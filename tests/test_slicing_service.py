import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_slicer_capabilities_prefer_curaengine_but_use_next_usable(monkeypatch, tmp_path):
    from services.edison_core.slicing import SlicerService

    definition_dir = tmp_path / "config" / "cura" / "definitions"
    profile_dir = tmp_path / "config" / "cura" / "profiles"
    definition_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    (definition_dir / "fdmprinter.def.json").write_text("{}")
    (profile_dir / "standard.inst.cfg").write_text("[values]")

    def _which(name):
        mapping = {
            "CuraEngine": "/usr/bin/CuraEngine",
            "prusa-slicer": "/usr/bin/prusa-slicer",
        }
        return mapping.get(name)

    monkeypatch.setattr("shutil.which", _which)

    service = SlicerService(tmp_path, config={
        "edison": {
            "printing": {
                "slicer": {
                    "cura_definition": "config/cura/definitions/fdmprinter.def.json",
                    "cura_profile": "config/cura/profiles/standard.inst.cfg",
                }
            }
        }
    })
    capabilities = service.get_capabilities()

    assert capabilities["preferred_engine"] == "curaengine"
    assert capabilities["active_engine"] == "curaengine"
    assert capabilities["engines"][0]["requires_definition"] is True
    assert capabilities["engines"][0]["definition_configured"] is True
    assert capabilities["cura_config"]["configured"] is True


def test_slice_options_normalize_request_values():
    from services.edison_core.slicing import SlicingOptions

    options = SlicingOptions.from_request({
        "profile": "0.1mm",
        "quality": "fine",
        "layer_height": "0.12",
        "material": "PETG",
        "infill": "35",
        "supports": "true",
        "adhesion": "brim",
        "nozzle": "0.6",
        "speed_profile": "fast",
    })

    assert options.layer_height == 0.12
    assert options.infill == 35
    assert options.supports is True
    assert options.material == "PETG"
    assert options.nozzle == "0.6"


def test_estimate_includes_cost_breakdown(tmp_path):
    from services.edison_core.slicing import SlicerService, SlicingOptions

    model_path = tmp_path / "cube.stl"
    model_path.write_text("solid cube")

    service = SlicerService(tmp_path, config={
        "edison": {
            "printing": {
                "slicer": {
                    "currency": "USD",
                    "material_cost_per_kg": {"pla": 22},
                }
            }
        }
    })
    estimate = service.estimate(model_path, SlicingOptions(material="PLA", infill=25, supports=True))

    assert estimate["currency"] == "USD"
    assert estimate["cost_breakdown"]["material_cost_per_kg"] == 22.0
    assert "spool_fraction" in estimate
    assert estimate["assumptions"]["adhesion"] == "skirt"


def test_printing_routes_use_slicer_service(monkeypatch, tmp_path):
    import services.edison_core.app as appmod

    model_path = tmp_path / "cube.stl"
    model_path.write_text("solid cube")

    class FakeSlicerService:
        def get_capabilities(self):
            return {
                "engines": [{"key": "prusa-slicer", "available": True, "usable": True, "status": "ready"}],
                "preferred_engine": "prusa-slicer",
                "active_engine": "prusa-slicer",
                "supports_estimates": True,
                "currency": "USD",
                "cura_config": {"configured": False, "definition_file": None, "profile_file": None},
            }

        def estimate(self, safe_model: Path, options):
            return {
                "duration_minutes": 42,
                "material_grams": 18.5,
                "estimated_cost": 0.56,
                "currency": "USD",
                "spool_fraction": 0.019,
                "cost_breakdown": {"material_cost_per_kg": 30.0, "cost_per_gram": 0.03, "spool_size_grams": 1000},
                "assumptions": {"material": options.material, "adhesion": options.adhesion, "nozzle": options.nozzle, "speed_profile": options.speed_profile, "supports": options.supports, "infill": options.infill, "layer_height": options.layer_height},
            }

        def slice(self, safe_model: Path, output_path: Path, options):
            output_path.write_text("G1 X0 Y0")
            return {"engine": "prusa-slicer", "output_path": str(output_path), "options": options.to_dict()}

    async def fake_execute_tool(name, payload, chat_id=None):
        return {"ok": True, "queued": True, "tool": name, "payload": payload}

    monkeypatch.setattr(appmod, "slicer_service_instance", FakeSlicerService())
    monkeypatch.setattr(appmod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(appmod, "_safe_workspace_path", lambda _: model_path)
    monkeypatch.setattr(appmod, "_workspace_relative", lambda p: str(Path(p).relative_to(tmp_path)))
    monkeypatch.setattr(appmod, "_execute_tool", fake_execute_tool)

    client = TestClient(appmod.app)

    capabilities = client.get("/printing/slicer/capabilities")
    assert capabilities.status_code == 200
    assert capabilities.json()["active_engine"] == "prusa-slicer"

    estimate = client.post("/printing/slice/estimate", json={
        "model_path": "cube.stl",
        "material": "PLA",
        "infill": 15,
        "supports": True,
    })
    assert estimate.status_code == 200
    estimate_payload = estimate.json()
    assert estimate_payload["estimate"]["duration_minutes"] == 42
    assert estimate_payload["options"]["supports"] is True

    sliced = client.post("/printing/slice", json={
        "model_path": "cube.stl",
        "profile": "0.2mm",
        "quality": "draft",
        "infill": 10,
    })
    assert sliced.status_code == 200
    sliced_payload = sliced.json()
    assert sliced_payload["ok"] is True
    assert sliced_payload["engine"] == "prusa-slicer"
    assert sliced_payload["options"]["quality"] == "draft"

    dispatched = client.post("/printing/slice-and-send", json={
        "model_path": "cube.stl",
        "printer_id": "p1",
        "material": "PETG",
        "supports": False,
    })
    assert dispatched.status_code == 200
    dispatched_payload = dispatched.json()
    assert dispatched_payload["dispatch"]["ok"] is True
    assert dispatched_payload["engine"] == "prusa-slicer"