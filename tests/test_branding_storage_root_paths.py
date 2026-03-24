import asyncio
import os
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _configure_branding_paths(core_app, monkeypatch, branding_root: Path, integrations_dir: Path):
    monkeypatch.setattr(core_app, "INTEGRATIONS_DIR", integrations_dir)
    monkeypatch.setattr(core_app, "CONNECTORS_DB", integrations_dir / "connectors.json")
    monkeypatch.setattr(core_app, "PRINTERS_DB", integrations_dir / "printers.json")
    monkeypatch.setattr(core_app, "PROMPTS_DB", integrations_dir / "prompts.json")
    monkeypatch.setattr(core_app, "BRANDING_DB", integrations_dir / "branding.json")
    monkeypatch.setattr(core_app, "BRANDING_ROOT", branding_root)
    monkeypatch.setattr(
        core_app,
        "MEDIA_ROOTS",
        [
            core_app.REPO_ROOT / "outputs",
            core_app.REPO_ROOT / "uploads",
            core_app.REPO_ROOT / "gallery",
            branding_root,
        ],
    )


def test_branding_create_client_with_root_inside_repo(monkeypatch, tmp_path):
    from services.edison_core import app as core_app

    inside_root = core_app.REPO_ROOT / "outputs" / f"pytest_branding_inside_{uuid.uuid4().hex[:8]}"
    integrations_dir = tmp_path / "integrations_inside"
    _configure_branding_paths(core_app, monkeypatch, inside_root, integrations_dir)

    try:
        data = asyncio.run(core_app.branding_create_client({"name": "adoropizza"}))

        assert data["ok"] is True
        assert data["created"] is True
        assert data["client"]["slug"] == "adoropizza"
        assert data["client"]["paths"]["base"].startswith("outputs/")
        assert (inside_root / "adoropizza" / "images").exists()
        assert (inside_root / "adoropizza" / "videos").exists()
        assert (inside_root / "adoropizza" / "files").exists()
    finally:
        shutil.rmtree(inside_root, ignore_errors=True)


def test_branding_paths_with_root_outside_repo(monkeypatch, tmp_path):
    from services.edison_core import app as core_app

    outside_root = tmp_path / "edison_data"
    integrations_dir = tmp_path / "integrations_outside"
    _configure_branding_paths(core_app, monkeypatch, outside_root, integrations_dir)

    create_data = asyncio.run(core_app.branding_create_client({"name": "adoropizza"}))
    assert create_data["ok"] is True
    assert create_data["created"] is True
    assert create_data["client"]["slug"] == "adoropizza"

    # Outside repo roots should serialize to absolute paths, not crash with relative_to().
    base_path = create_data["client"]["paths"]["base"]
    assert base_path.startswith(str(outside_root.resolve()))

    source_name = f"pytest_branding_src_{uuid.uuid4().hex[:8]}.txt"
    source_path = core_app.REPO_ROOT / "outputs" / source_name
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("branding test")

    try:
        add_data = asyncio.run(
            core_app.branding_add_existing_asset(
                create_data["client"]["id"],
                {"source_path": str(source_path), "asset_type": "files", "move": False},
            )
        )
        assert add_data["ok"] is True
        assert add_data["stored_path"].startswith(str(outside_root.resolve()))

        assets_data = asyncio.run(core_app.branding_list_assets(create_data["client"]["id"]))
        listed_paths = [item["path"] for item in assets_data["assets"]["files"]]
        assert add_data["stored_path"] in listed_paths
    finally:
        source_path.unlink(missing_ok=True)
