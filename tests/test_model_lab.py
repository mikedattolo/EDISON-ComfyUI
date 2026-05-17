from pathlib import Path

from services.edison_core import model_lab


def test_model_lab_summary_uses_hardware_and_installed_models(tmp_path, monkeypatch):
    models_dir = tmp_path / "models" / "llm"
    models_dir.mkdir(parents=True)
    (models_dir / "Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf").write_bytes(b"x" * 1024)

    monkeypatch.setattr(model_lab, "detect_system_ram_gb", lambda: 128.0)
    monkeypatch.setattr(
        model_lab,
        "detect_gpus",
        lambda: [
            {"index": 2, "name": "NVIDIA GeForce RTX 3090", "vram_total_mb": 24576},
            {"index": 0, "name": "NVIDIA GeForce RTX 5060 Ti", "vram_total_mb": 16311},
        ],
    )

    summary = model_lab.summarize_model_lab(
        tmp_path,
        {"edison": {"core": {"models_path": "models/llm"}}},
    )

    assert summary["ok"] is True
    assert summary["policy"]["no_safeguard_bypass"] is False
    assert summary["hardware"]["ram_upgrade_detected"] is True
    qwen3 = next(item for item in summary["recommendations"] if item["id"] == "qwen3-30b-a3b-instruct")
    assert qwen3["installed"] is True
    assert qwen3["fit"] == "ready"


def test_model_lab_install_plan_returns_hf_command(tmp_path):
    plan = model_lab.install_plan(
        "qwen2-5-coder-32b",
        tmp_path,
        {"edison": {"core": {"models_path": "models/llm"}}},
    )

    assert "huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct-GGUF" in plan["commands"][1]
    assert plan["config_hint"]["slot"] == "reasoning"


def test_detect_gpus_uses_absolute_nvidia_smi_fallback(monkeypatch):
    class FakePath:
        def __init__(self, path):
            self.path = path

        def exists(self):
            return self.path == "/usr/bin/nvidia-smi"

    class FakeProc:
        returncode = 0
        stdout = (
            "0, NVIDIA GeForce RTX 5060 Ti, 16311 MiB, 7817 MiB, 24, 50 %, 19.50 W\n"
            "2, NVIDIA GeForce RTX 3090, 24576 MiB, 8789 MiB, 30, 60 %, 93.85 W\n"
        )

    monkeypatch.setattr(model_lab.shutil, "which", lambda _name: None)
    monkeypatch.setattr(model_lab, "Path", FakePath)
    monkeypatch.setattr(model_lab.subprocess, "run", lambda *args, **kwargs: FakeProc())

    gpus = model_lab.detect_gpus()

    assert [gpu["name"] for gpu in gpus] == ["NVIDIA GeForce RTX 5060 Ti", "NVIDIA GeForce RTX 3090"]
    assert gpus[0]["vram_total_mb"] == 16311
    assert gpus[1]["fan_speed_percent"] == 60
    assert gpus[1]["power_draw_w"] == 93.85


def test_unknown_model_lab_profile_raises_key_error(tmp_path):
    try:
        model_lab.install_plan("missing", tmp_path, {})
    except KeyError as exc:
        assert exc.args[0] == "missing"
    else:
        raise AssertionError("Expected KeyError")
