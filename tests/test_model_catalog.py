import os
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_model_catalog_discovers_llm_and_comfyui_models(tmp_path, monkeypatch):
    from services.edison_core.model_catalog import build_model_catalog

    repo_root = tmp_path / "repo"
    llm_dir = repo_root / "models" / "llm"
    comfy_dir = repo_root / "ComfyUI" / "models"
    custom_nodes_dir = repo_root / "ComfyUI" / "custom_nodes"
    (comfy_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (comfy_dir / "loras").mkdir(parents=True, exist_ok=True)
    (custom_nodes_dir / "ComfyUI-TriposR").mkdir(parents=True, exist_ok=True)
    llm_dir.mkdir(parents=True, exist_ok=True)

    (llm_dir / "qwen2.5-14b-instruct-q4_k_m.gguf").write_bytes(b"llm")
    (comfy_dir / "checkpoints" / "flux1-dev.safetensors").write_bytes(b"image")
    (comfy_dir / "loras" / "product-photo-detail.safetensors").write_bytes(b"lora")
    (comfy_dir / "checkpoints" / "hunyuan3d-std.safetensors").write_bytes(b"mesh")

    catalog = build_model_catalog(repo_root, {
        "edison": {
            "core": {"models_path": "models/llm", "fast_model": "qwen2.5-14b-instruct-q4_k_m.gguf"},
            "comfyui": {"host": "127.0.0.1", "port": 8188},
            "video": {"cogvideox_model": "THUDM/CogVideoX-5b"},
            "music": {"model_size": "medium"},
            "voice": {"stt_model": "base", "tts_voice": "en-US-GuyNeural"},
        }
    })

    assert catalog["summary"]["llm_installed"] == 1
    assert catalog["summary"]["image_checkpoints_installed"] == 2
    assert catalog["summary"]["image_loras_installed"] == 1
    assert catalog["summary"]["mesh_model_candidates"] == 1
    assert catalog["summary"]["mesh_custom_nodes"] == 1
    assert catalog["image"]["checkpoints"][0]["filename"] == "flux1-dev.safetensors"
    assert "text_to_image" in catalog["image"]["workflows"][0]["workflow"]
    assert catalog["mesh"]["readiness"]["usable"] is True
    assert catalog["mesh"]["model_candidates"][0]["filename"] == "hunyuan3d-std.safetensors"


def test_model_catalog_route_returns_recommendations(monkeypatch, tmp_path):
    from services.edison_core.routes import business_platform

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(business_platform, "REPO_ROOT", repo_root)
    monkeypatch.setattr(business_platform, "_load_config", lambda: {
        "edison": {
            "core": {"models_path": "models/llm"},
            "comfyui": {"host": "127.0.0.1", "port": 8188},
            "video": {"cogvideox_model": "THUDM/CogVideoX-5b"},
            "music": {"model_size": "medium"},
            "voice": {"stt_model": "base", "tts_voice": "en-US-GuyNeural"},
        }
    })

    app = FastAPI()
    app.include_router(business_platform.router)
    client = TestClient(app)

    response = client.get("/system-awareness/models", params={"task": "product images"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["recommendation"]["matches"][0]["task"] == "product_images"


def test_model_catalog_recommends_mesh_generation(tmp_path):
    from services.edison_core.model_catalog import build_model_catalog, recommend_models_for_task

    repo_root = tmp_path / "repo"
    (repo_root / "ComfyUI" / "models" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (repo_root / "ComfyUI" / "custom_nodes" / "InstantMesh").mkdir(parents=True, exist_ok=True)
    (repo_root / "ComfyUI" / "models" / "checkpoints" / "stable-fast-3d.safetensors").write_bytes(b"mesh")

    catalog = build_model_catalog(repo_root, {"edison": {"comfyui": {"host": "127.0.0.1", "port": 8188}}})
    recommendation = recommend_models_for_task("which 3d model should I use for a product mockup", catalog)

    assert recommendation["matches"][0]["task"] == "mesh_generation"
    assert recommendation["matches"][0]["readiness"]["usable"] is True