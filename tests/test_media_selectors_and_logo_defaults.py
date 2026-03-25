import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_video_recent_returns_filtered_recent_items(monkeypatch, tmp_path):
    from services.edison_core import app as core_app

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    media_root = tmp_path / "media"
    media_root.mkdir(parents=True, exist_ok=True)

    older_video = media_root / "older.mp4"
    newer_audio = media_root / "newer.mp3"
    ignored_text = media_root / "notes.txt"
    older_video.write_bytes(b"video")
    newer_audio.write_bytes(b"audio")
    ignored_text.write_text("ignore me")

    os.utime(older_video, (100, 100))
    os.utime(newer_audio, (200, 200))
    os.utime(ignored_text, (300, 300))

    monkeypatch.setattr(core_app, "REPO_ROOT", repo_root)
    monkeypatch.setattr(core_app, "MEDIA_ROOTS", [media_root])

    payload = asyncio.run(core_app.video_recent(limit=2, kind="video,audio"))

    assert payload["ok"] is True
    assert [item["name"] for item in payload["items"]] == ["newer.mp3", "older.mp4"]
    assert {item["kind"] for item in payload["items"]} == {"audio", "video"}


def test_video_edit_accepts_media_root_outside_repo(monkeypatch, tmp_path):
    from services.edison_core import app as core_app

    repo_root = tmp_path / "repo"
    outputs_dir = repo_root / "outputs" / "videos"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    outside_root = tmp_path / "external_media"
    outside_root.mkdir(parents=True, exist_ok=True)
    source = outside_root / "clip.mp4"
    source.write_bytes(b"fake video")

    monkeypatch.setattr(core_app, "REPO_ROOT", repo_root)
    monkeypatch.setattr(core_app, "MEDIA_ROOTS", [outside_root, repo_root / "outputs"])

    def fake_run(cmd, capture_output=True, text=True, timeout=300):
        Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[-1]).write_bytes(b"rendered")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(core_app.subprocess, "run", fake_run)

    payload = asyncio.run(core_app.video_edit({"source_path": str(source), "operation": "trim"}))

    assert payload["ok"] is True
    assert payload["source_path"] == str(source.resolve())
    assert payload["output_path"].startswith("outputs/videos/")


def test_logo_requests_get_stronger_generation_defaults():
    from services.edison_core.app import _enhance_image_prompt, _image_generation_defaults

    defaults = _image_generation_defaults(
        prompt="make a modern pizza shop logo for Adoro",
        style_preset="auto",
        steps=20,
        guidance_scale=3.5,
        negative_prompt="",
    )
    enhanced_prompt = _enhance_image_prompt("make a modern pizza shop logo for Adoro", defaults["style_preset"])

    assert defaults["is_logo_request"] is True
    assert defaults["style_preset"] == "logo"
    assert defaults["steps"] >= 28
    assert defaults["guidance_scale"] >= 6.5
    assert "photorealistic" in defaults["negative_prompt"]
    assert "vector-style branding artwork" in enhanced_prompt
    assert "professional vector logo" in enhanced_prompt


def test_comfyui_base_url_normalizes_override_scheme_and_host():
    from services.edison_core.app import _comfyui_base_url

    assert _comfyui_base_url("https://100.67.221.112:8188/") == "https://100.67.221.112:8188"
    assert _comfyui_base_url("100.67.221.112:8188") == "http://100.67.221.112:8188"
    assert _comfyui_base_url("http://0.0.0.0:8188") == "http://127.0.0.1:8188"


def test_submit_comfyui_prompt_retries_http_on_ssl_mismatch(monkeypatch):
    import requests
    from services.edison_core import app as core_app

    calls = []

    class FakeResponse:
        ok = True

    def fake_post(url, json=None, timeout=0):
        calls.append(url)
        if url.startswith("https://"):
            raise requests.exceptions.SSLError("[SSL: WRONG_VERSION_NUMBER] wrong version number (_ssl.c:1000)")
        return FakeResponse()

    monkeypatch.setattr(core_app.requests, "post", fake_post)

    response, final_url = core_app._submit_comfyui_prompt({"prompt": {}}, "https://100.67.221.112:8188", timeout=5)

    assert response.ok is True
    assert final_url == "http://100.67.221.112:8188"
    assert calls == [
        "https://100.67.221.112:8188/prompt",
        "http://100.67.221.112:8188/prompt",
    ]


def test_image_status_returns_gallery_metadata_for_follow_up_edits(monkeypatch, tmp_path):
    from services.edison_core import app as core_app

    gallery_dir = tmp_path / "gallery"
    gallery_db = gallery_dir / "gallery.json"

    monkeypatch.setattr(core_app, "GALLERY_DIR", gallery_dir)
    monkeypatch.setattr(core_app, "GALLERY_DB", gallery_db)
    monkeypatch.setattr(core_app, "config", {"edison": {"comfyui": {"host": "127.0.0.1", "port": 8188}}})
    monkeypatch.setattr(core_app, "_on_image_generation_complete", lambda prompt_id=None: None)
    monkeypatch.setattr(core_app, "provenance_tracker_instance", None)
    monkeypatch.setattr(core_app.uuid, "uuid4", lambda: "test-image-id")

    class FakeResponse:
        def __init__(self, ok, payload=None, content=b"", status_code=200):
            self.ok = ok
            self._payload = payload or {}
            self.content = content
            self.status_code = status_code

        def json(self):
            return self._payload

    def fake_get(url, timeout=0):
        if url.endswith("/history/prompt-1"):
            return FakeResponse(
                True,
                {
                    "prompt-1": {
                        "outputs": {
                            "9": {
                                "images": [
                                    {"filename": "comfy-output.png", "subfolder": "", "type": "output"}
                                ]
                            }
                        },
                        "prompt": {
                            "1": {
                                "class_type": "CLIPTextEncode",
                                "inputs": {"text": "modern pizza logo"},
                            }
                        },
                    }
                },
            )
        if "/view?filename=comfy-output.png" in url:
            return FakeResponse(True, content=b"png-bytes")
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(core_app.requests, "get", fake_get)

    payload = asyncio.run(core_app.image_status("prompt-1"))

    assert payload["status"] == "completed"
    assert payload["saved_to_gallery"] is True
    assert payload["gallery_image_id"] == "test-image-id"
    assert payload["gallery_filename"] == "test-image-id.png"
    assert payload["gallery_image_url"] == "/gallery/image/test-image-id.png"
    assert payload["source_path"] == str(gallery_dir / "test-image-id.png")
    assert (gallery_dir / "test-image-id.png").read_bytes() == b"png-bytes"


def test_edit_image_auto_saves_result_to_gallery(monkeypatch, tmp_path):
    from services.edison_core import app as core_app

    edited_path = tmp_path / "outputs" / "edits" / "edited.png"
    edited_path.parent.mkdir(parents=True, exist_ok=True)
    edited_path.write_bytes(b"edited-bytes")

    saved_calls = []

    class FakeEditRecord:
        def __init__(self, output_path):
            self.edit_id = "edit-1"
            self.output_path = output_path
            self.model_used = "SDXL"

        def to_dict(self):
            return {
                "edit_id": self.edit_id,
                "output_path": self.output_path,
            }

    class FakeEditor:
        def img2img(self, **kwargs):
            return FakeEditRecord(str(edited_path))

    class FakeRequest:
        async def json(self):
            return {
                "source_path": str(edited_path),
                "prompt": "make it flatter",
                "parameters": {
                    "auto_refine": False,
                    "auto_mask": False,
                    "auto_save_gallery": True,
                },
            }

    def fake_save(image_path, prompt, settings=None):
        saved_calls.append((image_path, prompt, settings))
        return {
            "id": "gallery-edit-1",
            "url": "/gallery/image/gallery-edit-1.png",
            "filename": "gallery-edit-1.png",
        }

    monkeypatch.setattr(core_app, "image_editor_instance", FakeEditor())
    monkeypatch.setattr(core_app, "_save_path_to_gallery", fake_save)

    payload = asyncio.run(core_app.edit_image(FakeRequest()))

    assert payload["ok"] is True
    assert payload["edit"]["image_url"] == "/images/edits/edited.png"
    assert payload["edit"]["gallery_image_id"] == "gallery-edit-1"
    assert payload["edit"]["gallery_image_url"] == "/gallery/image/gallery-edit-1.png"
    assert payload["edit"]["gallery_filename"] == "gallery-edit-1.png"
    assert saved_calls[0][0] == str(edited_path)
    assert saved_calls[0][1] == "make it flatter"
