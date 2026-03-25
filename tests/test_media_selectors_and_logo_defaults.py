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
