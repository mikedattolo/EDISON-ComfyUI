"""Integration test: ensure all phase routers mount cleanly into a
FastAPI app and the basic GET endpoints respond.

This deliberately builds a *fresh* FastAPI app with only the phase
routers, so it doesn't pull in the full edison_core.app (which would
require GPU/model setup).
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")


def _build_app():
    from fastapi import FastAPI
    from services.edison_core.routes.phase1 import router as r1
    from services.edison_core.routes.phase2 import router as r2
    from services.edison_core.routes.phase3 import router as r3

    app = FastAPI()
    app.include_router(r1)
    app.include_router(r2)
    app.include_router(r3)
    return app


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    return TestClient(_build_app())


def test_all_expected_routes_exist():
    app = _build_app()
    paths = {r.path for r in app.routes}
    expected = {
        "/api/phase1/health",
        "/api/phase1/scheduler/telemetry",
        "/api/phase1/scheduler/lanes",
        "/api/phase1/citations/bundle",
        "/api/phase2/jobs",
        "/api/phase2/jobs/summary",
        "/api/phase2/palette",
        "/api/phase2/palette/search",
        "/api/phase2/conversations/search",
        "/api/phase2/context/usage",
        "/api/phase3/cad/qa",
        "/api/phase3/video/presets",
        "/api/phase3/video/shotlist",
        "/api/phase3/video/sequences",
    }
    missing = expected - paths
    assert not missing, f"missing routes: {missing}"


def test_phase1_health(client):
    r = client.get("/api/phase1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["modules"]["scheduler"] is True


def test_phase1_scheduler_telemetry(client):
    r = client.get("/api/phase1/scheduler/telemetry")
    assert r.status_code == 200
    body = r.json()
    assert "lanes" in body
    assert "chat" in body["lanes"]


def test_phase1_citations_bundle_endpoint(client):
    r = client.post("/api/phase1/citations/bundle", json={
        "hits": [{"url": "https://x.test", "title": "X", "snippet": "y"}],
        "source": "web",
        "request_id": "req-1",
        "append_to": "Body of answer.",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert "**Sources**" in body["text_with_sources"]


def test_phase2_palette_endpoint(client):
    r = client.get("/api/phase2/palette")
    assert r.status_code == 200
    body = r.json()
    assert any(c["id"] == "branding.logo" for c in body["commands"])


def test_phase2_palette_search_endpoint(client):
    r = client.get("/api/phase2/palette/search", params={"q": "make a logo"})
    assert r.status_code == 200
    body = r.json()
    assert body["matches"], "expected at least one match"
    assert body["matches"][0]["category"] == "branding"


def test_phase2_context_usage_endpoint(client):
    r = client.post("/api/phase2/context/usage", json={
        "messages": [{"role": "user", "content": "hello world"}],
        "context_window": 8192,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "warn", "critical"}
    assert "available_tokens" in body


def test_phase3_video_presets_endpoint(client):
    r = client.get("/api/phase3/video/presets")
    assert r.status_code == 200
    presets = r.json()["presets"]
    names = {p["name"] for p in presets}
    assert {"Instagram Reel", "TikTok"}.issubset(names)


def test_phase3_video_shotlist_endpoint(client):
    r = client.post("/api/phase3/video/shotlist", json={
        "topic": "Adoro Pizza promo",
        "preset": "instagram_reel",
        "beat_count": 4,
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["shots"]) == 4
    assert body["preset"] == "instagram_reel"


def test_phase3_cad_qa_missing_file(client):
    r = client.post("/api/phase3/cad/qa", json={
        "mesh_path": "/no/such/path.stl",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["passed"] is False


def test_phase3_video_sequences_lifecycle(client, tmp_path, monkeypatch):
    # Redirect ClipSequence.DEFAULT_ROOT to an isolated tmp dir so the
    # endpoint test doesn't touch the repo's data/.
    from services.edison_core.video_timeline import ClipSequence
    monkeypatch.setattr(ClipSequence, "DEFAULT_ROOT", tmp_path)

    r = client.post("/api/phase3/video/sequences", json={
        "title": "Test", "preset": "instagram_reel"
    })
    assert r.status_code == 200
    seq_id = r.json()["sequence_id"]

    r = client.post(
        f"/api/phase3/video/sequences/{seq_id}/clips",
        json={"source": "/a.mp4", "in_s": 0, "out_s": 3.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total_duration_s"] == pytest.approx(3.0, rel=1e-3)

    r = client.post(f"/api/phase3/video/sequences/{seq_id}/export-plan")
    assert r.status_code == 200
    plan = r.json()
    assert plan["clip_count"] == 1
    assert plan["preset"]["name"] == "Instagram Reel"
