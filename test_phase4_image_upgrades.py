"""Tests for Phase 4: image-gen presets, rewriter, style sheets, ETA, variations, print-this planner."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.edison_core.image_gen_presets import (
    ASPECT_PRESETS,
    StyleSheet,
    StyleSheetStore,
    build_grid,
    estimate_lane_eta_ms,
    pick_negative_prompt,
    plan_print_this,
    resolve_aspect,
    rewrite_prompt,
    sampler_defaults,
    variation_seeds,
)
from services.edison_core.routes.phase4 import router as phase4_router


# ── Aspect presets ──────────────────────────────────────────────

def test_resolve_aspect_known():
    out = resolve_aspect("square")
    assert out["width"] == 1024 and out["height"] == 1024
    assert out["upscale"] is False

def test_resolve_aspect_upscale_target():
    out = resolve_aspect("poster_18x24")
    assert out["upscale"] is True

def test_resolve_aspect_custom_string():
    out = resolve_aspect("768x1024")
    assert out == {"width": 768, "height": 1024, "hint": "custom", "upscale": False}

def test_resolve_aspect_unknown_raises():
    with pytest.raises(KeyError):
        resolve_aspect("definitely-not-a-preset")

def test_aspect_presets_are_64_aligned():
    # Generation-time presets must be /8-aligned for SDXL/Flux latents.
    # Platform-spec presets (twitter header, fb cover, etc.) skip this
    # check because they are render-then-downscale targets.
    from services.edison_core.image_gen_presets import PRESETS_NEEDING_UPSCALE
    for name, (w, h, _) in ASPECT_PRESETS.items():
        if name in PRESETS_NEEDING_UPSCALE:
            continue
        assert w % 8 == 0, f"{name} width {w} not /8"
        assert h % 8 == 0, f"{name} height {h} not /8"


# ── Negative prompt + sampler defaults ────────────────────────

def test_negative_prompt_flux_is_empty():
    assert pick_negative_prompt("flux_dev") == ""
    assert pick_negative_prompt("flux") == ""

def test_negative_prompt_sdxl_portrait_specialised():
    portrait = pick_negative_prompt("sdxl", intent="portrait")
    plain = pick_negative_prompt("sdxl")
    assert "asymmetrical" in portrait
    assert portrait != plain

def test_negative_prompt_logo_intent_overrides_family():
    out = pick_negative_prompt("flux", intent="logo")
    assert "photorealistic" in out

def test_sampler_defaults_known_family():
    out = sampler_defaults("flux_schnell")
    assert out["steps"] == 4
    assert out["sampler"] == "euler"

def test_sampler_defaults_unknown_falls_back():
    out = sampler_defaults("not-a-real-model")
    assert out["matched"] == "sdxl_default"


# ── Prompt rewriter + style sheet ─────────────────────────────

def test_rewrite_prompt_appends_quality_for_sdxl():
    out = rewrite_prompt("a cat on a chair", model_family="sdxl")
    assert "a cat on a chair" in out.positive
    assert "masterpiece" in out.positive  # sdxl quality suffix
    assert out.negative  # sdxl gets a negative prompt

def test_rewrite_prompt_flux_no_negative():
    out = rewrite_prompt("a cat on a chair", model_family="flux_dev")
    assert out.negative == ""

def test_rewrite_prompt_detects_logo_intent():
    out = rewrite_prompt("logo for adoro pizza", model_family="sdxl")
    assert out.intent == "logo"
    # logo intent means logo negatives, not generic sdxl
    assert "photorealistic" in out.negative

def test_style_sheet_strip_banned_is_word_aware():
    sheet = StyleSheet(project_id="p1", banned=["pink", "neon"])
    out = sheet.strip_banned("a pink and neon sign")
    assert "pink" not in out.lower()
    assert "neon" not in out.lower()
    # but a partial word like 'pinkish' would be banned too only if exact;
    # we use word boundaries → 'pinker' would be removed, 'pinkish' would not
    out2 = sheet.strip_banned("pinkish hue")
    assert "pinkish" in out2

def test_rewrite_prompt_with_style_sheet(tmp_path: Path):
    StyleSheetStore.reset_instance()
    store = StyleSheetStore.get_instance(root=tmp_path)
    sheet = StyleSheet(
        project_id="adoro",
        palette=["#ff0000", "#00aa55"],
        fonts=["Inter"],
        vibe=["bold", "italian"],
        banned=["cheap"],
    )
    store.save(sheet)
    loaded = store.get("adoro")
    assert loaded is not None
    out = rewrite_prompt(
        "cheap pizza ad with stripes", model_family="sdxl", style_sheet=loaded
    )
    assert "cheap" not in out.positive.lower()
    assert "bold" in out.positive
    assert "#ff0000" in out.positive

def test_style_sheet_store_persists(tmp_path: Path):
    StyleSheetStore.reset_instance()
    store = StyleSheetStore.get_instance(root=tmp_path)
    store.save(StyleSheet(project_id="x", palette=["#fff"]))
    StyleSheetStore.reset_instance()
    store2 = StyleSheetStore.get_instance(root=tmp_path)
    assert store2.get("x").palette == ["#fff"]
    assert "x" in store2.list_projects()
    assert store2.delete("x") is True
    assert store2.get("x") is None


# ── Seed variations + grid ────────────────────────────────────

def test_variation_seeds_deterministic():
    a = variation_seeds(42, count=4)
    b = variation_seeds(42, count=4)
    assert a == b
    assert len(a) == 4

def test_variation_seeds_different_for_different_input():
    assert variation_seeds(1) != variation_seeds(2)

def test_build_grid_dedupes():
    g = build_grid(samplers=("a", "a"), cfgs=(1.0, 1.0), steps=(10,))
    assert len(g) == 1


# ── Queue ETA ─────────────────────────────────────────────────

def test_estimate_lane_eta_no_history_uses_fallback():
    tel = {"lanes": {"image": {"queued": 2, "in_flight": 0, "max_concurrent": 1}},
           "history": []}
    out = estimate_lane_eta_ms(tel, "image", fallback_ms=10_000)
    assert out["known"] is False
    assert out["avg_duration_ms"] == 10_000
    assert out["eta_ms"] == 20_000

def test_estimate_lane_eta_with_history():
    tel = {
        "lanes": {"image": {"queued": 1, "in_flight": 1, "max_concurrent": 1}},
        "history": [
            {"lane": "image", "status": "done", "duration_ms": 4000},
            {"lane": "image", "status": "done", "duration_ms": 6000},
            {"lane": "chat",  "status": "done", "duration_ms": 100},  # ignored
        ],
    }
    out = estimate_lane_eta_ms(tel, "image")
    assert out["known"] is True
    assert out["avg_duration_ms"] == 5000
    # queued=1, over=0 → eta = 5000 * 1
    assert out["eta_ms"] == 5000

def test_estimate_lane_eta_unknown_lane():
    out = estimate_lane_eta_ms({"lanes": {}, "history": []}, "image")
    assert out["queued"] == 0
    assert out["known"] is False


# ── Print-this planner ────────────────────────────────────────

def test_plan_print_this_keychain_includes_extrude_and_qa():
    plan = plan_print_this(artifact_id="img1", intent="keychain")
    ops = [s["op"] for s in plan]
    assert "remove_background" in ops
    assert "vectorize" in ops
    assert "extrude_svg" in ops
    assert "cad_qa" in ops
    assert "queue_print" in ops
    # extrude has the keychain ring
    extrude = next(s for s in plan if s["op"] == "extrude_svg")
    assert "ring_diameter_mm" in extrude["params"]

def test_plan_print_this_sticker_skips_stl():
    plan = plan_print_this(artifact_id="img1", intent="sticker")
    ops = [s["op"] for s in plan]
    assert "extrude_svg" not in ops
    assert "export_print_png" in ops


# ── Phase 4 routes via TestClient ─────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    StyleSheetStore.reset_instance()
    monkeypatch.setattr(StyleSheetStore, "DEFAULT_ROOT", tmp_path / "ss")
    app = FastAPI()
    app.include_router(phase4_router)
    return TestClient(app)


def test_route_image_presets(client):
    r = client.get("/api/phase4/image/presets")
    assert r.status_code == 200
    presets = r.json()["presets"]
    names = {p["name"] for p in presets}
    assert {"square", "story", "logo", "tshirt_front"} <= names

def test_route_image_preset_by_name(client):
    r = client.get("/api/phase4/image/presets/square")
    assert r.status_code == 200
    assert r.json()["width"] == 1024

def test_route_image_preset_unknown_404(client):
    r = client.get("/api/phase4/image/presets/foo-bar")
    assert r.status_code == 404

def test_route_image_defaults(client):
    r = client.get("/api/phase4/image/defaults", params={"model_family": "flux_schnell"})
    assert r.status_code == 200
    body = r.json()
    assert body["sampler"]["steps"] == 4
    assert body["negative_prompt"] == ""

def test_route_image_rewrite(client):
    r = client.post("/api/phase4/image/rewrite",
                    json={"prompt": "logo for cafe", "model_family": "sdxl"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "logo"
    assert "photorealistic" in body["negative"]

def test_route_style_crud(client):
    pid = "demo-co"
    # 404 first
    assert client.get(f"/api/phase4/style/{pid}").status_code == 404
    # PUT
    r = client.put("/api/phase4/style", json={
        "project_id": pid, "palette": ["#abc"], "fonts": [], "vibe": ["clean"],
        "banned": [],
    })
    assert r.status_code == 200
    # GET
    r = client.get(f"/api/phase4/style/{pid}")
    assert r.status_code == 200
    assert r.json()["palette"] == ["#abc"]
    # LIST
    r = client.get("/api/phase4/style")
    assert pid in r.json()["projects"]
    # DELETE
    assert client.delete(f"/api/phase4/style/{pid}").json()["deleted"] is True

def test_route_queue_eta(client):
    # GPU scheduler is real; just verify the endpoint speaks JSON
    r = client.get("/api/phase4/queue/eta", params={"lane": "image"})
    assert r.status_code == 200
    body = r.json()
    assert body["lane"] == "image"
    assert "eta_ms" in body

def test_route_image_variations(client):
    r = client.get("/api/phase4/image/variations",
                   params={"seed": 1234, "count": 3})
    assert r.status_code == 200
    body = r.json()
    assert len(body["seeds"]) == 3

def test_route_print_this_plan(client):
    r = client.post("/api/phase4/print-this/plan",
                    json={"artifact_id": "abc", "intent": "keychain"})
    assert r.status_code == 200
    ops = [s["op"] for s in r.json()["steps"]]
    assert "extrude_svg" in ops
