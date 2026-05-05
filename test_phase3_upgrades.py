"""Tests for Phase 3 modules: CAD QA gates and video timeline."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest


# ── STL helpers ─────────────────────────────────────────────────────

def _write_binary_stl(path: Path, triangles: list[tuple[tuple[float, float, float], ...]]):
    with path.open("wb") as f:
        f.write(b" " * 80)
        f.write(struct.pack("<I", len(triangles)))
        for tri in triangles:
            # normal (zeroed)
            f.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for v in tri:
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))


def _cube_triangles(size: float = 10.0):
    """Return 12 triangles forming a closed cube of the given size."""
    s = size
    v = {
        0: (0, 0, 0), 1: (s, 0, 0), 2: (s, s, 0), 3: (0, s, 0),
        4: (0, 0, s), 5: (s, 0, s), 6: (s, s, s), 7: (0, s, s),
    }
    faces = [
        (0, 1, 2), (0, 2, 3),  # bottom
        (4, 6, 5), (4, 7, 6),  # top
        (0, 4, 5), (0, 5, 1),  # front
        (1, 5, 6), (1, 6, 2),  # right
        (2, 6, 7), (2, 7, 3),  # back
        (3, 7, 4), (3, 4, 0),  # left
    ]
    return [(v[a], v[b], v[c]) for a, b, c in faces]


# ── CAD QA tests ────────────────────────────────────────────────────

def test_cad_qa_clean_cube_passes(tmp_path: Path):
    from services.edison_core.cad_qa import run_qa

    p = tmp_path / "cube.stl"
    _write_binary_stl(p, _cube_triangles(20.0))

    report = run_qa(p, min_wall_thickness_mm=0.4)
    assert report.triangles == 12
    assert report.is_manifold is True
    assert report.open_edges == 0
    assert report.passed is True
    assert report.bounding_box["size"] == [20.0, 20.0, 20.0]


def test_cad_qa_open_mesh_fails(tmp_path: Path):
    from services.edison_core.cad_qa import run_qa

    # cube minus one face = non-manifold
    tris = _cube_triangles(10.0)[:-2]
    p = tmp_path / "open.stl"
    _write_binary_stl(p, tris)

    report = run_qa(p)
    assert report.is_manifold is False
    assert report.open_edges > 0
    assert report.passed is False


def test_cad_qa_build_volume_check(tmp_path: Path):
    from services.edison_core.cad_qa import run_qa

    p = tmp_path / "big.stl"
    _write_binary_stl(p, _cube_triangles(300.0))
    report = run_qa(p, build_volume_mm=(220, 220, 250))
    assert report.fits_build_volume is False
    assert any("build volume" in e for e in report.errors)


def test_cad_qa_missing_file_returns_error():
    from services.edison_core.cad_qa import run_qa

    report = run_qa("/nonexistent/path.stl")
    assert report.passed is False
    assert any("not found" in e for e in report.errors)


# ── Video timeline ──────────────────────────────────────────────────

def test_export_presets_present():
    from services.edison_core.video_timeline import list_presets, get_preset

    presets = list_presets()
    names = {p["name"] for p in presets}
    assert "Instagram Reel" in names
    p = get_preset("instagram_reel")
    assert p is not None
    assert p.aspect == "9:16"


def test_generate_shot_list_default():
    from services.edison_core.video_timeline import generate_shot_list

    sl = generate_shot_list("Adoro Pizza promo", preset="instagram_reel")
    assert sl.preset == "instagram_reel"
    assert len(sl.shots) == 7
    total = sum(s.duration_s for s in sl.shots)
    assert total > 0
    assert all("Adoro Pizza promo" in s.description for s in sl.shots)


def test_generate_shot_list_custom_beats():
    from services.edison_core.video_timeline import generate_shot_list

    sl = generate_shot_list("test", preset="tiktok", beat_count=3, target_duration_s=15)
    assert len(sl.shots) == 3
    assert sl.target_duration_s == 15.0


def test_generate_shot_list_unknown_preset():
    from services.edison_core.video_timeline import generate_shot_list

    with pytest.raises(ValueError):
        generate_shot_list("x", preset="fake_preset_999")


def test_clip_sequence_persistence(tmp_path: Path):
    from services.edison_core.video_timeline import Clip, ClipSequence

    seq = ClipSequence(title="Promo", preset="instagram_reel", root=tmp_path)
    seq.add_clip(Clip(id="c1", source="/a.mp4", in_s=0, out_s=3))
    seq.add_clip(Clip(id="c2", source="/b.mp4", in_s=0, out_s=2.5))
    seq.save()

    loaded = ClipSequence.load(seq.sequence_id, root=tmp_path)
    assert len(loaded.clips) == 2
    assert loaded.total_duration_s == pytest.approx(5.5, rel=1e-3)


def test_clip_sequence_export_plan_warns_on_overlong(tmp_path: Path):
    from services.edison_core.video_timeline import Clip, ClipSequence

    seq = ClipSequence(preset="instagram_feed", root=tmp_path)
    # feed preset is 60s; add a 70s clip
    seq.add_clip(Clip(id="c1", source="/a.mp4", in_s=0, out_s=70))
    plan = seq.export_plan()
    assert plan["preset"]["name"] == "Instagram Feed"
    assert any("trimmed" in w for w in plan["warnings"])


def test_clip_sequence_remove_and_move(tmp_path: Path):
    from services.edison_core.video_timeline import Clip, ClipSequence

    seq = ClipSequence(root=tmp_path)
    seq.add_clip(Clip(id="a", source="/a"))
    seq.add_clip(Clip(id="b", source="/b"))
    seq.add_clip(Clip(id="c", source="/c"))

    assert seq.move_clip("c", 0)
    assert [c.id for c in seq.clips] == ["c", "a", "b"]
    assert seq.remove_clip("a")
    assert [c.id for c in seq.clips] == ["c", "b"]
    assert not seq.remove_clip("nope")
