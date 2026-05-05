"""Regression tests for audit fixes:

* CAD QA: float-noise tolerance for shared edges (manifold detection).
* Retry: give_up_on exceptions propagate raw, not wrapped in RetryError.
* scheduled_ops: cancelled JobStore rows skip the coroutine.
"""

from __future__ import annotations

import asyncio
import struct
from pathlib import Path

import pytest


# ── CAD QA: jittered cube must still be detected as manifold ────────

def _write_binary_stl_jittered(path: Path, triangles, jitter: float = 1e-6):
    """Write the same STL but with each shared vertex perturbed slightly
    so we exercise the rounding code path in _edge_key.
    """
    import random
    random.seed(0)
    with path.open("wb") as f:
        f.write(b" " * 80)
        f.write(struct.pack("<I", len(triangles)))
        for tri in triangles:
            f.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for v in tri:
                # Perturb each vertex by < ndigits=4 rounding tolerance
                jv = (
                    v[0] + random.uniform(-jitter, jitter),
                    v[1] + random.uniform(-jitter, jitter),
                    v[2] + random.uniform(-jitter, jitter),
                )
                f.write(struct.pack("<3f", *jv))
            f.write(struct.pack("<H", 0))


def _cube():
    s = 20.0
    v = {
        0: (0, 0, 0), 1: (s, 0, 0), 2: (s, s, 0), 3: (0, s, 0),
        4: (0, 0, s), 5: (s, 0, s), 6: (s, s, s), 7: (0, s, s),
    }
    faces = [
        (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0),
    ]
    return [(v[a], v[b], v[c]) for a, b, c in faces]


def test_cad_qa_handles_float_noise(tmp_path: Path):
    """A cube with sub-rounding-tolerance jitter should still be manifold."""
    from services.edison_core.cad_qa import run_qa

    p = tmp_path / "jittered.stl"
    _write_binary_stl_jittered(p, _cube(), jitter=1e-6)
    report = run_qa(p)
    assert report.is_manifold is True, f"open edges: {report.open_edges}"
    assert report.passed is True


# ── Retry: give_up_on must propagate raw ────────────────────────────

def test_retry_sync_propagates_keyboard_interrupt():
    from services.edison_core.retry import RetryPolicy, retry_sync, RetryError

    def boom():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        retry_sync(boom, policy=RetryPolicy(max_attempts=3, base_delay=0.0, jitter=0.0))


@pytest.mark.asyncio
async def test_retry_async_propagates_cancellation():
    from services.edison_core.retry import RetryPolicy, retry_async

    async def boom():
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await retry_async(boom, policy=RetryPolicy(max_attempts=3, base_delay=0.0, jitter=0.0))


# ── scheduled_ops: cancellation propagation via JobStore ───────────

@pytest.mark.asyncio
async def test_scheduled_ops_skips_cancelled_jobs(monkeypatch):
    from services.edison_core import scheduled_ops
    from services.edison_core.gpu_scheduler import GPUScheduler

    GPUScheduler.reset_instance()
    GPUScheduler.get_instance()

    # Pretend the JobStore says this job is already cancelled.
    monkeypatch.setattr(
        scheduled_ops, "_job_already_cancelled", lambda jid: jid == "cancelled-1"
    )

    started = {"flag": False}

    async def work():
        started["flag"] = True
        return "should not run"

    with pytest.raises(asyncio.CancelledError):
        await scheduled_ops.run_scheduled("image", work, job_id="cancelled-1")
    assert started["flag"] is False


# ── Citations: id collisions are de-collided ────────────────────────

def test_citations_unique_ids_when_input_ids_collide():
    from services.edison_core.citations import normalize_hits

    hits = [
        {"id": "same", "url": "https://a.test", "title": "A"},
        {"id": "same", "url": "https://b.test", "title": "B"},
        {"id": "same", "url": "https://c.test", "title": "C"},
    ]
    cits = normalize_hits(hits)
    ids = [c.id for c in cits]
    assert len(set(ids)) == len(ids), f"duplicate ids: {ids}"


# ── Artifact revisions: tmp_path is honoured per-test ──────────────

def test_artifact_revisions_singleton_honours_root_change(tmp_path: Path):
    from services.edison_core.artifact_revisions import ArtifactRevisionStore

    ArtifactRevisionStore.reset_instance()
    a = ArtifactRevisionStore.get_instance(root=tmp_path / "a")
    a.add_revision("x", "first")

    b = ArtifactRevisionStore.get_instance(root=tmp_path / "b")
    # New instance because root differs
    assert b.root == (tmp_path / "b").resolve()
    # Original artifact is invisible from the new root
    assert b.list_revisions("x") == []
