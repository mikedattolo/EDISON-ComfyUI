"""Tests for Phase 2 modules: jobs center, artifact revisions, command
palette, conversation index, scheduled-ops wrappers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


# ── Scheduled ops ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_scheduled_routes_to_correct_lane():
    from services.edison_core.gpu_scheduler import GPUScheduler
    from services.edison_core.scheduled_ops import run_scheduled, lane_for

    GPUScheduler.reset_instance()
    sch = GPUScheduler.get_instance()

    assert lane_for("image") == "image"
    assert lane_for("video") == "video"
    assert lane_for("nonsense") == "background"

    seen = {}

    async def work():
        seen["ran"] = True
        return "ok"

    res = await run_scheduled("image", work, job_id="job-123")
    assert res == "ok"
    assert seen["ran"]
    info = sch.lane_info("image")
    assert info["completed"] == 1


# ── Artifact revisions ─────────────────────────────────────────────

def test_artifact_revisions_roundtrip(tmp_path: Path):
    from services.edison_core.artifact_revisions import ArtifactRevisionStore

    ArtifactRevisionStore.reset_instance()
    store = ArtifactRevisionStore.get_instance(root=tmp_path)
    rec1 = store.add_revision("artA", "hello", title="Test", kind="document")
    rec2 = store.add_revision("artA", "hello world", title="Test")

    revs = store.list_revisions("artA")
    assert len(revs) == 2

    fetched = store.get_revision("artA", rec1["revision_id"])
    assert fetched is not None
    assert fetched["content"] == "hello"

    diff = store.diff("artA", rec1["revision_id"], rec2["revision_id"])
    assert diff["a_exists"] and diff["b_exists"]
    assert diff["b_chars"] > diff["a_chars"]

    restored = store.restore("artA", rec1["revision_id"])
    assert restored["metadata"]["restored_from"] == rec1["revision_id"]
    assert len(store.list_revisions("artA")) == 3


def test_artifact_revisions_rejects_bad_id(tmp_path: Path):
    from services.edison_core.artifact_revisions import ArtifactRevisionStore

    ArtifactRevisionStore.reset_instance()
    store = ArtifactRevisionStore.get_instance(root=tmp_path)
    with pytest.raises(ValueError):
        store.add_revision("../escape", "x")


# ── Command palette ────────────────────────────────────────────────

def test_command_palette_search_finds_matches():
    from services.edison_core.command_palette import get_palette

    p = get_palette()
    cmds = p.search("make a logo for adoro pizza")
    ids = [c.id for c in cmds]
    assert "branding.logo" in ids


def test_command_palette_categories():
    from services.edison_core.command_palette import get_palette

    p = get_palette()
    branding = p.by_category("branding")
    assert branding
    assert all(c.category == "branding" for c in branding)


def test_command_palette_empty_query():
    from services.edison_core.command_palette import get_palette

    assert get_palette().search("") == []


# ── Conversation index + context usage ─────────────────────────────

def _seed_chat(tmp_path: Path, name: str, title: str, messages: list[dict]) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"id": name, "title": title, "messages": messages}))
    return path


def test_conversation_search_finds_topical_match(tmp_path: Path):
    from services.edison_core.conversation_index import search_conversations

    _seed_chat(tmp_path, "c1", "Adoro Pizza brand kit", [
        {"role": "user", "content": "Build a brand kit for Adoro Pizza"},
        {"role": "assistant", "content": "Sure, here are some logo concepts."},
    ])
    _seed_chat(tmp_path, "c2", "Generic chat", [
        {"role": "user", "content": "What's the weather?"},
    ])

    hits = search_conversations("adoro pizza brand", chats_dir=tmp_path)
    assert hits
    assert hits[0].chat_id == "c1"
    assert "Adoro" in hits[0].snippet or "pizza" in hits[0].snippet.lower()


def test_context_usage_thresholds():
    from services.edison_core.conversation_index import context_usage

    big = " ".join(["word"] * 4000)
    msgs = [{"role": "user", "content": big}]
    res = context_usage(msgs, context_window=2048)
    assert res["status"] == "critical"
    assert res["used_tokens"] >= 1000

    res2 = context_usage([{"role": "user", "content": "hi"}], context_window=8192)
    assert res2["status"] == "ok"
    assert res2["available_tokens"] > 0


# ── Jobs center ────────────────────────────────────────────────────

def test_jobs_summary_shape():
    from services.edison_core.jobs_center import summary

    out = summary()
    assert "counts_by_status" in out
    assert "counts_by_type" in out
    assert "lanes" in out
