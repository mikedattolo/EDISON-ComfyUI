"""Tests for Phase 1 additive modules: scheduler, retry, citations, and
artifact streaming.

These tests are self-contained and do not require vLLM, GPUs, or any of the
heavy EDISON services to be running.
"""

from __future__ import annotations

import asyncio

import pytest


# ── GPU scheduler ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scheduler_runs_lane_with_telemetry():
    from services.edison_core.gpu_scheduler import GPUScheduler

    GPUScheduler.reset_instance()
    sch = GPUScheduler.get_instance()

    async def work():
        await asyncio.sleep(0.01)
        return "ok"

    result = await sch.run("chat", work, metadata={"who": "test"})
    assert result == "ok"

    info = sch.lane_info("chat")
    assert info["completed"] == 1
    assert info["in_flight"] == 0
    telemetry = sch.telemetry()
    assert "chat" in telemetry["lanes"]
    assert telemetry["lanes"]["chat"]["completed"] == 1


@pytest.mark.asyncio
async def test_scheduler_concurrency_limits():
    from services.edison_core.gpu_scheduler import GPUScheduler

    GPUScheduler.reset_instance()
    sch = GPUScheduler(lane_concurrency={"image": 1, **{
        k: 4 for k in ["chat", "code", "vision", "tool", "video", "music",
                       "mesh", "cad", "background"]
    }})
    GPUScheduler._instance = sch

    counter = {"max": 0, "active": 0}

    async def slow():
        counter["active"] += 1
        counter["max"] = max(counter["max"], counter["active"])
        await asyncio.sleep(0.05)
        counter["active"] -= 1
        return True

    await asyncio.gather(
        sch.run("image", slow),
        sch.run("image", slow),
        sch.run("image", slow),
    )
    assert counter["max"] == 1


@pytest.mark.asyncio
async def test_scheduler_records_errors():
    from services.edison_core.gpu_scheduler import GPUScheduler

    GPUScheduler.reset_instance()
    sch = GPUScheduler.get_instance()

    async def boom():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        await sch.run("tool", boom)

    info = sch.lane_info("tool")
    assert info["errors"] == 1


# ── Retry ───────────────────────────────────────────────────────────

def test_retry_sync_eventually_succeeds():
    from services.edison_core.retry import RetryPolicy, retry_sync

    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    result = retry_sync(flaky, policy=RetryPolicy(max_attempts=5, base_delay=0.0, jitter=0.0))
    assert result == "ok"
    assert attempts["n"] == 3


def test_retry_sync_gives_up_and_raises():
    from services.edison_core.retry import RetryError, RetryPolicy, retry_sync

    def always_fail():
        raise ValueError("nope")

    with pytest.raises(RetryError) as excinfo:
        retry_sync(always_fail, policy=RetryPolicy(max_attempts=2, base_delay=0.0, jitter=0.0))
    assert len(excinfo.value.attempts) == 2


@pytest.mark.asyncio
async def test_retry_async_succeeds():
    from services.edison_core.retry import RetryPolicy, retry_async

    state = {"n": 0}

    async def flaky():
        state["n"] += 1
        if state["n"] < 2:
            raise TimeoutError("slow")
        return 42

    result = await retry_async(
        flaky,
        policy=RetryPolicy(max_attempts=3, base_delay=0.0, jitter=0.0),
    )
    assert result == 42


def test_friendly_error_messages():
    from services.edison_core.retry import friendly_error, RetryError

    assert "timed out" in friendly_error(asyncio.TimeoutError(), action="search")
    assert "Could not reach" in friendly_error(ConnectionError(), action="connector")
    msg = friendly_error(ValueError("bad input"), action="parse")
    assert "ValueError" in msg


# ── Citations ───────────────────────────────────────────────────────

def test_citations_normalize_and_dedupe():
    from services.edison_core.citations import normalize_hits

    hits = [
        {"url": "https://a.com", "title": "A", "snippet": "first"},
        {"link": "https://a.com", "name": "A duplicate"},  # dedup
        {"href": "https://b.com", "summary": "second"},
    ]
    cits = normalize_hits(hits, source="web")
    assert len(cits) == 2
    assert cits[0].url == "https://a.com"
    assert cits[1].snippet == "second"


def test_citations_attach_to_text_appends_block():
    from services.edison_core.citations import normalize_hits, attach_citations_to_text

    cits = normalize_hits([{"url": "https://x.test", "title": "X"}])
    text = attach_citations_to_text("Body of answer.", cits)
    assert "**Sources**" in text
    assert "https://x.test" in text


def test_citations_bundle_shape():
    from services.edison_core.citations import normalize_hits, bundle

    cits = normalize_hits([{"url": "https://x.test", "title": "X"}])
    payload = bundle(cits, request_id="req1")
    assert payload["count"] == 1
    assert payload["request_id"] == "req1"
    assert payload["items"][0]["title"] == "X"


# ── Artifact stream ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_artifact_stream_emits_events_and_revision():
    from services.edison_core.artifact_stream import ArtifactStream

    stream = ArtifactStream(kind="document", title="Test")

    async def producer():
        await stream.push("Hello ")
        await stream.push("world")
        await stream.commit(metadata={"final": True})
        await stream.close()

    events = []

    async def consumer():
        async for ev in stream:
            events.append(ev)

    await asyncio.gather(producer(), consumer())
    types = [e["type"] for e in events]
    assert types[0] == "artifact.delta"
    assert "artifact.revision" in types
    assert types[-1] == "artifact.done"
    assert stream.current_text == "Hello world"
    assert len(stream.revisions) == 1


@pytest.mark.asyncio
async def test_artifact_stream_diff_helper():
    from services.edison_core.artifact_stream import ArtifactStream

    stream = ArtifactStream()
    await stream.push("v1\n")
    rev1 = await stream.commit()
    await stream.push("v2 line\n")
    rev2 = await stream.commit()
    await stream.close()

    diff = stream.diff(rev1.revision_id, rev2.revision_id)
    assert diff["a_exists"] and diff["b_exists"]
    assert diff["b_lines"] >= diff["a_lines"]
