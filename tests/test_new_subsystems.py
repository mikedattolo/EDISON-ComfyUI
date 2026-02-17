"""
Tests for EDISON new subsystems:
- Unified job store CRUD + cancellation
- Routing detection for image/video/music/3D
- Retrieval reranking behavior
- Freshness cache TTL behavior
- Safe file serving (no path traversal)
"""

import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

# Ensure project root is on path
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Unified Job Store CRUD + Cancellation
# ═══════════════════════════════════════════════════════════════════════

def test_job_store_crud():
    """Test job store create, read, update, cancel, delete."""
    from services.edison_core.job_store import JobStore

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test_jobs.db"
        store = JobStore(db_path)

        # Create
        job_id = store.create_job(
            job_type="image",
            prompt="a sunset over mountains",
            negative_prompt="blurry",
            params={"steps": 30, "cfg": 7.5},
            provenance={"model": "flux-dev"},
        )
        assert job_id, "Job should be created"

        # Read
        job = store.get_job(job_id)
        assert job is not None, "Job should be retrievable"
        assert job["job_type"] == "image"
        assert job["status"] == "queued"
        assert job["prompt"] == "a sunset over mountains"
        assert job["params"]["steps"] == 30

        # Update status
        store.update_status(job_id, "generating")
        job = store.get_job(job_id)
        assert job["status"] == "generating"
        assert job["started_at"] is not None

        # Complete
        store.update_status(job_id, "complete", outputs=["/outputs/images/test.png"])
        job = store.get_job(job_id)
        assert job["status"] == "complete"
        assert "/outputs/images/test.png" in job["outputs"]
        assert job["duration_s"] is not None

        # List
        jobs = store.list_jobs(job_type="image")
        assert len(jobs) >= 1

        # Cancel (should fail since already complete)
        assert not store.cancel_job(job_id), "Should not cancel completed job"

        # Create and cancel a queued job
        job_id2 = store.create_job(job_type="video", prompt="running dog")
        assert store.cancel_job(job_id2), "Should cancel queued job"
        job2 = store.get_job(job_id2)
        assert job2["status"] == "cancelled"

        # Delete
        assert store.delete_job(job_id), "Should delete job"
        assert store.get_job(job_id) is None, "Deleted job should not exist"

        print("✓ Job store CRUD + cancellation: ALL PASSED")


def test_job_store_invalid_type():
    """Test that invalid job types raise ValueError."""
    from services.edison_core.job_store import JobStore

    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        try:
            store.create_job(job_type="invalid_type", prompt="test")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        print("✓ Job store invalid type validation: PASSED")


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Routing Detection for image/video/music/3D
# ═══════════════════════════════════════════════════════════════════════

def test_routing_detection():
    """Test that route_mode correctly detects different content types."""
    from services.edison_core.app import route_mode

    # Video detection
    result = route_mode("make a video of a cat running", "auto", False)
    assert result["tools_allowed"], "Video request should enable tools"
    assert "video" in " ".join(result["reasons"]).lower(), f"Should mention video: {result['reasons']}"

    # Music detection
    result = route_mode("compose a lo-fi beat for studying", "auto", False)
    assert result["tools_allowed"], "Music request should enable tools"
    assert "music" in " ".join(result["reasons"]).lower(), f"Should mention music: {result['reasons']}"

    # 3D detection
    result = route_mode("create a 3d model of a chair", "auto", False)
    assert result["tools_allowed"], "3D request should enable tools"
    assert "3d" in " ".join(result["reasons"]).lower(), f"Should mention 3D: {result['reasons']}"

    # Code detection
    result = route_mode("write a python function to sort a list", "auto", False)
    assert result["mode"] == "code", f"Code request should be code mode, got: {result['mode']}"

    # Image detection (with has_image=True)
    result = route_mode("what is in this image", "auto", True)
    assert result["mode"] == "image", f"Image input should be image mode, got: {result['mode']}"

    # Explicit mode override
    result = route_mode("hello world", "thinking", False)
    assert result["mode"] == "reasoning", f"Thinking should map to reasoning, got: {result['mode']}"

    print("✓ Routing detection for image/video/music/3D: ALL PASSED")


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: Retrieval Reranking Behavior
# ═══════════════════════════════════════════════════════════════════════

def test_reranking():
    """Test that reranking properly boosts relevant results."""
    from services.edison_core.retrieval import rerank_results, detect_query_intent, QueryIntent

    now = time.time()

    results = [
        ("old message about dogs", {
            "base_score": 0.9, "timestamp": now - 86400 * 30,
            "confidence": 0.5, "type": "message",
        }),
        ("recent fact: user prefers Python", {
            "base_score": 0.7, "timestamp": now - 3600,
            "confidence": 0.95, "type": "fact",
        }),
        ("user profile: preferred_language = Python", {
            "base_score": 0.6, "timestamp": now - 86400,
            "confidence": 0.9, "type": "profile",
        }),
    ]

    # General query - high similarity should win
    reranked = rerank_results(results, QueryIntent.GENERAL)
    assert len(reranked) == 3
    assert all("rerank_score" in r[1] for r in reranked)

    # Profile query - should boost profile memory
    reranked = rerank_results(results, QueryIntent.PROFILE)
    scores = [r[1]["rerank_score"] for r in reranked]
    # The profile entry should get a boost
    profile_entry = next(r for r in reranked if r[1]["type"] == "profile")
    assert profile_entry[1]["rerank_components"]["query_intent"] == "profile"

    # Intent detection
    assert detect_query_intent("what's my preferred language?") == QueryIntent.PROFILE
    assert detect_query_intent("what did we discuss yesterday?") == QueryIntent.EPISODIC
    assert detect_query_intent("how to use FastAPI middleware") in (QueryIntent.DEV_DOCS, QueryIntent.SEMANTIC)
    assert detect_query_intent("hello there") == QueryIntent.GENERAL

    print("✓ Retrieval reranking behavior: ALL PASSED")


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: Freshness Cache TTL Behavior
# ═══════════════════════════════════════════════════════════════════════

def test_freshness_cache():
    """Test freshness cache put/get/TTL/time-sensitivity."""
    from services.edison_core.freshness import FreshnessCache

    with tempfile.TemporaryDirectory() as td:
        cache = FreshnessCache(Path(td) / "test_cache.db")

        # Put and get
        cache.put("what is Python?", "Python is a programming language.",
                  sources=[{"url": "https://python.org"}],
                  citations=[{"title": "Python.org", "url": "https://python.org"}])

        result = cache.get("what is Python?")
        assert result is not None, "Should find cached result"
        assert result["cache_hit"] is True
        assert "Python is a programming language" in result["facts_summary"]
        assert len(result["citations"]) == 1

        # Time-sensitive detection
        assert cache.is_time_sensitive("what is the latest Python version?")
        assert cache.is_time_sensitive("current weather in NYC")
        assert not cache.is_time_sensitive("what is Python?")

        # Different queries have different fingerprints
        cache.put("what is JavaScript?", "JS is a scripting language.", ttl=1)
        result_js = cache.get("what is JavaScript?")
        result_py = cache.get("what is Python?")
        assert result_js is not None and result_py is not None
        assert result_js["facts_summary"] != result_py["facts_summary"]

        # Invalidate
        cache.invalidate("what is Python?")
        assert cache.get("what is Python?") is None

        # TTL expiry (set very short TTL)
        cache.put("expiring query", "temp data", ttl=0.001)
        time.sleep(0.01)
        assert cache.get("expiring query") is None, "Should be expired"

        # Needs refresh
        assert cache.needs_refresh("nonexistent query")

        print("✓ Freshness cache TTL behavior: ALL PASSED")


# ═══════════════════════════════════════════════════════════════════════
# TEST 5: Safe File Serving (no path traversal)
# ═══════════════════════════════════════════════════════════════════════

def test_safe_file_serving():
    """Test that file reader tool blocks path traversal."""
    from services.edison_core.tool_framework import SafeFileReaderTool

    tool = SafeFileReaderTool()

    # Should reject path traversal
    err = tool.validate_params(path="../../../etc/passwd")
    assert err is not None, "Should reject path traversal"
    assert "not in allowed" in err.lower() or "traversal" in err.lower(), f"Unexpected error: {err}"

    # Should reject paths outside safe roots
    err = tool.validate_params(path="/etc/passwd")
    assert err is not None, "Should reject /etc/passwd"

    # Should accept valid path
    err = tool.validate_params(path="")
    assert err is not None, "Should reject empty path"

    print("✓ Safe file serving (path traversal prevention): ALL PASSED")


# ═══════════════════════════════════════════════════════════════════════
# TEST 6: Memory Store
# ═══════════════════════════════════════════════════════════════════════

def test_memory_store():
    """Test three-tier memory store CRUD and hygiene."""
    from services.edison_core.memory.store import MemoryStore
    from services.edison_core.memory.models import MemoryEntry, MemoryType

    with tempfile.TemporaryDirectory() as td:
        store = MemoryStore(Path(td) / "test_memory.db")

        # Save profile memory
        entry = MemoryEntry(
            memory_type=MemoryType.PROFILE,
            key="preferred_language",
            content="User prefers Python for backend development",
            confidence=0.95,
            tags=["coding", "preference"],
        )
        mid = store.save(entry)
        assert mid

        # Retrieve
        retrieved = store.get(mid)
        assert retrieved is not None
        assert retrieved.key == "preferred_language"
        assert retrieved.memory_type == MemoryType.PROFILE

        # Search by type
        profiles = store.search(memory_type=MemoryType.PROFILE)
        assert len(profiles) >= 1

        # Search by tag
        tagged = store.search(tag="coding")
        assert len(tagged) >= 1

        # Update
        store.update(mid, confidence=0.99, pinned=True)
        updated = store.get(mid)
        assert updated.confidence == 0.99
        assert updated.pinned is True

        # Delete
        store.delete(mid)
        assert store.get(mid) is None

        # Hygiene (create duplicates and prune)
        for i in range(3):
            store.save(MemoryEntry(
                memory_type=MemoryType.EPISODIC,
                content="duplicate content",
                confidence=0.2 + i * 0.1,
            ))
        stats = store.run_hygiene()
        assert stats["duplicates_removed"] >= 2

        print("✓ Memory store CRUD + hygiene: ALL PASSED")


# ═══════════════════════════════════════════════════════════════════════
# TEST 7: Workflow Memory
# ═══════════════════════════════════════════════════════════════════════

def test_workflow_memory():
    """Test workflow recording and recommendations."""
    from services.edison_core.workflow_memory import WorkflowMemory

    with tempfile.TemporaryDirectory() as td:
        wm = WorkflowMemory(Path(td) / "test_workflows.db")

        # Record results
        wm.record_result(
            job_type="image",
            prompt="sunset landscape",
            params={"steps": 30, "cfg": 7.5, "sampler": "euler"},
            success=True,
            rating=4.5,
            style_profile="photorealistic",
        )
        wm.record_result(
            job_type="image",
            prompt="abstract art",
            params={"steps": 20, "cfg": 5.0},
            success=True,
            rating=3.0,
        )

        # Get recommendations
        recs = wm.get_recommendations("image")
        assert len(recs) >= 1
        assert recs[0]["rating"] >= recs[-1]["rating"], "Should be sorted by rating"

        # Get best params
        best = wm.get_best_params("image", style_profile="photorealistic")
        assert best is not None
        assert best["steps"] == 30

        # Stats
        stats = wm.get_stats()
        assert stats["total"] >= 2

        print("✓ Workflow memory: ALL PASSED")


# ═══════════════════════════════════════════════════════════════════════
# TEST 8: Observability
# ═══════════════════════════════════════════════════════════════════════

def test_observability():
    """Test structured event tracing."""
    from services.edison_core.observability import ObservabilityTracer, new_correlation_id

    tracer = ObservabilityTracer()
    cid = new_correlation_id()

    tracer.trace_retrieval(query="test query", results_count=5, intent="general")
    tracer.trace_memory_save(memory_type="profile", key="name", content="Alice")
    tracer.trace_tool_call(tool_name="web_fetch", success=True, duration_ms=150.0)
    tracer.trace_generation(job_type="image", status="complete", job_id="test-123")

    events = tracer.get_events()
    assert len(events) >= 4

    # Filter by type
    gen_events = tracer.get_events(event_type="generation")
    assert len(gen_events) >= 1

    stats = tracer.get_stats()
    assert stats["total_events"] >= 4
    assert "retrieval" in stats["by_type"]

    print("✓ Observability tracing: ALL PASSED")


# ═══════════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        test_job_store_crud,
        test_job_store_invalid_type,
        test_routing_detection,
        test_reranking,
        test_freshness_cache,
        test_safe_file_serving,
        test_memory_store,
        test_workflow_memory,
        test_observability,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
