import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_idle_resource_manager_runs_cleanup_after_idle_period():
    from services.edison_core.resource_protocol import IdleResourceManager

    manager = IdleResourceManager(idle_seconds=0)
    called = []

    def cleanup_callback():
        called.append("media")
        return {"status": "cleaned"}

    manager.register_cleanup("media", cleanup_callback)
    manager.begin_task("image_generation")
    manager.end_task("image_generation")

    result = manager.cleanup_if_idle()

    assert result["ran"] is True
    assert result["callbacks"]["media"] == {"status": "cleaned"}
    assert called == ["media"]


def test_idle_resource_manager_skips_cleanup_while_active():
    from services.edison_core.resource_protocol import IdleResourceManager

    manager = IdleResourceManager(idle_seconds=0)
    manager.register_cleanup("noop", lambda: {"status": "should_not_run"})
    manager.begin_task("swarm")

    result = manager.cleanup_if_idle()

    assert result["ran"] is False
    assert result["reason"] == "active_tasks"
    assert result["active_tasks"] == {"swarm": 1}


def test_swarm_session_prune_and_compact():
    from services.edison_core.swarm_engine import (
        SwarmSession,
        end_session,
        get_session,
        prune_sessions,
        register_session,
    )

    compact_session = SwarmSession(user_request="compact me")
    compact_session.status = "done"
    compact_session.conversation = [{"agent": "A", "response": str(i)} for i in range(40)]
    compact_session.shared_notes = [f"note {i}" for i in range(20)]
    compact_session._shared_note_set = {f"note {i}" for i in range(20)}
    register_session(compact_session)

    stale_session = SwarmSession(user_request="expire me")
    stale_session.status = "done"
    stale_session.last_activity_at = time.time() - 10_000
    register_session(stale_session)
    stale_session.last_activity_at = time.time() - 10_000

    stats = prune_sessions(ttl_seconds=60, compact_completed=True)

    assert stats["removed"] >= 1
    assert len(compact_session.conversation) <= 24
    assert len(compact_session.shared_notes) <= 12
    assert get_session(stale_session.session_id) is None

    end_session(compact_session.session_id)