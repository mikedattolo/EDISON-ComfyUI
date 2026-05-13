def test_agent_session_store_tracks_events():
    from services.edison_core.routes.agent_live import AgentSessionStore

    store = AgentSessionStore()
    session = store.create("research a page", session_id="agent_test")

    store.record_event("agent_test", {"type": "agent_step", "title": "Open browser", "status": "running", "ts": 10})
    store.record_event("agent_test", {"type": "file_diff", "path": "/tmp/out.md", "ts": 11})

    current = store.get(session["session_id"])

    assert current["state"] == "running"
    assert current["recent_actions"][0]["title"] == "Open browser"
    assert current["artifacts"][0]["path"] == "/tmp/out.md"


def test_agent_session_store_validates_state():
    import pytest
    from services.edison_core.routes.agent_live import AgentSessionStore

    store = AgentSessionStore()
    store.create(session_id="agent_test")

    with pytest.raises(ValueError):
        store.update("agent_test", state="mystery")
