def test_memory_scope_precedence():
    from services.edison_core.memory_scopes import MemoryScopeManager

    manager = MemoryScopeManager()

    assert manager.get_scope(project_id="p1", chat_id="c1").scope_id == "project:p1"
    assert manager.get_scope(chat_id="c1").scope_id == "chat:c1"
    assert manager.get_scope(session_id="s1").scope_id == "session:s1"
    assert manager.get_scope().scope_id == "global"


def test_memory_hit_dedupe_keeps_highest_confidence():
    from services.edison_core.memory_scopes import dedupe_memory_hits

    hits = [
        {"content": "Client likes blue", "score": 0.5, "confidence": 0.3, "source": "a"},
        {"content": "Client likes blue", "score": 0.4, "confidence": 0.9, "source": "b"},
    ]

    deduped = dedupe_memory_hits(hits)

    assert len(deduped) == 1
    assert deduped[0]["source"] == "b"


def test_memory_hit_ranking_applies_scope_bonus():
    from services.edison_core.memory_scopes import rank_memory_hits

    hits = [
        {"content": "global fact", "score": 0.8, "confidence": 0.5, "scope_id": "global"},
        {"content": "project fact", "score": 0.75, "confidence": 0.5, "scope_id": "project:p1"},
    ]

    ranked = rank_memory_hits(hits, preferred_scope_ids=["project:p1"])

    assert ranked[0]["content"] == "project fact"
    assert ranked[0]["rank_explanation"]["scope_bonus"] == 0.1
