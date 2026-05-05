"""Regression tests for the audit pass.

Covers the shared atomic-write helper and the conversation_state
locking fix.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from services.edison_core.safe_io import (
    atomic_write_json,
    atomic_write_text,
    read_json,
)


def test_atomic_write_json_creates_file(tmp_path: Path):
    p = tmp_path / "sub" / "data.json"
    atomic_write_json(p, {"a": 1, "b": [2, 3]})
    assert p.exists()
    assert json.loads(p.read_text()) == {"a": 1, "b": [2, 3]}


def test_atomic_write_does_not_leave_tempfile(tmp_path: Path):
    p = tmp_path / "data.json"
    atomic_write_json(p, {"x": 1})
    leftover = [name for name in os.listdir(tmp_path) if name != "data.json"]
    assert leftover == []


def test_atomic_write_overwrites_existing(tmp_path: Path):
    p = tmp_path / "data.json"
    atomic_write_json(p, {"v": 1})
    atomic_write_json(p, {"v": 2})
    assert json.loads(p.read_text()) == {"v": 2}


def test_atomic_write_text_works(tmp_path: Path):
    p = tmp_path / "note.txt"
    atomic_write_text(p, "hello world")
    assert p.read_text() == "hello world"


def test_read_json_missing_returns_default(tmp_path: Path):
    assert read_json(tmp_path / "nope.json", default={"d": 1}) == {"d": 1}


def test_read_json_corrupt_returns_default(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    assert read_json(p, default=[]) == []


def test_read_json_roundtrip(tmp_path: Path):
    p = tmp_path / "ok.json"
    atomic_write_json(p, {"k": "v"})
    assert read_json(p) == {"k": "v"}


# ---------------------------------------------------------------------------
# Conversation state thread-safety
# ---------------------------------------------------------------------------


def test_conversation_state_concurrent_increment():
    from services.state.conversation_state import ConversationStateManager

    mgr = ConversationStateManager()
    session = "concurrent-test"
    mgr.get_state(session)

    iterations = 200
    threads = 8

    def worker():
        for _ in range(iterations):
            mgr.increment_turn(session)

    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    state = mgr.get_state(session)
    # If the lock works, we get exactly threads * iterations increments.
    assert state.turn_count == threads * iterations


def test_conversation_state_concurrent_history_notes():
    from services.state.conversation_state import ConversationStateManager

    mgr = ConversationStateManager()
    session = "history-test"
    mgr.get_state(session)

    threads = 4
    notes_per_thread = 50

    def worker(idx: int):
        for i in range(notes_per_thread):
            mgr.add_history_note(session, f"thread-{idx}-note-{i}", max_notes=10000)

    workers = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    state = mgr.get_state(session)
    assert len(state.history_summary) == threads * notes_per_thread
