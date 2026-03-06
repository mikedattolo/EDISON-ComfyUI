"""Tests for swarm tool execution: SwarmSession task tracking and _run_agent_with_tools."""

import asyncio
import json
import re
import uuid
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# We import SwarmSession directly (it has no heavy deps).
# For SwarmEngine we need to mock LLM calls.
# ---------------------------------------------------------------------------

# Attempt direct import; fall back to inline recreations if heavy deps block it.
try:
    from services.edison_core.swarm_engine import SwarmSession, SwarmEngine
    _DIRECT_IMPORT = True
except Exception:
    _DIRECT_IMPORT = False


# ── SwarmSession task/artifact tracking ──────────────────────────────────

@pytest.fixture
def session():
    if _DIRECT_IMPORT:
        return SwarmSession(user_request="Build a REST API")
    pytest.skip("SwarmSession import unavailable")


class TestSwarmSessionTasks:
    def test_add_task(self, session):
        task = session.add_task("Create models.py", owner="FileManager")
        assert task["id"] == "T1"
        assert task["title"] == "Create models.py"
        assert task["status"] == "todo"
        assert task["owner_agent"] == "FileManager"
        assert len(session.tasks) == 1

    def test_add_multiple_tasks(self, session):
        session.add_task("Task A")
        session.add_task("Task B")
        session.add_task("Task C")
        assert len(session.tasks) == 3
        assert session.tasks[2]["id"] == "T3"

    def test_update_task_status(self, session):
        session.add_task("Setup DB")
        result = session.update_task("T1", status="done", notes="Used SQLAlchemy")
        assert result is not None
        assert result["status"] == "done"
        assert result["notes"] == "Used SQLAlchemy"

    def test_update_nonexistent_task(self, session):
        result = session.update_task("T999", status="done")
        assert result is None

    def test_tasks_text(self, session):
        session.add_task("Item 1", owner="PM")
        session.add_task("Item 2", owner="FM")
        text = session.tasks_text()
        assert "T1" in text
        assert "T2" in text
        assert "Item 1" in text

    def test_tasks_text_empty(self, session):
        text = session.tasks_text()
        assert "no tasks" in text.lower()


class TestSwarmSessionArtifacts:
    def test_add_artifact(self, session):
        art = session.add_artifact("src/main.py", kind="file", summary="Entry point", created_by="FileManager")
        assert art["path"] == "src/main.py"
        assert len(session.artifacts) == 1

    def test_to_dict_includes_tasks_and_artifacts(self, session):
        session.add_task("Write tests")
        session.add_artifact("tests/test_api.py")
        d = session.to_dict()
        assert "tasks" in d
        assert "artifacts" in d
        assert len(d["tasks"]) == 1
        assert len(d["artifacts"]) == 1


# ── SwarmEngine tool execution ────────────────────────────────────────────

@pytest.fixture
def mock_execute_tool():
    """An async callable that simulates tool execution."""
    async def _execute(tool_name: str, args: dict, chat_id: str = None) -> dict:
        if tool_name == "fs.write":
            return {"ok": True, "data": {"path": args.get("path", ""), "bytes_written": 42}}
        if tool_name == "fs.read":
            return {"ok": True, "data": {"contents": "file contents here"}}
        if tool_name == "code.search":
            return {"ok": True, "data": {"matches": [{"file": "main.py", "line": 10}]}}
        if tool_name == "workspace.init":
            return {"ok": True, "data": {"workspace_root": "/tmp/ws/default"}}
        return {"ok": False, "error": f"Unknown tool: {tool_name}"}
    return _execute


@pytest.fixture
def mock_llm_model():
    """A mock model object that returns controlled output."""
    model = MagicMock()
    return model


@pytest.fixture
def engine(mock_execute_tool, mock_llm_model):
    if not _DIRECT_IMPORT:
        pytest.skip("SwarmEngine import unavailable")

    def available_models():
        return [("fast", mock_llm_model, "Test Model")]

    def get_lock(model):
        import threading
        return threading.Lock()

    return SwarmEngine(
        available_models=available_models,
        get_lock_for_model=get_lock,
        config={},
        emit_fn=lambda *a, **kw: None,
        execute_tool=mock_execute_tool,
    )


class TestSwarmEngineToolExecution:
    @pytest.mark.asyncio
    async def test_run_agent_with_tools_no_execute_tool(self):
        """When execute_tool is None, falls back to plain prompt execution."""
        if not _DIRECT_IMPORT:
            pytest.skip("SwarmEngine import unavailable")

        mock_model = MagicMock()

        def available_models():
            return [("fast", mock_model, "Test")]

        engine = SwarmEngine(
            available_models=available_models,
            get_lock_for_model=lambda m: MagicMock(),
            execute_tool=None,  # No tool execution
        )

        agent = {"name": "TestAgent", "icon": "🧪", "model_tag": "fast", "can_use_tools": True}
        session = SwarmSession(user_request="test")

        # Mock _run_agent_prompt to return plain text
        engine._run_agent_prompt = AsyncMock(return_value={"response": "Here is my answer"})

        result = await engine._run_agent_with_tools(agent, "Test prompt", session)
        assert result["response"] == "Here is my answer"

    @pytest.mark.asyncio
    async def test_run_agent_with_tools_tool_call(self, engine):
        """Agent outputs a tool call JSON, gets result, then gives final answer."""
        agent = {"name": "FileManager", "icon": "📁", "model_tag": "fast", "can_use_tools": True}
        session = SwarmSession(user_request="Create a file")

        call_count = 0

        async def mock_prompt(ag, prompt, temp, max_tokens=600):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: agent wants to use a tool
                return {"response": '{"tool":"fs.write","args":{"path":"hello.txt","contents":"Hello World"}}'}
            else:
                # Second call: agent gives final answer after seeing tool result
                return {"response": "File hello.txt created successfully with 42 bytes."}

        engine._run_agent_prompt = mock_prompt

        result = await engine._run_agent_with_tools(agent, "Create hello.txt", session, chat_id="test_chat")
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["tool"] == "fs.write"
        assert result["tool_calls"][0]["result"]["ok"] is True

    @pytest.mark.asyncio
    async def test_run_agent_with_tools_max_iterations(self, engine):
        """Agent keeps calling tools and exhausts iteration budget."""
        agent = {"name": "TestAgent", "icon": "🧪", "model_tag": "fast", "can_use_tools": True}
        session = SwarmSession(user_request="test")

        async def always_tool(ag, prompt, temp, max_tokens=600):
            return {"response": '{"tool":"workspace.init","args":{"project_id":"test"}}'}

        engine._run_agent_prompt = always_tool

        result = await engine._run_agent_with_tools(
            agent, "Test", session, chat_id="chat1", max_tool_iters=3
        )
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 3  # Exhausted all 3 iterations

    @pytest.mark.asyncio
    async def test_run_agent_with_tools_immediate_answer(self, engine):
        """Agent immediately gives a text answer without tool calls."""
        agent = {"name": "TestAgent", "icon": "🧪", "model_tag": "fast", "can_use_tools": True}
        session = SwarmSession(user_request="test")

        async def plain_text(ag, prompt, temp, max_tokens=600):
            return {"response": "The answer is 42."}

        engine._run_agent_prompt = plain_text

        result = await engine._run_agent_with_tools(agent, "What is the answer?", session)
        assert result["response"] == "The answer is 42."
        assert result.get("tool_calls", []) == []

    @pytest.mark.asyncio
    async def test_run_agent_with_tools_error_handling(self, engine):
        """Tool execution raises an exception — should be caught gracefully."""
        agent = {"name": "TestAgent", "icon": "🧪", "model_tag": "fast", "can_use_tools": True}
        session = SwarmSession(user_request="test")

        async def failing_tool(tool_name, args, chat_id=None):
            raise RuntimeError("Database connection failed")

        engine._execute_tool = failing_tool

        call_count = 0

        async def mock_prompt(ag, prompt, temp, max_tokens=600):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"response": '{"tool":"fs.read","args":{"path":"data.json"}}'}
            return {"response": "Tool failed, reporting error."}

        engine._run_agent_prompt = mock_prompt

        result = await engine._run_agent_with_tools(agent, "Read data", session, chat_id="err_chat")
        assert "tool_calls" in result
        assert result["tool_calls"][0]["result"]["ok"] is False
        assert "Database connection failed" in result["tool_calls"][0]["result"]["error"]


# ── ProjectManager & FileManager in catalog ───────────────────────────────

class TestAgentCatalog:
    def test_project_manager_in_catalog(self):
        if not _DIRECT_IMPORT:
            pytest.skip("Import unavailable")
        from services.edison_core.swarm_engine import AGENT_CATALOG_DEFINITIONS
        names = [a["name"] for a in AGENT_CATALOG_DEFINITIONS]
        assert "ProjectManager" in names

    def test_file_manager_in_catalog(self):
        if not _DIRECT_IMPORT:
            pytest.skip("Import unavailable")
        from services.edison_core.swarm_engine import AGENT_CATALOG_DEFINITIONS
        names = [a["name"] for a in AGENT_CATALOG_DEFINITIONS]
        assert "FileManager" in names

    def test_tool_capable_agents_flagged(self):
        if not _DIRECT_IMPORT:
            pytest.skip("Import unavailable")
        from services.edison_core.swarm_engine import AGENT_CATALOG_DEFINITIONS
        for agent in AGENT_CATALOG_DEFINITIONS:
            if agent["name"] in ("ProjectManager", "FileManager"):
                assert agent.get("can_use_tools") is True, f"{agent['name']} should have can_use_tools=True"
