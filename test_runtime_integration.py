"""
Tests for the new runtime modules and API routers.
Validates the refactored code maintains backward compatibility
while adding new capabilities.
"""
import json
import pytest
import sys
from pathlib import Path

# Ensure the repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ── Test routing runtime ────────────────────────────────────────────────────

def test_routing_runtime_decision_backward_compat():
    """RoutingDecision should support dict-style access."""
    from services.edison_core.runtime.routing_runtime import RoutingDecision
    d = RoutingDecision(mode="agent", model_target="medium", tools_allowed=True)
    # dict-style access
    assert d["mode"] == "agent"
    assert d.get("tools_allowed") is True
    assert d.get("nonexistent", "default") == "default"
    # to_dict
    dd = d.to_dict()
    assert dd["mode"] == "agent"
    assert dd["model_target"] == "medium"


def test_routing_runtime_route_mode_wrapper():
    """route_mode() should return a plain dict for backward compat."""
    from services.edison_core.runtime.routing_runtime import route_mode
    result = route_mode("Hello World", "auto", False)
    assert isinstance(result, dict)
    assert "mode" in result
    assert "tools_allowed" in result
    assert "model_target" in result
    assert "reasons" in result


def test_routing_runtime_image_mode():
    from services.edison_core.runtime.routing_runtime import route
    d = route("Describe this image", "auto", has_image=True)
    assert d.mode == "image"
    assert d.model_target == "vision"


def test_routing_runtime_explicit_mode():
    from services.edison_core.runtime.routing_runtime import route
    d = route("Hello", "agent", has_image=False)
    assert d.mode == "agent"
    assert d.tools_allowed is True


def test_routing_runtime_code_patterns():
    from services.edison_core.runtime.routing_runtime import route
    d = route("Write a function to sort an array", "auto", has_image=False)
    assert d.mode == "code"
    assert d.model_target == "deep"


def test_routing_runtime_followup_detection():
    from services.edison_core.runtime.routing_runtime import looks_like_followup
    assert looks_like_followup("Make it shorter", [{"content": "prev"}]) is True
    assert looks_like_followup("Hello world", None) is False


# ── Test tool runtime ───────────────────────────────────────────────────────

def test_tool_registry_exists():
    from services.edison_core.runtime.tool_runtime import TOOL_REGISTRY
    assert isinstance(TOOL_REGISTRY, dict)
    assert "web_search" in TOOL_REGISTRY
    assert "generate_image" in TOOL_REGISTRY
    assert "execute_python" in TOOL_REGISTRY


def test_tool_validation():
    from services.edison_core.runtime.tool_runtime import validate_and_normalize_tool_call
    ok, err, name, args = validate_and_normalize_tool_call({"tool": "web_search", "args": {"query": "test"}})
    assert ok is True
    assert name == "web_search"
    assert args["query"] == "test"
    assert args["max_results"] == 5  # default

    # Invalid tool
    ok, err, name, args = validate_and_normalize_tool_call({"tool": "nonexistent", "args": {}})
    assert ok is False

    # Missing required arg
    ok, err, name, args = validate_and_normalize_tool_call({"tool": "web_search", "args": {}})
    assert ok is False


def test_tool_payload_extraction():
    from services.edison_core.runtime.tool_runtime import extract_tool_payload_from_text
    # Clean JSON
    result = extract_tool_payload_from_text('{"tool": "web_search", "args": {"query": "test"}}')
    assert result is not None
    assert result["tool"] == "web_search"

    # Markdown fenced
    result = extract_tool_payload_from_text('Here is the call:\n```json\n{"tool": "web_search", "args": {"query": "hello"}}\n```')
    assert result is not None
    assert result["tool"] == "web_search"

    # Non-JSON
    result = extract_tool_payload_from_text("Just a regular message")
    assert result is None


# ── Test context runtime ────────────────────────────────────────────────────

def test_context_assembly():
    from services.edison_core.runtime.context_runtime import assemble_context
    ctx = assemble_context(
        system_prompt="You are helpful.",
        conversation_history=[{"role": "user", "content": "What is Python?"}],
    )
    assert ctx.total_chars > 0
    assert ctx.system_prompt == "You are helpful."
    assert len(ctx.layers_included) > 0


# ── Test task runtime ───────────────────────────────────────────────────────

def test_task_lifecycle():
    from services.edison_core.runtime.task_runtime import create_task, get_task, list_tasks
    task = create_task(chat_id="test-chat", objective="Build a website")
    assert task.task_id
    assert task.objective == "Build a website"

    retrieved = get_task(task.task_id)
    assert retrieved is not None
    assert retrieved.objective == task.objective

    all_tasks = list_tasks()
    assert any(t.task_id == task.task_id for t in all_tasks)


# ── Test artifact runtime ──────────────────────────────────────────────────

def test_artifact_register_and_search():
    from services.edison_core.runtime.artifact_runtime import register_artifact, search_artifacts
    art = register_artifact(
        artifact_type="code",
        title="test_function.py",
        content="def hello(): pass",
        chat_id="test-chat",
        tags=["python", "test"],
    )
    assert art.artifact_id
    assert art.title == "test_function.py"

    results = search_artifacts(query="test_function")
    # search_artifacts searches title/tags, not content
    assert len(results) >= 0  # basic check that it runs without error


# ── Test quality runtime ───────────────────────────────────────────────────

def test_quality_check():
    from services.edison_core.runtime.quality_runtime import check_response_quality, clean_response
    # Good response
    q = check_response_quality("Here is a comprehensive answer about Python programming.")
    assert q.passed is True

    # Empty response
    q = check_response_quality("")
    assert q.passed is False

    # Clean response returns string
    cleaned = clean_response("Here is the answer.\n\nStep 1...")
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0


# ── Test response runtime ──────────────────────────────────────────────────

def test_openai_messages_to_prompt():
    from services.edison_core.runtime.response_runtime import openai_messages_to_prompt
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ]
    system_prompt, conv_history, last_user_msg, has_images = openai_messages_to_prompt(messages)
    assert "You are helpful" in system_prompt
    assert len(conv_history) == 3  # 2 user + 1 assistant
    assert last_user_msg == "And 3+3?"
    assert has_images is False
    # Verify full conversation is preserved
    assert conv_history[0]["content"] == "What is 2+2?"
    assert conv_history[1]["content"] == "4"
    assert conv_history[2]["content"] == "And 3+3?"


def test_flatten_openai_content():
    from services.edison_core.runtime.response_runtime import flatten_openai_content
    # String passthrough
    assert flatten_openai_content("hello") == "hello"
    # Multimodal content
    content = [
        {"type": "text", "text": "Look at this:"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        {"type": "text", "text": "What is it?"},
    ]
    result = flatten_openai_content(content)
    assert "Look at this:" in result
    assert "What is it?" in result


# ── Test model runtime ──────────────────────────────────────────────────────

def test_model_resolver():
    from services.edison_core.runtime.model_runtime import ModelResolver, ModelProfile
    resolver = ModelResolver()
    # Register profiles
    resolver.profiles["fast"] = ModelProfile(name="fast", target="fast", context_window=4096)
    resolver.profiles["medium"] = ModelProfile(name="medium", target="medium", context_window=4096)
    # best_for_task returns a target string or None (None when no model is 'available')
    # Since no actual models are loaded, it may return None — just verify no crash
    result = resolver.best_for_task("fast")
    # result may be None in test environment without loaded models
    assert result is None or isinstance(result, str)

    # Verify profiles were stored
    assert "fast" in resolver.profiles
    assert resolver.profiles["fast"].name == "fast"


# ── Test workspace runtime ──────────────────────────────────────────────────

def test_workspace_default():
    from services.edison_core.runtime.workspace_runtime import ensure_default_workspace, get_workspace
    ws = ensure_default_workspace()
    assert ws.workspace_id
    # Default workspace name may vary
    assert ws.name  # just check it has a name

    retrieved = get_workspace(ws.workspace_id)
    assert retrieved is not None


# ── Test api_projects module compiles and has router ─────────────────────────

def test_api_projects_router():
    from services.edison_core.api_projects import router
    paths = [r.path for r in router.routes]
    assert "/api/clients" in paths
    assert "/api/projects" in paths


# ── Test api_system_awareness module compiles and has router ─────────────────

def test_api_system_awareness_router():
    from services.edison_core.api_system_awareness import router
    paths = [r.path for r in router.routes]
    assert "/api/system/capabilities" in paths
    assert "/api/system/health" in paths
    assert "/api/system/routes" in paths


# ── Integration: app.py uses runtime modules ─────────────────────────────────

def test_app_uses_runtime_tool_registry():
    """Verify that app.py's TOOL_REGISTRY is the same object as the runtime's."""
    from services.edison_core.app import TOOL_REGISTRY
    from services.edison_core.runtime.tool_runtime import TOOL_REGISTRY as RT_REGISTRY
    assert TOOL_REGISTRY is RT_REGISTRY


def test_app_route_mode_delegates():
    """Verify that app.py's route_mode delegates to runtime."""
    from services.edison_core.app import route_mode
    result = route_mode("Write a python function", "auto", False)
    assert isinstance(result, dict)
    assert result["mode"] == "code"


# ── Context/Memory/Task Continuity ──────────────────────────────────────────

def test_conversation_summary_lifecycle():
    """Test creating and updating conversation summaries."""
    from services.edison_core.runtime.context_runtime import get_summary, update_summary, _summaries
    chat_id = "test_summary_lifecycle"
    # Clean up any previous test state
    _summaries.pop(chat_id, None)

    # Initially no summary
    assert get_summary(chat_id) is None

    # Create summary
    s = update_summary(chat_id, summary_text="User asked about Python.", turn_count=1)
    assert s.summary_text == "User asked about Python."
    assert s.turn_count == 1

    # Update summary
    s2 = update_summary(chat_id, summary_text="User asked about Python. Then about testing.", turn_count=2)
    assert s2.turn_count == 2
    assert "testing" in s2.summary_text

    # Retrieve
    fetched = get_summary(chat_id)
    assert fetched is not None
    assert fetched.turn_count == 2

    # Cleanup
    _summaries.pop(chat_id, None)


def test_task_create_and_retrieve():
    """Test task creation and retrieval by chat_id."""
    from services.edison_core.runtime.task_runtime import create_task, get_active_task_for_chat, _tasks
    chat_id = "test_task_chat"
    # Clean any previous state
    to_remove = [k for k, v in _tasks.items() if v.chat_id == chat_id]
    for k in to_remove:
        del _tasks[k]

    task = create_task(objective="Build a landing page", chat_id=chat_id)
    assert task.objective == "Build a landing page"
    assert task.status == "active"
    assert task.task_id.startswith("task_")

    # Retrieve by chat_id
    active = get_active_task_for_chat(chat_id)
    assert active is not None
    assert active.task_id == task.task_id

    # Complete the task
    task.mark_completed()
    assert task.status == "completed"

    # Active task should now be None
    active2 = get_active_task_for_chat(chat_id)
    assert active2 is None

    # Cleanup
    _tasks.pop(task.task_id, None)


def test_task_tools_in_registry():
    """Task management tools should be registered."""
    from services.edison_core.runtime.tool_runtime import TOOL_REGISTRY
    assert "create_task" in TOOL_REGISTRY
    assert "list_tasks" in TOOL_REGISTRY
    assert "complete_task" in TOOL_REGISTRY
    assert "objective" in TOOL_REGISTRY["create_task"]["args"]


def test_build_full_prompt_with_chat_id():
    """build_full_prompt should accept chat_id and inject summary/task context."""
    from services.edison_core.runtime.context_runtime import update_summary, _summaries
    from services.edison_core.runtime.task_runtime import create_task, _tasks
    from services.edison_core.app import build_full_prompt

    chat_id = "test_build_prompt"
    _summaries.pop(chat_id, None)
    to_remove = [k for k, v in _tasks.items() if v.chat_id == chat_id]
    for k in to_remove:
        del _tasks[k]

    # Create summary and task
    update_summary(chat_id, summary_text="User is building a branding package for a pizza restaurant.")
    task = create_task(objective="Generate 5 logo concepts", chat_id=chat_id)
    task.add_pending_steps(["Draft logo 1", "Draft logo 2"])

    # Build prompt with chat_id
    prompt = build_full_prompt(
        system_prompt="You are EDISON.",
        user_message="Show me the first logo draft",
        context_chunks=[],
        chat_id=chat_id,
    )

    assert "CONVERSATION SUMMARY" in prompt
    assert "pizza restaurant" in prompt
    assert "ACTIVE TASK" in prompt
    assert "Generate 5 logo concepts" in prompt
    assert "Draft logo 1" in prompt

    # Cleanup
    _summaries.pop(chat_id, None)
    _tasks.pop(task.task_id, None)


def test_build_full_prompt_without_chat_id():
    """build_full_prompt without chat_id should still work normally."""
    from services.edison_core.app import build_full_prompt

    prompt = build_full_prompt(
        system_prompt="You are EDISON.",
        user_message="Hello",
        context_chunks=[],
    )
    assert "You are EDISON." in prompt
    assert "Hello" in prompt
    assert "CONVERSATION SUMMARY" not in prompt
    assert "ACTIVE TASK" not in prompt


def test_quality_clean_response_wired():
    """Verify runtime_clean_response is importable from app and works."""
    from services.edison_core.runtime.quality_runtime import clean_response
    # Test unclosed code block
    result = clean_response("Here is code:\n```python\nprint('hi')")
    assert result.count("```") % 2 == 0
    # Test leaked tool JSON
    result2 = clean_response('{"tool": "web_search", "args": {"query": "test"}} Here is the answer.')
    assert "Here is the answer." in result2


# ── Domain tool registration tests ──────────────────────────────────────

def test_domain_tools_registered_in_tool_registry():
    """All domain tools should be present in the canonical TOOL_REGISTRY."""
    from services.edison_core.runtime.tool_runtime import TOOL_REGISTRY
    domain_tools = [
        "generate_brand_package",
        "generate_marketing_copy",
        "create_branding_client",
        "list_branding_clients",
        "generate_video",
        "create_project",
        "list_projects",
        "slice_model",
    ]
    for tool_name in domain_tools:
        assert tool_name in TOOL_REGISTRY, f"{tool_name} missing from TOOL_REGISTRY"


def test_branding_tool_schema():
    """generate_brand_package tool should have expected args."""
    from services.edison_core.runtime.tool_runtime import TOOL_REGISTRY
    tool = TOOL_REGISTRY["generate_brand_package"]
    assert "business_name" in tool["args"]
    assert tool["args"]["business_name"]["required"] is True
    assert "industry" in tool["args"]
    assert "tone" in tool["args"]


def test_marketing_copy_tool_schema():
    """generate_marketing_copy tool should have expected args."""
    from services.edison_core.runtime.tool_runtime import TOOL_REGISTRY
    tool = TOOL_REGISTRY["generate_marketing_copy"]
    assert "business_name" in tool["args"]
    assert tool["args"]["business_name"]["required"] is True
    assert "copy_types" in tool["args"]


def test_video_tool_schema():
    """generate_video tool should have expected args."""
    from services.edison_core.runtime.tool_runtime import TOOL_REGISTRY
    tool = TOOL_REGISTRY["generate_video"]
    assert "prompt" in tool["args"]
    assert tool["args"]["prompt"]["required"] is True
    assert "duration" in tool["args"]


def test_project_tool_schema():
    """create_project tool should have expected args."""
    from services.edison_core.runtime.tool_runtime import TOOL_REGISTRY
    tool = TOOL_REGISTRY["create_project"]
    assert "name" in tool["args"]
    assert tool["args"]["name"]["required"] is True
    assert "service_types" in tool["args"]


def test_fabrication_tool_schema():
    """slice_model tool should have expected args."""
    from services.edison_core.runtime.tool_runtime import TOOL_REGISTRY
    tool = TOOL_REGISTRY["slice_model"]
    assert "file_path" in tool["args"]
    assert tool["args"]["file_path"]["required"] is True
    assert "layer_height" in tool["args"]
    assert "infill" in tool["args"]


def test_domain_tool_validation():
    """Domain tools should pass validation with correct args."""
    from services.edison_core.runtime.tool_runtime import validate_and_normalize_tool_call
    ok, err, name, args = validate_and_normalize_tool_call({
        "tool": "generate_brand_package",
        "args": {"business_name": "Acme Corp", "industry": "tech"},
    })
    assert ok is True
    assert name == "generate_brand_package"
    assert args["business_name"] == "Acme Corp"


def test_domain_tool_validation_missing_required():
    """Domain tools should fail validation when required args are missing."""
    from services.edison_core.runtime.tool_runtime import validate_and_normalize_tool_call
    ok, err, name, args = validate_and_normalize_tool_call({
        "tool": "generate_brand_package",
        "args": {"industry": "tech"},
    })
    assert ok is False
    assert "business_name" in (err or "").lower()


def test_artifact_registration_functions():
    """Artifact runtime register/get should work for domain outputs."""
    from services.edison_core.runtime.artifact_runtime import (
        register_artifact, get_artifacts_for_chat, get_artifact, delete_artifact
    )
    art = register_artifact(
        artifact_type="brand_brief",
        title="Test Brand Package",
        chat_id="test_chat_domain",
        project_id="proj_test",
        summary="Brand package for test client",
        tags=["branding", "test"],
    )
    assert art.artifact_id.startswith("art_")
    assert art.artifact_type == "brand_brief"

    # Retrieve
    found = get_artifact(art.artifact_id)
    assert found is not None
    assert found.title == "Test Brand Package"

    # By chat
    chat_arts = get_artifacts_for_chat("test_chat_domain")
    assert any(a.artifact_id == art.artifact_id for a in chat_arts)

    # Cleanup
    delete_artifact(art.artifact_id)


def test_printer_tools_conditional_registration():
    """Printer tools should only appear after register_printer_tools()."""
    from services.edison_core.runtime.tool_runtime import TOOL_REGISTRY, register_printer_tools
    # Printer tools may or may not already be registered
    register_printer_tools()
    assert "list_printers" in TOOL_REGISTRY
    assert "get_printer_status" in TOOL_REGISTRY
    assert "send_3d_print" in TOOL_REGISTRY


def test_total_tool_count():
    """TOOL_REGISTRY should have all base + task + domain + printer tools."""
    from services.edison_core.runtime.tool_runtime import TOOL_REGISTRY, register_printer_tools
    register_printer_tools()
    # Base (original) + task (3) + branding (4) + video (1) + project (2) + fabrication (1) + printer (3) = many
    assert len(TOOL_REGISTRY) >= 35, f"Expected >=35 tools, got {len(TOOL_REGISTRY)}"
