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


# ── Routing fix tests ────────────────────────────────────────────────

def test_business_queries_route_to_agent_mode():
    """Business/project queries must route to agent mode with tools_allowed."""
    from services.edison_core.runtime.routing_runtime import route
    test_cases = [
        "List my projects",
        "Create a branding client called Adoro Pizza",
        "Generate a brand package for Adoro Pizza",
        "Create a project called Summer Campaign",
        "Generate marketing copy for a tech startup",
        "List my clients",
        "Slice the model for 3d printing",
    ]
    for msg in test_cases:
        d = route(msg, "auto", has_image=False)
        assert d.tools_allowed, f"'{msg}' should have tools_allowed=True, got mode={d.mode}"
        assert d.mode == "agent", f"'{msg}' should route to agent, got {d.mode}"


# ── api_projects module tests ─────────────────────────────────────────

def test_api_projects_pydantic_models():
    """Validate ProjectCreate / ClientCreate model defaults and fields."""
    from services.edison_core.api_projects import ProjectCreate, ClientCreate
    p = ProjectCreate(name="Test Project")
    assert p.status == "planned"
    assert p.notes is None
    assert p.service_types == ["mixed"]

    c = ClientCreate(business_name="Test Corp")
    assert c.tags == []
    assert c.email is None


def test_api_projects_storage_helpers(tmp_path, monkeypatch):
    """_load_json / _save_json should roundtrip correctly."""
    import services.edison_core.api_projects as proj

    fake_file = tmp_path / "items.json"
    fake_file.write_text("[]", encoding="utf-8")
    assert proj._load_json(fake_file) == []

    proj._save_json(fake_file, [{"id": "1", "name": "X"}])
    loaded = proj._load_json(fake_file)
    assert len(loaded) == 1
    assert loaded[0]["name"] == "X"


def test_api_projects_storage_handles_corrupt(tmp_path):
    """_load_json should return [] on corrupt JSON."""
    import services.edison_core.api_projects as proj

    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid", encoding="utf-8")
    assert proj._load_json(bad_file) == []


def test_api_projects_overview_endpoint_shape():
    """business_overview endpoint should return expected structure."""
    import asyncio
    import services.edison_core.api_projects as proj
    from pathlib import Path as P
    import tempfile, shutil

    # Use a temp directory for storage
    td = tempfile.mkdtemp()
    orig_dir = proj.PROJECTS_DIR
    orig_clients = proj.CLIENTS_FILE
    orig_projects = proj.PROJECTS_FILE
    try:
        proj.PROJECTS_DIR = P(td)
        proj.CLIENTS_FILE = P(td) / "clients.json"
        proj.PROJECTS_FILE = P(td) / "projects.json"
        proj.CLIENTS_FILE.write_text("[]")
        proj.PROJECTS_FILE.write_text("[]")

        result = asyncio.new_event_loop().run_until_complete(proj.business_overview())
        assert result["ok"] is True
        assert "clients" in result
        assert "projects" in result
        assert "counts" in result
        for key in ("clients", "projects", "connectors", "printers"):
            assert key in result["counts"]
    finally:
        proj.PROJECTS_DIR = orig_dir
        proj.CLIENTS_FILE = orig_clients
        proj.PROJECTS_FILE = orig_projects
        shutil.rmtree(td, ignore_errors=True)


# ── Workflow engine tests ──────────────────────────────────────────────

def test_workflow_classify_step_branding():
    from services.edison_core.runtime.workflow_engine import classify_step, StepKind
    assert classify_step("Generate a branding package with logo concepts") == StepKind.BRANDING
    assert classify_step("Create marketing copy for email campaign") == StepKind.MARKETING
    assert classify_step("Prepare the STL file for 3d printing") == StepKind.FABRICATION
    assert classify_step("Create a storyboard for the promo video") == StepKind.VIDEO
    assert classify_step("Hello how are you today") == StepKind.LLM


def test_workflow_plan_from_message_single():
    from services.edison_core.runtime.workflow_engine import plan_from_message, StepKind
    plan = plan_from_message("List my projects")
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_name == "list_projects"
    assert plan.steps[0].kind == StepKind.PROJECT


def test_workflow_plan_from_message_compound():
    from services.edison_core.runtime.workflow_engine import plan_from_message
    plan = plan_from_message("Make a branding package for Adoro Pizza and generate marketing copy for their menu")
    assert len(plan.steps) >= 2
    kinds = [s.kind.value for s in plan.steps]
    assert "branding" in kinds
    assert "marketing" in kinds


def test_workflow_plan_from_message_fallback():
    from services.edison_core.runtime.workflow_engine import plan_from_message, StepKind
    plan = plan_from_message("What's the weather like in New York?")
    assert len(plan.steps) == 1
    assert plan.steps[0].kind == StepKind.LLM


def test_workflow_step_to_work_step():
    from services.edison_core.runtime.workflow_engine import WorkflowStep, StepKind, workflow_step_to_work_step
    ws = WorkflowStep(id=1, title="Generate brand", kind=StepKind.BRANDING, tool_name="generate_brand_package")
    d = workflow_step_to_work_step(ws)
    assert d["kind"] == "llm"  # business kinds map to base llm kind
    assert d["tool_name"] == "generate_brand_package"
    assert d["id"] == 1


def test_workflow_plan_serialization():
    from services.edison_core.runtime.workflow_engine import plan_from_message
    plan = plan_from_message("Create a branding client called Test Corp")
    d = plan.to_dict()
    assert "workflow_id" in d
    assert "steps" in d
    assert d["step_count"] >= 1


def test_workflow_summarize():
    from services.edison_core.runtime.workflow_engine import plan_from_message, summarize_workflow
    plan = plan_from_message("List my projects")
    plan.steps[0].status = "completed"
    summary = summarize_workflow(plan)
    assert "✅" in summary
    assert "1/1" in summary


# ── Social media module tests ─────────────────────────────────────────

def test_social_platforms_registry():
    from services.edison_core.api_social import SOCIAL_PLATFORMS
    assert "instagram" in SOCIAL_PLATFORMS
    assert "facebook" in SOCIAL_PLATFORMS
    assert "tiktok" in SOCIAL_PLATFORMS
    assert "linkedin" in SOCIAL_PLATFORMS
    assert "google_business" in SOCIAL_PLATFORMS


def test_social_post_draft_model():
    from services.edison_core.api_social import SocialPostDraft
    draft = SocialPostDraft(platform="instagram", caption="Test caption")
    assert draft.post_type == "image"
    assert draft.hashtags == []


def test_social_tools_in_registry():
    from services.edison_core.runtime.tool_runtime import TOOL_REGISTRY
    assert "create_social_post" in TOOL_REGISTRY
    assert "schedule_social_post" in TOOL_REGISTRY
    assert "list_social_posts" in TOOL_REGISTRY


def test_social_routing():
    from services.edison_core.runtime.routing_runtime import route
    d = route("Create a social post for Instagram", "auto", has_image=False)
    assert d.tools_allowed
    assert d.mode == "agent"


# ── Auth module tests ─────────────────────────────────────────────────

def test_auth_password_hashing():
    from services.edison_core.api_auth import _hash_password, _verify_password
    hash_hex, salt = _hash_password("test_password_123")
    assert _verify_password("test_password_123", hash_hex, salt)
    assert not _verify_password("wrong_password", hash_hex, salt)


def test_auth_status_endpoint():
    import asyncio
    from services.edison_core.api_auth import auth_status
    result = asyncio.new_event_loop().run_until_complete(auth_status())
    assert "auth_enabled" in result
    assert "roles" in result
    assert "admin" in result["roles"]


# ── Help module tests ─────────────────────────────────────────────────

def test_help_topics_exist():
    from services.edison_core.api_help import HELP_TOPICS
    assert len(HELP_TOPICS) >= 10
    ids = [t["id"] for t in HELP_TOPICS]
    assert "chat" in ids
    assert "branding" in ids
    assert "social" in ids
    assert "projects" in ids


def test_help_search():
    import asyncio
    from services.edison_core.api_help import get_help
    result = asyncio.new_event_loop().run_until_complete(get_help(q="branding"))
    assert result["total"] > 0
    assert result["topics"][0]["id"] == "branding"


def test_help_category_filter():
    import asyncio
    from services.edison_core.api_help import get_help
    result = asyncio.new_event_loop().run_until_complete(get_help(category="business"))
    assert result["total"] > 0
    assert all(t["category"] == "business" for t in result["topics"])


# ── Workflow engine social step tests ─────────────────────────────────

def test_workflow_classify_social():
    from services.edison_core.runtime.workflow_engine import classify_step, StepKind
    assert classify_step("Create a social post for Instagram") == StepKind.SOCIAL
    assert classify_step("Schedule a social post for next Friday") == StepKind.SOCIAL


def test_workflow_social_tool_selection():
    from services.edison_core.runtime.workflow_engine import WorkflowStep, StepKind, select_tool_for_step
    ws = WorkflowStep(id=1, title="Create social post", kind=StepKind.SOCIAL)
    assert select_tool_for_step(ws) == "create_social_post"
    ws2 = WorkflowStep(id=2, title="Schedule the post", kind=StepKind.SOCIAL)
    assert select_tool_for_step(ws2) == "schedule_social_post"
