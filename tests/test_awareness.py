"""
Tests for Edison awareness subsystems (Parts 1-10).

Run: python tests/test_awareness.py
"""

import os
import sys
import tempfile
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed = 0
failed = 0


def run_test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"✓ {name}: PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ {name}: FAILED — {e}")
        failed += 1


# ── Part 1: Conversation State ───────────────────────────────────────────

def test_conversation_state_crud():
    from services.state.conversation_state import ConversationStateManager

    mgr = ConversationStateManager()

    # Get creates new state
    state = mgr.get_state("sess_1")
    assert state.session_id == "sess_1"
    assert state.active_domain == "unknown"
    assert state.task_stage == "idle"
    assert state.turn_count == 0

    # Update state
    mgr.update_state("sess_1", {
        "current_project": "my_app",
        "active_domain": "code",
        "task_stage": "executing",
        "last_intent": "debug",
    })
    state = mgr.get_state("sess_1")
    assert state.current_project == "my_app"
    assert state.active_domain == "code"
    assert state.task_stage == "executing"
    assert state.last_intent == "debug"

    # Invalid domain rejected
    mgr.update_state("sess_1", {"active_domain": "NOT_VALID"})
    assert mgr.get_state("sess_1").active_domain == "code"  # unchanged

    # Reset
    mgr.reset_state("sess_1")
    assert mgr.get_state("sess_1").active_domain == "unknown"
    assert mgr.get_state("sess_1").turn_count == 0

    # Increment turn
    count = mgr.increment_turn("sess_1")
    assert count == 1
    count = mgr.increment_turn("sess_1")
    assert count == 2

    # Error tracking
    mgr.record_error("sess_1", "Something went wrong")
    assert mgr.get_state("sess_1").error_count == 1
    assert mgr.get_state("sess_1").last_error == "Something went wrong"


def test_domain_detection():
    from services.state.conversation_state import detect_domain

    assert detect_domain("fix the bug in my python function") == "code"
    assert detect_domain("generate an image of a sunset") == "image"
    assert detect_domain("make a video of a cat") == "video"
    assert detect_domain("compose a lo-fi beat") == "music"
    assert detect_domain("create a 3d mesh of a chair") == "mesh"
    assert detect_domain("how's the weather today") == "conversation"
    assert detect_domain("check gpu temperature") == "hardware"

    # Context string generation
    from services.state.conversation_state import ConversationState
    state = ConversationState(
        session_id="test",
        current_project="my_app",
        active_domain="code",
        task_stage="debugging",
        last_tool_used="web_search",
    )
    ctx_str = state.to_context_string()
    assert "my_app" in ctx_str
    assert "code" in ctx_str
    assert "debugging" in ctx_str


# ── Part 2+3: Intent + Continuation Detection ───────────────────────────

def test_goal_detection():
    from services.state.intent_detection import detect_goal, Goal

    goal, conf = detect_goal("fix the crash in my app")
    assert goal == Goal.DEBUG_CODE
    assert conf > 0.4

    goal, conf = detect_goal("change it to be brighter and more colorful", last_goal="generate_new_artifact")
    assert goal == Goal.MODIFY_PREVIOUS_OUTPUT
    assert conf > 0.4

    goal, conf = detect_goal("generate an image of a dragon", coral_intent="generate_image")
    assert goal == Goal.GENERATE_NEW_ARTIFACT
    assert conf >= 0.85

    goal, conf = detect_goal("what is machine learning")
    assert goal == Goal.RESEARCH_TOPIC
    assert conf > 0.4

    goal, conf = detect_goal("explain how neural networks work")
    assert goal == Goal.EXPLAIN_CONCEPT
    assert conf > 0.4


def test_continuation_detection():
    from services.state.intent_detection import detect_continuation, ContinuationType

    # First turn = always new
    ct, conf = detect_continuation("hello edison", turn_count=0)
    assert ct == ContinuationType.NEW_TASK

    # Modification signals
    ct, conf = detect_continuation(
        "make it bigger and darker",
        last_intent="generate_image",
        turn_count=3,
    )
    assert ct == ContinuationType.MODIFY_PREVIOUS

    # Continuation signals
    ct, conf = detect_continuation(
        "and also add a border",
        last_intent="generate_image",
        turn_count=3,
    )
    assert ct == ContinuationType.CONTINUE_PREVIOUS

    # New topic
    ct, conf = detect_continuation(
        "forget that, let's talk about something else",
        last_intent="generate_image",
        turn_count=5,
    )
    assert ct == ContinuationType.NEW_TASK


def test_unified_classifier():
    from services.state.intent_detection import classify_intent_with_goal, Goal, ContinuationType

    result = classify_intent_with_goal(
        "generate an epic dragon image",
        coral_intent="generate_image",
        turn_count=0,
    )
    assert result.goal == Goal.GENERATE_NEW_ARTIFACT
    assert result.continuation == ContinuationType.NEW_TASK
    assert result.confidence > 0.5
    assert result.intent == "generate_image"


# ── Part 4: System State ────────────────────────────────────────────────

def test_system_state():
    from services.state.system_state import (
        get_system_state, record_system_error, get_recent_errors,
    )

    # Should return a snapshot without crashing
    snapshot = get_system_state(force_refresh=True)
    assert snapshot.timestamp > 0
    assert isinstance(snapshot.disks, list)
    assert isinstance(snapshot.gpus, list)
    assert isinstance(snapshot.models_loaded, list)

    # Context string should not crash
    ctx = snapshot.to_context_string()
    assert isinstance(ctx, str)

    # Error tracking
    record_system_error("test_error", "something broke", source="test")
    errors = get_recent_errors(limit=5)
    assert len(errors) >= 1
    assert errors[-1]["type"] == "test_error"


# ── Part 6: Project State ───────────────────────────────────────────────

def test_project_state():
    from services.state.project_state import ProjectStateManager

    mgr = ProjectStateManager()

    mgr.detect_project_from_message("sess_1", "I'm working on project my_app and editing main.py and utils.py")
    ctx = mgr.get_context("sess_1")
    assert ctx.name == "my_app", f"Expected 'my_app', got '{ctx.name}'"
    assert any("main.py" in f for f in ctx.recent_files), f"main.py not in {ctx.recent_files}"

    mgr.add_file_reference("sess_1", "utils/helpers.py")
    assert "utils/helpers.py" in ctx.recent_files
    assert ctx.language == "python"

    # Context string
    ctx_str = ctx.to_context_string()
    assert "my_app" in ctx_str
    assert "python" in ctx_str


# ── Part 7: Suggestions ─────────────────────────────────────────────────

def test_suggestion_engine():
    from services.awareness.suggestions import SuggestionEngine

    engine = SuggestionEngine()

    # Repeated errors
    sug = engine.check_repeated_errors(5, "TypeError: foo is not a function")
    assert sug is not None
    assert "5 errors" in sug.message
    assert sug.category == "error_help"

    # No suggestion for low error counts
    sug = engine.check_repeated_errors(1)
    assert sug is None

    # Idle after failure
    sug = engine.check_idle_after_failure(120, "error")
    assert sug is not None
    assert sug.category == "idle_hint"

    # Low resources
    sug = engine.check_system_resource_warning(free_gpu_mb=1000)
    assert sug is not None
    assert "GPU" in sug.message

    # Pending suggestions
    pending = engine.get_pending()
    assert len(pending) >= 2

    # Dismiss
    first_id = pending[0]["id"]
    assert engine.dismiss(first_id) is True
    pending2 = engine.get_pending()
    assert all(s["id"] != first_id for s in pending2)


# ── Part 8: Planner ─────────────────────────────────────────────────────

def test_planner_routing():
    from services.planner.planner import Planner, PlanComplexity

    planner = Planner()

    # Simple chat → trivial plan
    plan = planner.create_plan(
        message="hello, how are you?",
        goal="casual_chat",
        mode="chat",
    )
    assert plan.complexity == PlanComplexity.TRIVIAL
    assert any(s.action == "llm_respond" for s in plan.steps)
    assert not any(s.action == "memory_write" for s in plan.steps)

    # Research with tools → multi-step
    plan = planner.create_plan(
        message="search for the latest AI papers",
        goal="research_topic",
        intent="web_search",
        mode="agent",
        tools_allowed=True,
    )
    assert plan.complexity in (PlanComplexity.MULTI_STEP, PlanComplexity.PARALLEL)
    assert any(s.action == "web_search" for s in plan.steps)
    assert any(s.action == "memory_write" for s in plan.steps)

    # Image generation → includes generate step
    plan = planner.create_plan(
        message="generate an image of a dragon",
        goal="generate_new_artifact",
        intent="generate_image",
        active_domain="image",
        tools_allowed=True,
    )
    assert any("generate" in s.action for s in plan.steps)

    # Plan dependencies work
    next_steps = plan.next_steps()
    assert len(next_steps) > 0
    assert all(len(s.depends_on) == 0 or s.step_id == min(s2.step_id for s2 in plan.steps) for s in next_steps)


# ── Part 9: Self-Evaluation ─────────────────────────────────────────────

def test_self_evaluation():
    from services.awareness.self_eval import SelfEvaluator
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = SelfEvaluator(db_path=Path(tmpdir) / "test_eval.db")

        # Record success
        eid1 = evaluator.record(
            session_id="sess_1", action="web_search", success=True,
            duration_s=1.5, intent="research", goal="research_topic",
        )
        assert eid1.startswith("eval_")

        # Record failure
        eid2 = evaluator.record(
            session_id="sess_1", action="code_exec", success=False,
            duration_s=0.3, intent="debug", goal="debug_code",
            error="SyntaxError: invalid syntax",
        )

        # Record correction
        evaluator.record_correction(eid1, "The search results were incomplete")

        # Query recent
        recent = evaluator.get_recent(limit=5, session_id="sess_1")
        assert len(recent) == 2

        # Stats
        stats = evaluator.get_stats()
        assert stats["total"] == 2
        assert stats["successes"] == 1
        assert stats["failures"] == 1
        assert 0 < stats["success_rate"] < 1

        # Correction rate
        cr = evaluator.get_correction_rate()
        assert cr["corrected"] == 1
        assert cr["correction_rate"] == 0.5


# ── Run all ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("Conversation state CRUD", test_conversation_state_crud),
        ("Domain detection", test_domain_detection),
        ("Goal detection", test_goal_detection),
        ("Continuation detection", test_continuation_detection),
        ("Unified intent classifier", test_unified_classifier),
        ("System state retrieval", test_system_state),
        ("Project state tracking", test_project_state),
        ("Suggestion engine", test_suggestion_engine),
        ("Planner routing decisions", test_planner_routing),
        ("Self-evaluation loop", test_self_evaluation),
    ]

    print(f"\nRunning {len(tests)} awareness tests...\n")
    for name, fn in tests:
        run_test(name, fn)

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    sys.exit(0 if failed == 0 else 1)
