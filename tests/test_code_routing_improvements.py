from services.edison_core.app import (
    _is_renderable_code_request,
    _response_looks_non_renderable,
    _retrieve_repo_code_context,
    route_mode,
)


def test_followup_preserves_code_mode_for_referential_edit():
    conversation_history = [
        {"role": "user", "content": "Build a React component for a task dashboard."},
        {"role": "assistant", "content": "```tsx\nexport function TaskDashboard() { return <div />; }\n```"},
    ]

    result = route_mode(
        "make it blue and add loading states",
        "auto",
        False,
        conversation_history=conversation_history,
    )

    assert result["mode"] == "code"
    assert result["model_target"] == "deep"
    assert any("follow-up detected" in reason.lower() for reason in result["reasons"])


def test_followup_does_not_override_explicit_realtime_switch():
    conversation_history = [
        {"role": "user", "content": "Write a FastAPI endpoint for project creation."},
        {"role": "assistant", "content": "Here is the endpoint implementation."},
    ]

    result = route_mode(
        "what's the weather in chicago right now?",
        "auto",
        False,
        conversation_history=conversation_history,
    )

    assert result["mode"] == "agent"
    assert result["tools_allowed"] is True


def test_renderable_code_quality_heuristics_are_narrow():
    assert _is_renderable_code_request("Build a self-contained HTML analytics dashboard") is True
    assert _response_looks_non_renderable("Here's a basic structure:\nimport App from './App'") is True
    assert _response_looks_non_renderable(
        "<!DOCTYPE html><html><head><style>body{margin:0}</style></head><body><script>console.log('ok')</script></body></html>"
    ) is False


def test_repo_code_context_prefers_explicit_file_reference():
    matches = _retrieve_repo_code_context(
        "How does route_mode work in services/edison_core/app.py?",
        conversation_history=[{"role": "assistant", "content": "Check the router implementation."}],
        max_snippets=2,
    )

    assert matches
    assert matches[0]["path"] == "services/edison_core/app.py"
    assert matches[0]["score"] == 100
    assert matches[0]["snippet"].strip()
