from services.edison_core.response_guard import (
    StreamingResponseGuard,
    assistant_stop_sequences,
    sanitize_assistant_response,
)
from services.edison_core.runtime.quality_runtime import clean_response


def test_sanitize_truncates_fabricated_followup_turns():
    text = "The job is complete.\n\nUser: actually do another thing\nAssistant: Sure."

    assert sanitize_assistant_response(text) == "The job is complete."


def test_sanitize_preserves_role_labels_inside_code_fences():
    text = "Use this:\n```text\nUser: sample input\nAssistant: sample output\n```\nDone."

    assert sanitize_assistant_response(text) == text


def test_sanitize_strips_leading_assistant_label_only():
    assert sanitize_assistant_response("Assistant: Done.\n\nUser: fake") == "Done."


def test_streaming_guard_holds_back_and_blocks_ghost_turn():
    guard = StreamingResponseGuard(holdback_chars=8)
    emitted = []
    for token in ["All set.", "\n\nUs", "er: fake prompt\nAssistant: fake answer"]:
        delta, stopped = guard.push(token)
        emitted.append(delta)
        if stopped:
            break
    emitted.append(guard.flush())

    assert "".join(emitted) == "All set."
    assert guard.stopped is True
    assert guard.stop_reason == "role_label:user"


def test_quality_clean_response_uses_response_guard():
    assert clean_response("Assistant: Ready.\nUser: fake") == "Ready."


def test_assistant_stop_sequences_include_common_chat_roles():
    stops = assistant_stop_sequences(["\n\n\n"])

    assert "\nUser:" in stops
    assert "\nSystem:" in stops
    assert "\n\n\n" in stops
