"""Regression tests for hallucination fixes:

1. OpenAI-compat endpoint preserves full conversation history
   (previously it dropped all assistant turns and only kept the last
   user message, causing the model to lose context).
2. build_full_prompt feeds last 10 turns (was 3).
3. Stop tokens include \\nAssistant: so the model can't sneak a fake
   continuation in the user's voice.
"""

from __future__ import annotations

import re
from pathlib import Path

APP_PATH = Path(__file__).parent / "services" / "edison_core" / "app.py"


def _read():
    return APP_PATH.read_text(encoding="utf-8")


def test_openai_compat_preserves_history():
    src = _read()
    # The fixed code must collect history_pairs, not just last user/system.
    assert "history_pairs" in src, "history_pairs collector missing"
    assert "history_pairs[-10:]" in src, "history truncation cap missing"
    # The old buggy line (`system_prompt = "You are a helpful assistant."`)
    # must be gone.
    assert 'system_prompt = "You are a helpful assistant."' not in src, \
        "generic stub system prompt is still present"
    # And the new EDISON system prompt builder must be invoked.
    assert "build_system_prompt(\"chat\")" in src


def test_build_full_prompt_uses_10_turn_window():
    src = _read()
    # Must contain the new conversation_history[-10:] window
    assert "conversation_history[-10:]" in src, \
        "build_full_prompt should keep last 10 turns, not 3"
    # The old 3-turn window must be gone in build_full_prompt
    # (other helpers may legitimately use [-3:] elsewhere; we only check
    # that the explicit `for msg in conversation_history[-3:]:` pattern
    # is no longer the inner loop).
    assert "for msg in conversation_history[-3:]:" not in src, \
        "old 3-turn window still present in build_full_prompt"


def test_stop_tokens_block_fake_assistant_turns():
    """Every chat-path llm() call must include \\nAssistant: in stop list."""
    src = _read()
    # Find every `stop=[...]` literal that contains "User:" or "Human:".
    # All of them must also have "\nAssistant:" — that's the only way to
    # stop the model from continuing into a fake follow-up turn.
    stop_blocks = re.findall(r"stop=\[(.*?)\]", src, flags=re.DOTALL)
    chat_stop_blocks = [b for b in stop_blocks if '"User:"' in b or '"Human:"' in b]
    assert chat_stop_blocks, "no chat stop-token blocks found — search regex broken?"
    for b in chat_stop_blocks:
        # Either the newline-prefixed form OR the bare form is acceptable;
        # both block a fake assistant continuation. What's NOT acceptable
        # is a stop list that has User:/Human: but NEITHER form of
        # Assistant:.
        has_assistant_guard = (
            r'"\nAssistant:"' in b
            or '"Assistant:"' in b
        )
        assert has_assistant_guard, (
            "Chat stop-token block missing Assistant: guard — "
            "model can hallucinate the next turn:\n" + b[:200]
        )


def test_no_legacy_bare_user_only_stop_remaining():
    """No call site should still use the old [\"User:\", \"Human:\"] pair only."""
    src = _read()
    # Exact legacy pattern: stop=["User:", "Human:"] (only those two)
    assert 'stop=["User:", "Human:"]' not in src, \
        "legacy bare User:/Human: only stop list still present"
    # The ["User:", "Human:", "\\n\\n\\n"] pattern (no \\nAssistant:) must
    # also be gone.
    assert 'stop=["User:", "Human:", "\\n\\n\\n"],' not in src, \
        "legacy 3-token stop list (no \\nAssistant:) still present"
