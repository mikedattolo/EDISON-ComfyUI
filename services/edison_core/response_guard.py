"""Response guardrails for assistant text.

This module catches a common local-LLM failure mode where generation continues
past the answer into a fabricated conversation:

    User: ...
    Assistant: ...

The helpers are intentionally small and dependency-free so they can run on both
non-streaming and streaming paths.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional


ROLE_NAMES = (
    "user",
    "human",
    "assistant",
    "edison",
    "mike",
    "michael",
    "system",
    "developer",
    "tool",
    "observation",
)

DEFAULT_STOP_SEQUENCES = (
    "\nUser:",
    "\nHuman:",
    "\nAssistant:",
    "\nSystem:",
    "\nDeveloper:",
    "\nTool:",
    "\nObservation:",
    "\nMike:",
    "\nMichael:",
    "User:",
    "Human:",
    "<|user|>",
    "<|im_start|>user",
    "<|start_header_id|>user<|end_header_id|>",
)

_ROLE_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?"
    r"(user|human|assistant|edison|mike|michael|system|developer|tool|observation)"
    r"\s*[:\uff1a]\s*",
    re.IGNORECASE,
)
_CHATML_RE = re.compile(
    r"^\s*(?:<\|im_start\|>\s*)?(user|human|assistant|system|developer|tool)"
    r"(?:\s*<\|im_sep\|>|\s*$)",
    re.IGNORECASE,
)
_LLAMA_HEADER_RE = re.compile(
    r"^\s*<\|start_header_id\|>\s*(user|assistant|system|developer|tool)"
    r"\s*<\|end_header_id\|>",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GuardResult:
    text: str
    stopped: bool = False
    reason: Optional[str] = None


def assistant_stop_sequences(extra: Optional[Iterable[str]] = None) -> list[str]:
    """Return robust stop sequences for llama-cpp style completions."""
    seen = set()
    stops: list[str] = []
    for seq in list(DEFAULT_STOP_SEQUENCES) + list(extra or []):
        if seq and seq not in seen:
            stops.append(seq)
            seen.add(seq)
    return stops


def _is_fence_start(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("```") or stripped.startswith("~~~")


def _strip_leading_assistant_label(text: str) -> str:
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        match = _ROLE_LINE_RE.match(line)
        if match and match.group(1).lower() in {"assistant", "edison"}:
            lines[index] = line[match.end():]
            return "".join(lines).strip()
        return text.strip()
    return text.strip()


def guard_assistant_response(text: str) -> GuardResult:
    """Trim fabricated follow-up turns while preserving fenced code blocks."""
    if not text:
        return GuardResult("")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _strip_leading_assistant_label(normalized)
    lines = normalized.splitlines(keepends=True)
    kept: list[str] = []
    in_fence = False
    first_nonempty_seen = False

    for line in lines:
        if _is_fence_start(line):
            in_fence = not in_fence
            kept.append(line)
            first_nonempty_seen = first_nonempty_seen or bool(line.strip())
            continue

        if not in_fence:
            role_match = _ROLE_LINE_RE.match(line)
            chatml_match = _CHATML_RE.match(line) or _LLAMA_HEADER_RE.match(line)
            if role_match or chatml_match:
                role = (role_match or chatml_match).group(1).lower()
                if first_nonempty_seen or role not in {"assistant", "edison"}:
                    return GuardResult("".join(kept).rstrip(), True, f"role_label:{role}")
                line = line[(role_match or chatml_match).end():]

        kept.append(line)
        if line.strip():
            first_nonempty_seen = True

    return GuardResult("".join(kept).strip(), False, None)


def sanitize_assistant_response(text: str) -> str:
    """Public convenience wrapper used by non-streaming paths."""
    return guard_assistant_response(text).text


class StreamingResponseGuard:
    """Incrementally emit only text that is safe from trailing role labels.

    A small holdback buffer prevents leaking partial labels such as ``\nUs`` to
    the browser before the next token reveals ``\nUser:``.
    """

    def __init__(self, holdback_chars: int = 64):
        self.holdback_chars = max(0, int(holdback_chars))
        self.raw_text = ""
        self.emitted_chars = 0
        self.stopped = False
        self.stop_reason: Optional[str] = None
        self._current_text = ""

    @property
    def text(self) -> str:
        return self._current_text

    def push(self, token: str) -> tuple[str, bool]:
        if not token or self.stopped:
            return "", self.stopped

        self.raw_text += token
        result = guard_assistant_response(self.raw_text)
        self._current_text = result.text
        if result.stopped:
            self.stopped = True
            self.stop_reason = result.reason

        emit_to = len(self._current_text)
        if not self.stopped:
            emit_to = max(self.emitted_chars, emit_to - self.holdback_chars)

        delta = self._current_text[self.emitted_chars:emit_to]
        self.emitted_chars = emit_to
        return delta, self.stopped

    def flush(self) -> str:
        result = guard_assistant_response(self.raw_text)
        self._current_text = result.text
        if result.stopped:
            self.stopped = True
            self.stop_reason = result.reason
        delta = self._current_text[self.emitted_chars:]
        self.emitted_chars = len(self._current_text)
        return delta
