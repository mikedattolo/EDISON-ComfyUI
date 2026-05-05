"""
Conversation index + context-window indicator.

Phase 2 goal:

* "search across all my chats" without a heavy DB — light token-overlap
  index over the existing on-disk chat JSON files.
* Surface a context-usage indicator the front-end can show next to the
  composer (similar to ChatGPT/Claude's context bar).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_CHATS_DIR = REPO_ROOT / "chats"

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


@dataclass
class ConversationHit:
    chat_id: str
    title: str
    score: int
    snippet: str
    last_modified: float
    user: Optional[str] = None
    path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "title": self.title,
            "score": self.score,
            "snippet": self.snippet,
            "last_modified": self.last_modified,
            "user": self.user,
            "path": self.path,
        }


def _walk_chat_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*.json") if p.is_file() and p.name != "users.json"]


def _load_chat(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _extract_messages(chat: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(chat.get("messages"), list):
        return [m for m in chat["messages"] if isinstance(m, dict)]
    if isinstance(chat.get("turns"), list):
        return [m for m in chat["turns"] if isinstance(m, dict)]
    return []


def _message_text(msg: Dict[str, Any]) -> str:
    content = msg.get("content") or msg.get("text") or msg.get("message") or ""
    if isinstance(content, list):
        parts: List[str] = []
        for piece in content:
            if isinstance(piece, dict):
                parts.append(str(piece.get("text") or piece.get("content") or ""))
            else:
                parts.append(str(piece))
        return " ".join(p for p in parts if p)
    return str(content)


def search_conversations(
    query: str,
    *,
    chats_dir: Optional[Path] = None,
    limit: int = 20,
) -> List[ConversationHit]:
    """Search chat JSON files for token overlap with ``query``."""
    root = chats_dir or DEFAULT_CHATS_DIR
    q_tokens = set(_tokens(query))
    if not q_tokens:
        return []

    hits: List[ConversationHit] = []
    for path in _walk_chat_files(root):
        chat = _load_chat(path)
        if not chat:
            continue
        title = str(chat.get("title") or chat.get("name") or path.stem)
        messages = _extract_messages(chat)
        body_text_parts = [title]
        for msg in messages:
            body_text_parts.append(_message_text(msg))
        body = " ".join(body_text_parts)
        body_tokens = _tokens(body)
        if not body_tokens:
            continue
        body_set = set(body_tokens)
        overlap = q_tokens & body_set
        if not overlap:
            continue
        score = sum(body_tokens.count(t) for t in overlap)
        snippet = _make_snippet(body, q_tokens)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = time.time()
        hits.append(ConversationHit(
            chat_id=str(chat.get("id") or path.stem),
            title=title,
            score=score,
            snippet=snippet,
            last_modified=mtime,
            user=chat.get("user"),
            path=str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        ))

    hits.sort(key=lambda h: (h.score, h.last_modified), reverse=True)
    return hits[:limit]


def _make_snippet(text: str, q_tokens: set, *, window: int = 80) -> str:
    lower = text.lower()
    for tok in q_tokens:
        idx = lower.find(tok)
        if idx >= 0:
            start = max(0, idx - window // 2)
            end = min(len(text), idx + window)
            snippet = text[start:end].strip()
            if start > 0:
                snippet = "…" + snippet
            if end < len(text):
                snippet = snippet + "…"
            return snippet
    return text[: window * 2].strip()


# ── Context-usage indicator ──────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Cheap token estimate — ~4 chars/token, with a floor of word count."""
    if not text:
        return 0
    char_estimate = max(1, len(text) // 4)
    word_estimate = len(text.split())
    return max(char_estimate, word_estimate)


def context_usage(
    messages: List[Dict[str, Any]],
    *,
    context_window: int,
) -> Dict[str, Any]:
    """Estimate how full the model's context window is.

    Mirrors ChatGPT/Claude-style indicators:

    * ``used_tokens`` — best-effort estimate of tokens consumed.
    * ``available_tokens`` — remaining capacity.
    * ``percent_used`` — 0.0 – 1.0.
    * ``status`` — "ok" | "warn" | "critical" depending on percent.
    """
    used = 0
    for msg in messages or []:
        used += estimate_tokens(_message_text(msg))
    cw = max(1, int(context_window))
    pct = min(1.0, used / cw)
    if pct >= 0.9:
        status = "critical"
    elif pct >= 0.7:
        status = "warn"
    else:
        status = "ok"
    return {
        "used_tokens": used,
        "context_window": cw,
        "available_tokens": max(0, cw - used),
        "percent_used": round(pct, 4),
        "status": status,
    }
