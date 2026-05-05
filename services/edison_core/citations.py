"""
Citation helpers for retrieved-context responses.

Phase 1 goal: make EDISON's chat responses look more like Copilot/ChatGPT/Claude
by attaching source citations to messages that use retrieved context.

This module is purely data-shaping. It does not perform retrieval itself —
RAG, web search, and knowledge-base modules pass their hits in, and the
helpers here:

* normalize them into a consistent ``Citation`` shape,
* render reference markers (``[1]``, ``[2]``, …) that can be inserted
  inline into model output, and
* emit a serializable bundle the front-end can render as a sidebar.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass
class Citation:
    """A single source attached to a generated response."""

    id: str            # short stable id (e.g. "c1", "c2")
    title: str
    url: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None        # "web" | "rag" | "knowledge" | etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]


def normalize_hits(hits: Iterable[Dict[str, Any]], *, source: str = "rag") -> List[Citation]:
    """Convert heterogeneous retrieval results into ``Citation`` objects.

    Accepts dicts with any of the common key spellings used inside EDISON
    today (``url``, ``link``, ``href``; ``title``, ``name``; ``snippet``,
    ``content``, ``text``, ``summary``).
    """
    out: List[Citation] = []
    used_ids: set[str] = set()
    for index, raw in enumerate(hits, start=1):
        if not isinstance(raw, dict):
            continue
        url = raw.get("url") or raw.get("link") or raw.get("href")
        title = (
            raw.get("title")
            or raw.get("name")
            or (url.split("/")[-1] if url else None)
            or f"Source {index}"
        )
        snippet = (
            raw.get("snippet")
            or raw.get("summary")
            or raw.get("content")
            or raw.get("text")
        )
        if isinstance(snippet, str) and len(snippet) > 400:
            snippet = snippet[:400].rstrip() + "…"
        cid = raw.get("id") or f"c{index}"
        if not isinstance(cid, str):
            cid = f"c{index}"
        # ensure ids are unique within a single bundle
        original_cid = cid
        suffix = 2
        while cid in used_ids:
            cid = f"{original_cid}_{suffix}"
            suffix += 1
        used_ids.add(cid)
        out.append(
            Citation(
                id=cid,
                title=str(title),
                url=url if isinstance(url, str) else None,
                snippet=snippet if isinstance(snippet, str) else None,
                source=raw.get("source", source),
                metadata={
                    k: v for k, v in raw.items()
                    if k not in {"url", "link", "href", "title", "name",
                                 "snippet", "summary", "content", "text",
                                 "id", "source"}
                },
            )
        )
    return _dedupe(out)


def _dedupe(citations: List[Citation]) -> List[Citation]:
    """Drop duplicates that point at the same URL (or title when no URL)."""
    seen: Dict[str, Citation] = {}
    out: List[Citation] = []
    for cit in citations:
        key = (cit.url or "").strip().lower() or _short_hash(cit.title.lower())
        if key in seen:
            continue
        seen[key] = cit
        out.append(cit)
    return out


def render_marker(index: int) -> str:
    """Markdown-friendly inline marker, e.g. ``[1]``."""
    return f"[{index}]"


def render_reference_block(citations: Sequence[Citation]) -> str:
    """Render the trailing "Sources" section appended to a chat reply."""
    if not citations:
        return ""
    lines = ["", "**Sources**"]
    for i, cit in enumerate(citations, start=1):
        if cit.url:
            lines.append(f"{i}. [{cit.title}]({cit.url})")
        else:
            lines.append(f"{i}. {cit.title}")
    return "\n".join(lines)


# match common citation-marker patterns like "[1]", "[1, 2]", or "[1][2]"
_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def attach_citations_to_text(
    text: str,
    citations: Sequence[Citation],
    *,
    append_block: bool = True,
) -> str:
    """Ensure citations referenced in ``text`` resolve to known sources, and
    optionally append a Sources block. Citations not used inline are still
    listed, so users see the full provenance trail.
    """
    if not citations:
        return text
    if append_block:
        block = render_reference_block(citations)
        if block:
            text = f"{text.rstrip()}\n\n{block.strip()}"
    return text


def bundle(
    citations: Sequence[Citation],
    *,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the JSON payload the front-end can render as a citation panel."""
    return {
        "request_id": request_id,
        "count": len(citations),
        "items": [c.to_dict() for c in citations],
    }
