"""
search_runtime.py — Multi-stage research pipeline.

Upgrades from raw snippet search to a structured research system
with query expansion, source ranking, and evidence extraction.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SearchSource:
    """A single search result source with metadata."""
    title: str = ""
    url: str = ""
    snippet: str = ""
    full_text: str = ""
    domain: str = ""
    relevance_score: float = 0.0
    freshness: str = ""  # e.g., "2 hours ago"
    fetched: bool = False

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet[:500],
            "domain": self.domain,
            "relevance_score": self.relevance_score,
        }


@dataclass
class ResearchResult:
    """Output of a research pipeline execution."""
    query: str = ""
    expanded_queries: List[str] = field(default_factory=list)
    sources: List[SearchSource] = field(default_factory=list)
    synthesis: str = ""
    citations: List[dict] = field(default_factory=list)
    mode: str = "fast"  # fast | deep | browser
    elapsed_sec: float = 0.0
    source_count: int = 0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "synthesis": self.synthesis,
            "sources": [s.to_dict() for s in self.sources[:10]],
            "citations": self.citations,
            "mode": self.mode,
            "source_count": self.source_count,
        }


def expand_query(query: str) -> List[str]:
    """
    Generate expanded queries for better search coverage.
    Lightweight heuristic, no LLM needed.
    """
    queries = [query]
    # Add a reformulation for questions
    if "?" in query:
        q_no_mark = query.rstrip("?").strip()
        queries.append(q_no_mark)
    # Add a "latest" variant for current-events style queries
    current_words = ["latest", "recent", "current", "today", "2026", "now"]
    if not any(w in query.lower() for w in current_words):
        if any(w in query.lower() for w in ["what", "who", "when", "where", "how"]):
            queries.append(f"{query} latest 2026")
    return queries[:3]


def deduplicate_sources(sources: List[SearchSource]) -> List[SearchSource]:
    """Remove duplicate sources by URL."""
    seen = set()
    deduped = []
    for s in sources:
        key = s.url.lower().rstrip("/")
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    return deduped


def rank_sources(sources: List[SearchSource], query: str) -> List[SearchSource]:
    """Simple relevance ranking based on query term overlap."""
    query_words = set(query.lower().split())
    for s in sources:
        text = f"{s.title} {s.snippet}".lower()
        overlap = sum(1 for w in query_words if w in text)
        s.relevance_score = overlap / max(len(query_words), 1)
    sources.sort(key=lambda s: s.relevance_score, reverse=True)
    return sources


def extract_citations(sources: List[SearchSource], limit: int = 5) -> List[dict]:
    """Extract citation metadata from top sources."""
    citations = []
    for i, s in enumerate(sources[:limit]):
        citations.append({
            "index": i + 1,
            "title": s.title,
            "url": s.url,
            "domain": s.domain,
        })
    return citations


async def run_fast_search(
    query: str,
    search_fn: Callable[[str, int], Any],
    max_results: int = 5,
) -> ResearchResult:
    """
    Fast search mode — snippets only, no page fetching.
    Suitable for quick factual queries.
    """
    t0 = time.time()
    result = ResearchResult(query=query, mode="fast")

    try:
        raw_results = await _call_search(search_fn, query, max_results)
        for r in raw_results:
            source = SearchSource(
                title=r.get("title", ""),
                url=r.get("href", r.get("url", "")),
                snippet=r.get("body", r.get("snippet", "")),
                domain=_extract_domain(r.get("href", r.get("url", ""))),
            )
            result.sources.append(source)
    except Exception as e:
        logger.warning(f"Search error: {e}")

    result.sources = deduplicate_sources(result.sources)
    result.sources = rank_sources(result.sources, query)
    result.citations = extract_citations(result.sources)
    result.source_count = len(result.sources)
    result.elapsed_sec = round(time.time() - t0, 2)
    return result


async def run_deep_research(
    query: str,
    search_fn: Callable[[str, int], Any],
    fetch_fn: Optional[Callable[[str], str]] = None,
    max_results: int = 8,
) -> ResearchResult:
    """
    Deep research mode — search + page fetch + extract + compare.
    """
    t0 = time.time()
    result = ResearchResult(query=query, mode="deep")
    result.expanded_queries = expand_query(query)

    # Search with expanded queries
    all_sources: List[SearchSource] = []
    for eq in result.expanded_queries:
        try:
            raw = await _call_search(search_fn, eq, max_results)
            for r in raw:
                source = SearchSource(
                    title=r.get("title", ""),
                    url=r.get("href", r.get("url", "")),
                    snippet=r.get("body", r.get("snippet", "")),
                    domain=_extract_domain(r.get("href", r.get("url", ""))),
                )
                all_sources.append(source)
        except Exception as e:
            logger.warning(f"Search error for '{eq}': {e}")

    result.sources = deduplicate_sources(all_sources)
    result.sources = rank_sources(result.sources, query)

    # Fetch top pages for deeper content
    if fetch_fn:
        for source in result.sources[:3]:
            try:
                full_text = await _call_fetch(fetch_fn, source.url)
                source.full_text = full_text[:8000]
                source.fetched = True
            except Exception as e:
                logger.warning(f"Fetch error for {source.url}: {e}")

    result.citations = extract_citations(result.sources)
    result.source_count = len(result.sources)
    result.elapsed_sec = round(time.time() - t0, 2)
    return result


def synthesize_sources(sources: List[SearchSource], query: str) -> str:
    """
    Produce a synthesis text from search sources.
    This is a simple concatenation — the LLM will do the real synthesis.
    """
    parts = []
    for i, s in enumerate(sources[:5]):
        text = s.full_text if s.fetched and s.full_text else s.snippet
        parts.append(f"[{i+1}] {s.title} ({s.domain}): {text[:1000]}")
    return "\n\n".join(parts)


def _extract_domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc.replace("www.", "")
    except Exception:
        return ""


async def _call_search(search_fn: Callable, query: str, max_results: int) -> list:
    """Call search function, handling both sync and async."""
    import asyncio
    import inspect
    if inspect.iscoroutinefunction(search_fn):
        return await search_fn(query, max_results)
    return await asyncio.to_thread(search_fn, query, max_results)


async def _call_fetch(fetch_fn: Callable, url: str) -> str:
    """Call fetch function, handling both sync and async."""
    import asyncio
    import inspect
    if inspect.iscoroutinefunction(fetch_fn):
        return await fetch_fn(url)
    return await asyncio.to_thread(fetch_fn, url)
