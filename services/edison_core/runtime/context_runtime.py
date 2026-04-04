"""
context_runtime.py — Layered context assembly for the unified chat pipeline.

Handles the structured assembly of context from multiple sources:
- Recent verbatim conversation turns
- Rolling conversation summary
- Task state summary 
- Workspace/project memory
- RAG retrieval
- Artifact references
- Tool result cache

Prevents runaway prompt growth via budget-based assembly.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Token budget defaults (conservative char-based estimates) ─────────
DEFAULT_MAX_CONTEXT_CHARS = 12_000  # ~3k tokens
SUMMARY_BUDGET = 2_000
RECENT_TURNS_BUDGET = 6_000
MEMORY_BUDGET = 2_000
RAG_BUDGET = 2_000


@dataclass
class ContextLayer:
    """A single named block of context with its priority and budget."""
    name: str
    content: str
    priority: int = 50  # higher = more important, included first
    char_budget: int = 2000
    source: str = ""  # e.g. "conversation", "memory", "rag", "task"

    @property
    def trimmed(self) -> str:
        if len(self.content) <= self.char_budget:
            return self.content
        return self.content[: self.char_budget - 3] + "..."


@dataclass
class AssembledContext:
    """The final assembled context ready for prompt construction."""
    system_prompt: str = ""
    context_blocks: List[ContextLayer] = field(default_factory=list)
    total_chars: int = 0
    layers_included: List[str] = field(default_factory=list)
    layers_dropped: List[str] = field(default_factory=list)

    @property
    def combined_context(self) -> str:
        """Produce a single context string from all included layers."""
        parts = []
        for layer in self.context_blocks:
            if layer.content.strip():
                parts.append(f"[{layer.name}]\n{layer.trimmed}")
        return "\n\n".join(parts)


@dataclass
class ConversationSummary:
    """Rolling summary of a conversation for context continuity."""
    chat_id: str
    summary_text: str = ""
    turn_count: int = 0
    last_updated: float = 0.0
    key_topics: List[str] = field(default_factory=list)
    active_task: Optional[str] = None


# ── Summary store (in-memory, keyed by chat_id) ─────────────────────
_summaries: Dict[str, ConversationSummary] = {}
_MAX_SUMMARIES = 500


def get_summary(chat_id: str) -> Optional[ConversationSummary]:
    return _summaries.get(chat_id)


def update_summary(
    chat_id: str,
    summary_text: str,
    turn_count: int = 0,
    key_topics: Optional[List[str]] = None,
    active_task: Optional[str] = None,
) -> ConversationSummary:
    """Create or update a conversation summary."""
    existing = _summaries.get(chat_id)
    if existing:
        existing.summary_text = summary_text
        existing.turn_count = turn_count or existing.turn_count
        existing.last_updated = time.time()
        if key_topics:
            existing.key_topics = key_topics
        if active_task is not None:
            existing.active_task = active_task
        return existing
    s = ConversationSummary(
        chat_id=chat_id,
        summary_text=summary_text,
        turn_count=turn_count,
        last_updated=time.time(),
        key_topics=key_topics or [],
        active_task=active_task,
    )
    _summaries[chat_id] = s
    # Prune oldest if too many
    if len(_summaries) > _MAX_SUMMARIES:
        oldest_key = min(_summaries, key=lambda k: _summaries[k].last_updated)
        _summaries.pop(oldest_key, None)
    return s


# ── Context assembler ────────────────────────────────────────────────
def assemble_context(
    *,
    system_prompt: str = "",
    conversation_history: Optional[List[dict]] = None,
    conversation_summary: Optional[str] = None,
    task_state: Optional[dict] = None,
    memory_facts: Optional[List[str]] = None,
    rag_results: Optional[List[dict]] = None,
    artifact_refs: Optional[List[dict]] = None,
    tool_cache: Optional[List[str]] = None,
    assistant_profile: Optional[dict] = None,
    max_total_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> AssembledContext:
    """
    Assemble a structured context from multiple layers, respecting a total
    character budget.  Layers are sorted by priority (highest first) and
    trimmed to fit.
    """
    ctx = AssembledContext(system_prompt=system_prompt)
    layers: List[ContextLayer] = []

    # ── Recent conversation turns (highest priority) ─────────────────
    if conversation_history:
        recent = conversation_history[-10:]  # keep last 10 turns
        turns_text = "\n".join(
            f"{m.get('role', 'user').capitalize()}: {str(m.get('content', ''))[:800]}"
            for m in recent
            if isinstance(m, dict) and m.get("content")
        )
        if turns_text.strip():
            layers.append(ContextLayer(
                name="Recent conversation",
                content=turns_text,
                priority=90,
                char_budget=RECENT_TURNS_BUDGET,
                source="conversation",
            ))

    # ── Conversation summary ─────────────────────────────────────────
    if conversation_summary:
        layers.append(ContextLayer(
            name="Conversation summary",
            content=conversation_summary,
            priority=80,
            char_budget=SUMMARY_BUDGET,
            source="summary",
        ))

    # ── Task state ───────────────────────────────────────────────────
    if task_state:
        task_text = (
            f"Objective: {task_state.get('objective', 'none')}\n"
            f"Completed: {', '.join(task_state.get('completed_steps', []))}\n"
            f"Pending: {', '.join(task_state.get('pending_steps', []))}\n"
        )
        layers.append(ContextLayer(
            name="Active task",
            content=task_text,
            priority=85,
            char_budget=1500,
            source="task",
        ))

    # ── Memory facts ─────────────────────────────────────────────────
    if memory_facts:
        mem_text = "\n".join(f"• {f}" for f in memory_facts[:20])
        layers.append(ContextLayer(
            name="Remembered facts",
            content=mem_text,
            priority=70,
            char_budget=MEMORY_BUDGET,
            source="memory",
        ))

    # ── RAG retrieval ────────────────────────────────────────────────
    if rag_results:
        rag_text = "\n".join(
            f"[{r.get('source', 'unknown')}] {r.get('text', '')[:500]}"
            for r in rag_results[:5]
        )
        layers.append(ContextLayer(
            name="Retrieved knowledge",
            content=rag_text,
            priority=65,
            char_budget=RAG_BUDGET,
            source="rag",
        ))

    # ── Artifact references ──────────────────────────────────────────
    if artifact_refs:
        art_text = "\n".join(
            f"[{a.get('type', 'artifact')}] {a.get('title', 'untitled')} ({a.get('artifact_id', '')})"
            for a in artifact_refs[:10]
        )
        layers.append(ContextLayer(
            name="Active artifacts",
            content=art_text,
            priority=60,
            char_budget=1000,
            source="artifact",
        ))

    # ── Tool result cache ────────────────────────────────────────────
    if tool_cache:
        cache_text = "\n".join(tool_cache[-5:])
        layers.append(ContextLayer(
            name="Recent tool results",
            content=cache_text,
            priority=55,
            char_budget=1500,
            source="tool_cache",
        ))

    # ── Assistant profile customization ──────────────────────────────
    if assistant_profile:
        profile_name = assistant_profile.get("name", "Custom Assistant")
        profile_prompt = assistant_profile.get("system_prompt", "")
        if profile_prompt:
            layers.append(ContextLayer(
                name=f"Assistant profile: {profile_name}",
                content=profile_prompt,
                priority=95,
                char_budget=1500,
                source="profile",
            ))

    # ── Sort by priority and trim to budget ──────────────────────────
    layers.sort(key=lambda l: l.priority, reverse=True)
    budget_remaining = max_total_chars
    for layer in layers:
        trimmed_content = layer.trimmed
        if len(trimmed_content) <= budget_remaining:
            ctx.context_blocks.append(layer)
            ctx.layers_included.append(layer.name)
            budget_remaining -= len(trimmed_content)
        else:
            # Try to include a smaller portion
            if budget_remaining > 200:
                layer.char_budget = budget_remaining
                ctx.context_blocks.append(layer)
                ctx.layers_included.append(f"{layer.name} (trimmed)")
                budget_remaining = 0
            else:
                ctx.layers_dropped.append(layer.name)

    ctx.total_chars = max_total_chars - budget_remaining
    return ctx
