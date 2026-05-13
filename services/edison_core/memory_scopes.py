"""Scoped memory scaffolding for global, project, conversation, and session RAG."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class MemoryScope:
    scope_id: str
    kind: str  # global | project | session


@dataclass
class ScopedMemoryHit:
    content: str
    score: float = 0.0
    source: str = ""
    scope_id: str = "global"
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryScopeManager:
    def __init__(self, global_scope: str = "global"):
        self.global_scope = global_scope

    def get_scope(
        self,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        chat_id: Optional[str] = None
    ) -> MemoryScope:
        if project_id:
            return MemoryScope(scope_id=f"project:{project_id}", kind="project")
        if chat_id:
            return MemoryScope(scope_id=f"chat:{chat_id}", kind="chat")
        if session_id:
            return MemoryScope(scope_id=f"session:{session_id}", kind="session")
        return MemoryScope(scope_id=self.global_scope, kind="global")


def _norm_text(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def dedupe_memory_hits(hits: Iterable[Dict[str, Any] | ScopedMemoryHit]) -> List[Dict[str, Any]]:
    """Remove duplicate retrieval hits while preserving the highest confidence/score."""
    best: Dict[str, Dict[str, Any]] = {}
    for hit in hits:
        item = hit.to_dict() if isinstance(hit, ScopedMemoryHit) else dict(hit)
        key = _norm_text(item.get("content", "")) or _norm_text(item.get("source", ""))
        if not key:
            continue
        current = best.get(key)
        candidate_rank = (float(item.get("confidence") or 0), float(item.get("score") or 0))
        current_rank = (
            float((current or {}).get("confidence") or 0),
            float((current or {}).get("score") or 0),
        )
        if current is None or candidate_rank > current_rank:
            best[key] = item
    return list(best.values())


def rank_memory_hits(
    hits: Iterable[Dict[str, Any] | ScopedMemoryHit],
    *,
    preferred_scope_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Rank hits with transparent score, confidence, and scope preference metadata."""
    preferred = set(preferred_scope_ids or [])
    ranked = []
    for item in dedupe_memory_hits(hits):
        base = float(item.get("score") or 0)
        confidence = float(item.get("confidence") or 0)
        scope_bonus = 0.1 if item.get("scope_id") in preferred else 0.0
        item["rank_score"] = round(base + (confidence * 0.25) + scope_bonus, 4)
        item["rank_explanation"] = {
            "score": base,
            "confidence": confidence,
            "scope_bonus": scope_bonus,
        }
        ranked.append(item)
    return sorted(ranked, key=lambda hit: hit.get("rank_score", 0), reverse=True)
