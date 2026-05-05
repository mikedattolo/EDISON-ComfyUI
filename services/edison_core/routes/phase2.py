"""
Phase 2 routes: jobs center, persistent artifact revisions, command
palette, and conversation/context indicators. All endpoints are additive.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/phase2", tags=["phase2"])


# ── Jobs center ─────────────────────────────────────────────────────

@router.get("/jobs")
async def jobs_list(
    status: Optional[str] = None,
    job_type: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    from ..jobs_center import list_jobs
    return {"jobs": list_jobs(status=status, job_type=job_type, limit=limit)}


@router.get("/jobs/summary")
async def jobs_summary() -> Dict[str, Any]:
    from ..jobs_center import summary
    return summary()


@router.post("/jobs/{job_id}/cancel")
async def jobs_cancel(job_id: str) -> Dict[str, Any]:
    from ..jobs_center import cancel
    return cancel(job_id)


# ── Persistent artifact revisions ──────────────────────────────────

class AddRevisionRequest(BaseModel):
    artifact_id: str
    content: str
    kind: str = "document"
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@router.post("/artifacts/revisions")
async def add_revision(req: AddRevisionRequest) -> Dict[str, Any]:
    from ..artifact_revisions import ArtifactRevisionStore
    store = ArtifactRevisionStore.get_instance()
    try:
        return store.add_revision(
            req.artifact_id,
            req.content,
            kind=req.kind,
            title=req.title,
            metadata=req.metadata or {},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/artifacts/{artifact_id}/revisions")
async def list_revisions_phase2(artifact_id: str) -> Dict[str, Any]:
    from ..artifact_revisions import ArtifactRevisionStore
    store = ArtifactRevisionStore.get_instance()
    return {"artifact_id": artifact_id, "revisions": store.list_revisions(artifact_id)}


@router.get("/artifacts/{artifact_id}/revisions/{revision_id}")
async def get_revision(artifact_id: str, revision_id: str) -> Dict[str, Any]:
    from ..artifact_revisions import ArtifactRevisionStore
    store = ArtifactRevisionStore.get_instance()
    rev = store.get_revision(artifact_id, revision_id)
    if rev is None:
        raise HTTPException(status_code=404, detail="revision not found")
    return rev


@router.get("/artifacts/{artifact_id}/diff")
async def diff_revisions_phase2(artifact_id: str, a: str, b: str) -> Dict[str, Any]:
    from ..artifact_revisions import ArtifactRevisionStore
    return ArtifactRevisionStore.get_instance().diff(artifact_id, a, b)


@router.post("/artifacts/{artifact_id}/restore/{revision_id}")
async def restore_revision(artifact_id: str, revision_id: str) -> Dict[str, Any]:
    from ..artifact_revisions import ArtifactRevisionStore
    try:
        return ArtifactRevisionStore.get_instance().restore(artifact_id, revision_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ── Command palette ────────────────────────────────────────────────

@router.get("/palette")
async def palette_list(category: Optional[str] = None) -> Dict[str, Any]:
    from ..command_palette import get_palette
    p = get_palette()
    cmds = p.by_category(category) if category else p.all()
    return {"commands": [c.to_dict() for c in cmds]}


@router.get("/palette/search")
async def palette_search(q: str, limit: int = 8) -> Dict[str, Any]:
    from ..command_palette import get_palette
    matches = get_palette().search(q, limit=limit)
    return {"query": q, "matches": [c.to_dict() for c in matches]}


# ── Conversation search + context usage ───────────────────────────

@router.get("/conversations/search")
async def conversation_search(q: str, limit: int = 20) -> Dict[str, Any]:
    from ..conversation_index import search_conversations
    hits = search_conversations(q, limit=limit)
    return {"query": q, "hits": [h.to_dict() for h in hits]}


class ContextUsageRequest(BaseModel):
    messages: List[Dict[str, Any]]
    context_window: int = 8192


@router.post("/context/usage")
async def context_usage_endpoint(req: ContextUsageRequest) -> Dict[str, Any]:
    from ..conversation_index import context_usage
    return context_usage(req.messages, context_window=req.context_window)
