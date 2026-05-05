"""
Phase 1 routes: GPU scheduler telemetry, citations bundling, and artifact
revision lookup. These are additive endpoints designed to be safe to mount
even when the underlying features are not configured (e.g. vLLM not
installed).

Mounted by ``services/edison_core/app.py`` next to the other route modules.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/phase1", tags=["phase1"])


# ── GPU scheduler telemetry ─────────────────────────────────────────

@router.get("/scheduler/telemetry")
async def scheduler_telemetry() -> Dict[str, Any]:
    """Return queue depth, in-flight counts, and recent job history per lane."""
    try:
        from ..gpu_scheduler import get_scheduler
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"scheduler unavailable: {exc}")
    return get_scheduler().telemetry()


@router.get("/scheduler/lanes")
async def scheduler_lanes() -> Dict[str, Any]:
    try:
        from ..gpu_scheduler import get_scheduler
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"scheduler unavailable: {exc}")
    sch = get_scheduler()
    return {"lanes": [sch.lane_info(lane) for lane in sch.lanes()]}


# ── Citations ──────────────────────────────────────────────────────

class CitationBundleRequest(BaseModel):
    hits: List[Dict[str, Any]]
    source: str = "rag"
    request_id: Optional[str] = None
    append_to: Optional[str] = None  # if provided, returns text+sources


@router.post("/citations/bundle")
async def citations_bundle(req: CitationBundleRequest) -> Dict[str, Any]:
    from ..citations import normalize_hits, bundle, attach_citations_to_text

    cits = normalize_hits(req.hits, source=req.source)
    payload = bundle(cits, request_id=req.request_id)
    if req.append_to is not None:
        payload["text_with_sources"] = attach_citations_to_text(req.append_to, cits)
    return payload


# ── Artifact revisions (in-memory registry) ─────────────────────────

# Light in-memory registry so the front-end can ask "give me revision X of
# artifact Y" while a generation is still streaming. Persistent storage is
# handled by the existing artifacts pipeline; this is just the live cache.
_ACTIVE_STREAMS: Dict[str, Any] = {}


def register_stream(stream) -> None:
    _ACTIVE_STREAMS[stream.artifact_id] = stream


def unregister_stream(artifact_id: str) -> None:
    _ACTIVE_STREAMS.pop(artifact_id, None)


@router.get("/artifacts/{artifact_id}/revisions")
async def list_revisions(artifact_id: str) -> Dict[str, Any]:
    stream = _ACTIVE_STREAMS.get(artifact_id)
    if stream is None:
        raise HTTPException(status_code=404, detail=f"unknown artifact_id: {artifact_id}")
    return {
        "artifact_id": artifact_id,
        "kind": stream.kind,
        "title": stream.title,
        "revisions": [rev.to_dict() for rev in stream.revisions],
    }


@router.get("/artifacts/{artifact_id}/diff")
async def diff_revisions(artifact_id: str, a: str, b: str) -> Dict[str, Any]:
    stream = _ACTIVE_STREAMS.get(artifact_id)
    if stream is None:
        raise HTTPException(status_code=404, detail=f"unknown artifact_id: {artifact_id}")
    return stream.diff(a, b)


# ── Health ──────────────────────────────────────────────────────────

@router.get("/health")
async def phase1_health() -> Dict[str, Any]:
    return {
        "ok": True,
        "modules": {
            "scheduler": True,
            "citations": True,
            "artifact_stream": True,
            "retry": True,
        },
    }
