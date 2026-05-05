"""
Phase 3 routes: CAD QA gates and video timeline / shot list / export
presets. All endpoints are additive.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/phase3", tags=["phase3"])


# ── CAD QA ─────────────────────────────────────────────────────────

class CadQARequest(BaseModel):
    mesh_path: str
    min_wall_thickness_mm: float = 0.4
    build_volume_mm: Optional[List[float]] = None  # [x, y, z]


@router.post("/cad/qa")
async def cad_qa(req: CadQARequest) -> Dict[str, Any]:
    from ..cad_qa import run_qa
    bv = None
    if req.build_volume_mm and len(req.build_volume_mm) == 3:
        bv = (
            float(req.build_volume_mm[0]),
            float(req.build_volume_mm[1]),
            float(req.build_volume_mm[2]),
        )
    report = run_qa(
        req.mesh_path,
        min_wall_thickness_mm=req.min_wall_thickness_mm,
        build_volume_mm=bv,
    )
    return report.to_dict()


# ── Video presets / shot list ──────────────────────────────────────

@router.get("/video/presets")
async def video_presets() -> Dict[str, Any]:
    from ..video_timeline import list_presets
    return {"presets": list_presets()}


class ShotListRequest(BaseModel):
    topic: str
    preset: str = "instagram_reel"
    target_duration_s: Optional[float] = None
    beat_count: Optional[int] = None


@router.post("/video/shotlist")
async def video_shotlist(req: ShotListRequest) -> Dict[str, Any]:
    from ..video_timeline import generate_shot_list
    try:
        sl = generate_shot_list(
            req.topic,
            preset=req.preset,
            target_duration_s=req.target_duration_s,
            beat_count=req.beat_count,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return sl.to_dict()


# ── Video clip sequences ───────────────────────────────────────────

class CreateSequenceRequest(BaseModel):
    title: Optional[str] = None
    preset: Optional[str] = None


class ClipPayload(BaseModel):
    id: Optional[str] = None
    source: str
    in_s: float = 0.0
    out_s: Optional[float] = None
    transition_in: Optional[str] = None
    transition_out: Optional[str] = None
    label: Optional[str] = None


@router.post("/video/sequences")
async def create_sequence(req: CreateSequenceRequest) -> Dict[str, Any]:
    from ..video_timeline import ClipSequence
    seq = ClipSequence(title=req.title, preset=req.preset)
    seq.save()
    return seq.to_dict()


@router.get("/video/sequences")
async def list_sequences() -> Dict[str, Any]:
    from ..video_timeline import ClipSequence
    return {"sequences": ClipSequence.list_sequences()}


@router.get("/video/sequences/{sequence_id}")
async def get_sequence(sequence_id: str) -> Dict[str, Any]:
    from ..video_timeline import ClipSequence
    try:
        seq = ClipSequence.load(sequence_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return seq.to_dict()


@router.post("/video/sequences/{sequence_id}/clips")
async def add_clip(sequence_id: str, clip: ClipPayload) -> Dict[str, Any]:
    from ..video_timeline import Clip, ClipSequence
    import uuid
    try:
        seq = ClipSequence.load(sequence_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    new_clip = Clip(
        id=clip.id or f"clip_{uuid.uuid4().hex[:8]}",
        source=clip.source,
        in_s=clip.in_s,
        out_s=clip.out_s,
        transition_in=clip.transition_in,
        transition_out=clip.transition_out,
        label=clip.label,
    )
    seq.add_clip(new_clip)
    seq.save()
    return seq.to_dict()


@router.delete("/video/sequences/{sequence_id}/clips/{clip_id}")
async def remove_clip(sequence_id: str, clip_id: str) -> Dict[str, Any]:
    from ..video_timeline import ClipSequence
    try:
        seq = ClipSequence.load(sequence_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if not seq.remove_clip(clip_id):
        raise HTTPException(status_code=404, detail="clip not found")
    seq.save()
    return seq.to_dict()


@router.post("/video/sequences/{sequence_id}/export-plan")
async def export_plan(sequence_id: str, preset: Optional[str] = None) -> Dict[str, Any]:
    from ..video_timeline import ClipSequence
    try:
        seq = ClipSequence.load(sequence_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    try:
        return seq.export_plan(preset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
