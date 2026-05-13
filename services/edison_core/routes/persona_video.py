"""Persona Video Studio API routes.

Direct core paths are mounted at ``/persona-video/*``. Through the Edison web
reverse proxy, browser code calls ``/api/persona-video/*``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import mimetypes
import yaml

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..persona_video import (
    PERSONA_PACK_TYPES,
    PersonaVideoService,
    PersonaVideoValidationError,
    RIGHTS_ACK_FIELDS,
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/persona-video", tags=["persona-video-studio"])

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "edison.yaml"
_service: Optional[PersonaVideoService] = None


def _load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return yaml.safe_load(CONFIG_PATH.read_text()) or {}
    except Exception:
        return {}


def get_service() -> PersonaVideoService:
    global _service
    if _service is None:
        _service = PersonaVideoService(REPO_ROOT, _load_config())
    return _service


class RightsAcknowledgementPayload(BaseModel):
    all_visible_performers_are_consenting_adults: bool = False
    source_footage_cleared_for_ai_transformation: bool = False
    persona_identity_is_synthetic_or_authorized: bool = False
    material_is_non_explicit_production_footage: bool = False
    material_does_not_depict_or_simulate_minors: bool = False
    no_unauthorized_real_person_impersonation: bool = False
    local_metadata_acknowledgment_understood: bool = False
    acknowledged_by: Optional[str] = None


class PersonaPackPayload(BaseModel):
    name: str
    type: str = Field(default="other", description="LoRA / reference_images / identity_model / adapter / other")
    paths: List[str] = Field(default_factory=list)
    notes: str = ""
    preferred_backend_compatibility: List[str] = Field(default_factory=list)
    thumbnail: Optional[str] = None
    persona_id: Optional[str] = None


class ProbeRequest(BaseModel):
    source_path: str
    create_preview: bool = True


class PersonaVideoJobRequest(BaseModel):
    project_title: str
    description: str = ""
    output_folder: Optional[str] = None
    source_path: str
    persona_id: str
    rights_acknowledgement: RightsAcknowledgementPayload
    target_selection: Dict[str, Any] = Field(default_factory=dict)
    settings: Dict[str, Any] = Field(default_factory=dict)
    autostart: bool = True


@router.get("/health")
async def health() -> Dict[str, Any]:
    service = get_service()
    settings = service.settings()
    return {
        "ok": True,
        "module": "Persona Video Studio",
        "enabled": settings.get("enabled"),
        "rights_ack_fields": list(RIGHTS_ACK_FIELDS),
        "video_extensions": sorted(SUPPORTED_VIDEO_EXTENSIONS),
        "audio_extensions": sorted(SUPPORTED_AUDIO_EXTENSIONS),
    }


@router.get("/settings")
async def settings() -> Dict[str, Any]:
    return {"ok": True, "settings": get_service().settings()}


@router.get("/gpus")
async def gpus() -> Dict[str, Any]:
    return {"ok": True, **get_service().list_gpus()}


@router.get("/backends")
async def backends() -> Dict[str, Any]:
    return {"ok": True, "backends": get_service().list_backends()}


@router.get("/personas")
async def list_personas() -> Dict[str, Any]:
    return {"ok": True, "persona_packs": get_service().registry.list_packs(), "pack_types": sorted(PERSONA_PACK_TYPES)}


@router.post("/personas")
async def register_persona(payload: PersonaPackPayload) -> Dict[str, Any]:
    try:
        pack = get_service().registry.register_pack(_model_dump(payload))
        return {"ok": True, "persona_pack": pack}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/personas/{persona_id}")
async def delete_persona(persona_id: str) -> Dict[str, Any]:
    deleted = get_service().registry.delete_pack(persona_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Persona pack not found")
    return {"ok": True, "deleted": True, "persona_id": persona_id}


@router.post("/upload-source")
async def upload_source(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        content = await file.read()
        return get_service().save_upload(file.filename or "source_video.mp4", content)
    except PersonaVideoValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Persona source upload failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/probe")
async def probe_media(payload: ProbeRequest) -> Dict[str, Any]:
    try:
        return {"ok": True, "probe": get_service().probe_media(payload.source_path, create_preview=payload.create_preview)}
    except PersonaVideoValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    return {"ok": True, "jobs": get_service().list_jobs(status=status, limit=limit)}


@router.post("/jobs")
async def create_job(payload: PersonaVideoJobRequest) -> Dict[str, Any]:
    try:
        job = get_service().create_job(_model_dump(payload), autostart=payload.autostart)
        return {"ok": True, "job": job}
    except (PersonaVideoValidationError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Persona Video Studio job creation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> Dict[str, Any]:
    job = get_service().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True, "job": job}


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> Dict[str, Any]:
    try:
        return get_service().cancel_job(job_id)
    except PersonaVideoValidationError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str) -> Dict[str, Any]:
    try:
        return get_service().pause_job(job_id)
    except PersonaVideoValidationError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str) -> Dict[str, Any]:
    try:
        return get_service().resume_job(job_id)
    except PersonaVideoValidationError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/jobs/{job_id}/segments/{segment_id}/retry")
async def retry_segment(job_id: str, segment_id: str) -> Dict[str, Any]:
    try:
        job = get_service().retry_segment(job_id, segment_id)
        return {"ok": True, "job": job}
    except PersonaVideoValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/jobs/{job_id}/logs")
async def job_logs(job_id: str) -> Dict[str, Any]:
    job = get_service().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True, "job_id": job_id, "logs": job.get("logs", []), "gpu_logs": (job.get("gpu") or {}).get("logs", [])}


@router.get("/jobs/{job_id}/report")
async def job_report(job_id: str) -> Dict[str, Any]:
    job = get_service().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True, "report": get_service().build_metadata_report(job)}


@router.get("/previews/{filename}")
async def serve_preview(filename: str) -> FileResponse:
    service = get_service()
    previews_dir = (service.output_root / "previews").resolve(strict=False)
    target = (previews_dir / Path(filename).name).resolve(strict=False)
    if not str(target).startswith(str(previews_dir)) or not target.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    media_type = mimetypes.guess_type(str(target))[0] or "image/jpeg"
    return FileResponse(str(target), media_type=media_type, filename=target.name)


@router.get("/outputs/{job_id}/{filename}")
async def serve_output(job_id: str, filename: str) -> FileResponse:
    job = get_service().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job_dir = Path(job.get("project", {}).get("job_dir", "")).resolve(strict=False)
    target = (job_dir / "final" / Path(filename).name).resolve(strict=False)
    if not str(target).startswith(str(job_dir.resolve(strict=False))) or not target.exists():
        raise HTTPException(status_code=404, detail="Output not found")
    media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    return FileResponse(str(target), media_type=media_type, filename=target.name)


def _model_dump(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
