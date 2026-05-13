"""Persona Video Studio service layer.

This module implements a local, consent-gated, source-coherent video
transformation pipeline. It is intentionally additive and does not require a
specific identity-transform model to be installed. The pipeline owns the
production plumbing — rights metadata, probing, segmentation, target-track
metadata, backend selection, GPU scheduling, QC, reassembly, audio handling,
and reports — while actual identity synthesis is delegated to pluggable
``PersonaTransformBackend`` adapters.
"""

from __future__ import annotations

import json
import logging
import math
import mimetypes
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .persona_video_backends import PersonaBackendRegistry, PersonaTransformBackend
from .persona_video_gpu import ExclusiveGPURenderManager, GPUStrategySelector, SegmentQueueScheduler
from .safe_io import atomic_write_json, read_json

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
PERSONA_PACK_TYPES = {"lora", "reference_images", "identity_model", "adapter", "other"}

RIGHTS_ACK_FIELDS = (
    "all_visible_performers_are_consenting_adults",
    "source_footage_cleared_for_ai_transformation",
    "persona_identity_is_synthetic_or_authorized",
    "material_is_non_explicit_production_footage",
    "material_does_not_depict_or_simulate_minors",
    "no_unauthorized_real_person_impersonation",
    "local_metadata_acknowledgment_understood",
)

JOB_STATUSES = {
    "queued",
    "validating",
    "running",
    "paused",
    "failed",
    "needs_review",
    "completed",
    "cancelled",
}

PIPELINE_STAGES = [
    "validating_source_and_rights_metadata",
    "inspecting_media_streams",
    "detecting_shots_scenes",
    "segmenting_footage",
    "tracking_target_performer",
    "preparing_persona_backend_model",
    "transforming_segments",
    "temporal_stabilization_consistency_cleanup",
    "quality_control_scoring",
    "reprocessing_failed_segments_if_needed",
    "stitching_final_video",
    "restoring_audio_remuxing",
    "final_export_packaging",
]

QUALITY_PRESETS = {
    "fast_preview": {"label": "Fast Preview", "crf": 23, "preset": "veryfast", "segment_length_s": 20.0},
    "balanced": {"label": "Balanced", "crf": 18, "preset": "medium", "segment_length_s": 45.0},
    "maximum_quality": {"label": "Maximum Quality", "crf": 14, "preset": "slow", "segment_length_s": 30.0},
}

TRANSFORMATION_SCOPES = {
    "face_identity_only": "Face identity only",
    "face_hair_head_region": "Face + hair/head-region identity where supported",
    "full_visible_persona_styling": "Full visible persona styling where supported",
    "metadata_validation_only": "Pipeline validation only; no identity transformation",
}


@dataclass
class PersonaPack:
    persona_id: str
    name: str
    pack_type: str
    paths: List[str] = field(default_factory=list)
    notes: str = ""
    preferred_backend_compatibility: List[str] = field(default_factory=list)
    thumbnail: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PersonaPackRegistry:
    """Lightweight local persona identity pack registry."""

    def __init__(self, repo_root: Path, registry_path: Optional[Path] = None) -> None:
        self.repo_root = repo_root.resolve()
        self.registry_path = registry_path or (self.repo_root / "config" / "integrations" / "persona_packs.json")
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            atomic_write_json(self.registry_path, {"persona_packs": []})
        self._lock = threading.RLock()

    def list_packs(self) -> List[Dict[str, Any]]:
        with self._lock:
            data = read_json(self.registry_path, {"persona_packs": []}) or {"persona_packs": []}
            return list(data.get("persona_packs", []))

    def get_pack(self, persona_id: str) -> Optional[Dict[str, Any]]:
        return next((pack for pack in self.list_packs() if pack.get("persona_id") == persona_id), None)

    def register_pack(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        name = str(payload.get("name") or "").strip()
        if not name:
            raise ValueError("Persona pack name is required")
        pack_type = str(payload.get("type") or payload.get("pack_type") or "other").strip().lower()
        if pack_type not in PERSONA_PACK_TYPES:
            raise ValueError(f"Unsupported persona pack type '{pack_type}'")
        now = _utc_now()
        persona_id = str(payload.get("persona_id") or f"persona_{uuid.uuid4().hex[:10]}")
        paths = [str(item).strip() for item in (payload.get("paths") or payload.get("local_file_paths") or []) if str(item).strip()]
        pack = PersonaPack(
            persona_id=persona_id,
            name=name,
            pack_type=pack_type,
            paths=paths,
            notes=str(payload.get("notes") or ""),
            preferred_backend_compatibility=[
                str(item).strip() for item in payload.get("preferred_backend_compatibility", []) if str(item).strip()
            ],
            thumbnail=payload.get("thumbnail") or payload.get("preview_image"),
            created_at=str(payload.get("created_at") or now),
            updated_at=now,
        ).to_dict()
        pack["path_status"] = self._path_status(paths)
        with self._lock:
            data = read_json(self.registry_path, {"persona_packs": []}) or {"persona_packs": []}
            packs = [item for item in data.get("persona_packs", []) if item.get("persona_id") != persona_id]
            packs.append(pack)
            data["persona_packs"] = sorted(packs, key=lambda item: item.get("name", "").lower())
            atomic_write_json(self.registry_path, data)
        return pack

    def delete_pack(self, persona_id: str) -> bool:
        with self._lock:
            data = read_json(self.registry_path, {"persona_packs": []}) or {"persona_packs": []}
            before = len(data.get("persona_packs", []))
            data["persona_packs"] = [item for item in data.get("persona_packs", []) if item.get("persona_id") != persona_id]
            if len(data["persona_packs"]) == before:
                return False
            atomic_write_json(self.registry_path, data)
            return True

    def _path_status(self, paths: Iterable[str]) -> List[Dict[str, Any]]:
        status = []
        for raw in paths:
            p = Path(raw)
            if not p.is_absolute():
                p = self.repo_root / p
            status.append({"path": raw, "exists": p.exists(), "resolved": str(p.resolve(strict=False))})
        return status


class PersonaVideoValidationError(ValueError):
    pass


class PersonaVideoService:
    """Production-oriented job service for Persona Video Studio."""

    def __init__(self, repo_root: Path, config: Optional[Dict[str, Any]] = None) -> None:
        self.repo_root = repo_root.resolve()
        self.config = _normalize_config(config or {})
        self.output_root = self._resolve_dir(self.config.get("default_output_directory", "outputs/persona_video"))
        self.temp_root = self._resolve_dir(self.config.get("temp_working_directory", "outputs/persona_video/tmp"))
        self.upload_root = self._resolve_dir(self.config.get("upload_directory", "uploads/persona_video"))
        self.jobs_root = self.output_root / "jobs"
        for folder in (self.output_root, self.temp_root, self.upload_root, self.jobs_root):
            folder.mkdir(parents=True, exist_ok=True)
        self.registry = PersonaPackRegistry(self.repo_root)
        self.backends = PersonaBackendRegistry(self.repo_root, self.config)
        self.gpu_manager = ExclusiveGPURenderManager()
        self._lock = threading.RLock()
        self._active_threads: Dict[str, threading.Thread] = {}

    # ── settings / discovery ───────────────────────────────────────────

    def settings(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.config.get("enabled", True)),
            "default_output_directory": str(self.output_root),
            "temp_working_directory": str(self.temp_root),
            "keep_intermediates_default": bool(self.config.get("keep_intermediates_default", False)),
            "default_quality_preset": self.config.get("default_quality_preset", "balanced"),
            "default_gpu_strategy": self.config.get("default_gpu_strategy", "auto"),
            "exclusive_gpu_render_mode_default": bool(self.config.get("exclusive_gpu_render_mode_default", False)),
            "auto_restore_suspended_services": bool(self.config.get("auto_restore_suspended_services", True)),
            "vram_min_free_mb_by_gpu": self.config.get("vram_min_free_mb_by_gpu", {}),
            "maximum_concurrent_segment_workers": int(self.config.get("maximum_concurrent_segment_workers", 1)),
            "automatic_failed_segment_rerender_threshold": float(self.config.get("automatic_failed_segment_rerender_threshold", 0.55)),
            "qc_thresholds": self.config.get("qc_thresholds", {"minimum_segment_score": 0.6}),
            "optional_backend_selection": self.config.get("backend", "metadata_only_passthrough"),
            "supported_video_extensions": sorted(SUPPORTED_VIDEO_EXTENSIONS),
            "supported_audio_extensions": sorted(SUPPORTED_AUDIO_EXTENSIONS),
            "rights_ack_fields": list(RIGHTS_ACK_FIELDS),
            "pipeline_stages": list(PIPELINE_STAGES),
            "quality_presets": QUALITY_PRESETS,
            "transformation_scopes": TRANSFORMATION_SCOPES,
        }

    def list_backends(self) -> List[Dict[str, Any]]:
        return self.backends.list()

    def list_gpus(self) -> Dict[str, Any]:
        gpus = [gpu.to_dict() for gpu in self.gpu_manager.detect_gpus()]
        return {
            "gpus": gpus,
            "gpu_count": len(gpus),
            "note": "VRAM is tracked per GPU; Edison does not treat heterogeneous GPUs as one pooled card.",
        }

    # ── media intake ───────────────────────────────────────────────────

    def save_upload(self, filename: str, content: bytes) -> Dict[str, Any]:
        safe_name = _safe_filename(filename or "source_video.mp4")
        ext = Path(safe_name).suffix.lower()
        if ext not in SUPPORTED_VIDEO_EXTENSIONS:
            raise PersonaVideoValidationError(f"Unsupported source video extension '{ext}'")
        target = self.upload_root / f"{uuid.uuid4().hex[:8]}_{safe_name}"
        target.write_bytes(content)
        probe = self.probe_media(str(target), create_preview=True)
        return {"ok": True, "source_path": str(target), "probe": probe}

    def probe_media(self, source_path: str, create_preview: bool = False) -> Dict[str, Any]:
        source = self._resolve_media_path(source_path)
        if not source.exists() or not source.is_file():
            raise PersonaVideoValidationError("Source video file does not exist")
        if source.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            raise PersonaVideoValidationError(f"Unsupported media type: {source.suffix}")
        probe = probe_video(source)
        probe["source_path"] = str(source)
        probe["supported"] = True
        if create_preview:
            preview_dir = self.output_root / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{source.stem}_{uuid.uuid4().hex[:8]}"
            thumbnail = preview_dir / f"{stem}.jpg"
            contact_sheet = preview_dir / f"{stem}_contact.jpg"
            probe["thumbnail_path"] = str(create_thumbnail(source, thumbnail, probe.get("duration_s")))
            probe["contact_sheet_path"] = str(create_contact_sheet(source, contact_sheet, probe.get("duration_s")))
        return probe

    # ── job lifecycle ──────────────────────────────────────────────────

    def create_job(self, payload: Dict[str, Any], autostart: bool = True) -> Dict[str, Any]:
        if not bool(self.config.get("enabled", True)):
            raise PersonaVideoValidationError("Persona Video Studio is disabled in configuration")
        rights_ack = payload.get("rights_acknowledgement") or payload.get("rights_ack") or {}
        validate_rights_acknowledgement(rights_ack)
        source = self._resolve_media_path(str(payload.get("source_path") or ""))
        if not source.exists() or source.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            raise PersonaVideoValidationError("A readable supported source video is required")
        persona_id = str(payload.get("persona_id") or "").strip()
        persona_pack = self.registry.get_pack(persona_id) if persona_id else None
        if not persona_pack:
            raise PersonaVideoValidationError("A registered persona pack is required before starting a job")
        settings = self._normalize_job_settings(payload.get("settings") or {})
        backend = self.backends.get(settings["backend"])
        capabilities = backend.get_capabilities().to_dict()
        scope = settings.get("transformation_scope")
        if scope not in capabilities.get("supported_transformation_scopes", []):
            raise PersonaVideoValidationError(
                f"Backend '{settings['backend']}' does not support transformation scope '{scope}'."
            )
        output_folder = self._resolve_output_folder(payload.get("output_folder"), payload.get("project_title"))
        job_id = f"pvs_{uuid.uuid4().hex[:12]}"
        job_dir = output_folder / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        now = _utc_now()
        job = {
            "job_id": job_id,
            "job_type": "persona_video",
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "project": {
                "title": str(payload.get("project_title") or "Persona Video Studio Job").strip(),
                "description": str(payload.get("description") or payload.get("notes") or ""),
                "output_folder": str(output_folder),
                "job_dir": str(job_dir),
            },
            "rights_acknowledgement": enrich_rights_acknowledgement(rights_ack),
            "source": {"path": str(source), "probe": None, "thumbnail_path": None, "contact_sheet_path": None},
            "persona": persona_pack,
            "target_selection": payload.get("target_selection") or {"mode": "auto_detect_primary_subject"},
            "settings": settings,
            "backend_capabilities": capabilities,
            "pipeline": {
                "stages": [{"id": stage, "status": "queued", "progress": 0.0} for stage in PIPELINE_STAGES],
                "current_stage": None,
                "overall_progress": 0.0,
            },
            "segments": [],
            "segment_manifest": None,
            "qc_summary": None,
            "gpu": {"strategy_plan": None, "snapshots": {}, "logs": []},
            "exclusive_render": {},
            "outputs": {},
            "logs": [],
            "warnings": [],
            "errors": [],
            "cancel_requested": False,
            "pause_requested": False,
            "rerender_history": [],
        }
        self._save_job(job)
        self._append_log(job_id, "Job queued.")
        if autostart:
            self._start_thread(job_id, self._run_job)
        return self.get_job(job_id) or job

    def list_jobs(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        jobs = []
        for path in sorted(self.jobs_root.glob("*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                job = read_json(path, {}) or {}
                if status and job.get("status") != status:
                    continue
                jobs.append(job)
            except Exception:
                continue
            if len(jobs) >= limit:
                break
        return jobs

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        path = self._job_path(job_id)
        if not path.exists():
            return None
        return read_json(path, None)

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        job = self._require_job(job_id)
        if job.get("status") in {"completed", "failed", "cancelled"}:
            return {"ok": False, "job_id": job_id, "status": job.get("status"), "reason": "job already terminal"}
        job["cancel_requested"] = True
        job["status"] = "cancelled" if job.get("status") == "queued" else job.get("status", "running")
        job["updated_at"] = _utc_now()
        self._save_job(job)
        self._append_log(job_id, "Cancellation requested.")
        return {"ok": True, "job_id": job_id, "status": job.get("status"), "cancel_requested": True}

    def pause_job(self, job_id: str) -> Dict[str, Any]:
        job = self._require_job(job_id)
        if job.get("status") not in {"queued", "running", "validating"}:
            return {"ok": False, "job_id": job_id, "reason": f"Cannot pause status {job.get('status')}"}
        job["pause_requested"] = True
        job["updated_at"] = _utc_now()
        self._save_job(job)
        self._append_log(job_id, "Pause requested; worker will pause at the next safe boundary.")
        return {"ok": True, "job_id": job_id, "pause_requested": True}

    def resume_job(self, job_id: str) -> Dict[str, Any]:
        job = self._require_job(job_id)
        job["pause_requested"] = False
        if job.get("status") == "paused":
            job["status"] = "running"
        job["updated_at"] = _utc_now()
        self._save_job(job)
        self._append_log(job_id, "Resume requested.")
        return {"ok": True, "job_id": job_id, "status": job.get("status")}

    def retry_segment(self, job_id: str, segment_id: str) -> Dict[str, Any]:
        job = self._require_job(job_id)
        job = mark_segment_for_retry(job, segment_id, reason="manual_retry_requested")
        self._save_job(job)
        self._append_log(job_id, f"Retry queued for segment {segment_id}.")
        self._start_thread(job_id, self._retry_segment_worker, segment_id)
        return self.get_job(job_id) or job

    # ── pure helpers exposed for tests/documented integration ──────────

    def build_metadata_report(self, job: Dict[str, Any]) -> Dict[str, Any]:
        return build_metadata_report(job)

    # ── worker internals ───────────────────────────────────────────────

    def _run_job(self, job_id: str) -> None:
        exclusive_state: Dict[str, Any] = {}
        try:
            job = self._require_job(job_id)
            self._set_status(job_id, "validating")
            self._stage(job_id, "validating_source_and_rights_metadata", "running", 0.02)
            validate_rights_acknowledgement(job.get("rights_acknowledgement") or {})
            self._checkpoint(job_id)
            self._stage(job_id, "validating_source_and_rights_metadata", "completed", 0.08)

            settings = job.get("settings") or {}
            if settings.get("exclusive_gpu_render_mode"):
                self._append_log(job_id, "Exclusive GPU Render Mode enabled: suspending Edison-controlled local AI services where possible.")
                exclusive_state = self.gpu_manager.enter_exclusive_mode(
                    thresholds_min_free_mb={str(k): float(v) for k, v in (self.config.get("vram_min_free_mb_by_gpu") or {}).items()},
                    wait_timeout_s=float(self.config.get("exclusive_wait_timeout_s", 60)),
                    poll_interval_s=float(self.config.get("exclusive_poll_interval_s", 2)),
                )
                job = self._require_job(job_id)
                job["exclusive_render"] = exclusive_state
                job.setdefault("gpu", {}).setdefault("snapshots", {})["exclusive_enter"] = exclusive_state
                self._save_job(job)

            self._set_status(job_id, "running")
            source = Path(job["source"]["path"])
            job_dir = Path(job["project"]["job_dir"])
            work_dir = job_dir / "work"
            segment_dir = work_dir / "segments"
            transformed_dir = work_dir / "transformed"
            stabilized_dir = work_dir / "stabilized"
            for folder in (work_dir, segment_dir, transformed_dir, stabilized_dir):
                folder.mkdir(parents=True, exist_ok=True)

            self._stage(job_id, "inspecting_media_streams", "running", 0.10)
            probe = self.probe_media(str(source), create_preview=True)
            job = self._require_job(job_id)
            job["source"]["probe"] = probe
            job["source"]["thumbnail_path"] = probe.get("thumbnail_path")
            job["source"]["contact_sheet_path"] = probe.get("contact_sheet_path")
            self._save_job(job)
            self._stage(job_id, "inspecting_media_streams", "completed", 0.16)

            self._stage(job_id, "detecting_shots_scenes", "running", 0.18)
            segments = build_segments(
                float(probe.get("duration_s") or 0),
                settings.get("segment_size_preference", "auto"),
                settings.get("quality_preset", "balanced"),
            )
            job = self._require_job(job_id)
            job["segments"] = segments
            self._save_job(job)
            self._stage(job_id, "detecting_shots_scenes", "completed", 0.24)

            self._stage(job_id, "segmenting_footage", "running", 0.26)
            split_segments = self._materialize_segments(job_id, source, segments, segment_dir)
            job = self._require_job(job_id)
            job["segments"] = split_segments
            self._save_job(job)
            self._stage(job_id, "segmenting_footage", "completed", 0.34)

            self._stage(job_id, "tracking_target_performer", "running", 0.36)
            tracked = self._track_target(job_id, split_segments)
            job = self._require_job(job_id)
            job["segments"] = tracked
            self._save_job(job)
            self._stage(job_id, "tracking_target_performer", "completed", 0.42)

            self._stage(job_id, "preparing_persona_backend_model", "running", 0.44)
            backend = self.backends.get(settings.get("backend", "metadata_only_passthrough"))
            prepare_result = backend.prepare(job, work_dir)
            job = self._require_job(job_id)
            job["backend_prepare"] = prepare_result
            self._save_job(job)
            self._stage(job_id, "preparing_persona_backend_model", "completed", 0.50)

            gpus = [gpu.to_dict() for gpu in self.gpu_manager.detect_gpus()]
            plan = GPUStrategySelector.select(
                gpus,
                job.get("backend_capabilities") or backend.get_capabilities().to_dict(),
                settings.get("gpu_strategy", "auto"),
                int(self.config.get("maximum_concurrent_segment_workers", 1)),
            )
            scheduler = SegmentQueueScheduler(plan, gpus)
            assigned = scheduler.assign_segments(tracked)
            log_lines = scheduler.stage_assignment_log(assigned)
            job = self._require_job(job_id)
            job["gpu"]["strategy_plan"] = plan.to_dict()
            job["gpu"].setdefault("logs", []).extend(log_lines)
            job["segments"] = assigned
            self._save_job(job)
            for line in log_lines:
                self._append_log(job_id, line)

            self._stage(job_id, "transforming_segments", "running", 0.52)
            transformed = self._transform_segments(job_id, backend, assigned, transformed_dir)
            job = self._require_job(job_id)
            job["segments"] = transformed
            self._save_job(job)
            self._stage(job_id, "transforming_segments", "completed", 0.68)

            self._stage(job_id, "temporal_stabilization_consistency_cleanup", "running", 0.70)
            stabilized = self._stabilize_segments(job_id, backend, transformed, stabilized_dir)
            job = self._require_job(job_id)
            job["segments"] = stabilized
            self._save_job(job)
            self._stage(job_id, "temporal_stabilization_consistency_cleanup", "completed", 0.76)

            self._stage(job_id, "quality_control_scoring", "running", 0.78)
            scored = self._score_segments(job_id, backend, stabilized)
            job = self._require_job(job_id)
            job["segments"] = scored
            job["qc_summary"] = summarize_qc(scored)
            self._save_job(job)
            self._stage(job_id, "quality_control_scoring", "completed", 0.84)

            self._stage(job_id, "reprocessing_failed_segments_if_needed", "running", 0.85)
            job = self._maybe_auto_retry_failed_segments(job_id, backend)
            self._stage(job_id, "reprocessing_failed_segments_if_needed", "completed", 0.88)

            self._stage(job_id, "stitching_final_video", "running", 0.90)
            stitched = self._stitch_segments(job_id)
            job = self._require_job(job_id)
            job["outputs"]["stitched_video_path"] = str(stitched)
            self._save_job(job)
            self._stage(job_id, "stitching_final_video", "completed", 0.93)

            self._stage(job_id, "restoring_audio_remuxing", "running", 0.94)
            final_video = self._remux_audio(job_id, stitched)
            job = self._require_job(job_id)
            job["outputs"]["final_video_path"] = str(final_video)
            self._save_job(job)
            self._stage(job_id, "restoring_audio_remuxing", "completed", 0.97)

            self._stage(job_id, "final_export_packaging", "running", 0.98)
            self._write_outputs(job_id, final_video)
            job = self._require_job(job_id)
            terminal_status = "needs_review" if (job.get("qc_summary") or {}).get("needs_review") else "completed"
            self._set_status(job_id, terminal_status)
            self._stage(job_id, "final_export_packaging", "completed", 1.0)
            self._append_log(job_id, f"Persona Video Studio job finished with status {terminal_status}.")
        except _JobCancelled:
            self._set_status(job_id, "cancelled")
            self._append_log(job_id, "Job cancelled at a safe checkpoint.")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Persona Video Studio job failed: %s", job_id)
            job = self.get_job(job_id) or {"job_id": job_id}
            job.setdefault("errors", []).append(str(exc))
            job["status"] = "failed"
            job["updated_at"] = _utc_now()
            self._save_job(job)
            self._append_log(job_id, f"Job failed: {exc}")
        finally:
            try:
                job = self.get_job(job_id) or {}
                if exclusive_state and bool(self.config.get("auto_restore_suspended_services", True)):
                    restore = self.gpu_manager.restore(exclusive_state)
                    job = self.get_job(job_id) or job
                    job.setdefault("exclusive_render", {})["restore"] = restore
                    self._save_job(job)
                    self._append_log(job_id, "Exclusive GPU Render Mode restore attempted.")
                if job and not (job.get("settings") or {}).get("keep_intermediate_files", False):
                    self._cleanup_intermediates(job)
            finally:
                with self._lock:
                    self._active_threads.pop(job_id, None)

    def _retry_segment_worker(self, job_id: str, segment_id: str) -> None:
        try:
            job = self._require_job(job_id)
            self._set_status(job_id, "running")
            backend = self.backends.get((job.get("settings") or {}).get("backend", "metadata_only_passthrough"))
            segments = job.get("segments", [])
            segment = next((item for item in segments if item.get("segment_id") == segment_id), None)
            if not segment:
                raise PersonaVideoValidationError("Segment not found")
            job_dir = Path(job["project"]["job_dir"])
            transformed_dir = job_dir / "work" / "transformed"
            stabilized_dir = job_dir / "work" / "stabilized"
            source_segment = Path(segment.get("source_segment_path") or "")
            if not source_segment.exists():
                raise PersonaVideoValidationError("Source segment missing; rerender requires retained segment media")
            transformed_path = transformed_dir / f"{segment_id}_retry_{len(job.get('rerender_history', []))}.mp4"
            result = backend.transform_segment(job, segment, source_segment, transformed_path, segment.get("gpu_assignment"))
            segment["status"] = "transformed" if result.get("ok") else "failed"
            segment["transform_result"] = result
            stable_path = stabilized_dir / f"{segment_id}_retry_stabilized.mp4"
            stab = backend.temporal_stabilize(job, segment, transformed_path, stable_path)
            segment["stabilized_path"] = str(stable_path)
            segment["temporal_stabilization"] = stab
            segment["qc"] = backend.score_segment(job, segment, stable_path)
            segment["status"] = "needs_review" if segment["qc"].get("needs_review") else "completed"
            job["segments"] = [segment if item.get("segment_id") == segment_id else item for item in segments]
            job["qc_summary"] = summarize_qc(job["segments"])
            self._save_job(job)
            stitched = self._stitch_segments(job_id)
            final = self._remux_audio(job_id, stitched)
            self._write_outputs(job_id, final)
            updated = self._require_job(job_id)
            updated["status"] = "needs_review" if (updated.get("qc_summary") or {}).get("needs_review") else "completed"
            updated["updated_at"] = _utc_now()
            self._save_job(updated)
            self._append_log(job_id, f"Retry finished for segment {segment_id}.")
        except Exception as exc:  # noqa: BLE001
            job = self.get_job(job_id) or {"job_id": job_id}
            job.setdefault("errors", []).append(f"retry {segment_id}: {exc}")
            job["status"] = "needs_review"
            job["updated_at"] = _utc_now()
            self._save_job(job)
            self._append_log(job_id, f"Retry failed for segment {segment_id}: {exc}")

    # ── processing stage helpers ───────────────────────────────────────

    def _materialize_segments(self, job_id: str, source: Path, segments: List[Dict[str, Any]], segment_dir: Path) -> List[Dict[str, Any]]:
        ffmpeg = find_binary("ffmpeg")
        if not ffmpeg and len(segments) > 1:
            raise PersonaVideoValidationError("FFmpeg is required for multi-segment processing")
        output: List[Dict[str, Any]] = []
        for segment in segments:
            self._checkpoint(job_id)
            seg = dict(segment)
            target = segment_dir / f"{seg['segment_id']}.mp4"
            if len(segments) == 1 and not ffmpeg:
                shutil.copy2(source, target)
            elif len(segments) == 1:
                shutil.copy2(source, target)
            else:
                cmd = [
                    ffmpeg,
                    "-y",
                    "-ss",
                    f"{float(seg['start_s']):.3f}",
                    "-to",
                    f"{float(seg['end_s']):.3f}",
                    "-i",
                    str(source),
                    "-map",
                    "0",
                    "-c",
                    "copy",
                    "-avoid_negative_ts",
                    "make_zero",
                    str(target),
                ]
                run_command(cmd, timeout=240)
            seg["source_segment_path"] = str(target)
            seg["status"] = "queued"
            output.append(seg)
        return output

    def _track_target(self, job_id: str, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        job = self._require_job(job_id)
        target = job.get("target_selection") or {}
        mode = target.get("mode") or "auto_detect_primary_subject"
        tracked = []
        for segment in segments:
            self._checkpoint(job_id)
            seg = dict(segment)
            seg["tracking"] = {
                "mode": mode,
                "track_id": target.get("track_id") or "primary_subject_auto_track",
                "confidence": 0.72 if mode == "auto_detect_primary_subject" else 0.82,
                "mask_path": None,
                "representative_frame_path": None,
                "notes": "Heuristic track scaffold; replace with detector/tracker plugin for mask editing.",
            }
            tracked.append(seg)
        return tracked

    def _transform_segments(
        self,
        job_id: str,
        backend: PersonaTransformBackend,
        segments: List[Dict[str, Any]],
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        total = max(1, len(segments))
        for index, segment in enumerate(segments, start=1):
            self._checkpoint(job_id)
            seg = dict(segment)
            source_segment = Path(seg.get("source_segment_path") or "")
            output_segment = output_dir / f"{seg['segment_id']}_transformed.mp4"
            try:
                result = backend.transform_segment(self._require_job(job_id), seg, source_segment, output_segment, seg.get("gpu_assignment"))
                seg["transform_result"] = result
                seg["transformed_path"] = str(output_segment)
                seg["status"] = "transformed" if result.get("ok") else "failed"
            except Exception as exc:  # noqa: BLE001
                seg["status"] = "failed"
                seg["transform_result"] = {"ok": False, "error": str(exc)}
            out.append(seg)
            self._update_overall_progress(job_id, 0.52 + (0.16 * index / total))
        return out

    def _stabilize_segments(self, job_id: str, backend: PersonaTransformBackend, segments: List[Dict[str, Any]], output_dir: Path) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for segment in segments:
            self._checkpoint(job_id)
            seg = dict(segment)
            transformed = Path(seg.get("transformed_path") or "")
            target = output_dir / f"{seg['segment_id']}_stabilized.mp4"
            if seg.get("status") == "failed" or not transformed.exists():
                seg["temporal_stabilization"] = {"ok": False, "error": "transform output missing"}
                out.append(seg)
                continue
            try:
                result = backend.temporal_stabilize(self._require_job(job_id), seg, transformed, target)
                seg["stabilized_path"] = str(target)
                seg["temporal_stabilization"] = result
                seg["status"] = "stabilized" if result.get("ok") else "failed"
            except Exception as exc:  # noqa: BLE001
                seg["status"] = "failed"
                seg["temporal_stabilization"] = {"ok": False, "error": str(exc)}
            out.append(seg)
        return out

    def _score_segments(self, job_id: str, backend: PersonaTransformBackend, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        thresholds = self.config.get("qc_thresholds", {}) or {}
        min_score = float(thresholds.get("minimum_segment_score", 0.6))
        out: List[Dict[str, Any]] = []
        for segment in segments:
            self._checkpoint(job_id)
            seg = dict(segment)
            stable = Path(seg.get("stabilized_path") or seg.get("transformed_path") or "")
            if seg.get("status") == "failed" or not stable.exists():
                qc = {
                    "persona_identity_confidence": None,
                    "target_tracking_confidence": (seg.get("tracking") or {}).get("confidence"),
                    "temporal_flicker_score": 1.0,
                    "frame_stability_score": 0.0,
                    "skipped_frame_count": 0,
                    "failed_frame_count": 1,
                    "warning_flags": ["missing_segment_output"],
                    "needs_review": True,
                    "score": 0.0,
                }
            else:
                qc = backend.score_segment(self._require_job(job_id), seg, stable)
                qc["needs_review"] = bool(qc.get("needs_review")) or float(qc.get("score") or 0.0) < min_score
            seg["qc"] = qc
            if qc.get("needs_review"):
                seg["status"] = "needs_review" if seg.get("status") != "failed" else "failed"
            else:
                seg["status"] = "completed"
            out.append(seg)
        return out

    def _maybe_auto_retry_failed_segments(self, job_id: str, backend: PersonaTransformBackend) -> Dict[str, Any]:
        job = self._require_job(job_id)
        auto_threshold = float(self.config.get("automatic_failed_segment_rerender_threshold", 0.55))
        auto_retry = bool((job.get("settings") or {}).get("auto_retry_failed_segments", True))
        if not auto_retry:
            return job
        failed = [seg for seg in job.get("segments", []) if float((seg.get("qc") or {}).get("score") or 0.0) < auto_threshold]
        for seg in failed[:2]:
            seg_id = seg.get("segment_id")
            job.setdefault("rerender_history", []).append({
                "segment_id": seg_id,
                "reason": "auto_retry_threshold",
                "score": (seg.get("qc") or {}).get("score"),
                "timestamp": _utc_now(),
                "attempted": False,
                "note": "Auto retry scaffold recorded; production backends may requeue failed segment only.",
            })
        self._save_job(job)
        return job

    def _stitch_segments(self, job_id: str) -> Path:
        job = self._require_job(job_id)
        job_dir = Path(job["project"]["job_dir"])
        final_dir = job_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        segments = job.get("segments", [])
        usable = [Path(seg.get("stabilized_path") or seg.get("transformed_path") or "") for seg in segments if Path(seg.get("stabilized_path") or seg.get("transformed_path") or "").exists()]
        if not usable:
            raise PersonaVideoValidationError("No transformed segments available for stitching")
        stitched = final_dir / f"{job_id}_stitched.mp4"
        if len(usable) == 1:
            shutil.copy2(usable[0], stitched)
            return stitched
        ffmpeg = find_binary("ffmpeg")
        if not ffmpeg:
            raise PersonaVideoValidationError("FFmpeg is required to stitch multiple segments")
        manifest = job_dir / "segment_concat_manifest.txt"
        manifest.write_text("".join(f"file '{str(path).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for path in usable))
        cmd = build_concat_command(manifest, stitched)
        run_command(cmd, timeout=600)
        job["segment_manifest"] = str(manifest)
        self._save_job(job)
        return stitched

    def _remux_audio(self, job_id: str, stitched: Path) -> Path:
        job = self._require_job(job_id)
        settings = job.get("settings") or {}
        source = Path(job["source"]["path"])
        final_dir = Path(job["project"]["job_dir"]) / "final"
        output = final_dir / f"{job_id}_final.mp4"
        audio_mode = settings.get("audio_mode", "preserve_original")
        alt_audio = settings.get("alternate_audio_path")
        if audio_mode == "preserve_original" and not (job.get("source", {}).get("probe") or {}).get("audio_present"):
            shutil.copy2(stitched, output)
            return output
        ffmpeg = find_binary("ffmpeg")
        if not ffmpeg:
            shutil.copy2(stitched, output)
            job.setdefault("warnings", []).append("FFmpeg missing; audio remux skipped and stitched video copied as final output.")
            self._save_job(job)
            return output
        cmd = build_remux_command(stitched, source, output, audio_mode, alternate_audio_path=Path(alt_audio) if alt_audio else None)
        run_command(cmd, timeout=600)
        return output

    def _write_outputs(self, job_id: str, final_video: Path) -> None:
        job = self._require_job(job_id)
        final_dir = Path(job["project"]["job_dir"]) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        report = build_metadata_report(job)
        report_path = final_dir / f"{job_id}_metadata_report.json"
        qc_path = final_dir / f"{job_id}_qc_summary.json"
        atomic_write_json(report_path, report)
        atomic_write_json(qc_path, job.get("qc_summary") or {})
        poster = final_dir / f"{job_id}_poster.jpg"
        poster_path = create_thumbnail(final_video, poster, duration_s=1.0)
        job = self._require_job(job_id)
        job["outputs"].update({
            "final_video_path": str(final_video),
            "metadata_report_path": str(report_path),
            "qc_summary_path": str(qc_path),
            "poster_path": str(poster_path) if poster_path else None,
        })
        job["updated_at"] = _utc_now()
        self._save_job(job)

    # ── state helpers ─────────────────────────────────────────────────

    def _start_thread(self, job_id: str, target: Any, *args: Any) -> None:
        with self._lock:
            existing = self._active_threads.get(job_id)
            if existing and existing.is_alive():
                return
            thread = threading.Thread(target=target, args=(job_id, *args), daemon=True, name=f"persona-video-{job_id}")
            self._active_threads[job_id] = thread
            thread.start()

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_root / _safe_filename(job_id) / "job.json"

    def _require_job(self, job_id: str) -> Dict[str, Any]:
        job = self.get_job(job_id)
        if not job:
            raise PersonaVideoValidationError("Job not found")
        return job

    def _save_job(self, job: Dict[str, Any]) -> None:
        job_id = job.get("job_id")
        if not job_id:
            raise PersonaVideoValidationError("Cannot save job without job_id")
        path = self._job_path(str(job_id))
        path.parent.mkdir(parents=True, exist_ok=True)
        job["updated_at"] = _utc_now()
        atomic_write_json(path, job)

    def _append_log(self, job_id: str, message: str) -> None:
        job = self.get_job(job_id)
        if not job:
            return
        job.setdefault("logs", []).append({"timestamp": _utc_now(), "message": message})
        self._save_job(job)

    def _set_status(self, job_id: str, status: str) -> None:
        if status not in JOB_STATUSES:
            raise PersonaVideoValidationError(f"Invalid job status '{status}'")
        job = self._require_job(job_id)
        job["status"] = status
        self._save_job(job)

    def _stage(self, job_id: str, stage_id: str, status: str, progress: float) -> None:
        job = self._require_job(job_id)
        for stage in job.get("pipeline", {}).get("stages", []):
            if stage.get("id") == stage_id:
                stage["status"] = status
                stage["progress"] = max(float(stage.get("progress") or 0.0), progress)
                stage["updated_at"] = _utc_now()
        job.setdefault("pipeline", {})["current_stage"] = stage_id if status == "running" else job.get("pipeline", {}).get("current_stage")
        job["pipeline"]["overall_progress"] = max(float(job["pipeline"].get("overall_progress") or 0.0), progress)
        self._save_job(job)
        self._append_log(job_id, f"Stage {stage_id}: {status}.")

    def _update_overall_progress(self, job_id: str, progress: float) -> None:
        job = self.get_job(job_id)
        if not job:
            return
        job.setdefault("pipeline", {})["overall_progress"] = max(float(job.get("pipeline", {}).get("overall_progress") or 0.0), progress)
        self._save_job(job)

    def _checkpoint(self, job_id: str) -> None:
        job = self._require_job(job_id)
        if job.get("cancel_requested") or job.get("status") == "cancelled":
            raise _JobCancelled()
        while job.get("pause_requested"):
            job["status"] = "paused"
            self._save_job(job)
            time.sleep(1.0)
            job = self._require_job(job_id)
            if job.get("cancel_requested"):
                raise _JobCancelled()
        if job.get("status") == "paused":
            job["status"] = "running"
            self._save_job(job)

    def _cleanup_intermediates(self, job: Dict[str, Any]) -> None:
        work_dir = Path(job.get("project", {}).get("job_dir", "")) / "work"
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
            self._append_log(job.get("job_id", ""), "Intermediate work files removed per job settings.")

    def _normalize_job_settings(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        quality = _normalize_choice(raw.get("quality_preset"), {"fast_preview", "balanced", "maximum_quality"}, self.config.get("default_quality_preset", "balanced"))
        backend = str(raw.get("backend") or self.config.get("backend") or "metadata_only_passthrough")
        default_scope = "metadata_validation_only" if backend == "metadata_only_passthrough" else "face_identity_only"
        scope = _normalize_choice(raw.get("transformation_scope"), set(TRANSFORMATION_SCOPES), default_scope)
        audio_mode = "preserve_original"
        if raw.get("strip_audio"):
            audio_mode = "strip"
        elif raw.get("replace_audio") or raw.get("alternate_audio_path"):
            audio_mode = "replace"
        elif raw.get("preserve_original_audio") is False:
            audio_mode = "strip"
        return {
            "backend": backend,
            "quality_preset": quality,
            "transformation_scope": scope,
            "temporal_consistency_priority": _normalize_choice(raw.get("temporal_consistency_priority"), {"low", "medium", "high", "maximum"}, "high"),
            "flicker_reduction": bool(raw.get("flicker_reduction", True)),
            "identity_stability_priority": float(raw.get("identity_stability_priority", 0.8)),
            "preserve_lighting_color_relationship": bool(raw.get("preserve_lighting_color_relationship", True)),
            "audio_mode": audio_mode,
            "alternate_audio_path": raw.get("alternate_audio_path"),
            "exclusive_gpu_render_mode": bool(raw.get("exclusive_gpu_render_mode", self.config.get("exclusive_gpu_render_mode_default", False))),
            "use_all_gpus_where_supported": bool(raw.get("use_all_gpus_where_supported", True)),
            "gpu_strategy": GPUStrategySelector.normalize_strategy(str(raw.get("gpu_strategy") or self.config.get("default_gpu_strategy", "auto"))),
            "keep_intermediate_files": bool(raw.get("keep_intermediate_files", self.config.get("keep_intermediates_default", False))),
            "segment_size_preference": _normalize_choice(raw.get("segment_size_preference"), {"auto", "short", "long"}, "auto"),
            "auto_retry_failed_segments": bool(raw.get("auto_retry_failed_segments", True)),
        }

    def _resolve_media_path(self, value: str) -> Path:
        if not value:
            raise PersonaVideoValidationError("Media path is required")
        p = Path(value).expanduser()
        if not p.is_absolute():
            p = self.repo_root / p
        return p.resolve(strict=False)

    def _resolve_dir(self, value: str) -> Path:
        p = Path(value).expanduser()
        if not p.is_absolute():
            p = self.repo_root / p
        p = p.resolve(strict=False)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _resolve_output_folder(self, value: Optional[str], project_title: Optional[str]) -> Path:
        if value:
            folder = self._resolve_dir(str(value))
        else:
            slug = _slugify(project_title or "persona-video-project")
            folder = self.output_root / "projects" / slug
            folder.mkdir(parents=True, exist_ok=True)
        return folder


class _JobCancelled(Exception):
    pass


# ── standalone helpers used by service, routes, and tests ─────────────────


def validate_rights_acknowledgement(payload: Dict[str, Any]) -> Dict[str, Any]:
    missing = [field for field in RIGHTS_ACK_FIELDS if payload.get(field) is not True]
    if missing:
        raise PersonaVideoValidationError(
            "Rights/consent acknowledgement is incomplete: " + ", ".join(missing)
        )
    return payload


def enrich_rights_acknowledgement(payload: Dict[str, Any]) -> Dict[str, Any]:
    validate_rights_acknowledgement(payload)
    enriched = {field: bool(payload.get(field)) for field in RIGHTS_ACK_FIELDS}
    enriched["acknowledged_at"] = payload.get("acknowledged_at") or _utc_now()
    enriched["acknowledged_by"] = payload.get("acknowledged_by") or "local_user"
    enriched["storage_note"] = "Stored only as local Persona Video Studio job metadata and final sidecar report."
    return enriched


def probe_video(source: Path) -> Dict[str, Any]:
    ffprobe = find_binary("ffprobe")
    stat = source.stat()
    base = {
        "filename": source.name,
        "path": str(source),
        "size_bytes": stat.st_size,
        "container": source.suffix.lower().lstrip("."),
        "duration_s": 0.0,
        "width": None,
        "height": None,
        "fps": None,
        "video_codec": None,
        "audio_present": False,
        "audio_codec": None,
        "probe_backend": "filesystem_only",
    }
    if not ffprobe:
        base["warning"] = "ffprobe not found; only filesystem metadata is available."
        return base
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_streams", "-show_format", "-print_format", "json", str(source)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            base["warning"] = result.stderr.strip()[:500]
            return base
        data = json.loads(result.stdout or "{}")
        streams = data.get("streams") or []
        fmt = data.get("format") or {}
        video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
        audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), {})
        duration = _safe_float(video.get("duration")) or _safe_float(fmt.get("duration"))
        fps = _parse_rate(video.get("avg_frame_rate") or video.get("r_frame_rate"))
        base.update({
            "duration_s": round(duration, 3),
            "width": video.get("width"),
            "height": video.get("height"),
            "fps": round(fps, 3) if fps else None,
            "video_codec": video.get("codec_name"),
            "audio_present": bool(audio),
            "audio_codec": audio.get("codec_name") if audio else None,
            "container": fmt.get("format_name") or base["container"],
            "bit_rate": _safe_int(fmt.get("bit_rate")),
            "probe_backend": "ffprobe",
        })
        return base
    except Exception as exc:  # noqa: BLE001
        base["warning"] = str(exc)
        return base


def build_segments(duration_s: float, preference: str = "auto", quality_preset: str = "balanced") -> List[Dict[str, Any]]:
    if duration_s <= 0:
        duration_s = float(QUALITY_PRESETS.get(quality_preset, QUALITY_PRESETS["balanced"])["segment_length_s"])
    if preference == "short":
        segment_len = 20.0
    elif preference == "long":
        segment_len = 90.0
    else:
        segment_len = float(QUALITY_PRESETS.get(quality_preset, QUALITY_PRESETS["balanced"])["segment_length_s"])
    count = max(1, int(math.ceil(duration_s / max(1.0, segment_len))))
    segments = []
    for idx in range(count):
        start = round(idx * segment_len, 3)
        end = round(min(duration_s, (idx + 1) * segment_len), 3)
        if end <= start:
            end = round(start + segment_len, 3)
        segments.append({
            "segment_id": f"seg_{idx + 1:03d}",
            "index": idx,
            "start_s": start,
            "end_s": end,
            "duration_s": round(end - start, 3),
            "status": "queued",
            "shot_detection": {"method": "time_based_fallback", "confidence": 0.5},
        })
    return segments


def summarize_qc(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [float((seg.get("qc") or {}).get("score") or 0.0) for seg in segments]
    needs_review = [seg.get("segment_id") for seg in segments if (seg.get("qc") or {}).get("needs_review") or seg.get("status") in {"failed", "needs_review"}]
    return {
        "segment_count": len(segments),
        "average_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "minimum_score": round(min(scores), 3) if scores else 0.0,
        "maximum_score": round(max(scores), 3) if scores else 0.0,
        "failed_segments": [seg.get("segment_id") for seg in segments if seg.get("status") == "failed"],
        "needs_review_segments": needs_review,
        "needs_review": bool(needs_review),
        "segments": [{"segment_id": seg.get("segment_id"), "status": seg.get("status"), "qc": seg.get("qc")} for seg in segments],
    }


def mark_segment_for_retry(job: Dict[str, Any], segment_id: str, reason: str = "manual_retry") -> Dict[str, Any]:
    found = False
    updated_segments = []
    for segment in job.get("segments", []):
        seg = dict(segment)
        if seg.get("segment_id") == segment_id:
            found = True
            seg["status"] = "queued"
            seg["retry_requested"] = True
            seg["retry_requested_at"] = _utc_now()
        updated_segments.append(seg)
    if not found:
        raise PersonaVideoValidationError("Segment not found")
    job["segments"] = updated_segments
    job["status"] = "running"
    job.setdefault("rerender_history", []).append({
        "segment_id": segment_id,
        "reason": reason,
        "timestamp": _utc_now(),
        "attempted": True,
    })
    job["updated_at"] = _utc_now()
    return job


def build_remux_command(
    video_input: Path,
    source_video: Path,
    output: Path,
    audio_mode: str = "preserve_original",
    alternate_audio_path: Optional[Path] = None,
) -> List[str]:
    ffmpeg = find_binary("ffmpeg") or "ffmpeg"
    mode = audio_mode or "preserve_original"
    if mode == "strip":
        return [ffmpeg, "-y", "-i", str(video_input), "-c:v", "copy", "-an", str(output)]
    if mode == "replace":
        if not alternate_audio_path:
            raise PersonaVideoValidationError("Alternate audio path is required when audio_mode is replace")
        return [
            ffmpeg,
            "-y",
            "-i",
            str(video_input),
            "-i",
            str(alternate_audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output),
        ]
    return [
        ffmpeg,
        "-y",
        "-i",
        str(video_input),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output),
    ]


def build_concat_command(manifest_path: Path, output_path: Path) -> List[str]:
    ffmpeg = find_binary("ffmpeg") or "ffmpeg"
    return [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(manifest_path), "-c", "copy", str(output_path)]


def build_metadata_report(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema": "edison.persona_video.report.v1",
        "job_id": job.get("job_id"),
        "job_type": "persona_video",
        "status": job.get("status"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "project": job.get("project"),
        "rights_acknowledgement": job.get("rights_acknowledgement"),
        "source": job.get("source"),
        "persona": job.get("persona"),
        "target_selection": job.get("target_selection"),
        "settings": job.get("settings"),
        "backend_capabilities": job.get("backend_capabilities"),
        "pipeline": job.get("pipeline"),
        "gpu": job.get("gpu"),
        "exclusive_render": job.get("exclusive_render"),
        "segment_manifest": job.get("segment_manifest"),
        "segments": job.get("segments"),
        "qc_summary": job.get("qc_summary"),
        "rerender_history": job.get("rerender_history", []),
        "outputs": job.get("outputs"),
        "warnings": job.get("warnings", []),
        "errors": job.get("errors", []),
        "local_metadata_notice": "Rights acknowledgement is stored only in local Edison job/report metadata.",
    }


def create_thumbnail(source: Path, output: Path, duration_s: Optional[float] = None) -> Optional[Path]:
    ffmpeg = find_binary("ffmpeg")
    if not ffmpeg:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    seek = max(0.0, min(float(duration_s or 2.0) * 0.1, 5.0))
    cmd = [ffmpeg, "-y", "-ss", f"{seek:.3f}", "-i", str(source), "-frames:v", "1", "-q:v", "2", str(output)]
    try:
        run_command(cmd, timeout=60)
        return output if output.exists() else None
    except Exception:
        return None


def create_contact_sheet(source: Path, output: Path, duration_s: Optional[float] = None) -> Optional[Path]:
    ffmpeg = find_binary("ffmpeg")
    if not ffmpeg:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(float(duration_s or 60.0), 1.0)
    interval = max(1.0, duration / 9.0)
    vf = f"fps=1/{interval:.3f},scale=240:-1,tile=3x3"
    cmd = [ffmpeg, "-y", "-i", str(source), "-vf", vf, "-frames:v", "1", "-q:v", "3", str(output)]
    try:
        run_command(cmd, timeout=90)
        return output if output.exists() else None
    except Exception:
        return None


def run_command(cmd: List[str], timeout: int = 120) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "command failed")[:1000])
    return result


def find_binary(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    for directory in ("/usr/bin", "/usr/local/bin", "/snap/bin", "/bin"):
        candidate = Path(directory) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    edison_cfg = config.get("edison", config)
    persona_cfg = dict(edison_cfg.get("persona_video", {}))
    return {
        "enabled": persona_cfg.get("enabled", True),
        "default_output_directory": persona_cfg.get("default_output_directory", "outputs/persona_video"),
        "temp_working_directory": persona_cfg.get("temp_working_directory", "outputs/persona_video/tmp"),
        "upload_directory": persona_cfg.get("upload_directory", "uploads/persona_video"),
        "keep_intermediates_default": persona_cfg.get("keep_intermediates_default", False),
        "default_quality_preset": persona_cfg.get("default_quality_preset", "balanced"),
        "default_gpu_strategy": persona_cfg.get("default_gpu_strategy", "auto"),
        "exclusive_gpu_render_mode_default": persona_cfg.get("exclusive_gpu_render_mode_default", False),
        "auto_restore_suspended_services": persona_cfg.get("auto_restore_suspended_services", True),
        "vram_min_free_mb_by_gpu": persona_cfg.get("vram_min_free_mb_by_gpu", {"0": 18000, "1": 10000, "2": 10000}),
        "maximum_concurrent_segment_workers": persona_cfg.get("maximum_concurrent_segment_workers", 2),
        "automatic_failed_segment_rerender_threshold": persona_cfg.get("automatic_failed_segment_rerender_threshold", 0.55),
        "qc_thresholds": persona_cfg.get("qc_thresholds", {"minimum_segment_score": 0.6}),
        "backend": persona_cfg.get("backend", "metadata_only_passthrough"),
        "exclusive_wait_timeout_s": persona_cfg.get("exclusive_wait_timeout_s", 60),
        "exclusive_poll_interval_s": persona_cfg.get("exclusive_poll_interval_s", 2),
        "comfyui_workflows_dir": persona_cfg.get("comfyui_workflows_dir", "config/persona_video/comfyui_workflows"),
    }


def _normalize_choice(value: Any, allowed: set[str], default: str) -> str:
    candidate = str(value or default).strip().lower().replace(" ", "_").replace("-", "_")
    return candidate if candidate in allowed else default


def _parse_rate(value: Any) -> float:
    text = str(value or "")
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        den = _safe_float(denominator)
        if den == 0:
            return 0.0
        return _safe_float(numerator) / den
    return _safe_float(text)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_filename(value: str) -> str:
    name = Path(value).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or f"file_{uuid.uuid4().hex[:8]}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower()).strip("-")
    return slug or f"persona-video-{uuid.uuid4().hex[:8]}"
