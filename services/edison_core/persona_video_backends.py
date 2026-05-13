"""Persona Video Studio backend interfaces.

The Persona Video Studio pipeline is intentionally backend-pluggable. Edison
owns intake, rights metadata, segmentation, GPU orchestration, job recovery,
QC, and packaging; specialized identity-transformation adapters can be added
without changing the orchestration layer.

The included ``MetadataOnlyPassthroughBackend`` is a validation/development
adapter. It copies input segments through the pipeline so operators can test
job handling, media probing, remuxing, GPU scheduling, reports, and UI flows
before installing a real consenting-persona transformation backend. It does
NOT claim to synthesize or transform identity.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .comfyui_integration import ComfyUIExecutionService, discover_workflow_templates, inject_workflow_variables


@dataclass(frozen=True)
class PersonaBackendCapabilities:
    """Capability metadata exposed to the UI and scheduler."""

    backend_id: str
    display_name: str
    available: bool
    description: str
    supported_input_types: List[str] = field(default_factory=list)
    supported_transformation_scopes: List[str] = field(default_factory=list)
    supports_segment_batch_mode: bool = False
    supports_parallel_segment_processing: bool = False
    supports_multi_gpu: bool = False
    supports_advanced_multigpu: bool = False
    supports_comfyui_workflow: bool = False
    supports_temporal_consistency_pass: bool = False
    produces_synthetic_persona: bool = False
    disabled_for_final_render: bool = False
    setup_required: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PersonaTransformBackend:
    """Interface every Persona Video Studio backend must implement."""

    backend_id = "base"

    def __init__(self, repo_root: Path, config: Optional[Dict[str, Any]] = None) -> None:
        self.repo_root = repo_root.resolve()
        self.config = config or {}

    def is_available(self) -> bool:
        return self.get_capabilities().available

    def get_capabilities(self) -> PersonaBackendCapabilities:
        raise NotImplementedError

    def prepare(self, job: Dict[str, Any], workspace: Path) -> Dict[str, Any]:
        raise NotImplementedError

    def transform_segment(
        self,
        job: Dict[str, Any],
        segment: Dict[str, Any],
        source_segment: Path,
        output_segment: Path,
        gpu_assignment: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def temporal_stabilize(
        self,
        job: Dict[str, Any],
        segment: Dict[str, Any],
        transformed_segment: Path,
        output_segment: Path,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def score_segment(
        self,
        job: Dict[str, Any],
        segment: Dict[str, Any],
        transformed_segment: Path,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def cleanup(self, job: Dict[str, Any], workspace: Path) -> Dict[str, Any]:
        return {"ok": True, "backend": self.backend_id, "cleaned": False}


class MetadataOnlyPassthroughBackend(PersonaTransformBackend):
    """Development backend used to validate the pipeline end-to-end.

    It preserves source pixels and timing. This is useful for local smoke tests
    and validating legal metadata/reporting, but it is not a production persona
    identity transform.
    """

    backend_id = "metadata_only_passthrough"

    def get_capabilities(self) -> PersonaBackendCapabilities:
        return PersonaBackendCapabilities(
            backend_id=self.backend_id,
            display_name="Metadata-only passthrough / pipeline validation",
            available=True,
            description=(
                "Validates Persona Video Studio intake, segmentation, GPU scheduling, "
                "QC metadata, stitching, and audio handling without altering identity."
            ),
            supported_input_types=["mp4", "mov", "mkv", "webm", "avi", "m4v"],
            supported_transformation_scopes=["metadata_validation_only"],
            supports_segment_batch_mode=True,
            supports_parallel_segment_processing=True,
            supports_multi_gpu=True,
            supports_advanced_multigpu=False,
            supports_comfyui_workflow=False,
            supports_temporal_consistency_pass=False,
            produces_synthetic_persona=False,
            disabled_for_final_render=True,
            setup_required=[
                "Install/register a real PersonaTransformBackend adapter for production identity transformation."
            ],
            notes="Does not modify frames; intended for development, testing, and dry-run validation only.",
        )

    def prepare(self, job: Dict[str, Any], workspace: Path) -> Dict[str, Any]:
        return {
            "ok": True,
            "backend": self.backend_id,
            "workspace": str(workspace),
            "warning": "Passthrough backend active: final video is not transformed.",
        }

    def transform_segment(
        self,
        job: Dict[str, Any],
        segment: Dict[str, Any],
        source_segment: Path,
        output_segment: Path,
        gpu_assignment: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        output_segment.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_segment, output_segment)
        return {
            "ok": True,
            "backend": self.backend_id,
            "output_path": str(output_segment),
            "gpu_assignment": gpu_assignment or {},
            "warning": "Segment copied without identity transformation.",
        }

    def temporal_stabilize(
        self, job: Dict[str, Any], segment: Dict[str, Any], transformed_segment: Path, output_segment: Path) -> Dict[str, Any]:
        output_segment.parent.mkdir(parents=True, exist_ok=True)
        if transformed_segment.resolve(strict=False) != output_segment.resolve(strict=False):
            shutil.copy2(transformed_segment, output_segment)
        return {
            "ok": True,
            "backend": self.backend_id,
            "output_path": str(output_segment),
            "applied": False,
            "reason": "Passthrough backend has no temporal consistency model.",
        }

    def score_segment(self, job: Dict[str, Any], segment: Dict[str, Any], transformed_segment: Path) -> Dict[str, Any]:
        exists = transformed_segment.exists()
        size = transformed_segment.stat().st_size if exists else 0
        tracking_confidence = float(segment.get("tracking", {}).get("confidence", 0.72))
        return {
            "persona_identity_confidence": None,
            "target_tracking_confidence": round(tracking_confidence, 3),
            "temporal_flicker_score": 0.12 if exists else 1.0,
            "frame_stability_score": 0.86 if exists and size > 0 else 0.0,
            "skipped_frame_count": 0,
            "failed_frame_count": 0 if exists and size > 0 else 1,
            "warning_flags": ["passthrough_backend_no_identity_transform"],
            "needs_review": True,
            "score": 0.78 if exists and size > 0 else 0.0,
        }


class ComfyUITemplatePersonaBackend(PersonaTransformBackend):
    """Integration placeholder for curated ComfyUI persona workflows.

    This adapter becomes available when workflow templates are present in
    ``config/persona_video/comfyui_workflows``. The actual execution bridge is
    intentionally left as a small, documented integration point because ComfyUI
    persona workflows vary heavily by installed custom nodes.
    """

    backend_id = "comfyui_persona_workflow"

    def _workflow_dir(self) -> Path:
        configured = self.config.get("comfyui_workflows_dir")
        if configured:
            p = Path(configured)
            return p if p.is_absolute() else (self.repo_root / p)
        return self.repo_root / "config" / "persona_video" / "comfyui_workflows"

    def _workflow_files(self) -> List[Path]:
        return [template.path for template in self._workflow_templates()]

    def _workflow_templates(self):
        return discover_workflow_templates(self._workflow_dir())

    def get_capabilities(self) -> PersonaBackendCapabilities:
        templates = self._workflow_templates()
        available_templates = [template for template in templates if template.validation.ok]
        available = bool(available_templates)
        setup_required: List[str] = []
        for template in templates:
            setup_required.extend(template.validation.errors)
            setup_required.extend(template.validation.warnings)
        return PersonaBackendCapabilities(
            backend_id=self.backend_id,
            display_name="Curated ComfyUI persona workflow adapter",
            available=available,
            description=(
                "Runs curated ComfyUI workflow templates with injected source segment, "
                "persona references/model paths, output path, and optional device hints."
            ),
            supported_input_types=["image_sequence", "video_segment"],
            supported_transformation_scopes=[
                "face_identity_only",
                "face_hair_head_region",
                "full_visible_persona_styling",
            ],
            supports_segment_batch_mode=True,
            supports_parallel_segment_processing=True,
            supports_multi_gpu=False,
            supports_advanced_multigpu=False,
            supports_comfyui_workflow=True,
            supports_temporal_consistency_pass=True,
            produces_synthetic_persona=True,
            disabled_for_final_render=not available,
            setup_required=[] if available else setup_required or [
                "Add curated ComfyUI workflow JSON templates under config/persona_video/comfyui_workflows/.",
                "Install the custom nodes and model weights required by those templates.",
            ],
            notes=(
                f"Detected {len(templates)} workflow template(s), {len(available_templates)} valid."
                if templates else
                "No workflow templates detected; adapter is not available."
            ),
        )

    def prepare(self, job: Dict[str, Any], workspace: Path) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("ComfyUI persona workflow backend is not available")
        templates = self._workflow_templates()
        return {
            "ok": True,
            "backend": self.backend_id,
            "workflow_templates": [template.to_dict() for template in templates],
            "integration_point": "Templates are injected and submitted through Edison ComfyUIExecutionService.",
        }

    def transform_segment(
        self,
        job: Dict[str, Any],
        segment: Dict[str, Any],
        source_segment: Path,
        output_segment: Path,
        gpu_assignment: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        templates = [template for template in self._workflow_templates() if template.validation.ok]
        if not templates:
            return {
                "ok": False,
                "backend": self.backend_id,
                "error": "No valid ComfyUI persona workflow templates are available.",
            }
        settings = job.get("settings") or {}
        requested = str(settings.get("workflow_template") or "").strip()
        template = next((item for item in templates if item.template_id == requested or item.name == requested), templates[0])
        output_segment.parent.mkdir(parents=True, exist_ok=True)
        persona = job.get("persona") or {}
        variables = {
            "source_segment_path": str(source_segment),
            "source_frame_path": str(source_segment),
            "persona_paths": persona.get("paths") or [],
            "persona_pack_path": ",".join(persona.get("paths") or []),
            "output_path": str(output_segment),
            "quality_preset": settings.get("quality_preset", "balanced"),
            "transformation_scope": settings.get("transformation_scope", "face_identity_only"),
            "segment_id": segment.get("segment_id"),
            "gpu_index": (gpu_assignment or {}).get("gpu_index", ""),
            "gpu_name": (gpu_assignment or {}).get("gpu_name", ""),
        }
        workflow = inject_workflow_variables(template.workflow, variables)
        comfy_cfg = (self.config.get("comfyui") or self.config.get("comfyui_backend") or {})
        base_url = str(comfy_cfg.get("base_url") or self.config.get("comfyui_base_url") or "http://127.0.0.1:8188")
        service = ComfyUIExecutionService(base_url=base_url, timeout_s=float(self.config.get("comfyui_timeout_s", 10)))
        submit = service.submit(workflow, extra_data={"edison_job_id": job.get("job_id"), "segment_id": segment.get("segment_id")})
        if not submit.get("ok"):
            return {
                "ok": False,
                "backend": self.backend_id,
                "workflow_template": template.to_dict(),
                "error": submit.get("error") or "ComfyUI submit failed",
                "submit": submit,
            }
        poll = service.poll_until_complete(
            str(submit.get("prompt_id")),
            timeout_s=float(self.config.get("comfyui_poll_timeout_s", 900)),
            poll_interval_s=float(self.config.get("comfyui_poll_interval_s", 2)),
        )
        if not output_segment.exists():
            return {
                "ok": False,
                "backend": self.backend_id,
                "workflow_template": template.to_dict(),
                "prompt_id": submit.get("prompt_id"),
                "poll": poll,
                "error": "ComfyUI completed or returned state, but the expected output_path was not created.",
            }
        return {
            "ok": bool(poll.get("ok")),
            "backend": self.backend_id,
            "workflow_template": template.to_dict(),
            "prompt_id": submit.get("prompt_id"),
            "output_path": str(output_segment),
            "poll": poll,
            "gpu_assignment": gpu_assignment or {},
        }

    def temporal_stabilize(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError("ComfyUI temporal stabilization is workflow-template specific.")

    def score_segment(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "persona_identity_confidence": None,
            "target_tracking_confidence": None,
            "temporal_flicker_score": None,
            "frame_stability_score": None,
            "skipped_frame_count": 0,
            "failed_frame_count": 0,
            "warning_flags": ["qc_adapter_not_configured"],
            "needs_review": True,
            "score": 0.0,
        }


class PersonaBackendRegistry:
    """Small registry for persona transformation backends."""

    def __init__(self, repo_root: Path, config: Optional[Dict[str, Any]] = None) -> None:
        self.repo_root = repo_root.resolve()
        self.config = config or {}
        self._backends: Dict[str, PersonaTransformBackend] = {}
        self.register(MetadataOnlyPassthroughBackend(self.repo_root, self.config))
        self.register(ComfyUITemplatePersonaBackend(self.repo_root, self.config))

    def register(self, backend: PersonaTransformBackend) -> None:
        self._backends[backend.backend_id] = backend

    def get(self, backend_id: str) -> PersonaTransformBackend:
        if backend_id not in self._backends:
            raise KeyError(f"Unknown persona transform backend: {backend_id}")
        return self._backends[backend_id]

    def list(self) -> List[Dict[str, Any]]:
        return [backend.get_capabilities().to_dict() for backend in self._backends.values()]

    def available_ids(self) -> List[str]:
        return [backend_id for backend_id, backend in self._backends.items() if backend.is_available()]

    def __iter__(self) -> Iterable[PersonaTransformBackend]:
        return iter(self._backends.values())
