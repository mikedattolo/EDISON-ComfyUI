"""GPU orchestration helpers for Persona Video Studio.

These utilities keep Edison honest about heterogeneous GPUs: VRAM is tracked
per card and scheduling is segment/job based unless a backend explicitly
advertises advanced multi-GPU placement support.
"""

from __future__ import annotations

import gc
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GPUDevice:
    index: int
    name: str
    total_mb: float
    free_mb: float = 0.0
    used_mb: float = 0.0
    utilization_percent: Optional[float] = None

    @property
    def is_primary_3090(self) -> bool:
        return "3090" in self.name.lower()

    @property
    def is_16gb_aux(self) -> bool:
        return 14500 <= float(self.total_mb or 0) <= 18000

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["role_hint"] = "primary_24gb" if self.is_primary_3090 else "auxiliary_16gb" if self.is_16gb_aux else "available"
        return payload


@dataclass
class GPUStrategyPlan:
    requested_strategy: str
    effective_strategy: str
    primary_gpu: Optional[int]
    worker_gpus: List[int]
    auxiliary_gpus: List[int] = field(default_factory=list)
    max_concurrent_workers: int = 1
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    advanced_mode_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GPUStrategySelector:
    """Select GPU worker plans for heterogeneous local render workloads."""

    AUTO = "auto"
    PRIMARY_AUX = "primary_aux"
    PARALLEL_ALL = "parallel_all"
    PRIMARY_ONLY = "3090_only"
    ADVANCED = "advanced_multigpu"

    @staticmethod
    def normalize_strategy(value: str) -> str:
        lowered = (value or "auto").strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "auto": GPUStrategySelector.AUTO,
            "rtx_3090_primary_+_other_gpus_for_auxiliary_tasks": GPUStrategySelector.PRIMARY_AUX,
            "3090_primary_auxiliary": GPUStrategySelector.PRIMARY_AUX,
            "primary_auxiliary": GPUStrategySelector.PRIMARY_AUX,
            "primary_aux": GPUStrategySelector.PRIMARY_AUX,
            "parallel_segment_processing_across_all_gpus": GPUStrategySelector.PARALLEL_ALL,
            "parallel_segments": GPUStrategySelector.PARALLEL_ALL,
            "parallel_all": GPUStrategySelector.PARALLEL_ALL,
            "3090_only": GPUStrategySelector.PRIMARY_ONLY,
            "rtx_3090_only": GPUStrategySelector.PRIMARY_ONLY,
            "advanced_multigpu_backend_mode": GPUStrategySelector.ADVANCED,
            "advanced_multigpu": GPUStrategySelector.ADVANCED,
        }
        return aliases.get(lowered, GPUStrategySelector.AUTO)

    @staticmethod
    def choose_primary_gpu(gpus: List[GPUDevice]) -> Optional[GPUDevice]:
        if not gpus:
            return None
        for gpu in gpus:
            if gpu.is_primary_3090:
                return gpu
        return max(gpus, key=lambda item: float(item.total_mb or 0))

    @classmethod
    def select(
        cls,
        gpus: Iterable[Dict[str, Any] | GPUDevice],
        backend_capabilities: Dict[str, Any],
        requested_strategy: str = "auto",
        max_concurrent_workers: int = 1,
    ) -> GPUStrategyPlan:
        devices = [gpu if isinstance(gpu, GPUDevice) else GPUDevice(**_normalize_gpu_payload(gpu)) for gpu in gpus]
        devices.sort(key=lambda item: item.index)
        requested = cls.normalize_strategy(requested_strategy)
        primary = cls.choose_primary_gpu(devices)
        primary_idx = primary.index if primary else None
        warnings: List[str] = []
        notes: List[str] = []

        if not devices:
            return GPUStrategyPlan(
                requested_strategy=requested,
                effective_strategy="cpu_or_unknown",
                primary_gpu=None,
                worker_gpus=[],
                auxiliary_gpus=[],
                max_concurrent_workers=1,
                warnings=["No NVIDIA GPUs detected; backend may run on CPU or fail if CUDA is required."],
            )

        aux = [gpu.index for gpu in devices if primary_idx is None or gpu.index != primary_idx]
        supports_parallel = bool(
            backend_capabilities.get("supports_parallel_segment_processing")
            or backend_capabilities.get("supports_segment_batch_mode")
        )
        supports_multi = bool(backend_capabilities.get("supports_multi_gpu") or supports_parallel)
        supports_advanced = bool(backend_capabilities.get("supports_advanced_multigpu"))

        if requested == cls.ADVANCED:
            if supports_advanced:
                return GPUStrategyPlan(
                    requested_strategy=requested,
                    effective_strategy=cls.ADVANCED,
                    primary_gpu=primary_idx,
                    worker_gpus=[gpu.index for gpu in devices],
                    auxiliary_gpus=aux,
                    max_concurrent_workers=max(1, int(max_concurrent_workers)),
                    notes=["Advanced backend-managed multi-GPU placement is enabled by the selected backend."],
                    advanced_mode_enabled=True,
                )
            warnings.append("Advanced MultiGPU mode requested, but the selected backend does not advertise safe advanced placement support; falling back to Auto.")
            requested = cls.AUTO

        if requested == cls.AUTO:
            if supports_parallel and len(devices) > 1:
                effective = cls.PARALLEL_ALL
                workers = [gpu.index for gpu in devices]
                notes.append("Auto selected parallel segment scheduling because the backend supports segment batches.")
            elif len(devices) > 1:
                effective = cls.PRIMARY_AUX
                workers = [primary_idx] if primary_idx is not None else [devices[0].index]
                notes.append("Auto selected 3090-primary scheduling with auxiliary GPUs reserved for lighter stages.")
            else:
                effective = cls.PRIMARY_ONLY
                workers = [devices[0].index]
                notes.append("Auto selected single-GPU scheduling.")
        elif requested == cls.PARALLEL_ALL:
            if not supports_parallel:
                effective = cls.PRIMARY_ONLY
                workers = [primary_idx] if primary_idx is not None else [devices[0].index]
                warnings.append("Parallel segment processing requested, but the backend does not support it; using primary GPU only.")
            else:
                effective = cls.PARALLEL_ALL
                workers = [gpu.index for gpu in devices]
        elif requested == cls.PRIMARY_AUX:
            effective = cls.PRIMARY_AUX
            workers = [primary_idx] if primary_idx is not None else [devices[0].index]
        elif requested == cls.PRIMARY_ONLY:
            effective = cls.PRIMARY_ONLY
            workers = [primary_idx] if primary_idx is not None else [devices[0].index]
        else:
            effective = cls.AUTO
            workers = [primary_idx] if primary_idx is not None else [devices[0].index]

        if supports_multi is False and len(workers) > 1:
            warnings.append("Backend does not advertise multi-GPU support; limiting transform workers to the primary GPU.")
            workers = [primary_idx] if primary_idx is not None else [devices[0].index]
            effective = cls.PRIMARY_ONLY

        capped_workers = max(1, min(max(1, int(max_concurrent_workers)), len(workers))) if workers else 1
        return GPUStrategyPlan(
            requested_strategy=requested,
            effective_strategy=effective,
            primary_gpu=primary_idx,
            worker_gpus=workers,
            auxiliary_gpus=aux,
            max_concurrent_workers=capped_workers,
            warnings=warnings,
            notes=notes,
            advanced_mode_enabled=False,
        )


class SegmentQueueScheduler:
    """Assign segments to GPUs without pretending VRAM is pooled."""

    def __init__(self, strategy_plan: GPUStrategyPlan, gpus: Iterable[Dict[str, Any] | GPUDevice]) -> None:
        self.plan = strategy_plan
        self.gpus = [gpu if isinstance(gpu, GPUDevice) else GPUDevice(**_normalize_gpu_payload(gpu)) for gpu in gpus]
        self.gpu_by_index = {gpu.index: gpu for gpu in self.gpus}

    def assign_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not segments:
            return []
        worker_gpus = list(self.plan.worker_gpus)
        if not worker_gpus:
            return [dict(seg, gpu_assignment={"gpu_index": None, "role": "cpu_or_unknown"}) for seg in segments]

        assignments: List[Dict[str, Any]] = []
        if self.plan.effective_strategy == GPUStrategySelector.PARALLEL_ALL:
            ranked = sorted(worker_gpus, key=lambda idx: self.gpu_by_index.get(idx, GPUDevice(idx, "GPU", 0)).total_mb, reverse=True)
            for idx, segment in enumerate(segments):
                gpu_idx = ranked[idx % len(ranked)]
                gpu = self.gpu_by_index.get(gpu_idx)
                assignments.append(dict(
                    segment,
                    gpu_assignment={
                        "gpu_index": gpu_idx,
                        "gpu_name": gpu.name if gpu else f"GPU {gpu_idx}",
                        "role": "parallel_segment_transform",
                    },
                ))
            return assignments

        primary = self.plan.primary_gpu if self.plan.primary_gpu is not None else worker_gpus[0]
        primary_gpu = self.gpu_by_index.get(primary)
        for segment in segments:
            assignments.append(dict(
                segment,
                gpu_assignment={
                    "gpu_index": primary,
                    "gpu_name": primary_gpu.name if primary_gpu else f"GPU {primary}",
                    "role": "primary_transform",
                    "auxiliary_gpus": list(self.plan.auxiliary_gpus),
                },
            ))
        return assignments

    def stage_assignment_log(self, assigned_segments: List[Dict[str, Any]]) -> List[str]:
        lines = []
        for segment in assigned_segments:
            gpu = segment.get("gpu_assignment") or {}
            segment_id = segment.get("segment_id") or segment.get("id") or "segment"
            if gpu.get("gpu_index") is None:
                lines.append(f"Segment {segment_id} → CPU/unknown device")
            else:
                lines.append(f"Segment {segment_id} → GPU {gpu.get('gpu_index')} {gpu.get('gpu_name', '')}".strip())
        for gpu_idx in self.plan.auxiliary_gpus:
            gpu = self.gpu_by_index.get(gpu_idx)
            lines.append(f"Aux/QC tasks eligible → GPU {gpu_idx} {gpu.name if gpu else ''}".strip())
        return lines


class EdisonServiceController:
    """Best-effort controller for Edison-owned GPU services.

    It only touches Edison-controlled globals that are already loaded in the
    current Python process. External/unmanaged processes are reported through
    GPU telemetry but are not killed.
    """

    def snapshot(self) -> Dict[str, Any]:
        app_mod = _get_running_app_module()
        if app_mod is None:
            return {"available": False, "reason": "edison app module not loaded in this process"}
        return {
            "available": True,
            "llm_fast_loaded": getattr(app_mod, "llm_fast", None) is not None,
            "llm_medium_loaded": getattr(app_mod, "llm_medium", None) is not None,
            "llm_deep_loaded": getattr(app_mod, "llm_deep", None) is not None,
            "llm_reasoning_loaded": getattr(app_mod, "llm_reasoning", None) is not None,
            "vision_loaded": getattr(app_mod, "llm_vision", None) is not None,
            "vision_code_loaded": getattr(app_mod, "llm_vision_code", None) is not None,
            "video_pipe_loaded": getattr(getattr(app_mod, "video_service", None), "_pipe", None) is not None,
            "music_model_loaded": bool(getattr(getattr(app_mod, "music_service", None), "_model_loaded", False)),
        }

    def suspend_for_render(self) -> Dict[str, Any]:
        app_mod = _get_running_app_module()
        actions: List[str] = []
        errors: List[str] = []
        if app_mod is None:
            return {"ok": False, "actions": actions, "errors": ["Edison app module not loaded; nothing to suspend."]}
        try:
            unload = getattr(app_mod, "unload_all_llm_models", None)
            if callable(unload):
                unloaded = unload()
                actions.append(f"unloaded_llm_models:{','.join(unloaded) if unloaded else 'none'}")
        except Exception as exc:  # pragma: no cover - app-specific best effort
            errors.append(f"llm_unload_failed:{exc}")
        try:
            video_service = getattr(app_mod, "video_service", None)
            if video_service is not None and getattr(video_service, "_pipe", None) is not None:
                video_service._unload_pipeline()
                actions.append("unloaded_video_pipeline")
        except Exception as exc:  # pragma: no cover
            errors.append(f"video_unload_failed:{exc}")
        try:
            music_service = getattr(app_mod, "music_service", None)
            if music_service is not None and bool(getattr(music_service, "_model_loaded", False)):
                music_service._unload_model()
                actions.append("unloaded_music_model")
        except Exception as exc:  # pragma: no cover
            errors.append(f"music_unload_failed:{exc}")
        _flush_cuda_memory()
        return {"ok": not errors, "actions": actions, "errors": errors}

    def restore_after_render(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        app_mod = _get_running_app_module()
        actions: List[str] = []
        errors: List[str] = []
        if app_mod is None or not snapshot.get("available"):
            return {"ok": True, "actions": ["restore_skipped_no_prior_snapshot"], "errors": []}
        try:
            should_restore_text = any(snapshot.get(key) for key in [
                "llm_fast_loaded", "llm_medium_loaded", "llm_deep_loaded", "llm_reasoning_loaded"
            ])
            reload_fn = getattr(app_mod, "reload_llm_models_background", None)
            if should_restore_text and callable(reload_fn):
                reload_fn(include_vision=False, include_vision_code=False)
                actions.append("requested_text_model_restore")
        except Exception as exc:  # pragma: no cover
            errors.append(f"text_model_restore_failed:{exc}")
        return {"ok": not errors, "actions": actions, "errors": errors}


class ExclusiveGPURenderManager:
    """Coordinates exclusive render mode snapshots, unload, readiness wait, restore."""

    def __init__(self, service_controller: Optional[EdisonServiceController] = None) -> None:
        self.service_controller = service_controller or EdisonServiceController()

    def detect_gpus(self) -> List[GPUDevice]:
        try:
            from .gpu_config import detect_gpus
        except Exception:
            try:
                from services.edison_core.gpu_config import detect_gpus  # type: ignore
            except Exception:
                detect_gpus = _detect_gpus_without_repo_helpers
        devices = []
        for payload in detect_gpus():
            devices.append(GPUDevice(**_normalize_gpu_payload(payload)))
        return devices

    def snapshot_gpu_state(self) -> Dict[str, Any]:
        return {
            "timestamp": time.time(),
            "gpus": [gpu.to_dict() for gpu in self.detect_gpus()],
            "unmanaged_processes": self.query_gpu_processes(),
        }

    def enter_exclusive_mode(
        self,
        thresholds_min_free_mb: Optional[Dict[str, float]] = None,
        wait_timeout_s: float = 60.0,
        poll_interval_s: float = 2.0,
    ) -> Dict[str, Any]:
        pre = self.snapshot_gpu_state()
        service_snapshot = self.service_controller.snapshot()
        suspend = self.service_controller.suspend_for_render()
        post_unload = self.snapshot_gpu_state()
        readiness = self.wait_for_vram_ready(thresholds_min_free_mb or {}, wait_timeout_s, poll_interval_s)
        return {
            "enabled": True,
            "service_snapshot": service_snapshot,
            "pre_render_gpu_state": pre,
            "suspend_result": suspend,
            "post_unload_gpu_state": post_unload,
            "readiness": readiness,
        }

    def restore(self, exclusive_state: Dict[str, Any]) -> Dict[str, Any]:
        snapshot = (exclusive_state or {}).get("service_snapshot") or {}
        restore = self.service_controller.restore_after_render(snapshot)
        post_restore = self.snapshot_gpu_state()
        return {"restore_result": restore, "post_restore_gpu_state": post_restore}

    def wait_for_vram_ready(
        self,
        thresholds_min_free_mb: Dict[str, float],
        wait_timeout_s: float = 60.0,
        poll_interval_s: float = 2.0,
    ) -> Dict[str, Any]:
        deadline = time.time() + max(0.0, wait_timeout_s)
        samples: List[Dict[str, Any]] = []
        ready = False
        warnings: List[str] = []
        while True:
            state = self.snapshot_gpu_state()
            samples.append(state)
            gpus = state.get("gpus", [])
            ready = True
            for gpu in gpus:
                idx = str(gpu.get("index"))
                min_free = thresholds_min_free_mb.get(idx)
                if min_free is None:
                    min_free = thresholds_min_free_mb.get(gpu.get("name", ""), 0)
                if min_free and float(gpu.get("free_mb") or 0) < float(min_free):
                    ready = False
            if ready or time.time() >= deadline:
                break
            time.sleep(max(0.2, poll_interval_s))
        if not ready:
            warnings.append("VRAM readiness threshold not fully met before timeout; unmanaged/external processes may still hold memory.")
        if samples and samples[-1].get("unmanaged_processes"):
            warnings.append("Detected NVIDIA processes in nvidia-smi; Edison did not terminate unmanaged external processes.")
        return {
            "ready": ready,
            "thresholds_min_free_mb": thresholds_min_free_mb,
            "samples": samples[-5:],
            "warnings": warnings,
        }

    def query_gpu_processes(self) -> List[Dict[str, Any]]:
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            return []
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return []
            rows = []
            for line in result.stdout.strip().splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 4:
                    rows.append({
                        "gpu_uuid": parts[0],
                        "pid": parts[1],
                        "process_name": parts[2],
                        "used_memory_mb": _safe_float(parts[3]),
                    })
            return rows
        except Exception:
            return []


def _normalize_gpu_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    total_mb = payload.get("total_mb")
    if total_mb is None and payload.get("memory_total_gb") is not None:
        total_mb = float(payload.get("memory_total_gb")) * 1024
    used_mb = payload.get("used_mb")
    if used_mb is None and payload.get("memory_used_gb") is not None:
        used_mb = float(payload.get("memory_used_gb")) * 1024
    free_mb = payload.get("free_mb")
    if free_mb is None and total_mb is not None and used_mb is not None:
        free_mb = float(total_mb) - float(used_mb)
    return {
        "index": int(payload.get("index", 0)),
        "name": str(payload.get("name", f"GPU {payload.get('index', 0)}")),
        "total_mb": float(total_mb or 0),
        "free_mb": float(free_mb or 0),
        "used_mb": float(used_mb or 0),
        "utilization_percent": payload.get("utilization_percent"),
    }


def _detect_gpus_without_repo_helpers() -> List[Dict[str, Any]]:
    """Fallback detector that avoids importing repo modules or PyYAML."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
        gpus: List[Dict[str, Any]] = []
        for line in result.stdout.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "total_mb": _safe_float(parts[2]),
                    "used_mb": _safe_float(parts[3]),
                    "free_mb": _safe_float(parts[4]),
                    "utilization_percent": _safe_float(parts[5]) if len(parts) > 5 else None,
                })
        return gpus
    except Exception:
        return []


def _get_running_app_module() -> Any:
    for name in ("services.edison_core.app", "edison_core.app", "app"):
        mod = sys.modules.get(name)
        if mod is not None:
            return mod
    return None


def _flush_cuda_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()


def _safe_float(value: Any) -> float:
    try:
        return float(str(value).replace("MiB", "").strip())
    except Exception:
        return 0.0
