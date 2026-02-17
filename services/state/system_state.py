"""
System State Awareness for Edison.

Tracks running jobs, recent errors, GPU utilization, ComfyUI queue,
and disk usage so Edison can reference system health in responses.
"""

import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()


# ── Data model ───────────────────────────────────────────────────────────

@dataclass
class GPUInfo:
    id: int
    name: str
    total_mb: int
    used_mb: int
    free_mb: int
    utilization_pct: float
    temperature_c: Optional[int] = None


@dataclass
class DiskInfo:
    path: str
    total_gb: float
    used_gb: float
    free_gb: float
    usage_pct: float


@dataclass
class SystemSnapshot:
    """Point-in-time snapshot of system state."""
    timestamp: float = field(default_factory=time.time)
    gpus: List[GPUInfo] = field(default_factory=list)
    disks: List[DiskInfo] = field(default_factory=list)
    running_jobs: int = 0
    queued_jobs: int = 0
    recent_errors: List[Dict[str, Any]] = field(default_factory=list)
    comfyui_reachable: bool = False
    comfyui_queue_size: int = 0
    models_loaded: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def to_context_string(self) -> str:
        """Compact summary for LLM context injection."""
        parts = []
        if self.gpus:
            gpu_strs = [
                f"GPU{g.id}: {g.free_mb}MB free/{g.total_mb}MB, {g.utilization_pct:.0f}% util"
                for g in self.gpus
            ]
            parts.append("GPUs: " + " | ".join(gpu_strs))
        if self.running_jobs or self.queued_jobs:
            parts.append(f"Jobs: {self.running_jobs} running, {self.queued_jobs} queued")
        if self.disks:
            for d in self.disks:
                parts.append(f"Disk({d.path}): {d.free_gb:.1f}GB free ({d.usage_pct:.0f}% used)")
        if self.recent_errors:
            parts.append(f"Recent errors: {len(self.recent_errors)}")
        if self.comfyui_reachable:
            parts.append(f"ComfyUI: online (queue={self.comfyui_queue_size})")
        if self.models_loaded:
            parts.append(f"Models loaded: {', '.join(self.models_loaded)}")
        return "; ".join(parts) if parts else "System state unknown"


# ── Collectors ───────────────────────────────────────────────────────────

def _collect_gpu_info() -> List[GPUInfo]:
    """Probe NVIDIA GPUs via pynvml or nvidia-smi fallback."""
    gpus: List[GPUInfo] = []

    # Try pynvml first
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except Exception:
                gpu_util = 0.0
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None
            gpus.append(GPUInfo(
                id=i,
                name=name,
                total_mb=int(mem.total / 1024 / 1024),
                used_mb=int(mem.used / 1024 / 1024),
                free_mb=int(mem.free / 1024 / 1024),
                utilization_pct=float(gpu_util),
                temperature_c=temp,
            ))
        pynvml.nvmlShutdown()
        return gpus
    except Exception:
        pass

    # Fallback: PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                free, total = torch.cuda.mem_get_info(i)
                used = total - free
                gpus.append(GPUInfo(
                    id=i,
                    name=name,
                    total_mb=int(total / 1024 / 1024),
                    used_mb=int(used / 1024 / 1024),
                    free_mb=int(free / 1024 / 1024),
                    utilization_pct=0.0,  # not available via torch
                ))
    except Exception:
        pass

    return gpus


def _collect_disk_info() -> List[DiskInfo]:
    """Check disk usage for key directories."""
    paths_to_check = [
        str(REPO_ROOT / "models"),
        str(REPO_ROOT / "outputs"),
        str(REPO_ROOT),
    ]
    disks: List[DiskInfo] = []
    seen_devices = set()
    for path in paths_to_check:
        if not Path(path).exists():
            continue
        try:
            usage = shutil.disk_usage(path)
            # Deduplicate by device (same filesystem)
            dev_key = (usage.total, path.split("/")[1] if "/" in path else path)
            if dev_key in seen_devices:
                continue
            seen_devices.add(dev_key)
            disks.append(DiskInfo(
                path=path,
                total_gb=round(usage.total / (1024**3), 1),
                used_gb=round(usage.used / (1024**3), 1),
                free_gb=round(usage.free / (1024**3), 1),
                usage_pct=round(usage.used / usage.total * 100, 1) if usage.total else 0,
            ))
        except Exception:
            pass
    return disks


def _check_comfyui(comfyui_url: str = "http://127.0.0.1:8188") -> Dict[str, Any]:
    """Check if ComfyUI is reachable and get queue size."""
    try:
        import urllib.request
        import json as _json
        req = urllib.request.Request(f"{comfyui_url}/queue", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = _json.loads(resp.read())
            running = len(data.get("queue_running", []))
            pending = len(data.get("queue_pending", []))
            return {"reachable": True, "queue_size": running + pending}
    except Exception:
        return {"reachable": False, "queue_size": 0}


def _get_loaded_models() -> List[str]:
    """Return names of currently loaded LLM models."""
    models = []
    try:
        # Import from app module globals
        from services.edison_core import app as _app
        for attr_name in ("llm_fast", "llm_medium", "llm_deep", "llm_reasoning", "llm_vision"):
            obj = getattr(_app, attr_name, None)
            if obj is not None:
                models.append(attr_name.replace("llm_", ""))
    except Exception:
        pass
    return models


def _get_job_counts() -> Dict[str, int]:
    """Query the unified job store for running/queued counts."""
    try:
        from services.edison_core.job_store import JobStore
        store = JobStore.get_instance()
        running = len(store.list_jobs(status="generating", limit=100))
        queued = len(store.list_jobs(status="queued", limit=100))
        return {"running": running, "queued": queued}
    except Exception:
        return {"running": 0, "queued": 0}


# ── Main API ─────────────────────────────────────────────────────────────

# Cache the snapshot to avoid hammering GPU/disk on every request
_cached_snapshot: Optional[SystemSnapshot] = None
_cache_lock = threading.Lock()
_CACHE_TTL = 10  # seconds


def get_system_state(force_refresh: bool = False) -> SystemSnapshot:
    """Collect and return a full system state snapshot.

    Results are cached for _CACHE_TTL seconds to avoid probe overhead.
    """
    global _cached_snapshot
    now = time.time()

    if not force_refresh and _cached_snapshot and (now - _cached_snapshot.timestamp) < _CACHE_TTL:
        return _cached_snapshot

    with _cache_lock:
        # Double-check after lock
        if not force_refresh and _cached_snapshot and (now - _cached_snapshot.timestamp) < _CACHE_TTL:
            return _cached_snapshot

        gpus = _collect_gpu_info()
        disks = _collect_disk_info()
        comfyui = _check_comfyui()
        jobs = _get_job_counts()
        models = _get_loaded_models()

        snapshot = SystemSnapshot(
            gpus=[GPUInfo(**asdict(g)) if isinstance(g, GPUInfo) else g for g in gpus],
            disks=disks,
            running_jobs=jobs["running"],
            queued_jobs=jobs["queued"],
            comfyui_reachable=comfyui["reachable"],
            comfyui_queue_size=comfyui["queue_size"],
            models_loaded=models,
        )

        _cached_snapshot = snapshot
        return snapshot


# ── Error tracking ───────────────────────────────────────────────────────

_recent_errors: List[Dict[str, Any]] = []
_errors_lock = threading.Lock()
_MAX_ERRORS = 50


def record_system_error(error_type: str, message: str, source: str = ""):
    """Record a system-level error for the awareness layer."""
    with _errors_lock:
        _recent_errors.append({
            "type": error_type,
            "message": message[:500],
            "source": source,
            "timestamp": time.time(),
        })
        if len(_recent_errors) > _MAX_ERRORS:
            del _recent_errors[:len(_recent_errors) - _MAX_ERRORS]


def get_recent_errors(limit: int = 10) -> List[Dict[str, Any]]:
    """Return the most recent system errors."""
    with _errors_lock:
        return list(_recent_errors[-limit:])
