"""GPU-aware ComfyUI worker registry and routing helpers.

ComfyUI does not pool heterogeneous GPU VRAM into one memory space. Edison gets
better throughput by running one ComfyUI process per GPU and dispatching jobs to
the worker that best matches the job's VRAM estimate, queue depth, and requested
GPU assignment.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import requests


QueueProbe = Callable[["ComfyUIWorker"], Dict[str, Any]]


def normalize_base_url(value: Optional[str] = None, *, host: str = "127.0.0.1", port: int = 8188) -> str:
    """Normalize a ComfyUI URL for client-side Edison requests."""

    raw = str(value or "").strip()
    if not raw:
        raw = f"http://{host}:{port}"
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urllib.parse.urlparse(raw)
    scheme = (parsed.scheme or "http").lower()
    hostname = parsed.hostname or "127.0.0.1"
    if hostname in {"0.0.0.0", "::"}:
        hostname = "127.0.0.1"
    resolved_port = parsed.port or int(port or 8188)
    return f"{scheme}://{hostname}:{resolved_port}"


@dataclass(frozen=True)
class ComfyUIWorker:
    id: str
    base_url: str
    display_name: str = ""
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8188
    gpu_index: Optional[int] = None
    gpu_uuid: str = ""
    gpu_name: str = ""
    cuda_visible_devices: str = ""
    role: str = "worker"
    total_vram_mb: Optional[int] = None
    usable_vram_mb: Optional[int] = None
    max_parallel_jobs: int = 1
    preferred_for: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_config(cls, payload: Dict[str, Any], *, default_host: str, default_port: int) -> "ComfyUIWorker":
        host = str(payload.get("host") or default_host or "127.0.0.1")
        port = int(payload.get("port") or default_port or 8188)
        base_url = normalize_base_url(payload.get("base_url"), host=host, port=port)
        worker_id = str(payload.get("id") or payload.get("name") or f"comfyui-{port}").strip()
        return cls(
            id=worker_id,
            base_url=base_url,
            display_name=str(payload.get("display_name") or payload.get("label") or worker_id),
            enabled=bool(payload.get("enabled", True)),
            host=host,
            port=port,
            gpu_index=_maybe_int(payload.get("gpu_index")),
            gpu_uuid=str(payload.get("gpu_uuid") or payload.get("uuid") or "").strip(),
            gpu_name=str(payload.get("gpu_name") or payload.get("name") or "").strip(),
            cuda_visible_devices=str(payload.get("cuda_visible_devices") or payload.get("gpu_uuid") or "").strip(),
            role=str(payload.get("role") or "worker").strip().lower(),
            total_vram_mb=_maybe_int(payload.get("total_vram_mb")),
            usable_vram_mb=_maybe_int(payload.get("usable_vram_mb") or payload.get("max_vram_mb")),
            max_parallel_jobs=max(1, int(payload.get("max_parallel_jobs") or 1)),
            preferred_for=[str(item).strip().lower() for item in (payload.get("preferred_for") or []) if str(item).strip()],
            tags=[str(item).strip().lower() for item in (payload.get("tags") or []) if str(item).strip()],
            notes=str(payload.get("notes") or ""),
        )

    @classmethod
    def legacy(cls, *, host: str, port: int) -> "ComfyUIWorker":
        base_url = normalize_base_url(None, host=host, port=port)
        return cls(
            id="default",
            base_url=base_url,
            display_name="Default ComfyUI",
            host=host,
            port=port,
            role="primary",
            preferred_for=["image", "mesh", "video", "persona_video"],
        )

    def accepts_job(self, job_type: str) -> bool:
        normalized = _normalize_job_type(job_type)
        return not self.preferred_for or normalized in self.preferred_for or "all" in self.preferred_for

    def has_capacity_for(self, estimated_vram_mb: Optional[int]) -> bool:
        if estimated_vram_mb is None:
            return True
        limit = self.usable_vram_mb or self.total_vram_mb
        return not limit or int(estimated_vram_mb) <= int(limit)

    def matches_gpu_assignment(self, assignment: Optional[Dict[str, Any]]) -> bool:
        if not assignment:
            return False
        assigned_uuid = str(assignment.get("gpu_uuid") or assignment.get("uuid") or "").strip()
        if assigned_uuid and self.gpu_uuid and assigned_uuid == self.gpu_uuid:
            return True
        assigned_index = _maybe_int(assignment.get("gpu_index"))
        return assigned_index is not None and self.gpu_index is not None and assigned_index == self.gpu_index

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkerSelection:
    worker: ComfyUIWorker
    reason: str
    estimated_vram_mb: Optional[int] = None
    queue: Dict[str, Any] = field(default_factory=dict)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    requested_worker: str = ""
    requested_gpu: Dict[str, Any] = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        return self.worker.base_url

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker": self.worker.to_dict(),
            "base_url": self.worker.base_url,
            "reason": self.reason,
            "estimated_vram_mb": self.estimated_vram_mb,
            "queue": self.queue,
            "rejected": self.rejected,
            "requested_worker": self.requested_worker,
            "requested_gpu": self.requested_gpu,
        }


class ComfyUIWorkerRegistry:
    def __init__(
        self,
        workers: Iterable[ComfyUIWorker],
        *,
        workers_enabled: bool = False,
        selection_strategy: str = "least_busy_weighted",
        queue_probe: Optional[QueueProbe] = None,
    ) -> None:
        self.workers = [worker for worker in workers]
        self.workers_enabled = bool(workers_enabled)
        self.selection_strategy = str(selection_strategy or "least_busy_weighted")
        self.queue_probe = queue_probe or probe_worker_queue

    @classmethod
    def from_config(cls, config: Dict[str, Any], *, queue_probe: Optional[QueueProbe] = None) -> "ComfyUIWorkerRegistry":
        comfy = _extract_comfyui_config(config)
        host = str(comfy.get("host") or "127.0.0.1")
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        port = int(comfy.get("port") or 8188)
        worker_rows = comfy.get("workers") or []
        workers_enabled = bool(comfy.get("workers_enabled", False)) and bool(worker_rows)
        if workers_enabled:
            workers = [ComfyUIWorker.from_config(row, default_host=host, default_port=port) for row in worker_rows]
        else:
            workers = [ComfyUIWorker.legacy(host=host, port=port)]
        return cls(
            workers,
            workers_enabled=workers_enabled,
            selection_strategy=str(comfy.get("worker_selection_strategy") or "least_busy_weighted"),
            queue_probe=queue_probe,
        )

    def enabled_workers(self) -> List[ComfyUIWorker]:
        return [worker for worker in self.workers if worker.enabled]

    def select(
        self,
        *,
        job_type: str = "image",
        params: Optional[Dict[str, Any]] = None,
        estimated_vram_mb: Optional[int] = None,
        requested_worker: Optional[str] = None,
        gpu_assignment: Optional[Dict[str, Any]] = None,
        require_reachable: bool = False,
    ) -> WorkerSelection:
        estimate = estimated_vram_mb
        if estimate is None:
            estimate = estimate_comfyui_vram_mb(job_type, params or {})
        workers = self.enabled_workers()
        if not workers:
            raise RuntimeError("No enabled ComfyUI workers are configured.")

        rejected: List[Dict[str, Any]] = []
        requested = str(requested_worker or "").strip()
        if requested:
            match = next((worker for worker in workers if worker.id == requested), None)
            if not match:
                rejected.append({"worker_id": requested, "reason": "requested worker is not configured or enabled"})
            elif not match.has_capacity_for(estimate):
                rejected.append({"worker_id": match.id, "reason": "requested worker does not meet VRAM estimate"})
            else:
                queue = self._safe_probe(match)
                if require_reachable and not queue.get("reachable", False):
                    rejected.append({"worker_id": match.id, "reason": "requested worker is not reachable", "queue": queue})
                else:
                    return WorkerSelection(match, "requested_worker", estimate, queue, rejected, requested, gpu_assignment or {})

        if gpu_assignment:
            for worker in workers:
                if not worker.matches_gpu_assignment(gpu_assignment):
                    continue
                if not worker.has_capacity_for(estimate):
                    rejected.append({"worker_id": worker.id, "reason": "assigned GPU worker does not meet VRAM estimate"})
                    continue
                queue = self._safe_probe(worker)
                if require_reachable and not queue.get("reachable", False):
                    rejected.append({"worker_id": worker.id, "reason": "assigned GPU worker is not reachable", "queue": queue})
                    continue
                return WorkerSelection(worker, "gpu_assignment_match", estimate, queue, rejected, requested, gpu_assignment)

        candidates: List[tuple[ComfyUIWorker, Dict[str, Any]]] = []
        normalized_type = _normalize_job_type(job_type)
        for worker in workers:
            if not worker.accepts_job(normalized_type):
                rejected.append({"worker_id": worker.id, "reason": f"worker is not preferred for {normalized_type}"})
                continue
            if not worker.has_capacity_for(estimate):
                rejected.append({"worker_id": worker.id, "reason": f"estimated VRAM {estimate} MB exceeds worker limit"})
                continue
            queue = self._safe_probe(worker)
            if require_reachable and not queue.get("reachable", False):
                rejected.append({"worker_id": worker.id, "reason": "worker is not reachable", "queue": queue})
                continue
            candidates.append((worker, queue))

        if not candidates:
            fallback = workers[0]
            queue = self._safe_probe(fallback)
            return WorkerSelection(fallback, "fallback_no_compatible_worker", estimate, queue, rejected, requested, gpu_assignment or {})

        reachable = [item for item in candidates if item[1].get("reachable", False)]
        ranked = reachable or candidates
        ranked.sort(key=lambda item: _worker_score(item[0], item[1]))
        reason = "least_busy_reachable_worker" if reachable else "least_busy_configured_worker"
        return WorkerSelection(ranked[0][0], reason, estimate, ranked[0][1], rejected, requested, gpu_assignment or {})

    def health(self) -> Dict[str, Any]:
        rows = []
        reachable = 0
        for worker in self.enabled_workers():
            queue = self._safe_probe(worker)
            if queue.get("reachable"):
                reachable += 1
            rows.append({**worker.to_dict(), "queue": queue})
        return {
            "workers_enabled": self.workers_enabled,
            "selection_strategy": self.selection_strategy,
            "worker_count": len(rows),
            "reachable_count": reachable,
            "workers": rows,
            "note": "Workers are separate ComfyUI processes/GPU VRAM pools; Edison does not pool VRAM across cards.",
        }

    def _safe_probe(self, worker: ComfyUIWorker) -> Dict[str, Any]:
        try:
            return self.queue_probe(worker)
        except Exception as exc:
            return {"reachable": False, "running": 0, "pending": 0, "idle": False, "error": str(exc)}


def probe_worker_queue(worker: ComfyUIWorker, timeout_s: float = 1.5) -> Dict[str, Any]:
    response = requests.get(f"{worker.base_url}/queue", timeout=timeout_s)
    if not response.ok:
        return {"reachable": False, "status_code": response.status_code, "running": 0, "pending": 0, "idle": False}
    data = response.json()
    running = len(data.get("queue_running", []) or [])
    pending = len(data.get("queue_pending", []) or [])
    return {
        "reachable": True,
        "status_code": response.status_code,
        "running": running,
        "pending": pending,
        "idle": (running + pending) == 0,
    }


def estimate_comfyui_vram_mb(job_type: str, params: Dict[str, Any]) -> int:
    normalized = _normalize_job_type(job_type)
    if normalized == "persona_video":
        return int(params.get("estimated_vram_mb") or 12000)
    if normalized == "mesh":
        return int(params.get("estimated_vram_mb") or 10000)
    if normalized == "video":
        return int(params.get("estimated_vram_mb") or 18000)
    width = max(64, int(params.get("width") or 1024))
    height = max(64, int(params.get("height") or 1024))
    steps = max(1, int(params.get("steps") or 20))
    megapixels = (width * height) / 1_000_000
    base = 6000 + int(megapixels * 1800) + min(2200, steps * 35)
    if str(params.get("style_preset") or "").lower() == "logo":
        base += 800
    if params.get("hires_fix") or params.get("refine"):
        base += 2500
    return max(6000, min(base, 24000))


def _extract_comfyui_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if "workers" in config or "workers_enabled" in config or "host" in config or "port" in config:
        return dict(config)
    edison = config.get("edison", config)
    return dict((edison or {}).get("comfyui") or {})


def _normalize_job_type(value: Any) -> str:
    text = str(value or "image").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"image_generation", "generate_image", "img2img", "inpaint", "texture"}:
        return "image"
    if text in {"persona", "persona_segment", "persona_video_segment"}:
        return "persona_video"
    return text or "image"


def _worker_score(worker: ComfyUIWorker, queue: Dict[str, Any]) -> tuple[float, int, int]:
    running = int(queue.get("running") or 0)
    pending = int(queue.get("pending") or 0)
    load = (running + pending) / max(1, worker.max_parallel_jobs)
    primary_penalty = 0 if worker.role == "primary" else 1
    vram_rank = -(worker.usable_vram_mb or worker.total_vram_mb or 0)
    return (load, primary_penalty, vram_rank)


def _maybe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(str(value).strip()))
    except Exception:
        return None
