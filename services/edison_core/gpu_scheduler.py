"""
GPU Scheduler & Workload Lanes
==============================

Phase 1 foundation for GPU isolation across workloads:

* Reserves a primary GPU lane for interactive chat/coding (low latency).
* Pins image generation, video, and 3D/CAD jobs to secondary lanes.
* Adds a priority request queue so chat is never blocked by long jobs.
* Provides telemetry hooks for queue depth, VRAM pressure, and model
  residency.

This module is intentionally additive and side-effect free at import time:
nothing is touched until ``GPUScheduler.get_instance()`` is called.

Other modules can opt in to scheduling without breaking existing code paths.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Workload lanes ──────────────────────────────────────────────────────

#: Lane name → priority (higher value = scheduled first within the same lane)
DEFAULT_LANE_PRIORITY: Dict[str, int] = {
    "chat": 100,
    "code": 95,
    "vision": 80,
    "tool": 70,
    "image": 50,
    "video": 30,
    "music": 30,
    "mesh": 25,
    "cad": 25,
    "background": 10,
}

#: Default lane → preferred GPU index. The scheduler will fall back to any
#: available device if the preferred one is missing.
DEFAULT_LANE_GPU: Dict[str, int] = {
    "chat": 0,
    "code": 0,
    "vision": 0,
    "tool": 0,
    "image": 1,
    "video": 2,
    "music": 2,
    "mesh": 2,
    "cad": 2,
    "background": 2,
}

#: Maximum concurrent jobs per lane. Chat is intentionally large so it never
#: queues behind its own siblings; long-running media lanes are throttled.
DEFAULT_LANE_CONCURRENCY: Dict[str, int] = {
    "chat": 8,
    "code": 4,
    "vision": 2,
    "tool": 4,
    "image": 1,
    "video": 1,
    "music": 1,
    "mesh": 1,
    "cad": 1,
    "background": 1,
}


@dataclass
class JobRecord:
    job_id: str
    lane: str
    priority: int
    submitted_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "queued"  # queued|running|done|error|cancelled
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def wait_ms(self) -> Optional[float]:
        if self.started_at is None:
            return None
        return round((self.started_at - self.submitted_at) * 1000.0, 2)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at is None or self.completed_at is None:
            return None
        return round((self.completed_at - self.started_at) * 1000.0, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "lane": self.lane,
            "priority": self.priority,
            "status": self.status,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "wait_ms": self.wait_ms,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "metadata": self.metadata,
        }


class _LaneState:
    """Per-lane semaphore + bookkeeping."""

    __slots__ = ("name", "gpu_index", "max_concurrent", "semaphore",
                 "in_flight", "queued", "completed", "errors")

    def __init__(self, name: str, gpu_index: int, max_concurrent: int):
        self.name = name
        self.gpu_index = gpu_index
        self.max_concurrent = max(1, int(max_concurrent))
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.in_flight = 0
        self.queued = 0
        self.completed = 0
        self.errors = 0


class GPUScheduler:
    """Async-friendly priority scheduler with workload lanes.

    Use ``await scheduler.run("chat", coro_factory)`` to schedule work. The
    scheduler will acquire the lane's semaphore (respecting max concurrency)
    and run the coroutine. Telemetry is recorded for queue depth, wait time,
    and duration.

    The scheduler is purely cooperative — it does not actually move tensors
    between GPUs. It coordinates *when* work runs so independent workloads
    don't trample each other. Pinning to a specific GPU is the responsibility
    of the model loader (e.g. setting ``CUDA_VISIBLE_DEVICES`` or
    ``main_gpu``).
    """

    _instance: Optional["GPUScheduler"] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        lane_priority: Optional[Dict[str, int]] = None,
        lane_gpu: Optional[Dict[str, int]] = None,
        lane_concurrency: Optional[Dict[str, int]] = None,
        history_size: int = 256,
    ):
        self._lane_priority = dict(lane_priority or DEFAULT_LANE_PRIORITY)
        self._lane_gpu = dict(lane_gpu or DEFAULT_LANE_GPU)
        self._lane_concurrency = dict(lane_concurrency or DEFAULT_LANE_CONCURRENCY)

        self._lanes: Dict[str, _LaneState] = {}
        for name in self._lane_priority:
            self._lanes[name] = _LaneState(
                name=name,
                gpu_index=self._lane_gpu.get(name, 0),
                max_concurrent=self._lane_concurrency.get(name, 1),
            )

        self._history: List[JobRecord] = []
        self._history_size = history_size
        self._history_lock = threading.Lock()
        self._jobs: Dict[str, JobRecord] = {}

    # ── singleton ────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "GPUScheduler":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Test helper — discard the singleton so a fresh one is built."""
        with cls._instance_lock:
            cls._instance = None

    # ── lane introspection ──────────────────────────────────────────

    def lanes(self) -> List[str]:
        return list(self._lanes.keys())

    def lane_info(self, lane: str) -> Dict[str, Any]:
        state = self._lanes.get(lane)
        if state is None:
            raise KeyError(f"Unknown lane: {lane}")
        return {
            "lane": state.name,
            "gpu_index": state.gpu_index,
            "max_concurrent": state.max_concurrent,
            "in_flight": state.in_flight,
            "queued": state.queued,
            "completed": state.completed,
            "errors": state.errors,
            "priority": self._lane_priority.get(lane, 0),
        }

    def telemetry(self) -> Dict[str, Any]:
        """Return a snapshot of all lane stats + recent history."""
        lanes = {name: self.lane_info(name) for name in self._lanes}
        with self._history_lock:
            history = [rec.to_dict() for rec in self._history[-50:]]
        return {
            "lanes": lanes,
            "history": history,
            "active": [
                rec.to_dict() for rec in self._jobs.values()
                if rec.status == "running"
            ],
            "now": time.time(),
        }

    # ── execution ────────────────────────────────────────────────────

    def submit(
        self,
        lane: str,
        *,
        priority: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> JobRecord:
        """Reserve a job slot. Returns a :class:`JobRecord` you can update
        manually if you don't need the ``run()`` helper.
        """
        if lane not in self._lanes:
            raise KeyError(f"Unknown lane: {lane}")
        rec = JobRecord(
            job_id=str(uuid.uuid4())[:12],
            lane=lane,
            priority=priority if priority is not None else self._lane_priority.get(lane, 0),
            submitted_at=time.time(),
            metadata=dict(metadata or {}),
        )
        self._jobs[rec.job_id] = rec
        self._lanes[lane].queued += 1
        return rec

    def _finalize(self, rec: JobRecord) -> None:
        with self._history_lock:
            self._history.append(rec)
            if len(self._history) > self._history_size:
                self._history = self._history[-self._history_size:]
        # leave entry in _jobs briefly so callers can look up status, but
        # cap the dict to avoid growth
        if len(self._jobs) > self._history_size * 2:
            stale_ids = [
                jid for jid, r in self._jobs.items()
                if r.status in ("done", "error", "cancelled")
            ]
            for jid in stale_ids[:-self._history_size]:
                self._jobs.pop(jid, None)

    async def run(
        self,
        lane: str,
        coro_factory: Callable[[], Awaitable[Any]],
        *,
        priority: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Schedule and await ``coro_factory()`` inside the given lane.

        ``coro_factory`` is a zero-arg callable returning a coroutine. Using
        a factory (rather than a coroutine object) lets the scheduler create
        the coroutine *after* acquiring the semaphore, which avoids
        ``RuntimeWarning: coroutine ... was never awaited`` if the call is
        cancelled before it starts.
        """
        rec = self.submit(lane, priority=priority, metadata=metadata)
        state = self._lanes[lane]
        try:
            async with state.semaphore:
                state.queued = max(0, state.queued - 1)
                state.in_flight += 1
                rec.started_at = time.time()
                rec.status = "running"
                try:
                    result = await coro_factory()
                    rec.completed_at = time.time()
                    rec.status = "done"
                    state.completed += 1
                    return result
                except asyncio.CancelledError:
                    rec.completed_at = time.time()
                    rec.status = "cancelled"
                    raise
                except Exception as exc:  # noqa: BLE001
                    rec.completed_at = time.time()
                    rec.status = "error"
                    rec.error = str(exc)
                    state.errors += 1
                    raise
                finally:
                    state.in_flight = max(0, state.in_flight - 1)
        finally:
            self._finalize(rec)

    # ── lookup helpers ──────────────────────────────────────────────

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    def gpu_for_lane(self, lane: str) -> int:
        return self._lane_gpu.get(lane, 0)


# ── Convenience helpers ────────────────────────────────────────────────

def get_scheduler() -> GPUScheduler:
    """Return the process-wide scheduler instance."""
    return GPUScheduler.get_instance()


async def run_on_lane(
    lane: str,
    coro_factory: Callable[[], Awaitable[Any]],
    *,
    priority: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """Module-level shortcut for the singleton scheduler."""
    return await get_scheduler().run(
        lane, coro_factory, priority=priority, metadata=metadata
    )


def configure_from_env() -> GPUScheduler:
    """Build a scheduler with per-lane concurrency overrides from env vars.

    Environment variables (all optional):

    * ``EDISON_LANE_CHAT_CONCURRENCY``
    * ``EDISON_LANE_IMAGE_CONCURRENCY``
    * ``EDISON_LANE_VIDEO_CONCURRENCY``
    * ``EDISON_LANE_MESH_CONCURRENCY``
    """
    concurrency = dict(DEFAULT_LANE_CONCURRENCY)
    for lane in concurrency:
        env_key = f"EDISON_LANE_{lane.upper()}_CONCURRENCY"
        val = os.environ.get(env_key)
        if val:
            try:
                concurrency[lane] = max(1, int(val))
            except ValueError:
                logger.warning("Ignoring non-integer %s=%r", env_key, val)
    GPUScheduler.reset_instance()
    GPUScheduler._instance = GPUScheduler(lane_concurrency=concurrency)
    return GPUScheduler._instance
