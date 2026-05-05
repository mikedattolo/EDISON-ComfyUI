"""
Scheduled-ops wrappers for long-running generation jobs.

Phase 2 goal: route image, video, music, and mesh jobs through the GPU
scheduler so they don't block chat. These are thin, *opt-in* wrappers —
existing callers can keep invoking the underlying ops directly; new code
paths and the unified jobs center should prefer the scheduled variants.

Each helper:

* picks a sensible lane,
* records lane metadata on the scheduler telemetry, and
* calls back through the existing :class:`JobStore` so the unified jobs
  center can surface progress.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)


# Map job_type → scheduler lane. Kept here (not in gpu_scheduler) so that
# the scheduler module itself stays free of business semantics.
JOB_TYPE_TO_LANE: Dict[str, str] = {
    "image": "image",
    "video": "video",
    "music": "music",
    "mesh": "mesh",
    "cad": "cad",
}

# Default priority overrides for special cases. Higher = scheduled first
# within the same lane semaphore.
JOB_TYPE_PRIORITY: Dict[str, int] = {
    "image": 60,    # bumps image above default lane priority of 50
    "video": 35,
    "music": 30,
    "mesh": 30,
    "cad": 30,
}


def lane_for(job_type: str) -> str:
    """Return the scheduler lane that should run a given job type."""
    return JOB_TYPE_TO_LANE.get(job_type, "background")


async def run_scheduled(
    job_type: str,
    coro_factory: Callable[[], Awaitable[Any]],
    *,
    job_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    priority: Optional[int] = None,
) -> Any:
    """Run ``coro_factory()`` on the lane that matches ``job_type``.

    The metadata dict is stored on the scheduler's job record so the
    unified jobs center can correlate scheduler entries back to the
    persistent :class:`JobStore` row identified by ``job_id``.
    """
    from .gpu_scheduler import get_scheduler

    sched = get_scheduler()
    lane = lane_for(job_type)
    meta = dict(metadata or {})
    if job_id:
        meta["job_id"] = job_id
    meta["job_type"] = job_type

    pri = priority if priority is not None else JOB_TYPE_PRIORITY.get(job_type)
    return await sched.run(lane, coro_factory, priority=pri, metadata=meta)
