"""
Unified Jobs Center.

Phase 2 goal: a single backend surface that aggregates *all* in-flight and
historical work across chat, image, video, music, mesh, CAD, printing, and
connectors. This module is read-only — it doesn't own any data, it just
joins the existing :class:`JobStore` (persistent generation jobs) with
the live :class:`GPUScheduler` telemetry (in-flight scheduler entries).

The result is the data shape the front-end's "global jobs center" needs:
each job carries its kind, status, lane, queue position, and links to
outputs.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _safe_job_store():
    try:
        from .job_store import JobStore
        return JobStore.get_instance()
    except Exception as exc:  # pragma: no cover
        logger.debug("JobStore unavailable: %s", exc)
        return None


def _safe_scheduler():
    try:
        from .gpu_scheduler import get_scheduler
        return get_scheduler()
    except Exception as exc:  # pragma: no cover
        logger.debug("Scheduler unavailable: %s", exc)
        return None


def list_jobs(
    *,
    status: Optional[str] = None,
    job_type: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Return persistent jobs filtered by status/type, decorated with any
    matching live scheduler entry (queue position, lane, wait_ms).
    """
    store = _safe_job_store()
    if store is None:
        return []

    rows = store.list_jobs(job_type=job_type, status=status, limit=limit)

    sched = _safe_scheduler()
    if sched is None:
        return rows

    # Build a lookup of live scheduler entries keyed by metadata.job_id
    live: Dict[str, Dict[str, Any]] = {}
    snapshot = sched.telemetry()
    for rec in snapshot.get("active", []) + snapshot.get("history", []):
        meta = rec.get("metadata") or {}
        jid = meta.get("job_id")
        if jid:
            live[jid] = rec

    for row in rows:
        match = live.get(row["job_id"])
        if match:
            row["scheduler"] = {
                "lane": match.get("lane"),
                "status": match.get("status"),
                "wait_ms": match.get("wait_ms"),
                "duration_ms": match.get("duration_ms"),
                "priority": match.get("priority"),
            }
    return rows


def summary() -> Dict[str, Any]:
    """High-level dashboard summary: counts by status/type and lane stats."""
    store = _safe_job_store()
    sched = _safe_scheduler()

    counts_by_status: Dict[str, int] = {}
    counts_by_type: Dict[str, int] = {}
    recent: List[Dict[str, Any]] = []
    if store is not None:
        recent = store.list_jobs(limit=100)
        for row in recent:
            counts_by_status[row["status"]] = counts_by_status.get(row["status"], 0) + 1
            counts_by_type[row["job_type"]] = counts_by_type.get(row["job_type"], 0) + 1

    lanes: Dict[str, Any] = {}
    if sched is not None:
        for lane in sched.lanes():
            lanes[lane] = sched.lane_info(lane)

    return {
        "now": time.time(),
        "counts_by_status": counts_by_status,
        "counts_by_type": counts_by_type,
        "lanes": lanes,
        "recent_count": len(recent),
    }


def cancel(job_id: str) -> Dict[str, Any]:
    """Best-effort cancel via the JobStore. The scheduler does not track
    cancellation directly because cooperative cancel is the producer's
    responsibility.
    """
    store = _safe_job_store()
    if store is None:
        return {"ok": False, "reason": "job store unavailable"}
    ok = store.cancel_job(job_id)
    return {"ok": ok, "job_id": job_id}
