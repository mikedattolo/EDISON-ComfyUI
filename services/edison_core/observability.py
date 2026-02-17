"""
Observability Module for EDISON
Structured logging and tracing with per-request correlation IDs.
Tracks: retrieval decisions, memory saves, tool calls, generation steps.
"""

import json
import logging
import threading
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Per-request correlation ID using contextvars (async-safe)
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def new_correlation_id() -> str:
    """Generate and set a new correlation ID for the current request."""
    cid = str(uuid.uuid4())[:12]
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return _correlation_id.get() or "none"


class StructuredEvent:
    """A structured observability event."""

    def __init__(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ):
        self.event_type = event_type
        self.data = data or {}
        self.correlation_id = correlation_id or get_correlation_id()
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "data": self.data,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class ObservabilityTracer:
    """Central tracing system for Edison operations."""

    def __init__(self, max_events: int = 10000):
        self._events: List[StructuredEvent] = []
        self._lock = threading.Lock()
        self._max_events = max_events

    def trace(self, event_type: str, **data) -> StructuredEvent:
        """Record a structured event."""
        event = StructuredEvent(event_type=event_type, data=data)
        with self._lock:
            self._events.append(event)
            # Trim if too many
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events // 2:]

        logger.debug(f"[{event.correlation_id}] {event_type}: {json.dumps(data, default=str)[:200]}")
        return event

    def trace_retrieval(
        self,
        query: str,
        results_count: int,
        rerank_scores: Optional[List[float]] = None,
        intent: str = "",
        **extra,
    ):
        """Trace a retrieval decision."""
        self.trace(
            "retrieval",
            query=query[:200],
            results_count=results_count,
            rerank_scores=rerank_scores or [],
            intent=intent,
            **extra,
        )

    def trace_memory_save(self, memory_type: str, key: Optional[str] = None, content: str = "", **extra):
        """Trace a memory save operation."""
        self.trace("memory_save", memory_type=memory_type, key=key, content=content[:200], **extra)

    def trace_tool_call(self, tool_name: str, success: bool, duration_ms: float = 0, **extra):
        """Trace a tool execution."""
        self.trace("tool_call", tool_name=tool_name, success=success, duration_ms=duration_ms, **extra)

    def trace_generation(self, job_type: str, status: str, job_id: str = "", **extra):
        """Trace a generation pipeline step."""
        self.trace("generation", job_type=job_type, status=status, job_id=job_id, **extra)

    def get_events(
        self,
        event_type: Optional[str] = None,
        correlation_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query recent events."""
        with self._lock:
            filtered = self._events
            if event_type:
                filtered = [e for e in filtered if e.event_type == event_type]
            if correlation_id:
                filtered = [e for e in filtered if e.correlation_id == correlation_id]
            return [e.to_dict() for e in filtered[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            by_type = {}
            for e in self._events:
                by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
            return {"total_events": len(self._events), "by_type": by_type}


# ── Global singleton ────────────────────────────────────────────────────

_tracer: Optional[ObservabilityTracer] = None


def get_tracer() -> ObservabilityTracer:
    global _tracer
    if _tracer is None:
        _tracer = ObservabilityTracer()
    return _tracer
