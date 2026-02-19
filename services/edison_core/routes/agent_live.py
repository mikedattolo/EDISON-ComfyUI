"""
Agent Live View — real-time streaming of agent activity to the UI.

Provides:
  GET  /agent/stream       — SSE stream of structured agent events
  GET  /agent/live-config  — whether live view is enabled
  WS   /ws/agent           — (optional) WebSocket transport

Event types emitted:
  agent_step       — high-level step update
  browser_open     — agent opened a URL
  browser_screenshot — base64 PNG thumbnail
  file_diff        — file edit preview
  log              — debug / info / warning message
"""

import asyncio
import json
import logging
import re
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agent_live_view"])


# ── In-memory event bus (process-local, sufficient for single-server) ────

class AgentEventBus:
    """Thread-safe fan-out event bus for agent activity."""

    def __init__(self):
        self._subscribers: Dict[str, asyncio.Queue] = {}
        self._history: Dict[str, list] = defaultdict(list)  # session → last N events
        self._max_history = 200

    def subscribe(self, session_id: str = "default") -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        sub_id = f"{session_id}_{uuid.uuid4().hex[:8]}"
        self._subscribers[sub_id] = q
        # Send history to new subscriber
        for evt in self._history.get(session_id, []):
            q.put_nowait(evt)
        return q

    def unsubscribe(self, queue: asyncio.Queue):
        to_remove = [k for k, v in self._subscribers.items() if v is queue]
        for k in to_remove:
            del self._subscribers[k]

    def emit(self, event: dict, session_id: str = "default"):
        """Emit an event to all subscribers watching this session."""
        event.setdefault("ts", time.time())
        event.setdefault("session_id", session_id)

        # Redact secrets
        event = _redact_secrets(event)

        # Store in history
        hist = self._history[session_id]
        hist.append(event)
        if len(hist) > self._max_history:
            self._history[session_id] = hist[-self._max_history:]

        # Fan out
        dead = []
        for sub_id, q in self._subscribers.items():
            if sub_id.startswith(session_id) or sub_id.startswith("default"):
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    dead.append(sub_id)
        for k in dead:
            del self._subscribers[k]


# Singleton
_event_bus: Optional[AgentEventBus] = None

def get_event_bus() -> AgentEventBus:
    global _event_bus
    if _event_bus is None:
        _event_bus = AgentEventBus()
    return _event_bus


# ── Secret redaction ─────────────────────────────────────────────────────

_SECRET_PATTERNS = [
    re.compile(r"(sk-[a-zA-Z0-9]{20,})", re.I),
    re.compile(r"(ghp_[a-zA-Z0-9]{36,})", re.I),
    re.compile(r"(Bearer\s+[a-zA-Z0-9._\-]{20,})", re.I),
    re.compile(r"(api[_-]?key[\"']?\s*[:=]\s*[\"']?[a-zA-Z0-9_\-]{16,})", re.I),
    re.compile(r"(token[\"']?\s*[:=]\s*[\"']?[a-zA-Z0-9._\-]{16,})", re.I),
]


def _redact_secrets(event: dict) -> dict:
    """Deep-redact known secret patterns from event values."""
    try:
        from services.edison_core.app import config as app_config
        cfg = (app_config or {}).get("edison", {}).get("agent_live_view", {})
        if not cfg.get("redact_secrets", True):
            return event
    except ImportError:
        pass

    def _scrub(val):
        if isinstance(val, str):
            for pat in _SECRET_PATTERNS:
                val = pat.sub("[REDACTED]", val)
            return val
        if isinstance(val, dict):
            return {k: _scrub(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_scrub(v) for v in val]
        return val

    return _scrub(event)


# ── Helper: emit from anywhere in the codebase ──────────────────────────

def emit_agent_step(
    title: str,
    status: str = "running",
    step_id: str = None,
    session_id: str = "default",
    **extra,
):
    """Convenience wrapper used by agent/tool code to emit a step event."""
    get_event_bus().emit({
        "type": "agent_step",
        "step_id": step_id or uuid.uuid4().hex[:12],
        "title": title,
        "status": status,
        **extra,
    }, session_id=session_id)


def emit_browser_event(url: str, session_id: str = "default", screenshot_b64: str = None):
    evt: Dict[str, Any] = {"type": "browser_open", "url": url}
    if screenshot_b64:
        evt["type"] = "browser_screenshot"
        evt["png_base64"] = screenshot_b64
    get_event_bus().emit(evt, session_id=session_id)


def emit_file_diff(path: str, diff: str, session_id: str = "default"):
    get_event_bus().emit({
        "type": "file_diff",
        "path": path,
        "diff": diff,
    }, session_id=session_id)


def emit_log(message: str, level: str = "info", session_id: str = "default"):
    get_event_bus().emit({
        "type": "log",
        "level": level,
        "message": message,
    }, session_id=session_id)


# ── SSE endpoint ─────────────────────────────────────────────────────────

@router.get("/agent/stream")
async def agent_stream(request: Request, session_id: str = "default"):
    """
    Server-Sent Events stream of agent activity.
    Connect from the UI with ``new EventSource('/agent/stream?session_id=...')``.
    """
    bus = get_event_bus()
    q = bus.subscribe(session_id)

    async def _generate():
        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f": keepalive {time.time()}\n\n"
        finally:
            bus.unsubscribe(q)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── WebSocket transport (optional) ──────────────────────────────────────

@router.websocket("/ws/agent")
async def agent_websocket(ws: WebSocket, session_id: str = "default"):
    await ws.accept()
    bus = get_event_bus()
    q = bus.subscribe(session_id)
    try:
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=30)
                await ws.send_text(json.dumps(event))
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"type": "keepalive"}))
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(q)


# ── Config endpoint ──────────────────────────────────────────────────────

@router.get("/agent/live-config")
async def agent_live_config():
    """Return live-view configuration to the frontend."""
    try:
        from services.edison_core.app import config as app_config
    except ImportError:
        app_config = None

    cfg = {}
    if app_config:
        cfg = app_config.get("edison", {}).get("agent_live_view", {})

    return {
        "enabled": cfg.get("enabled", True),
        "redact_secrets": cfg.get("redact_secrets", True),
        "screenshot_interval_s": cfg.get("screenshot_interval_s", 5),
        "sse_url": "/agent/stream",
        "ws_url": "/ws/agent",
    }
