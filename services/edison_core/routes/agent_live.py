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
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agent_live_view"])


@dataclass
class AgentSessionState:
    session_id: str
    state: str = "running"
    current_objective: str = ""
    recent_actions: list[Dict[str, Any]] = field(default_factory=list)
    artifacts: list[Dict[str, Any]] = field(default_factory=list)
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgentSessionStore:
    """Small process-local state model for recoverable live agent sessions."""

    TERMINAL = {"completed", "failed", "cancelled"}
    STATES = {"planned", "running", "paused", "interrupted", "completed", "failed", "cancelled"}

    def __init__(self):
        self._sessions: Dict[str, AgentSessionState] = {}

    def create(self, objective: str = "", session_id: Optional[str] = None) -> Dict[str, Any]:
        sid = session_id or f"agent_{uuid.uuid4().hex[:12]}"
        state = AgentSessionState(session_id=sid, current_objective=objective or "", state="planned")
        self._sessions[sid] = state
        return state.to_dict()

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        state = self._sessions.get(session_id)
        return state.to_dict() if state else None

    def list(self) -> list[Dict[str, Any]]:
        return sorted((s.to_dict() for s in self._sessions.values()), key=lambda item: item["updated_at"], reverse=True)

    def update(self, session_id: str, *, state: Optional[str] = None, error: str = "", objective: Optional[str] = None) -> Dict[str, Any]:
        current = self._sessions.setdefault(session_id, AgentSessionState(session_id=session_id))
        if state:
            if state not in self.STATES:
                raise ValueError(f"Unsupported agent session state '{state}'")
            current.state = state
        if objective is not None:
            current.current_objective = objective
        if error:
            current.error = error
        current.updated_at = time.time()
        return current.to_dict()

    def record_event(self, session_id: str, event: Dict[str, Any]):
        current = self._sessions.setdefault(session_id, AgentSessionState(session_id=session_id))
        event_type = event.get("type")
        if event_type in {"agent_step", "browser_open", "browser_screenshot", "tool_call", "log"}:
            current.recent_actions.append({
                "type": event_type,
                "title": event.get("title") or event.get("url") or event.get("message") or event_type,
                "status": event.get("status"),
                "ts": event.get("ts", time.time()),
            })
            current.recent_actions = current.recent_actions[-50:]
        if event_type in {"file_diff", "artifact"}:
            current.artifacts.append({k: event.get(k) for k in ("type", "path", "url", "title", "ts") if k in event})
            current.artifacts = current.artifacts[-50:]
        if event.get("status") in self.STATES:
            current.state = event["status"]
        if event.get("error"):
            current.error = str(event["error"])
            current.state = "failed"
        current.updated_at = time.time()


_session_store: Optional[AgentSessionStore] = None


def get_session_store() -> AgentSessionStore:
    global _session_store
    if _session_store is None:
        _session_store = AgentSessionStore()
    return _session_store


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

        get_session_store().record_event(session_id, event)

        # Fan out to all subscribers (session filtering relaxed so
        # "default" events reach every subscriber and vice-versa)
        dead = []
        for sub_id, q in self._subscribers.items():
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


@router.post("/agent/sessions")
async def create_agent_session(payload: Dict[str, Any] | None = None):
    payload = payload or {}
    session = get_session_store().create(
        objective=str(payload.get("objective") or payload.get("current_objective") or ""),
        session_id=payload.get("session_id"),
    )
    get_event_bus().emit(
        {"type": "agent_step", "title": "Agent session created", "status": "planned"},
        session_id=session["session_id"],
    )
    return {"ok": True, "session": session}


@router.get("/agent/sessions")
async def list_agent_sessions():
    return {"sessions": get_session_store().list()}


@router.get("/agent/sessions/{session_id}")
async def get_agent_session(session_id: str):
    session = get_session_store().get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Agent session not found")
    return {"session": session}


@router.post("/agent/sessions/{session_id}/state")
async def update_agent_session_state(session_id: str, payload: Dict[str, Any]):
    try:
        session = get_session_store().update(
            session_id,
            state=payload.get("state"),
            error=str(payload.get("error") or ""),
            objective=payload.get("current_objective"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    get_event_bus().emit(
        {"type": "agent_step", "title": f"Agent session {session['state']}", "status": session["state"]},
        session_id=session_id,
    )
    return {"ok": True, "session": session}


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
