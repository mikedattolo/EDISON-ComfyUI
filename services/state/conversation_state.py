"""
Conversation State Layer for Edison.

Tracks per-session structured state: current project, task, domain,
last tool/artifact, task stage, and intent — enabling context-aware
routing and continuation detection.
"""

import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Data model ───────────────────────────────────────────────────────────

VALID_DOMAINS = {
    "code", "image", "video", "music", "mesh", "hardware",
    "research", "writing", "data", "system", "conversation", "unknown",
}

VALID_TASK_STAGES = {
    "idle", "planning", "executing", "reviewing", "iterating",
    "debugging", "waiting_for_input", "complete",
}


@dataclass
class ConversationState:
    """Structured state for a single session/chat."""

    session_id: str
    current_project: Optional[str] = None
    current_task: Optional[str] = None
    last_tool_used: Optional[str] = None
    last_generated_artifact: Optional[str] = None
    active_domain: str = "unknown"
    task_stage: str = "idle"
    last_intent: Optional[str] = None
    last_goal: Optional[str] = None
    last_confidence: float = 0.0
    turn_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    history_summary: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_context_string(self) -> str:
        """Compact string for injection into LLM system prompts."""
        parts = []
        if self.current_project:
            parts.append(f"Project: {self.current_project}")
        if self.current_task:
            parts.append(f"Task: {self.current_task}")
        if self.active_domain != "unknown":
            parts.append(f"Domain: {self.active_domain}")
        if self.task_stage != "idle":
            parts.append(f"Stage: {self.task_stage}")
        if self.last_tool_used:
            parts.append(f"Last tool: {self.last_tool_used}")
        if self.last_generated_artifact:
            parts.append(f"Last artifact: {self.last_generated_artifact}")
        if self.last_intent:
            parts.append(f"Last intent: {self.last_intent}")
        if self.error_count > 0:
            parts.append(f"Errors: {self.error_count}")
        return "; ".join(parts) if parts else "No active context"


# ── State store ──────────────────────────────────────────────────────────

# Max entries before pruning oldest idle sessions
_MAX_SESSIONS = 500
# Idle timeout (seconds) before a session is eligible for pruning
_IDLE_TIMEOUT = 3600 * 4  # 4 hours


class ConversationStateManager:
    """In-memory per-session state store with thread safety."""

    def __init__(self):
        self._states: Dict[str, ConversationState] = {}
        self._lock = threading.Lock()
        logger.info("ConversationStateManager initialized")

    # ── Public API ───────────────────────────────────────────────────────

    def get_state(self, session_id: str) -> ConversationState:
        """Return the state for *session_id*, creating it if needed."""
        with self._lock:
            if session_id not in self._states:
                self._states[session_id] = ConversationState(session_id=session_id)
                logger.debug(f"Created new conversation state for session {session_id}")
            return self._states[session_id]

    def update_state(self, session_id: str, updates: Dict[str, Any]) -> ConversationState:
        """Merge *updates* into an existing (or new) session state.

        Only known fields are applied; unknown keys are silently ignored.
        """
        state = self.get_state(session_id)
        applied = []
        for key, value in updates.items():
            if not hasattr(state, key) or key in ("session_id", "created_at"):
                continue
            # Validate constrained fields
            if key == "active_domain" and value not in VALID_DOMAINS:
                logger.warning(f"Invalid domain '{value}', ignoring")
                continue
            if key == "task_stage" and value not in VALID_TASK_STAGES:
                logger.warning(f"Invalid task_stage '{value}', ignoring")
                continue
            setattr(state, key, value)
            applied.append(key)
        state.updated_at = time.time()
        if applied:
            logger.debug(f"State update [{session_id}]: {', '.join(applied)}")
        return state

    def reset_state(self, session_id: str) -> ConversationState:
        """Reset a session back to a blank state, keeping the session_id."""
        with self._lock:
            self._states[session_id] = ConversationState(session_id=session_id)
        logger.info(f"Reset conversation state for session {session_id}")
        return self._states[session_id]

    def increment_turn(self, session_id: str) -> int:
        """Bump turn counter and return new count."""
        state = self.get_state(session_id)
        state.turn_count += 1
        state.updated_at = time.time()
        return state.turn_count

    def record_error(self, session_id: str, error_msg: str):
        """Track an error occurrence in the session."""
        state = self.get_state(session_id)
        state.error_count += 1
        state.last_error = error_msg[:500]
        state.updated_at = time.time()

    def add_history_note(self, session_id: str, note: str, max_notes: int = 20):
        """Append a concise history note (capped at *max_notes*)."""
        state = self.get_state(session_id)
        state.history_summary.append(note[:200])
        if len(state.history_summary) > max_notes:
            state.history_summary = state.history_summary[-max_notes:]

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Return summaries of all active sessions."""
        with self._lock:
            return [
                {
                    "session_id": s.session_id,
                    "domain": s.active_domain,
                    "task_stage": s.task_stage,
                    "turn_count": s.turn_count,
                    "updated_at": s.updated_at,
                }
                for s in self._states.values()
            ]

    def prune_idle(self):
        """Remove sessions idle longer than _IDLE_TIMEOUT, down to _MAX_SESSIONS."""
        now = time.time()
        with self._lock:
            before = len(self._states)
            self._states = {
                sid: s
                for sid, s in self._states.items()
                if (now - s.updated_at) < _IDLE_TIMEOUT
            }
            # Hard cap
            if len(self._states) > _MAX_SESSIONS:
                sorted_sessions = sorted(
                    self._states.items(), key=lambda kv: kv[1].updated_at
                )
                self._states = dict(sorted_sessions[-_MAX_SESSIONS:])
            pruned = before - len(self._states)
            if pruned:
                logger.info(f"Pruned {pruned} idle conversation states")


# ── Singleton accessor ───────────────────────────────────────────────────

_manager: Optional[ConversationStateManager] = None
_manager_lock = threading.Lock()


def get_conversation_state_manager() -> ConversationStateManager:
    """Return the global ConversationStateManager singleton."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = ConversationStateManager()
    return _manager


# ── Domain detection helper ──────────────────────────────────────────────

_DOMAIN_PATTERNS: Dict[str, List[str]] = {
    "code": [
        "code", "function", "class", "debug", "error", "stacktrace", "compile",
        "python", "javascript", "typescript", "rust", "java", "bug", "refactor",
        "api", "endpoint", "deploy", "docker", "git", "commit", "pull request",
        "lint", "test", "unittest", "pytest",
    ],
    "image": [
        "image", "photo", "picture", "draw", "sketch", "illustration",
        "portrait", "landscape", "render", "photorealistic", "anime",
        "flux", "stable diffusion", "comfyui",
    ],
    "video": [
        "video", "clip", "animate", "animation", "motion",
        "film", "movie", "footage", "cogvideo",
    ],
    "music": [
        "music", "song", "beat", "melody", "compose", "track",
        "instrumental", "lo-fi", "lofi", "hip hop", "edm", "soundtrack",
    ],
    "mesh": [
        "3d", "mesh", "model 3d", "stl", "glb", "sculpt", "blender",
    ],
    "hardware": [
        "gpu", "cpu", "ram", "disk", "temperature", "fan", "power",
        "sensor", "coral", "tpu", "usb", "pcie",
    ],
    "research": [
        "research", "paper", "arxiv", "study", "abstract", "cite",
        "survey", "literature", "journal",
    ],
    "writing": [
        "essay", "blog", "article", "story", "poem", "letter", "email",
        "document", "report", "summary", "draft",
    ],
    "data": [
        "data", "dataset", "csv", "dataframe", "sql", "database",
        "analytics", "visualization", "chart", "graph", "statistics",
    ],
    "system": [
        "system", "settings", "config", "restart", "status", "health",
        "update", "install", "memory usage", "disk space",
    ],
}


def detect_domain(message: str) -> str:
    """Detect the active domain from message text using keyword matching.

    Returns the domain with the highest hit count, or "conversation".
    """
    msg_lower = message.lower()
    scores: Dict[str, int] = {}
    for domain, keywords in _DOMAIN_PATTERNS.items():
        score = sum(1 for kw in keywords if kw in msg_lower)
        if score > 0:
            scores[domain] = score
    if not scores:
        return "conversation"
    return max(scores, key=scores.get)
