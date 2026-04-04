"""
browser_runtime.py — Browser session management and agent execution.

Wraps the Playwright-based browser system with better session state,
task narration, and step logging.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BrowserStep:
    """One step in a browser agent session."""
    step_number: int
    action: str  # navigate | click | type | scroll | observe | screenshot
    description: str = ""
    url: str = ""
    selector: str = ""
    text_input: str = ""
    result_summary: str = ""
    screenshot_b64: str = ""
    elapsed_sec: float = 0.0
    ok: bool = True
    error: str = ""


@dataclass
class BrowserSession:
    """Tracked browser session state."""
    session_id: str = ""
    task_description: str = ""
    chat_id: str = ""
    current_url: str = ""
    current_title: str = ""
    steps: List[BrowserStep] = field(default_factory=list)
    status: str = "active"  # active | paused | completed | failed
    created_at: float = 0.0
    updated_at: float = 0.0
    extracted_content: str = ""

    def __post_init__(self):
        if not self.session_id:
            self.session_id = f"browser_{uuid.uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = time.time()
        self.updated_at = time.time()

    @property
    def narration(self) -> str:
        """Produce a human-readable narration of what happened."""
        if not self.steps:
            return "No browser actions taken yet."
        lines = [f"Browser task: {self.task_description}"]
        for s in self.steps:
            status = "✓" if s.ok else "✗"
            if s.action == "navigate":
                lines.append(f"{status} Navigated to {s.url}")
            elif s.action == "click":
                lines.append(f"{status} Clicked: {s.description or s.selector}")
            elif s.action == "type":
                lines.append(f"{status} Typed: '{s.text_input[:50]}'")
            elif s.action == "observe":
                lines.append(f"{status} Observed page content")
            elif s.action == "screenshot":
                lines.append(f"{status} Took screenshot")
            elif s.action == "scroll":
                lines.append(f"{status} Scrolled page")
            else:
                lines.append(f"{status} {s.action}: {s.description}")
        return "\n".join(lines)

    def add_step(self, step: BrowserStep) -> None:
        self.steps.append(step)
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "task": self.task_description,
            "current_url": self.current_url,
            "current_title": self.current_title,
            "status": self.status,
            "steps": len(self.steps),
            "narration": self.narration,
        }


# In-memory session store
_sessions: Dict[str, BrowserSession] = {}


def create_session(
    chat_id: str,
    task_description: str = "",
    url: str = "",
) -> BrowserSession:
    """Create a new tracked browser session."""
    session = BrowserSession(
        chat_id=chat_id,
        task_description=task_description,
        current_url=url,
    )
    _sessions[session.session_id] = session
    logger.info(f"Created browser session {session.session_id}: {task_description}")
    return session


def get_session(session_id: str) -> Optional[BrowserSession]:
    return _sessions.get(session_id)


def get_sessions_for_chat(chat_id: str) -> List[BrowserSession]:
    return [s for s in _sessions.values() if s.chat_id == chat_id]


def record_step(
    session_id: str,
    action: str,
    description: str = "",
    url: str = "",
    selector: str = "",
    text_input: str = "",
    result_summary: str = "",
    screenshot_b64: str = "",
    elapsed_sec: float = 0.0,
    ok: bool = True,
    error: str = "",
) -> Optional[BrowserStep]:
    """Record a step in a browser session."""
    session = get_session(session_id)
    if not session:
        return None

    step = BrowserStep(
        step_number=len(session.steps) + 1,
        action=action,
        description=description,
        url=url,
        selector=selector,
        text_input=text_input,
        result_summary=result_summary,
        screenshot_b64=screenshot_b64,
        elapsed_sec=elapsed_sec,
        ok=ok,
        error=error,
    )
    session.add_step(step)

    if url:
        session.current_url = url

    return step


def complete_session(session_id: str, extracted_content: str = "") -> Optional[BrowserSession]:
    session = get_session(session_id)
    if session:
        session.status = "completed"
        session.extracted_content = extracted_content
        session.updated_at = time.time()
    return session


def fail_session(session_id: str, error: str = "") -> Optional[BrowserSession]:
    session = get_session(session_id)
    if session:
        session.status = "failed"
        session.updated_at = time.time()
    return session


def cleanup_old_sessions(max_age_sec: int = 3600) -> int:
    """Remove sessions older than max_age_sec."""
    now = time.time()
    to_remove = [
        sid for sid, s in _sessions.items()
        if now - s.updated_at > max_age_sec
    ]
    for sid in to_remove:
        _sessions.pop(sid, None)
    return len(to_remove)
