"""
task_runtime.py — Persistent task state that tracks across turns.

A task represents an ongoing objective with steps, artifacts, and metadata.
Tasks persist across conversation turns and can be resumed.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# In-memory task store, keyed by task_id
_tasks: Dict[str, "TaskState"] = {}
_MAX_TASKS = 1000


@dataclass
class TaskState:
    """Tracks an ongoing task across conversation turns."""
    task_id: str = ""
    chat_id: str = ""
    workspace_id: str = "default"
    project_id: str = ""
    objective: str = ""
    completed_steps: List[str] = field(default_factory=list)
    pending_steps: List[str] = field(default_factory=list)
    active_artifacts: List[str] = field(default_factory=list)
    referenced_files: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    model_used: str = ""
    mode_used: str = ""
    confidence: float = 0.5
    status: str = "active"  # active | paused | completed | failed
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.task_id:
            self.task_id = f"task_{uuid.uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = time.time()
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    def add_completed_step(self, step: str) -> None:
        self.completed_steps.append(step)
        if step in self.pending_steps:
            self.pending_steps.remove(step)
        self.updated_at = time.time()

    def add_pending_steps(self, steps: List[str]) -> None:
        for s in steps:
            if s not in self.pending_steps and s not in self.completed_steps:
                self.pending_steps.append(s)
        self.updated_at = time.time()

    def add_artifact(self, artifact_id: str) -> None:
        if artifact_id not in self.active_artifacts:
            self.active_artifacts.append(artifact_id)
        self.updated_at = time.time()

    def mark_completed(self) -> None:
        self.status = "completed"
        self.updated_at = time.time()

    def mark_failed(self, reason: str = "") -> None:
        self.status = "failed"
        self.metadata["failure_reason"] = reason
        self.updated_at = time.time()

    @property
    def summary(self) -> str:
        completed = ", ".join(self.completed_steps[-3:]) if self.completed_steps else "none"
        pending = ", ".join(self.pending_steps[:3]) if self.pending_steps else "none"
        return (
            f"Task: {self.objective}\n"
            f"Status: {self.status}\n"
            f"Completed: {completed}\n"
            f"Pending: {pending}"
        )

    def context_dict(self) -> dict:
        """Produce a dict suitable for context_runtime injection."""
        return {
            "objective": self.objective,
            "completed_steps": self.completed_steps[-5:],
            "pending_steps": self.pending_steps[:5],
            "status": self.status,
        }


# ── Task store operations ────────────────────────────────────────────

def create_task(
    chat_id: str,
    objective: str,
    workspace_id: str = "default",
    project_id: str = "",
    pending_steps: Optional[List[str]] = None,
) -> TaskState:
    """Create and store a new task."""
    task = TaskState(
        chat_id=chat_id,
        workspace_id=workspace_id,
        project_id=project_id,
        objective=objective,
        pending_steps=pending_steps or [],
    )
    _tasks[task.task_id] = task
    _prune_tasks()
    logger.info(f"Created task {task.task_id}: {objective}")
    return task


def get_task(task_id: str) -> Optional[TaskState]:
    return _tasks.get(task_id)


def get_active_task_for_chat(chat_id: str) -> Optional[TaskState]:
    """Return the most recent active task for a given chat."""
    candidates = [
        t for t in _tasks.values()
        if t.chat_id == chat_id and t.status == "active"
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t.updated_at)


def list_tasks(
    chat_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
) -> List[TaskState]:
    """List tasks with optional filters."""
    tasks = list(_tasks.values())
    if chat_id:
        tasks = [t for t in tasks if t.chat_id == chat_id]
    if workspace_id:
        tasks = [t for t in tasks if t.workspace_id == workspace_id]
    if status:
        tasks = [t for t in tasks if t.status == status]
    tasks.sort(key=lambda t: t.updated_at, reverse=True)
    return tasks[:limit]


def complete_task(task_id: str) -> Optional[TaskState]:
    task = get_task(task_id)
    if task:
        task.mark_completed()
    return task


def _prune_tasks():
    """Remove old completed/failed tasks to prevent unbounded growth."""
    if len(_tasks) <= _MAX_TASKS:
        return
    prunable = sorted(
        [(tid, t) for tid, t in _tasks.items() if t.status in ("completed", "failed")],
        key=lambda x: x[1].updated_at,
    )
    while len(_tasks) > _MAX_TASKS and prunable:
        tid, _ = prunable.pop(0)
        _tasks.pop(tid, None)
