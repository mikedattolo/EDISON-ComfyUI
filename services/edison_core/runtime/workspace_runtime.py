"""
workspace_runtime.py — Logical workspace layer over existing filesystem.

Does NOT move or restructure any files. Provides a logical grouping of
projects, artifacts, memory scope, and tool preferences.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_workspaces: Dict[str, "Workspace"] = {}


@dataclass
class Workspace:
    """A logical workspace grouping projects, artifacts, and preferences."""
    workspace_id: str = ""
    name: str = "Default"
    description: str = ""
    client_name: str = ""
    project_ids: List[str] = field(default_factory=list)
    active_artifact_ids: List[str] = field(default_factory=list)
    preferred_tools: List[str] = field(default_factory=list)
    default_mode: str = "auto"
    default_style: str = ""
    memory_scope: str = ""  # will map to memory_scopes.py
    recent_task_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.workspace_id:
            self.workspace_id = f"ws_{uuid.uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = time.time()
        self.updated_at = time.time()
        if not self.memory_scope:
            self.memory_scope = f"workspace:{self.workspace_id}"

    def to_dict(self) -> dict:
        return asdict(self)


def ensure_default_workspace() -> Workspace:
    """Ensure the default workspace exists."""
    if "default" not in _workspaces:
        ws = Workspace(workspace_id="default", name="Edison Core")
        _workspaces["default"] = ws
    return _workspaces["default"]


def create_workspace(
    name: str,
    client_name: str = "",
    description: str = "",
    tags: Optional[List[str]] = None,
) -> Workspace:
    ws = Workspace(
        name=name,
        client_name=client_name,
        description=description,
        tags=tags or [],
    )
    _workspaces[ws.workspace_id] = ws
    logger.info(f"Created workspace {ws.workspace_id}: {name}")
    return ws


def get_workspace(workspace_id: str) -> Optional[Workspace]:
    return _workspaces.get(workspace_id)


def list_workspaces() -> List[Workspace]:
    return sorted(_workspaces.values(), key=lambda w: w.updated_at, reverse=True)


def update_workspace(workspace_id: str, **kwargs) -> Optional[Workspace]:
    ws = get_workspace(workspace_id)
    if not ws:
        return None
    for k, v in kwargs.items():
        if hasattr(ws, k) and k not in ("workspace_id", "created_at"):
            setattr(ws, k, v)
    ws.updated_at = time.time()
    return ws


def delete_workspace(workspace_id: str) -> bool:
    if workspace_id == "default":
        return False
    return _workspaces.pop(workspace_id, None) is not None
