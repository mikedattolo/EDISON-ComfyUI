"""
artifact_runtime.py — First-class artifact registry and lifecycle.

Anything EDISON creates (documents, images, code, briefs, etc.) is tracked
as an artifact with metadata, revision history, and project linkage.
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

# In-memory artifact store
_artifacts: Dict[str, "Artifact"] = {}
_MAX_ARTIFACTS = 5000


@dataclass
class Artifact:
    """A tracked output produced by EDISON."""
    artifact_id: str = ""
    artifact_type: str = "note"  # note | report | prompt | brand_brief | code | image | ...
    title: str = ""
    workspace_id: str = "default"
    project_id: str = ""
    task_id: str = ""
    chat_id: str = ""
    path: str = ""              # filesystem path if applicable
    content: str = ""           # inline content for small artifacts
    summary: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    revision_parent_id: str = ""
    revision_number: int = 1
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.artifact_id:
            self.artifact_id = f"art_{uuid.uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = time.time()
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        return asdict(self)


# ── Registry operations ──────────────────────────────────────────────

def register_artifact(
    *,
    artifact_type: str = "note",
    title: str = "",
    workspace_id: str = "default",
    project_id: str = "",
    task_id: str = "",
    chat_id: str = "",
    path: str = "",
    content: str = "",
    summary: str = "",
    tags: Optional[List[str]] = None,
    metadata: Optional[dict] = None,
) -> Artifact:
    """Create and store a new artifact."""
    art = Artifact(
        artifact_type=artifact_type,
        title=title or f"Untitled {artifact_type}",
        workspace_id=workspace_id,
        project_id=project_id,
        task_id=task_id,
        chat_id=chat_id,
        path=path,
        content=content,
        summary=summary,
        tags=tags or [],
        metadata=metadata or {},
    )
    _artifacts[art.artifact_id] = art
    _prune_artifacts()
    logger.info(f"Registered artifact {art.artifact_id}: {art.title} ({art.artifact_type})")
    return art


def get_artifact(artifact_id: str) -> Optional[Artifact]:
    return _artifacts.get(artifact_id)


def get_artifacts_for_chat(chat_id: str, limit: int = 20) -> List[Artifact]:
    arts = [a for a in _artifacts.values() if a.chat_id == chat_id]
    arts.sort(key=lambda a: a.updated_at, reverse=True)
    return arts[:limit]


def get_artifacts_for_workspace(workspace_id: str, limit: int = 50) -> List[Artifact]:
    arts = [a for a in _artifacts.values() if a.workspace_id == workspace_id]
    arts.sort(key=lambda a: a.updated_at, reverse=True)
    return arts[:limit]


def get_artifacts_for_task(task_id: str) -> List[Artifact]:
    return [a for a in _artifacts.values() if a.task_id == task_id]


def search_artifacts(query: str, limit: int = 10) -> List[Artifact]:
    """Simple text-based artifact search across title, summary, tags."""
    q = query.lower()
    scored = []
    for a in _artifacts.values():
        score = 0
        if q in a.title.lower():
            score += 3
        if q in a.summary.lower():
            score += 2
        if any(q in t.lower() for t in a.tags):
            score += 1
        if score > 0:
            scored.append((score, a))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [a for _, a in scored[:limit]]


def revise_artifact(
    artifact_id: str,
    *,
    content: str = "",
    summary: str = "",
    title: str = "",
    metadata: Optional[dict] = None,
) -> Optional[Artifact]:
    """Create a new revision of an existing artifact."""
    parent = get_artifact(artifact_id)
    if not parent:
        return None
    new_art = Artifact(
        artifact_type=parent.artifact_type,
        title=title or parent.title,
        workspace_id=parent.workspace_id,
        project_id=parent.project_id,
        task_id=parent.task_id,
        chat_id=parent.chat_id,
        path=parent.path,
        content=content or parent.content,
        summary=summary or parent.summary,
        revision_parent_id=parent.artifact_id,
        revision_number=parent.revision_number + 1,
        tags=list(parent.tags),
        metadata={**parent.metadata, **(metadata or {})},
    )
    _artifacts[new_art.artifact_id] = new_art
    logger.info(f"Revised artifact {artifact_id} → {new_art.artifact_id} (rev {new_art.revision_number})")
    return new_art


def delete_artifact(artifact_id: str) -> bool:
    return _artifacts.pop(artifact_id, None) is not None


def list_recent_artifacts(limit: int = 20) -> List[Artifact]:
    arts = sorted(_artifacts.values(), key=lambda a: a.updated_at, reverse=True)
    return arts[:limit]


def artifact_refs_for_context(chat_id: str, limit: int = 5) -> List[dict]:
    """Produce artifact reference dicts suitable for context_runtime."""
    arts = get_artifacts_for_chat(chat_id, limit=limit)
    return [
        {
            "artifact_id": a.artifact_id,
            "type": a.artifact_type,
            "title": a.title,
            "summary": a.summary[:200] if a.summary else "",
        }
        for a in arts
    ]


def _prune_artifacts():
    if len(_artifacts) <= _MAX_ARTIFACTS:
        return
    sorted_arts = sorted(_artifacts.values(), key=lambda a: a.updated_at)
    while len(_artifacts) > _MAX_ARTIFACTS:
        oldest = sorted_arts.pop(0)
        _artifacts.pop(oldest.artifact_id, None)
