"""
Project Awareness for Edison.

Tracks recent files edited, repositories referenced, and working
directory context to bias retrieval and suggestions toward the
user's active project.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()


@dataclass
class ProjectContext:
    """Tracks the state of the user's active project."""

    name: Optional[str] = None
    root_path: Optional[str] = None
    language: Optional[str] = None
    recent_files: List[str] = field(default_factory=list)
    recent_repos: List[str] = field(default_factory=list)
    working_directory: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    _MAX_RECENT = 20

    def add_file(self, filepath: str):
        """Record a referenced/edited file (deduplicated, capped)."""
        if filepath in self.recent_files:
            self.recent_files.remove(filepath)
        self.recent_files.insert(0, filepath)
        if len(self.recent_files) > self._MAX_RECENT:
            self.recent_files = self.recent_files[: self._MAX_RECENT]
        self.updated_at = time.time()
        self._infer_language(filepath)

    def add_repo(self, repo_identifier: str):
        """Record a repository reference."""
        if repo_identifier in self.recent_repos:
            self.recent_repos.remove(repo_identifier)
        self.recent_repos.insert(0, repo_identifier)
        if len(self.recent_repos) > 10:
            self.recent_repos = self.recent_repos[:10]
        self.updated_at = time.time()

    def _infer_language(self, filepath: str):
        """Infer primary language from file extension."""
        ext_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".java": "java", ".rs": "rust", ".go": "go", ".cpp": "cpp",
            ".c": "c", ".rb": "ruby", ".php": "php", ".swift": "swift",
            ".kt": "kotlin", ".cs": "csharp", ".html": "html", ".css": "css",
        }
        ext = Path(filepath).suffix.lower()
        if ext in ext_map:
            self.language = ext_map[ext]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_context_string(self) -> str:
        """Compact summary for LLM context."""
        parts = []
        if self.name:
            parts.append(f"Project: {self.name}")
        if self.language:
            parts.append(f"Language: {self.language}")
        if self.recent_files:
            parts.append(f"Recent files: {', '.join(self.recent_files[:5])}")
        if self.recent_repos:
            parts.append(f"Repos: {', '.join(self.recent_repos[:3])}")
        if self.working_directory:
            parts.append(f"CWD: {self.working_directory}")
        return "; ".join(parts) if parts else "No project context"


# ── Project state manager ────────────────────────────────────────────────

class ProjectStateManager:
    """Thread-safe per-session project context tracker."""

    def __init__(self):
        self._contexts: Dict[str, ProjectContext] = {}
        self._lock = threading.Lock()
        logger.info("ProjectStateManager initialized")

    def get_context(self, session_id: str) -> ProjectContext:
        with self._lock:
            if session_id not in self._contexts:
                self._contexts[session_id] = ProjectContext()
            return self._contexts[session_id]

    def update_context(self, session_id: str, updates: Dict[str, Any]) -> ProjectContext:
        ctx = self.get_context(session_id)
        for key, value in updates.items():
            if hasattr(ctx, key) and key not in ("recent_files", "recent_repos"):
                setattr(ctx, key, value)
        ctx.updated_at = time.time()
        return ctx

    def add_file_reference(self, session_id: str, filepath: str):
        ctx = self.get_context(session_id)
        ctx.add_file(filepath)

    def add_repo_reference(self, session_id: str, repo: str):
        ctx = self.get_context(session_id)
        ctx.add_repo(repo)

    def detect_project_from_message(self, session_id: str, message: str):
        """Extract project and file references from a message."""
        import re
        ctx = self.get_context(session_id)

        # Detect file paths mentioned in the message
        file_patterns = re.findall(
            r'(?:^|\s)([a-zA-Z0-9_/\\.-]+\.(?:py|js|ts|java|rs|go|cpp|c|rb|php|html|css|json|yaml|yml|md|sh|sql))\b',
            message
        )
        for fp in file_patterns:
            ctx.add_file(fp)

        # Detect GitHub-style repo references
        repo_patterns = re.findall(
            r'(?:github\.com/|gitlab\.com/|https?://)?([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)',
            message
        )
        for repo in repo_patterns:
            # Filter out false positives (common patterns that look like repos)
            if "/" in repo and not repo.startswith("http"):
                ctx.add_repo(repo)

        # Detect project name from "working on X" patterns
        project_match = re.search(
            r'\b(?:working on|project|repo|repository)\s+(?:(?:called|project|repo|repository)\s+)*["\']?(\w[\w.-]*)',
            message, re.IGNORECASE
        )
        if project_match:
            ctx.name = project_match.group(1)

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {"session_id": sid, **ctx.to_dict()}
                for sid, ctx in self._contexts.items()
            ]


# ── Singleton ────────────────────────────────────────────────────────────

_manager: Optional[ProjectStateManager] = None
_manager_lock = threading.Lock()


def get_project_state_manager() -> ProjectStateManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = ProjectStateManager()
    return _manager
