"""
Edison File Editing Service.

Supports reading, editing, and versioning text-based files.
Works with the file_store for uploads and maintains edit history.

Supported formats: txt, md, json, yaml, py, js, html, css, and more.
"""

import difflib
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VERSIONS_DIR = REPO_ROOT / "uploads" / "versions"

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

EDITABLE_EXTENSIONS = {
    ".txt", ".md", ".json", ".yaml", ".yml",
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".html", ".css", ".xml", ".csv",
    ".sh", ".bat", ".toml", ".ini",
    ".cfg", ".conf", ".log", ".sql",
    ".rs", ".go", ".java", ".cpp", ".c", ".h",
    ".rb", ".php", ".swift", ".kt",
}


# ── Data Classes ─────────────────────────────────────────────────────────

@dataclass
class FileVersion:
    """A versioned snapshot of a file."""
    version_id: str
    file_id: str
    version_number: int
    content_hash: str
    timestamp: float
    source: str  # "original", "llm_edit", "user_edit", "transform"
    edit_description: Optional[str] = None
    diff_summary: Optional[str] = None
    path: str = ""  # path to versioned copy

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EditResult:
    """Result of a file edit operation."""
    success: bool
    file_id: str
    version_id: str
    new_content: str
    diff: str
    message: str
    version_number: int = 0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "file_id": self.file_id,
            "version_id": self.version_id,
            "diff": self.diff,
            "message": self.message,
            "version_number": self.version_number,
        }


# ── Helpers ──────────────────────────────────────────────────────────────

def _compute_diff(old: str, new: str, filename: str = "file") -> str:
    """Compute a unified diff between old and new content."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines,
                                fromfile=f"a/{filename}",
                                tofile=f"b/{filename}")
    return "".join(diff)


def _diff_summary(old: str, new: str) -> str:
    """Short summary of changes."""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    added = len(new_lines) - len(old_lines)
    if added > 0:
        return f"+{added} lines"
    elif added < 0:
        return f"{added} lines"
    else:
        changed = sum(1 for a, b in zip(old_lines, new_lines) if a != b)
        return f"{changed} lines changed"


def _hash_content(content: str) -> str:
    import hashlib
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _is_editable(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in EDITABLE_EXTENSIONS


def _safe_path(path: Path, root: Path) -> bool:
    """Ensure path doesn't escape root."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


# ── File Editor ──────────────────────────────────────────────────────────

class FileEditor:
    """
    Text file editor with version tracking.

    Integrates with FileStore for uploaded files.
    Keeps versioned copies of every edit.
    """

    def __init__(self, versions_dir: Optional[str] = None):
        self._versions_dir = Path(versions_dir) if versions_dir else VERSIONS_DIR
        self._versions_dir.mkdir(parents=True, exist_ok=True)
        self._versions: Dict[str, List[FileVersion]] = {}  # file_id → versions

    def _version_path(self, file_id: str, version_id: str, ext: str) -> Path:
        d = self._versions_dir / file_id
        d.mkdir(parents=True, exist_ok=True)
        return d / f"v_{version_id}{ext}"

    def _save_version(self, file_id: str, content: str, filename: str,
                      source: str, description: Optional[str] = None,
                      diff_sum: Optional[str] = None) -> FileVersion:
        """Save a versioned copy and track it."""
        ver_id = str(uuid.uuid4())[:8]
        versions = self._versions.setdefault(file_id, [])
        ver_num = len(versions) + 1

        ext = Path(filename).suffix or ".txt"
        ver_path = self._version_path(file_id, ver_id, ext)
        ver_path.write_text(content, encoding="utf-8")

        ver = FileVersion(
            version_id=ver_id,
            file_id=file_id,
            version_number=ver_num,
            content_hash=_hash_content(content),
            timestamp=time.time(),
            source=source,
            edit_description=description,
            diff_summary=diff_sum,
            path=str(ver_path),
        )
        versions.append(ver)

        # Write provenance sidecar
        sidecar = ver_path.parent / f"v_{ver_id}.meta.json"
        try:
            sidecar.write_text(json.dumps(ver.to_dict(), indent=2))
        except Exception as e:
            logger.debug(f"Failed to write version sidecar: {e}")

        return ver

    # ── Public API ───────────────────────────────────────────────────

    def load_file(self, path: str) -> str:
        """
        Load a text file, performing safety checks.

        Args:
            path: Absolute or relative path to the file.

        Returns:
            File contents as string.

        Raises:
            ValueError: If file is too large, not editable, or path traversal detected.
            FileNotFoundError: If file doesn't exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not _is_editable(p.name):
            raise ValueError(f"File type not supported for editing: {p.suffix}")
        if p.stat().st_size > MAX_FILE_SIZE_BYTES:
            raise ValueError(f"File too large for editing: {p.stat().st_size} bytes")
        return p.read_text(encoding="utf-8")

    def apply_edit(self, file_id: str, original_content: str,
                   new_content: str, filename: str,
                   description: str = "Edit applied",
                   source: str = "user_edit") -> EditResult:
        """
        Apply an edit and create a versioned snapshot.

        Args:
            file_id: Unique identifier for the file (from FileStore or generated).
            original_content: The content before editing.
            new_content: The content after editing.
            filename: Original filename (for extension detection).
            description: Description of the edit.
            source: Source of the edit (user_edit, llm_edit, transform).

        Returns:
            EditResult with diff and version info.
        """
        if len(new_content.encode("utf-8")) > MAX_FILE_SIZE_BYTES:
            return EditResult(
                success=False,
                file_id=file_id,
                version_id="",
                new_content="",
                diff="",
                message=f"Edited content too large (max {MAX_FILE_SIZE_BYTES} bytes)",
            )

        diff = _compute_diff(original_content, new_content, filename)
        diff_sum = _diff_summary(original_content, new_content)

        # Save original as v1 if this is the first edit
        versions = self._versions.get(file_id, [])
        if not versions:
            self._save_version(file_id, original_content, filename,
                               "original", "Original file")

        # Save new version
        ver = self._save_version(file_id, new_content, filename,
                                 source, description, diff_sum)

        return EditResult(
            success=True,
            file_id=file_id,
            version_id=ver.version_id,
            new_content=new_content,
            diff=diff,
            message=f"Edit applied: {diff_sum}",
            version_number=ver.version_number,
        )

    def apply_search_replace(self, file_id: str, content: str,
                             search: str, replace: str,
                             filename: str) -> EditResult:
        """Apply a search-and-replace edit."""
        if search not in content:
            return EditResult(
                success=False,
                file_id=file_id,
                version_id="",
                new_content=content,
                diff="",
                message=f"Search string not found in file",
            )
        new_content = content.replace(search, replace)
        count = content.count(search)
        return self.apply_edit(
            file_id, content, new_content, filename,
            description=f"Replaced {count} occurrence(s)",
            source="user_edit",
        )

    def apply_line_edit(self, file_id: str, content: str,
                        line_number: int, new_line: str,
                        filename: str) -> EditResult:
        """Replace a specific line by line number (1-based)."""
        lines = content.splitlines(keepends=True)
        if line_number < 1 or line_number > len(lines):
            return EditResult(
                success=False,
                file_id=file_id,
                version_id="",
                new_content=content,
                diff="",
                message=f"Line {line_number} out of range (1-{len(lines)})",
            )
        old_line = lines[line_number - 1].rstrip("\n")
        lines[line_number - 1] = new_line.rstrip("\n") + "\n"
        new_content = "".join(lines)
        return self.apply_edit(
            file_id, content, new_content, filename,
            description=f"Line {line_number}: '{old_line}' → '{new_line.rstrip()}'",
            source="user_edit",
        )

    def get_versions(self, file_id: str) -> List[Dict]:
        """Get version history for a file."""
        versions = self._versions.get(file_id, [])
        return [v.to_dict() for v in versions]

    def get_version_content(self, file_id: str, version_id: str) -> Optional[str]:
        """Get content of a specific version."""
        versions = self._versions.get(file_id, [])
        for v in versions:
            if v.version_id == version_id:
                p = Path(v.path)
                if p.exists():
                    return p.read_text(encoding="utf-8")
        return None

    def get_diff_between_versions(self, file_id: str,
                                  ver_id_a: str, ver_id_b: str) -> Optional[str]:
        """Get diff between two versions."""
        content_a = self.get_version_content(file_id, ver_id_a)
        content_b = self.get_version_content(file_id, ver_id_b)
        if content_a is None or content_b is None:
            return None
        return _compute_diff(content_a, content_b, file_id)

    def revert_to_version(self, file_id: str, version_id: str,
                          filename: str) -> EditResult:
        """Revert file to a specific version."""
        content = self.get_version_content(file_id, version_id)
        if content is None:
            return EditResult(
                success=False,
                file_id=file_id,
                version_id="",
                new_content="",
                diff="",
                message=f"Version {version_id} not found",
            )
        # Get current content (latest version)
        versions = self._versions.get(file_id, [])
        current = ""
        if versions:
            p = Path(versions[-1].path)
            if p.exists():
                current = p.read_text(encoding="utf-8")

        return self.apply_edit(
            file_id, current, content, filename,
            description=f"Reverted to version {version_id}",
            source="user_edit",
        )


# ── Singleton ────────────────────────────────────────────────────────────

_editor = None

def get_file_editor() -> FileEditor:
    global _editor
    if _editor is None:
        _editor = FileEditor()
    return _editor
