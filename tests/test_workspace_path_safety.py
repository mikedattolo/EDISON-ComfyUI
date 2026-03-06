"""Tests for workspace path safety helpers: _safe_workspace_path, _safe_project_path, get_workspace_root."""

import os
import sys
import re
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# We extract the pure-function logic from app.py and test it directly.
# This avoids importing the full FastAPI app (which requires GPU libs, etc.).
# The functions under test are copied here verbatim so we can test the logic.
# ---------------------------------------------------------------------------

# Simulated REPO_ROOT for tests
_TEST_REPO_ROOT = Path(tempfile.mkdtemp(prefix="edison_test_repo_"))


def _safe_workspace_path(path_str: str) -> Path:
    """Resolve a path and keep it scoped to repository root."""
    path_obj = Path(path_str or ".")
    candidate = (path_obj if path_obj.is_absolute() else (_TEST_REPO_ROOT / path_obj)).resolve()
    if not str(candidate).startswith(str(_TEST_REPO_ROOT.resolve())):
        raise ValueError("Access denied: path outside workspace")
    return candidate


def get_workspace_root(chat_id: str = None, project_id: str = None) -> Path:
    """Return a per-project/chat workspace root, creating it if needed."""
    base = _TEST_REPO_ROOT / "outputs"
    folder = project_id or chat_id or "default"
    folder = re.sub(r"[^a-zA-Z0-9_\-]", "_", folder)[:64]
    root = (base / "workspaces" / folder).resolve()
    if not str(root).startswith(str(base.resolve())):
        raise ValueError("Access denied: workspace path escape")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_project_path(workspace_root: Path, relative_path: str) -> Path:
    """Resolve *relative_path* inside *workspace_root* with traversal protection."""
    raw = Path(relative_path)
    if raw.is_absolute():
        candidate = raw.resolve()
    else:
        candidate = (workspace_root / raw).resolve()
    ws_resolved = workspace_root.resolve()
    if candidate != ws_resolved and ws_resolved not in candidate.parents:
        raise ValueError("Access denied: path outside workspace root")
    return candidate


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_repo_root():
    """Ensure a fresh test repo directory for each test."""
    _TEST_REPO_ROOT.mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree(_TEST_REPO_ROOT, ignore_errors=True)
    _TEST_REPO_ROOT.mkdir(parents=True, exist_ok=True)


# ── _safe_workspace_path tests ────────────────────────────────────────────


class TestSafeWorkspacePath:
    def test_relative_path_stays_inside(self):
        result = _safe_workspace_path("subdir/file.txt")
        assert str(result).startswith(str(_TEST_REPO_ROOT.resolve()))

    def test_dot_resolves_to_root(self):
        result = _safe_workspace_path(".")
        assert result == _TEST_REPO_ROOT.resolve()

    def test_empty_resolves_to_root(self):
        result = _safe_workspace_path("")
        assert result == _TEST_REPO_ROOT.resolve()

    def test_traversal_blocked(self):
        with pytest.raises(ValueError, match="path outside workspace"):
            _safe_workspace_path("../../../etc/passwd")

    def test_absolute_traversal_blocked(self):
        with pytest.raises(ValueError, match="path outside workspace"):
            _safe_workspace_path("/etc/passwd")

    def test_double_dot_hidden(self):
        with pytest.raises(ValueError, match="path outside workspace"):
            _safe_workspace_path("subdir/../../../../../../etc/shadow")

    def test_deeply_nested_ok(self):
        result = _safe_workspace_path("a/b/c/d/e")
        assert result == (_TEST_REPO_ROOT / "a" / "b" / "c" / "d" / "e").resolve()


# ── get_workspace_root tests ──────────────────────────────────────────────


class TestGetWorkspaceRoot:
    def test_creates_directory(self):
        root = get_workspace_root(chat_id="test_chat_123")
        assert root.exists()
        assert root.is_dir()

    def test_folder_name_sanitised(self):
        root = get_workspace_root(chat_id="../../evil")
        assert ".." not in root.name
        assert "evil" in root.name

    def test_project_id_takes_priority(self):
        root = get_workspace_root(chat_id="chat1", project_id="proj_42")
        assert "proj_42" in str(root)
        assert "chat1" not in str(root)

    def test_default_fallback(self):
        root = get_workspace_root()
        assert root.name == "default"

    def test_long_name_truncated(self):
        long_id = "a" * 200
        root = get_workspace_root(project_id=long_id)
        assert len(root.name) <= 64

    def test_special_chars_stripped(self):
        root = get_workspace_root(chat_id="hello world/bar@baz")
        # All special chars become underscore
        assert "/" not in root.name
        assert "@" not in root.name
        assert " " not in root.name


# ── _safe_project_path tests ─────────────────────────────────────────────


class TestSafeProjectPath:
    def test_relative_inside_workspace(self):
        ws = get_workspace_root(project_id="safe_test")
        result = _safe_project_path(ws, "src/main.py")
        assert str(result).startswith(str(ws.resolve()))

    def test_traversal_above_workspace(self):
        ws = get_workspace_root(project_id="safe_test2")
        with pytest.raises(ValueError, match="path outside workspace"):
            _safe_project_path(ws, "../../../etc/passwd")

    def test_absolute_outside_blocked(self):
        ws = get_workspace_root(project_id="safe_test3")
        with pytest.raises(ValueError, match="path outside workspace"):
            _safe_project_path(ws, "/tmp/evil.txt")

    def test_dot_resolves_to_workspace(self):
        ws = get_workspace_root(project_id="safe_test4")
        result = _safe_project_path(ws, ".")
        assert result == ws.resolve()

    def test_symlink_traversal_blocked(self):
        """If a symlink escapes the workspace, the resolved path should be caught."""
        ws = get_workspace_root(project_id="safe_test_symlink")
        link_path = ws / "escape_link"
        try:
            link_path.symlink_to("/tmp")
        except OSError:
            pytest.skip("Cannot create symlinks in this environment")
        with pytest.raises(ValueError, match="path outside workspace"):
            _safe_project_path(ws, "escape_link/evil.txt")

    def test_double_traversal(self):
        ws = get_workspace_root(project_id="safe_test5")
        with pytest.raises(ValueError, match="path outside workspace"):
            _safe_project_path(ws, "foo/../../../../../../etc/shadow")
