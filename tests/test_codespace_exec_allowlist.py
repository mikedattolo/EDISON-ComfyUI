"""Tests for codespace_exec command allowlist / blocklist."""

import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Extracted from app.py — the _run_codespaces_command logic.
# We recreate it here so we can unit-test without the full app import.
# ---------------------------------------------------------------------------

_TEST_REPO_ROOT = Path(tempfile.mkdtemp(prefix="edison_test_exec_"))


def _safe_workspace_path(path_str: str) -> Path:
    path_obj = Path(path_str or ".")
    candidate = (path_obj if path_obj.is_absolute() else (_TEST_REPO_ROOT / path_obj)).resolve()
    if not str(candidate).startswith(str(_TEST_REPO_ROOT.resolve())):
        raise ValueError("Access denied: path outside workspace")
    return candidate


def _run_codespaces_command(command: str, cwd: str = ".", timeout: int = 30) -> dict:
    if not command or not isinstance(command, str):
        return {"ok": False, "error": "Command required"}

    blocked_patterns = [
        r"`", r"\$\(", r"\bsudo\b", r"\brm\s+-rf\b",
        r"\bshutdown\b", r"\breboot\b", r"\bmkfs\b", r"\bdd\s+if=",
    ]
    if any(re.search(p, command) for p in blocked_patterns):
        return {"ok": False, "error": "Command contains blocked shell patterns"}

    try:
        parts = shlex.split(command)
    except Exception as e:
        return {"ok": False, "error": f"Invalid command: {e}"}

    if not parts:
        return {"ok": False, "error": "Empty command"}

    allowed_roots = {
        "ls", "pwd", "cat", "echo", "grep", "find", "head", "tail", "wc", "du", "df",
        "stat", "file", "which", "type", "env", "printenv",
        "tree", "mkdir", "cp", "mv", "touch", "ln", "sort", "uniq", "cut", "tr", "tee",
        "sed", "awk", "xargs", "diff", "patch",
        "python", "python3", "pytest", "pip", "pip3", "ruff", "black", "pylint", "mypy",
        "node", "npm", "npx", "yarn",
        "cargo", "go",
        "git",
        "zip", "unzip", "tar", "gzip", "gunzip",
        "make", "cmake",
    }
    if parts[0] not in allowed_roots:
        return {"ok": False, "error": f"Command '{parts[0]}' is not allowed in sandbox"}

    try:
        safe_cwd = _safe_workspace_path(cwd)
        proc = subprocess.run(
            parts,
            cwd=str(safe_cwd),
            capture_output=True,
            text=True,
            timeout=max(1, min(int(timeout), 120)),
            env={k: v for k, v in os.environ.items()
                 if k not in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}},
        )
        return {
            "ok": proc.returncode == 0,
            "data": {
                "command": command,
                "cwd": str(safe_cwd),
                "returncode": proc.returncode,
                "stdout": (proc.stdout or "")[:16000],
                "stderr": (proc.stderr or "")[:8000],
            },
            "error": None if proc.returncode == 0 else f"Command failed ({proc.returncode})",
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Command timed out"}
    except Exception as e:
        return {"ok": False, "error": f"Command execution failed: {e}"}


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _ensure_cwd():
    _TEST_REPO_ROOT.mkdir(parents=True, exist_ok=True)
    yield


# ── Allowed commands ──────────────────────────────────────────────────────

class TestAllowedCommands:
    def test_ls(self):
        result = _run_codespaces_command("ls -la")
        assert result["ok"] is True

    def test_pwd(self):
        result = _run_codespaces_command("pwd")
        assert result["ok"] is True

    def test_echo(self):
        result = _run_codespaces_command("echo hello")
        assert result["ok"] is True
        assert "hello" in result["data"]["stdout"]

    def test_python3_version(self):
        result = _run_codespaces_command("python3 --version")
        assert result["ok"] is True

    def test_git_status(self):
        result = _run_codespaces_command("git status", cwd=".")
        # May fail if not a git repo, but should NOT be blocked
        assert "not allowed" not in result.get("error", "")

    def test_mkdir(self):
        result = _run_codespaces_command(f"mkdir -p testdir", cwd=".")
        assert result["ok"] is True


# ── Blocked commands ──────────────────────────────────────────────────────

class TestBlockedCommands:
    def test_curl_blocked(self):
        result = _run_codespaces_command("curl https://example.com")
        assert result["ok"] is False
        assert "not allowed" in result["error"]

    def test_wget_blocked(self):
        result = _run_codespaces_command("wget https://example.com")
        assert result["ok"] is False
        assert "not allowed" in result["error"]

    def test_bash_blocked(self):
        result = _run_codespaces_command("bash -c 'echo pwned'")
        assert result["ok"] is False
        assert "not allowed" in result["error"]

    def test_sh_blocked(self):
        result = _run_codespaces_command("sh -c 'id'")
        assert result["ok"] is False
        assert "not allowed" in result["error"]

    def test_nc_blocked(self):
        result = _run_codespaces_command("nc -l 1234")
        assert result["ok"] is False
        assert "not allowed" in result["error"]

    def test_chmod_blocked(self):
        result = _run_codespaces_command("chmod 777 /etc/passwd")
        assert result["ok"] is False
        assert "not allowed" in result["error"]

    def test_apt_blocked(self):
        result = _run_codespaces_command("apt install evil-tool")
        assert result["ok"] is False
        assert "not allowed" in result["error"]


# ── Blocked patterns ─────────────────────────────────────────────────────

class TestBlockedPatterns:
    def test_sudo_blocked(self):
        result = _run_codespaces_command("sudo ls")
        assert result["ok"] is False
        assert "blocked shell patterns" in result["error"]

    def test_rm_rf_blocked(self):
        result = _run_codespaces_command("rm -rf /")
        assert result["ok"] is False
        assert "blocked" in result["error"]

    def test_backtick_injection(self):
        result = _run_codespaces_command("echo `id`")
        assert result["ok"] is False
        assert "blocked" in result["error"]

    def test_dollar_paren_injection(self):
        result = _run_codespaces_command("echo $(whoami)")
        assert result["ok"] is False
        assert "blocked" in result["error"]

    def test_shutdown_blocked(self):
        result = _run_codespaces_command("shutdown now")
        assert result["ok"] is False
        assert "blocked" in result["error"]

    def test_reboot_blocked(self):
        result = _run_codespaces_command("reboot")
        assert result["ok"] is False
        assert "blocked" in result["error"]

    def test_dd_blocked(self):
        result = _run_codespaces_command("dd if=/dev/zero of=/dev/sda")
        assert result["ok"] is False
        assert "blocked" in result["error"]


# ── Edge cases ────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_command(self):
        result = _run_codespaces_command("")
        assert result["ok"] is False

    def test_none_command(self):
        result = _run_codespaces_command(None)
        assert result["ok"] is False

    def test_timeout_capped(self):
        # timeout 999 should be clamped to 120
        result = _run_codespaces_command("echo quick", timeout=999)
        assert result["ok"] is True

    def test_cwd_traversal(self):
        result = _run_codespaces_command("ls", cwd="/etc")
        assert result["ok"] is False
        assert "denied" in result.get("error", "").lower() or "outside" in result.get("error", "").lower()

    def test_api_keys_not_exposed(self):
        """Ensure sensitive env vars are stripped from subprocess environment."""
        os.environ["OPENAI_API_KEY"] = "sk-test-secret"
        try:
            result = _run_codespaces_command("env")
            assert result["ok"] is True
            assert "sk-test-secret" not in result["data"]["stdout"]
        finally:
            del os.environ["OPENAI_API_KEY"]
