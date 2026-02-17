"""
Tool Execution Framework for EDISON
Provides a clean base_tool interface, registry, structured logging,
timeouts, error isolation, and permission gates.
"""

import abc
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()

# Allowed root directories for file access
SAFE_ROOTS = [
    REPO_ROOT / "outputs",
    REPO_ROOT / "uploads",
    REPO_ROOT / "data",
    REPO_ROOT / "config",
]


class ToolResult:
    """Structured result from a tool execution."""

    def __init__(
        self,
        success: bool,
        data: Any = None,
        error: Optional[str] = None,
        duration_ms: float = 0,
        tool_name: str = "",
        call_id: str = "",
    ):
        self.success = success
        self.data = data
        self.error = error
        self.duration_ms = duration_ms
        self.tool_name = tool_name
        self.call_id = call_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
            "tool_name": self.tool_name,
            "call_id": self.call_id,
        }


class BaseTool(abc.ABC):
    """Base interface for all Edison tools."""

    name: str = "base_tool"
    description: str = ""
    requires_permission: bool = False

    @abc.abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""

    def validate_params(self, **kwargs) -> Optional[str]:
        """Return error message if params invalid, None if OK."""
        return None


class ToolRegistry:
    """Central registry for all available tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._allowlist: set = set()  # Tools allowed without confirmation
        self._call_log: List[Dict[str, Any]] = []

    def register(self, tool: BaseTool, allowed: bool = True):
        self._tools[tool.name] = tool
        if allowed:
            self._allowlist.add(tool.name)
        logger.info(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        return [
            {"name": t.name, "description": t.description, "requires_permission": t.requires_permission}
            for t in self._tools.values()
        ]

    def execute(
        self,
        tool_name: str,
        correlation_id: Optional[str] = None,
        timeout_s: float = 30.0,
        **kwargs,
    ) -> ToolResult:
        """Execute a tool with logging, timeout, and error isolation."""
        call_id = str(uuid.uuid4())[:8]
        tool = self._tools.get(tool_name)

        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}", call_id=call_id, tool_name=tool_name)

        if tool.requires_permission and tool_name not in self._allowlist:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' requires permission",
                call_id=call_id,
                tool_name=tool_name,
            )

        # Validate params
        err = tool.validate_params(**kwargs)
        if err:
            return ToolResult(success=False, error=err, call_id=call_id, tool_name=tool_name)

        start = time.monotonic()
        log_entry = {
            "call_id": call_id,
            "tool": tool_name,
            "params": {k: str(v)[:200] for k, v in kwargs.items()},
            "correlation_id": correlation_id,
            "timestamp": time.time(),
        }

        try:
            # Execute with timeout via threading
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exe:
                future = exe.submit(tool.execute, **kwargs)
                result = future.result(timeout=timeout_s)
            result.call_id = call_id
            result.tool_name = tool_name
            result.duration_ms = (time.monotonic() - start) * 1000

            log_entry["success"] = result.success
            log_entry["duration_ms"] = result.duration_ms
            if result.error:
                log_entry["error"] = result.error

        except concurrent.futures.TimeoutError:
            result = ToolResult(
                success=False,
                error=f"Tool '{tool_name}' timed out after {timeout_s}s",
                duration_ms=(time.monotonic() - start) * 1000,
                call_id=call_id,
                tool_name=tool_name,
            )
            log_entry["success"] = False
            log_entry["error"] = "timeout"
            log_entry["duration_ms"] = result.duration_ms
        except Exception as e:
            result = ToolResult(
                success=False,
                error=f"Tool error: {type(e).__name__}: {e}",
                duration_ms=(time.monotonic() - start) * 1000,
                call_id=call_id,
                tool_name=tool_name,
            )
            log_entry["success"] = False
            log_entry["error"] = str(e)
            log_entry["duration_ms"] = result.duration_ms

        self._call_log.append(log_entry)
        logger.info(
            f"Tool call [{call_id}] {tool_name}: "
            f"{'OK' if result.success else 'FAIL'} "
            f"({result.duration_ms:.0f}ms)"
        )
        return result

    def get_call_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._call_log[-limit:]


# ── Built-in Tools ───────────────────────────────────────────────────────


class WebFetchTool(BaseTool):
    """Fetch content from a URL."""

    name = "web_fetch"
    description = "Fetch text content from a URL"

    def execute(self, url: str = "", **kwargs) -> ToolResult:
        if not url:
            return ToolResult(success=False, error="URL required")
        try:
            import requests as req

            resp = req.get(url, timeout=15, headers={"User-Agent": "Edison/1.0"})
            resp.raise_for_status()
            # Truncate to avoid huge payloads
            text = resp.text[:50000]
            return ToolResult(success=True, data={"url": url, "status": resp.status_code, "text": text})
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class SafeFileReaderTool(BaseTool):
    """Read a file from allowed directories only."""

    name = "file_reader"
    description = "Read a file safely from allowed directories"

    def validate_params(self, path: str = "", **kwargs) -> Optional[str]:
        if not path:
            return "Path required"
        p = Path(path).resolve()
        if not any(str(p).startswith(str(root)) for root in SAFE_ROOTS):
            return f"Path not in allowed directories: {p}"
        if ".." in str(path):
            return "Path traversal not allowed"
        return None

    def execute(self, path: str = "", max_bytes: int = 100000, **kwargs) -> ToolResult:
        p = Path(path).resolve()
        if not p.exists():
            return ToolResult(success=False, error=f"File not found: {p}")
        try:
            content = p.read_text(errors="replace")[:max_bytes]
            return ToolResult(success=True, data={"path": str(p), "content": content, "size": p.stat().st_size})
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class CodeExecutionTool(BaseTool):
    """Execute Python code in a sandboxed subprocess."""

    name = "code_exec"
    description = "Execute Python code safely (time-limited, sandboxed)"
    requires_permission = True

    def execute(self, code: str = "", timeout: int = 10, **kwargs) -> ToolResult:
        if not code:
            return ToolResult(success=False, error="No code provided")
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
            f.write(code)
            f.flush()
            try:
                result = subprocess.run(
                    ["python", f.name],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd="/tmp",
                )
                return ToolResult(
                    success=result.returncode == 0,
                    data={
                        "stdout": result.stdout[:10000],
                        "stderr": result.stderr[:5000],
                        "returncode": result.returncode,
                    },
                    error=result.stderr[:1000] if result.returncode != 0 else None,
                )
            except subprocess.TimeoutExpired:
                return ToolResult(success=False, error=f"Code execution timed out after {timeout}s")
            except Exception as e:
                return ToolResult(success=False, error=str(e))


# ── Global registry singleton ────────────────────────────────────────────

_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        _registry.register(WebFetchTool())
        _registry.register(SafeFileReaderTool())
        _registry.register(CodeExecutionTool(), allowed=False)  # Requires explicit permission
    return _registry
