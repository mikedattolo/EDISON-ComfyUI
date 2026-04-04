"""
api_system_awareness.py — FastAPI router for EDISON self-inspection.

Gives the assistant (and human users) the ability to understand its own
architecture, modules, endpoints, services, and runtime state.
"""
from __future__ import annotations

import importlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/system", tags=["system-awareness"])

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── Capability registry ──────────────────────────────────────────────────────
# Static map of EDISON's known subsystems.  The assistant consults this when
# answering "what can you do?" or "which endpoint handles X?" questions.

_CAPABILITY_MAP: Dict[str, Dict[str, Any]] = {
    "chat": {
        "description": "Conversational AI with multi-mode routing (chat, reasoning, code, agent, work, swarm)",
        "endpoints": ["/chat", "/chat/stream"],
        "frontend_page": "index.html",
        "backend_module": "services/edison_core/app.py",
    },
    "openai_compat": {
        "description": "OpenAI-compatible /v1/chat/completions endpoint for drop-in client support",
        "endpoints": ["/v1/chat/completions", "/v1/models"],
        "backend_module": "services/edison_core/app.py",
    },
    "image_generation": {
        "description": "Text-to-image via ComfyUI FLUX pipeline",
        "endpoints": ["/generate-image", "/gallery", "/gallery/images"],
        "frontend_page": "gallery.html",
        "backend_module": "services/edison_core/app.py",
    },
    "branding": {
        "description": "Client branding packages — logos, palettes, slogans, style guides",
        "endpoints": ["/api/branding/clients", "/api/branding/clients/{client_id}"],
        "storage_file": "config/integrations/branding.json",
        "frontend_page": "branding.html",
    },
    "printing": {
        "description": "3D printer management, slicing, and job tracking",
        "endpoints": ["/api/printers", "/api/printers/discover", "/api/printers/{id}/print"],
        "storage_file": "config/integrations/printers.json",
        "frontend_page": "printing.html",
    },
    "video": {
        "description": "Video editing, trimming, storyboard generation",
        "endpoints": ["/api/video/upload", "/api/video/trim", "/api/video/storyboard"],
        "frontend_page": "video.html",
    },
    "connectors": {
        "description": "External service integrations (GitHub, Slack, Gmail, etc.)",
        "endpoints": ["/api/connectors", "/api/connectors/{id}/test"],
        "storage_file": "config/integrations/connectors.json",
        "frontend_page": "connectors.html",
    },
    "file_manager": {
        "description": "File upload, gallery browsing, and asset organization",
        "endpoints": ["/upload", "/files", "/gallery/images"],
        "frontend_page": "files.html",
    },
    "search": {
        "description": "Web search (DuckDuckGo) and RAG retrieval",
        "endpoints": ["/search"],
        "backend_module": "services/edison_core/app.py",
    },
    "knowledge_base": {
        "description": "Long-term memory / knowledge learning from documents",
        "endpoints": ["/api/knowledge/learn", "/api/knowledge/search"],
    },
    "settings": {
        "description": "User preferences, model selection, system diagnostics",
        "endpoints": ["/settings", "/api/readiness"],
        "frontend_page": "settings.html",
    },
    "projects": {
        "description": "Client/project management with tasks, assets, and deliverables",
        "endpoints": ["/api/projects", "/api/projects/{id}", "/api/clients"],
        "frontend_page": "projects.html",
    },
    "runtime": {
        "description": "Extracted runtime modules — routing, tools, context, quality, model resolver",
        "backend_module": "services/edison_core/runtime/",
        "modules": [
            "routing_runtime", "tool_runtime", "context_runtime", "task_runtime",
            "artifact_runtime", "workspace_runtime", "model_runtime",
            "quality_runtime", "response_runtime", "chat_runtime",
            "search_runtime", "browser_runtime",
        ],
    },
}


def _discover_api_routes() -> List[Dict[str, str]]:
    """Dynamically list all registered FastAPI routes (lazy import of app)."""
    try:
        # Avoid circular import — only used at request time
        from services.edison_core.app import app as _app
        routes = []
        for route in _app.routes:
            methods = getattr(route, "methods", None)
            path = getattr(route, "path", None)
            if path and methods:
                for m in sorted(methods):
                    routes.append({"method": m, "path": path})
        return routes
    except Exception as exc:
        logger.warning(f"Could not discover routes: {exc}")
        return []


def _discover_frontend_pages() -> List[Dict[str, str]]:
    """List HTML pages in the web/ directory."""
    web_dir = REPO_ROOT / "web"
    pages = []
    if web_dir.is_dir():
        for f in sorted(web_dir.glob("*.html")):
            pages.append({"file": f.name, "path": f"/web/{f.name}"})
    return pages


def _discover_config_files() -> List[Dict[str, str]]:
    """List config & integration JSON files."""
    results = []
    config_dir = REPO_ROOT / "config"
    if config_dir.is_dir():
        for f in sorted(config_dir.rglob("*")):
            if f.is_file() and f.suffix in (".json", ".yaml", ".yml"):
                results.append({"file": str(f.relative_to(REPO_ROOT)), "size": f.stat().st_size})
    return results


def _discover_env_vars() -> Dict[str, Optional[str]]:
    """Return EDISON-relevant environment variables (values redacted for secrets)."""
    relevant = [
        "EDISON_MODE", "EDISON_HOST", "EDISON_PORT",
        "EDISON_CLIENTS_DIR", "EDISON_BACKUP_DIR",
        "COMFYUI_URL", "COMFYUI_OUTPUT_DIR",
        "MODEL_REPO_ROOT", "VLLM_URL",
        "QDRANT_URL", "QDRANT_COLLECTION",
    ]
    result = {}
    for key in relevant:
        val = os.environ.get(key)
        if val is not None:
            # Redact anything that looks like a token/secret
            if any(s in key.lower() for s in ("token", "secret", "key", "password")):
                result[key] = "***"
            else:
                result[key] = val
        else:
            result[key] = None
    return result


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/capabilities")
async def get_capabilities():
    """Return the full capability map — used by the assistant for self-awareness."""
    return {"capabilities": _CAPABILITY_MAP}


@router.get("/routes")
async def get_routes():
    """Return all registered API routes."""
    return {"routes": _discover_api_routes()}


@router.get("/pages")
async def get_pages():
    """Return all frontend HTML pages."""
    return {"pages": _discover_frontend_pages()}


@router.get("/config-files")
async def get_config_files():
    """Return config/integration file list."""
    return {"files": _discover_config_files()}


@router.get("/environment")
async def get_environment():
    """Return relevant environment variables (secrets redacted)."""
    return {"environment": _discover_env_vars()}


@router.get("/runtime-modules")
async def get_runtime_modules():
    """Return list of runtime modules and their status."""
    modules = _CAPABILITY_MAP.get("runtime", {}).get("modules", [])
    status = {}
    for mod_name in modules:
        full_name = f"services.edison_core.runtime.{mod_name}"
        try:
            m = importlib.import_module(full_name)
            status[mod_name] = {"loaded": True, "path": getattr(m, "__file__", "?")}
        except Exception as exc:
            status[mod_name] = {"loaded": False, "error": str(exc)}
    return {"runtime_modules": status}


@router.get("/health")
async def system_health():
    """Aggregated health check combining system state + model readiness."""
    result: Dict[str, Any] = {"status": "ok"}
    try:
        from services.state.system_state import get_system_state
        snap = get_system_state()
        result["system"] = {
            "gpus": [{"name": g.name, "memory_used_mb": g.memory_used_mb, "memory_total_mb": g.memory_total_mb} for g in snap.gpus],
            "disks": [{"path": d.path, "used_gb": round(d.used_bytes / 1e9, 1), "total_gb": round(d.total_bytes / 1e9, 1)} for d in snap.disks],
            "comfyui_reachable": snap.comfyui_reachable,
            "loaded_models": snap.loaded_models,
            "recent_errors": snap.recent_errors[-3:],
        }
    except Exception as exc:
        result["system"] = {"error": str(exc)}
    return result


@router.get("/code-search")
async def code_search(query: str, max_results: int = 10):
    """Safe grep-style search across EDISON source files (read-only).

    Searches .py and .js files for the given string/pattern.
    """
    results = []
    allowed_extensions = {".py", ".js", ".html", ".yaml", ".yml", ".json", ".md"}
    search_dirs = [
        REPO_ROOT / "services",
        REPO_ROOT / "web",
        REPO_ROOT / "config",
    ]
    pattern = re.compile(re.escape(query), re.IGNORECASE)

    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for fpath in search_dir.rglob("*"):
            if len(results) >= max_results:
                break
            if not fpath.is_file() or fpath.suffix not in allowed_extensions:
                continue
            # Skip large files (>500KB) and binary
            if fpath.stat().st_size > 500_000:
                continue
            try:
                text = fpath.read_text(errors="replace")
                for lineno, line in enumerate(text.splitlines(), 1):
                    if pattern.search(line):
                        results.append({
                            "file": str(fpath.relative_to(REPO_ROOT)),
                            "line": lineno,
                            "text": line.strip()[:200],
                        })
                        if len(results) >= max_results:
                            break
            except Exception:
                continue

    return {"query": query, "results": results, "count": len(results)}


@router.get("/inspect-file")
async def inspect_file(path: str, start_line: int = 1, end_line: int = 50):
    """Safely read a section of an EDISON source file.

    Restricted to files inside the repository root with allowed extensions.
    """
    allowed_extensions = {".py", ".js", ".html", ".yaml", ".yml", ".json", ".md", ".css", ".txt"}

    # Normalize and validate path
    target = (REPO_ROOT / path).resolve()
    if not str(target).startswith(str(REPO_ROOT)):
        return {"error": "Path traversal not allowed"}
    if not target.is_file():
        return {"error": "File not found"}
    if target.suffix not in allowed_extensions:
        return {"error": f"File type {target.suffix} not allowed"}
    if target.stat().st_size > 2_000_000:
        return {"error": "File too large (>2MB)"}

    start_line = max(1, start_line)
    end_line = max(start_line, min(end_line, start_line + 200))  # Cap at 200 lines

    try:
        lines = target.read_text(errors="replace").splitlines()
        section = lines[start_line - 1:end_line]
        return {
            "file": str(target.relative_to(REPO_ROOT)),
            "start_line": start_line,
            "end_line": min(end_line, len(lines)),
            "total_lines": len(lines),
            "content": "\n".join(section),
        }
    except Exception as exc:
        return {"error": str(exc)}
