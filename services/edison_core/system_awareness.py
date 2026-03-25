"""
System awareness helpers for Edison self-inspection and capability mapping.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import os
import re

from .model_catalog import build_model_catalog


DEFAULT_TOOL_CATEGORIES = {
    "browser": ["open_sandbox_browser", "browser.create_session", "browser.navigate"],
    "content_generation": ["generate_image", "generate_video", "generate_music"],
    "code_execution": ["execute_python", "codespace_exec"],
    "data_access": ["list_files", "read_file", "rag_search", "web_search"],
    "fabrication": ["list_printers", "send_3d_print", "get_printer_status"],
}


def build_capability_map(repo_root: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    repo_root = repo_root.resolve()
    web_dir = repo_root / "web"
    app_path = repo_root / "services" / "edison_core" / "app.py"
    routes_dir = repo_root / "services" / "edison_core" / "routes"
    integrations_dir = repo_root / "config" / "integrations"

    pages = _discover_pages(web_dir)
    routes = _discover_routes([app_path] + sorted(routes_dir.glob("*.py")))
    storage = _discover_storage(repo_root, config)
    service_modules = _discover_service_modules(repo_root / "services")
    environment = _discover_environment(config)
    readiness = _discover_readiness(repo_root)
    model_catalog = build_model_catalog(repo_root, config)

    return {
        "summary": {
            "page_count": len(pages),
            "route_count": len(routes),
            "service_module_count": len(service_modules),
            "storage_count": len(storage),
            "installed_llm_models": model_catalog.get("summary", {}).get("llm_installed", 0),
            "installed_image_checkpoints": model_catalog.get("summary", {}).get("image_checkpoints_installed", 0),
        },
        "pages": pages,
        "routes": routes,
        "core_service_modules": service_modules,
        "storage": storage,
        "integrations": _discover_integrations(integrations_dir),
        "environment": environment,
        "modes": _discover_modes(config),
        "tools": DEFAULT_TOOL_CATEGORIES,
        "models": model_catalog,
        "readiness": readiness,
    }


def _discover_pages(web_dir: Path) -> List[Dict[str, Any]]:
    route_map = {
        "index.html": "/",
        "branding.html": "/branding",
        "connectors.html": "/connectors",
        "printing.html": "/printing",
        "projects.html": "/projects",
        "video_editor.html": "/video-editor",
    }
    pages = []
    for page in sorted(web_dir.glob("*.html")):
        pages.append({
            "file": page.name,
            "route": route_map.get(page.name, f"/{page.stem}"),
            "title": page.stem.replace("_", " ").title(),
        })
    return pages


def _discover_routes(files: List[Path]) -> List[Dict[str, Any]]:
    pattern = re.compile(r"@(app|router)\.(?:get|post|put|delete|patch|api_route)\((?:\s*)[\"']([^\"']+)[\"']")
    routes: List[Dict[str, Any]] = []
    seen = set()
    for file_path in files:
        if not file_path.exists():
            continue
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        for _target, route in pattern.findall(text):
            if route in seen:
                continue
            seen.add(route)
            category = route.strip("/").split("/", 1)[0] if route.strip("/") else "root"
            routes.append({
                "path": route,
                "category": category or "root",
                "source": file_path.name,
            })
    return sorted(routes, key=lambda item: item["path"])


def _discover_storage(repo_root: Path, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    projects_root = Path(config.get("projects", {}).get("root", "outputs"))
    if not projects_root.is_absolute():
        projects_root = (repo_root / projects_root).resolve()
    if projects_root.name != "projects":
        projects_root = projects_root / "projects"

    candidates = [
        repo_root / "config" / "integrations" / "branding.json",
        repo_root / "config" / "integrations" / "connectors.json",
        repo_root / "config" / "integrations" / "printers.json",
        repo_root / "config" / "integrations" / "prompts.json",
        projects_root,
        repo_root / "outputs" / "clients",
        repo_root / "outputs" / "gallery",
    ]
    storage = []
    for path in candidates:
        storage.append({
            "path": _relative_or_absolute(path, repo_root),
            "exists": path.exists(),
            "kind": "directory" if path.suffix == "" else "file",
        })
    return storage


def _discover_integrations(integrations_dir: Path) -> Dict[str, Any]:
    branding = _load_json(integrations_dir / "branding.json", {"clients": []})
    connectors = _load_json(integrations_dir / "connectors.json", {"connectors": []})
    printers = _load_json(integrations_dir / "printers.json", {"printers": []})
    return {
        "branding_clients": len(branding.get("clients", [])),
        "connected_integrations": len(connectors.get("connectors", [])),
        "registered_printers": len(printers.get("printers", [])),
    }


def _discover_environment(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "important_variables": [
            "EDISON_CLIENTS_DIR",
            "EDISON_BACKUP_DIR",
            "EDISON_CORE_URL",
            "OAUTH_*",
        ],
        "project_root": config.get("projects", {}).get("root", "outputs"),
        "comfyui_host": config.get("comfyui", {}).get("host", "127.0.0.1"),
        "comfyui_port": config.get("comfyui", {}).get("port", 8188),
        "coral_enabled": bool(config.get("coral", {}).get("enabled", False)),
    }


def _discover_modes(config: Dict[str, Any]) -> Dict[str, Any]:
    configured_modes = sorted((config.get("modes") or {}).keys())
    return {
        "configured": configured_modes,
        "business_domains": ["branding", "marketing", "printing", "video", "connectors", "projects"],
    }


def _discover_service_modules(services_root: Path) -> List[Dict[str, Any]]:
    modules = []
    for py_file in sorted(services_root.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue
        modules.append({
            "module": py_file.stem,
            "path": str(py_file.relative_to(services_root.parent)),
        })
    return modules


def _discover_readiness(repo_root: Path) -> Dict[str, Any]:
    return {
        "config_present": (repo_root / "config" / "edison.yaml").exists(),
        "web_present": (repo_root / "web" / "index.html").exists(),
        "outputs_present": (repo_root / "outputs").exists(),
        "branding_storage_present": (repo_root / "config" / "integrations" / "branding.json").exists(),
        "workspace_path": str(repo_root),
        "python_env": os.environ.get("VIRTUAL_ENV") or "system",
    }


def _load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except Exception:
        return str(path.resolve())