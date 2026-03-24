"""
Business-platform routes for projects, overview data, and capability inspection.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import json
import yaml

from fastapi import APIRouter, HTTPException, Query

from ..contracts import ProjectCreateRequest, ProjectUpdateRequest
from ..projects import ProjectWorkspaceManager
from ..system_awareness import build_capability_map

router = APIRouter(tags=["business-platform"])

REPO_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
CONFIG_PATH = REPO_ROOT / "config" / "edison.yaml"
INTEGRATIONS_DIR = REPO_ROOT / "config" / "integrations"


def _load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        payload = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    except Exception:
        return {}
    return payload.get("edison", payload)


def get_project_manager() -> ProjectWorkspaceManager:
    return ProjectWorkspaceManager(
        repo_root=REPO_ROOT,
        config=_load_config(),
        branding_db_path=INTEGRATIONS_DIR / "branding.json",
    )


def get_capability_map() -> Dict[str, Any]:
    return build_capability_map(REPO_ROOT, _load_config())


def _load_json(name: str, default: Dict[str, Any]) -> Dict[str, Any]:
    path = INTEGRATIONS_DIR / name
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _model_dump(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@router.get("/projects")
async def list_projects(
    client_id: Optional[str] = None,
    status: Optional[str] = None,
    service: Optional[str] = None,
    query: Optional[str] = None,
):
    manager = get_project_manager()
    projects = manager.list_projects(client_id=client_id, status=status, service=service, query=query)
    return {"ok": True, "projects": [_model_dump(project) for project in projects]}


@router.post("/projects")
async def create_project(request: ProjectCreateRequest):
    manager = get_project_manager()
    project = manager.create_project(request)
    return {"ok": True, "project": _model_dump(project), "created": True}


@router.get("/projects/{project_id}")
async def get_project(project_id: str):
    manager = get_project_manager()
    project = manager.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"ok": True, "project": _model_dump(project)}


@router.put("/projects/{project_id}")
async def update_project(project_id: str, request: ProjectUpdateRequest):
    manager = get_project_manager()
    project = manager.update_project(project_id, request)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"ok": True, "project": _model_dump(project), "updated": True}


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    manager = get_project_manager()
    deleted = manager.delete_project(project_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"ok": True, "deleted": True, "project_id": project_id}


@router.get("/projects/{project_id}/status")
async def get_project_status(project_id: str):
    manager = get_project_manager()
    status = manager.get_project_status(project_id)
    if not status:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"ok": True, "status": status}


@router.get("/projects/by-client/{client_id}")
async def list_projects_by_client(client_id: str):
    manager = get_project_manager()
    projects = manager.get_projects_by_client(client_id)
    return {"ok": True, "client_id": client_id, "projects": [_model_dump(project) for project in projects]}


@router.get("/business/overview")
async def business_overview(limit: int = Query(10, ge=1, le=100)):
    manager = get_project_manager()
    branding = _load_json("branding.json", {"clients": []})
    connectors = _load_json("connectors.json", {"connectors": []})
    printers = _load_json("printers.json", {"printers": []})
    projects = manager.list_projects()[:limit]
    projects_by_client = {}
    for project in manager.list_projects():
        if project.client_id:
            projects_by_client[project.client_id] = projects_by_client.get(project.client_id, 0) + 1

    clients = []
    for client in branding.get("clients", []):
        clients.append({
            "id": client.get("id"),
            "name": client.get("business_name") or client.get("name"),
            "slug": client.get("slug"),
            "project_count": projects_by_client.get(client.get("id"), 0),
            "tags": client.get("tags", []),
            "paths": client.get("paths", {}),
        })

    return {
        "ok": True,
        "clients": clients,
        "projects": [_model_dump(project) for project in projects],
        "counts": {
            "clients": len(clients),
            "projects": len(manager.list_projects()),
            "connectors": len(connectors.get("connectors", [])),
            "printers": len(printers.get("printers", [])),
        },
        "capabilities": get_capability_map().get("summary", {}),
    }


@router.get("/system-awareness/capabilities")
async def system_awareness_capabilities():
    return {"ok": True, "capabilities": get_capability_map()}