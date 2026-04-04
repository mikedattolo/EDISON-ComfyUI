"""
api_projects.py — FastAPI router for client & project CRUD.

Provides /api/clients and /api/projects endpoints for managing
business relationships, projects, tasks, and deliverables.

Storage: JSON files under config/projects/ (migration-safe, backward-compatible).
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["projects"])

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECTS_DIR = REPO_ROOT / "config" / "projects"
CLIENTS_FILE = PROJECTS_DIR / "clients.json"
PROJECTS_FILE = PROJECTS_DIR / "projects.json"


def _ensure_storage():
    """Create storage directory and seed files if needed."""
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    for fpath in (CLIENTS_FILE, PROJECTS_FILE):
        if not fpath.exists():
            fpath.write_text("[]", encoding="utf-8")


def _load_json(fpath: Path) -> list:
    _ensure_storage()
    try:
        data = json.loads(fpath.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_json(fpath: Path, data: list):
    _ensure_storage()
    fpath.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ── Pydantic models ──────────────────────────────────────────────────────────

class ClientCreate(BaseModel):
    business_name: str = Field(..., min_length=1, max_length=200)
    contact_person: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    industry: Optional[str] = None
    notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class ClientUpdate(BaseModel):
    business_name: Optional[str] = None
    contact_person: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    industry: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None


class ProjectCreate(BaseModel):
    client_id: Optional[str] = None
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    service_types: List[str] = Field(
        default_factory=lambda: ["mixed"],
        description="branding, printing, video, marketing, mixed"
    )
    due_date: Optional[str] = None
    status: str = Field(default="planning", description="planning, active, review, completed, archived")
    tags: List[str] = Field(default_factory=list)


class ProjectUpdate(BaseModel):
    client_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    service_types: Optional[List[str]] = None
    due_date: Optional[str] = None
    status: Optional[str] = None
    tags: Optional[List[str]] = None


class TaskCreate(BaseModel):
    project_id: str
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    assigned_to: Optional[str] = None
    status: str = Field(default="pending", description="pending, in_progress, review, done")
    priority: str = Field(default="normal", description="low, normal, high, urgent")


class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    assigned_to: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None


# ── Client endpoints ──────────────────────────────────────────────────────────

@router.get("/api/clients")
async def list_clients(industry: Optional[str] = None, tag: Optional[str] = None):
    """List all clients, optionally filtered by industry or tag."""
    clients = _load_json(CLIENTS_FILE)
    if industry:
        clients = [c for c in clients if (c.get("industry") or "").lower() == industry.lower()]
    if tag:
        clients = [c for c in clients if tag.lower() in [t.lower() for t in c.get("tags", [])]]
    return {"clients": clients, "total": len(clients)}


@router.post("/api/clients")
async def create_client(body: ClientCreate):
    """Create a new client."""
    clients = _load_json(CLIENTS_FILE)

    # Check for duplicate business name
    for c in clients:
        if c.get("business_name", "").lower() == body.business_name.lower():
            raise HTTPException(status_code=409, detail=f"Client '{body.business_name}' already exists")

    client = {
        "id": str(uuid.uuid4()),
        "business_name": body.business_name,
        "contact_person": body.contact_person,
        "email": body.email,
        "phone": body.phone,
        "website": body.website,
        "industry": body.industry,
        "notes": body.notes,
        "tags": body.tags,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    clients.append(client)
    _save_json(CLIENTS_FILE, clients)
    logger.info(f"Created client: {client['id']} ({body.business_name})")
    return {"client": client}


@router.get("/api/clients/{client_id}")
async def get_client(client_id: str):
    """Get a single client by ID."""
    clients = _load_json(CLIENTS_FILE)
    client = next((c for c in clients if c.get("id") == client_id), None)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    return {"client": client}


@router.put("/api/clients/{client_id}")
async def update_client(client_id: str, body: ClientUpdate):
    """Update a client's fields."""
    clients = _load_json(CLIENTS_FILE)
    idx = next((i for i, c in enumerate(clients) if c.get("id") == client_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="Client not found")

    updates = body.dict(exclude_none=True)
    for key, val in updates.items():
        clients[idx][key] = val
    clients[idx]["updated_at"] = time.time()

    _save_json(CLIENTS_FILE, clients)
    return {"client": clients[idx]}


@router.delete("/api/clients/{client_id}")
async def delete_client(client_id: str):
    """Delete a client and their associated projects."""
    clients = _load_json(CLIENTS_FILE)
    before = len(clients)
    clients = [c for c in clients if c.get("id") != client_id]
    if len(clients) == before:
        raise HTTPException(status_code=404, detail="Client not found")
    _save_json(CLIENTS_FILE, clients)

    # Also remove associated projects
    projects = _load_json(PROJECTS_FILE)
    projects = [p for p in projects if p.get("client_id") != client_id]
    _save_json(PROJECTS_FILE, projects)

    logger.info(f"Deleted client: {client_id}")
    return {"deleted": True}


# ── Project endpoints ─────────────────────────────────────────────────────────

@router.get("/api/projects")
async def list_projects(
    client_id: Optional[str] = None,
    status: Optional[str] = None,
    service_type: Optional[str] = None,
):
    """List all projects, optionally filtered."""
    projects = _load_json(PROJECTS_FILE)
    if client_id:
        projects = [p for p in projects if p.get("client_id") == client_id]
    if status:
        projects = [p for p in projects if p.get("status") == status]
    if service_type:
        projects = [p for p in projects if service_type in p.get("service_types", [])]
    return {"projects": projects, "total": len(projects)}


@router.post("/api/projects")
async def create_project(body: ProjectCreate):
    """Create a new project, optionally linked to a client."""
    # Validate client exists if provided
    if body.client_id:
        clients = _load_json(CLIENTS_FILE)
        if not any(c.get("id") == body.client_id for c in clients):
            raise HTTPException(status_code=404, detail=f"Client {body.client_id} not found")

    projects = _load_json(PROJECTS_FILE)
    project = {
        "id": str(uuid.uuid4()),
        "client_id": body.client_id,
        "name": body.name,
        "description": body.description,
        "service_types": body.service_types,
        "due_date": body.due_date,
        "status": body.status,
        "tags": body.tags,
        "tasks": [],
        "assets": [],
        "deliverables": [],
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    projects.append(project)
    _save_json(PROJECTS_FILE, projects)
    logger.info(f"Created project: {project['id']} ({body.name})")
    return {"project": project}


@router.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    """Get a single project by ID."""
    projects = _load_json(PROJECTS_FILE)
    project = next((p for p in projects if p.get("id") == project_id), None)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"project": project}


@router.put("/api/projects/{project_id}")
async def update_project(project_id: str, body: ProjectUpdate):
    """Update a project's fields."""
    projects = _load_json(PROJECTS_FILE)
    idx = next((i for i, p in enumerate(projects) if p.get("id") == project_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="Project not found")

    updates = body.dict(exclude_none=True)
    for key, val in updates.items():
        projects[idx][key] = val
    projects[idx]["updated_at"] = time.time()

    _save_json(PROJECTS_FILE, projects)
    return {"project": projects[idx]}


@router.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project."""
    projects = _load_json(PROJECTS_FILE)
    before = len(projects)
    projects = [p for p in projects if p.get("id") != project_id]
    if len(projects) == before:
        raise HTTPException(status_code=404, detail="Project not found")
    _save_json(PROJECTS_FILE, projects)
    logger.info(f"Deleted project: {project_id}")
    return {"deleted": True}


# ── Task sub-endpoints (nested under projects) ───────────────────────────────

@router.post("/api/projects/{project_id}/tasks")
async def create_task(project_id: str, body: TaskCreate):
    """Add a task to a project."""
    projects = _load_json(PROJECTS_FILE)
    idx = next((i for i, p in enumerate(projects) if p.get("id") == project_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="Project not found")

    task = {
        "id": str(uuid.uuid4()),
        "title": body.title,
        "description": body.description,
        "assigned_to": body.assigned_to,
        "status": body.status,
        "priority": body.priority,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    if "tasks" not in projects[idx]:
        projects[idx]["tasks"] = []
    projects[idx]["tasks"].append(task)
    projects[idx]["updated_at"] = time.time()
    _save_json(PROJECTS_FILE, projects)
    return {"task": task}


@router.get("/api/projects/{project_id}/tasks")
async def list_tasks(project_id: str):
    """List tasks for a project."""
    projects = _load_json(PROJECTS_FILE)
    project = next((p for p in projects if p.get("id") == project_id), None)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    tasks = project.get("tasks", [])
    return {"tasks": tasks, "total": len(tasks)}


@router.put("/api/projects/{project_id}/tasks/{task_id}")
async def update_task(project_id: str, task_id: str, body: TaskUpdate):
    """Update a task in a project."""
    projects = _load_json(PROJECTS_FILE)
    proj_idx = next((i for i, p in enumerate(projects) if p.get("id") == project_id), None)
    if proj_idx is None:
        raise HTTPException(status_code=404, detail="Project not found")

    tasks = projects[proj_idx].get("tasks", [])
    task_idx = next((i for i, t in enumerate(tasks) if t.get("id") == task_id), None)
    if task_idx is None:
        raise HTTPException(status_code=404, detail="Task not found")

    updates = body.dict(exclude_none=True)
    for key, val in updates.items():
        tasks[task_idx][key] = val
    tasks[task_idx]["updated_at"] = time.time()
    projects[proj_idx]["tasks"] = tasks
    projects[proj_idx]["updated_at"] = time.time()
    _save_json(PROJECTS_FILE, projects)
    return {"task": tasks[task_idx]}


@router.delete("/api/projects/{project_id}/tasks/{task_id}")
async def delete_task(project_id: str, task_id: str):
    """Delete a task from a project."""
    projects = _load_json(PROJECTS_FILE)
    proj_idx = next((i for i, p in enumerate(projects) if p.get("id") == project_id), None)
    if proj_idx is None:
        raise HTTPException(status_code=404, detail="Project not found")

    tasks = projects[proj_idx].get("tasks", [])
    before = len(tasks)
    tasks = [t for t in tasks if t.get("id") != task_id]
    if len(tasks) == before:
        raise HTTPException(status_code=404, detail="Task not found")

    projects[proj_idx]["tasks"] = tasks
    projects[proj_idx]["updated_at"] = time.time()
    _save_json(PROJECTS_FILE, projects)
    return {"deleted": True}


# ── Asset linkage ─────────────────────────────────────────────────────────────

@router.post("/api/projects/{project_id}/assets")
async def add_asset(project_id: str, body: dict):
    """Link an asset (file path, URL, artifact ID) to a project."""
    projects = _load_json(PROJECTS_FILE)
    idx = next((i for i, p in enumerate(projects) if p.get("id") == project_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="Project not found")

    asset = {
        "id": str(uuid.uuid4()),
        "type": body.get("type", "file"),
        "path": body.get("path", ""),
        "label": body.get("label", ""),
        "created_at": time.time(),
    }
    if "assets" not in projects[idx]:
        projects[idx]["assets"] = []
    projects[idx]["assets"].append(asset)
    projects[idx]["updated_at"] = time.time()
    _save_json(PROJECTS_FILE, projects)
    return {"asset": asset}


@router.get("/api/projects/{project_id}/assets")
async def list_assets(project_id: str):
    """List assets for a project."""
    projects = _load_json(PROJECTS_FILE)
    project = next((p for p in projects if p.get("id") == project_id), None)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    assets = project.get("assets", [])
    return {"assets": assets, "total": len(assets)}
