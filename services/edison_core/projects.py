"""
Project workspace management for business-oriented Edison workflows.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, UTC
import json
import re
import shutil
import uuid

from .contracts import ProjectCreateRequest, ProjectResponse, ProjectUpdateRequest


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _model_dump(model: Any, **kwargs) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(**kwargs)
    return model.dict(**kwargs)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower()).strip("-")
    return slug or f"project-{uuid.uuid4().hex[:8]}"


class ProjectWorkspaceManager:
    def __init__(self, repo_root: Path, config: Dict[str, Any], branding_db_path: Optional[Path] = None):
        self.repo_root = repo_root.resolve()
        self.config = config or {}
        root_candidate = Path(self.config.get("projects", {}).get("root", "outputs"))
        if not root_candidate.is_absolute():
            root_candidate = (self.repo_root / root_candidate).resolve()
        self.projects_root = root_candidate if root_candidate.name == "projects" else (root_candidate / "projects")
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self.branding_db_path = (branding_db_path or (self.repo_root / "config" / "integrations" / "branding.json")).resolve()

    def list_projects(
        self,
        client_id: Optional[str] = None,
        status: Optional[str] = None,
        service: Optional[str] = None,
        query: Optional[str] = None,
    ) -> List[ProjectResponse]:
        projects: List[ProjectResponse] = []
        for project_dir in sorted(self.projects_root.glob("*")):
            meta_path = project_dir / "project.json"
            if not meta_path.exists():
                continue
            try:
                project = ProjectResponse(**self._load_project_payload(meta_path))
            except Exception:
                continue
            if client_id and project.client_id != client_id:
                continue
            if status and project.status != status:
                continue
            if service and service not in project.service_types:
                continue
            if query:
                haystack = " ".join([
                    project.name,
                    project.description or "",
                    project.client_name or "",
                    " ".join(project.tags),
                ]).lower()
                if query.lower() not in haystack:
                    continue
            projects.append(project)
        return sorted(projects, key=lambda item: item.created_at, reverse=True)

    def create_project(self, req: ProjectCreateRequest) -> ProjectResponse:
        project_id = f"proj_{uuid.uuid4().hex[:8]}"
        slug = self._unique_project_slug(req.name)
        project_dir = self.projects_root / project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        workspace_paths = self._create_workspace_dirs(project_dir)
        client = self._get_branding_client(req.client_id)
        config_path = project_dir / "project.json"
        now = _utc_now()
        payload = {
            "project_id": project_id,
            "name": req.name,
            "slug": slug,
            "description": req.description,
            "template": req.template,
            "client_id": client.get("id") if client else req.client_id,
            "client_name": (client.get("business_name") or client.get("name")) if client else None,
            "client_slug": client.get("slug") if client else None,
            "service_types": self._normalize_service_types(req.service_types),
            "due_date": req.due_date,
            "status": req.status,
            "notes": req.notes or "",
            "tags": self._normalize_tags(req.tags),
            "assets": req.assets,
            "tasks": req.tasks,
            "approvals": req.approvals,
            "deliverables": req.deliverables,
            "workspace_paths": workspace_paths,
            "root_path": str(project_dir.resolve()),
            "created_at": now,
            "updated_at": now,
            "config_path": str(config_path.resolve()),
        }
        self._write_json(config_path, payload)
        return ProjectResponse(**payload)

    def get_project(self, project_id: str) -> Optional[ProjectResponse]:
        meta_path = self._meta_path(project_id)
        if not meta_path.exists():
            return None
        return ProjectResponse(**self._load_project_payload(meta_path))

    def update_project(self, project_id: str, req: ProjectUpdateRequest) -> Optional[ProjectResponse]:
        meta_path = self._meta_path(project_id)
        if not meta_path.exists():
            return None

        payload = self._load_project_payload(meta_path)
        updates = _model_dump(req, exclude_unset=True)
        if "name" in updates and updates["name"]:
            payload["name"] = updates["name"]
            payload["slug"] = payload.get("slug") or self._unique_project_slug(updates["name"], exclude_project_id=project_id)
        if "client_id" in updates:
            client = self._get_branding_client(updates["client_id"])
            payload["client_id"] = updates["client_id"]
            payload["client_name"] = (client.get("business_name") or client.get("name")) if client else None
            payload["client_slug"] = client.get("slug") if client else None
        if "service_types" in updates and updates["service_types"] is not None:
            payload["service_types"] = self._normalize_service_types(updates["service_types"])
        if "tags" in updates and updates["tags"] is not None:
            payload["tags"] = self._normalize_tags(updates["tags"])
        for key in [
            "description",
            "template",
            "due_date",
            "status",
            "notes",
            "assets",
            "tasks",
            "approvals",
            "deliverables",
        ]:
            if key in updates:
                payload[key] = updates[key]
        payload["updated_at"] = _utc_now()
        self._write_json(meta_path, payload)
        return ProjectResponse(**payload)

    def delete_project(self, project_id: str) -> bool:
        project_dir = self.projects_root / project_id
        if not project_dir.exists():
            return False
        shutil.rmtree(project_dir, ignore_errors=True)
        return True

    def get_projects_by_client(self, client_id: str) -> List[ProjectResponse]:
        return self.list_projects(client_id=client_id)

    def get_project_with_client(self, project_id: str) -> Optional[Dict[str, Any]]:
        project = self.get_project(project_id)
        if not project:
            return None
        return {
            "project": _model_dump(project),
            "client": self._get_branding_client(project.client_id),
        }

    def get_project_status(self, project_id: str) -> Optional[Dict[str, Any]]:
        project = self.get_project(project_id)
        if not project:
            return None
        task_counts = {
            "total": len(project.tasks),
            "completed": len([task for task in project.tasks if task.get("status") == "completed"]),
            "pending": len([task for task in project.tasks if task.get("status") in {None, "pending", "in_progress"}]),
        }
        deliverable_counts = {
            "total": len(project.deliverables),
            "approved": len([item for item in project.deliverables if item.get("status") == "approved"]),
        }
        return {
            "project_id": project.project_id,
            "name": project.name,
            "status": project.status,
            "client_id": project.client_id,
            "client_name": project.client_name,
            "service_types": project.service_types,
            "task_counts": task_counts,
            "deliverable_counts": deliverable_counts,
            "asset_count": len(project.assets),
            "paths": project.workspace_paths,
            "updated_at": project.updated_at,
        }

    def register_output(
        self,
        project_id: str,
        path: str,
        category: str,
        title: str,
        status: str = "generated",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ProjectResponse]:
        meta_path = self._meta_path(project_id)
        if not meta_path.exists():
            return None
        payload = self._load_project_payload(meta_path)
        asset_entry = {
            "path": path,
            "category": category,
            "title": title,
            "status": status,
            "metadata": metadata or {},
        }
        if not any(item.get("path") == path for item in payload.get("assets", [])):
            payload.setdefault("assets", []).append(asset_entry)
        if not any(item.get("path") == path for item in payload.get("deliverables", [])):
            payload.setdefault("deliverables", []).append(asset_entry)
        payload["updated_at"] = _utc_now()
        self._write_json(meta_path, payload)
        return ProjectResponse(**payload)

    def _meta_path(self, project_id: str) -> Path:
        return self.projects_root / project_id / "project.json"

    def _load_project_payload(self, meta_path: Path) -> Dict[str, Any]:
        data = json.loads(meta_path.read_text())
        project_dir = meta_path.parent.resolve()
        data.setdefault("slug", _slugify(data.get("name") or meta_path.parent.name))
        data.setdefault("template", None)
        data.setdefault("client_id", None)
        data.setdefault("client_name", None)
        data.setdefault("client_slug", None)
        data.setdefault("service_types", [])
        data.setdefault("due_date", None)
        data.setdefault("status", "planned")
        data.setdefault("notes", "")
        data.setdefault("tags", [])
        data.setdefault("assets", [])
        data.setdefault("tasks", [])
        data.setdefault("approvals", [])
        data.setdefault("deliverables", [])
        data.setdefault("workspace_paths", self._default_workspace_paths(project_dir))
        data.setdefault("root_path", str(project_dir))
        data.setdefault("config_path", str(meta_path.resolve()))
        data.setdefault("updated_at", data.get("created_at"))
        return data

    def _create_workspace_dirs(self, project_dir: Path) -> Dict[str, str]:
        dirs = self._default_workspace_paths(project_dir)
        for folder in dirs.values():
            Path(folder).mkdir(parents=True, exist_ok=True)
        return dirs

    def _default_workspace_paths(self, project_dir: Path) -> Dict[str, str]:
        return {
            "planning": str((project_dir / "planning").resolve()),
            "assets": str((project_dir / "assets").resolve()),
            "branding": str((project_dir / "branding").resolve()),
            "marketing": str((project_dir / "marketing").resolve()),
            "printing": str((project_dir / "printing").resolve()),
            "video": str((project_dir / "video").resolve()),
            "deliverables": str((project_dir / "deliverables").resolve()),
        }

    def _load_branding_clients(self) -> List[Dict[str, Any]]:
        if not self.branding_db_path.exists():
            return []
        try:
            payload = json.loads(self.branding_db_path.read_text())
        except Exception:
            return []
        return payload.get("clients", [])

    def _get_branding_client(self, client_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not client_id:
            return None
        return next((client for client in self._load_branding_clients() if client.get("id") == client_id), None)

    def _unique_project_slug(self, name: str, exclude_project_id: Optional[str] = None) -> str:
        base_slug = _slugify(name)
        used = set()
        for project in self.list_projects():
            if exclude_project_id and project.project_id == exclude_project_id:
                continue
            used.add(project.slug)
        if base_slug not in used:
            return base_slug
        index = 2
        while True:
            candidate = f"{base_slug}-{index}"
            if candidate not in used:
                return candidate
            index += 1

    def _normalize_service_types(self, values: List[str]) -> List[str]:
        allowed = {"branding", "printing", "video", "marketing", "mixed"}
        normalized = []
        for value in values or []:
            lowered = str(value).strip().lower()
            if lowered in allowed and lowered not in normalized:
                normalized.append(lowered)
        return normalized

    def _normalize_tags(self, values: List[str]) -> List[str]:
        normalized = []
        for value in values or []:
            cleaned = str(value).strip()
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
        return normalized

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
