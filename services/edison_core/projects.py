"""
Project workspaces and sandbox management scaffolding.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import uuid

from .contracts import ProjectCreateRequest, ProjectResponse


class ProjectWorkspaceManager:
    def __init__(self, repo_root: Path, config: Dict[str, Any]):
        self.repo_root = repo_root
        self.config = config
        self.projects_root = Path(config.get("projects", {}).get("root", "outputs")).resolve()
        if not self.projects_root.is_absolute():
            self.projects_root = (self.repo_root / self.projects_root).resolve()
        self.projects_root.mkdir(parents=True, exist_ok=True)

    def list_projects(self) -> List[ProjectResponse]:
        projects = []
        for project_dir in self.projects_root.glob("*"):
            meta_path = project_dir / "project.json"
            if meta_path.exists():
                data = json.loads(meta_path.read_text())
                projects.append(ProjectResponse(**data))
        return projects

    def create_project(self, req: ProjectCreateRequest) -> ProjectResponse:
        project_id = f"proj_{uuid.uuid4().hex[:8]}"
        project_dir = self.projects_root / project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        config_path = project_dir / "project.json"
        payload = {
            "project_id": project_id,
            "name": req.name,
            "description": req.description,
            "root_path": str(project_dir),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "config_path": str(config_path)
        }
        config_path.write_text(json.dumps(payload, indent=2))
        return ProjectResponse(**payload)

    def get_project(self, project_id: str) -> Optional[ProjectResponse]:
        project_dir = self.projects_root / project_id
        meta_path = project_dir / "project.json"
        if not meta_path.exists():
            return None
        data = json.loads(meta_path.read_text())
        return ProjectResponse(**data)
