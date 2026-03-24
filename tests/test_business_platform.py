import os
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_project_workspace_manager_crud_and_client_linking(tmp_path):
    from services.edison_core.contracts import ProjectCreateRequest, ProjectUpdateRequest
    from services.edison_core.projects import ProjectWorkspaceManager

    repo_root = tmp_path / "repo"
    branding_db = repo_root / "config" / "integrations" / "branding.json"
    branding_db.parent.mkdir(parents=True, exist_ok=True)
    branding_db.write_text(
        """
{
  "clients": [
    {
      "id": "client_adoro",
      "name": "Adoro Pizza",
      "business_name": "Adoro Pizza",
      "slug": "adoro-pizza",
      "tags": ["restaurant"]
    }
  ]
}
        """.strip()
    )

    manager = ProjectWorkspaceManager(
        repo_root=repo_root,
        config={"projects": {"root": "outputs"}},
        branding_db_path=branding_db,
    )

    created = manager.create_project(ProjectCreateRequest(
        name="Spring Launch Campaign",
        description="Brand refresh and promo rollout",
        client_id="client_adoro",
        service_types=["branding", "marketing", "video"],
        tags=["launch", "social"],
        tasks=[{"id": "task_1", "title": "Draft brand voice", "status": "pending"}],
    ))

    assert created.client_name == "Adoro Pizza"
    assert created.client_slug == "adoro-pizza"
    assert created.root_path.endswith(created.project_id)
    assert (repo_root / "outputs" / "projects" / created.project_id / "branding").exists()
    assert (repo_root / "outputs" / "projects" / created.project_id / "deliverables").exists()

    fetched = manager.get_project(created.project_id)
    assert fetched is not None
    assert fetched.name == "Spring Launch Campaign"
    assert fetched.service_types == ["branding", "marketing", "video"]

    updated = manager.update_project(created.project_id, ProjectUpdateRequest(
        status="active",
        deliverables=[{"id": "deliv_1", "name": "Logo concepts", "status": "approved"}],
    ))
    assert updated is not None
    assert updated.status == "active"
    assert updated.deliverables[0]["status"] == "approved"

    status = manager.get_project_status(created.project_id)
    assert status is not None
    assert status["task_counts"]["total"] == 1
    assert status["deliverable_counts"]["approved"] == 1

    by_client = manager.get_projects_by_client("client_adoro")
    assert len(by_client) == 1
    assert by_client[0].project_id == created.project_id

    deleted = manager.delete_project(created.project_id)
    assert deleted is True
    assert manager.get_project(created.project_id) is None


def test_capability_map_discovers_pages_routes_and_storage(tmp_path):
    from services.edison_core.system_awareness import build_capability_map

    repo_root = tmp_path / "repo"
    (repo_root / "web").mkdir(parents=True)
    (repo_root / "services" / "edison_core" / "routes").mkdir(parents=True)
    (repo_root / "config" / "integrations").mkdir(parents=True)
    (repo_root / "web" / "index.html").write_text("<html></html>")
    (repo_root / "web" / "projects.html").write_text("<html></html>")
    (repo_root / "services" / "edison_core" / "app.py").write_text(
        "@app.get('/awareness/system')\nasync def awareness():\n    return {}\n"
    )
    (repo_root / "services" / "edison_core" / "routes" / "business_platform.py").write_text(
        "@router.get('/projects')\nasync def projects():\n    return {}\n"
    )
    (repo_root / "config" / "integrations" / "branding.json").write_text('{"clients": []}')
    (repo_root / "config" / "integrations" / "connectors.json").write_text('{"connectors": []}')
    (repo_root / "config" / "integrations" / "printers.json").write_text('{"printers": []}')

    capabilities = build_capability_map(repo_root, {"projects": {"root": "outputs"}, "modes": {"chat": {}, "work": {}}})

    page_routes = {page["route"] for page in capabilities["pages"]}
    route_paths = {route["path"] for route in capabilities["routes"]}

    assert "/" in page_routes
    assert "/projects" in page_routes
    assert "/projects" in route_paths
    assert any(item["path"].endswith("branding.json") for item in capabilities["storage"])
    assert capabilities["summary"]["page_count"] >= 2
    assert "browser" in capabilities["tools"]


def test_business_router_exposes_projects_and_capabilities(monkeypatch, tmp_path):
    from services.edison_core.contracts import ProjectCreateRequest
    from services.edison_core.projects import ProjectWorkspaceManager
    from services.edison_core.routes import business_platform

    repo_root = tmp_path / "repo"
    branding_db = repo_root / "config" / "integrations" / "branding.json"
    branding_db.parent.mkdir(parents=True, exist_ok=True)
    branding_db.write_text('{"clients": []}')
    manager = ProjectWorkspaceManager(repo_root, {"projects": {"root": "outputs"}}, branding_db)
    created = manager.create_project(ProjectCreateRequest(name="Ops Dashboard"))

    monkeypatch.setattr(business_platform, "get_project_manager", lambda: manager)
    monkeypatch.setattr(business_platform, "get_capability_map", lambda: {"summary": {"page_count": 3, "route_count": 7}})
    monkeypatch.setattr(business_platform, "_load_json", lambda name, default: default)

    app = FastAPI()
    app.include_router(business_platform.router)
    client = TestClient(app)

    projects_response = client.get("/projects")
    assert projects_response.status_code == 200
    assert projects_response.json()["projects"][0]["project_id"] == created.project_id

    status_response = client.get(f"/projects/{created.project_id}/status")
    assert status_response.status_code == 200
    assert status_response.json()["status"]["name"] == "Ops Dashboard"

    capabilities_response = client.get("/system-awareness/capabilities")
    assert capabilities_response.status_code == 200
    assert capabilities_response.json()["capabilities"]["summary"]["route_count"] == 7