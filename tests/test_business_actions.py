import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_branding_store_supports_richer_client_fields(tmp_path):
    from services.edison_core.branding_store import BrandingClientStore

    repo_root = tmp_path / "repo"
    branding_root = repo_root / "outputs" / "clients"
    branding_db = repo_root / "config" / "integrations" / "branding.json"

    store = BrandingClientStore(
        repo_root=repo_root,
        branding_root=branding_root,
        branding_db_path=branding_db,
        media_roots=[repo_root / "outputs"],
    )

    created = store.create_client({
        "business_name": "Adoro Pizza",
        "contact_person": "Mia",
        "email": "mia@adoro.test",
        "phone": "555-1000",
        "website": "https://adoro.test",
        "industry": "restaurant",
        "notes": "Needs signage and campaign assets",
        "tags": "food, local, pizza",
    })

    assert created["created"] is True
    client = created["client"]
    assert client["business_name"] == "Adoro Pizza"
    assert client["contact_person"] == "Mia"
    assert client["email"] == "mia@adoro.test"
    assert client["industry"] == "restaurant"
    assert client["tags"] == ["food", "local", "pizza"]
    assert (branding_root / "adoro-pizza" / "images").exists()
    assert (branding_root / "adoro-pizza" / "videos").exists()
    assert (branding_root / "adoro-pizza" / "files").exists()

    updated = store.update_client(client["id"], {
        "phone": "555-2000",
        "website": "https://new.adoro.test",
        "tags": ["food", "campaign"],
    })

    assert updated is not None
    assert updated["phone"] == "555-2000"
    assert updated["website"] == "https://new.adoro.test"
    assert updated["tags"] == ["food", "campaign"]


def test_business_actions_execute_client_project_branding_and_marketing_flows(tmp_path):
    from services.edison_core.branding_store import BrandingClientStore
    from services.edison_core.business_actions import execute_business_action
    from services.edison_core.projects import ProjectWorkspaceManager

    repo_root = tmp_path / "repo"
    branding_root = repo_root / "outputs" / "clients"
    branding_db = repo_root / "config" / "integrations" / "branding.json"
    branding_db.parent.mkdir(parents=True, exist_ok=True)

    store = BrandingClientStore(
        repo_root=repo_root,
        branding_root=branding_root,
        branding_db_path=branding_db,
        media_roots=[repo_root / "outputs"],
    )
    manager = ProjectWorkspaceManager(
        repo_root=repo_root,
        config={"projects": {"root": "outputs"}, "modes": {"chat": {}, "work": {}}},
        branding_db_path=branding_db,
    )

    client_action = execute_business_action(
        message="create a branding client for Adoro Pizza",
        repo_root=repo_root,
        config={"projects": {"root": "outputs"}, "modes": {"chat": {}, "work": {}}},
        branding_store=store,
        project_manager=manager,
    )
    assert client_action is not None
    assert client_action["business_action"]["type"] == "create_client"

    project_action = execute_business_action(
        message="create a project for Adoro Pizza called Spring Launch Campaign",
        repo_root=repo_root,
        config={"projects": {"root": "outputs"}, "modes": {"chat": {}, "work": {}}},
        branding_store=store,
        project_manager=manager,
    )
    assert project_action is not None
    assert project_action["business_action"]["type"] == "create_project"
    project_id = project_action["business_action"]["project"]["project_id"]

    branding_action = execute_business_action(
        message="make a branding package for Adoro Pizza for Spring Launch Campaign",
        repo_root=repo_root,
        config={"projects": {"root": "outputs"}, "modes": {"chat": {}, "work": {}}},
        branding_store=store,
        project_manager=manager,
    )
    assert branding_action is not None
    assert branding_action["business_action"]["type"] == "branding_package"
    branding_result = branding_action["business_action"]["result"]
    assert branding_result["project_id"] == project_id
    assert len(branding_result["outputs"]) >= 5

    project = manager.get_project(project_id)
    assert project is not None
    assert any(item["category"] == "logo_concepts" for item in project.deliverables)

    marketing_action = execute_business_action(
        message="generate social captions and ad copy for Adoro Pizza for Spring Launch Campaign",
        repo_root=repo_root,
        config={"projects": {"root": "outputs"}, "modes": {"chat": {}, "work": {}}},
        branding_store=store,
        project_manager=manager,
    )
    assert marketing_action is not None
    assert marketing_action["business_action"]["type"] == "marketing_copy"
    marketing_result = marketing_action["business_action"]["result"]
    assert marketing_result["project_id"] == project_id
    assert any(output["title"] == "social_captions" for output in marketing_result["outputs"])


def test_business_actions_auto_routes_simple_cad_request_to_rhino_node(tmp_path):
    from services.edison_core.branding_store import BrandingClientStore
    from services.edison_core.business_actions import execute_business_action
    from services.edison_core.projects import ProjectWorkspaceManager

    class FakeNodeManager:
        def __init__(self):
            self.commands = []

        def mark_stale(self):
            return None

        def list_nodes(self):
            return {
                "nodes": [
                    {
                        "id": "cad-laptop",
                        "name": "CAD-Laptop",
                        "status": "online",
                        "capabilities": ["cad", "3d-modeling", "rhino"],
                        "software": {"rhino": {"version": "7"}},
                    }
                ]
            }

        def find_best_node_for_task(self, task_description, required_capabilities=None, preferred_software=None):
            return self.list_nodes()["nodes"][0]

        def send_command(self, node_id, command, params=None):
            self.commands.append({"node_id": node_id, "command": command, "params": params or {}})
            return {
                "ok": True,
                "node_id": node_id,
                "response": {
                    "ok": True,
                    "output_paths": (params or {}).get("output_paths", []),
                },
            }

    repo_root = tmp_path / "repo"
    branding_root = repo_root / "outputs" / "clients"
    branding_db = repo_root / "config" / "integrations" / "branding.json"
    branding_db.parent.mkdir(parents=True, exist_ok=True)

    store = BrandingClientStore(
        repo_root=repo_root,
        branding_root=branding_root,
        branding_db_path=branding_db,
        media_roots=[repo_root / "outputs"],
    )
    manager = ProjectWorkspaceManager(
        repo_root=repo_root,
        config={"projects": {"root": "outputs"}, "modes": {"chat": {}, "work": {}}},
        branding_db_path=branding_db,
    )
    node_manager = FakeNodeManager()

    action = execute_business_action(
        message="Edison generate a 3d model of a vase on CAD Laptop",
        repo_root=repo_root,
        config={"projects": {"root": "outputs"}, "modes": {"chat": {}, "work": {}}},
        branding_store=store,
        project_manager=manager,
        node_manager=node_manager,
    )

    assert action is not None
    assert action["business_action"]["type"] == "node_model_request"
    assert action["business_action"]["ok"] is True
    assert node_manager.commands[0]["node_id"] == "cad-laptop"
    assert node_manager.commands[0]["command"] == "rhino_script"
    assert "AddRevSrf" in node_manager.commands[0]["params"]["script_content"]
    assert action["business_action"]["output_paths"] == node_manager.commands[0]["params"]["output_paths"]


def test_business_actions_falls_back_to_queued_node_task_when_direct_dispatch_fails(tmp_path):
    from services.edison_core.branding_store import BrandingClientStore
    from services.edison_core.business_actions import execute_business_action
    from services.edison_core.projects import ProjectWorkspaceManager

    class FakeNodeManager:
        def __init__(self):
            self.commands = []
            self.delegations = []

        def mark_stale(self):
            return None

        def list_nodes(self):
            return {
                "nodes": [
                    {
                        "id": "cad-laptop",
                        "name": "CAD-Laptop",
                        "status": "online",
                        "capabilities": ["cad", "3d-modeling", "rhino"],
                        "software": {"rhino": {"version": "7"}},
                    }
                ]
            }

        def find_best_node_for_task(self, task_description, required_capabilities=None, preferred_software=None):
            return self.list_nodes()["nodes"][0]

        def send_command(self, node_id, command, params=None):
            self.commands.append({"node_id": node_id, "command": command, "params": params or {}})
            return {"ok": False, "error": "HTTPConnectionPool(host=192.168.1.50, port=9200): timed out"}

        def delegate_task(self, task_description, task_type, payload, required_capabilities=None, preferred_software=None, node_id=None):
            task = {"id": "task_queued_123", "node_id": node_id, "task_type": task_type, "payload": payload, "status": "pending"}
            self.delegations.append({
                "task_description": task_description,
                "task_type": task_type,
                "payload": payload,
                "required_capabilities": required_capabilities,
                "preferred_software": preferred_software,
                "node_id": node_id,
            })
            return {"ok": True, "node": self.list_nodes()["nodes"][0], "task": task}

    repo_root = tmp_path / "repo"
    branding_root = repo_root / "outputs" / "clients"
    branding_db = repo_root / "config" / "integrations" / "branding.json"
    branding_db.parent.mkdir(parents=True, exist_ok=True)

    store = BrandingClientStore(
        repo_root=repo_root,
        branding_root=branding_root,
        branding_db_path=branding_db,
        media_roots=[repo_root / "outputs"],
    )
    manager = ProjectWorkspaceManager(
        repo_root=repo_root,
        config={"projects": {"root": "outputs"}, "modes": {"chat": {}, "work": {}}},
        branding_db_path=branding_db,
    )
    node_manager = FakeNodeManager()

    action = execute_business_action(
        message="Edison generate a 3d model of a vase",
        repo_root=repo_root,
        config={"projects": {"root": "outputs"}, "modes": {"chat": {}, "work": {}}},
        branding_store=store,
        project_manager=manager,
        node_manager=node_manager,
    )

    assert action is not None
    assert action["business_action"]["type"] == "node_model_request"
    assert action["business_action"]["ok"] is True
    assert action["business_action"]["status"] == "queued"
    assert node_manager.commands[0]["command"] == "rhino_script"
    assert node_manager.delegations[0]["task_type"] == "rhino_script"
    assert node_manager.delegations[0]["node_id"] == "cad-laptop"
    assert action["business_action"]["task"]["id"] == "task_queued_123"