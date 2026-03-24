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