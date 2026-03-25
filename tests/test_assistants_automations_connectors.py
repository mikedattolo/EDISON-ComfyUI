import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _configure_temp_integrations(monkeypatch, core_app, tmp_path):
    repo_root = tmp_path / "repo"
    integrations_dir = repo_root / "config" / "integrations"
    branding_root = repo_root / "outputs" / "branding"
    backup_root = repo_root / "backups"
    connectors_db = integrations_dir / "connectors.json"
    prompts_db = integrations_dir / "prompts.json"

    monkeypatch.setattr(core_app, "REPO_ROOT", repo_root)
    monkeypatch.setattr(core_app, "INTEGRATIONS_DIR", integrations_dir)
    monkeypatch.setattr(core_app, "BRANDING_ROOT", branding_root)
    monkeypatch.setattr(core_app, "SELF_EDIT_BACKUP_DIR", backup_root)
    monkeypatch.setattr(core_app, "CONNECTORS_DB", connectors_db)
    monkeypatch.setattr(core_app, "PROMPTS_DB", prompts_db)

    return repo_root, integrations_dir, connectors_db, prompts_db


def test_assistant_profiles_round_trip_and_prompt_injection(monkeypatch, tmp_path):
    from services.edison_core import app as core_app

    _configure_temp_integrations(monkeypatch, core_app, tmp_path)

    payload = asyncio.run(core_app.upsert_assistant({
        "name": "Brand Director",
        "description": "Luxury restaurant brand strategist",
        "system_prompt": "Focus on premium positioning and concise direction.",
        "default_mode": "chat",
        "starter_prompts": ["Create a brand voice", "Write a slogan"],
    }))

    assistant_id = payload["assistant"]["id"]
    assistant = core_app._resolve_assistant_profile(assistant_id)
    prompt = core_app.build_system_prompt("chat", assistant_profile=assistant)

    assert assistant["name"] == "Brand Director"
    assert "custom assistant 'Brand Director'" in prompt
    assert "premium positioning" in prompt


def test_easy_connect_builds_custom_header_connector(monkeypatch, tmp_path):
    from services.edison_core import app as core_app

    _configure_temp_integrations(monkeypatch, core_app, tmp_path)

    payload = asyncio.run(core_app.easy_connect_connector({
        "provider": "custom",
        "name": "inventory_api",
        "base_url": "https://inventory.example.com/api",
        "auth_type": "custom-header",
        "header_name": "X-Inventory-Key",
        "token": "secret-token",
    }))

    connectors = core_app._load_connectors()["connectors"]
    connector = next(item for item in connectors if item["name"] == "inventory_api")

    assert payload["ok"] is True
    assert connector["headers"]["X-Inventory-Key"] == "secret-token"
    assert connector["base_url"] == "https://inventory.example.com/api"


def test_automation_matches_message_and_calls_connector(monkeypatch, tmp_path):
    from services.edison_core import app as core_app

    _configure_temp_integrations(monkeypatch, core_app, tmp_path)

    asyncio.run(core_app.upsert_connector({
        "name": "orders_api",
        "provider": "custom",
        "base_url": "https://orders.example.com",
        "headers": {"Authorization": "Bearer token"},
    }))

    asyncio.run(core_app.upsert_automation({
        "name": "Latest Orders",
        "connector_name": "orders_api",
        "trigger_phrases": ["sync latest orders"],
        "method": "POST",
        "path": "/orders/search",
        "body_template": {"query": "{{message}}"},
        "response_template": "Automation '{{automation_name}}' completed.",
    }))

    calls = []

    def fake_call(connector, method, path, body=None):
        calls.append((connector["name"], method, path, body))
        return {"ok": True, "status_code": 200, "body": {"count": 3}}

    monkeypatch.setattr(core_app, "_call_connector_http", fake_call)

    result = core_app._maybe_execute_automation("please sync latest orders for me")

    assert result["mode_used"] == "automation"
    assert result["automation"]["name"] == "Latest Orders"
    assert calls == [
        ("orders_api", "POST", "/orders/search", {"query": "please sync latest orders for me"})
    ]
    assert 'Automation \"' not in result["response"]