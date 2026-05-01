from services.edison_core.nodes import NodeManager


def test_dispatch_command_falls_back_to_queue(tmp_path, monkeypatch):
    manager = NodeManager(tmp_path / "nodes.json")
    manager.register_node({
        "id": "cad-laptop",
        "name": "CAD Laptop",
        "host": "192.168.1.50",
        "port": 9200,
        "capabilities": ["cad", "rhino"],
        "software": {"rhino": "7"},
    })

    monkeypatch.setattr(
        manager,
        "send_command",
        lambda node_id, command, params=None: {"ok": False, "node_id": node_id, "error": "connect timed out"},
    )

    result = manager.dispatch_command("cad-laptop", "ping", {})

    assert result["ok"] is True
    assert result["status"] == "queued"
    assert result["task"]["node_id"] == "cad-laptop"
    assert result["task"]["task_type"] == "ping"
    assert result["direct_error"] == "connect timed out"
    assert manager.list_tasks(node_id="cad-laptop", limit=10)[0]["status"] == "pending"


def test_dispatch_command_returns_direct_success_without_queuing(tmp_path, monkeypatch):
    manager = NodeManager(tmp_path / "nodes.json")
    manager.register_node({
        "id": "cad-laptop",
        "name": "CAD Laptop",
        "host": "192.168.1.50",
        "port": 9200,
        "capabilities": ["cad", "rhino"],
        "software": {"rhino": "7"},
    })

    monkeypatch.setattr(
        manager,
        "send_command",
        lambda node_id, command, params=None: {
            "ok": True,
            "node_id": node_id,
            "response": {"ok": True, "pong": True},
        },
    )

    result = manager.dispatch_command("cad-laptop", "ping", {})

    assert result["ok"] is True
    assert result["status"] == "completed"
    assert result["response"]["pong"] is True
    assert manager.list_tasks(node_id="cad-laptop", limit=10) == []