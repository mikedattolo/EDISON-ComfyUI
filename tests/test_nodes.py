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


def test_node_registration_accepts_rich_capabilities_and_dispatch_envelope(tmp_path):
    manager = NodeManager(tmp_path / "nodes.json")
    node = manager.register_node({
        "id": "win-cad",
        "name": "Windows CAD Node",
        "host": "192.168.1.60",
        "os": "windows",
        "ram_gb": 32,
        "gpus": [{"name": "RTX A3000", "vram_gb": 6}],
        "capabilities": ["cad", "rhino"],
        "installed_apps": {"rhino": "8", "solidworks": "2024"},
        "allowed_tools": ["rhino", "file_transfer"],
        "accepted_job_types": ["rhino_grasshopper"],
    })
    task = manager.submit_task("win-cad", "rhino_grasshopper", {"gh_path": "design.gh"})
    envelope = manager.build_dispatch_request(task, node)

    assert node["gpus"][0]["vram_gb"] == 6
    assert envelope["schema_version"] == 1
    assert envelope["node_capabilities"]["installed_apps"]["solidworks"] == "2024"
    assert envelope["task_type"] == "rhino_grasshopper"


def test_delegate_respects_node_accepted_job_types(tmp_path):
    manager = NodeManager(tmp_path / "nodes.json")
    manager.register_node({
        "id": "storage-node",
        "name": "Storage Node",
        "host": "192.168.1.61",
        "capabilities": ["indexing"],
        "accepted_job_types": ["index_files"],
    })

    result = manager.delegate_task("run a render", "gpu_render", {}, node_id="storage-node")

    assert result["ok"] is False
    assert "does not accept" in result["error"]
