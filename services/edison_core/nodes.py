"""EDISON Node Manager — distributed worker-node registration & orchestration.

Mirrors the printer-manager pattern: nodes self-register over the LAN,
the main EDISON instance tracks them, and tasks can be dispatched.
"""

from __future__ import annotations

import json
import logging
import socket
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


# ── Data Models ──────────────────────────────────────────────────────────

@dataclass
class NodeProfile:
    id: str
    name: str
    host: str
    port: int = 9200
    role: str = "general"            # general | cad | render | compute | print
    os: str = ""                     # windows | linux | macos
    cpu: str = ""
    ram_gb: int = 0
    gpu: str = ""
    gpu_vram_gb: int = 0
    capabilities: List[str] = field(default_factory=list)  # rhino, blender, comfyui…
    software: Dict[str, str] = field(default_factory=dict)  # {"rhino": "7", …}
    enabled: bool = True
    status: str = "offline"          # online | offline | busy | error
    last_seen: float = 0.0          # epoch timestamp


@dataclass
class NodeTask:
    id: str
    node_id: str
    task_type: str                  # rhino_command, file_transfer, shell, heartbeat
    payload: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"         # pending | running | done | failed
    result: Dict[str, Any] = field(default_factory=dict)
    created: float = 0.0
    completed: float = 0.0


# ── Node Manager ─────────────────────────────────────────────────────────

class NodeManager:
    """Manages registered worker nodes and dispatches tasks to them."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._ensure_db()

    # ── persistence ──

    def _ensure_db(self):
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._db_path.exists():
            self._db_path.write_text(json.dumps({"nodes": [], "tasks": []}, indent=2))

    def _load(self) -> Dict[str, Any]:
        try:
            return json.loads(self._db_path.read_text())
        except Exception:
            return {"nodes": [], "tasks": []}

    def _save(self, data: Dict[str, Any]):
        self._db_path.write_text(json.dumps(data, indent=2))

    # ── node CRUD ──

    def list_nodes(self) -> Dict[str, Any]:
        db = self._load()
        return {"nodes": db.get("nodes", [])}

    def get_node(self, node_id: str) -> Dict[str, Any]:
        db = self._load()
        node = next((n for n in db.get("nodes", []) if n.get("id") == node_id), None)
        if node is None:
            raise KeyError(f"Node '{node_id}' not found")
        return node

    def register_node(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Register or update a worker node (called by the node agent)."""
        db = self._load()
        node_id = str(payload.get("id") or "").strip()
        if not node_id:
            node_id = f"node_{uuid.uuid4().hex[:8]}"

        profile = NodeProfile(
            id=node_id,
            name=str(payload.get("name") or node_id),
            host=str(payload.get("host") or ""),
            port=int(payload.get("port", 9200)),
            role=str(payload.get("role") or "general"),
            os=str(payload.get("os") or ""),
            cpu=str(payload.get("cpu") or ""),
            ram_gb=int(payload.get("ram_gb", 0)),
            gpu=str(payload.get("gpu") or ""),
            gpu_vram_gb=int(payload.get("gpu_vram_gb", 0)),
            capabilities=list(payload.get("capabilities") or []),
            software=dict(payload.get("software") or {}),
            enabled=bool(payload.get("enabled", True)),
            status="online",
            last_seen=time.time(),
        )

        nodes = db.get("nodes", [])
        existing = next((n for n in nodes if n.get("id") == node_id), None)
        serialized = asdict(profile)
        if existing:
            existing.update(serialized)
        else:
            nodes.append(serialized)

        db["nodes"] = nodes
        self._save(db)
        return serialized

    def heartbeat(self, node_id: str, status_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update last_seen and optional status info for a node."""
        db = self._load()
        node = next((n for n in db.get("nodes", []) if n.get("id") == node_id), None)
        if node is None:
            raise KeyError(f"Node '{node_id}' not found")
        node["last_seen"] = time.time()
        node["status"] = "online"
        if status_info:
            for k in ("cpu_usage", "ram_usage", "gpu_usage"):
                if k in status_info:
                    node[k] = status_info[k]
        self._save(db)
        return {"ok": True, "node_id": node_id}

    def remove_node(self, node_id: str) -> Dict[str, Any]:
        db = self._load()
        before = len(db.get("nodes", []))
        db["nodes"] = [n for n in db.get("nodes", []) if n.get("id") != node_id]
        self._save(db)
        removed = len(db["nodes"]) < before
        return {"ok": removed, "node_id": node_id}

    def mark_stale(self, timeout_sec: float = 120.0):
        """Mark nodes as offline if they haven't checked in recently."""
        db = self._load()
        now = time.time()
        changed = False
        for n in db.get("nodes", []):
            if n.get("status") == "online" and (now - n.get("last_seen", 0)) > timeout_sec:
                n["status"] = "offline"
                changed = True
        if changed:
            self._save(db)

    # ── task dispatch ──

    def submit_task(self, node_id: str, task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Queue a task for a specific node."""
        db = self._load()
        node = next((n for n in db.get("nodes", []) if n.get("id") == node_id), None)
        if node is None:
            raise KeyError(f"Node '{node_id}' not found")

        task = NodeTask(
            id=f"task_{uuid.uuid4().hex[:12]}",
            node_id=node_id,
            task_type=task_type,
            payload=payload,
            status="pending",
            created=time.time(),
        )

        tasks = db.get("tasks", [])
        tasks.append(asdict(task))
        db["tasks"] = tasks
        self._save(db)
        return asdict(task)

    def get_pending_tasks(self, node_id: str) -> List[Dict[str, Any]]:
        """Return pending tasks for a node (polled by the agent)."""
        db = self._load()
        return [
            t for t in db.get("tasks", [])
            if t.get("node_id") == node_id and t.get("status") == "pending"
        ]

    def update_task(self, task_id: str, status: str, result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update a task's status and result."""
        db = self._load()
        task = next((t for t in db.get("tasks", []) if t.get("id") == task_id), None)
        if task is None:
            raise KeyError(f"Task '{task_id}' not found")
        task["status"] = status
        if result:
            task["result"] = result
        if status in ("done", "failed"):
            task["completed"] = time.time()
        self._save(db)
        return task

    # ── network discovery ──

    def discover_nodes(self, subnet: str = "", timeout_sec: float = 0.3, max_hosts: int = 64) -> Dict[str, Any]:
        """Scan the LAN for running EDISON node agents (port 9200)."""
        import ipaddress

        target_subnet = (subnet or "").strip()
        if not target_subnet:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
            base = ".".join(local_ip.split(".")[:3])
            target_subnet = f"{base}.0/24"

        net = ipaddress.ip_network(target_subnet, strict=False)
        agent_port = 9200
        discovered: List[Dict[str, Any]] = []
        scanned = 0

        for host in list(net.hosts())[:max(1, max_hosts)]:
            scanned += 1
            host_str = str(host)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout_sec)
            try:
                if sock.connect_ex((host_str, agent_port)) == 0:
                    # Port open — try to identify as an EDISON node
                    try:
                        resp = requests.get(
                            f"http://{host_str}:{agent_port}/info",
                            timeout=2.0,
                        )
                        if resp.ok:
                            info = resp.json()
                            info["host"] = host_str
                            discovered.append(info)
                            continue
                    except Exception:
                        pass
                    discovered.append({
                        "host": host_str,
                        "port": agent_port,
                        "name": f"Unknown Agent {host_str}",
                        "status": "detected",
                    })
            finally:
                sock.close()

        return {
            "subnet": target_subnet,
            "scanned_hosts": scanned,
            "discovered": discovered,
        }

    # ── remote command dispatch ──

    def send_command(self, node_id: str, command: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a command directly to a node agent's HTTP API."""
        node = self.get_node(node_id)
        host = node.get("host")
        port = node.get("port", 9200)
        if not host:
            raise ValueError(f"Node '{node_id}' has no host configured")

        url = f"http://{host}:{port}/execute"
        payload = {"command": command, "params": params or {}}

        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            return {"ok": True, "node_id": node_id, "response": resp.json()}
        except requests.RequestException as e:
            return {"ok": False, "node_id": node_id, "error": str(e)}

    # ── capability matching & delegation ──

    # Maps task intent keywords → capability tags the node must have
    CAPABILITY_MAP: Dict[str, List[str]] = {
        # 3D CAD / modeling
        "rhino": ["rhino"],
        "grasshopper": ["grasshopper"],
        "solidworks": ["solidworks"],
        "blender": ["blender"],
        "cad": ["cad"],
        "3d_model": ["3d-modeling"],
        "3d_modeling": ["3d-modeling"],
        "nurbs": ["nurbs"],
        "parametric": ["parametric-design", "cad"],
        "mechanical": ["mechanical-cad"],
        "sheet_metal": ["sheet-metal"],
        "assembly": ["assembly-design"],
        "render": ["rendering", "gpu-render"],
        "animation": ["animation"],
        # Compute
        "gpu": ["gpu-compute", "cuda"],
        "cuda": ["cuda"],
        # Generic
        "compute": ["gpu-compute"],
    }

    def find_nodes_with_capability(self, *capabilities: str) -> List[Dict[str, Any]]:
        """Return all online enabled nodes that have ALL of the given capabilities."""
        self.mark_stale()
        db = self._load()
        result = []
        for node in db.get("nodes", []):
            if not node.get("enabled", True):
                continue
            if node.get("status") != "online":
                continue
            node_caps = set(node.get("capabilities", []))
            node_sw   = set((node.get("software") or {}).keys())
            combined  = node_caps | node_sw
            if all(cap in combined for cap in capabilities):
                result.append(node)
        return result

    def find_best_node_for_task(
        self,
        task_description: str,
        required_capabilities: Optional[List[str]] = None,
        preferred_software: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Given a natural-language task description and optional explicit capability
        requirements, find the best available online node.

        Returns the best matching node dict, or None if no suitable node is online.
        """
        self.mark_stale()
        db = self._load()
        nodes = [n for n in db.get("nodes", [])
                 if n.get("enabled", True) and n.get("status") == "online"]
        if not nodes:
            return None

        # Build a set of required capabilities from the description
        required = set(required_capabilities or [])
        preferred = set(sw.lower() for sw in (preferred_software or []))

        desc_lower = task_description.lower()
        for keyword, caps in self.CAPABILITY_MAP.items():
            if keyword.replace("_", " ") in desc_lower or keyword in desc_lower:
                required.update(caps)

        def score(node: Dict[str, Any]) -> int:
            node_caps = set(node.get("capabilities", []))
            node_sw   = set((node.get("software") or {}).keys())
            combined  = node_caps | node_sw
            s = sum(1 for r in required if r in combined)
            s += sum(2 for p in preferred if p in combined)   # bonus for preferred SW
            # Prefer less busy nodes
            cpu = node.get("cpu_usage", 50)
            if cpu < 40:
                s += 1
            return s

        candidates = sorted(nodes, key=score, reverse=True)
        best = candidates[0]

        # Require at least minimal match when requirements were derived
        if required:
            node_caps = set(best.get("capabilities", []))
            node_sw = set((best.get("software") or {}).keys())
            combined = node_caps | node_sw
            if not any(r in combined for r in required):
                return None

        return best

    def delegate_task(
        self,
        task_description: str,
        task_type: str,
        payload: Dict[str, Any],
        required_capabilities: Optional[List[str]] = None,
        preferred_software: Optional[List[str]] = None,
        node_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Intelligently delegate a task to the best available node.

        If node_id is given, use that node directly.
        Otherwise, find the best node based on task_description and capabilities.
        Returns a dict with node info and task info.
        """
        if node_id:
            target_node = self.get_node(node_id)
        else:
            target_node = self.find_best_node_for_task(
                task_description,
                required_capabilities=required_capabilities,
                preferred_software=preferred_software,
            )

        if target_node is None:
            return {
                "ok": False,
                "error": "No suitable online node found for this task. "
                         "Make sure the node agent is running on your remote machine.",
            }

        task = self.submit_task(target_node["id"], task_type, payload)
        return {
            "ok": True,
            "node": {
                "id": target_node["id"],
                "name": target_node.get("name"),
                "host": target_node.get("host"),
                "capabilities": target_node.get("capabilities", []),
                "software": target_node.get("software", {}),
            },
            "task": task,
        }

    def list_tasks(
        self,
        node_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List tasks, optionally filtered by node and/or status."""
        db = self._load()
        tasks = db.get("tasks", [])
        if node_id:
            tasks = [t for t in tasks if t.get("node_id") == node_id]
        if status:
            tasks = [t for t in tasks if t.get("status") == status]
        # Return most-recent first
        tasks.sort(key=lambda t: t.get("created", 0), reverse=True)
        return tasks[:limit]
