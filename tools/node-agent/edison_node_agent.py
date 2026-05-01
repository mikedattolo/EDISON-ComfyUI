"""EDISON Node Agent — runs on remote worker machines.

This lightweight agent runs on your Windows/Linux/Mac workstation and:
  1. Auto-detects hardware (CPU, RAM, GPU)
  2. Registers with the main EDISON server
  3. Sends heartbeats every 30 seconds
  4. Polls for and executes tasks from EDISON
  5. Exposes a local HTTP API for direct commands

Rhino 7 Integration:
  If Rhino 7 is detected, the agent can relay commands to Rhino via
  its COM automation interface (Windows) or command-line.

Usage:
  python edison_node_agent.py --server 192.168.1.100
  python edison_node_agent.py --server 192.168.1.100 --name "Engineering-Laptop" --role cad
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional imports for GPU detection
try:
    import wmi  # type: ignore
    HAS_WMI = True
except ImportError:
    HAS_WMI = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EDISON-Node] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("edison-node")

# ── Configuration ────────────────────────────────────────────────────────

DEFAULT_AGENT_PORT = 9200
HEARTBEAT_INTERVAL = 30  # seconds
TASK_POLL_INTERVAL = 10  # seconds

# Persistent node ID stored locally
NODE_ID_FILE = Path.home() / ".edison" / "node_id"


def get_or_create_node_id() -> str:
    """Get persistent node ID or create one."""
    NODE_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    if NODE_ID_FILE.exists():
        return NODE_ID_FILE.read_text().strip()
    nid = f"node_{uuid.uuid4().hex[:8]}"
    NODE_ID_FILE.write_text(nid)
    return nid


# ── Hardware Detection ───────────────────────────────────────────────────

def detect_cpu() -> str:
    """Detect CPU model string."""
    if platform.system() == "Windows":
        try:
            output = subprocess.check_output(
                ["wmic", "cpu", "get", "name"],
                text=True, timeout=5
            ).strip()
            lines = [l.strip() for l in output.split("\n") if l.strip() and l.strip() != "Name"]
            if lines:
                return lines[0]
        except Exception:
            pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def detect_ram_gb() -> int:
    """Detect total system RAM in GB."""
    try:
        import psutil  # type: ignore
        return round(psutil.virtual_memory().total / (1024 ** 3))
    except ImportError:
        pass
    if platform.system() == "Windows":
        try:
            output = subprocess.check_output(
                ["wmic", "os", "get", "TotalVisibleMemorySize"],
                text=True, timeout=5
            )
            for line in output.split("\n"):
                line = line.strip()
                if line.isdigit():
                    return round(int(line) / (1024 * 1024))
        except Exception:
            pass
    return 0


def detect_gpu() -> tuple[str, int]:
    """Detect GPU name and VRAM in GB. Returns (name, vram_gb)."""
    # Try nvidia-smi first
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=10
        ).strip()
        if output:
            parts = output.split(",")
            name = parts[0].strip()
            vram_mb = int(parts[1].strip()) if len(parts) > 1 else 0
            return name, round(vram_mb / 1024)
    except Exception:
        pass

    # Windows WMI fallback
    if platform.system() == "Windows":
        try:
            output = subprocess.check_output(
                ["wmic", "path", "win32_videocontroller", "get", "name,adapterram"],
                text=True, timeout=5
            )
            lines = [l.strip() for l in output.split("\n") if l.strip() and "Name" not in l]
            if lines:
                # Find NVIDIA GPU preferentially
                for line in lines:
                    if "nvidia" in line.lower() or "rtx" in line.lower() or "gtx" in line.lower():
                        parts = line.rsplit(None, 1)
                        name = parts[0].strip()
                        try:
                            vram = round(int(parts[1]) / (1024 ** 3))
                        except (ValueError, IndexError):
                            vram = 0
                        return name, vram
                # Fallback to first GPU
                parts = lines[0].rsplit(None, 1)
                name = parts[0].strip()
                try:
                    vram = round(int(parts[1]) / (1024 ** 3))
                except (ValueError, IndexError):
                    vram = 0
                return name, vram
        except Exception:
            pass

    return "Unknown GPU", 0


def detect_software() -> Dict[str, str]:
    """Detect installed CAD/creative software."""
    software: Dict[str, str] = {}

    # Rhino 7/8
    if platform.system() == "Windows":
        rhino_paths = [
            Path(r"C:\Program Files\Rhino 7\System\Rhino.exe"),
            Path(r"C:\Program Files\Rhino 8\System\Rhino.exe"),
            Path(r"C:\Program Files\McNeel\Rhinoceros\7.0\System\Rhino.exe"),
        ]
        for rp in rhino_paths:
            if rp.exists():
                ver = "8" if "8" in str(rp) else "7"
                software["rhino"] = ver
                break
    else:
        if shutil.which("rhinoceros") or shutil.which("rhino"):
            software["rhino"] = "detected"

    # Blender
    if shutil.which("blender"):
        try:
            out = subprocess.check_output(["blender", "--version"], text=True, timeout=5)
            ver = out.strip().split("\n")[0].replace("Blender ", "")
            software["blender"] = ver
        except Exception:
            software["blender"] = "detected"

    # FreeCAD
    if shutil.which("freecad") or shutil.which("FreeCAD"):
        software["freecad"] = "detected"

    # SolidWorks (Windows only)
    if platform.system() == "Windows":
        sw_paths = [
            r"C:\Program Files\SOLIDWORKS Corp\SOLIDWORKS\SLDWORKS.exe",
            r"C:\Program Files\SolidWorks Corp\SolidWorks\SLDWORKS.exe",
        ]
        # Also check versioned installs 2020-2025
        for yr in range(2020, 2026):
            sw_paths.append(rf"C:\Program Files\SOLIDWORKS Corp\SOLIDWORKS {yr}\SLDWORKS.exe")
            sw_paths.append(rf"C:\Program Files\SolidWorks Corp\SolidWorks {yr}\SLDWORKS.exe")
        for sw_path_str in sw_paths:
            sw_path = Path(sw_path_str)
            if sw_path.exists():
                # Extract year from path component if possible
                for part in sw_path.parts:
                    if "solidworks" in part.lower() and any(c.isdigit() for c in part):
                        digits = [c for c in part if c.isdigit()]
                        if len(digits) >= 4:
                            ver = "".join(digits[:4])
                            software["solidworks"] = ver
                            break
                if "solidworks" not in software:
                    software["solidworks"] = "detected"
                break
        if "solidworks" not in software:
            # Fallback: registry check
            try:
                import winreg  # type: ignore
                for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
                    for key_path in [
                        r"SOFTWARE\SolidWorks",
                        r"SOFTWARE\SolidWorks Corp\SolidWorks",
                        r"SOFTWARE\WOW6432Node\SolidWorks Corp\SolidWorks",
                    ]:
                        try:
                            with winreg.OpenKey(hive, key_path):
                                software["solidworks"] = "detected"
                                break
                        except FileNotFoundError:
                            continue
                    if "solidworks" in software:
                        break
            except ImportError:
                pass

    # Grasshopper (comes with Rhino)
    if "rhino" in software:
        software["grasshopper"] = software["rhino"]

    # Python
    software["python"] = platform.python_version()

    return software


def detect_capabilities(software: Dict[str, str]) -> List[str]:
    """Derive capability tags from detected software."""
    caps = []
    if "rhino" in software:
        caps.extend(["cad", "rhino", "3d-modeling", "nurbs", "surface-modeling"])
    if "grasshopper" in software:
        caps.extend(["grasshopper", "parametric-design"])
    if "blender" in software:
        caps.extend(["3d-modeling", "rendering", "blender", "animation", "sculpting"])
    if "freecad" in software:
        caps.extend(["cad", "freecad", "parametric-cad"])
    if "solidworks" in software:
        caps.extend(["cad", "solidworks", "mechanical-cad", "3d-modeling",
                     "sheet-metal", "assembly-design", "technical-drawing"])

    # GPU capabilities
    gpu_name, _ = detect_gpu()
    if any(x in gpu_name.lower() for x in ["nvidia", "rtx", "gtx", "quadro", "a3000", "a4000", "a5000"]):
        caps.extend(["gpu-compute", "cuda", "gpu-render"])
    if any(x in gpu_name.lower() for x in ["quadro", "a3000", "a4000", "a5000", "rtx a"]):
        caps.append("workstation-gpu")

    return list(set(caps))


def get_system_status() -> Dict[str, Any]:
    """Get current CPU/RAM/GPU usage."""
    status: Dict[str, Any] = {}
    try:
        import psutil  # type: ignore
        status["cpu_usage"] = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        status["ram_usage"] = mem.percent
        status["ram_available_gb"] = round(mem.available / (1024 ** 3), 1)
    except ImportError:
        pass

    # GPU usage via nvidia-smi
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5
        ).strip()
        if output:
            parts = [p.strip() for p in output.split(",")]
            status["gpu_usage"] = int(parts[0])
            status["gpu_mem_used_mb"] = int(parts[1])
            status["gpu_mem_total_mb"] = int(parts[2])
    except Exception:
        pass

    return status


# ── Rhino 7 Integration ─────────────────────────────────────────────────

class RhinoController:
    """Interface to Rhino 7 via COM automation (Windows) or command-line."""

    def __init__(self):
        self.available = False
        self._rhino = None
        if platform.system() == "Windows":
            try:
                import win32com.client  # type: ignore
                self._rhino = win32com.client.Dispatch("Rhino.Application")
                self.available = True
                logger.info("✓ Connected to Rhino 7 via COM")
            except Exception as e:
                logger.info(f"Rhino COM not available (start Rhino first): {e}")

    def run_command(self, command: str) -> Dict[str, Any]:
        """Send a Rhino command string."""
        if not self.available or self._rhino is None:
            return {"ok": False, "error": "Rhino is not connected"}
        try:
            self._rhino.RunScript(f"-_RunScript ({command})", 0)
            return {"ok": True, "command": command}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def run_python_script(self, script_path: str) -> Dict[str, Any]:
        """Run a Python script inside Rhino."""
        if not self.available or self._rhino is None:
            return {"ok": False, "error": "Rhino is not connected"}
        try:
            self._rhino.RunScript(f'-_RunPythonScript "{script_path}"', 0)
            return {"ok": True, "script": script_path}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def run_grasshopper_definition(self, gh_path: str) -> Dict[str, Any]:
        """Open a Grasshopper definition."""
        if not self.available or self._rhino is None:
            return {"ok": False, "error": "Rhino is not connected"}
        try:
            self._rhino.RunScript(f'-_Grasshopper _Open "{gh_path}"', 0)
            return {"ok": True, "definition": gh_path}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def export_file(self, filepath: str, format: str = "3dm") -> Dict[str, Any]:
        """Export the current document."""
        if not self.available or self._rhino is None:
            return {"ok": False, "error": "Rhino is not connected"}
        try:
            self._rhino.RunScript(f'-_Export "{filepath}"', 0)
            return {"ok": True, "exported": filepath, "format": format}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def open_file(self, filepath: str) -> Dict[str, Any]:
        """Open a file in Rhino."""
        if not self.available or self._rhino is None:
            return {"ok": False, "error": "Rhino is not connected"}
        try:
            self._rhino.RunScript(f'-_Open "{filepath}"', 0)
            return {"ok": True, "opened": filepath}
        except Exception as e:
            return {"ok": False, "error": str(e)}


# ── SolidWorks Controller ────────────────────────────────────────────────

class SolidWorksController:
    """Interface to SolidWorks via COM automation (Windows only)."""

    def __init__(self):
        self.available = False
        self._sw = None
        if platform.system() == "Windows":
            try:
                import win32com.client  # type: ignore
                # Connect to running instance first, then launch
                try:
                    self._sw = win32com.client.GetActiveObject("SldWorks.Application")
                    logger.info("✓ Connected to running SolidWorks instance via COM")
                except Exception:
                    self._sw = win32com.client.Dispatch("SldWorks.Application")
                    self._sw.Visible = True
                    logger.info("✓ Launched SolidWorks via COM")
                self.available = True
            except Exception as e:
                logger.info(f"SolidWorks COM not available (start SolidWorks first): {e}")

    def open_file(self, filepath: str) -> Dict[str, Any]:
        """Open a SolidWorks part/assembly/drawing file."""
        if not self.available or self._sw is None:
            return {"ok": False, "error": "SolidWorks is not connected"}
        try:
            import win32com.client  # type: ignore
            ext = Path(filepath).suffix.lower()
            doc_type = {".sldprt": 1, ".sldasm": 2, ".slddrw": 3}.get(ext, 1)
            errors_var = win32com.client.VARIANT(0x4003, 0)   # VT_BYREF|VT_I4
            warns_var  = win32com.client.VARIANT(0x4003, 0)
            doc = self._sw.OpenDoc6(filepath, doc_type, 0, "", errors_var, warns_var)
            if doc:
                return {"ok": True, "opened": filepath}
            return {"ok": False, "error": f"Could not open {filepath}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def run_macro(self, macro_path: str) -> Dict[str, Any]:
        """Run a SolidWorks VBA macro (.swp file)."""
        if not self.available or self._sw is None:
            return {"ok": False, "error": "SolidWorks is not connected"}
        try:
            ret = self._sw.RunMacro(macro_path, "", "")
            return {"ok": True, "macro": macro_path, "result": ret}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def export_file(self, output_path: str, fmt: str = "step") -> Dict[str, Any]:
        """Export the active SolidWorks document to another format."""
        if not self.available or self._sw is None:
            return {"ok": False, "error": "SolidWorks is not connected"}
        try:
            import win32com.client  # type: ignore
            doc = self._sw.ActiveDoc
            if doc is None:
                return {"ok": False, "error": "No active document in SolidWorks"}
            # swExportDataFormat enum values
            fmt_map = {"step": 20, "stl": 26, "iges": 3, "dxf": 9,
                       "dwg": 10, "obj": 41, "3ds": 25, "parasolid": 31}
            fmt_id = fmt_map.get(fmt.lower(), 20)
            data_options = self._sw.GetExportFileData(fmt_id)
            errors_var = win32com.client.VARIANT(0x4003, 0)
            warns_var  = win32com.client.VARIANT(0x4003, 0)
            doc.Extension.SaveAs(output_path, 0, 0, data_options, errors_var, warns_var)
            return {"ok": True, "exported": output_path, "format": fmt}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def run_script(self, script_content: str) -> Dict[str, Any]:
        """Run a Python script that uses solidworks-python-api (swpg) or standalone."""
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tf:
                tf.write(script_content)
                tmp_path = tf.name
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True, timeout=120,
            )
            Path(tmp_path).unlink(missing_ok=True)
            return {
                "ok": result.returncode == 0,
                "stdout": result.stdout[:4000],
                "stderr": result.stderr[:1000],
                "returncode": result.returncode,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}


# ── Blender Controller ────────────────────────────────────────────────────

class BlenderController:
    """Run Blender in background mode or via headless script execution."""

    def __init__(self):
        self.blender_exe = shutil.which("blender") or ""
        self.available = bool(self.blender_exe)
        if self.available:
            logger.info(f"✓ Blender found at {self.blender_exe}")
        else:
            logger.info("Blender not found in PATH")

    def run_script(self, script_content: str, blend_file: str = "") -> Dict[str, Any]:
        """Run a Python script inside Blender (background mode)."""
        if not self.available:
            return {"ok": False, "error": "Blender not found"}
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tf:
                tf.write(script_content)
                script_path = tf.name
            cmd = [self.blender_exe, "--background"]
            if blend_file:
                cmd.append(blend_file)
            cmd += ["--python", script_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            Path(script_path).unlink(missing_ok=True)
            return {
                "ok": result.returncode == 0,
                "stdout": result.stdout[:4000],
                "stderr": result.stderr[:1000],
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Blender script timed out (300s)"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def render(self, blend_file: str, output_path: str = "", frame: int = 1) -> Dict[str, Any]:
        """Render a frame from a .blend file."""
        if not self.available:
            return {"ok": False, "error": "Blender not found"}
        try:
            cmd = [self.blender_exe, "--background", blend_file,
                   "--render-output", output_path or "/tmp/render_",
                   "--render-frame", str(frame)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            return {
                "ok": result.returncode == 0,
                "stdout": result.stdout[:2000],
                "stderr": result.stderr[:500],
                "output_path": output_path,
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Blender render timed out (600s)"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def export(self, blend_file: str, output_path: str, fmt: str = "gltf") -> Dict[str, Any]:
        """Export a .blend to another format using a generated inline script."""
        fmt_lower = fmt.lower()
        export_scripts = {
            "stl": (
                "import bpy\n"
                f"bpy.ops.export_mesh.stl(filepath=r'{output_path}')\n"
            ),
            "obj": (
                "import bpy\n"
                f"bpy.ops.export_scene.obj(filepath=r'{output_path}')\n"
            ),
            "gltf": (
                "import bpy\n"
                f"bpy.ops.export_scene.gltf(filepath=r'{output_path}', export_format='GLTF_SEPARATE')\n"
            ),
            "fbx": (
                "import bpy\n"
                f"bpy.ops.export_scene.fbx(filepath=r'{output_path}')\n"
            ),
        }
        script = export_scripts.get(fmt_lower)
        if not script:
            return {"ok": False, "error": f"Unsupported export format: {fmt}"}
        return self.run_script(script, blend_file=blend_file)


# ── Task Executor ────────────────────────────────────────────────────────

class TaskExecutor:
    """Executes tasks received from the EDISON server."""

    def __init__(
        self,
        rhino: Optional[RhinoController] = None,
        solidworks: Optional["SolidWorksController"] = None,
        blender: Optional["BlenderController"] = None,
    ):
        self.rhino = rhino
        self.solidworks = solidworks
        self.blender = blender
        self._handlers = {
            # Rhino
            "rhino_command": self._handle_rhino_command,
            "rhino_script": self._handle_rhino_script,
            "rhino_grasshopper": self._handle_rhino_grasshopper,
            "rhino_export": self._handle_rhino_export,
            "rhino_open": self._handle_rhino_open,
            # SolidWorks
            "solidworks_open": self._handle_sw_open,
            "solidworks_macro": self._handle_sw_macro,
            "solidworks_export": self._handle_sw_export,
            "solidworks_script": self._handle_sw_script,
            # Blender
            "blender_script": self._handle_blender_script,
            "blender_render": self._handle_blender_render,
            "blender_export": self._handle_blender_export,
            # Generic
            "shell": self._handle_shell,
            "file_transfer": self._handle_file_transfer,
            "ping": self._handle_ping,
            "system_status": self._handle_system_status,
        }

    def execute(self, task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        handler = self._handlers.get(task_type)
        if handler is None:
            return {"ok": False, "error": f"Unknown task type: {task_type}"}
        try:
            return handler(payload)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _handle_rhino_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.rhino is None or not self.rhino.available:
            return {"ok": False, "error": "Rhino is not connected on this node"}
        return self.rhino.run_command(payload.get("command", ""))

    def _handle_rhino_script(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.rhino is None or not self.rhino.available:
            return {"ok": False, "error": "Rhino is not connected on this node"}
        return self.rhino.run_python_script(payload.get("script_path", ""))

    def _handle_rhino_grasshopper(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.rhino is None or not self.rhino.available:
            return {"ok": False, "error": "Rhino is not connected on this node"}
        return self.rhino.run_grasshopper_definition(payload.get("gh_path", ""))

    def _handle_rhino_export(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.rhino is None or not self.rhino.available:
            return {"ok": False, "error": "Rhino is not connected on this node"}
        return self.rhino.export_file(
            payload.get("filepath", ""),
            payload.get("format", "3dm"),
        )

    def _handle_rhino_open(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.rhino is None or not self.rhino.available:
            return {"ok": False, "error": "Rhino is not connected on this node"}
        return self.rhino.open_file(payload.get("filepath", ""))

    # ── SolidWorks handlers ──

    def _handle_sw_open(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.solidworks is None or not self.solidworks.available:
            return {"ok": False, "error": "SolidWorks is not connected on this node"}
        return self.solidworks.open_file(payload.get("filepath", ""))

    def _handle_sw_macro(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.solidworks is None or not self.solidworks.available:
            return {"ok": False, "error": "SolidWorks is not connected on this node"}
        return self.solidworks.run_macro(payload.get("macro_path", ""))

    def _handle_sw_export(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.solidworks is None or not self.solidworks.available:
            return {"ok": False, "error": "SolidWorks is not connected on this node"}
        return self.solidworks.export_file(
            payload.get("output_path", ""),
            payload.get("format", "step"),
        )

    def _handle_sw_script(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.solidworks is None or not self.solidworks.available:
            return {"ok": False, "error": "SolidWorks is not connected on this node"}
        return self.solidworks.run_script(payload.get("script", ""))

    # ── Blender handlers ──

    def _handle_blender_script(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.blender is None or not self.blender.available:
            return {"ok": False, "error": "Blender not found on this node"}
        return self.blender.run_script(
            payload.get("script", ""),
            payload.get("blend_file", ""),
        )

    def _handle_blender_render(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.blender is None or not self.blender.available:
            return {"ok": False, "error": "Blender not found on this node"}
        return self.blender.render(
            payload.get("blend_file", ""),
            payload.get("output_path", ""),
            int(payload.get("frame", 1)),
        )

    def _handle_blender_export(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.blender is None or not self.blender.available:
            return {"ok": False, "error": "Blender not found on this node"}
        return self.blender.export(
            payload.get("blend_file", ""),
            payload.get("output_path", ""),
            payload.get("format", "gltf"),
        )

    # ── Generic handlers ──

    def _handle_shell(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        cmd = payload.get("command", "")
        if not cmd:
            return {"ok": False, "error": "No command specified"}
        # Safety: only allow whitelisted commands
        allowed_prefixes = ["dir", "ls", "echo", "hostname", "ipconfig", "systeminfo", "whoami", "python"]
        cmd_lower = cmd.strip().lower()
        if not any(cmd_lower.startswith(p) for p in allowed_prefixes):
            return {"ok": False, "error": f"Command not in allowlist. Allowed: {allowed_prefixes}"}
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30
            )
            return {
                "ok": True,
                "stdout": result.stdout[:4000],
                "stderr": result.stderr[:1000],
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Command timed out (30s)"}

    def _handle_file_transfer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Placeholder for file transfer
        return {"ok": True, "message": "File transfer not yet implemented"}

    def _handle_ping(self, _payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True, "pong": True, "time": time.time()}

    def _handle_system_status(self, _payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True, **get_system_status()}


# ── HTTP Handler (local agent API) ──────────────────────────────────────

class NodeAgentHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the local node agent API."""

    executor: TaskExecutor
    node_info: Dict[str, Any]

    def log_message(self, format, *args):
        logger.debug(f"HTTP: {format % args}")

    def _send_json(self, data: Dict[str, Any], status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw)

    def do_GET(self):
        if self.path == "/info":
            self._send_json(self.__class__.node_info)
        elif self.path == "/status":
            self._send_json({"ok": True, **get_system_status()})
        elif self.path == "/health":
            self._send_json({"ok": True, "status": "running"})
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path == "/execute":
            body = self._read_body()
            command = body.get("command", "")
            params = body.get("params", {})
            result = self.__class__.executor.execute(command, params)
            self._send_json(result)
        else:
            self._send_json({"error": "Not found"}, 404)


# ── Main Agent Loop ─────────────────────────────────────────────────────

class EdisonNodeAgent:
    """Main agent that registers with EDISON and maintains connection."""

    def __init__(
        self,
        server_host: str,
        server_port: int = 8080,
        agent_port: int = DEFAULT_AGENT_PORT,
        name: str = "",
        role: str = "general",
    ):
        # Normalize server URL — handle cases like "192.168.1.20:8811" or "http://192.168.1.20"
        host = server_host.strip()
        if host.startswith("http://") or host.startswith("https://"):
            # Already has scheme — strip it to parse
            scheme_end = host.index("://") + 3
            scheme = host[:scheme_end]
            rest = host[scheme_end:]
        else:
            scheme = "http://"
            rest = host
        # If host already includes a port, use it; otherwise append server_port
        if ":" in rest:
            self.server_url = f"{scheme}{rest}"
        else:
            self.server_url = f"{scheme}{rest}:{server_port}"
        self.agent_port = agent_port
        self.node_id = get_or_create_node_id()
        self.name = name or f"{platform.node()}"
        self.role = role
        self.running = False

        # Detect hardware
        logger.info("Detecting hardware...")
        self.cpu = detect_cpu()
        self.ram_gb = detect_ram_gb()
        self.gpu_name, self.gpu_vram = detect_gpu()
        self.software = detect_software()
        self.capabilities = detect_capabilities(self.software)

        logger.info(f"  CPU: {self.cpu}")
        logger.info(f"  RAM: {self.ram_gb} GB")
        logger.info(f"  GPU: {self.gpu_name} ({self.gpu_vram} GB VRAM)")
        logger.info(f"  Software: {self.software}")
        logger.info(f"  Capabilities: {self.capabilities}")

        # Rhino controller
        self.rhino = RhinoController() if "rhino" in self.software else None

        # SolidWorks controller
        self.solidworks = SolidWorksController() if "solidworks" in self.software else None

        # Blender controller
        self.blender = BlenderController() if "blender" in self.software else None

        # Task executor
        self.executor = TaskExecutor(
            rhino=self.rhino,
            solidworks=self.solidworks,
            blender=self.blender,
        )

    def _get_local_ip(self) -> str:
        """Get the LAN IP that can reach the EDISON server."""
        try:
            # Parse server host to connect to
            host = self.server_url.replace("http://", "").replace("https://", "").split(":")[0]
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((host, 80))
                return s.getsockname()[0]
        except Exception:
            return socket.gethostbyname(socket.gethostname())

    def _build_registration(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "name": self.name,
            "host": self._get_local_ip(),
            "port": self.agent_port,
            "role": self.role,
            "os": platform.system().lower(),
            "cpu": self.cpu,
            "ram_gb": self.ram_gb,
            "gpu": self.gpu_name,
            "gpu_vram_gb": self.gpu_vram,
            "capabilities": self.capabilities,
            "software": self.software,
            "enabled": True,
        }

    def register(self) -> bool:
        """Register with the EDISON server."""
        payload = self._build_registration()
        try:
            resp = requests.post(
                f"{self.server_url}/nodes/register",
                json=payload,
                timeout=10,
            )
            if resp.ok:
                logger.info(f"✓ Registered with EDISON server as '{self.node_id}'")
                return True
            else:
                logger.error(f"Registration failed: {resp.status_code} {resp.text[:200]}")
                return False
        except Exception as e:
            logger.error(f"Cannot reach EDISON server at {self.server_url}: {e}")
            return False

    def _heartbeat_loop(self):
        """Send heartbeats and poll for tasks."""
        import requests as req

        while self.running:
            try:
                status = get_system_status()
                resp = req.post(
                    f"{self.server_url}/nodes/{self.node_id}/heartbeat",
                    json=status,
                    timeout=10,
                )
                if resp.ok:
                    data = resp.json()
                    pending = data.get("pending_tasks", [])
                    for task in pending:
                        self._execute_remote_task(task)
                else:
                    logger.warning(f"Heartbeat failed: {resp.status_code}")
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")

            time.sleep(HEARTBEAT_INTERVAL)

    def _execute_remote_task(self, task: Dict[str, Any]):
        """Execute a task received from the server and report back."""
        import requests as req

        task_id = task.get("id", "")
        task_type = task.get("task_type", "")
        payload = task.get("payload", {})

        logger.info(f"Executing task {task_id}: {task_type}")

        # Mark as running
        try:
            req.post(
                f"{self.server_url}/nodes/tasks/{task_id}/update",
                json={"status": "running"},
                timeout=5,
            )
        except Exception:
            pass

        # Execute
        result = self.executor.execute(task_type, payload)
        status = "done" if result.get("ok") else "failed"

        # Report result
        try:
            req.post(
                f"{self.server_url}/nodes/tasks/{task_id}/update",
                json={"status": status, "result": result},
                timeout=10,
            )
            logger.info(f"Task {task_id} completed: {status}")
        except Exception as e:
            logger.error(f"Failed to report task result: {e}")

    def _start_http_server(self):
        """Start the local HTTP API for direct commands."""
        NodeAgentHandler.executor = self.executor
        NodeAgentHandler.node_info = self._build_registration()

        server = HTTPServer(("0.0.0.0", self.agent_port), NodeAgentHandler)
        logger.info(f"✓ Agent HTTP API listening on port {self.agent_port}")
        server.serve_forever()

    def start(self):
        """Start the agent: register, heartbeat, and serve."""
        self.running = True

        # Register
        registered = False
        for attempt in range(5):
            if self.register():
                registered = True
                break
            logger.info(f"Retrying registration in 5s (attempt {attempt + 2}/5)...")
            time.sleep(5)

        if not registered:
            logger.error("Failed to register after 5 attempts. Running in standalone mode.")
            logger.info("The agent will retry on heartbeat.")

        # Start heartbeat thread
        hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        hb_thread.start()

        # Start HTTP server (blocks)
        try:
            self._start_http_server()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.running = False


# ── CLI ──────────────────────────────────────────────────────────────────

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package is required. Install it with:")
    print("  pip install requests")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="EDISON Node Agent — connect this machine to your EDISON AI server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python edison_node_agent.py --server 192.168.1.100
  python edison_node_agent.py --server 192.168.1.100 --name "CAD-Laptop" --role cad
  python edison_node_agent.py --server 192.168.1.100 --port 9201
        """,
    )
    parser.add_argument(
        "--server", required=True,
        help="IP address or hostname of the main EDISON server",
    )
    parser.add_argument(
        "--server-port", type=int, default=8080,
        help="EDISON server port (default: 8080 — the EDISON web proxy)",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_AGENT_PORT,
        help=f"Local agent HTTP port (default: {DEFAULT_AGENT_PORT})",
    )
    parser.add_argument(
        "--name", default="",
        help="Friendly name for this node (default: hostname)",
    )
    parser.add_argument(
        "--role", default="general",
        choices=["general", "cad", "render", "compute", "print"],
        help="Node role (default: general)",
    )

    args = parser.parse_args()

    print(r"""
    ███████╗██████╗ ██╗███████╗ ██████╗ ███╗   ██╗
    ██╔════╝██╔══██╗██║██╔════╝██╔═══██╗████╗  ██║
    █████╗  ██║  ██║██║███████╗██║   ██║██╔██╗ ██║
    ██╔══╝  ██║  ██║██║╚════██║██║   ██║██║╚██╗██║
    ███████╗██████╔╝██║███████║╚██████╔╝██║ ╚████║
    ╚══════╝╚═════╝ ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
                    Node Agent
    """)

    agent = EdisonNodeAgent(
        server_host=args.server,
        server_port=args.server_port,
        agent_port=args.port,
        name=args.name,
        role=args.role,
    )
    agent.start()


if __name__ == "__main__":
    main()
