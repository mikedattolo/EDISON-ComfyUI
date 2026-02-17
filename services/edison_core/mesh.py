"""
3D Mesh Generation Service for EDISON
Generates 3D meshes (GLB/STL) from text prompts.
Supports ComfyUI 3D workflow backend with graceful fallback.
"""

import json
import logging
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
MESH_OUTPUT_DIR = REPO_ROOT / "outputs" / "meshes"
MESH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class MeshGenerationService:
    """
    3D mesh generation service.
    Primary backend: ComfyUI 3D workflow (if available)
    Saves GLB by default, STL optional via conversion.
    """

    def __init__(self, config: Dict[str, Any]):
        mesh_cfg = config.get("edison", {}).get("mesh", {})
        comfyui_cfg = config.get("edison", {}).get("comfyui", {})

        host = comfyui_cfg.get("host", "127.0.0.1")
        if host == "0.0.0.0":
            host = "127.0.0.1"
        port = comfyui_cfg.get("port", 8188)
        self.comfyui_url = f"http://{host}:{port}"

        self.default_format = mesh_cfg.get("default_format", "glb")
        self._available = False
        self._check_availability()

        logger.info(f"3D mesh generation service initialized (available={self._available})")

    def _check_availability(self):
        """Check if 3D generation backend is available."""
        try:
            import requests
            resp = requests.get(f"{self.comfyui_url}/system_stats", timeout=3)
            self._available = resp.status_code == 200
        except Exception:
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def generate(
        self,
        prompt: str,
        output_format: str = "glb",
        params: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a 3D mesh from a text prompt.

        Returns dict with job info including output path on success.
        """
        if job_id is None:
            job_id = str(uuid.uuid4())

        params = params or {}
        output_format = output_format.lower()
        if output_format not in ("glb", "stl"):
            output_format = "glb"

        # Try to register with unified job store
        try:
            from .job_store import JobStore
            store = JobStore.get_instance()
            store.create_job(
                job_type="mesh",
                prompt=prompt,
                params=params,
                provenance={"backend": "comfyui", "format": output_format},
            )
        except Exception:
            pass

        result = {
            "job_id": job_id,
            "status": "error",
            "prompt": prompt,
            "format": output_format,
        }

        if not self._available:
            result["error"] = (
                "3D generation backend not available. "
                "Ensure ComfyUI is running with 3D nodes installed."
            )
            self._update_job_status(job_id, "error", error_log=result["error"])
            return result

        try:
            self._update_job_status(job_id, "generating")

            # Build ComfyUI 3D workflow
            workflow = self._build_workflow(prompt, output_format, params)

            import requests
            resp = requests.post(
                f"{self.comfyui_url}/prompt",
                json={"prompt": workflow},
                timeout=30,
            )
            resp.raise_for_status()
            comfy_result = resp.json()

            prompt_id = comfy_result.get("prompt_id", "")
            result["prompt_id"] = prompt_id
            result["status"] = "generating"
            return result

        except Exception as e:
            error_msg = f"3D generation failed: {type(e).__name__}: {e}"
            logger.error(error_msg)
            result["error"] = error_msg
            self._update_job_status(job_id, "error", error_log=error_msg)
            return result

    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a 3D generation job."""
        try:
            from .job_store import JobStore
            store = JobStore.get_instance()
            job = store.get_job(job_id)
            if job:
                return job
        except Exception:
            pass
        return {"job_id": job_id, "status": "unknown"}

    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get result of a completed 3D generation job."""
        try:
            from .job_store import JobStore
            store = JobStore.get_instance()
            job = store.get_job(job_id)
            if job and job["status"] == "complete":
                return job
        except Exception:
            pass
        return None

    def save_output(self, job_id: str, data: bytes, output_format: str = "glb") -> Path:
        """Save generated mesh data and write metadata sidecar."""
        filename = f"{job_id}.{output_format}"
        output_path = MESH_OUTPUT_DIR / filename
        output_path.write_bytes(data)

        # Write sidecar metadata
        try:
            from .job_store import JobStore
            store = JobStore.get_instance()
            store.update_status(job_id, "complete", outputs=[str(output_path)])
            store.write_metadata_sidecar(job_id, output_path)
        except Exception as e:
            logger.warning(f"Could not write sidecar for {job_id}: {e}")

        logger.info(f"Saved 3D mesh: {output_path}")
        return output_path

    def _build_workflow(self, prompt: str, output_format: str, params: Dict) -> Dict:
        """Build a ComfyUI workflow for 3D generation."""
        # Basic text-to-3D workflow structure
        # This would be customized based on available ComfyUI 3D nodes
        return {
            "1": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["2", 0]},
            },
            "2": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": params.get("model", "default_3d_model.safetensors")},
            },
        }

    def _update_job_status(self, job_id: str, status: str, **kwargs):
        try:
            from .job_store import JobStore
            store = JobStore.get_instance()
            store.update_status(job_id, status, **kwargs)
        except Exception:
            pass
