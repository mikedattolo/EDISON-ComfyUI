"""GPU fan control API routes for EDISON."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..gpu_fan_control import GpuFanController, load_yaml_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gpu-fans", tags=["gpu-fan-control"])

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "edison.yaml"
_controller: Optional[GpuFanController] = None


def get_controller(force_reload: bool = False) -> GpuFanController:
    global _controller
    if _controller is None or force_reload:
        try:
            config = load_yaml_config(CONFIG_PATH)
        except Exception:
            config = {}
        _controller = GpuFanController(REPO_ROOT, config)
    return _controller


class FanApplyRequest(BaseModel):
    apply: bool = Field(default=False, description="Actually write fan speeds. False returns dry-run decisions only.")
    poll_interval_s: Optional[float] = Field(default=None, ge=1, le=120)


@router.get("/health")
async def health() -> Dict[str, Any]:
    controller = get_controller()
    diagnostics = controller.diagnostics()
    return {
        "ok": True,
        "module": "GPU Fan Control",
        "enabled": controller.config.get("enabled", True),
        "running": controller.is_running(),
        "apply_default": controller.config.get("apply_enabled_default", False),
        "fan_control_ready": diagnostics.get("fan_control_ready"),
        "gpu_access": diagnostics.get("container_or_host_has_gpu_access"),
    }


@router.get("/config")
async def config() -> Dict[str, Any]:
    return {"ok": True, "config": get_controller().public_config()}


@router.get("/diagnostics")
async def diagnostics() -> Dict[str, Any]:
    return {"ok": True, "diagnostics": get_controller().diagnostics()}


@router.get("/status")
async def status(refresh: bool = True, apply: bool = False) -> Dict[str, Any]:
    controller = get_controller()
    if refresh:
        return controller.evaluate_once(apply=apply)
    return controller.status()


@router.post("/apply-once")
async def apply_once(payload: FanApplyRequest) -> Dict[str, Any]:
    if not payload.apply:
        return get_controller().evaluate_once(apply=False)
    return get_controller().evaluate_once(apply=True)


@router.post("/start")
async def start(payload: FanApplyRequest) -> Dict[str, Any]:
    try:
        return get_controller().start(apply=payload.apply, poll_interval_s=payload.poll_interval_s)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to start GPU fan controller")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/stop")
async def stop() -> Dict[str, Any]:
    return get_controller().stop()


@router.post("/reload")
async def reload_config() -> Dict[str, Any]:
    controller = get_controller(force_reload=True)
    return {"ok": True, "config": controller.public_config(), "diagnostics": controller.diagnostics()}
