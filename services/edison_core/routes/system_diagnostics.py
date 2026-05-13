"""System diagnostics/doctor API routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse

from ..gpu_fan_control import load_yaml_config
from ..system_diagnostics import build_system_diagnostic_report, report_as_text

router = APIRouter(prefix="/system", tags=["system-diagnostics"])

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "edison.yaml"


def _load_config() -> Dict[str, Any]:
    try:
        return load_yaml_config(CONFIG_PATH)
    except Exception:
        return {}


@router.get("/doctor")
async def diagnostics(run_cuda_allocation: bool = Query(default=False)) -> Dict[str, Any]:
    """Return a JSON doctor report for runtime, CUDA, ComfyUI, paths, and cooling."""

    return {
        "ok": True,
        "report": build_system_diagnostic_report(
            REPO_ROOT,
            _load_config(),
            run_cuda_allocation=run_cuda_allocation,
        ),
    }


@router.get("/doctor/text", response_class=PlainTextResponse)
async def diagnostics_text(run_cuda_allocation: bool = Query(default=False)) -> str:
    report = build_system_diagnostic_report(
        REPO_ROOT,
        _load_config(),
        run_cuda_allocation=run_cuda_allocation,
    )
    return report_as_text(report)
