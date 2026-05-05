"""
Phase 4 routes: image-gen quality-of-life endpoints.

Aspect presets, prompt rewriter, project style sheets, queue ETA,
seed-variation helper, A/B grid, and the "print this" pipeline planner.

Additive — safe to mount alongside phase1/2/3.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/phase4", tags=["phase4"])


# ── Aspect presets ──────────────────────────────────────────────────

@router.get("/image/presets")
async def image_presets() -> Dict[str, Any]:
    from ..image_gen_presets import list_presets
    return {"presets": list_presets()}


@router.get("/image/presets/{name}")
async def image_preset(name: str) -> Dict[str, Any]:
    from ..image_gen_presets import resolve_aspect
    try:
        return resolve_aspect(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ── Negative prompt + sampler defaults ─────────────────────────────

@router.get("/image/defaults")
async def image_defaults(
    model_family: str = "sdxl",
    intent: Optional[str] = None,
) -> Dict[str, Any]:
    from ..image_gen_presets import pick_negative_prompt, sampler_defaults
    return {
        "model_family": model_family,
        "intent": intent,
        "negative_prompt": pick_negative_prompt(model_family, intent=intent),
        "sampler": sampler_defaults(model_family),
    }


# ── Prompt rewriter ─────────────────────────────────────────────────

class RewriteRequest(BaseModel):
    prompt: str
    model_family: str = "sdxl"
    project_id: Optional[str] = None
    intent: Optional[str] = None


@router.post("/image/rewrite")
async def image_rewrite(req: RewriteRequest) -> Dict[str, Any]:
    from ..image_gen_presets import rewrite_prompt, StyleSheetStore
    sheet = None
    if req.project_id:
        sheet = StyleSheetStore.get_instance().get(req.project_id)
    rewritten = rewrite_prompt(
        req.prompt,
        model_family=req.model_family,
        style_sheet=sheet,
        explicit_intent=req.intent,
    )
    return rewritten.to_dict()


# ── Style sheet CRUD ────────────────────────────────────────────────

class StyleSheetIn(BaseModel):
    project_id: str
    palette: List[str] = Field(default_factory=list)
    fonts: List[str] = Field(default_factory=list)
    vibe: List[str] = Field(default_factory=list)
    banned: List[str] = Field(default_factory=list)


@router.get("/style/{project_id}")
async def style_get(project_id: str) -> Dict[str, Any]:
    from ..image_gen_presets import StyleSheetStore
    sheet = StyleSheetStore.get_instance().get(project_id)
    if sheet is None:
        raise HTTPException(status_code=404, detail="style sheet not found")
    return sheet.to_dict()


@router.put("/style")
async def style_put(payload: StyleSheetIn) -> Dict[str, Any]:
    from ..image_gen_presets import StyleSheet, StyleSheetStore
    sheet = StyleSheet(
        project_id=payload.project_id,
        palette=payload.palette,
        fonts=payload.fonts,
        vibe=payload.vibe,
        banned=payload.banned,
    )
    StyleSheetStore.get_instance().save(sheet)
    return sheet.to_dict()


@router.delete("/style/{project_id}")
async def style_delete(project_id: str) -> Dict[str, Any]:
    from ..image_gen_presets import StyleSheetStore
    ok = StyleSheetStore.get_instance().delete(project_id)
    return {"deleted": ok, "project_id": project_id}


@router.get("/style")
async def style_list() -> Dict[str, Any]:
    from ..image_gen_presets import StyleSheetStore
    return {"projects": StyleSheetStore.get_instance().list_projects()}


# ── Queue ETA ───────────────────────────────────────────────────────

@router.get("/queue/eta")
async def queue_eta(lane: str = "image") -> Dict[str, Any]:
    from ..image_gen_presets import estimate_lane_eta_ms
    try:
        from ..gpu_scheduler import get_scheduler
        telemetry = get_scheduler().telemetry()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"scheduler unavailable: {exc}")
    return estimate_lane_eta_ms(telemetry, lane)


# ── Seed variations + A/B grid ─────────────────────────────────────

@router.get("/image/variations")
async def image_variations(seed: int, count: int = 4, spread: int = 1) -> Dict[str, Any]:
    from ..image_gen_presets import variation_seeds
    seeds = variation_seeds(seed, count=count, spread=spread)
    return {"seed": seed, "count": count, "spread": spread, "seeds": seeds}


@router.get("/image/grid")
async def image_grid() -> Dict[str, Any]:
    from ..image_gen_presets import build_grid
    return {"grid": build_grid()}


# ── "Print this" pipeline planner ──────────────────────────────────

class PrintThisRequest(BaseModel):
    artifact_id: str
    intent: str = "keychain"
    material: str = "PLA"
    color: str = "black"


@router.post("/print-this/plan")
async def print_this_plan(req: PrintThisRequest) -> Dict[str, Any]:
    from ..image_gen_presets import plan_print_this
    return {
        "artifact_id": req.artifact_id,
        "intent": req.intent,
        "material": req.material,
        "color": req.color,
        "steps": plan_print_this(
            artifact_id=req.artifact_id,
            intent=req.intent,
            material=req.material,
            color=req.color,
        ),
    }
