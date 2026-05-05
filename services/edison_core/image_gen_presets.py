"""
Image generation presets, prompt rewriter, and seed-variation helpers.

This module is the cheap-wins layer for image generation:

* :func:`resolve_aspect` — preset → native (W,H) for SDXL/Flux.
* :func:`rewrite_prompt` — turns a freeform prompt into a structured
  Subject / Style / Lighting / Lens / Negatives block, optionally injecting
  a project style sheet.
* :class:`StyleSheet` — per-project palette/fonts/banned tokens, persisted
  as JSON next to the branding store.
* :func:`pick_negative_prompt` — model-aware negative prompt library.
* :func:`variation_seeds` — deterministic neighbouring seeds for the
  "more like this" workflow.
* :func:`build_grid` — sampler/CFG/step grid for A/B test runs.

Pure-Python, no torch/diffusers deps. Anything that needs a GPU goes
through :mod:`gpu_scheduler` and lives in the route layer.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ── Aspect-ratio presets ────────────────────────────────────────────

#: Preset name → (width, height, hint). Sizes chosen to be native for
#: SDXL (multiples of 64, ~1024² total area) and friendly for Flux.
ASPECT_PRESETS: Dict[str, Tuple[int, int, str]] = {
    "square":            (1024, 1024, "Instagram post / album cover"),
    "portrait":          ( 832, 1216, "phone wallpaper / poster"),
    "landscape":         (1216,  832, "desktop banner / hero image"),
    "wide":              (1344,  768, "YouTube thumbnail / cinematic"),
    "tall":              ( 768, 1344, "Instagram story / TikTok / Reels"),
    "story":             (1080, 1920, "Instagram / Snapchat story (upscale target)"),
    "reel":              (1080, 1920, "Instagram Reel / TikTok"),
    "facebook_cover":    (1640,  624, "Facebook page cover"),
    "twitter_header":    (1500,  500, "X / Twitter header"),
    "linkedin_banner":   (1584,  396, "LinkedIn profile banner"),
    "youtube_thumb":     (1280,  720, "YouTube thumbnail"),
    "business_card":     (1050,  600, "3.5x2 in @ 300dpi"),
    "flyer_us_letter":   (2550, 3300, "8.5x11 in @ 300dpi"),
    "flyer_a4":          (2480, 3508, "A4 @ 300dpi"),
    "poster_18x24":      (5400, 7200, "18x24 in poster (upscale target)"),
    "sticker":           (1024, 1024, "die-cut sticker, transparent bg"),
    "tshirt_front":      (3000, 3000, "12x12 in print area @ 300dpi"),
    "logo":              (1024, 1024, "logo on transparent background"),
    "menu_us_letter":    (2550, 3300, "restaurant menu page"),
}

#: Sizes are SDXL/Flux native at gen time; the bigger 'targets' get there
#: via tile upscale rather than huge initial latents. Mark which presets
#: should auto-trigger upscale.
PRESETS_NEEDING_UPSCALE = {
    "story", "reel", "flyer_us_letter", "flyer_a4",
    "poster_18x24", "tshirt_front", "menu_us_letter",
    # Platform-spec sizes that aren't latent-friendly: render larger,
    # then downscale to spec.
    "facebook_cover", "twitter_header", "linkedin_banner",
    "business_card",
}


def resolve_aspect(preset: str) -> Dict[str, object]:
    """Return ``{"width","height","hint","upscale"}`` for a preset name."""
    key = (preset or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key not in ASPECT_PRESETS:
        # graceful fallback: try parsing "WxH"
        m = re.match(r"^(\d{3,5})x(\d{3,5})$", key)
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            return {"width": w, "height": h, "hint": "custom",
                    "upscale": w * h > 1024 * 1024}
        raise KeyError(f"Unknown aspect preset: {preset!r}")
    w, h, hint = ASPECT_PRESETS[key]
    return {"width": w, "height": h, "hint": hint,
            "upscale": key in PRESETS_NEEDING_UPSCALE}


def list_presets() -> List[Dict[str, object]]:
    return [
        {"name": k, "width": w, "height": h, "hint": hint,
         "upscale": k in PRESETS_NEEDING_UPSCALE}
        for k, (w, h, hint) in ASPECT_PRESETS.items()
    ]


# ── Negative prompt library ─────────────────────────────────────────

#: Family → curated negative prompt. Empty string means the model doesn't
#: benefit from negatives (Flux schnell, most distilled models).
NEGATIVE_PROMPTS: Dict[str, str] = {
    "sdxl":
        "blurry, lowres, jpeg artifacts, watermark, text, signature, "
        "deformed, extra fingers, bad anatomy, malformed hands, "
        "out of frame, duplicate, cropped, low quality, ugly",
    "sdxl_portrait":
        "deformed, extra fingers, malformed hands, bad anatomy, "
        "asymmetrical eyes, extra limbs, plastic skin, lowres, "
        "watermark, text, signature, oversaturated",
    "sd15":
        "blurry, lowres, jpeg artifacts, watermark, text, deformed, "
        "extra fingers, bad anatomy, ugly, duplicate, cropped",
    "flux":   "",     # Flux dev/pro/schnell don't benefit
    "flux_dev": "",
    "flux_schnell": "",
    "pixart": "blurry, lowres, watermark, text, deformed, ugly",
    "logo":
        "photorealistic, photograph, gradient, drop shadow, 3d render, "
        "blurry, lowres, watermark, text, jpeg artifacts",
    "sticker":
        "background, photorealistic, blurry, lowres, watermark, "
        "deformed, jpeg artifacts",
}


def pick_negative_prompt(model_family: str, *, intent: Optional[str] = None) -> str:
    """Choose the right negative prompt for a model family + intent."""
    fam = (model_family or "").strip().lower()
    intent = (intent or "").strip().lower()
    if intent in {"logo", "sticker"} and intent in NEGATIVE_PROMPTS:
        return NEGATIVE_PROMPTS[intent]
    if intent == "portrait" and fam.startswith("sdxl"):
        return NEGATIVE_PROMPTS["sdxl_portrait"]
    if fam in NEGATIVE_PROMPTS:
        return NEGATIVE_PROMPTS[fam]
    # heuristic match
    for key in NEGATIVE_PROMPTS:
        if fam.startswith(key):
            return NEGATIVE_PROMPTS[key]
    return NEGATIVE_PROMPTS["sdxl"]  # safe default


# ── Sampler / scheduler defaults ────────────────────────────────────

#: Family → (sampler, scheduler, steps, cfg). Verified-good defaults.
SAMPLER_DEFAULTS: Dict[str, Tuple[str, str, int, float]] = {
    "sdxl":         ("dpmpp_2m",   "karras", 30, 6.5),
    "sdxl_turbo":   ("euler_a",    "normal",  6, 1.5),
    "sd15":         ("dpmpp_2m",   "karras", 25, 7.0),
    "flux":         ("euler",      "simple", 25, 1.0),
    "flux_dev":     ("euler",      "simple", 25, 1.0),
    "flux_schnell": ("euler",      "simple",  4, 1.0),
    "pixart":       ("dpmpp_2m",   "karras", 25, 4.5),
}


def sampler_defaults(model_family: str) -> Dict[str, object]:
    fam = (model_family or "").strip().lower()
    for key in (fam, *(k for k in SAMPLER_DEFAULTS if fam.startswith(k))):
        if key in SAMPLER_DEFAULTS:
            sampler, sched, steps, cfg = SAMPLER_DEFAULTS[key]
            return {"sampler": sampler, "scheduler": sched,
                    "steps": steps, "cfg": cfg, "matched": key}
    sampler, sched, steps, cfg = SAMPLER_DEFAULTS["sdxl"]
    return {"sampler": sampler, "scheduler": sched,
            "steps": steps, "cfg": cfg, "matched": "sdxl_default"}


# ── Prompt rewriter ─────────────────────────────────────────────────

#: Light keyword routing for "intent" detection. No model required.
_INTENT_KEYWORDS = {
    "logo":     ("logo", "wordmark", "monogram", "brandmark"),
    "sticker":  ("sticker", "die-cut", "decal"),
    "portrait": ("portrait", "headshot", "face of", "selfie"),
    "product":  ("product shot", "ecommerce", "studio shot"),
    "social":   ("instagram", "tiktok", "facebook ad", "social post"),
}


def detect_intent(prompt: str) -> Optional[str]:
    p = (prompt or "").lower()
    for intent, kws in _INTENT_KEYWORDS.items():
        if any(k in p for k in kws):
            return intent
    return None


@dataclass
class RewrittenPrompt:
    positive: str
    negative: str
    intent: Optional[str]
    style_tokens: List[str] = field(default_factory=list)
    raw: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def rewrite_prompt(
    prompt: str,
    *,
    model_family: str = "sdxl",
    style_sheet: Optional["StyleSheet"] = None,
    explicit_intent: Optional[str] = None,
) -> RewrittenPrompt:
    """Structure a freeform prompt and inject project style if provided.

    Conservative: we never *replace* user words, we only append style and
    quality tokens. If a banned token from the style sheet is present,
    it's removed (case-insensitive whole-word).
    """
    raw = (prompt or "").strip()
    intent = explicit_intent or detect_intent(raw)
    style_tokens: List[str] = []
    cleaned = raw

    if style_sheet:
        cleaned = style_sheet.strip_banned(cleaned)
        style_tokens.extend(style_sheet.style_tokens())

    # quality suffix — model-aware
    fam = (model_family or "").lower()
    if fam.startswith("flux"):
        quality = "highly detailed, sharp focus, professional photography"
    elif fam.startswith("sdxl"):
        quality = "8k, masterpiece, ultra detailed, sharp focus, dramatic lighting"
    else:
        quality = "high quality, detailed, sharp focus"

    parts = [cleaned]
    if style_tokens:
        parts.append(", ".join(style_tokens))
    parts.append(quality)
    positive = ", ".join(p for p in parts if p)

    negative = pick_negative_prompt(model_family, intent=intent)
    return RewrittenPrompt(
        positive=positive,
        negative=negative,
        intent=intent,
        style_tokens=style_tokens,
        raw=raw,
    )


# ── Project style sheet ─────────────────────────────────────────────

@dataclass
class StyleSheet:
    """A reusable per-project style sheet.

    Persisted as ``data/style_sheets/<project_id>.json``.
    """

    project_id: str
    palette: List[str] = field(default_factory=list)        # hex strings
    fonts: List[str] = field(default_factory=list)
    vibe: List[str] = field(default_factory=list)           # adjectives
    banned: List[str] = field(default_factory=list)         # case-insensitive

    def style_tokens(self) -> List[str]:
        out: List[str] = []
        if self.vibe:
            out.append(", ".join(self.vibe))
        if self.palette:
            out.append("brand palette " + " ".join(self.palette))
        if self.fonts:
            out.append("typography " + ", ".join(self.fonts))
        return out

    def strip_banned(self, text: str) -> str:
        if not self.banned or not text:
            return text
        out = text
        for term in self.banned:
            term = (term or "").strip()
            if not term:
                continue
            out = re.sub(rf"\b{re.escape(term)}\b", "", out, flags=re.IGNORECASE)
        # collapse double spaces / commas left behind
        out = re.sub(r"\s+,", ",", out)
        out = re.sub(r",\s*,", ",", out)
        out = re.sub(r"\s{2,}", " ", out).strip(" ,")
        return out

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class StyleSheetStore:
    """File-backed JSON store for :class:`StyleSheet`."""

    DEFAULT_ROOT = Path("data/style_sheets")

    _instance: Optional["StyleSheetStore"] = None
    _lock = threading.Lock()

    def __init__(self, root: Optional[Path] = None):
        self.root = (root or self.DEFAULT_ROOT).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_instance(cls, root: Optional[Path] = None) -> "StyleSheetStore":
        with cls._lock:
            target = (root or cls.DEFAULT_ROOT).resolve()
            if cls._instance is None or cls._instance.root != target:
                cls._instance = cls(root=target)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            cls._instance = None

    def _path(self, project_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", project_id) or "default"
        return self.root / f"{safe}.json"

    def get(self, project_id: str) -> Optional[StyleSheet]:
        p = self._path(project_id)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        return StyleSheet(
            project_id=data.get("project_id", project_id),
            palette=list(data.get("palette") or []),
            fonts=list(data.get("fonts") or []),
            vibe=list(data.get("vibe") or []),
            banned=list(data.get("banned") or []),
        )

    def save(self, sheet: StyleSheet) -> StyleSheet:
        p = self._path(sheet.project_id)
        p.write_text(json.dumps(sheet.to_dict(), indent=2), encoding="utf-8")
        return sheet

    def delete(self, project_id: str) -> bool:
        p = self._path(project_id)
        if p.exists():
            p.unlink()
            return True
        return False

    def list_projects(self) -> List[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))


# ── Seed variations & A/B grid ──────────────────────────────────────

def variation_seeds(seed: int, *, count: int = 4, spread: int = 1) -> List[int]:
    """Return ``count`` neighbouring seeds for a "more like this" UX.

    The seeds are deterministic so re-running variations on the same image
    twice yields the same set.
    """
    if count <= 0:
        return []
    base = int(seed) & 0xFFFFFFFF
    spread = max(1, int(spread))
    out: List[int] = []
    # pseudo-randomly walk away from the seed; deterministic via SHA256
    for i in range(1, count + 1):
        h = hashlib.sha256(f"{base}:{i}:{spread}".encode()).digest()
        delta = int.from_bytes(h[:4], "big") % (spread * 32 + 1) - spread * 16
        out.append((base + delta) & 0xFFFFFFFF)
    return out


def build_grid(
    *,
    samplers: Iterable[str] = ("dpmpp_2m", "euler_a"),
    cfgs: Iterable[float] = (4.0, 7.0),
    steps: Iterable[int] = (20, 30),
) -> List[Dict[str, object]]:
    """Return a sampler × CFG × steps grid (de-duplicated)."""
    grid = []
    seen = set()
    for s in samplers:
        for c in cfgs:
            for n in steps:
                key = (s, float(c), int(n))
                if key in seen:
                    continue
                seen.add(key)
                grid.append({"sampler": s, "cfg": float(c), "steps": int(n)})
    return grid


# ── Queue ETA ───────────────────────────────────────────────────────

def estimate_lane_eta_ms(
    telemetry: Dict[str, object],
    lane: str,
    *,
    fallback_ms: int = 30_000,
) -> Dict[str, object]:
    """Compute an ETA snapshot for a lane from a scheduler telemetry dict.

    Uses the lane's recent completed history average duration when
    available, otherwise ``fallback_ms``. Wait = avg_duration * (queued
    + max(0, in_flight − concurrency)).
    """
    lanes = telemetry.get("lanes", {}) if isinstance(telemetry, dict) else {}
    lane_info = lanes.get(lane) if isinstance(lanes, dict) else None
    history = telemetry.get("history", []) if isinstance(telemetry, dict) else []

    if not isinstance(lane_info, dict):
        return {
            "lane": lane, "queued": 0, "in_flight": 0,
            "avg_duration_ms": fallback_ms, "eta_ms": 0, "known": False,
        }

    durations = [
        float(rec.get("duration_ms") or 0)
        for rec in history
        if isinstance(rec, dict)
        and rec.get("lane") == lane
        and rec.get("status") == "done"
        and (rec.get("duration_ms") or 0) > 0
    ]
    avg = sum(durations) / len(durations) if durations else float(fallback_ms)

    queued = int(lane_info.get("queued") or 0)
    in_flight = int(lane_info.get("in_flight") or 0)
    capacity = int(lane_info.get("max_concurrent") or 1)
    over = max(0, in_flight - capacity)
    eta = avg * (queued + over)
    return {
        "lane": lane,
        "queued": queued,
        "in_flight": in_flight,
        "max_concurrent": capacity,
        "avg_duration_ms": int(avg),
        "samples": len(durations),
        "eta_ms": int(eta),
        "known": bool(durations),
    }


# ── "Print this" pipeline planner ──────────────────────────────────

def plan_print_this(
    *,
    artifact_id: str,
    intent: str = "keychain",
    material: str = "PLA",
    color: str = "black",
) -> List[Dict[str, object]]:
    """Plan the chain of ops needed to take an image to a printable STL.

    Returns a list of structured steps the workflow engine can execute
    via existing endpoints. Pure data — no side effects.
    """
    steps: List[Dict[str, object]] = [
        {"op": "remove_background",
         "input": artifact_id, "tool": "rmbg"},
        {"op": "vectorize",
         "input": "$prev",      "tool": "potrace",
         "params": {"posterize_levels": 2}},
    ]
    intent = (intent or "keychain").lower()
    if intent == "keychain":
        steps += [
            {"op": "extrude_svg",
             "input": "$prev", "tool": "stl_emboss",
             "params": {"thickness_mm": 4.0, "ring_diameter_mm": 5.0}},
        ]
    elif intent == "plaque":
        steps += [
            {"op": "extrude_svg",
             "input": "$prev", "tool": "stl_emboss",
             "params": {"thickness_mm": 8.0, "border_mm": 5.0}},
        ]
    elif intent == "sticker":
        # Stickers don't need STL — return a print-ready PNG instead
        return [
            {"op": "remove_background", "input": artifact_id, "tool": "rmbg"},
            {"op": "export_print_png",  "input": "$prev",     "tool": "rasterize",
             "params": {"dpi": 300}},
        ]
    else:  # generic emboss
        steps += [
            {"op": "extrude_svg",
             "input": "$prev", "tool": "stl_emboss",
             "params": {"thickness_mm": 3.0}},
        ]
    steps += [
        {"op": "cad_qa",
         "input": "$prev", "tool": "cad_qa.run_qa"},
        {"op": "slice",
         "input": "$prev", "tool": "slicer",
         "params": {"material": material, "color": color}},
        {"op": "queue_print",
         "input": "$prev", "tool": "printer.queue"},
    ]
    return steps
