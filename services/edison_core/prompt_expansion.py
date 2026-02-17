"""
Prompt Expansion and Style Profile System for EDISON Image Generation
Improves short prompts with composition, lighting, and material details.
Loads style profiles from config/style_profiles/*.yaml.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
STYLE_PROFILES_DIR = REPO_ROOT / "config" / "style_profiles"

# ── Prompt Enhancement Keywords ──────────────────────────────────────────

COMPOSITION_HINTS = [
    "rule of thirds", "centered composition", "dynamic angle",
    "bird's eye view", "worm's eye view", "symmetrical", "dutch angle",
    "wide shot", "close-up", "medium shot", "establishing shot",
]

LIGHTING_HINTS = [
    "natural lighting", "golden hour", "studio lighting",
    "dramatic lighting", "soft diffused light", "rim lighting",
    "volumetric lighting", "backlit", "chiaroscuro",
]

MATERIAL_HINTS = [
    "fine detail", "sharp focus", "high resolution",
    "intricate texture", "smooth gradients", "rich colors",
]

# Short prompt threshold (word count)
SHORT_PROMPT_THRESHOLD = 8


def load_style_profiles() -> Dict[str, Dict[str, Any]]:
    """Load all style profiles from config/style_profiles/*.yaml."""
    profiles = {}
    if not STYLE_PROFILES_DIR.exists():
        return profiles

    for f in STYLE_PROFILES_DIR.glob("*.yaml"):
        try:
            data = yaml.safe_load(f.read_text())
            if data and "name" in data:
                profiles[data["name"]] = data
        except Exception as e:
            logger.warning(f"Failed to load style profile {f}: {e}")

    logger.info(f"Loaded {len(profiles)} style profiles: {list(profiles.keys())}")
    return profiles


_cached_profiles: Optional[Dict[str, Dict[str, Any]]] = None


def get_style_profiles() -> Dict[str, Dict[str, Any]]:
    """Get cached style profiles (loaded once)."""
    global _cached_profiles
    if _cached_profiles is None:
        _cached_profiles = load_style_profiles()
    return _cached_profiles


def expand_prompt(
    prompt: str,
    style: Optional[str] = None,
    enhance: bool = True,
) -> Tuple[str, str]:
    """
    Expand a short image prompt with composition, lighting, and material hints.
    Returns (expanded_prompt, negative_prompt).
    """
    profiles = get_style_profiles()
    negative = ""

    # Apply style profile if specified
    if style and style in profiles:
        profile = profiles[style]
        suffix = profile.get("prompt_suffix", "").strip()
        negative = profile.get("negative_prompt", "").strip()
        if suffix and suffix not in prompt:
            prompt = f"{prompt}, {suffix}"
        logger.info(f"Applied style profile: {style}")
        return prompt, negative

    # Only enhance short prompts
    words = prompt.split()
    if not enhance or len(words) > SHORT_PROMPT_THRESHOLD:
        return prompt, negative

    # Check if prompt already has quality/style keywords
    existing_lower = prompt.lower()
    has_lighting = any(lh in existing_lower for lh in [
        "lighting", "light", "backlit", "golden hour",
        "volumetric", "dramatic", "soft",
    ])
    has_composition = any(ch in existing_lower for ch in [
        "close-up", "wide shot", "angle", "view",
        "composition", "centered", "symmetr",
    ])
    has_quality = any(qh in existing_lower for qh in [
        "detailed", "quality", "resolution", "sharp",
        "professional", "8k", "4k", "uhd",
    ])

    additions = []
    if not has_quality:
        additions.append("highly detailed")
    if not has_lighting:
        additions.append("professional lighting")
    if not has_composition:
        additions.append("balanced composition")

    if additions:
        prompt = f"{prompt}, {', '.join(additions)}"

    return prompt, negative


def get_style_defaults(style: str) -> Dict[str, Any]:
    """Get generation parameter defaults for a style profile."""
    profiles = get_style_profiles()
    if style in profiles:
        return profiles[style].get("defaults", {})
    return {}
