"""
Edison Swarm Memory Safety.

Provides utilities for memory-safe swarm execution:
- Time-sliced execution: group agents by model, load one heavy model at a time
- Degraded mode: when memory is tight, use single model for all agents
- Vision-on-demand: load vision model only when images are present

Integrates with ModelManager v2 MemoryGate when available.
"""

import gc
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────

# Minimum free VRAM (approx) to consider loading another heavy model
MIN_FREE_VRAM_MB = 3000

# If memory is this tight, switch to degraded (single-model) mode
DEGRADED_THRESHOLD_MB = 5000


# ── Memory probing (standalone, no hard dependency on v2) ────────────────

def _get_free_vram_mb() -> float:
    """Best-effort free VRAM probe. Returns 0 if no GPU info available."""
    try:
        import pynvml
        pynvml.nvmlInit()
        total = 0.0
        for i in range(pynvml.nvmlDeviceGetCount()):
            info = pynvml.nvmlDeviceGetMemoryInfo(
                pynvml.nvmlDeviceGetHandleByIndex(i)
            )
            total += info.free / (1024 ** 2)
        pynvml.nvmlShutdown()
        return total
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            total = 0.0
            for i in range(torch.cuda.device_count()):
                free, _ = torch.cuda.mem_get_info(i)
                total += free / (1024 ** 2)
            return total
    except Exception:
        pass
    return 0.0


def _flush_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ── Swarm Memory Policy ─────────────────────────────────────────────────

class SwarmMemoryPolicy:
    """
    Decides how to run swarm agents memory-safely.

    Modes:
    - NORMAL:    All assigned models are loaded (existing behaviour)
    - TIME_SLICE: Group agents by model tag, load one heavy model at a time
    - DEGRADED:  Use a single model for all agents
    """

    def __init__(self):
        pass

    def assess(self, agents: List[dict],
               loaded_models: Dict[str, bool]) -> str:
        """
        Decide the execution mode.

        Args:
            agents: List of agent dicts with 'model' and 'model_name' keys.
            loaded_models: Map of model_key → is_loaded (e.g. {"fast": True, "deep": True}).

        Returns:
            "normal", "time_slice", or "degraded"
        """
        free_vram = _get_free_vram_mb()

        # Count how many distinct heavy models the agents want
        model_tags = set()
        for a in agents:
            name = a.get("model_name", "")
            for tag in ["deep", "medium", "fast"]:
                if tag.lower() in name.lower():
                    model_tags.add(tag)
                    break

        heavy_needed = len([t for t in model_tags if t in ("deep", "medium")])

        if heavy_needed <= 1:
            return "normal"

        # Multiple heavy models needed — check if we can afford them
        if free_vram > DEGRADED_THRESHOLD_MB * 2:
            return "normal"
        elif free_vram > DEGRADED_THRESHOLD_MB:
            return "time_slice"
        else:
            return "degraded"


def group_agents_by_model(agents: List[dict]) -> Dict[str, List[dict]]:
    """
    Group agents by their model tag for time-sliced execution.

    Returns: dict mapping model_tag → [agents]
    """
    groups: Dict[str, List[dict]] = {}
    for agent in agents:
        name = agent.get("model_name", "Unknown")
        tag = "fast"  # default
        for t in ["deep", "medium", "fast"]:
            if t.lower() in name.lower():
                tag = t
                break
        groups.setdefault(tag, []).append(agent)
    return groups


def apply_degraded_mode(agents: List[dict],
                        fallback_model: Any,
                        fallback_name: str) -> List[dict]:
    """
    Replace all agent models with a single fallback model.
    Used when memory is too tight for multiple models.

    Returns: modified agents list (mutated in place).
    """
    for agent in agents:
        agent["model"] = fallback_model
        agent["model_name"] = f"{fallback_name} (degraded)"
    logger.info(f"Swarm degraded mode: all {len(agents)} agents using {fallback_name}")
    return agents


def should_load_vision(has_images: bool, request_message: str = "") -> bool:
    """
    Vision-on-demand rule: only load vision model when images are present
    or explicitly requested.
    """
    if has_images:
        return True
    msg = request_message.lower()
    vision_keywords = [
        "describe this image", "what do you see", "analyze this photo",
        "look at this", "what is in this picture", "vision", "describe the image",
        "what's in this image", "read this screenshot",
    ]
    return any(kw in msg for kw in vision_keywords)


# ── Singleton ────────────────────────────────────────────────────────────

_policy = None

def get_swarm_memory_policy() -> SwarmMemoryPolicy:
    global _policy
    if _policy is None:
        _policy = SwarmMemoryPolicy()
    return _policy
