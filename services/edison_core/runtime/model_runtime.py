"""
model_runtime.py — Task-aware model selection, fallback, and profile management.

Replaces brittle hardcoded model switching with a profile-based system.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelProfile:
    """Describes a model's capabilities and when to use it."""
    name: str
    target: str                    # fast | medium | deep | reasoning | vision | vision_code
    context_window: int = 4096
    strengths: List[str] = field(default_factory=list)
    preferred_tasks: List[str] = field(default_factory=list)
    latency_class: str = "normal"  # fast | normal | slow
    supports_tools: bool = False
    supports_vision: bool = False
    gpu_layers: int = -1
    fallback_targets: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "target": self.target,
            "context_window": self.context_window,
            "strengths": self.strengths,
            "preferred_tasks": self.preferred_tasks,
            "latency_class": self.latency_class,
            "supports_tools": self.supports_tools,
            "supports_vision": self.supports_vision,
        }


# ── Default profiles ─────────────────────────────────────────────────
DEFAULT_PROFILES: Dict[str, ModelProfile] = {
    "fast": ModelProfile(
        name="fast",
        target="fast",
        context_window=4096,
        strengths=["quick chat", "simple Q&A", "greetings"],
        preferred_tasks=["chat", "instant"],
        latency_class="fast",
        supports_tools=False,
        fallback_targets=["medium", "deep"],
    ),
    "medium": ModelProfile(
        name="medium",
        target="medium",
        context_window=4096,
        strengths=["tool use", "agent tasks", "search synthesis"],
        preferred_tasks=["agent"],
        latency_class="normal",
        supports_tools=True,
        fallback_targets=["fast", "deep"],
    ),
    "deep": ModelProfile(
        name="deep",
        target="deep",
        context_window=8192,
        strengths=["coding", "complex reasoning", "work mode", "swarm"],
        preferred_tasks=["code", "work", "swarm"],
        latency_class="slow",
        supports_tools=True,
        fallback_targets=["medium", "fast"],
    ),
    "reasoning": ModelProfile(
        name="reasoning",
        target="reasoning",
        context_window=8192,
        strengths=["deep analysis", "step-by-step thinking"],
        preferred_tasks=["reasoning", "thinking"],
        latency_class="slow",
        supports_tools=False,
        fallback_targets=["deep", "medium"],
    ),
    "vision": ModelProfile(
        name="vision",
        target="vision",
        context_window=4096,
        strengths=["image understanding", "visual Q&A"],
        preferred_tasks=["image"],
        latency_class="normal",
        supports_vision=True,
        fallback_targets=["vision_code"],
    ),
    "vision_code": ModelProfile(
        name="vision_code",
        target="vision_code",
        context_window=4096,
        strengths=["image to code", "UI mockup understanding"],
        preferred_tasks=["vision-to-code"],
        latency_class="normal",
        supports_vision=True,
        fallback_targets=["vision"],
    ),
}


class ModelResolver:
    """
    Resolves which model to use for a given task, with graceful fallback.
    
    This is designed to wrap around the existing model_manager_v2 while
    providing a cleaner task-aware interface.
    """

    def __init__(self, profiles: Optional[Dict[str, ModelProfile]] = None):
        self.profiles = profiles or dict(DEFAULT_PROFILES)
        self._available: Dict[str, bool] = {}  # target → loaded

    def set_available(self, target: str, available: bool) -> None:
        """Mark a model target as available or unavailable."""
        self._available[target] = available

    def is_available(self, target: str) -> bool:
        return self._available.get(target, False)

    def get_profile(self, target: str) -> Optional[ModelProfile]:
        return self.profiles.get(target)

    def resolve(
        self,
        target: str,
        task_type: Optional[str] = None,
    ) -> Optional[str]:
        """
        Resolve a model target to an available model, with fallback.
        Returns the target name of the model to use, or None if nothing available.
        """
        # Direct match
        if self.is_available(target):
            return target

        # Fallback chain
        profile = self.get_profile(target)
        if profile:
            for fb in profile.fallback_targets:
                if self.is_available(fb):
                    logger.info(f"Model {target} unavailable, falling back to {fb}")
                    return fb

        # Last resort: return any available model
        for t, avail in self._available.items():
            if avail:
                logger.warning(f"Model {target} unavailable, using any available: {t}")
                return t

        return None

    def best_for_task(self, task_type: str) -> Optional[str]:
        """Find the best available model for a given task type."""
        # Check profiles that list this task_type as preferred
        for profile in self.profiles.values():
            if task_type in profile.preferred_tasks and self.is_available(profile.target):
                return profile.target

        # Fall back to task-target map
        task_map = {
            "chat": "fast",
            "instant": "fast",
            "code": "deep",
            "work": "deep",
            "swarm": "deep",
            "agent": "medium",
            "reasoning": "reasoning",
            "thinking": "reasoning",
            "image": "vision",
            "vision-to-code": "vision_code",
        }
        preferred = task_map.get(task_type, "fast")
        return self.resolve(preferred, task_type=task_type)

    def get_context_limit(self, target: str) -> int:
        """Return the effective context window for a model target."""
        profile = self.get_profile(target)
        return profile.context_window if profile else 4096

    def list_profiles(self) -> List[dict]:
        return [p.to_dict() for p in self.profiles.values()]


# Module-level default resolver instance
_default_resolver = ModelResolver()


def get_resolver() -> ModelResolver:
    return _default_resolver


def configure_from_yaml(core_config: dict) -> None:
    """Update profiles from edison.yaml core config section."""
    r = _default_resolver
    ctx_overrides = {
        "fast": core_config.get("fast_n_ctx", 4096),
        "medium": core_config.get("medium_n_ctx", 4096),
        "deep": core_config.get("deep_n_ctx", 8192),
        "reasoning": core_config.get("reasoning_n_ctx", 8192),
        "vision": core_config.get("vision_n_ctx", 4096),
        "vision_code": core_config.get("vision_code_n_ctx", 4096),
    }
    for target, ctx in ctx_overrides.items():
        profile = r.get_profile(target)
        if profile:
            profile.context_window = ctx
