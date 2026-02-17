"""
Proactive Suggestion Engine for Edison.

Generates non-intrusive suggestions when conditions are met:
  - Long-running task detected
  - Repeated errors detected
  - User idle after generation failure
  - Large memory recall opportunity detected

Suggestions are logged and returned to the caller — never injected
into the user's conversation unsolicited.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Suggestion:
    """A proactive suggestion for the user."""
    id: str
    category: str          # "error_help", "optimization", "memory", "idle_hint", "system"
    message: str           # Human-readable suggestion text
    confidence: float      # 0.0-1.0
    timestamp: float = field(default_factory=time.time)
    dismissed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "message": self.message,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "dismissed": self.dismissed,
            "metadata": self.metadata,
        }


class SuggestionEngine:
    """Lightweight rule-based suggestion generator."""

    def __init__(self):
        self._suggestions: List[Suggestion] = []
        self._MAX_SUGGESTIONS = 50
        self._counter = 0
        logger.info("SuggestionEngine initialized")

    def _next_id(self) -> str:
        self._counter += 1
        return f"sug_{self._counter}"

    def _add(self, category: str, message: str, confidence: float,
             metadata: Optional[Dict[str, Any]] = None) -> Suggestion:
        sug = Suggestion(
            id=self._next_id(),
            category=category,
            message=message,
            confidence=confidence,
            metadata=metadata or {},
        )
        self._suggestions.append(sug)
        if len(self._suggestions) > self._MAX_SUGGESTIONS:
            self._suggestions = self._suggestions[-self._MAX_SUGGESTIONS:]
        logger.info(f"SUGGESTION [{category}]: {message} (conf={confidence:.2f})")
        return sug

    # ── Trigger methods ──────────────────────────────────────────────────

    def check_repeated_errors(
        self, error_count: int, last_error: Optional[str] = None
    ) -> Optional[Suggestion]:
        """Suggest help when the user encounters repeated errors."""
        if error_count >= 3:
            msg = (
                f"I notice you've encountered {error_count} errors in this session. "
                f"Would you like me to help debug the issue"
            )
            if last_error:
                msg += f" (last error: {last_error[:100]})"
            msg += "?"
            return self._add("error_help", msg, min(0.5 + error_count * 0.1, 0.95))
        return None

    def check_long_running_task(
        self, task_name: str, elapsed_seconds: float, expected_seconds: float = 300
    ) -> Optional[Suggestion]:
        """Suggest alternatives when a task is taking longer than expected."""
        if elapsed_seconds > expected_seconds * 1.5:
            ratio = elapsed_seconds / expected_seconds
            msg = (
                f"The {task_name} task has been running for "
                f"{elapsed_seconds / 60:.1f} minutes ({ratio:.1f}× the expected time). "
                f"Would you like to check its status or try an alternative approach?"
            )
            return self._add("optimization", msg, min(0.5 + ratio * 0.1, 0.90),
                             {"task": task_name, "elapsed_s": elapsed_seconds})
        return None

    def check_idle_after_failure(
        self, idle_seconds: float, last_status: str
    ) -> Optional[Suggestion]:
        """Suggest re-trying or alternative when user is idle after a failure."""
        if last_status == "error" and idle_seconds > 60:
            msg = (
                "It looks like the last generation failed and you've been idle. "
                "Would you like me to retry with different settings, or help "
                "troubleshoot the issue?"
            )
            return self._add("idle_hint", msg, 0.65,
                             {"idle_s": idle_seconds})
        return None

    def check_memory_opportunity(
        self, message: str, recall_count: int
    ) -> Optional[Suggestion]:
        """Suggest when many relevant memories are found."""
        if recall_count >= 5:
            msg = (
                f"I found {recall_count} relevant memories related to your query. "
                f"Would you like a summary of what I remember about this topic?"
            )
            return self._add("memory", msg, 0.60,
                             {"recall_count": recall_count})
        return None

    def check_system_resource_warning(
        self, free_gpu_mb: int = 0, free_disk_gb: float = 0,
    ) -> Optional[Suggestion]:
        """Warn about low resources before the user runs into problems."""
        if 0 < free_gpu_mb < 2048:
            return self._add(
                "system",
                f"GPU memory is low ({free_gpu_mb}MB free). "
                f"Large generation tasks may fail. Consider closing other GPU programs.",
                0.80, {"free_gpu_mb": free_gpu_mb}
            )
        if 0 < free_disk_gb < 5.0:
            return self._add(
                "system",
                f"Disk space is low ({free_disk_gb:.1f}GB free). "
                f"Consider cleaning up old outputs in the outputs/ directory.",
                0.75, {"free_disk_gb": free_disk_gb}
            )
        return None

    # ── Evaluate all triggers at once ────────────────────────────────────

    def evaluate(
        self,
        error_count: int = 0,
        last_error: Optional[str] = None,
        idle_seconds: float = 0,
        last_status: str = "",
        recall_count: int = 0,
        message: str = "",
        free_gpu_mb: int = -1,
        free_disk_gb: float = -1,
    ) -> List[Suggestion]:
        """Run all suggestion triggers and return any that fire."""
        fired: List[Suggestion] = []
        for fn, kwargs in [
            (self.check_repeated_errors, {"error_count": error_count, "last_error": last_error}),
            (self.check_idle_after_failure, {"idle_seconds": idle_seconds, "last_status": last_status}),
            (self.check_memory_opportunity, {"message": message, "recall_count": recall_count}),
            (self.check_system_resource_warning, {"free_gpu_mb": free_gpu_mb, "free_disk_gb": free_disk_gb}),
        ]:
            sug = fn(**kwargs)
            if sug:
                fired.append(sug)
        return fired

    # ── Query / dismiss ──────────────────────────────────────────────────

    def get_pending(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Return the most recent un-dismissed suggestions."""
        return [
            s.to_dict()
            for s in reversed(self._suggestions)
            if not s.dismissed
        ][:limit]

    def dismiss(self, suggestion_id: str) -> bool:
        for s in self._suggestions:
            if s.id == suggestion_id:
                s.dismissed = True
                return True
        return False

    def clear(self):
        self._suggestions.clear()


# ── Singleton ────────────────────────────────────────────────────────────

_engine: Optional[SuggestionEngine] = None


def get_suggestion_engine() -> SuggestionEngine:
    global _engine
    if _engine is None:
        _engine = SuggestionEngine()
    return _engine
