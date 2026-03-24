from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict


CleanupCallback = Callable[[], Any]


class IdleResourceManager:
    """Coordinate idle cleanup for non-LLM resources.

    The manager tracks active heavy tasks such as image generation and swarm.
    When the system has been idle for a configured grace period, registered
    cleanup callbacks are invoked to release non-LLM resources.
    """

    def __init__(self, idle_seconds: float = 45.0):
        self._idle_seconds = max(0.0, float(idle_seconds))
        self._lock = threading.RLock()
        self._active_tasks: Dict[str, int] = {}
        self._cleanup_callbacks: Dict[str, CleanupCallback] = {}
        now = time.time()
        self._last_activity_at = now
        self._idle_since = now
        self._last_cleanup_at = 0.0
        self._last_cleanup_result: Dict[str, Any] = {"ran": False}

    def set_idle_seconds(self, idle_seconds: float) -> None:
        with self._lock:
            self._idle_seconds = max(0.0, float(idle_seconds))

    def register_cleanup(self, name: str, callback: CleanupCallback) -> None:
        with self._lock:
            self._cleanup_callbacks[name] = callback

    def begin_task(self, task_name: str) -> None:
        now = time.time()
        with self._lock:
            self._active_tasks[task_name] = self._active_tasks.get(task_name, 0) + 1
            self._last_activity_at = now
            self._idle_since = 0.0

    def end_task(self, task_name: str) -> None:
        now = time.time()
        with self._lock:
            count = self._active_tasks.get(task_name, 0)
            if count <= 1:
                self._active_tasks.pop(task_name, None)
            else:
                self._active_tasks[task_name] = count - 1
            self._last_activity_at = now
            if not self._active_tasks:
                self._idle_since = now

    def force_idle(self, task_name: str | None = None) -> None:
        now = time.time()
        with self._lock:
            if task_name is None:
                self._active_tasks.clear()
            else:
                self._active_tasks.pop(task_name, None)
            self._last_activity_at = now
            if not self._active_tasks:
                self._idle_since = now

    def touch(self) -> None:
        now = time.time()
        with self._lock:
            self._last_activity_at = now
            if not self._active_tasks:
                self._idle_since = now

    def total_active_tasks(self) -> int:
        with self._lock:
            return sum(self._active_tasks.values())

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            idle_for = 0.0
            if not self._active_tasks and self._idle_since > 0:
                idle_for = max(0.0, time.time() - self._idle_since)
            return {
                "idle_seconds": self._idle_seconds,
                "active_tasks": dict(self._active_tasks),
                "total_active_tasks": sum(self._active_tasks.values()),
                "last_activity_at": self._last_activity_at,
                "idle_since": self._idle_since,
                "idle_for_seconds": idle_for,
                "last_cleanup_at": self._last_cleanup_at,
                "last_cleanup_result": dict(self._last_cleanup_result),
            }

    def cleanup_if_idle(self, force: bool = False) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            active_tasks = dict(self._active_tasks)
            idle_for = 0.0
            if not active_tasks and self._idle_since > 0:
                idle_for = max(0.0, now - self._idle_since)

            if not force:
                if active_tasks:
                    result = {
                        "ran": False,
                        "reason": "active_tasks",
                        "active_tasks": active_tasks,
                    }
                    self._last_cleanup_result = result
                    return result
                if idle_for < self._idle_seconds:
                    result = {
                        "ran": False,
                        "reason": "idle_grace_period",
                        "idle_for_seconds": idle_for,
                        "required_idle_seconds": self._idle_seconds,
                    }
                    self._last_cleanup_result = result
                    return result

            callbacks = list(self._cleanup_callbacks.items())

        callbacks_result: Dict[str, Any] = {}
        for name, callback in callbacks:
            try:
                callbacks_result[name] = callback()
            except Exception as exc:
                callbacks_result[name] = {"error": str(exc)}

        result = {
            "ran": True,
            "force": force,
            "idle_for_seconds": idle_for,
            "callbacks": callbacks_result,
        }
        with self._lock:
            self._last_cleanup_at = time.time()
            self._last_cleanup_result = result
        return result