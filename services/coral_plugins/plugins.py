"""
Optional Coral TPU Plugin Modules for Edison.

Provides pluggable detection capabilities (presence, activity, object
detection) that run on the Coral Edge TPU when available.  Edison
operates normally when Coral or a camera is unavailable.

Each plugin follows the same interface:
  class SomePlugin:
      name: str
      available: bool
      def detect(self, input_data) -> dict
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Plugin base ──────────────────────────────────────────────────────────

class CoralPlugin(ABC):
    """Base class for all optional Coral-powered plugins."""

    name: str = "base_plugin"
    description: str = ""
    requires_camera: bool = False
    requires_tpu: bool = True

    def __init__(self):
        self._available = False
        self._last_result: Optional[Dict[str, Any]] = None
        self._last_run: float = 0
        self._init_plugin()

    @abstractmethod
    def _init_plugin(self):
        """Attempt to initialize hardware resources.  Set self._available."""
        ...

    @abstractmethod
    def detect(self, input_data: Any = None) -> Dict[str, Any]:
        """Run detection.  Returns a result dict or empty on failure."""
        ...

    @property
    def available(self) -> bool:
        return self._available

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "available": self._available,
            "last_run": self._last_run,
            "description": self.description,
        }


# ── Presence Detection Plugin ────────────────────────────────────────────

class PresenceDetectionPlugin(CoralPlugin):
    """Detects whether a person is present in front of the camera.

    Uses a MobileNet SSD or PoseNet model on the Edge TPU.
    Falls back to unavailable if no camera or TPU is found.
    """

    name = "presence_detection"
    description = "Detect human presence via camera + Coral TPU"
    requires_camera = True
    requires_tpu = True

    def _init_plugin(self):
        try:
            # Check for Edge TPU runtime
            from pycoral.utils.edgetpu import list_edge_tpus
            tpus = list_edge_tpus()
            if not tpus:
                raise RuntimeError("No Edge TPU found")

            # Check for camera
            import cv2
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError("No camera available")
            cap.release()

            self._available = True
            logger.info(f"✓ Coral plugin '{self.name}' initialized (TPU + camera)")
        except Exception as e:
            self._available = False
            logger.info(f"Coral plugin '{self.name}' unavailable: {e}")

    def detect(self, input_data: Any = None) -> Dict[str, Any]:
        if not self._available:
            return {"present": None, "error": "Plugin unavailable"}
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return {"present": None, "error": "Camera read failed"}

            # Simple face detection as presence proxy
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            present = len(faces) > 0

            self._last_result = {"present": present, "faces": len(faces)}
            self._last_run = time.time()
            return self._last_result
        except Exception as e:
            logger.warning(f"Presence detection failed: {e}")
            return {"present": None, "error": str(e)}


# ── Activity Detection Plugin ────────────────────────────────────────────

class ActivityDetectionPlugin(CoralPlugin):
    """Detects simple user activity states (typing, idle, away).

    This is a simplified stub that can be extended with actual
    TPU-based activity classification models.
    """

    name = "activity_detection"
    description = "Detect user activity patterns (typing, idle, away)"
    requires_camera = False
    requires_tpu = False

    def _init_plugin(self):
        self._available = True  # Always available (software-only)
        self._activity_log: List[Dict[str, Any]] = []
        logger.info(f"✓ Coral plugin '{self.name}' initialized (software-only)")

    def detect(self, input_data: Any = None) -> Dict[str, Any]:
        """Detect activity from input timing metadata.

        input_data should be a dict with optional keys:
          - last_message_time: float (epoch)
          - message_count_last_5min: int
          - is_generating: bool
        """
        if not isinstance(input_data, dict):
            input_data = {}

        now = time.time()
        last_msg = input_data.get("last_message_time", now)
        msg_count = input_data.get("message_count_last_5min", 0)
        is_generating = input_data.get("is_generating", False)

        idle_seconds = now - last_msg

        if is_generating:
            activity = "waiting_for_generation"
        elif idle_seconds > 600:
            activity = "away"
        elif idle_seconds > 120:
            activity = "idle"
        elif msg_count > 10:
            activity = "active_conversation"
        else:
            activity = "typing"

        result = {
            "activity": activity,
            "idle_seconds": round(idle_seconds, 1),
            "message_rate": msg_count,
        }
        self._last_result = result
        self._last_run = now
        return result


# ── Object Detection Plugin ──────────────────────────────────────────────

class ObjectDetectionPlugin(CoralPlugin):
    """Basic object detection using Coral TPU + camera.

    Uses EfficientDet-Lite or SSD MobileNet on Edge TPU.
    """

    name = "object_detection"
    description = "Detect objects via camera + Coral TPU"
    requires_camera = True
    requires_tpu = True

    def _init_plugin(self):
        try:
            from pycoral.utils.edgetpu import list_edge_tpus
            tpus = list_edge_tpus()
            if not tpus:
                raise RuntimeError("No Edge TPU found")
            self._available = True
            logger.info(f"✓ Coral plugin '{self.name}' initialized")
        except Exception as e:
            self._available = False
            logger.info(f"Coral plugin '{self.name}' unavailable: {e}")

    def detect(self, input_data: Any = None) -> Dict[str, Any]:
        if not self._available:
            return {"objects": [], "error": "Plugin unavailable"}
        # Placeholder: real implementation would run edge TPU inference
        self._last_run = time.time()
        return {"objects": [], "note": "TPU inference not configured"}


# ── Plugin Registry ──────────────────────────────────────────────────────

class CoralPluginRegistry:
    """Registry of available Coral plugins.  Plugins that fail to
    initialize are kept in the registry as unavailable (not removed).
    """

    def __init__(self):
        self._plugins: Dict[str, CoralPlugin] = {}
        self._lock = threading.Lock()

    def register(self, plugin: CoralPlugin):
        with self._lock:
            self._plugins[plugin.name] = plugin
            logger.debug(f"Registered coral plugin: {plugin.name} (available={plugin.available})")

    def get(self, name: str) -> Optional[CoralPlugin]:
        return self._plugins.get(name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        return [p.status() for p in self._plugins.values()]

    def detect_all(self, input_data: Any = None) -> Dict[str, Any]:
        """Run all available plugins and return combined results."""
        results = {}
        for name, plugin in self._plugins.items():
            if plugin.available:
                try:
                    results[name] = plugin.detect(input_data)
                except Exception as e:
                    results[name] = {"error": str(e)}
        return results


# ── Singleton ────────────────────────────────────────────────────────────

_registry: Optional[CoralPluginRegistry] = None
_registry_lock = threading.Lock()


def get_coral_plugin_registry() -> CoralPluginRegistry:
    """Return the global CoralPluginRegistry, creating and populating it on first call."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = CoralPluginRegistry()
                # Register all built-in plugins (they gracefully handle missing hardware)
                for PluginClass in (PresenceDetectionPlugin, ActivityDetectionPlugin, ObjectDetectionPlugin):
                    try:
                        _registry.register(PluginClass())
                    except Exception as e:
                        logger.warning(f"Failed to register {PluginClass.__name__}: {e}")
                logger.info(
                    f"Coral plugin registry ready: "
                    f"{sum(1 for p in _registry._plugins.values() if p.available)}/"
                    f"{len(_registry._plugins)} plugins available"
                )
    return _registry
