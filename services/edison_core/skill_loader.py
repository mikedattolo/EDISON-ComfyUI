"""Dynamic skill/plugin loader for Edison tools."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
import threading
import time
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional

from .tool_framework import ToolRegistry

logger = logging.getLogger(__name__)


class SkillLoader:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        skills_dir: Path,
        config_getter: Callable[[], Dict[str, Any]],
        poll_interval_sec: int = 3,
    ):
        self._tool_registry = tool_registry
        self._skills_dir = skills_dir
        self._config_getter = config_getter
        self._poll_interval_sec = max(1, int(poll_interval_sec))
        self._loaded: Dict[str, Dict[str, Any]] = {}
        self._watch_thread: Optional[threading.Thread] = None
        self._watch_stop = threading.Event()
        self._lock = threading.Lock()

    def _skill_runtime_config(self) -> Dict[str, Any]:
        cfg = self._config_getter() or {}
        ed = cfg.get("edison", {}) if isinstance(cfg, dict) else {}
        return ed.get("skills", {}) if isinstance(ed, dict) else {}

    def _is_skill_allowed(self, metadata: Dict[str, Any]) -> tuple[bool, str]:
        runtime = self._skill_runtime_config()
        disabled = set(runtime.get("disabled_skills", []) or [])
        allowed_permissions = set(runtime.get("allowed_permissions", []) or [])
        name = str(metadata.get("name") or "")

        if name and name in disabled:
            return False, f"Skill '{name}' is disabled by config"
        if metadata.get("enabled") is False:
            return False, f"Skill '{name}' is marked disabled"

        required = set(metadata.get("required_permissions", []) or [])
        if required and not required.issubset(allowed_permissions):
            missing = sorted(required - allowed_permissions)
            return False, f"Skill '{name}' missing permissions: {', '.join(missing)}"

        return True, ""

    def _load_module_from_file(self, path: Path) -> ModuleType:
        module_name = f"services.edison_core.skills.{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load skill module spec for {path.name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def load_all(self) -> Dict[str, Any]:
        self._skills_dir.mkdir(parents=True, exist_ok=True)
        loaded = []
        skipped = []

        with self._lock:
            for py_file in sorted(self._skills_dir.glob("*_skill.py")):
                try:
                    mod = self._load_module_from_file(py_file)
                    metadata = getattr(mod, "SKILL_METADATA", {"name": py_file.stem})
                    allowed, reason = self._is_skill_allowed(metadata)
                    if not allowed:
                        skipped.append({"module": py_file.name, "reason": reason})
                        continue

                    register = getattr(mod, "register", None)
                    if not callable(register):
                        skipped.append({"module": py_file.name, "reason": "No register(tool_registry) function"})
                        continue

                    registered_tools = register(self._tool_registry) or []
                    self._loaded[py_file.stem] = {
                        "metadata": metadata,
                        "tools": registered_tools,
                        "path": str(py_file),
                        "mtime": py_file.stat().st_mtime,
                    }
                    loaded.append({"module": py_file.name, "tools": registered_tools})
                    logger.info("✓ Skill loaded: %s (%s)", py_file.stem, ", ".join(registered_tools) or "no tools")
                except Exception as e:
                    skipped.append({"module": py_file.name, "reason": str(e)})
                    logger.warning("⚠ Failed to load skill %s: %s", py_file.name, e)

        return {"loaded": loaded, "skipped": skipped}

    def reload_if_changed(self) -> List[str]:
        changed: List[str] = []
        with self._lock:
            known = dict(self._loaded)

        for py_file in sorted(self._skills_dir.glob("*_skill.py")):
            stem = py_file.stem
            mtime = py_file.stat().st_mtime
            if stem not in known or mtime > float(known[stem].get("mtime", 0)):
                changed.append(stem)

        if not changed:
            return []

        # Simple strategy: re-run loader; tools are additive in current registry model.
        self.load_all()
        return changed

    def start_watcher(self):
        if self._watch_thread and self._watch_thread.is_alive():
            return

        self._watch_stop.clear()

        def _watch_loop():
            while not self._watch_stop.is_set():
                try:
                    changed = self.reload_if_changed()
                    if changed:
                        logger.info("Skill reload detected for: %s", ", ".join(changed))
                except Exception as e:
                    logger.debug("Skill watcher iteration failed: %s", e)
                self._watch_stop.wait(self._poll_interval_sec)

        self._watch_thread = threading.Thread(target=_watch_loop, daemon=True, name="skill-loader-watch")
        self._watch_thread.start()

    def stop_watcher(self):
        self._watch_stop.set()
        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=2)

    def list_skills(self) -> List[Dict[str, Any]]:
        with self._lock:
            out = []
            for key, item in self._loaded.items():
                meta = item.get("metadata", {})
                out.append(
                    {
                        "module": key,
                        "name": meta.get("name", key),
                        "description": meta.get("description", ""),
                        "tools": item.get("tools", []),
                        "version": meta.get("version", ""),
                    }
                )
            return out
