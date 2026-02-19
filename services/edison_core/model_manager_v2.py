"""
Memory-aware model manager for Edison  (v2).

Single source of truth for acquiring and releasing LLM models.

Implements:
- Resident set policy (one fast model always-resident, one heavy slot)
- Memory probing (RAM + VRAM)
- Fallback ladder (full → reduced GPU layers → reduced context → smaller model)
- Thread-safe, lazy-loading, graceful degradation
- ``resolve_model(target)`` — canonical way to get a Llama instance with fallback
- ``pre_heavy_task`` budget gate for ComfyUI / video / vision / 3D tasks
"""

import gc
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Re-export helpers so callers can ``from model_manager_v2 import get_memory_snapshot``
__all__ = [
    "ModelManager",
    "MemoryGate",
    "MemorySnapshot",
    "LoadedModel",
    "get_model_manager",
    "get_memory_gate",
    "get_memory_snapshot",
]


# ── Data Classes ─────────────────────────────────────────────────────────

@dataclass
class MemorySnapshot:
    """System memory state."""
    ram_total_mb: float = 0
    ram_available_mb: float = 0
    ram_used_mb: float = 0
    gpus: List[Dict[str, float]] = field(default_factory=list)  # [{total_mb, used_mb, free_mb}]
    timestamp: float = 0

    def total_vram_free_mb(self) -> float:
        return sum(g.get("free_mb", 0) for g in self.gpus)

    def total_vram_total_mb(self) -> float:
        return sum(g.get("total_mb", 0) for g in self.gpus)

    def to_dict(self) -> dict:
        return {
            "ram_total_mb": round(self.ram_total_mb, 1),
            "ram_available_mb": round(self.ram_available_mb, 1),
            "ram_used_mb": round(self.ram_used_mb, 1),
            "gpus": self.gpus,
            "total_vram_free_mb": round(self.total_vram_free_mb(), 1),
            "timestamp": self.timestamp,
        }


@dataclass
class LoadedModel:
    """Metadata for a loaded model."""
    key: str                    # e.g. "fast", "medium", "deep", "vision"
    path: str
    model: Any = None           # The actual Llama object
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    is_heavy: bool = False      # Is this a heavy-slot model?
    loaded_at: float = 0
    estimated_vram_mb: float = 0
    clip_path: Optional[str] = None  # For vision models


# ── Constants ────────────────────────────────────────────────────────────

HEAVY_KEYS = {"medium", "deep", "reasoning", "vision", "vision_code"}
FAST_KEY = "fast"

# Fallback ladder: try progressively lighter configurations
FALLBACK_GPU_LAYERS = [None, 32, 20, 12, 4, 0]  # None = use config default
FALLBACK_CTX_SIZES = [None, 4096, 2048, 1024]     # None = use config default


# ── Memory Probing ───────────────────────────────────────────────────────

def get_memory_snapshot() -> MemorySnapshot:
    """Probe system RAM and GPU VRAM."""
    snap = MemorySnapshot(timestamp=time.time())

    # RAM
    try:
        import psutil
        vm = psutil.virtual_memory()
        snap.ram_total_mb = vm.total / (1024 ** 2)
        snap.ram_available_mb = vm.available / (1024 ** 2)
        snap.ram_used_mb = vm.used / (1024 ** 2)
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"RAM probe failed: {e}")

    # VRAM via pynvml (most reliable)
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            snap.gpus.append({
                "index": i,
                "total_mb": round(info.total / (1024 ** 2), 1),
                "used_mb": round(info.used / (1024 ** 2), 1),
                "free_mb": round(info.free / (1024 ** 2), 1),
            })
        pynvml.nvmlShutdown()
        return snap
    except Exception:
        pass

    # Fallback: PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                snap.gpus.append({
                    "index": i,
                    "total_mb": round(total / (1024 ** 2), 1),
                    "used_mb": round((total - free) / (1024 ** 2), 1),
                    "free_mb": round(free / (1024 ** 2), 1),
                })
            return snap
    except Exception:
        pass

    # Fallback: nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for i, line in enumerate(result.stdout.strip().split("\n")):
                parts = [float(x.strip()) for x in line.split(",")]
                if len(parts) == 3:
                    snap.gpus.append({
                        "index": i,
                        "total_mb": parts[0],
                        "used_mb": parts[1],
                        "free_mb": parts[2],
                    })
    except Exception:
        pass

    return snap


def _estimate_model_vram_mb(model_path: str) -> float:
    """Rough estimate of VRAM needed for a GGUF model (≈85% of file size)."""
    try:
        size_bytes = os.path.getsize(model_path)
        return (size_bytes / (1024 ** 2)) * 0.85
    except Exception:
        return 0


def _flush_gpu():
    """Force GC + CUDA cache flush."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


# ── ModelManager 2.0 ─────────────────────────────────────────────────────

class ModelManager:
    """
    Memory-aware model manager.

    Policies:
    - The "fast" model is always-resident when loaded.
    - Only ONE heavy model (medium/deep/reasoning/vision/vision_code) occupies
      the heavy slot at a time.
    - Loading a new heavy model evicts the previous one.
    - Fallback ladder on OOM: reduce GPU layers → reduce context → try smaller model.
    """

    def __init__(self):
        self._models: Dict[str, LoadedModel] = {}
        self._lock = threading.Lock()
        self._heavy_slot: Optional[str] = None  # current heavy model key
        self._model_configs: Dict[str, dict] = {}  # key → {path, n_ctx, n_gpu_layers, ...}
        self._fallback_order: List[str] = ["deep", "medium", "fast"]

    # ── Configuration ────────────────────────────────────────────────

    def register_model(self, key: str, path: str, n_ctx: int = 4096,
                       n_gpu_layers: int = -1, tensor_split: Optional[list] = None,
                       clip_path: Optional[str] = None,
                       use_flash_attn: bool = False, **kwargs):
        """Register a model configuration (does not load it)."""
        self._model_configs[key] = {
            "path": path,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "tensor_split": tensor_split,
            "clip_path": clip_path,
            "use_flash_attn": use_flash_attn,
            **kwargs,
        }

    # ── Core Operations ──────────────────────────────────────────────

    def ensure_model(self, key: str) -> Optional[Any]:
        """
        Ensure a model is loaded and return the Llama instance.
        If it's a heavy model, evicts the current heavy-slot occupant first.
        Uses fallback ladder on failure.
        """
        with self._lock:
            # Already loaded?
            if key in self._models and self._models[key].model is not None:
                return self._models[key].model

            cfg = self._model_configs.get(key)
            if not cfg:
                logger.warning(f"Model '{key}' not registered")
                return None

            if not os.path.exists(cfg["path"]):
                logger.warning(f"Model file not found: {cfg['path']}")
                return None

            is_heavy = key in HEAVY_KEYS

            # Evict current heavy-slot occupant
            if is_heavy and self._heavy_slot and self._heavy_slot != key:
                self._unload_model_unsafe(self._heavy_slot)

            # Try loading with fallback ladder
            model = self._load_with_fallback(key, cfg)
            if model is not None:
                estimated_vram = _estimate_model_vram_mb(cfg["path"])
                self._models[key] = LoadedModel(
                    key=key,
                    path=cfg["path"],
                    model=model,
                    n_ctx=cfg["n_ctx"],
                    n_gpu_layers=cfg["n_gpu_layers"],
                    is_heavy=is_heavy,
                    loaded_at=time.time(),
                    estimated_vram_mb=estimated_vram,
                    clip_path=cfg.get("clip_path"),
                )
                if is_heavy:
                    self._heavy_slot = key
                logger.info(f"✓ Model '{key}' loaded successfully")
            else:
                logger.error(f"✗ Failed to load model '{key}' after all fallback attempts")

            return model

    def get_model(self, key: str) -> Optional[Any]:
        """Get a loaded model (does not load on demand)."""
        with self._lock:
            lm = self._models.get(key)
            return lm.model if lm else None

    def is_loaded(self, key: str) -> bool:
        with self._lock:
            return key in self._models and self._models[key].model is not None

    def unload_heavy_slot(self):
        """Unload whatever is in the heavy slot."""
        with self._lock:
            if self._heavy_slot:
                self._unload_model_unsafe(self._heavy_slot)
                self._heavy_slot = None

    def unload_all_except_fast(self):
        """Unload all models except the fast model."""
        with self._lock:
            keys = [k for k in list(self._models.keys()) if k != FAST_KEY]
            for k in keys:
                self._unload_model_unsafe(k)
            self._heavy_slot = None

    def unload_all(self):
        """Unload every model."""
        with self._lock:
            for k in list(self._models.keys()):
                self._unload_model_unsafe(k)
            self._heavy_slot = None

    # ── Unified resolve ──────────────────────────────────────────────

    # Canonical fallback chains per target
    _FALLBACK_CHAINS: Dict[str, List[str]] = {
        "fast":      ["fast", "medium", "deep"],
        "medium":    ["medium", "fast", "deep"],
        "deep":      ["deep", "medium", "fast"],
        "reasoning": ["reasoning", "deep", "medium", "fast"],
        "vision":    ["vision"],
        "vision_code": ["vision_code", "vision"],
    }

    def resolve_model(self, target: str) -> Tuple[Optional[Any], str]:
        """
        The **single entry-point** for acquiring a model.

        Tries to ensure the target model, then walks a fallback chain.
        Returns ``(llama_instance, actual_key)`` or ``(None, "")``.

        This replaces all ``llm = llm_deep or llm_medium or llm_fast`` patterns.
        """
        chain = self._FALLBACK_CHAINS.get(target, [target, "fast"])
        for key in chain:
            model = self.ensure_model(key)
            if model is not None:
                if key != target:
                    logger.warning(f"Resolved '{target}' → '{key}' (fallback)")
                return model, key
        return None, ""

    def loaded_models(self) -> Dict[str, dict]:
        """Return info about currently loaded models."""
        with self._lock:
            result = {}
            for k, lm in self._models.items():
                if lm.model is not None:
                    result[k] = {
                        "path": lm.path,
                        "n_ctx": lm.n_ctx,
                        "n_gpu_layers": lm.n_gpu_layers,
                        "is_heavy": lm.is_heavy,
                        "estimated_vram_mb": round(lm.estimated_vram_mb, 1),
                        "loaded_at": lm.loaded_at,
                    }
            return result

    def heavy_slot_occupant(self) -> Optional[str]:
        with self._lock:
            return self._heavy_slot

    # ── Fallback Ladder ──────────────────────────────────────────────

    def _load_with_fallback(self, key: str, cfg: dict) -> Optional[Any]:
        """Try loading with progressively lighter settings."""
        try:
            from llama_cpp import Llama
        except ImportError:
            logger.error("llama-cpp-python not installed — cannot load models")
            return None

        original_gpu = cfg["n_gpu_layers"]
        original_ctx = cfg["n_ctx"]

        for gpu_layers in FALLBACK_GPU_LAYERS:
            for ctx_size in FALLBACK_CTX_SIZES:
                actual_gpu = gpu_layers if gpu_layers is not None else original_gpu
                actual_ctx = ctx_size if ctx_size is not None else original_ctx

                # Skip if this is the same as a previously-failed combo
                # (the first iteration uses original settings)
                try:
                    kwargs = {
                        "model_path": cfg["path"],
                        "n_ctx": actual_ctx,
                        "n_gpu_layers": actual_gpu,
                        "verbose": False,
                    }
                    if cfg.get("tensor_split"):
                        kwargs["tensor_split"] = cfg["tensor_split"]
                    if cfg.get("clip_path") and os.path.exists(cfg["clip_path"]):
                        kwargs["clip_model_path"] = cfg["clip_path"]
                    if cfg.get("use_flash_attn"):
                        kwargs["use_flash_attn"] = True

                    label = f"key={key} gpu_layers={actual_gpu} n_ctx={actual_ctx}"
                    logger.info(f"Attempting model load: {label}")

                    model = Llama(**kwargs)

                    if actual_gpu != original_gpu or actual_ctx != original_ctx:
                        logger.warning(
                            f"Model '{key}' loaded with reduced settings: "
                            f"gpu_layers={actual_gpu} (was {original_gpu}), "
                            f"n_ctx={actual_ctx} (was {original_ctx})"
                        )
                    return model

                except Exception as e:
                    err_str = str(e).lower()
                    is_oom = "out of memory" in err_str or "oom" in err_str or "alloc" in err_str
                    logger.warning(f"Load attempt failed ({label}): {e}")

                    _flush_gpu()

                    if not is_oom:
                        # Non-OOM error — don't bother retrying lighter configs
                        return None

        # All attempts exhausted
        return None

    # ── Internal ─────────────────────────────────────────────────────

    def _unload_model_unsafe(self, key: str):
        """Unload a model without acquiring the lock (caller holds it)."""
        lm = self._models.pop(key, None)
        if lm and lm.model is not None:
            logger.info(f"Unloading model '{key}'")
            try:
                del lm.model
            except Exception:
                pass
            lm.model = None
        _flush_gpu()
        # Brief pause to let CUDA release memory
        time.sleep(0.3)


# ── Memory Gate ──────────────────────────────────────────────────────────

class MemoryGate:
    """
    Pre-flight memory check before heavy tasks (image gen, video, music, 3D, vision).
    Optionally unloads LLMs to free VRAM, or fails fast with a structured error.
    """

    def __init__(self, model_manager: Optional[ModelManager] = None):
        self._mm = model_manager

    # ── Public API ───────────────────────────────────────────────────

    def pre_heavy_task(
        self,
        required_vram_mb: float = 4000,
        required_ram_mb: float = 0,
        reason: str = "heavy GPU task",
        unload_llms: bool = True,
        allow_cpu_fallback: bool = False,
    ) -> Dict[str, Any]:
        """
        Prepare system for a heavy GPU task.

        Returns a dict:
            ok: True  → proceed
            ok: False → caller must return the ``error`` dict to the UI

        The ``error`` dict is structured so the frontend can display it nicely
        and offer an "Unload & retry" button.
        """
        snap = get_memory_snapshot()
        freed = 0.0

        # Check RAM first
        if required_ram_mb > 0 and snap.ram_available_mb < required_ram_mb:
            return self._fail(
                snap, freed, reason,
                required_vram_mb, required_ram_mb,
                f"Not enough RAM for {reason}: need {required_ram_mb:.0f} MB, "
                f"have {snap.ram_available_mb:.0f} MB free.",
            )

        # VRAM check
        if snap.total_vram_free_mb() >= required_vram_mb:
            logger.info(
                f"MemoryGate OK for '{reason}': "
                f"{snap.total_vram_free_mb():.0f} MB free ≥ {required_vram_mb:.0f} MB required"
            )
            return {"ok": True, "freed_mb": 0, "snapshot": snap}

        # Not enough VRAM — try unloading
        if unload_llms and self._mm:
            # Unload heavy-slot model first
            heavy = self._mm.heavy_slot_occupant()
            if heavy:
                estimated = self._mm._models.get(heavy, LoadedModel(key="", path="")).estimated_vram_mb
                self._mm.unload_heavy_slot()
                freed += estimated

            snap2 = get_memory_snapshot()
            if snap2.total_vram_free_mb() >= required_vram_mb:
                logger.info(
                    f"MemoryGate OK for '{reason}' after unloading heavy slot "
                    f"(freed ~{freed:.0f} MB)"
                )
                return {"ok": True, "freed_mb": freed, "snapshot": snap2}

            # Still not enough — unload everything
            self._mm.unload_all()
            freed += sum(
                lm.estimated_vram_mb
                for lm in self._mm._models.values()
                if lm.model is not None
            )
            _flush_gpu()

        final_snap = get_memory_snapshot()
        if final_snap.total_vram_free_mb() >= required_vram_mb:
            logger.info(
                f"MemoryGate OK for '{reason}' after full unload (freed ~{freed:.0f} MB)"
            )
            return {"ok": True, "freed_mb": freed, "snapshot": final_snap}

        # CPU fallback explicitly disallowed for heavy tasks
        if not allow_cpu_fallback:
            return self._fail(
                final_snap, freed, reason,
                required_vram_mb, required_ram_mb,
                f"Not enough VRAM for {reason} even after unloading all models. "
                f"Need {required_vram_mb:.0f} MB, have {final_snap.total_vram_free_mb():.0f} MB free. "
                f"CPU fallback is not allowed for this task.",
            )

        # If we reach here, allow_cpu_fallback is True
        logger.warning(f"MemoryGate: proceeding on CPU for '{reason}' (insufficient VRAM)")
        return {"ok": True, "freed_mb": freed, "snapshot": final_snap, "cpu_fallback": True}

    def post_heavy_task(self):
        """Called after heavy task completes — reload fast model if needed."""
        if self._mm and not self._mm.is_loaded(FAST_KEY):
            self._mm.ensure_model(FAST_KEY)
        _flush_gpu()

    # ── Private helpers ──────────────────────────────────────────────

    @staticmethod
    def _fail(
        snap: "MemorySnapshot",
        freed_mb: float,
        reason: str,
        required_vram_mb: float,
        required_ram_mb: float,
        message: str,
    ) -> Dict[str, Any]:
        """Build a structured error result for the UI."""
        logger.error(f"MemoryGate FAIL: {message}")
        return {
            "ok": False,
            "freed_mb": freed_mb,
            "snapshot": snap,
            "error": {
                "message": message,
                "reason": reason,
                "required_vram_mb": required_vram_mb,
                "required_ram_mb": required_ram_mb,
                "current_free_vram_mb": round(snap.total_vram_free_mb(), 1),
                "current_free_ram_mb": round(snap.ram_available_mb, 1),
                "action": "unload_and_retry",
                "action_label": "Unload models & retry",
            },
        }


# ── Singleton ────────────────────────────────────────────────────────────

_instance: Optional[ModelManager] = None
_instance_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ModelManager()
    return _instance


_gate_instance: Optional[MemoryGate] = None
_gate_lock = threading.Lock()


def get_memory_gate() -> MemoryGate:
    global _gate_instance
    if _gate_instance is None:
        with _gate_lock:
            if _gate_instance is None:
                _gate_instance = MemoryGate(get_model_manager())
    return _gate_instance
