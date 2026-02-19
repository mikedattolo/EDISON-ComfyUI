"""
GPU configuration validator and tensor_split normalizer.

On startup:
  1. Detect available GPUs + VRAM (torch → pynvml → nvidia-smi fallback).
  2. Compare against config/gpu_map.yaml and config/edison.yaml.
  3. Auto-normalize tensor_split if its length != GPU count.
  4. Log clear warnings for any mismatches.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


# ── GPU detection ────────────────────────────────────────────────────────

def detect_gpus() -> List[Dict[str, Any]]:
    """
    Detect available GPUs and return a list of dicts:
    [{"index": 0, "name": "...", "total_mb": ..., "free_mb": ..., "used_mb": ...}, ...]
    """
    gpus: List[Dict[str, Any]] = []

    # Try torch first (most common in this repo)
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free, total = torch.cuda.mem_get_info(i)
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "total_mb": round(total / (1024 ** 2), 1),
                    "free_mb": round(free / (1024 ** 2), 1),
                    "used_mb": round((total - free) / (1024 ** 2), 1),
                })
            if gpus:
                return gpus
    except Exception:
        pass

    # Try pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            gpus.append({
                "index": i,
                "name": name,
                "total_mb": round(info.total / (1024 ** 2), 1),
                "free_mb": round(info.free / (1024 ** 2), 1),
                "used_mb": round(info.used / (1024 ** 2), 1),
            })
        pynvml.nvmlShutdown()
        if gpus:
            return gpus
    except Exception:
        pass

    # Fallback: nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "total_mb": float(parts[2]),
                        "used_mb": float(parts[3]),
                        "free_mb": float(parts[4]),
                    })
    except Exception:
        pass

    return gpus


# ── tensor_split normalization ───────────────────────────────────────────

def normalize_tensor_split(
    tensor_split: Optional[List[float]],
    gpu_count: int,
) -> List[float]:
    """
    Normalize tensor_split so its length matches gpu_count.

    Rules:
      - If tensor_split is None or empty → equal split across all GPUs
      - If len(tensor_split) == gpu_count → return as-is (normalized to sum=1)
      - If len(tensor_split) != gpu_count → redistribute proportionally and warn
    """
    if not tensor_split or gpu_count < 1:
        # Equal split
        if gpu_count < 1:
            return [1.0]
        equal = round(1.0 / gpu_count, 4)
        result = [equal] * gpu_count
        logger.info(f"tensor_split not configured; using equal split: {result}")
        return result

    # Normalize to sum=1
    total = sum(tensor_split)
    if total <= 0:
        equal = round(1.0 / gpu_count, 4)
        return [equal] * gpu_count

    normalized = [round(v / total, 4) for v in tensor_split]

    if len(normalized) == gpu_count:
        return normalized

    # Length mismatch — redistribute
    logger.warning(
        f"tensor_split length ({len(normalized)}) != GPU count ({gpu_count}). "
        f"Auto-normalizing."
    )

    if len(normalized) < gpu_count:
        # Fewer splits than GPUs: distribute remaining equally among new GPUs
        existing_sum = sum(normalized)
        remaining = max(1.0 - existing_sum, 0)
        extra_count = gpu_count - len(normalized)
        extra_share = round(remaining / extra_count, 4) if extra_count > 0 else 0
        result = normalized + [extra_share] * extra_count
    else:
        # More splits than GPUs: merge extras into the last GPU
        result = normalized[:gpu_count]
        overflow = sum(normalized[gpu_count:])
        result[-1] = round(result[-1] + overflow, 4)

    # Re-normalize to sum=1
    s = sum(result)
    if s > 0:
        result = [round(v / s, 4) for v in result]

    logger.warning(f"Auto-normalized tensor_split: {result}")
    return result


# ── Config validation ────────────────────────────────────────────────────

def validate_config(
    repo_root: Path,
    edison_config: dict,
    detected_gpus: List[Dict[str, Any]],
) -> List[str]:
    """
    Validate edison.yaml and gpu_map.yaml against actual hardware.

    Returns a list of warning/error strings (empty = all good).
    """
    warnings: List[str] = []
    gpu_count = len(detected_gpus)
    core = edison_config.get("edison", {}).get("core", {})

    # 1. Check tensor_split length
    ts = core.get("tensor_split")
    if ts and len(ts) != gpu_count:
        warnings.append(
            f"edison.yaml tensor_split has {len(ts)} entries but {gpu_count} GPU(s) detected. "
            f"Will auto-normalize."
        )

    # 2. Check gpu_map.yaml
    gpu_map_path = repo_root / "config" / "gpu_map.yaml"
    if gpu_map_path.exists():
        try:
            with open(gpu_map_path) as f:
                gpu_map = yaml.safe_load(f) or {}
            map_gpus = gpu_map.get("gpus", [])
            if len(map_gpus) != gpu_count:
                warnings.append(
                    f"gpu_map.yaml lists {len(map_gpus)} GPU(s) but {gpu_count} detected. "
                    f"Update gpu_map.yaml to match your hardware."
                )
        except Exception as e:
            warnings.append(f"Failed to parse gpu_map.yaml: {e}")

    # 3. Check model file existence
    models_path_rel = core.get("models_path", "models/llm")
    models_path = repo_root / models_path_rel
    for key in ["fast_model", "medium_model", "deep_model", "vision_model"]:
        name = core.get(key)
        if name:
            full = models_path / name
            if not full.exists():
                warnings.append(f"Model file for '{key}' not found: {full}")

    # 4. Check VRAM sufficiency (rough)
    total_vram = sum(g.get("total_mb", 0) for g in detected_gpus)
    if total_vram > 0 and total_vram < 8000:
        warnings.append(
            f"Total VRAM is only {total_vram:.0f} MB. "
            f"EDISON needs ≥8 GB for the fast model. Consider CPU offloading."
        )

    # 5. Check n_gpu_layers sanity
    deep_layers = core.get("deep_n_gpu_layers", core.get("n_gpu_layers", -1))
    if isinstance(deep_layers, int) and deep_layers > 80:
        warnings.append(
            f"deep_n_gpu_layers={deep_layers} is very high. "
            f"72B models typically have ~80 layers. Using -1 offloads all."
        )

    return warnings


# ── Startup runner ───────────────────────────────────────────────────────

def run_startup_validation(repo_root: Path, edison_config: dict) -> Tuple[List[Dict], List[float]]:
    """
    Run full GPU detection + config validation at startup.

    Returns:
        (detected_gpus, normalized_tensor_split)
    """
    # Detect GPUs
    gpus = detect_gpus()
    if gpus:
        logger.info(f"Detected {len(gpus)} GPU(s):")
        for g in gpus:
            logger.info(
                f"  GPU {g['index']}: {g['name']} — "
                f"{g['total_mb']:.0f} MB total, {g['free_mb']:.0f} MB free"
            )
    else:
        logger.warning("No GPUs detected! Models will run on CPU (very slow).")

    # Validate config
    warnings = validate_config(repo_root, edison_config, gpus)
    for w in warnings:
        logger.warning(f"CONFIG: {w}")

    # Normalize tensor_split
    core = edison_config.get("edison", {}).get("core", {})
    raw_ts = core.get("tensor_split")
    normalized_ts = normalize_tensor_split(raw_ts, len(gpus))

    return gpus, normalized_ts
