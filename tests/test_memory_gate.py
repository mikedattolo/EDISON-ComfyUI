"""
Tests for MemoryGate and ModelManager v2 core logic.

Run: python tests/test_memory_gate.py
"""

import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed = 0
failed = 0


def run_test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"✓ {name}: PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ {name}: FAILED — {e}")
        failed += 1


from services.edison_core.model_manager_v2 import (
    MemoryGate,
    MemorySnapshot,
    ModelManager,
    LoadedModel,
    get_memory_snapshot,
    HEAVY_KEYS,
    FAST_KEY,
)


# ── Helper to create fake snapshots ─────────────────────────────────────

def _make_snap(free_vram_mb=20000, ram_avail_mb=32000, gpu_count=1):
    """Build a MemorySnapshot with controllable VRAM/RAM."""
    gpus = []
    per_gpu = free_vram_mb / max(gpu_count, 1)
    for i in range(gpu_count):
        gpus.append({
            "index": i,
            "total_mb": per_gpu * 2,
            "used_mb": per_gpu,
            "free_mb": per_gpu,
        })
    snap = MemorySnapshot(
        ram_total_mb=ram_avail_mb * 2,
        ram_available_mb=ram_avail_mb,
        ram_used_mb=ram_avail_mb,
        gpus=gpus,
        timestamp=time.time(),
    )
    return snap


# ── MemoryGate tests ────────────────────────────────────────────────────

def test_memgate_ok_plenty_of_vram():
    """Pre-heavy task passes when plenty of VRAM available."""
    gate = MemoryGate(model_manager=None)
    with patch("services.edison_core.model_manager_v2.get_memory_snapshot",
               return_value=_make_snap(free_vram_mb=20000)):
        result = gate.pre_heavy_task(required_vram_mb=4000, reason="test")
    assert result["ok"] is True, f"Expected ok=True, got {result}"

run_test("MemoryGate: OK with plenty of VRAM", test_memgate_ok_plenty_of_vram)


def test_memgate_fail_not_enough_vram_no_mm():
    """Fail when VRAM too low and no ModelManager to unload from."""
    gate = MemoryGate(model_manager=None)
    with patch("services.edison_core.model_manager_v2.get_memory_snapshot",
               return_value=_make_snap(free_vram_mb=1000)):
        result = gate.pre_heavy_task(required_vram_mb=4000, reason="test")
    assert result["ok"] is False, f"Expected ok=False, got {result}"
    assert "error" in result, "Expected 'error' key in result"
    err = result["error"]
    assert err["action"] == "unload_and_retry", f"Expected action='unload_and_retry', got {err}"
    assert err["required_vram_mb"] == 4000

run_test("MemoryGate: FAIL with low VRAM, no ModelManager", test_memgate_fail_not_enough_vram_no_mm)


def test_memgate_fail_not_enough_ram():
    """Fail when RAM is insufficient, even before checking VRAM."""
    gate = MemoryGate(model_manager=None)
    with patch("services.edison_core.model_manager_v2.get_memory_snapshot",
               return_value=_make_snap(free_vram_mb=20000, ram_avail_mb=500)):
        result = gate.pre_heavy_task(required_vram_mb=4000, required_ram_mb=8000, reason="test")
    assert result["ok"] is False, f"Expected ok=False for low RAM, got {result}"
    assert "RAM" in result["error"]["message"], f"Error should mention RAM: {result['error']['message']}"

run_test("MemoryGate: FAIL with low RAM", test_memgate_fail_not_enough_ram)


def test_memgate_structured_error():
    """Verify the error structure matches the expected UI schema."""
    gate = MemoryGate(model_manager=None)
    with patch("services.edison_core.model_manager_v2.get_memory_snapshot",
               return_value=_make_snap(free_vram_mb=500)):
        result = gate.pre_heavy_task(required_vram_mb=8000, reason="image generation")
    assert result["ok"] is False
    err = result["error"]
    required_keys = {"message", "reason", "required_vram_mb", "required_ram_mb",
                     "current_free_vram_mb", "current_free_ram_mb", "action", "action_label"}
    missing = required_keys - set(err.keys())
    assert not missing, f"Error dict missing keys: {missing}"
    assert err["reason"] == "image generation"
    assert err["action_label"] == "Unload models & retry"

run_test("MemoryGate: structured error schema", test_memgate_structured_error)


def test_memgate_cpu_fallback_allowed():
    """When allow_cpu_fallback=True, proceed even with low VRAM."""
    gate = MemoryGate(model_manager=None)
    with patch("services.edison_core.model_manager_v2.get_memory_snapshot",
               return_value=_make_snap(free_vram_mb=500)):
        result = gate.pre_heavy_task(
            required_vram_mb=8000, reason="test", allow_cpu_fallback=True
        )
    assert result["ok"] is True, f"Expected ok=True with CPU fallback, got {result}"
    assert result.get("cpu_fallback") is True, "Expected cpu_fallback flag"

run_test("MemoryGate: CPU fallback allowed succeeds", test_memgate_cpu_fallback_allowed)


def test_memgate_unloads_heavy_slot():
    """MemoryGate unloads heavy-slot model to free VRAM, then succeeds."""
    mm = ModelManager()
    # Simulate a heavy model in the slot
    mm._models["deep"] = LoadedModel(
        key="deep", path="/fake/deep.gguf",
        model=MagicMock(), estimated_vram_mb=10000,
        is_heavy=True,
    )
    mm._heavy_slot = "deep"

    call_count = [0]
    def _snap_side_effect():
        call_count[0] += 1
        if call_count[0] <= 1:
            return _make_snap(free_vram_mb=2000)  # Before unload: tight
        else:
            return _make_snap(free_vram_mb=15000)  # After unload: plenty

    gate = MemoryGate(model_manager=mm)
    with patch("services.edison_core.model_manager_v2.get_memory_snapshot",
               side_effect=_snap_side_effect):
        result = gate.pre_heavy_task(required_vram_mb=6000, reason="video gen")

    assert result["ok"] is True, f"Expected OK after unload, got {result}"
    assert mm._heavy_slot is None, "Heavy slot should be cleared"

run_test("MemoryGate: unloads heavy slot to free VRAM", test_memgate_unloads_heavy_slot)


# ── ModelManager tests ───────────────────────────────────────────────────

def test_mm_register_and_list():
    """Register models and verify configs stored."""
    mm = ModelManager()
    mm.register_model("fast", "/fake/fast.gguf", n_ctx=8192, n_gpu_layers=99)
    mm.register_model("deep", "/fake/deep.gguf", n_ctx=4096, n_gpu_layers=-1)
    assert "fast" in mm._model_configs
    assert "deep" in mm._model_configs
    assert mm._model_configs["fast"]["n_ctx"] == 8192
    assert mm._model_configs["deep"]["n_gpu_layers"] == -1

run_test("ModelManager: register_model stores configs", test_mm_register_and_list)


def test_mm_resolve_model_fallback_chain():
    """resolve_model walks fallback chain when primary unavailable."""
    mm = ModelManager()

    # Register only fast, not deep
    mm.register_model("fast", "/fake/fast.gguf")
    fake_llama = MagicMock()

    with patch.object(mm, "ensure_model", side_effect=lambda key: fake_llama if key == "fast" else None):
        model, actual_key = mm.resolve_model("deep")

    assert model is fake_llama, "Should resolve to fast model"
    assert actual_key == "fast", f"Expected fallback to 'fast', got '{actual_key}'"

run_test("ModelManager: resolve_model fallback chain", test_mm_resolve_model_fallback_chain)


def test_mm_resolve_model_primary():
    """resolve_model returns primary when available."""
    mm = ModelManager()
    mm.register_model("deep", "/fake/deep.gguf")
    fake_llama = MagicMock()

    with patch.object(mm, "ensure_model", side_effect=lambda key: fake_llama if key == "deep" else None):
        model, actual_key = mm.resolve_model("deep")

    assert model is fake_llama
    assert actual_key == "deep"

run_test("ModelManager: resolve_model returns primary", test_mm_resolve_model_primary)


def test_mm_resolve_model_all_fail():
    """resolve_model returns (None, '') when all models fail."""
    mm = ModelManager()
    with patch.object(mm, "ensure_model", return_value=None):
        model, actual_key = mm.resolve_model("deep")

    assert model is None
    assert actual_key == ""

run_test("ModelManager: resolve_model all fail → None", test_mm_resolve_model_all_fail)


def test_mm_unload_heavy_slot():
    """unload_heavy_slot clears the heavy slot."""
    mm = ModelManager()
    mm._models["medium"] = LoadedModel(
        key="medium", path="/fake/medium.gguf",
        model=MagicMock(), is_heavy=True,
    )
    mm._heavy_slot = "medium"

    mm.unload_heavy_slot()
    assert mm._heavy_slot is None
    assert "medium" not in mm._models

run_test("ModelManager: unload_heavy_slot", test_mm_unload_heavy_slot)


def test_mm_unload_all_except_fast():
    """unload_all_except_fast preserves fast model."""
    mm = ModelManager()
    mm._models["fast"] = LoadedModel(key="fast", path="/f.gguf", model=MagicMock())
    mm._models["deep"] = LoadedModel(key="deep", path="/d.gguf", model=MagicMock(), is_heavy=True)
    mm._heavy_slot = "deep"

    mm.unload_all_except_fast()
    assert "fast" in mm._models, "Fast model should be preserved"
    assert "deep" not in mm._models, "Deep model should be unloaded"
    assert mm._heavy_slot is None

run_test("ModelManager: unload_all_except_fast", test_mm_unload_all_except_fast)


def test_mm_fallback_chains_exist():
    """All expected fallback chains are defined."""
    mm = ModelManager()
    for key in ["fast", "medium", "deep", "reasoning", "vision", "vision_code"]:
        assert key in mm._FALLBACK_CHAINS, f"Missing fallback chain for '{key}'"
        chain = mm._FALLBACK_CHAINS[key]
        assert chain[0] == key, f"Fallback chain for '{key}' should start with itself"
        assert len(chain) >= 1, f"Chain for '{key}' is empty"

run_test("ModelManager: all fallback chains defined", test_mm_fallback_chains_exist)


def test_mm_heavy_keys_classification():
    """Verify heavy vs fast key classification."""
    assert "fast" not in HEAVY_KEYS, "fast should not be a heavy key"
    for k in ["medium", "deep", "reasoning", "vision", "vision_code"]:
        assert k in HEAVY_KEYS, f"'{k}' should be a heavy key"
    assert FAST_KEY == "fast"

run_test("ModelManager: heavy/fast key classification", test_mm_heavy_keys_classification)


# ── Memory snapshot ──────────────────────────────────────────────────────

def test_memory_snapshot_vram_calc():
    """MemorySnapshot correctly sums multi-GPU VRAM."""
    snap = MemorySnapshot(
        gpus=[
            {"total_mb": 24000, "used_mb": 8000, "free_mb": 16000},
            {"total_mb": 24000, "used_mb": 12000, "free_mb": 12000},
        ]
    )
    assert snap.total_vram_free_mb() == 28000, f"Expected 28000, got {snap.total_vram_free_mb()}"
    assert snap.total_vram_total_mb() == 48000

run_test("MemorySnapshot: multi-GPU VRAM sum", test_memory_snapshot_vram_calc)


def test_memory_snapshot_to_dict():
    """MemorySnapshot.to_dict() includes all expected keys."""
    snap = MemorySnapshot(
        ram_total_mb=64000, ram_available_mb=32000, ram_used_mb=32000,
        gpus=[{"total_mb": 24000, "used_mb": 8000, "free_mb": 16000}],
        timestamp=1234567890.0,
    )
    d = snap.to_dict()
    for key in ["ram_total_mb", "ram_available_mb", "ram_used_mb", "gpus",
                "total_vram_free_mb", "timestamp"]:
        assert key in d, f"Missing key '{key}' in to_dict()"

run_test("MemorySnapshot: to_dict keys", test_memory_snapshot_to_dict)


# ── Summary ──────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"MemoryGate + ModelManager Tests: {passed} passed, {failed} failed")
print(f"{'='*60}")
sys.exit(1 if failed > 0 else 0)
