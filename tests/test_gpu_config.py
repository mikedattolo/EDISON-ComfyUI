"""
Tests for GPU config validation and tensor_split normalization.

Run: python tests/test_gpu_config.py
"""

import os
import sys
import tempfile
from pathlib import Path

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


# ── tensor_split normalization ───────────────────────────────────────────

from services.edison_core.gpu_config import (
    normalize_tensor_split,
    validate_config,
)


def test_equal_split_none_input():
    """None tensor_split → equal split."""
    result = normalize_tensor_split(None, 2)
    assert len(result) == 2, f"Expected 2 entries, got {len(result)}"
    assert abs(sum(result) - 1.0) < 0.01, f"Sum should be ~1.0, got {sum(result)}"
    assert abs(result[0] - 0.5) < 0.01, f"Each should be ~0.5, got {result}"

run_test("equal split (None input, 2 GPUs)", test_equal_split_none_input)


def test_equal_split_empty_input():
    """Empty list → equal split."""
    result = normalize_tensor_split([], 3)
    assert len(result) == 3, f"Expected 3 entries, got {len(result)}"
    for v in result:
        assert abs(v - 1.0/3) < 0.01, f"Each should be ~0.333, got {v}"

run_test("equal split (empty input, 3 GPUs)", test_equal_split_empty_input)


def test_passthrough_matching_length():
    """tensor_split with correct length → normalized to sum=1."""
    result = normalize_tensor_split([3, 1], 2)
    assert len(result) == 2, f"Expected 2, got {len(result)}"
    assert abs(result[0] - 0.75) < 0.01, f"GPU 0 should be ~0.75, got {result[0]}"
    assert abs(result[1] - 0.25) < 0.01, f"GPU 1 should be ~0.25, got {result[1]}"

run_test("normalize matching-length tensor_split", test_passthrough_matching_length)


def test_already_normalized():
    """Already-normalized values pass through."""
    result = normalize_tensor_split([0.6, 0.4], 2)
    assert abs(result[0] - 0.6) < 0.01, f"GPU 0: expected 0.6, got {result[0]}"
    assert abs(result[1] - 0.4) < 0.01, f"GPU 1: expected 0.4, got {result[1]}"

run_test("already normalized passthrough", test_already_normalized)


def test_expand_fewer_splits_than_gpus():
    """2 splits for 3 GPUs → expand and redistribute."""
    result = normalize_tensor_split([0.6, 0.4], 3)
    assert len(result) == 3, f"Expected 3, got {len(result)}"
    assert abs(sum(result) - 1.0) < 0.01, f"Sum should be ~1.0, got {sum(result)}"
    # The first two should retain their relative proportions
    assert result[0] > result[1], f"GPU 0 should be > GPU 1: {result}"

run_test("expand: fewer splits than GPUs", test_expand_fewer_splits_than_gpus)


def test_shrink_more_splits_than_gpus():
    """4 splits for 2 GPUs → merge overflow into last GPU."""
    result = normalize_tensor_split([0.3, 0.3, 0.2, 0.2], 2)
    assert len(result) == 2, f"Expected 2, got {len(result)}"
    assert abs(sum(result) - 1.0) < 0.01, f"Sum should be ~1.0, got {sum(result)}"

run_test("shrink: more splits than GPUs", test_shrink_more_splits_than_gpus)


def test_single_gpu():
    """1 GPU → always [1.0]."""
    result = normalize_tensor_split([0.7, 0.3], 1)
    assert len(result) == 1, f"Expected 1, got {len(result)}"
    assert abs(result[0] - 1.0) < 0.01, f"Expected 1.0, got {result[0]}"

run_test("single GPU collapse", test_single_gpu)


def test_zero_gpus():
    """0 GPUs → fallback [1.0]."""
    result = normalize_tensor_split(None, 0)
    assert len(result) == 1, f"Expected [1.0], got {result}"

run_test("zero GPUs fallback", test_zero_gpus)


def test_zero_values():
    """All-zero tensor_split → equal split."""
    result = normalize_tensor_split([0, 0], 2)
    assert len(result) == 2
    assert abs(result[0] - 0.5) < 0.01

run_test("all-zero tensor_split → equal split", test_zero_values)


# ── Config validation ────────────────────────────────────────────────────

def test_validate_config_no_warnings():
    """Clean config produces no warnings."""
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        config = {
            "edison": {
                "core": {
                    "tensor_split": [0.5, 0.5],
                    "models_path": "models/llm",
                }
            }
        }
        gpus = [
            {"index": 0, "name": "GPU-0", "total_mb": 24000, "free_mb": 20000, "used_mb": 4000},
            {"index": 1, "name": "GPU-1", "total_mb": 24000, "free_mb": 20000, "used_mb": 4000},
        ]
        warnings = validate_config(tmpdir, config, gpus)
        # gpu_map.yaml doesn't exist in tmpdir, so no gpu_map warning
        # No model files specified, so no model warnings
        # 24 GB per GPU is plenty
        ts_warns = [w for w in warnings if "tensor_split" in w]
        assert len(ts_warns) == 0, f"Expected no tensor_split warnings: {warnings}"

run_test("validate_config: clean 2-GPU config", test_validate_config_no_warnings)


def test_validate_config_tensor_split_mismatch():
    """tensor_split length != GPU count → warning."""
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        config = {
            "edison": {
                "core": {
                    "tensor_split": [0.5, 0.3, 0.2],  # 3 entries
                }
            }
        }
        gpus = [
            {"index": 0, "name": "GPU-0", "total_mb": 24000, "free_mb": 20000, "used_mb": 4000},
            {"index": 1, "name": "GPU-1", "total_mb": 24000, "free_mb": 20000, "used_mb": 4000},
        ]
        warnings = validate_config(tmpdir, config, gpus)
        ts_warns = [w for w in warnings if "tensor_split" in w]
        assert len(ts_warns) >= 1, f"Expected a tensor_split mismatch warning: {warnings}"

run_test("validate_config: tensor_split mismatch", test_validate_config_tensor_split_mismatch)


def test_validate_config_low_vram_warning():
    """Total VRAM < 8 GB → warning."""
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        config = {"edison": {"core": {}}}
        gpus = [
            {"index": 0, "name": "GPU-0", "total_mb": 4000, "free_mb": 3000, "used_mb": 1000},
        ]
        warnings = validate_config(tmpdir, config, gpus)
        vram_warns = [w for w in warnings if "VRAM" in w]
        assert len(vram_warns) >= 1, f"Expected VRAM sufficiency warning: {warnings}"

run_test("validate_config: low VRAM warning", test_validate_config_low_vram_warning)


def test_validate_config_high_gpu_layers_warning():
    """Extremely high n_gpu_layers → warning."""
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        config = {
            "edison": {
                "core": {
                    "deep_n_gpu_layers": 200,
                }
            }
        }
        gpus = [
            {"index": 0, "name": "GPU-0", "total_mb": 48000, "free_mb": 40000, "used_mb": 8000},
        ]
        warnings = validate_config(tmpdir, config, gpus)
        layer_warns = [w for w in warnings if "n_gpu_layers" in w.lower() or "layers" in w.lower()]
        assert len(layer_warns) >= 1, f"Expected high gpu_layers warning: {warnings}"

run_test("validate_config: high n_gpu_layers warning", test_validate_config_high_gpu_layers_warning)


# ── Summary ──────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"GPU Config Tests: {passed} passed, {failed} failed")
print(f"{'='*60}")
sys.exit(1 if failed > 0 else 0)
