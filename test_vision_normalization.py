#!/usr/bin/env python3
"""Tests for vision image payload normalization."""

import sys

sys.path.insert(0, '.')


def test_normalize_image_data_uri():
    from services.edison_core.app import _normalize_image_data_uri

    # Already-prefixed data URI should be preserved (and trimmed)
    prefixed = "data:image/jpeg;base64, abc123  "
    normalized = _normalize_image_data_uri(prefixed)
    assert normalized == "data:image/jpeg;base64,abc123"

    # Raw base64 should get png data URI prefix
    raw = "ZmFrZV9pbWFnZV9ieXRlcw=="
    normalized = _normalize_image_data_uri(raw)
    assert normalized == f"data:image/png;base64,{raw}"

    # Empty/invalid values should return None
    assert _normalize_image_data_uri("") is None
    assert _normalize_image_data_uri("   ") is None
    assert _normalize_image_data_uri(None) is None


if __name__ == "__main__":
    test_normalize_image_data_uri()
    print("✓ test_vision_normalization passed")