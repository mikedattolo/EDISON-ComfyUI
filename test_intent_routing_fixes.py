"""Regression tests for intent routing fixes.

Two specific issues addressed:

1. Coding requests were silently routed to the Rhino/CAD pipeline because
   the alias table (box, cube, cylinder, sphere, ball, cup, mug, ...)
   matched verbs like "build" / "create" / "generate" without requiring
   any 3D / fabrication signal. So "build a Box class",
   "create a cylinder utility", "generate a sphere component", etc.
   ended up trying to model a 3D shape.

2. Genuine 3D requests (with words like "stl", "3d print", "rhino",
   "cad", "printable") must still resolve correctly.
"""
from __future__ import annotations

from services.edison_core.business_actions import (
    _looks_like_code_request,
    _parse_model_request,
)


# ---------------------------------------------------------------------------
# _looks_like_code_request
# ---------------------------------------------------------------------------


def test_code_request_detection_positive():
    coding_phrases = [
        "write a function to sort a list",
        "build a Box class in python",
        "create a cylinder utility function",
        "generate a sphere component for react",
        "implement a queue class",
        "fix this bug",
        "debug this code",
        "review this code",
        "optimize this loop",
        "rewrite this in typescript",
        "convert this script to python",
        "make a fastapi endpoint",
        "add a new module",
        "create a router for /users",
    ]
    for phrase in coding_phrases:
        assert _looks_like_code_request(phrase.lower()), (
            f"expected coding phrase to be detected: {phrase!r}"
        )


def test_code_request_detection_negative():
    non_coding_phrases = [
        "design a vase for printing",
        "model a planter for the garden",
        "create a 3d print of a coaster",
        "build a stl of a keychain",
        "sculpt a small sphere paperweight",
    ]
    for phrase in non_coding_phrases:
        assert not _looks_like_code_request(phrase.lower()), (
            f"phrase should NOT be flagged as code: {phrase!r}"
        )


# ---------------------------------------------------------------------------
# _parse_model_request — coding requests must NOT match
# ---------------------------------------------------------------------------


def test_coding_requests_do_not_route_to_cad():
    coding_requests = [
        "build a Box class",
        "create a cylinder utility function",
        "generate a sphere component for the dashboard",
        "make a Cube object pool in python",
        "build me a mug class with __init__",
        "write a function that returns a sphere of radius r",
        "implement a Pot class for the inventory module",
        "create a glass component in react",
        "build the auth module",
        "create a new service",
        "generate a fastapi endpoint",
    ]
    for text in coding_requests:
        result = _parse_model_request(text.lower(), text, has_node_manager=True)
        assert result is None, (
            f"coding request was misrouted to CAD: {text!r} -> {result!r}"
        )


# ---------------------------------------------------------------------------
# _parse_model_request — genuine 3D requests still match
# ---------------------------------------------------------------------------


def test_alias_request_with_strong_3d_hint_matches():
    # "make a 3d model of a cup" must still route to the cup preset.
    text = "make a 3d model of a cup"
    result = _parse_model_request(text.lower(), text, has_node_manager=True)
    assert result is not None
    assert result["shape"] == "cup"
    assert result["matched_alias"] == "cup"
    assert result["needs_llm"] is False


def test_stl_request_matches_alias():
    text = "design an stl file of a cube"
    result = _parse_model_request(text.lower(), text, has_node_manager=True)
    assert result is not None
    assert result["shape"] == "box"  # cube is aliased to box preset


def test_3d_print_request_matches_alias():
    text = "model a planter for a 3d print"
    result = _parse_model_request(text.lower(), text, has_node_manager=True)
    assert result is not None
    # Either "planter" (free-form) or one of the aliased presets is fine.
    assert result["shape"]


def test_freeform_request_with_strong_3d_hint():
    text = "create a 3d model of a small dragon"
    result = _parse_model_request(text.lower(), text, has_node_manager=True)
    assert result is not None
    assert result["needs_llm"] is True


def test_alias_without_3d_hint_rejected():
    # No 3d/cad/stl/print signal -> must NOT match the alias table.
    text = "create a cup"
    result = _parse_model_request(text.lower(), text, has_node_manager=False)
    assert result is None
