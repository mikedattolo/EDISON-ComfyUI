#!/usr/bin/env python3
"""Route and tool registration tests for persistent sandbox browser sessions."""

import sys

sys.path.insert(0, ".")


def _route_methods_map(app):
    route_map = {}
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = set(getattr(route, "methods", set()))
        if path:
            route_map.setdefault(path, set()).update(methods)
    return route_map


def test_sandbox_browser_session_routes_present():
    from services.edison_core.app import app

    route_map = _route_methods_map(app)
    expected_post_routes = {
        "/sandbox/browser/session/create",
        "/sandbox/browser/session/navigate",
        "/sandbox/browser/session/click",
        "/sandbox/browser/session/type",
        "/sandbox/browser/session/key",
        "/sandbox/browser/session/scroll",
        "/sandbox/browser/session/move",
        "/sandbox/browser/session/screenshot",
        "/sandbox/browser/session/close",
    }

    for path in expected_post_routes:
        assert path in route_map, f"Missing route: {path}"
        assert "POST" in route_map[path], f"Route missing POST: {path}"


def test_browser_tools_registered():
    from services.edison_core.app import TOOL_REGISTRY

    expected_tools = {
        "browser_create_session",
        "browser_screenshot",
        "browser_navigate",
        "browser_click",
        "browser_type",
        "browser_press",
        "browser_scroll",
        "browser_close",
    }
    missing = expected_tools - set(TOOL_REGISTRY.keys())
    assert not missing, f"Missing browser tools: {missing}"


def test_browser_url_auto_routing():
    """Verify that explicit URL-opening messages are detected for auto-routing."""
    import re

    pattern_full = (
        r'(?:open|visit|browse(?:\s+to)?|go\s+to|navigate\s+to|show\s+me|load|pull\s+up)\s+'
        r'(https?://\S+|(?:www\.)\S+)'
    )
    pattern_bare = (
        r'(?:open|visit|browse(?:\s+to)?|go\s+to|navigate\s+to|show\s+me|load|pull\s+up)\s+'
        r'([a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:/\S*)?)'
    )

    # Should match (explicit URL opening intent)
    should_match = [
        ("open https://github.com", "https://github.com"),
        ("visit https://example.org/page", "https://example.org/page"),
        ("browse to http://localhost:8080", "http://localhost:8080"),
        ("go to https://news.ycombinator.com", "https://news.ycombinator.com"),
        ("navigate to https://docs.python.org", "https://docs.python.org"),
        ("open github.com", "github.com"),
        ("visit reddit.com/r/python", "reddit.com/r/python"),
    ]
    for msg, expected_url in should_match:
        m = re.search(pattern_full, msg, re.IGNORECASE)
        if not m:
            m = re.search(pattern_bare, msg, re.IGNORECASE)
        assert m is not None, f"Should match: {msg!r}"
        assert m.group(1) == expected_url, f"URL mismatch for {msg!r}: got {m.group(1)!r}"

    # Should NOT match (general search/chat intent)
    should_not_match = [
        "search for python tutorials",
        "what is the latest news about AI",
        "tell me about github.com",
        "how to use github",
    ]
    for msg in should_not_match:
        m = re.search(pattern_full, msg, re.IGNORECASE)
        if not m:
            m = re.search(pattern_bare, msg, re.IGNORECASE)
        assert m is None, f"Should NOT match: {msg!r}"


if __name__ == "__main__":
    test_sandbox_browser_session_routes_present()
    test_browser_tools_registered()
    test_browser_url_auto_routing()
    print("✓ test_sandbox_browser_sessions passed")
