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
        "browser.create_session",
        "browser.observe",
        "browser.navigate",
        "browser.click",
        "browser.type",
        "browser.press",
        "browser.scroll",
    }
    missing = expected_tools - set(TOOL_REGISTRY.keys())
    assert not missing, f"Missing browser tools: {missing}"


if __name__ == "__main__":
    test_sandbox_browser_session_routes_present()
    test_browser_tools_registered()
    print("✓ test_sandbox_browser_sessions passed")
