#!/usr/bin/env python3
"""Route existence checks for integrations/printing API additions."""

import sys

sys.path.insert(0, '.')


def _route_methods_map(app):
    route_map = {}
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = set(getattr(route, "methods", set()))
        if path:
            route_map.setdefault(path, set()).update(methods)
    return route_map


def test_integration_and_printing_routes_present():
    from services.edison_core.app import app

    route_map = _route_methods_map(app)

    expected = {
        "/integrations/connectors": {"GET", "POST"},
        "/integrations/connectors/{name}": {"PATCH", "DELETE"},
        "/printing/printers": {"GET", "POST"},
        "/printing/printers/{printer_id}": {"PATCH", "DELETE"},
        "/printing/slice": {"POST"},
        "/printing/slice/{job_id}": {"GET"},
        "/system/memory": {"GET"},
        "/system/diagnostics": {"GET"},
        "/video/files": {"GET"},
        "/video/edit": {"POST"},
        "/video/asset": {"GET"},
    }

    for path, methods in expected.items():
        assert path in route_map, f"Missing route: {path}"
        assert methods.issubset(route_map[path]), f"Route {path} missing methods {methods - route_map[path]}"


if __name__ == "__main__":
    test_integration_and_printing_routes_present()
    print("✓ test_api_routes_new_modes passed")