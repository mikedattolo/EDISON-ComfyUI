import pytest


def test_resolve_safe_path_accepts_nested_relative_path(tmp_path):
    from services.edison_core.safe_io import resolve_safe_path

    target = resolve_safe_path(tmp_path, "projects/demo/file.txt")

    assert target == tmp_path / "projects" / "demo" / "file.txt"


def test_resolve_safe_path_rejects_traversal(tmp_path):
    from services.edison_core.safe_io import resolve_safe_path

    with pytest.raises(ValueError):
        resolve_safe_path(tmp_path, "../outside.txt")


def test_resolve_safe_path_rejects_absolute_outside_path(tmp_path):
    from services.edison_core.safe_io import resolve_safe_path

    outside = tmp_path.parent / "outside.txt"

    with pytest.raises(ValueError):
        resolve_safe_path(tmp_path, outside)
