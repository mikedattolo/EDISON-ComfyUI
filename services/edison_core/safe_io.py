"""Shared safe-IO helpers.

Centralised because many modules in services/edison_core/ were doing
``path.write_text(json.dumps(...))`` directly, which corrupts the file
if the process crashes mid-write or the disk fills up. Using a temp
file in the same directory followed by ``os.replace`` makes the swap
atomic on POSIX and Windows.

Also exposes ``read_json`` which gracefully handles a missing or
corrupted file (returns the supplied default) so callers don't need to
duplicate try/except boilerplate.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

PathLike = Union[str, os.PathLike, Path]

_log = logging.getLogger("edison.safe_io")


def atomic_write_text(path: PathLike, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically.

    Writes to a temp file in the same directory, fsyncs, then renames.
    A crash partway through leaves the original file untouched.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dest.name}.", suffix=".tmp", dir=str(dest.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(text)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                # Some filesystems (procfs, certain network mounts) don't
                # support fsync. The atomic rename still gives durability
                # vs. a partial truncate.
                pass
        os.replace(tmp_path, dest)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(
    path: PathLike,
    data: Any,
    *,
    indent: int = 2,
    default: Optional[Any] = str,
    encoding: str = "utf-8",
) -> None:
    """JSON-serialise ``data`` and write atomically to ``path``."""
    text = json.dumps(data, indent=indent, default=default)
    atomic_write_text(path, text, encoding=encoding)


def read_json(path: PathLike, default: Any = None) -> Any:
    """Read JSON from ``path``. Returns ``default`` on missing/corrupt file."""
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except (json.JSONDecodeError, OSError) as e:
        _log.warning("read_json: failed to load %s (%s); using default", p, e)
        return default


__all__ = ["atomic_write_text", "atomic_write_json", "read_json"]
