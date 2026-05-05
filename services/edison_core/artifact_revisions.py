"""
Persistent artifact revision store.

Phase 2 goal: support "revise / compare versions / restore" controls in the
chat UI. Where :class:`ArtifactStream` handled live deltas during one
generation, this module persists committed revisions across sessions so
users can roll back, diff, and branch from any historical version.

Storage is intentionally JSON-on-disk to match EDISON's existing
local-first pattern. Each artifact is a directory under
``data/artifact_revisions/<artifact_id>/`` containing one ``rev_<id>.json``
file per revision plus a ``manifest.json`` that lists them in order.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_ROOT = REPO_ROOT / "data" / "artifact_revisions"


class ArtifactRevisionStore:
    """File-backed store for artifact revisions."""

    _instance: Optional["ArtifactRevisionStore"] = None
    _lock = threading.Lock()

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = (root or DEFAULT_ROOT).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_instance(cls, root: Optional[Path] = None) -> "ArtifactRevisionStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(root)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            cls._instance = None

    # ── path helpers ─────────────────────────────────────────────────

    def _artifact_dir(self, artifact_id: str) -> Path:
        safe = "".join(c for c in artifact_id if c.isalnum() or c in ("-", "_"))
        if not safe or safe != artifact_id:
            raise ValueError(f"invalid artifact_id: {artifact_id!r}")
        return self.root / safe

    def _manifest_path(self, artifact_id: str) -> Path:
        return self._artifact_dir(artifact_id) / "manifest.json"

    def _read_manifest(self, artifact_id: str) -> Dict[str, Any]:
        path = self._manifest_path(artifact_id)
        if not path.exists():
            return {"artifact_id": artifact_id, "revisions": [], "created_at": time.time()}
        try:
            return json.loads(path.read_text())
        except Exception:
            logger.exception("corrupt manifest at %s; starting fresh", path)
            return {"artifact_id": artifact_id, "revisions": [], "created_at": time.time()}

    def _write_manifest(self, artifact_id: str, manifest: Dict[str, Any]) -> None:
        path = self._manifest_path(artifact_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, default=str))

    # ── public API ───────────────────────────────────────────────────

    def add_revision(
        self,
        artifact_id: str,
        content: str,
        *,
        kind: str = "document",
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist a new revision. Returns the revision record."""
        adir = self._artifact_dir(artifact_id)
        adir.mkdir(parents=True, exist_ok=True)
        revision_id = f"rev_{uuid.uuid4().hex[:10]}"
        record = {
            "revision_id": revision_id,
            "artifact_id": artifact_id,
            "kind": kind,
            "title": title,
            "created_at": time.time(),
            "metadata": metadata or {},
            "content_path": f"{revision_id}.json",
        }
        (adir / record["content_path"]).write_text(json.dumps({
            "revision_id": revision_id,
            "content": content,
        }, indent=2))

        manifest = self._read_manifest(artifact_id)
        manifest.setdefault("revisions", []).append({
            k: v for k, v in record.items() if k != "content"
        })
        manifest["updated_at"] = time.time()
        if title:
            manifest["title"] = title
        manifest["kind"] = kind
        self._write_manifest(artifact_id, manifest)
        return record

    def list_revisions(self, artifact_id: str) -> List[Dict[str, Any]]:
        manifest = self._read_manifest(artifact_id)
        return list(manifest.get("revisions", []))

    def get_revision(self, artifact_id: str, revision_id: str) -> Optional[Dict[str, Any]]:
        adir = self._artifact_dir(artifact_id)
        path = adir / f"{revision_id}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except Exception:
            logger.exception("corrupt revision file %s", path)
            return None
        manifest_entry = next(
            (r for r in self.list_revisions(artifact_id) if r["revision_id"] == revision_id),
            {},
        )
        return {**manifest_entry, "content": payload.get("content", "")}

    def diff(self, artifact_id: str, a: str, b: str) -> Dict[str, Any]:
        ra = self.get_revision(artifact_id, a)
        rb = self.get_revision(artifact_id, b)
        return {
            "artifact_id": artifact_id,
            "a": a,
            "b": b,
            "a_exists": ra is not None,
            "b_exists": rb is not None,
            "a_lines": (ra or {}).get("content", "").count("\n") + (1 if ra else 0),
            "b_lines": (rb or {}).get("content", "").count("\n") + (1 if rb else 0),
            "a_chars": len((ra or {}).get("content", "")),
            "b_chars": len((rb or {}).get("content", "")),
        }

    def restore(self, artifact_id: str, revision_id: str) -> Dict[str, Any]:
        """Mark a revision as the new tip by appending a copy with a fresh id."""
        rev = self.get_revision(artifact_id, revision_id)
        if rev is None:
            raise KeyError(f"unknown revision {revision_id!r} for {artifact_id}")
        return self.add_revision(
            artifact_id,
            rev["content"],
            kind=rev.get("kind", "document"),
            title=rev.get("title"),
            metadata={"restored_from": revision_id},
        )

    def delete_artifact(self, artifact_id: str) -> bool:
        adir = self._artifact_dir(artifact_id)
        if not adir.exists():
            return False
        for child in adir.iterdir():
            try:
                child.unlink()
            except IsADirectoryError:
                pass
        try:
            adir.rmdir()
        except OSError:
            pass
        return True
