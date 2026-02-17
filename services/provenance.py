"""
Edison Provenance Tracker.

Stores metadata sidecar JSON for every generation or edit, including:
- model used, parameters, timestamp, source artifact
- memory usage and fallback triggers
- retries and error context
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROVENANCE_DIR = REPO_ROOT / "data" / "provenance"


@dataclass
class ProvenanceRecord:
    """Metadata record for a generation or edit."""
    record_id: str
    action: str  # "generate_image", "edit_image", "edit_file", "generate_video", etc.
    timestamp: float = 0
    model_used: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    source_artifact: Optional[str] = None  # file_id or path of input
    output_artifact: Optional[str] = None  # file_id or path of output
    memory_snapshot: Optional[Dict[str, Any]] = None
    fallback_used: bool = False
    fallback_details: Optional[str] = None
    retries: int = 0
    error: Optional[str] = None
    session_id: Optional[str] = None
    duration_seconds: float = 0

    def to_dict(self) -> dict:
        return asdict(self)


class ProvenanceTracker:
    """Write and read provenance sidecar files."""

    def __init__(self, base_dir: Optional[Any] = None):
        self._dir = Path(base_dir) if base_dir else PROVENANCE_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def record(self, record: Optional["ProvenanceRecord"] = None, *,
               action: str = "", model_used: Optional[str] = None,
               parameters: Optional[Dict[str, Any]] = None,
               source_artifacts: Optional[List[str]] = None,
               output_artifacts: Optional[List[str]] = None,
               session_id: Optional[str] = None,
               **kwargs) -> ProvenanceRecord:
        """Write a provenance record to disk.

        Accepts either a ProvenanceRecord or keyword arguments.
        Returns the ProvenanceRecord.
        """
        import uuid as _uuid
        if record is None:
            record = ProvenanceRecord(
                record_id=str(_uuid.uuid4()),
                action=action,
                timestamp=time.time(),
                model_used=model_used,
                parameters=parameters or {},
                source_artifact=",".join(source_artifacts) if source_artifacts else None,
                output_artifact=",".join(output_artifacts) if output_artifacts else None,
                session_id=session_id,
                **{k: v for k, v in kwargs.items()
                   if k in ProvenanceRecord.__dataclass_fields__},
            )
        fname = f"{record.record_id}.json"
        path = self._dir / fname
        try:
            path.write_text(json.dumps(record.to_dict(), indent=2, default=str))
        except Exception as e:
            logger.warning(f"Failed to write provenance record: {e}")
        return record

    def write_sidecar(self, record: "ProvenanceRecord",
                      artifact_path: Optional[str] = None) -> Optional[Path]:
        """Write a .provenance.json sidecar next to an artifact."""
        target = artifact_path or record.output_artifact
        if not target:
            return None
        sidecar = Path(target + ".provenance.json")
        try:
            sidecar.write_text(json.dumps(record.to_dict(), indent=2, default=str))
        except Exception as e:
            logger.warning(f"Failed to write sidecar: {e}")
        return sidecar

    def get(self, record_id: str) -> Optional[ProvenanceRecord]:
        """Read a provenance record."""
        path = self._dir / f"{record_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return ProvenanceRecord(**{k: v for k, v in data.items()
                                       if k in ProvenanceRecord.__dataclass_fields__})
        except Exception as e:
            logger.warning(f"Failed to read provenance record: {e}")
            return None

    def list_recent(self, limit: int = 50) -> List[Dict]:
        """List recent provenance records."""
        records = []
        try:
            files = sorted(self._dir.glob("*.json"), key=os.path.getmtime, reverse=True)
            for f in files[:limit]:
                try:
                    data = json.loads(f.read_text())
                    records.append(data)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Failed to list provenance records: {e}")
        return records


# ── Singleton ────────────────────────────────────────────────────────────

_tracker: Optional[ProvenanceTracker] = None

def get_provenance_tracker() -> ProvenanceTracker:
    global _tracker
    if _tracker is None:
        _tracker = ProvenanceTracker()
    return _tracker
