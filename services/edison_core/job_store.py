"""
Unified Generation Job Store for EDISON
SQLite-backed job tracking for all generation types: image, video, music, mesh.
"""

import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
DB_PATH = REPO_ROOT / "data" / "jobs.db"

# Valid statuses for generation jobs
VALID_STATUSES = {"queued", "loading", "generating", "encoding", "complete", "error", "cancelled"}
VALID_JOB_TYPES = {"image", "video", "music", "mesh", "persona_video", "agent", "swarm", "comfyui", "print", "node"}


def normalize_status_for_store(status: str) -> str:
    value = str(status or "queued").strip().lower()
    mapping = {
        "validating": "loading",
        "running": "generating",
        "paused": "generating",
        "needs_review": "complete",
        "completed": "complete",
        "done": "complete",
        "failed": "error",
        "cancel_requested": "cancelled",
    }
    return mapping.get(value, value if value in VALID_STATUSES else "queued")


class JobStore:
    """SQLite-backed unified generation job store."""

    _instance: Optional["JobStore"] = None
    _lock = threading.Lock()

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    @classmethod
    def get_instance(cls, db_path: Optional[Path] = None) -> "JobStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(db_path)
        return cls._instance

    # ── connection per thread ────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    # ── schema ───────────────────────────────────────────────────────────

    def _init_db(self):
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id        TEXT PRIMARY KEY,
                job_type      TEXT NOT NULL,
                status        TEXT NOT NULL DEFAULT 'queued',
                prompt        TEXT,
                negative_prompt TEXT,
                params        TEXT,          -- JSON
                outputs       TEXT,          -- JSON list of paths/urls
                provenance    TEXT,          -- JSON (model, workflow, seed, versions)
                error_log     TEXT,
                created_at    REAL NOT NULL,
                started_at    REAL,
                completed_at  REAL,
                duration_s    REAL,
                correlation_id TEXT,
                title          TEXT,
                summary        TEXT,
                stage          TEXT,
                progress       REAL DEFAULT 0,
                updated_at     REAL,
                logs           TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(job_type);
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
        """)
        self._ensure_columns(conn)
        conn.commit()
        logger.info(f"Job store initialized at {self.db_path}")

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        desired = {
            "title": "TEXT",
            "summary": "TEXT",
            "stage": "TEXT",
            "progress": "REAL DEFAULT 0",
            "updated_at": "REAL",
            "logs": "TEXT",
        }
        for name, ddl in desired.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {ddl}")

    # ── CRUD ─────────────────────────────────────────────────────────────

    def create_job(
        self,
        job_type: str,
        prompt: str = "",
        negative_prompt: str = "",
        params: Optional[Dict] = None,
        provenance: Optional[Dict] = None,
        correlation_id: Optional[str] = None,
        job_id: Optional[str] = None,
        title: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> str:
        if job_type not in VALID_JOB_TYPES:
            raise ValueError(f"Invalid job_type '{job_type}', must be one of {VALID_JOB_TYPES}")
        job_id = str(job_id or uuid.uuid4())
        now = time.time()
        conn = self._conn()
        conn.execute(
            """INSERT INTO jobs
               (job_id, job_type, status, prompt, negative_prompt, params,
                outputs, provenance, created_at, correlation_id, title, summary, stage, progress, updated_at, logs)
               VALUES (?, ?, 'queued', ?, ?, ?, '[]', ?, ?, ?, ?, ?, ?, 0, ?, '[]')""",
            (
                job_id,
                job_type,
                prompt,
                negative_prompt,
                json.dumps(params or {}),
                json.dumps(provenance or {}),
                now,
                correlation_id,
                title or (params or {}).get("title") or prompt[:80],
                summary or (params or {}).get("summary") or "",
                (params or {}).get("stage") or "queued",
                now,
            ),
        )
        conn.commit()
        logger.info(f"Created {job_type} job {job_id}")
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn().execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def list_jobs(
        self,
        job_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        clauses, params_list = [], []
        if job_type:
            clauses.append("job_type = ?")
            params_list.append(job_type)
        if status:
            clauses.append("status = ?")
            params_list.append(status)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn().execute(
            f"SELECT * FROM jobs {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (*params_list, limit, offset),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def update_status(
        self,
        job_id: str,
        status: str,
        error_log: Optional[str] = None,
        outputs: Optional[List[str]] = None,
        provenance: Optional[Dict] = None,
    ):
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'")
        conn = self._conn()
        now = time.time()

        sets = ["status = ?", "updated_at = ?"]
        vals: list = [status]
        vals.append(now)

        if status == "generating":
            sets.append("started_at = COALESCE(started_at, ?)")
            vals.append(now)
        if status in ("complete", "error", "cancelled"):
            sets.append("completed_at = ?")
            vals.append(now)
            sets.append("duration_s = ? - COALESCE(started_at, created_at)")
            vals.append(now)
        if error_log is not None:
            sets.append("error_log = ?")
            vals.append(error_log)
        if outputs is not None:
            sets.append("outputs = ?")
            vals.append(json.dumps(outputs))
        if provenance is not None:
            sets.append("provenance = ?")
            vals.append(json.dumps(provenance))

        vals.append(job_id)
        conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?", vals)
        conn.commit()

    def set_progress(self, job_id: str, progress: float = 0.0, stage: Optional[str] = None):
        conn = self._conn()
        progress = max(0.0, min(1.0, float(progress or 0.0)))
        conn.execute(
            "UPDATE jobs SET progress = ?, stage = COALESCE(?, stage), updated_at = ? WHERE job_id = ?",
            (progress, stage, time.time(), job_id),
        )
        conn.commit()

    def append_log(self, job_id: str, message: str, level: str = "info"):
        job = self.get_job(job_id)
        if not job:
            return
        logs = job.get("logs") or []
        if not isinstance(logs, list):
            logs = []
        logs.append({"timestamp": time.time(), "level": level, "message": str(message)})
        self._conn().execute(
            "UPDATE jobs SET logs = ?, updated_at = ? WHERE job_id = ?",
            (json.dumps(logs[-500:]), time.time(), job_id),
        )
        self._conn().commit()

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job if it is not already complete/error."""
        job = self.get_job(job_id)
        if not job:
            return False
        if job["status"] in ("complete", "error", "cancelled"):
            return False
        self.update_status(job_id, "cancelled")
        logger.info(f"Cancelled job {job_id}")
        return True

    def delete_job(self, job_id: str) -> bool:
        conn = self._conn()
        cur = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        conn.commit()
        return cur.rowcount > 0

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for key in ("params", "outputs", "provenance", "logs"):
            if d.get(key):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    def write_metadata_sidecar(self, job_id: str, output_path: Path):
        """Write a .meta.json sidecar file next to an output artifact."""
        job = self.get_job(job_id)
        if not job:
            return
        meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
        meta = {
            "job_id": job["job_id"],
            "job_type": job["job_type"],
            "prompt": job["prompt"],
            "negative_prompt": job.get("negative_prompt", ""),
            "params": job.get("params", {}),
            "provenance": job.get("provenance", {}),
            "created_at": job["created_at"],
            "completed_at": job.get("completed_at"),
            "duration_s": job.get("duration_s"),
        }
        try:
            meta_path.write_text(json.dumps(meta, indent=2, default=str))
            logger.debug(f"Wrote sidecar metadata to {meta_path}")
        except Exception as e:
            logger.warning(f"Failed to write sidecar: {e}")
