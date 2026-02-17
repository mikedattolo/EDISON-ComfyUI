"""
Workflow Intelligence for EDISON
Tracks generation settings that produced good results,
stores recommended workflows, and enables reuse via style/workflow selection.
"""

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
DB_PATH = REPO_ROOT / "data" / "workflows.db"


class WorkflowMemory:
    """Track and recommend generation workflows based on past success."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def _init_db(self):
        self._conn().executescript("""
            CREATE TABLE IF NOT EXISTS workflow_records (
                id              TEXT PRIMARY KEY,
                job_type        TEXT NOT NULL,
                prompt          TEXT,
                params          TEXT DEFAULT '{}',
                style_profile   TEXT,
                model_used      TEXT,
                rating          REAL DEFAULT 0.0,
                success         INTEGER DEFAULT 0,
                created_at      REAL NOT NULL,
                job_id          TEXT,
                tags            TEXT DEFAULT '[]',
                notes           TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_wf_type ON workflow_records(job_type);
            CREATE INDEX IF NOT EXISTS idx_wf_rating ON workflow_records(rating DESC);
            CREATE INDEX IF NOT EXISTS idx_wf_style ON workflow_records(style_profile);
        """)
        self._conn().commit()
        logger.info("Workflow intelligence initialized")

    def record_result(
        self,
        job_type: str,
        prompt: str,
        params: Dict[str, Any],
        success: bool = True,
        rating: float = 0.0,
        style_profile: Optional[str] = None,
        model_used: Optional[str] = None,
        job_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Record a generation result for workflow learning."""
        import uuid
        record_id = str(uuid.uuid4())
        self._conn().execute(
            """INSERT INTO workflow_records
               (id, job_type, prompt, params, style_profile, model_used,
                rating, success, created_at, job_id, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record_id, job_type, prompt, json.dumps(params),
                style_profile, model_used, rating,
                int(success), time.time(), job_id, json.dumps(tags or []),
            ),
        )
        self._conn().commit()
        return record_id

    def get_recommendations(
        self,
        job_type: str,
        limit: int = 5,
        min_rating: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Get recommended workflows ranked by success and rating."""
        rows = self._conn().execute(
            """SELECT * FROM workflow_records
               WHERE job_type = ? AND success = 1 AND rating >= ?
               ORDER BY rating DESC, created_at DESC
               LIMIT ?""",
            (job_type, min_rating, limit),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d["params"] = json.loads(d["params"])
                d["tags"] = json.loads(d["tags"])
            except (json.JSONDecodeError, TypeError):
                pass
            results.append(d)
        return results

    def get_best_params(self, job_type: str, style_profile: Optional[str] = None) -> Optional[Dict]:
        """Get the best parameters for a job type, optionally filtered by style."""
        clauses = ["job_type = ?", "success = 1"]
        params = [job_type]
        if style_profile:
            clauses.append("style_profile = ?")
            params.append(style_profile)

        row = self._conn().execute(
            f"""SELECT params FROM workflow_records
                WHERE {' AND '.join(clauses)}
                ORDER BY rating DESC, created_at DESC
                LIMIT 1""",
            params,
        ).fetchone()

        if row:
            try:
                return json.loads(row["params"])
            except (json.JSONDecodeError, TypeError):
                pass
        return None

    def rate_result(self, record_id: str, rating: float) -> bool:
        """Update rating for a workflow record."""
        cur = self._conn().execute(
            "UPDATE workflow_records SET rating = ? WHERE id = ?",
            (min(5.0, max(0.0, rating)), record_id),
        )
        self._conn().commit()
        return cur.rowcount > 0

    def get_stats(self) -> Dict[str, Any]:
        conn = self._conn()
        total = conn.execute("SELECT COUNT(*) as c FROM workflow_records").fetchone()["c"]
        by_type = {}
        for r in conn.execute(
            "SELECT job_type, COUNT(*) as c, AVG(rating) as avg_rating "
            "FROM workflow_records GROUP BY job_type"
        ).fetchall():
            by_type[r["job_type"]] = {"count": r["c"], "avg_rating": round(r["avg_rating"] or 0, 2)}
        return {"total": total, "by_type": by_type}
