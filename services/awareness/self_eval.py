"""
Self-Evaluation Loop for Edison.

After each tool execution or generation, logs:
  - success / failure
  - execution time
  - user follow-up correction

Stores evaluations and exposes aggregate statistics to improve
routing confidence over time.
"""

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
DB_PATH = REPO_ROOT / "data" / "self_eval.db"


@dataclass
class EvalEntry:
    """Record of a single tool/generation execution and its outcome."""
    eval_id: str
    session_id: str
    action: str              # tool name, generation type, or "llm_respond"
    intent: str
    goal: str
    success: bool
    duration_s: float
    error: Optional[str] = None
    user_correction: Optional[str] = None  # set later if user corrects
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eval_id": self.eval_id,
            "session_id": self.session_id,
            "action": self.action,
            "intent": self.intent,
            "goal": self.goal,
            "success": self.success,
            "duration_s": round(self.duration_s, 3),
            "error": self.error,
            "user_correction": self.user_correction,
            "timestamp": self.timestamp,
        }


class SelfEvaluator:
    """Tracks execution outcomes in SQLite for post-hoc analysis."""

    def __init__(self, db_path=None):
        if db_path is not None:
            self.db_path = Path(db_path) if not isinstance(db_path, Path) else db_path
        else:
            self.db_path = DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()
        self._counter = 0
        self._lock = threading.Lock()
        logger.info(f"SelfEvaluator initialized at {self.db_path}")

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    def _init_db(self):
        self._conn().executescript("""
            CREATE TABLE IF NOT EXISTS evaluations (
                eval_id       TEXT PRIMARY KEY,
                session_id    TEXT NOT NULL,
                action        TEXT NOT NULL,
                intent        TEXT,
                goal          TEXT,
                success       INTEGER NOT NULL,
                duration_s    REAL,
                error         TEXT,
                user_correction TEXT,
                timestamp     REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_eval_session ON evaluations(session_id);
            CREATE INDEX IF NOT EXISTS idx_eval_action ON evaluations(action);
            CREATE INDEX IF NOT EXISTS idx_eval_ts ON evaluations(timestamp DESC);
        """)
        self._conn().commit()

    # ── Record outcomes ──────────────────────────────────────────────────

    def _next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"eval_{int(time.time())}_{self._counter}"

    def record(
        self,
        session_id: str,
        action: str,
        success: bool,
        duration_s: float,
        intent: str = "",
        goal: str = "",
        error: Optional[str] = None,
    ) -> str:
        """Record an execution outcome.  Returns the eval_id."""
        eval_id = self._next_id()
        now = time.time()
        try:
            self._conn().execute(
                """INSERT INTO evaluations
                   (eval_id, session_id, action, intent, goal, success,
                    duration_s, error, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (eval_id, session_id, action, intent, goal,
                 1 if success else 0, duration_s, error, now),
            )
            self._conn().commit()
            log_fn = logger.info if success else logger.warning
            log_fn(
                f"EVAL [{eval_id}]: action={action}, success={success}, "
                f"duration={duration_s:.2f}s"
                + (f", error={error[:100]}" if error else "")
            )
        except Exception as e:
            logger.warning(f"Failed to record evaluation: {e}")
        return eval_id

    def record_correction(self, eval_id: str, correction: str):
        """Record that the user corrected the output of a previous action."""
        try:
            self._conn().execute(
                "UPDATE evaluations SET user_correction = ? WHERE eval_id = ?",
                (correction[:1000], eval_id),
            )
            self._conn().commit()
            logger.info(f"EVAL correction [{eval_id}]: {correction[:100]}")
        except Exception as e:
            logger.warning(f"Failed to record correction: {e}")

    # ── Query ────────────────────────────────────────────────────────────

    def get_recent(self, limit: int = 20, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return recent evaluations."""
        if session_id:
            rows = self._conn().execute(
                "SELECT * FROM evaluations WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        else:
            rows = self._conn().execute(
                "SELECT * FROM evaluations ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self, action: Optional[str] = None, window_hours: int = 24) -> Dict[str, Any]:
        """Aggregate success/failure statistics."""
        cutoff = time.time() - (window_hours * 3600)
        if action:
            rows = self._conn().execute(
                "SELECT success, COUNT(*) as cnt, AVG(duration_s) as avg_dur "
                "FROM evaluations WHERE action = ? AND timestamp > ? GROUP BY success",
                (action, cutoff),
            ).fetchall()
        else:
            rows = self._conn().execute(
                "SELECT success, COUNT(*) as cnt, AVG(duration_s) as avg_dur "
                "FROM evaluations WHERE timestamp > ? GROUP BY success",
                (cutoff,),
            ).fetchall()
        total = sum(r["cnt"] for r in rows)
        successes = sum(r["cnt"] for r in rows if r["success"])
        avg_dur = sum(r["avg_dur"] * r["cnt"] for r in rows) / total if total else 0
        return {
            "total": total,
            "successes": successes,
            "failures": total - successes,
            "success_rate": round(successes / total, 3) if total else 0,
            "avg_duration_s": round(avg_dur, 3),
            "window_hours": window_hours,
            "action_filter": action,
        }

    def get_correction_rate(self, window_hours: int = 24) -> Dict[str, Any]:
        """How often users correct Edison's output."""
        cutoff = time.time() - (window_hours * 3600)
        row = self._conn().execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN user_correction IS NOT NULL THEN 1 ELSE 0 END) as corrected "
            "FROM evaluations WHERE timestamp > ?",
            (cutoff,),
        ).fetchone()
        total = row["total"]
        corrected = row["corrected"]
        return {
            "total": total,
            "corrected": corrected,
            "correction_rate": round(corrected / total, 3) if total else 0,
            "window_hours": window_hours,
        }


# ── Singleton ────────────────────────────────────────────────────────────

_evaluator: Optional[SelfEvaluator] = None
_eval_lock = threading.Lock()


def get_self_evaluator() -> SelfEvaluator:
    global _evaluator
    if _evaluator is None:
        with _eval_lock:
            if _evaluator is None:
                _evaluator = SelfEvaluator()
    return _evaluator
