"""
Freshness Router & Cache for EDISON
Ensures time-sensitive queries get fresh data while caching stable answers.
"""

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
DB_PATH = REPO_ROOT / "data" / "freshness_cache.db"

# Keywords that indicate a time-sensitive query
TIME_SENSITIVE_KEYWORDS = [
    "latest", "current", "today", "now", "recent", "new release",
    "price", "stock", "weather", "score", "version", "update",
    "just released", "breaking", "live", "real-time", "driver",
    "this week", "this month", "2025", "2026",
]

DEFAULT_TTL = 3600       # 1 hour for general cache
FRESH_TTL = 300          # 5 minutes for time-sensitive queries
STALE_TTL = 86400        # 24 hours before forced refresh


class FreshnessCache:
    """SQLite-backed cache for web-sourced facts with TTL."""

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
            CREATE TABLE IF NOT EXISTS freshness_cache (
                fingerprint     TEXT PRIMARY KEY,
                query           TEXT NOT NULL,
                sources         TEXT DEFAULT '[]',
                facts_summary   TEXT,
                citations       TEXT DEFAULT '[]',
                retrieved_at    REAL NOT NULL,
                ttl             REAL NOT NULL,
                last_verified   REAL
            );
            CREATE INDEX IF NOT EXISTS idx_fc_retrieved ON freshness_cache(retrieved_at);
        """)
        self._conn().commit()
        logger.info("Freshness cache initialized")

    def fingerprint(self, query: str) -> str:
        normalized = " ".join(query.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached result if not stale. Returns None if expired/missing."""
        fp = self.fingerprint(query)
        row = self._conn().execute(
            "SELECT * FROM freshness_cache WHERE fingerprint = ?", (fp,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        age = time.time() - d["retrieved_at"]
        if age > d["ttl"]:
            logger.debug(f"Cache stale for query fingerprint {fp} (age={age:.0f}s, ttl={d['ttl']:.0f}s)")
            return None
        # Parse JSON fields
        for k in ("sources", "citations"):
            try:
                d[k] = json.loads(d[k]) if isinstance(d[k], str) else d[k]
            except (json.JSONDecodeError, TypeError):
                d[k] = []
        d["cache_hit"] = True
        d["age_s"] = round(age, 1)
        return d

    def put(
        self,
        query: str,
        facts_summary: str,
        sources: Optional[List[Dict[str, str]]] = None,
        citations: Optional[List[Dict[str, str]]] = None,
        ttl: Optional[float] = None,
    ):
        """Store a query result in the cache."""
        fp = self.fingerprint(query)
        now = time.time()
        if ttl is None:
            ttl = FRESH_TTL if self.is_time_sensitive(query) else DEFAULT_TTL
        self._conn().execute(
            """INSERT OR REPLACE INTO freshness_cache
               (fingerprint, query, sources, facts_summary, citations,
                retrieved_at, ttl, last_verified)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fp,
                query,
                json.dumps(sources or []),
                facts_summary,
                json.dumps(citations or []),
                now,
                ttl,
                now,
            ),
        )
        self._conn().commit()

    def invalidate(self, query: str):
        fp = self.fingerprint(query)
        self._conn().execute("DELETE FROM freshness_cache WHERE fingerprint = ?", (fp,))
        self._conn().commit()

    def cleanup_expired(self) -> int:
        """Remove entries older than STALE_TTL regardless of their original TTL."""
        cutoff = time.time() - STALE_TTL
        cur = self._conn().execute(
            "DELETE FROM freshness_cache WHERE retrieved_at < ?", (cutoff,)
        )
        self._conn().commit()
        return cur.rowcount

    @staticmethod
    def is_time_sensitive(query: str) -> bool:
        q = query.lower()
        return any(kw in q for kw in TIME_SENSITIVE_KEYWORDS)

    def needs_refresh(self, query: str) -> bool:
        """Check if query needs fresh web data."""
        cached = self.get(query)
        if cached is None:
            return True
        if self.is_time_sensitive(query):
            age = time.time() - cached["retrieved_at"]
            return age > FRESH_TTL
        return False

    def format_citations(self, citations: List[Dict[str, str]]) -> str:
        """Format citations for inclusion in LLM responses."""
        if not citations:
            return ""
        lines = ["\n\n**Sources:**"]
        for i, c in enumerate(citations, 1):
            title = c.get("title", "Source")
            url = c.get("url", "")
            date = c.get("date", "")
            line = f"[{i}] {title}"
            if url:
                line += f" — {url}"
            if date:
                line += f" ({date})"
            lines.append(line)
        return "\n".join(lines)
