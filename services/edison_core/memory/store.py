"""
Memory Store for EDISON
SQLite-backed storage for three-tier memory entries with CRUD + hygiene.
"""

import json
import logging
import math
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import MemoryEntry, MemoryType

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
DB_PATH = REPO_ROOT / "data" / "memory.db"


class MemoryStore:
    """Persistent three-tier memory store."""

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
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    def _init_db(self):
        self._conn().executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id                     TEXT PRIMARY KEY,
                memory_type            TEXT NOT NULL,
                key                    TEXT,
                content                TEXT NOT NULL,
                timestamp              REAL NOT NULL,
                confidence             REAL NOT NULL DEFAULT 0.8,
                tags                   TEXT DEFAULT '[]',
                source_conversation_id TEXT,
                pinned                 INTEGER DEFAULT 0,
                metadata               TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_mem_key ON memories(key);
            CREATE INDEX IF NOT EXISTS idx_mem_ts ON memories(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_mem_conv ON memories(source_conversation_id);
        """)
        self._conn().commit()
        logger.info(f"Memory store initialized at {self.db_path}")

    # ── CRUD ─────────────────────────────────────────────────────────────

    def save(self, entry: MemoryEntry) -> str:
        conn = self._conn()
        conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, memory_type, key, content, timestamp, confidence,
                tags, source_conversation_id, pinned, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.id,
                entry.memory_type.value,
                entry.key,
                entry.content,
                entry.timestamp,
                entry.confidence,
                json.dumps(entry.tags),
                entry.source_conversation_id,
                int(entry.pinned),
                json.dumps(entry.metadata),
            ),
        )
        conn.commit()
        return entry.id

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        row = self._conn().execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_entry(row) if row else None

    def search(
        self,
        memory_type: Optional[MemoryType] = None,
        key: Optional[str] = None,
        tag: Optional[str] = None,
        conversation_id: Optional[str] = None,
        pinned_only: bool = False,
        limit: int = 50,
    ) -> List[MemoryEntry]:
        clauses, params = [], []
        if memory_type:
            clauses.append("memory_type = ?")
            params.append(memory_type.value)
        if key:
            clauses.append("key = ?")
            params.append(key)
        if tag:
            clauses.append("tags LIKE ?")
            params.append(f'%"{tag}"%')
        if conversation_id:
            clauses.append("source_conversation_id = ?")
            params.append(conversation_id)
        if pinned_only:
            clauses.append("pinned = 1")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn().execute(
            f"SELECT * FROM memories {where} ORDER BY timestamp DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def delete(self, memory_id: str) -> bool:
        cur = self._conn().execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn().commit()
        return cur.rowcount > 0

    def update(self, memory_id: str, **kwargs) -> bool:
        sets, vals = [], []
        allowed = {"content", "confidence", "tags", "pinned", "key", "metadata"}
        for k, v in kwargs.items():
            if k not in allowed:
                continue
            if k == "tags":
                v = json.dumps(v)
            elif k == "metadata":
                v = json.dumps(v)
            elif k == "pinned":
                v = int(v)
            sets.append(f"{k} = ?")
            vals.append(v)
        if not sets:
            return False
        vals.append(memory_id)
        cur = self._conn().execute(
            f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", vals
        )
        self._conn().commit()
        return cur.rowcount > 0

    # ── Hygiene Worker ───────────────────────────────────────────────────

    def run_hygiene(self) -> Dict[str, int]:
        """Deduplicate, merge, and prune memory entries."""
        stats = {"duplicates_removed": 0, "merged": 0, "pruned": 0}

        # 1. Remove exact content duplicates (keep highest confidence)
        conn = self._conn()
        dupes = conn.execute("""
            SELECT content, memory_type, COUNT(*) as cnt
            FROM memories
            GROUP BY content, memory_type
            HAVING cnt > 1
        """).fetchall()

        for dupe in dupes:
            rows = conn.execute(
                """SELECT id, confidence, timestamp, pinned FROM memories
                   WHERE content = ? AND memory_type = ?
                   ORDER BY pinned DESC, confidence DESC, timestamp DESC""",
                (dupe["content"], dupe["memory_type"]),
            ).fetchall()
            # Keep the first (best), delete rest
            for row in rows[1:]:
                conn.execute("DELETE FROM memories WHERE id = ?", (row["id"],))
                stats["duplicates_removed"] += 1

        # 2. Prune low-confidence, old, unpinned entries
        cutoff = time.time() - 90 * 86400  # 90 days
        cur = conn.execute(
            """DELETE FROM memories
               WHERE pinned = 0 AND confidence < 0.3 AND timestamp < ?""",
            (cutoff,),
        )
        stats["pruned"] = cur.rowcount

        # 3. Merge similar profile entries with same key
        profile_keys = conn.execute(
            """SELECT key, COUNT(*) as cnt FROM memories
               WHERE memory_type = 'profile' AND key IS NOT NULL
               GROUP BY key HAVING cnt > 1"""
        ).fetchall()

        for pk in profile_keys:
            rows = conn.execute(
                """SELECT * FROM memories
                   WHERE memory_type = 'profile' AND key = ?
                   ORDER BY pinned DESC, confidence DESC, timestamp DESC""",
                (pk["key"],),
            ).fetchall()
            if len(rows) <= 1:
                continue
            # Keep most confident/recent, boost its confidence
            best = rows[0]
            new_conf = min(1.0, best["confidence"] + 0.05)
            conn.execute(
                "UPDATE memories SET confidence = ? WHERE id = ?",
                (new_conf, best["id"]),
            )
            for row in rows[1:]:
                conn.execute("DELETE FROM memories WHERE id = ?", (row["id"],))
                stats["merged"] += 1

        conn.commit()
        logger.info(f"Memory hygiene: {stats}")
        return stats

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        d = dict(row)
        tags = d.get("tags", "[]")
        meta = d.get("metadata", "{}")
        try:
            tags = json.loads(tags) if isinstance(tags, str) else tags
        except json.JSONDecodeError:
            tags = []
        try:
            meta = json.loads(meta) if isinstance(meta, str) else meta
        except json.JSONDecodeError:
            meta = {}
        return MemoryEntry(
            id=d["id"],
            memory_type=MemoryType(d["memory_type"]),
            key=d.get("key"),
            content=d["content"],
            timestamp=d["timestamp"],
            confidence=d["confidence"],
            tags=tags,
            source_conversation_id=d.get("source_conversation_id"),
            pinned=bool(d.get("pinned", 0)),
            metadata=meta,
        )

    def get_stats(self) -> Dict[str, Any]:
        conn = self._conn()
        total = conn.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
        by_type = {}
        for row in conn.execute(
            "SELECT memory_type, COUNT(*) as c FROM memories GROUP BY memory_type"
        ).fetchall():
            by_type[row["memory_type"]] = row["c"]
        pinned = conn.execute("SELECT COUNT(*) as c FROM memories WHERE pinned = 1").fetchone()["c"]
        return {"total": total, "by_type": by_type, "pinned": pinned}
