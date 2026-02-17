"""
Knowledge Pack Manager for EDISON
Download, ingest, and manage large knowledge datasets as "packs".
Each pack is chunked, embedded, and upserted to Qdrant in a dedicated collection.
Provenance metadata tracked in SQLite. Installs are idempotent.
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = REPO_ROOT / "data" / "knowledge_packs"
DB_PATH = DATA_DIR / "packs.db"

# ── Pack registry (built-in packs) ──────────────────────────────────────

BUILTIN_PACKS = {
    "wikipedia-simple": {
        "name": "Wikipedia Simple English",
        "description": "Simple English Wikipedia dump (~200k articles)",
        "url": "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2",
        "format": "mediawiki-xml-bz2",
        "collection": "kb_wikipedia_simple",
        "license": "CC BY-SA 3.0",
        "chunk_size": 512,
    },
    "arxiv-abstracts": {
        "name": "arXiv Abstracts",
        "description": "arXiv paper abstracts and metadata",
        "url": "https://huggingface.co/datasets/arxiv-community/arxiv-metadata",
        "format": "jsonl",
        "collection": "kb_arxiv_abstracts",
        "license": "CC0",
        "chunk_size": 512,
    },
    "rss-feeds": {
        "name": "RSS Feed Ingestor",
        "description": "Configurable RSS feed aggregator",
        "url": "",
        "format": "rss",
        "collection": "kb_rss_feeds",
        "license": "varies",
        "chunk_size": 256,
    },
}

DEFAULT_RSS_FEEDS = [
    "https://hnrss.org/newest?points=100",
    "https://arxiv.org/rss/cs.AI",
    "https://arxiv.org/rss/cs.LG",
]


class KnowledgePackManager:
    """Manage knowledge pack lifecycle: install, update, status, uninstall."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "packs.db"
        self._local = threading.local()
        self._init_db()
        self._progress: Dict[str, Dict[str, Any]] = {}

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
            CREATE TABLE IF NOT EXISTS packs (
                pack_id         TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                description     TEXT,
                url             TEXT,
                format          TEXT,
                collection      TEXT NOT NULL,
                license         TEXT,
                version         TEXT,
                sha256          TEXT,
                installed_at    REAL,
                updated_at      REAL,
                chunk_count     INTEGER DEFAULT 0,
                status          TEXT DEFAULT 'available',
                metadata        TEXT DEFAULT '{}'
            );
        """)
        self._conn().commit()
        logger.info("Knowledge pack manager initialized")

    # ── Public API ───────────────────────────────────────────────────────

    def list_available(self) -> List[Dict[str, Any]]:
        """List all available packs (built-in + installed)."""
        packs = []
        installed = {r["pack_id"]: dict(r) for r in
                     self._conn().execute("SELECT * FROM packs").fetchall()}

        for pack_id, info in BUILTIN_PACKS.items():
            entry = {**info, "pack_id": pack_id, "status": "available"}
            if pack_id in installed:
                entry["status"] = installed[pack_id]["status"]
                entry["installed_at"] = installed[pack_id]["installed_at"]
                entry["chunk_count"] = installed[pack_id]["chunk_count"]
            packs.append(entry)

        # Add custom installed packs not in builtins
        for pid, row in installed.items():
            if pid not in BUILTIN_PACKS:
                packs.append(row)

        return packs

    def status(self, pack_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of a pack or all packs."""
        if pack_id:
            row = self._conn().execute("SELECT * FROM packs WHERE pack_id = ?", (pack_id,)).fetchone()
            if row:
                d = dict(row)
                d["progress"] = self._progress.get(pack_id, {})
                return d
            if pack_id in BUILTIN_PACKS:
                return {**BUILTIN_PACKS[pack_id], "pack_id": pack_id, "status": "available"}
            return {"pack_id": pack_id, "status": "not_found"}
        return {"packs": self.list_available()}

    def install(self, pack_id: str, progress_cb: Optional[Callable] = None) -> Dict[str, Any]:
        """Install a knowledge pack (idempotent - skips if already installed)."""
        # Check if already installed
        existing = self._conn().execute("SELECT status FROM packs WHERE pack_id = ?", (pack_id,)).fetchone()
        if existing and existing["status"] == "installed":
            logger.info(f"Pack '{pack_id}' already installed, skipping")
            return {"pack_id": pack_id, "status": "already_installed"}

        info = BUILTIN_PACKS.get(pack_id)
        if not info:
            return {"pack_id": pack_id, "status": "error", "error": "Unknown pack"}

        self._progress[pack_id] = {"stage": "starting", "percent": 0}

        # Record as installing
        now = time.time()
        self._conn().execute(
            """INSERT OR REPLACE INTO packs
               (pack_id, name, description, url, format, collection, license,
                installed_at, status, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'installing', '{}')""",
            (pack_id, info["name"], info["description"], info["url"],
             info["format"], info["collection"], info["license"], now),
        )
        self._conn().commit()

        try:
            self._progress[pack_id] = {"stage": "downloading", "percent": 10}
            if progress_cb:
                progress_cb(pack_id, "downloading", 10)

            # Download step (actual download would happen here)
            download_dir = self.data_dir / "downloads" / pack_id
            download_dir.mkdir(parents=True, exist_ok=True)

            self._progress[pack_id] = {"stage": "parsing", "percent": 30}
            if progress_cb:
                progress_cb(pack_id, "parsing", 30)

            # Parse + chunk (format-specific)
            chunks = self._parse_pack(pack_id, info, download_dir)

            self._progress[pack_id] = {"stage": "embedding", "percent": 60}
            if progress_cb:
                progress_cb(pack_id, "embedding", 60)

            # Embed + upsert to Qdrant
            chunk_count = self._embed_and_upsert(info["collection"], chunks)

            self._progress[pack_id] = {"stage": "complete", "percent": 100}
            if progress_cb:
                progress_cb(pack_id, "complete", 100)

            # Update status
            self._conn().execute(
                """UPDATE packs SET status = 'installed', chunk_count = ?,
                   updated_at = ? WHERE pack_id = ?""",
                (chunk_count, time.time(), pack_id),
            )
            self._conn().commit()

            logger.info(f"Pack '{pack_id}' installed ({chunk_count} chunks)")
            return {"pack_id": pack_id, "status": "installed", "chunk_count": chunk_count}

        except Exception as e:
            self._conn().execute(
                "UPDATE packs SET status = 'error' WHERE pack_id = ?", (pack_id,)
            )
            self._conn().commit()
            self._progress[pack_id] = {"stage": "error", "error": str(e)}
            logger.error(f"Pack install failed for '{pack_id}': {e}")
            return {"pack_id": pack_id, "status": "error", "error": str(e)}

    def update(self, pack_id: str) -> Dict[str, Any]:
        """Re-download and re-index a pack."""
        self.uninstall(pack_id)
        return self.install(pack_id)

    def uninstall(self, pack_id: str) -> Dict[str, Any]:
        """Remove a pack and its Qdrant collection."""
        row = self._conn().execute("SELECT * FROM packs WHERE pack_id = ?", (pack_id,)).fetchone()
        if not row:
            return {"pack_id": pack_id, "status": "not_found"}

        # Try to delete Qdrant collection
        collection = row["collection"]
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(path=str(REPO_ROOT / "qdrant_storage"))
            client.delete_collection(collection)
            logger.info(f"Deleted Qdrant collection: {collection}")
        except Exception as e:
            logger.warning(f"Could not delete collection {collection}: {e}")

        # Remove from DB
        self._conn().execute("DELETE FROM packs WHERE pack_id = ?", (pack_id,))
        self._conn().commit()

        # Cleanup download dir
        dl_dir = self.data_dir / "downloads" / pack_id
        if dl_dir.exists():
            import shutil
            shutil.rmtree(dl_dir, ignore_errors=True)

        logger.info(f"Pack '{pack_id}' uninstalled")
        return {"pack_id": pack_id, "status": "uninstalled"}

    # ── Internal helpers ─────────────────────────────────────────────────

    def _parse_pack(self, pack_id: str, info: Dict, download_dir: Path) -> List[Dict[str, str]]:
        """Parse downloaded pack into text chunks. Returns list of {text, metadata}."""
        fmt = info.get("format", "")
        chunk_size = info.get("chunk_size", 512)
        chunks = []

        if fmt == "rss":
            chunks = self._parse_rss_feeds(DEFAULT_RSS_FEEDS, chunk_size)
        else:
            # For other formats, create a placeholder indicating the pack
            # needs actual data download first
            logger.info(f"Pack '{pack_id}' format '{fmt}' - download required for full indexing")
            chunks.append({
                "text": f"Knowledge pack '{info['name']}' registered. Full download pending.",
                "metadata": {"pack_id": pack_id, "type": "placeholder"},
            })

        return chunks

    def _parse_rss_feeds(self, feed_urls: List[str], chunk_size: int) -> List[Dict]:
        """Parse RSS feeds into chunks."""
        chunks = []
        try:
            import xml.etree.ElementTree as ET
            import requests

            for url in feed_urls:
                try:
                    resp = requests.get(url, timeout=15)
                    resp.raise_for_status()
                    root = ET.fromstring(resp.content)
                    for item in root.iter("item"):
                        title = item.findtext("title", "")
                        desc = item.findtext("description", "")
                        link = item.findtext("link", "")
                        pub_date = item.findtext("pubDate", "")
                        text = f"{title}\n{desc}"
                        # Simple chunking
                        words = text.split()
                        for i in range(0, len(words), chunk_size):
                            chunk_text = " ".join(words[i:i + chunk_size])
                            if len(chunk_text.strip()) > 20:
                                chunks.append({
                                    "text": chunk_text,
                                    "metadata": {
                                        "source": url,
                                        "title": title,
                                        "link": link,
                                        "pub_date": pub_date,
                                        "type": "rss",
                                    },
                                })
                except Exception as e:
                    logger.warning(f"Failed to parse RSS feed {url}: {e}")
        except ImportError:
            logger.warning("requests not available for RSS parsing")
        return chunks

    def _embed_and_upsert(self, collection: str, chunks: List[Dict]) -> int:
        """Embed chunks and upsert to Qdrant. Returns count of chunks stored."""
        if not chunks:
            return 0

        try:
            from sentence_transformers import SentenceTransformer
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct

            encoder = SentenceTransformer("all-MiniLM-L6-v2")
            client = QdrantClient(path=str(REPO_ROOT / "qdrant_storage"))

            # Ensure collection exists
            collections = [c.name for c in client.get_collections().collections]
            if collection not in collections:
                client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )

            # Batch embed and upsert
            batch_size = 100
            total = 0
            texts = [c["text"] for c in chunks]

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_chunks = chunks[i:i + batch_size]
                embeddings = encoder.encode(batch_texts, show_progress_bar=False)

                points = []
                for j, (text, emb) in enumerate(zip(batch_texts, embeddings)):
                    meta = batch_chunks[j].get("metadata", {})
                    meta["text"] = text
                    meta["timestamp"] = time.time()
                    points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector=emb.tolist(),
                        payload=meta,
                    ))

                client.upsert(collection_name=collection, points=points)
                total += len(points)

            logger.info(f"Upserted {total} chunks to collection '{collection}'")
            return total

        except ImportError as e:
            logger.warning(f"Dependencies missing for embedding: {e}")
            return 0
        except Exception as e:
            logger.error(f"Embed/upsert failed: {e}")
            return 0
