"""
Developer Knowledge Base for EDISON
Ingests code, docs, and specs for accurate coding assistance.
Chunks code by function/class boundaries where possible.
Extracts symbols (classes, functions, imports) into metadata.
"""

import ast
import hashlib
import json
import logging
import os
import re
import sqlite3
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
DATA_DIR = REPO_ROOT / "data" / "dev_kb"
DB_PATH = DATA_DIR / "dev_kb.db"

# Collections for different KB domains
COLLECTIONS = {
    "python": "kb_dev_python",
    "js_ts": "kb_dev_js_ts",
    "frameworks": "kb_dev_frameworks",
    "code_examples": "kb_dev_code_examples",
}

# File extensions to index
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go",
    ".java", ".cpp", ".c", ".h", ".hpp", ".cs", ".rb",
    ".yaml", ".yml", ".toml", ".json", ".md", ".rst", ".txt",
}


class DevKBManager:
    """Manage developer knowledge base: ingest repos, docs, and code examples."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
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

    @property
    def db_path(self) -> Path:
        return self.data_dir / "dev_kb.db"

    def _init_db(self):
        self._conn().executescript("""
            CREATE TABLE IF NOT EXISTS repos (
                name            TEXT PRIMARY KEY,
                path            TEXT NOT NULL,
                git_url         TEXT,
                collection      TEXT NOT NULL,
                commit_hash     TEXT,
                indexed_at      REAL,
                chunk_count     INTEGER DEFAULT 0,
                status          TEXT DEFAULT 'pending'
            );
            CREATE TABLE IF NOT EXISTS doc_packs (
                pack_id         TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                description     TEXT,
                collection      TEXT NOT NULL,
                version         TEXT,
                installed_at    REAL,
                chunk_count     INTEGER DEFAULT 0,
                status          TEXT DEFAULT 'available'
            );
        """)
        self._conn().commit()
        logger.info("Developer KB initialized")

    # ── Repo Management ──────────────────────────────────────────────────

    def add_repo(self, name: str, path_or_url: str, collection: str = "code_examples") -> Dict[str, Any]:
        """Add a local or git repo for indexing."""
        coll_name = COLLECTIONS.get(collection, f"kb_dev_{collection}")
        local_path = Path(path_or_url)
        git_url = None

        if not local_path.exists():
            # Assume it's a git URL - clone it
            git_url = path_or_url
            clone_dir = self.data_dir / "repos" / name
            clone_dir.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run(
                    ["git", "clone", "--depth=1", git_url, str(clone_dir)],
                    check=True, capture_output=True, timeout=120,
                )
                local_path = clone_dir
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                return {"name": name, "status": "error", "error": f"Clone failed: {e}"}

        # Get commit hash
        commit = self._get_commit_hash(local_path)

        self._conn().execute(
            """INSERT OR REPLACE INTO repos
               (name, path, git_url, collection, commit_hash, status)
               VALUES (?, ?, ?, ?, ?, 'pending')""",
            (name, str(local_path), git_url, coll_name, commit),
        )
        self._conn().commit()

        # Index the repo
        return self._index_repo(name, local_path, coll_name)

    def update_repo(self, name: str) -> Dict[str, Any]:
        """Pull latest and re-index a repo."""
        row = self._conn().execute("SELECT * FROM repos WHERE name = ?", (name,)).fetchone()
        if not row:
            return {"name": name, "status": "not_found"}

        path = Path(row["path"])
        if row["git_url"] and path.exists():
            try:
                subprocess.run(
                    ["git", "-C", str(path), "pull", "--ff-only"],
                    check=True, capture_output=True, timeout=60,
                )
            except Exception as e:
                logger.warning(f"Git pull failed for {name}: {e}")

        return self._index_repo(name, path, row["collection"])

    def status(self, name: Optional[str] = None) -> Dict[str, Any]:
        if name:
            row = self._conn().execute("SELECT * FROM repos WHERE name = ?", (name,)).fetchone()
            if row:
                return dict(row)
            return {"name": name, "status": "not_found"}
        repos = [dict(r) for r in self._conn().execute("SELECT * FROM repos").fetchall()]
        packs = [dict(r) for r in self._conn().execute("SELECT * FROM doc_packs").fetchall()]
        return {"repos": repos, "packs": packs}

    def uninstall(self, name: str) -> Dict[str, Any]:
        """Remove a repo or doc pack."""
        row = self._conn().execute("SELECT * FROM repos WHERE name = ?", (name,)).fetchone()
        if row:
            self._delete_collection(row["collection"])
            self._conn().execute("DELETE FROM repos WHERE name = ?", (name,))
            self._conn().commit()
            return {"name": name, "status": "uninstalled"}
        return {"name": name, "status": "not_found"}

    # ── Indexing ─────────────────────────────────────────────────────────

    def _index_repo(self, name: str, path: Path, collection: str) -> Dict[str, Any]:
        """Index a repository: walk files, extract symbols, chunk, embed, upsert."""
        chunks = []
        for fpath in self._walk_code_files(path):
            try:
                content = fpath.read_text(errors="replace")
                rel_path = str(fpath.relative_to(path))

                if fpath.suffix == ".py":
                    file_chunks = self._chunk_python(content, rel_path)
                else:
                    file_chunks = self._chunk_text(content, rel_path, fpath.suffix)

                chunks.extend(file_chunks)
            except Exception as e:
                logger.debug(f"Skipping {fpath}: {e}")

        # Embed and upsert
        count = self._embed_and_upsert(collection, chunks)

        self._conn().execute(
            """UPDATE repos SET status = 'indexed', chunk_count = ?,
               indexed_at = ?, commit_hash = ?
               WHERE name = ?""",
            (count, time.time(), self._get_commit_hash(path), name),
        )
        self._conn().commit()

        logger.info(f"Indexed repo '{name}': {count} chunks")
        return {"name": name, "status": "indexed", "chunk_count": count}

    def _chunk_python(self, source: str, filepath: str) -> List[Dict]:
        """Chunk Python source by function/class boundaries + extract symbols."""
        chunks = []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return self._chunk_text(source, filepath, ".py")

        symbols = {"classes": [], "functions": [], "imports": []}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                symbols["classes"].append(node.name)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                symbols["functions"].append(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    symbols["imports"].append(alias.name)

        lines = source.split("\n")

        # Extract top-level classes and functions as individual chunks
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.lineno - 1
                end = getattr(node, "end_lineno", start + 20)
                chunk_text = "\n".join(lines[start:end])
                if len(chunk_text.strip()) > 30:
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "filepath": filepath,
                            "type": "code",
                            "language": "python",
                            "symbol": node.name,
                            "symbol_type": type(node).__name__,
                            "line_start": start + 1,
                            "line_end": end,
                            "symbols": symbols,
                        },
                    })

        # Also chunk remaining module-level code
        if not chunks:
            chunks = self._chunk_text(source, filepath, ".py")

        return chunks

    def _chunk_text(self, text: str, filepath: str, ext: str, chunk_size: int = 512) -> List[Dict]:
        """Simple text chunking by word count."""
        words = text.split()
        chunks = []
        lang = {"py": "python", "js": "javascript", "ts": "typescript"}.get(ext.lstrip("."), "text")

        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i + chunk_size])
            if len(chunk_text.strip()) > 30:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "filepath": filepath,
                        "type": "code" if ext in {".py", ".js", ".ts"} else "doc",
                        "language": lang,
                    },
                })
        return chunks

    def _walk_code_files(self, root: Path, max_files: int = 5000) -> List[Path]:
        """Walk a directory for indexable code files, skipping hidden/venv/node_modules."""
        skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", "dist", "build"}
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fn in filenames:
                if len(files) >= max_files:
                    return files
                p = Path(dirpath) / fn
                if p.suffix in CODE_EXTENSIONS and p.stat().st_size < 500_000:
                    files.append(p)
        return files

    def _get_commit_hash(self, path: Path) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def _embed_and_upsert(self, collection: str, chunks: List[Dict]) -> int:
        if not chunks:
            return 0
        try:
            from sentence_transformers import SentenceTransformer
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct

            encoder = SentenceTransformer("all-MiniLM-L6-v2")
            client = QdrantClient(path=str(REPO_ROOT / "qdrant_storage"))

            collections = [c.name for c in client.get_collections().collections]
            if collection not in collections:
                client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )

            texts = [c["text"] for c in chunks]
            batch_size = 100
            total = 0

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_meta = chunks[i:i + batch_size]
                embeddings = encoder.encode(batch, show_progress_bar=False)

                points = [
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=emb.tolist(),
                        payload={**m.get("metadata", {}), "text": t, "timestamp": time.time()},
                    )
                    for t, emb, m in zip(batch, embeddings, batch_meta)
                ]
                client.upsert(collection_name=collection, points=points)
                total += len(points)

            return total
        except ImportError as e:
            logger.warning(f"Dependencies missing for embedding: {e}")
            return 0
        except Exception as e:
            logger.error(f"Embed/upsert failed: {e}")
            return 0

    def _delete_collection(self, collection: str):
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(path=str(REPO_ROOT / "qdrant_storage"))
            client.delete_collection(collection)
        except Exception as e:
            logger.warning(f"Could not delete collection {collection}: {e}")
