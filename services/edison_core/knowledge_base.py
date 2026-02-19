"""
Knowledge Base System for EDISON
Provides offline knowledge lookup using Wikipedia text dumps and other knowledge sources.
Stores knowledge in dedicated Qdrant collections with SQLite metadata index.

Inspired by how ChatGPT/Claude handle knowledge:
- Pre-indexed knowledge base for instant factual lookup
- Semantic search over large knowledge corpora
- Chunked documents with overlap for better retrieval
- Hierarchical retrieval: fast keyword → semantic reranking
"""

import logging
import sqlite3
import json
import os
import re
import time
import threading
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeChunk:
    """A chunk of knowledge with metadata"""
    text: str
    source: str  # e.g. "wikipedia", "web_search", "user_upload"
    title: str = ""
    url: str = ""
    category: str = ""
    chunk_index: int = 0
    total_chunks: int = 1
    timestamp: float = 0.0
    confidence: float = 1.0


class KnowledgeBase:
    """
    Offline knowledge base using Qdrant for vector search + SQLite for metadata.
    
    Architecture (similar to ChatGPT's retrieval plugin):
    - Qdrant collection "edison_knowledge" for semantic vector search
    - SQLite DB for fast metadata queries, dedup tracking, source management
    - Chunking with overlap for better retrieval of long articles
    - Separate from conversation memory to avoid noise
    """

    COLLECTION_NAME = "edison_knowledge"
    SEARCH_CACHE_COLLECTION = "edison_search_cache"
    VECTOR_SIZE = 384  # all-MiniLM-L6-v2

    def __init__(self, storage_path: str = "./knowledge_storage", qdrant_path: str = None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.qdrant_path = qdrant_path or str(self.storage_path / "qdrant")
        self.db_path = str(self.storage_path / "knowledge_index.db")

        self.client = None
        self.encoder = None
        self._lock = threading.Lock()

        self._init_sqlite()
        self._init_encoder()
        self._init_qdrant()

    def _init_sqlite(self):
        """Initialize SQLite metadata index"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_sources (
                    id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    title TEXT,
                    url TEXT,
                    category TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    content_hash TEXT,
                    created_at REAL,
                    updated_at REAL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_stats (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    source TEXT NOT NULL,
                    result_count INTEGER,
                    results_stored BOOLEAN DEFAULT 0,
                    timestamp REAL,
                    quality_score REAL DEFAULT 0.0
                )
            """)
            # Index for fast lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sources_type ON knowledge_sources(source_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sources_hash ON knowledge_sources(content_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_search_query ON search_history(query)")
            conn.commit()
            conn.close()
            logger.info(f"✓ Knowledge SQLite DB initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite: {e}")

    def _init_encoder(self):
        """Initialize sentence transformer (shared with RAG)"""
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✓ Knowledge base encoder loaded")
        except Exception as e:
            logger.error(f"Failed to load encoder for knowledge base: {e}")
            self.encoder = None

    def _init_qdrant(self):
        """Initialize Qdrant collections for knowledge"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self.client = QdrantClient(path=self.qdrant_path)

            for coll_name in [self.COLLECTION_NAME, self.SEARCH_CACHE_COLLECTION]:
                collections = self.client.get_collections().collections
                if coll_name not in [c.name for c in collections]:
                    self.client.create_collection(
                        collection_name=coll_name,
                        vectors_config=VectorParams(
                            size=self.VECTOR_SIZE,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"Created Qdrant collection: {coll_name}")

            logger.info("✓ Knowledge base Qdrant initialized")
        except Exception as e:
            logger.error(f"Failed to initialize knowledge Qdrant: {e}")
            self.client = None

    def is_ready(self) -> bool:
        return self.client is not None and self.encoder is not None

    # ── Chunking ──────────────────────────────────────────────────────────

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval.
        Uses sentence-aware splitting to avoid mid-sentence breaks.
        
        Similar to LangChain's RecursiveCharacterTextSplitter approach
        used by ChatGPT retrieval plugin.
        """
        if not text or len(text) <= chunk_size:
            return [text] if text else []

        # Split on sentence boundaries first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence exceeds chunk_size, save current and start new
            if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep overlap from end of previous chunk
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = (current_chunk + " " + sentence).strip()

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    # ── Content hashing for dedup ─────────────────────────────────────────

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    def _is_duplicate(self, content_hash: str) -> bool:
        """Check if content already exists in knowledge base"""
        try:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute(
                "SELECT id FROM knowledge_sources WHERE content_hash = ?",
                (content_hash,)
            ).fetchone()
            conn.close()
            return row is not None
        except Exception:
            return False

    # ── Adding Knowledge ──────────────────────────────────────────────────

    def add_knowledge(
        self,
        text: str,
        source_type: str,
        title: str = "",
        url: str = "",
        category: str = "",
        chunk_size: int = 512,
        skip_dedup: bool = False
    ) -> int:
        """
        Add a piece of knowledge to the base. Chunks, embeds, and stores.
        
        Returns number of chunks stored.
        """
        if not self.is_ready():
            logger.warning("Knowledge base not ready, skipping add")
            return 0

        if not text or len(text.strip()) < 20:
            return 0

        content_hash = self._content_hash(text)
        if not skip_dedup and self._is_duplicate(content_hash):
            logger.debug(f"Duplicate content skipped: {title[:50]}")
            return 0

        try:
            import uuid
            from qdrant_client.models import PointStruct

            chunks = self.chunk_text(text, chunk_size=chunk_size)
            if not chunks:
                return 0

            source_id = str(uuid.uuid4())
            now = time.time()

            # Embed all chunks
            with self._lock:
                embeddings = self.encoder.encode(chunks, show_progress_bar=False)

            # Build points
            points = []
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                point_id = str(uuid.uuid4())
                payload = {
                    "text": chunk,
                    "source_id": source_id,
                    "source_type": source_type,
                    "title": title,
                    "url": url,
                    "category": category,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "timestamp": now,
                    "type": "knowledge"
                }
                points.append(PointStruct(id=point_id, vector=emb.tolist(), payload=payload))

            # Upsert to Qdrant
            collection = self.SEARCH_CACHE_COLLECTION if source_type == "web_search" else self.COLLECTION_NAME
            self.client.upsert(collection_name=collection, points=points)

            # Record in SQLite
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO knowledge_sources
                (id, source_type, title, url, category, chunk_count, content_hash, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (source_id, source_type, title, url, category, len(chunks), content_hash, now, now))
            conn.commit()
            conn.close()

            logger.info(f"Added {len(chunks)} knowledge chunks: [{source_type}] {title[:60]}")
            return len(chunks)

        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return 0

    def add_search_results(self, query: str, results: List[Dict[str, str]]) -> int:
        """
        Store web search results in the knowledge base for future retrieval.
        This is a key gap fix — ChatGPT and Claude both cache retrieved info.
        
        Args:
            query: The search query
            results: List of dicts with 'title', 'url', 'snippet'
        
        Returns:
            Number of results stored
        """
        if not self.is_ready():
            return 0

        stored = 0
        now = time.time()

        for result in results:
            title = result.get('title', '')
            url = result.get('url', '')
            snippet = result.get('snippet', '')

            if not snippet or len(snippet.strip()) < 20:
                continue

            # Build enriched text for better embedding
            enriched = f"{title}. {snippet}"
            if url:
                enriched += f" [Source: {url}]"

            count = self.add_knowledge(
                text=enriched,
                source_type="web_search",
                title=title,
                url=url,
                category="search_result",
                chunk_size=1024,  # Search snippets are already short
                skip_dedup=False
            )
            stored += count

        # Log search history
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO search_history (query, source, result_count, results_stored, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (query, "web_search", len(results), stored > 0, now))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to log search history: {e}")

        logger.info(f"Stored {stored} search result chunks for query: {query[:50]}")
        return stored

    # ── Querying Knowledge ────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        source_types: Optional[List[str]] = None,
        min_score: float = 0.25,
        include_search_cache: bool = True
    ) -> List[Tuple[str, Dict]]:
        """
        Query the knowledge base with semantic search.
        
        Similar to how ChatGPT's retrieval works:
        1. Encode query
        2. Search Qdrant for nearest neighbors
        3. Apply relevance threshold (min_score)
        4. Optionally filter by source type
        5. Return ranked results
        
        Returns list of (text, metadata) tuples sorted by relevance.
        """
        if not self.is_ready():
            return []

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

            with self._lock:
                query_vector = self.encoder.encode([query_text], show_progress_bar=False)[0]

            # Build optional filter
            query_filter = None
            if source_types:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="source_type",
                            match=MatchAny(any=source_types)
                        )
                    ]
                )

            all_results = []

            # Search main knowledge collection
            results = self.client.query_points(
                collection_name=self.COLLECTION_NAME,
                query=query_vector.tolist(),
                limit=n_results,
                query_filter=query_filter,
                with_payload=True
            )
            points = results.points if hasattr(results, 'points') else results
            for r in points:
                if r.score >= min_score:
                    all_results.append((r.payload.get("text", ""), {
                        **{k: v for k, v in r.payload.items() if k != "text"},
                        "score": r.score,
                        "collection": self.COLLECTION_NAME
                    }))

            # Also search search cache if requested
            if include_search_cache:
                try:
                    cache_results = self.client.query_points(
                        collection_name=self.SEARCH_CACHE_COLLECTION,
                        query=query_vector.tolist(),
                        limit=n_results,
                        with_payload=True
                    )
                    cache_points = cache_results.points if hasattr(cache_results, 'points') else cache_results
                    for r in cache_points:
                        if r.score >= min_score:
                            all_results.append((r.payload.get("text", ""), {
                                **{k: v for k, v in r.payload.items() if k != "text"},
                                "score": r.score,
                                "collection": self.SEARCH_CACHE_COLLECTION
                            }))
                except Exception:
                    pass  # Search cache might not exist yet

            # Sort by score descending
            all_results.sort(key=lambda x: x[1]["score"], reverse=True)

            # Update access counts
            self._update_access_counts([r[1].get("source_id") for r in all_results if r[1].get("source_id")])

            return all_results[:n_results]

        except Exception as e:
            logger.error(f"Knowledge query error: {e}")
            return []

    def _update_access_counts(self, source_ids: List[str]):
        """Track which knowledge gets accessed most (for maintenance)"""
        if not source_ids:
            return
        try:
            conn = sqlite3.connect(self.db_path)
            now = time.time()
            for sid in source_ids:
                if sid:
                    conn.execute(
                        "UPDATE knowledge_sources SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                        (now, sid)
                    )
            conn.commit()
            conn.close()
        except Exception:
            pass

    # ── Wikipedia Loading ─────────────────────────────────────────────────

    def load_wikipedia_dump(
        self,
        dump_path: str,
        max_articles: int = 0,
        min_article_length: int = 200,
        categories: Optional[List[str]] = None,
        batch_size: int = 100,
        progress_callback=None
    ) -> Dict:
        """
        Load Wikipedia text dump into the knowledge base.
        
        Supports multiple formats:
        - Wikipedia XML dump (parsed with mwxml or mwparserfromhell)
        - Pre-processed JSON lines (one article per line: {"title": ..., "text": ...})
        - Plain text files (one article per file in a directory)
        - Kiwix ZIM files (with zimply)
        - Simple Wikipedia dump from HuggingFace datasets
        
        Args:
            dump_path: Path to Wikipedia dump file or directory
            max_articles: Max articles to load (0 = all)
            min_article_length: Skip short articles
            categories: Only load articles in these categories
            batch_size: Number of articles to process before committing
            progress_callback: Optional callback(articles_loaded, total) for progress
            
        Returns:
            Dict with stats about loading
        """
        dump_path = Path(dump_path)
        stats = {"articles_loaded": 0, "chunks_stored": 0, "skipped": 0, "errors": 0}

        if not dump_path.exists():
            logger.error(f"Wikipedia dump not found: {dump_path}")
            return {**stats, "error": "File not found"}

        if dump_path.suffix == '.jsonl' or dump_path.suffix == '.json':
            stats = self._load_jsonl_dump(dump_path, max_articles, min_article_length, batch_size, progress_callback)
        elif dump_path.is_dir():
            stats = self._load_directory_dump(dump_path, max_articles, min_article_length, batch_size, progress_callback)
        elif dump_path.suffix == '.xml' or dump_path.suffix == '.bz2':
            stats = self._load_xml_dump(dump_path, max_articles, min_article_length, batch_size, progress_callback)
        elif dump_path.suffix == '.zim':
            stats = self._load_zim_dump(dump_path, max_articles, min_article_length, batch_size, progress_callback)
        elif dump_path.suffix == '.parquet':
            stats = self._load_parquet_dump(dump_path, max_articles, min_article_length, batch_size, progress_callback)
        else:
            logger.error(f"Unsupported dump format: {dump_path.suffix}")
            return {**stats, "error": f"Unsupported format: {dump_path.suffix}"}

        # Update stats
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO knowledge_stats (key, value, updated_at)
                VALUES (?, ?, ?)
            """, ("wikipedia_load", json.dumps(stats), time.time()))
            conn.commit()
            conn.close()
        except Exception:
            pass

        logger.info(f"Wikipedia loading complete: {stats}")
        return stats

    def _load_jsonl_dump(self, path, max_articles, min_length, batch_size, callback):
        """Load JSON Lines format (most common pre-processed format)"""
        stats = {"articles_loaded": 0, "chunks_stored": 0, "skipped": 0, "errors": 0}

        try:
            import gzip
            opener = gzip.open if str(path).endswith('.gz') else open

            with opener(path, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if max_articles > 0 and stats["articles_loaded"] >= max_articles:
                        break

                    try:
                        article = json.loads(line.strip())
                        title = article.get('title', f'article_{line_num}')
                        text = article.get('text', '')

                        if len(text) < min_length:
                            stats["skipped"] += 1
                            continue

                        chunks = self.add_knowledge(
                            text=text,
                            source_type="wikipedia",
                            title=title,
                            url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                            category=article.get('category', 'general')
                        )
                        stats["chunks_stored"] += chunks
                        stats["articles_loaded"] += 1

                        if callback and stats["articles_loaded"] % batch_size == 0:
                            callback(stats["articles_loaded"], max_articles or -1)

                    except json.JSONDecodeError:
                        stats["errors"] += 1
                    except Exception as e:
                        stats["errors"] += 1
                        if stats["errors"] <= 5:
                            logger.warning(f"Error loading article: {e}")

        except Exception as e:
            logger.error(f"Error reading JSONL dump: {e}")
            stats["error"] = str(e)

        return stats

    def _load_directory_dump(self, path, max_articles, min_length, batch_size, callback):
        """Load from directory of text files"""
        stats = {"articles_loaded": 0, "chunks_stored": 0, "skipped": 0, "errors": 0}

        text_files = sorted(path.glob("**/*.txt")) + sorted(path.glob("**/*.md"))

        for filepath in text_files:
            if max_articles > 0 and stats["articles_loaded"] >= max_articles:
                break

            try:
                text = filepath.read_text(encoding='utf-8', errors='replace')
                if len(text) < min_length:
                    stats["skipped"] += 1
                    continue

                title = filepath.stem.replace('_', ' ').replace('-', ' ')
                chunks = self.add_knowledge(
                    text=text,
                    source_type="wikipedia",
                    title=title,
                    category="general"
                )
                stats["chunks_stored"] += chunks
                stats["articles_loaded"] += 1

                if callback and stats["articles_loaded"] % batch_size == 0:
                    callback(stats["articles_loaded"], max_articles or len(text_files))

            except Exception as e:
                stats["errors"] += 1
                if stats["errors"] <= 5:
                    logger.warning(f"Error loading {filepath.name}: {e}")

        return stats

    def _load_xml_dump(self, path, max_articles, min_length, batch_size, callback):
        """Load from Wikipedia XML dump (optionally bz2 compressed)"""
        stats = {"articles_loaded": 0, "chunks_stored": 0, "skipped": 0, "errors": 0}

        try:
            # Try using mwxml for efficient Wikipedia XML parsing
            import mwxml
            import bz2

            if str(path).endswith('.bz2'):
                f = bz2.open(path, 'rt', encoding='utf-8')
            else:
                f = open(path, 'rt', encoding='utf-8')

            dump = mwxml.Dump.from_file(f)

            for page in dump:
                if max_articles > 0 and stats["articles_loaded"] >= max_articles:
                    break

                # Skip non-article namespaces
                if page.namespace != 0:
                    continue

                title = page.title
                # Get latest revision
                for revision in page:
                    text = revision.text or ""
                    break  # Only latest revision

                # Strip wiki markup (basic)
                text = self._strip_wiki_markup(text)

                if len(text) < min_length:
                    stats["skipped"] += 1
                    continue

                chunks = self.add_knowledge(
                    text=text,
                    source_type="wikipedia",
                    title=title,
                    url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    category="general"
                )
                stats["chunks_stored"] += chunks
                stats["articles_loaded"] += 1

                if callback and stats["articles_loaded"] % batch_size == 0:
                    callback(stats["articles_loaded"], max_articles or -1)

            f.close()

        except ImportError:
            logger.error("mwxml not installed. Install with: pip install mwxml")
            stats["error"] = "mwxml package required for XML dumps"
        except Exception as e:
            logger.error(f"Error reading XML dump: {e}")
            stats["error"] = str(e)

        return stats

    def _load_zim_dump(self, path, max_articles, min_length, batch_size, callback):
        """Load from Kiwix ZIM file (offline Wikipedia)"""
        stats = {"articles_loaded": 0, "chunks_stored": 0, "skipped": 0, "errors": 0}

        try:
            from libzim.reader import Archive
            from bs4 import BeautifulSoup

            zim = Archive(str(path))
            entry_count = zim.entry_count

            for i in range(entry_count):
                if max_articles > 0 and stats["articles_loaded"] >= max_articles:
                    break

                try:
                    entry = zim._get_entry_by_id(i)
                    if not entry.is_redirect:
                        item = entry.get_item()
                        mimetype = item.mimetype

                        if mimetype == "text/html":
                            html = bytes(item.content).decode('utf-8', errors='replace')
                            soup = BeautifulSoup(html, 'html.parser')

                            # Remove scripts, styles, navigation
                            for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                                tag.decompose()

                            text = soup.get_text(separator='\n', strip=True)
                            title = entry.title or entry.path

                            if len(text) < min_length:
                                stats["skipped"] += 1
                                continue

                            chunks = self.add_knowledge(
                                text=text,
                                source_type="wikipedia",
                                title=title,
                                category="general"
                            )
                            stats["chunks_stored"] += chunks
                            stats["articles_loaded"] += 1

                            if callback and stats["articles_loaded"] % batch_size == 0:
                                callback(stats["articles_loaded"], max_articles or entry_count)

                except Exception:
                    stats["errors"] += 1

        except ImportError:
            logger.error("libzim not installed. Install with: pip install libzim")
            stats["error"] = "libzim package required for ZIM files"
        except Exception as e:
            logger.error(f"Error reading ZIM file: {e}")
            stats["error"] = str(e)

        return stats

    def _load_parquet_dump(self, path, max_articles, min_length, batch_size, callback):
        """Load from Parquet file (e.g., HuggingFace datasets)"""
        stats = {"articles_loaded": 0, "chunks_stored": 0, "skipped": 0, "errors": 0}

        try:
            import pyarrow.parquet as pq

            table = pq.read_table(str(path))
            df = table.to_pandas()

            # Try common column names
            text_col = None
            title_col = None
            for col in ['text', 'content', 'article', 'body']:
                if col in df.columns:
                    text_col = col
                    break
            for col in ['title', 'name', 'heading']:
                if col in df.columns:
                    title_col = col
                    break

            if not text_col:
                stats["error"] = f"No text column found. Columns: {list(df.columns)}"
                return stats

            for idx, row in df.iterrows():
                if max_articles > 0 and stats["articles_loaded"] >= max_articles:
                    break

                text = str(row[text_col])
                title = str(row[title_col]) if title_col else f"article_{idx}"

                if len(text) < min_length:
                    stats["skipped"] += 1
                    continue

                chunks = self.add_knowledge(
                    text=text,
                    source_type="wikipedia",
                    title=title,
                    url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    category="general"
                )
                stats["chunks_stored"] += chunks
                stats["articles_loaded"] += 1

                if callback and stats["articles_loaded"] % batch_size == 0:
                    callback(stats["articles_loaded"], max_articles or len(df))

        except ImportError:
            logger.error("pyarrow not installed. Install with: pip install pyarrow")
            stats["error"] = "pyarrow package required for Parquet files"
        except Exception as e:
            logger.error(f"Error reading Parquet file: {e}")
            stats["error"] = str(e)

        return stats

    @staticmethod
    def _strip_wiki_markup(text: str) -> str:
        """Basic wiki markup stripping"""
        if not text:
            return ""
        # Remove templates {{...}}
        text = re.sub(r'\{\{[^}]*\}\}', '', text)
        # Remove [[File:...]] and [[Image:...]]
        text = re.sub(r'\[\[(File|Image):[^\]]*\]\]', '', text, flags=re.IGNORECASE)
        # Convert [[link|text]] to text, [[link]] to link
        text = re.sub(r'\[\[([^|\]]*\|)?([^\]]*)\]\]', r'\2', text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove ref tags and content
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(r'<ref[^/]*/>', '', text)
        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    # ── Stats & Maintenance ───────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        stats = {"status": "not_ready"}

        if not self.is_ready():
            return stats

        try:
            # Qdrant stats
            knowledge_info = self.client.get_collection(self.COLLECTION_NAME)
            stats["knowledge_points"] = knowledge_info.points_count

            try:
                cache_info = self.client.get_collection(self.SEARCH_CACHE_COLLECTION)
                stats["search_cache_points"] = cache_info.points_count
            except Exception:
                stats["search_cache_points"] = 0

            # SQLite stats
            conn = sqlite3.connect(self.db_path)
            row = conn.execute("SELECT COUNT(*) FROM knowledge_sources").fetchone()
            stats["total_sources"] = row[0] if row else 0

            row = conn.execute("SELECT COUNT(*) FROM search_history").fetchone()
            stats["total_searches"] = row[0] if row else 0

            # Source type breakdown
            rows = conn.execute(
                "SELECT source_type, COUNT(*), SUM(chunk_count) FROM knowledge_sources GROUP BY source_type"
            ).fetchall()
            stats["by_source"] = {r[0]: {"sources": r[1], "chunks": r[2]} for r in rows}

            conn.close()
            stats["status"] = "ready"

        except Exception as e:
            stats["error"] = str(e)

        return stats

    def cleanup_old_search_cache(self, max_age_days: int = 30):
        """Remove old search cache entries to prevent bloat"""
        if not self.is_ready():
            return

        try:
            from qdrant_client.models import Filter, FieldCondition, Range

            cutoff = time.time() - (max_age_days * 86400)

            self.client.delete(
                collection_name=self.SEARCH_CACHE_COLLECTION,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="timestamp",
                            range=Range(lt=cutoff)
                        )
                    ]
                )
            )
            logger.info(f"Cleaned up search cache entries older than {max_age_days} days")
        except Exception as e:
            logger.warning(f"Search cache cleanup failed: {e}")
