#!/usr/bin/env python3
"""
Index extracted Wikipedia JSONL files into Qdrant for RAG retrieval.

This script reads the JSONL files produced by extract_wikipedia.py,
chunks them, generates embeddings with all-MiniLM-L6-v2, and upserts
into a dedicated 'wikipedia' Qdrant collection.

Usage:
    python3 scripts/index_wikipedia.py [--wiki-dir DIR] [--qdrant-path DIR]
                                       [--batch-size N] [--chunk-size N]
                                       [--max-articles N] [--resume]

Defaults read from config/edison.yaml (rag.wikipedia_path, rag.storage_path).
"""
import argparse
import json
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Defaults ───────────────────────────────────────────────────────────────

DEFAULT_WIKI_DIR = "/opt/data/wikipedia/text"
DEFAULT_QDRANT_PATH = "/opt/data/rag"
COLLECTION_NAME = "wikipedia"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2
CHUNK_SIZE = 500   # ~500 chars per chunk (a decent paragraph)
CHUNK_OVERLAP = 50  # overlap between chunks for context continuity
EMBED_BATCH_SIZE = 256  # articles per encoder.encode() call
UPSERT_BATCH_SIZE = 512  # points per Qdrant upsert call
PROGRESS_FILE = "wiki_index_progress.json"


def load_config():
    """Try to read paths from edison.yaml."""
    config_path = REPO_ROOT / "config" / "edison.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            rag = cfg.get("edison", {}).get("rag", {})
            return rag
        except Exception:
            pass
    return {}


def chunk_text(text: str, title: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split article text into overlapping chunks with metadata."""
    if not text or len(text) < 50:
        return []

    chunks = []
    # Split on paragraph boundaries first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            # If a single paragraph is too long, split further
            if len(para) > chunk_size:
                words = para.split()
                sub = ""
                for w in words:
                    if len(sub) + len(w) + 1 > chunk_size:
                        if sub:
                            chunks.append(sub)
                        sub = w
                    else:
                        sub = (sub + " " + w).strip() if sub else w
                current = sub
            else:
                current = para

    if current:
        chunks.append(current)

    # Build output dicts with title prefix for embedding quality
    results = []
    for i, chunk in enumerate(chunks):
        # Prepend title so the embedding captures what article this is about
        embed_text = f"{title}: {chunk}" if title else chunk
        results.append({
            "text": embed_text,
            "title": title,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "type": "wikipedia",
            "source": "wikipedia",
        })
    return results


def iter_jsonl_files(wiki_dir: str):
    """Yield (filepath, line_number, article_dict) for all JSONL files."""
    wiki_path = Path(wiki_dir)
    files = sorted(wiki_path.rglob("*.jsonl"))
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    article = json.loads(line)
                    yield str(fpath.relative_to(wiki_path)), line_no, article
                except json.JSONDecodeError:
                    continue


def save_progress(progress_path: Path, data: dict):
    with open(progress_path, "w") as f:
        json.dump(data, f)


def load_progress(progress_path: Path) -> dict:
    if progress_path.exists():
        try:
            with open(progress_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"articles_indexed": 0, "chunks_indexed": 0, "last_file": "", "last_line": -1}


def main():
    parser = argparse.ArgumentParser(description="Index Wikipedia into Qdrant")
    rag_cfg = load_config()

    parser.add_argument("--wiki-dir", default=rag_cfg.get("wikipedia_path", DEFAULT_WIKI_DIR),
                        help="Directory with extracted JSONL files")
    parser.add_argument("--qdrant-path", default=rag_cfg.get("storage_path", DEFAULT_QDRANT_PATH),
                        help="Qdrant local storage directory")
    parser.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE,
                        help="Articles per embedding batch")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help="Max characters per chunk")
    parser.add_argument("--max-articles", type=int, default=0,
                        help="Max articles to index (0 = all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    wiki_dir = args.wiki_dir
    qdrant_path = args.qdrant_path

    if not Path(wiki_dir).exists():
        logger.error(f"Wikipedia directory not found: {wiki_dir}")
        sys.exit(1)

    progress_path = Path(qdrant_path) / PROGRESS_FILE

    # ── Load encoder ────────────────────────────────────────────────────────
    logger.info("Loading sentence-transformers model: all-MiniLM-L6-v2 ...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Encoder loaded.")

    # ── Init Qdrant ─────────────────────────────────────────────────────────
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    os.makedirs(qdrant_path, exist_ok=True)
    client = QdrantClient(path=qdrant_path)

    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")
    else:
        info = client.get_collection(COLLECTION_NAME)
        logger.info(f"Using existing collection: {COLLECTION_NAME} ({info.points_count:,} points)")

    # ── Resume support ──────────────────────────────────────────────────────
    progress = load_progress(progress_path) if args.resume else {
        "articles_indexed": 0, "chunks_indexed": 0, "last_file": "", "last_line": -1
    }
    skip_until_file = progress["last_file"]
    skip_until_line = progress["last_line"]
    skipping = bool(skip_until_file)
    articles_indexed = progress["articles_indexed"]
    chunks_indexed = progress["chunks_indexed"]

    if skipping:
        logger.info(f"Resuming from {skip_until_file} line {skip_until_line} "
                     f"({articles_indexed:,} articles, {chunks_indexed:,} chunks already done)")

    # ── Process articles ────────────────────────────────────────────────────
    chunk_buffer: list[dict] = []  # accumulated chunks waiting to be embedded+upserted
    start_time = time.time()
    total_articles = 0
    max_articles = args.max_articles

    logger.info(f"Indexing from: {wiki_dir}")
    logger.info(f"Qdrant storage: {qdrant_path}")
    logger.info(f"Chunk size: {args.chunk_size} chars")
    if max_articles:
        logger.info(f"Max articles: {max_articles:,}")
    logger.info("")

    for rel_path, line_no, article in iter_jsonl_files(wiki_dir):
        # Resume: skip already-processed files/lines
        if skipping:
            if rel_path == skip_until_file and line_no >= skip_until_line:
                skipping = False
            else:
                if rel_path < skip_until_file:
                    continue
                elif rel_path == skip_until_file and line_no < skip_until_line:
                    continue
                else:
                    skipping = False

        title = article.get("title", "")
        text = article.get("text", "")

        if not text or len(text) < 100:
            continue

        # Chunk the article
        article_chunks = chunk_text(text, title, chunk_size=args.chunk_size)
        chunk_buffer.extend(article_chunks)
        total_articles += 1
        articles_indexed += 1

        # When buffer is large enough, embed and upsert
        if len(chunk_buffer) >= UPSERT_BATCH_SIZE:
            _embed_and_upsert(client, encoder, chunk_buffer)
            chunks_indexed += len(chunk_buffer)
            chunk_buffer = []

            # Save checkpoint
            save_progress(progress_path, {
                "articles_indexed": articles_indexed,
                "chunks_indexed": chunks_indexed,
                "last_file": rel_path,
                "last_line": line_no,
            })

            # Progress report
            if articles_indexed % 10000 == 0:
                elapsed = time.time() - start_time
                rate = total_articles / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  {articles_indexed:>10,} articles  |  "
                    f"{chunks_indexed:>12,} chunks  |  "
                    f"{elapsed/60:.0f} min  |  "
                    f"{rate:.0f} art/sec"
                )

        if max_articles and total_articles >= max_articles:
            logger.info(f"Reached max_articles limit ({max_articles:,})")
            break

    # Flush remaining buffer
    if chunk_buffer:
        _embed_and_upsert(client, encoder, chunk_buffer)
        chunks_indexed += len(chunk_buffer)

    elapsed = time.time() - start_time

    # Final save
    save_progress(progress_path, {
        "articles_indexed": articles_indexed,
        "chunks_indexed": chunks_indexed,
        "last_file": "DONE",
        "last_line": -1,
    })

    # Final stats
    info = client.get_collection(COLLECTION_NAME)
    logger.info("")
    logger.info("Done!")
    logger.info(f"  Articles indexed: {articles_indexed:,}")
    logger.info(f"  Chunks indexed:   {chunks_indexed:,}")
    logger.info(f"  Qdrant points:    {info.points_count:,}")
    logger.info(f"  Time:             {elapsed/3600:.1f} hours")
    logger.info(f"  Storage:          {qdrant_path}")


def _embed_and_upsert(client, encoder, chunk_buffer: list[dict]):
    """Embed a batch of chunks and upsert into Qdrant."""
    from qdrant_client.models import PointStruct

    texts = [c["text"] for c in chunk_buffer]
    vectors = encoder.encode(texts, show_progress_bar=False, batch_size=128)

    points = []
    for chunk_meta, vec in zip(chunk_buffer, vectors):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vec.tolist(),
            payload={
                "text": chunk_meta["text"],
                "title": chunk_meta["title"],
                "chunk_index": chunk_meta["chunk_index"],
                "total_chunks": chunk_meta["total_chunks"],
                "type": "wikipedia",
                "source": "wikipedia",
                "timestamp": 0,  # Not time-sensitive
            },
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)


if __name__ == "__main__":
    main()
