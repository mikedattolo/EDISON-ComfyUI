# Knowledge Packs

Knowledge packs let you download and ingest large external datasets into
EDISON's vector store so it can answer questions about topics beyond its
training data.

## Built-in Packs

| Pack | Description | Size |
|------|-------------|------|
| `wikipedia-simple` | Simple English Wikipedia (~200k articles) | ~700MB |
| `arxiv-abstracts` | ArXiv paper abstracts (CS, ML, AI) | ~500MB |
| `rss-feeds` | Live RSS feeds (Hacker News, ArXiv, TechCrunch, etc.) | Variable |

## CLI Usage

```bash
# List available packs
python -m services.knowledge_packs.cli list

# Install a pack
python -m services.knowledge_packs.cli install wikipedia-simple

# Check status
python -m services.knowledge_packs.cli status wikipedia-simple

# Update (re-ingest with latest data)
python -m services.knowledge_packs.cli update rss-feeds

# Uninstall
python -m services.knowledge_packs.cli uninstall wikipedia-simple
```

## API Endpoints

```
GET  /knowledge-packs              # List all packs + status
POST /knowledge-packs/{name}/install   # Install a pack (async)
```

## How It Works

1. **Download** — data is fetched from the source (Wikipedia dump, ArXiv API,
   RSS feeds)
2. **Chunk** — text is split into ~500-token chunks with overlap
3. **Embed** — each chunk is embedded using `all-MiniLM-L6-v2` (384-dim)
4. **Store** — embeddings are upserted into a Qdrant collection
   (`kb_wikipedia`, `kb_arxiv`, `kb_rss`)
5. **Index** — provenance metadata (source, version, chunk count) is recorded
   in SQLite (`data/knowledge_packs/packs.db`)

## Adding Custom Packs

Create a new entry in `KnowledgePackManager.PACKS` with:
- `name` — unique identifier
- `description` — human-readable description  
- `source_url` — download URL or API endpoint
- `collection` — Qdrant collection name
- `install_fn` — async function that downloads, chunks, and embeds

## RSS Feeds

The `rss-feeds` pack includes these default feeds:
- Hacker News (top stories)
- ArXiv CS.AI
- ArXiv CS.LG
- TechCrunch
- The Verge

Add custom feeds by appending to `DEFAULT_RSS_FEEDS` in
`services/knowledge_packs/manager.py`.

## Storage

- **Vectors**: Qdrant local storage at `./qdrant_storage/`
- **Metadata**: SQLite at `data/knowledge_packs/packs.db`
- **Embeddings model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
