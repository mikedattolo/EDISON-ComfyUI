# Developer Knowledge Base

The Developer KB gives EDISON code-aware retrieval for programming
assistance.  It indexes local repositories using AST-based chunking
(Python) and symbol extraction, enabling precise answers about codebases.

## Features

- **AST-based chunking** — Python files are split at function/class
  boundaries (not arbitrary line counts), preserving semantic coherence
- **Symbol extraction** — function names, class names, decorators, and
  docstrings are stored as metadata for targeted lookup
- **Multi-language support** — Python (AST), JavaScript/TypeScript, and
  generic text files (line-based fallback)
- **Git integration** — clone remote repos directly via URL

## CLI Usage

```bash
# Index a local repository
python -m services.edison_core.dev_kb.cli add /path/to/my-project

# Index a remote repository (cloned automatically)
python -m services.edison_core.dev_kb.cli add https://github.com/user/repo

# List indexed repositories
python -m services.edison_core.dev_kb.cli list

# Search indexed code
python -m services.edison_core.dev_kb.cli search "how does authentication work"
```

## API Endpoints

```
GET  /dev-kb/repos                 # List indexed repos
POST /dev-kb/index                 # Index a repo: {"path": "/path/to/repo"}
GET  /dev-kb/search?q=auth+flow    # Search across indexed code
```

## How It Works

### 1. Repository Scanning

When you add a repo, the Dev KB:
- Walks the file tree (respects `.gitignore` patterns)
- Filters by supported extensions (`.py`, `.js`, `.ts`, `.jsx`, `.tsx`)
- Skips `node_modules`, `__pycache__`, `.git`, `venv`, etc.

### 2. AST-Based Chunking (Python)

For `.py` files, the chunker uses Python's `ast` module to:
- Identify top-level functions and classes
- Extract each as a self-contained chunk with:
  - Source code
  - Function/class name
  - Decorators
  - Docstring
  - Line range (start, end)
- Module-level code between definitions is captured as separate chunks

### 3. Embedding & Storage

Each chunk is:
- Embedded using `all-MiniLM-L6-v2` (384 dimensions)
- Stored in a Qdrant collection based on language:
  - `kb_dev_python` — Python code
  - `kb_dev_js_ts` — JavaScript/TypeScript
  - `kb_dev_frameworks` — Framework-specific docs
  - `kb_dev_code_examples` — Tutorials and examples

### 4. Search

Queries are embedded and matched via cosine similarity against all indexed
chunks.  Results include the source code, file path, symbol name, and
relevance score.

## Qdrant Collections

| Collection | Contents |
|------------|----------|
| `kb_dev_python` | Python functions, classes, modules |
| `kb_dev_js_ts` | JavaScript and TypeScript code |
| `kb_dev_frameworks` | Framework documentation chunks |
| `kb_dev_code_examples` | Code examples and tutorials |

## Storage

- **Vectors**: Qdrant local storage at `./qdrant_storage/`
- **Metadata**: SQLite at `data/dev_kb/dev_kb.db`
