# New Subsystems Overview

This document summarizes all new subsystems added in the latest overhaul.

## 1. Memory Intelligence (`services/edison_core/memory/`)

Three-tier memory system that learns from conversations:

- **Profile** — long-term user preferences (name, language, style)
- **Episodic** — session-specific context (what was discussed today)
- **Semantic** — extracted facts and knowledge

Features: CRUD API, search by type/key/tag, pin important memories,
automatic hygiene (dedup, merge, prune stale entries).

**API:** `GET/POST/PUT/DELETE /memory/*`

## 2. Retrieval Reranking (`services/edison_core/retrieval.py`)

Two-stage retrieval pipeline with intent-aware scoring:

- Detects query intent (profile, episodic, semantic, dev_docs, general)
- Multi-factor reranking: 50% relevance + 20% recency + 15% confidence + 15% type weight
- Intent-based boosting (e.g., "what's my name?" boosts profile memories)

## 3. Freshness Cache (`services/edison_core/freshness.py`)

Prevents stale answers for time-sensitive queries:

- Detects time-sensitive queries ("latest", "today", "current price")
- TTL management: 300s for time-sensitive, 3600s general, 86400s stale cutoff
- Citation formatting for sourced answers

**API:** `GET /freshness/check?query=...`

## 4. Tool Execution Framework (`services/edison_core/tool_framework.py`)

Clean tool architecture with safety controls:

- `BaseTool` ABC with `name`, `description`, `execute()`, `validate_params()`
- `ToolRegistry` with timeout enforcement and permission gates
- Built-in tools: `WebFetchTool`, `SafeFileReaderTool`, `CodeExecutionTool`
- Path traversal prevention in file reader

**API:** `GET /tools/available`, `GET /tools/{name}/schema`

## 5. Knowledge Packs (`services/knowledge_packs/`)

See [KNOWLEDGE_PACKS.md](KNOWLEDGE_PACKS.md).

## 6. Developer KB (`services/edison_core/dev_kb/`)

See [DEV_KB.md](DEV_KB.md).

## 7. Unified Job Store (`services/edison_core/job_store.py`)

See [GENERATIONS.md](GENERATIONS.md).

## 8. Style Profiles & Prompt Expansion (`services/edison_core/prompt_expansion.py`)

Enriches short image prompts (< 8 words) with style-specific defaults:
- 4 built-in profiles: photorealistic, cinematic, anime, digital_art
- YAML-configurable at `config/style_profiles/*.yaml`

## 9. 3D Mesh Generation (`services/edison_core/mesh.py`)

ComfyUI-backed 3D generation with GLB/STL output.
Integrated with unified job store.

## 10. Workflow Intelligence (`services/edison_core/workflow_memory.py`)

Tracks which generation settings produce good results:
- Records outcomes with parameters
- Recommends best parameters for similar prompts
- User rating support (1-5 stars)

**API:** `POST/GET /workflows/*`

## 11. Observability (`services/edison_core/observability.py`)

Structured logging with correlation IDs:
- `ContextVar`-based correlation propagation
- Traces: retrieval, memory_save, tool_call, generation
- In-memory event buffer (last 10,000 events)

**API:** `GET /observability/events?limit=100`, `GET /observability/stats`

## File Map

```
services/edison_core/
├── memory/
│   ├── __init__.py
│   ├── models.py          # MemoryEntry, MemoryType
│   └── store.py           # MemoryStore (SQLite)
├── dev_kb/
│   ├── __init__.py
│   ├── manager.py         # DevKBManager (AST chunking)
│   └── cli.py
├── job_store.py            # Unified JobStore (SQLite)
├── retrieval.py            # QueryIntent + reranking
├── freshness.py            # FreshnessCache (SQLite)
├── tool_framework.py       # BaseTool + ToolRegistry
├── mesh.py                 # MeshGenerationService
├── workflow_memory.py      # WorkflowMemory (SQLite)
├── observability.py        # ObservabilityTracer
├── prompt_expansion.py     # Style profiles + expansion
├── video.py                # ← refactored (JobStore integration)
└── music.py                # ← refactored (JobStore integration)

services/knowledge_packs/
├── __init__.py
├── manager.py              # KnowledgePackManager
└── cli.py

config/style_profiles/
├── photorealistic.yaml
├── cinematic.yaml
├── anime.yaml
└── digital_art.yaml

tests/
└── test_new_subsystems.py  # 9 tests, all passing
```
