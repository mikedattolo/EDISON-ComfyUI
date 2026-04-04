# EDISON Runtime Architecture

## Overview

The EDISON runtime layer extracts shared business logic out of `app.py` into reusable, testable modules under `services/edison_core/runtime/`. All modules are imported at startup and wired into the existing route handlers via thin wrappers — the monolithic `app.py` still handles HTTP concerns, but delegates core logic to these modules.

## Module Map

| Module | Purpose |
|--------|---------|
| `routing_runtime.py` | Intent classification → mode/model/tool decisions |
| `tool_runtime.py` | Tool registry, validation, domain tool schemas |
| `context_runtime.py` | Layered context assembly (conversation, RAG, task, artifact) |
| `task_runtime.py` | Persistent task state across conversation turns |
| `artifact_runtime.py` | Artifact registry and lifecycle tracking |
| `workspace_runtime.py` | Logical workspace/project scoping |
| `model_runtime.py` | Task-aware model selection and fallback chains |
| `quality_runtime.py` | Response quality checks, cleanup, trust signals |
| `response_runtime.py` | OpenAI-format response formatting, SSE helpers |
| `chat_runtime.py` | Unified 12-stage chat pipeline (future) |
| `search_runtime.py` | Multi-stage research pipeline |
| `browser_runtime.py` | Browser session management |

## Tool Registry

All tools are registered in `tool_runtime.py:TOOL_REGISTRY`. Domain-specific tools are organized into groups:

### Core Tools
`web_search`, `rag_search`, `knowledge_search`, `generate_image`, `system_stats`, `execute_python`, `read_file`, `list_files`, `analyze_csv`, `get_current_time`, `get_weather`, `get_news`, `generate_music`, `call_external_api`, `open_sandbox_browser`, `browser.*`, `write_file`, `summarize_url`

### Task Management Tools
`create_task`, `list_tasks`, `complete_task`

### Branding Tools
- `generate_brand_package` — Full brand package generation (logos, palette, typography, slogans, moodboard, style guide)
- `generate_marketing_copy` — Multi-channel marketing copy (ad copy, social captions, email campaigns, etc.)
- `create_branding_client` — Create a new branding client record
- `list_branding_clients` — List all clients

### Video Tools
- `generate_video` — Generate video from a text prompt

### Project Tools
- `create_project` — Create a new project with client linkage and service types
- `list_projects` — List projects with optional status filter

### Fabrication Tools
- `slice_model` — Slice a 3D model file for printing

### Printer Tools (conditional)
- `list_printers`, `get_printer_status`, `send_3d_print`

## Artifact Tracking

When domain tools execute successfully, they register artifacts in the in-memory store via `artifact_runtime.register_artifact()`. This enables:

- Chat-scoped artifact queries (`get_artifacts_for_chat`)
- Project-scoped artifact queries (`get_artifacts_for_task`)
- Context injection — artifacts are summarized into the prompt for follow-up turns

Tracked artifact types: `brand_brief`, `marketing_copy`, `audio`, `video`, `project`, `print_job`, `note`, `report`, `code`, `image`

## API Routers

### System Awareness (`api_system_awareness.py`)
`/api/system/capabilities`, `/routes`, `/pages`, `/config-files`, `/environment`, `/runtime-modules`, `/health`, `/code-search`, `/inspect-file`

### Projects & Clients (`api_projects.py`)
`/api/clients` (CRUD), `/api/projects` (CRUD with tasks, assets, deliverables)

## Wiring Pattern

The runtime modules are connected to `app.py` via:

1. **Import aliases** at the top of `app.py`:
   ```python
   from services.edison_core.runtime.routing_runtime import route_mode as runtime_route_mode
   from services.edison_core.runtime.tool_runtime import TOOL_REGISTRY as RUNTIME_TOOL_REGISTRY
   ```

2. **Thin wrappers** in `app.py` that delegate to runtime:
   ```python
   def route_mode(msg, mode, has_image):
       return runtime_route_mode(msg, mode, has_image)
   ```

3. **Direct calls** in `_execute_tool` for domain tool dispatch

4. **Quality checks** injected into response streaming paths via `runtime_clean_response()`

5. **Trust signals** generated in streaming done payloads via `format_trust_signals()`

## Testing

`test_runtime_integration.py` — 38 tests covering:
- Routing decisions and backward compatibility
- Tool registry completeness and validation
- Context assembly and conversation summaries
- Task CRUD operations
- Artifact registration and retrieval
- Quality checks and response cleanup
- Domain tool schemas and validation
- API router mounting
- App.py integration verification
