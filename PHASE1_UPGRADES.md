# Phase 1 Upgrades

This document describes the additive Phase 1 upgrades that move EDISON
closer to Copilot/ChatGPT/Claude quality without breaking any existing
functionality. Everything here is optional — none of the existing chat,
image, video, branding, fabrication, or connector code paths were
modified.

## What was added

### 1. GPU scheduler with workload lanes
**Module:** [services/edison_core/gpu_scheduler.py](services/edison_core/gpu_scheduler.py)

Coordinates concurrent jobs across logical lanes so chat/code stay
responsive while image, video, and CAD work runs in the background.

Lanes (with default concurrency / preferred GPU):

| Lane         | Concurrency | Preferred GPU |
|--------------|-------------|---------------|
| chat         | 8           | 0             |
| code         | 4           | 0             |
| vision       | 2           | 0             |
| tool         | 4           | 0             |
| image        | 1           | 1             |
| video        | 1           | 2             |
| music        | 1           | 2             |
| mesh / cad   | 1           | 2             |
| background   | 1           | 2             |

Override per-lane concurrency at startup with environment variables:

```
EDISON_LANE_CHAT_CONCURRENCY=12
EDISON_LANE_VIDEO_CONCURRENCY=2
```

Usage from any async code path:

```python
from services.edison_core.gpu_scheduler import run_on_lane

async def render_image():
    ...

result = await run_on_lane("image", render_image, metadata={"prompt": "..."})
```

### 2. Hardened vLLM serving bridge
**Module:** [services/edison_vllm/bridge.py](services/edison_vllm/bridge.py)

* Per-lane engines (`fast`, `deep`, `vision`) with proportional GPU
  memory utilization and per-lane tensor-parallel sizing.
* OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, and
  `/v1/models` endpoints.
* Streaming responses (Server-Sent Events).
* `/healthz` and `/readyz` for orchestration.
* Optional prefix-cache and attention flags via env vars.

The original `services/edison_vllm/server.py` is kept for backward
compatibility.

### 3. Retry / backoff utility
**Module:** [services/edison_core/retry.py](services/edison_core/retry.py)

`retry_sync`, `retry_async`, `RetryPolicy`, and `friendly_error()` for
clearer chat-facing failure messages.

### 4. Citations helper
**Module:** [services/edison_core/citations.py](services/edison_core/citations.py)

Normalizes RAG / web / knowledge hits into a consistent `Citation` shape,
renders inline reference markers, and emits a JSON bundle the front-end
can render as a sources panel.

### 5. Artifact streaming
**Module:** [services/edison_core/artifact_stream.py](services/edison_core/artifact_stream.py)

`ArtifactStream` lets producers push partial deltas to consumers while a
generation is in flight, with built-in revision tracking and a
diff-descriptor helper.

### 6. Phase 1 HTTP routes
**Module:** [services/edison_core/routes/phase1.py](services/edison_core/routes/phase1.py)

Mounted into `app.py` with the same defensive pattern as the other route
modules. Endpoints:

* `GET  /api/phase1/health`
* `GET  /api/phase1/scheduler/telemetry`
* `GET  /api/phase1/scheduler/lanes`
* `POST /api/phase1/citations/bundle`
* `GET  /api/phase1/artifacts/{id}/revisions`
* `GET  /api/phase1/artifacts/{id}/diff`

## Tests

[test_phase1_upgrades.py](test_phase1_upgrades.py) — 12 tests covering
scheduler, retry, citations, and artifact streams. Run with:

```
venv/bin/python -m pytest test_phase1_upgrades.py -q
```

## What was *not* changed

* `app.py` was modified only to register the new optional route module.
* No existing endpoints, modes, prompts, storage formats, or front-end
  pages were touched.
* The original vLLM `server.py` is still in place; the new `bridge.py`
  runs alongside it on the same port if you choose to switch.

## Roadmap pointer

This delivers Phase 1 of the upgrade plan. Phase 2 (workload queues
wired into image/video pipelines, artifact revision UI in the front-end,
unified workspace shell) and Phase 3 (advanced throughput, richer video
timeline, CAD QA gates) build on these foundations.
