# Edison Awareness & Context System

Edison's awareness layer makes the assistant more context-sensitive, proactive, and self-improving. All modules are **opt-in, graceful-fallback**: if any component fails to initialize, the rest of Edison continues to function normally.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                      chat_stream()                      │
│                                                         │
│  1. coral intent  ──►  2. classify_intent_with_goal()   │
│                            ▼                            │
│  3. conversation_state     4. project_state             │
│                            ▼                            │
│  5. planner.create_plan()  6. route_mode()              │
│                            ▼                            │
│  7. LLM generation                                      │
│                            ▼                            │
│  8. self_eval.record()     9. suggestion_engine          │
│  10. structured logging                                  │
└─────────────────────────────────────────────────────────┘
```

---

## Modules

### 1. Conversation State (`services/state/conversation_state.py`)

Tracks per-session structured state across turns.

**Key class:** `ConversationStateManager`

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | str | Maps to `chat_id` from frontend |
| `current_task` | str | Current goal being worked on |
| `active_domain` | str | Detected domain (image, code, music, etc.) |
| `turn_count` | int | Number of turns in session |
| `error_count` | int | Errors encountered |
| `last_intent` | str | Last classified intent |
| `last_goal` | str | Last detected goal |
| `task_stage` | str | `idle` / `in_progress` / `completed` |

**Usage:**
```python
from services.state.conversation_state import get_conversation_state_manager

mgr = get_conversation_state_manager()
state = mgr.get_state("session_123")
mgr.update_state("session_123", {"active_domain": "image"})
mgr.increment_turn("session_123")
```

**Automatic cleanup:** Sessions idle >4 hours are pruned. Max 500 concurrent sessions.

---

### 2. Goal-Level Intent Detection (`services/state/intent_detection.py`)

Classifies user messages into high-level goals beyond raw Coral intents.

**9 Goals:**
- `DEBUG_CODE` — fix, error, crash, traceback
- `MODIFY_PREVIOUS_OUTPUT` — change it, make it brighter, redo
- `CONTINUE_PREVIOUS_TASK` — keep going, next step, also
- `RESEARCH_TOPIC` — what is, look up, latest news
- `GENERATE_NEW_ARTIFACT` — create, generate, make, build
- `EXPLAIN_CONCEPT` — explain, describe, how does
- `CONFIGURE_SYSTEM` — configure, settings, install, restart
- `RECALL_MEMORY` — remember, recall, what did I say
- `CASUAL_CHAT` — fallback

**Continuation Detection:**
- `NEW_TASK` — user starts a new topic
- `MODIFY_PREVIOUS` — user wants to adjust last output
- `CONTINUE_PREVIOUS` — user continues the same task

**Usage:**
```python
from services.state.intent_detection import classify_intent_with_goal

result = classify_intent_with_goal(
    message="make it brighter",
    coral_intent="generate_image",
    last_intent="generate_image",
    turn_count=3,
)
# result.goal == Goal.MODIFY_PREVIOUS_OUTPUT
# result.continuation == ContinuationType.MODIFY_PREVIOUS
```

---

### 3. System Awareness (`services/state/system_state.py`)

Monitors GPU, disk, ComfyUI, and job status with a 10-second cache.

```python
from services.state.system_state import get_system_state

snapshot = get_system_state()
# snapshot.gpus[0].memory_used_mb
# snapshot.disks[0].free_gb
# snapshot.comfyui_running
# snapshot.active_jobs
```

**GPU detection:** pynvml → PyTorch CUDA fallback → empty list.

---

### 4. Coral Awareness Plugins (`services/coral_plugins/plugins.py`)

Optional plugins that extend Coral TPU capabilities. All gracefully unavailable when hardware is missing.

| Plugin | Description | Requires |
|--------|-------------|----------|
| `PresenceDetectionPlugin` | Camera-based face detection | OpenCV + camera |
| `ActivityDetectionPlugin` | Idle/typing/away detection | Software only |
| `ObjectDetectionPlugin` | Edge TPU object detection | Coral TPU |

```python
from services.coral_plugins.plugins import get_coral_plugin_registry

registry = get_coral_plugin_registry()
registry.register(PresenceDetectionPlugin())
results = registry.detect_all()
```

---

### 5. Project Awareness (`services/state/project_state.py`)

Tracks files, repos, and language context mentioned in conversation.

```python
from services.state.project_state import get_project_state_manager

mgr = get_project_state_manager()
mgr.detect_project_from_message("sess_1", "editing main.py in project my_app")
ctx = mgr.get_context("sess_1")
# ctx.name == "my_app"
# ctx.language == "python"
# ctx.recent_files == ["main.py"]
```

---

### 6. Proactive Suggestions (`services/awareness/suggestions.py`)

Non-intrusive suggestion engine that triggers on specific conditions.

| Trigger | Condition | Category |
|---------|-----------|----------|
| Repeated errors | ≥3 errors in session | `error_help` |
| Long-running task | Task >300s | `performance` |
| Idle after failure | Idle >60s after error | `idle_hint` |
| Memory opportunity | First conversation turn | `memory` |
| System resources | GPU >90% or disk <5GB | `resources` |

```python
from services.awareness.suggestions import get_suggestion_engine

engine = get_suggestion_engine()
engine.evaluate(error_count=5, last_error_message="OOM", ...)
pending = engine.get_pending()
engine.dismiss(pending[0].suggestion_id)
```

---

### 7. Planner Layer (`services/planner/planner.py`)

Rule-based planner that creates ordered execution plans between intent detection and routing.

**Plan steps:** `memory_retrieve` → `web_search` → `tool_execute` → `generate_*` → `llm_respond` → `memory_write`

**Complexity levels:** TRIVIAL / SIMPLE / MULTI_STEP / PARALLEL

```python
from services.planner.planner import get_planner

planner = get_planner()
plan = planner.create_plan(
    intent="generate_image",
    goal="generate_new_artifact",
    message="a dragon in watercolor style",
    has_image=False,
)
# plan.steps[0].action == "generate_image"
# plan.complexity == PlanComplexity.SIMPLE
```

---

### 8. Self-Evaluation (`services/awareness/self_eval.py`)

SQLite-backed outcome tracking for every request.

```python
from services.awareness.self_eval import get_self_evaluator

evaluator = get_self_evaluator()
evaluator.record(session_id="s1", intent="generate_image", outcome="success")
evaluator.record_correction(session_id="s1", original_intent="chat", corrected_intent="generate_image")

stats = evaluator.get_stats()
# {"total": 100, "success": 92, "error": 8, "success_rate": 0.92}
correction_rate = evaluator.get_correction_rate()
# 0.03
```

**DB location:** `data/self_eval.db`

---

### 9. Structured Logging (`services/awareness/structured_logging.py`)

JSON-formatted logging for all awareness decisions. Writes to the `edison.awareness` logger.

```python
from services.awareness.structured_logging import log_intent_decision

log_intent_decision(
    intent="generate_image",
    goal="generate_new_artifact",
    continuation="new_task",
    confidence=0.92,
    session_id="s1",
)
# Output: {"event": "intent_decision", "intent": "generate_image", ...}
```

**Convenience wrappers:** `log_intent_decision()`, `log_planner_decision()`, `log_routing_decision()`, `log_state_update()`, `log_eval_outcome()`

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/awareness/state/{session_id}` | Conversation state for a session |
| `GET` | `/awareness/project/{session_id}` | Project context for a session |
| `GET` | `/awareness/suggestions` | Pending proactive suggestions |
| `POST` | `/awareness/suggestions/{id}/dismiss` | Dismiss a suggestion |
| `GET` | `/awareness/system` | System state snapshot (GPU, disk, jobs) |
| `GET` | `/awareness/eval/stats` | Self-evaluation statistics |
| `GET` | `/awareness/eval/recent` | Recent evaluation records |
| `GET` | `/awareness/plan` | Planner status |
| `GET` | `/awareness/coral-plugins` | List Coral plugins |

---

## Integration Points in `app.py`

1. **Globals** (L432-443): Six new awareness globals initialized to `None`
2. **`_init_new_subsystems()`**: Each module initialized in a try/except block with graceful fallback
3. **`chat_stream()`** after `get_intent_from_coral()`: Runs `classify_intent_with_goal()`, updates conversation state, detects project context
4. **`chat_stream()`** after `route_mode()`: Creates execution plan, logs routing decision
5. **`chat_stream()`** after `store_conversation_exchange()`: Records outcome in self-evaluator, updates task stage, evaluates suggestions
6. **`chat_stream()`** exception handler: Records errors in conversation state and self-evaluator
7. **`chat()`** non-streaming: Lightweight intent classification hook

---

## Safety & Stability

- **All awareness code is wrapped in try/except** — failures log a debug message and the request continues normally
- **No blocking I/O** in the hot path — system state is cached for 10s, self-eval writes are fast SQLite inserts
- **Session limits** — max 500 sessions, 4-hour idle timeout with automatic pruning
- **Thread-safe** — all managers use threading locks for concurrent access
- **Graceful degradation** — if pynvml/PyTorch/OpenCV/Coral are unavailable, those features simply report as unavailable

---

## Testing

```bash
python tests/test_awareness.py
```

10 tests covering: conversation state CRUD, domain detection, goal detection, continuation detection, unified classifier, system state, project state, suggestion engine, planner routing, self-evaluation.

---

## File Inventory

| File | Part | Purpose |
|------|------|---------|
| `services/state/__init__.py` | — | Package init |
| `services/state/conversation_state.py` | 1 | Per-session state tracking |
| `services/state/intent_detection.py` | 2,3 | Goal + continuation detection |
| `services/state/system_state.py` | 4 | GPU/disk/job monitoring |
| `services/state/project_state.py` | 6 | Project context tracking |
| `services/coral_plugins/__init__.py` | — | Package init |
| `services/coral_plugins/plugins.py` | 5 | Coral TPU plugins |
| `services/awareness/__init__.py` | — | Package init |
| `services/awareness/suggestions.py` | 7 | Proactive suggestions |
| `services/awareness/self_eval.py` | 9 | Self-evaluation loop |
| `services/awareness/structured_logging.py` | 10 | JSON structured logging |
| `services/planner/__init__.py` | — | Package init |
| `services/planner/planner.py` | 8 | Rule-based planner |
| `tests/test_awareness.py` | 12 | Test suite |
| `docs/AWARENESS.md` | 13 | This document |
