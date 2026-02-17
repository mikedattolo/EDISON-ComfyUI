# Memory Management

## Architecture

Edison uses a layered memory management system to prevent GPU VRAM crashes
while maintaining maximum performance.

### Layer 1: Existing Model Loader (`load_llm_models()`)

The original eager loader at startup that loads fast → medium → deep → reasoning
→ vision models. This remains the primary startup path.

### Layer 2: ModelManager v2 (`model_manager_v2.py`)

A memory-aware model manager that runs alongside the existing loader:

- **MemorySnapshot** — probes RAM (via `psutil`) and VRAM (via `pynvml` →
  `torch.cuda` → `nvidia-smi` fallback ladder) to know current memory state.
- **LoadedModel** — metadata for each loaded model (key, path, VRAM estimate,
  GPU layers, context size).
- **ModelManager** — manages a "heavy slot" (only one heavy model at a time)
  with a fallback ladder for GPU layers and context sizes:

  ```
  GPU layers: [config_default, 32, 20, 12, 4, 0]
  Context:    [config_default, 4096, 2048, 1024]
  ```

  If a model fails to load, it automatically tries with fewer GPU layers and
  smaller context sizes until it succeeds or exhausts options.

### Layer 3: Memory Gate

Pre-flight VRAM check before heavy GPU tasks (image/video/music generation):

```
pre_heavy_task(required_vram_mb=4000)
  → check free VRAM
  → if insufficient: unload heavy slot
  → if still insufficient: unload all models
  → final VRAM check
  → returns {ok: bool, freed_mb: float, snapshot: MemorySnapshot}
```

After the heavy task completes:

```
post_heavy_task()
  → reload fast model if needed
  → flush GPU caches
```

This is integrated into:
- `/generate-image` — 4000 MB VRAM threshold
- `/generate-video` — 6000 MB VRAM threshold
- `/generate-music` — 4000 MB VRAM threshold

### Layer 4: Swarm Memory Policy (`swarm_safety.py`)

When Swarm mode selects multiple agents (each potentially needing different
models), the policy assesses available VRAM:

| Mode | Condition | Behavior |
|------|-----------|----------|
| **normal** | ≤1 heavy model needed OR VRAM > 2×threshold | All agents get their preferred model |
| **time_slice** | VRAM between threshold and 2×threshold | Agents grouped by model, run sequentially |
| **degraded** | VRAM < threshold (5000 MB default) | All agents share a single fallback model |

Additional features:
- **Vision-on-demand** (`should_load_vision()`) — only loads the vision model
  when images are present or vision keywords detected in the prompt
- **Agent grouping** (`group_agents_by_model()`) — groups agents by model tag
  for efficient time-sliced execution

## Existing VRAM Safety

The existing `unload_all_llm_models()` → `_flush_gpu_memory()` → `gc.collect()`
→ `torch.cuda.empty_cache()` pattern remains the primary safety mechanism.
ModelManager v2 and the Memory Gate are secondary layers that provide additional
protection.

## Configuration

ModelManager v2 reads from the same `config/edison.yaml` as the existing
loader. No new configuration is required — it uses the existing model paths,
GPU layers, and context sizes from config.

## Monitoring

- `GET /system/memory` — returns current MemorySnapshot with RAM, VRAM, and
  loaded model information.
- Logs include memory gate decisions: `Memory gate: ok=True, freed=0MB`
- Swarm policy decisions logged: `Swarm policy: normal`
