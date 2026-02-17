# Repo Stability Pass (Phase 0)

## Overview

Before adding new features, a stability audit was performed to identify and fix
issues that could cause runtime failures or mask bugs.

## What Was Found & Fixed

### 1. Bare `except:` Blocks → `except Exception:`

**7 instances** across two files were changed from bare `except:` to
`except Exception:` so that `KeyboardInterrupt` and `SystemExit` are no longer
accidentally swallowed.

| File | Lines |
|------|-------|
| `services/edison_core/app.py` | L5283, L5382, L6630, L6641, L6655, L6675 |
| `services/edison_core/sandbox.py` | L200 |

### 2. Compile-All Check

`python -m compileall -q services/` passes with zero errors, confirming no
syntax issues exist anywhere in the service tree.

### 3. `scripts/check.sh`

A new repo-health script was added/enhanced:

```bash
bash scripts/check.sh
```

It runs:
- `python -m compileall -q services/` — syntax check
- `ruff check services/` (if installed) — lint
- Merge-conflict marker scan (`<<<<<<<`)
- Unit tests for awareness subsystems and memory/file subsystems

### 4. Dead Code: `model_manager_v2.py`

`services/edison_core/model_manager_v2.py` (474 lines) existed but was never
imported. It is now registered in `_init_new_subsystems()` and available as
`model_manager_v2_instance` and `memory_gate_instance`. It does **not** replace
the existing `load_llm_models()` startup path to avoid breaking anything — it
runs alongside as a secondary safety layer.

### 5. Missing Directories

The `uploads/` directory tree (referenced by file upload code) is now
auto-created by `FileStore.__init__()`.

## Validation

- All bare `except:` blocks fixed
- `compileall` clean
- No merge conflict markers
- 41 new tests pass in `tests/test_memory_and_files.py`
- 26 tests pass in `tests/test_awareness.py`
