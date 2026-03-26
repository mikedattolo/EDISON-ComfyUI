# Cura Printing Page Handoff

## Current Status

No code changes have been applied yet.

This handoff captures the investigation completed so far for improving the 3D printing page with Cura/CuraEngine-style workflow support.

## Relevant Files Identified

- `web/printing.html`
  - Current 3D printing UI.
  - Has printer discovery, printer registration, basic slicing, and slice-and-send controls.
  - Current slicing UI is minimal and only exposes `model_path` and a simple `profile` field.

- `services/edison_core/app.py`
  - FastAPI routes for printing are defined inline on the main app.
  - Relevant route handlers found:
    - `GET /printing/printers`
    - `POST /printing/discover`
    - `GET /printing/setup-guide/{printer_type}`
    - `POST /printing/printers`
    - `GET /printing/printers/{printer_id}/status`
    - `POST /printing/printers/send`
    - `PATCH /printing/printers/{printer_id}`
    - `DELETE /printing/printers/{printer_id}`
    - `POST /printing/slice`
    - `GET /printing/slice/{job_id}`
    - `POST /printing/slice-and-send`

- `services/edison_core/printer.py`
  - Existing printer manager and driver abstraction.
  - Handles configured printers and dispatch to Bambu/OctoPrint/generic backends.
  - Does not currently provide a dedicated slicer abstraction.

- `tests/test_printer_manager.py`
  - Existing tests for printer listing, status, and sending jobs.

## What Exists Today

### Current slicing implementation

The current slicing flow in `services/edison_core/app.py`:

- uses `prusa-slicer`, `orca-slicer`, or `slic3r` if found on `PATH`
- writes `.gcode` beside the source model
- stores transient slice status in `_SLICE_STATUS`
- has no CuraEngine-first integration
- has no capabilities endpoint for frontend settings/bootstrap
- has no estimate/preview endpoint for print time/material/cost
- has no structured slicing profile model beyond a single `profile` string

### Current UI limitations

The page in `web/printing.html` currently lacks:

- material selection
- infill control
- support toggles
- nozzle selection
- bed adhesion controls
- layer-height preset handling beyond free text
- print estimate preview
- slicer capability/status visibility
- Cura-like workflow organization

## Recommended Implementation Plan

### 1. Add a dedicated slicer service

Create a new module, likely:

- `services/edison_core/slicing.py`

Responsibilities:

- detect available slicers with preference order:
  1. `CuraEngine`
  2. `prusa-slicer`
  3. `orca-slicer`
  4. `slic3r`
- expose available engine metadata to the UI
- accept structured slicing options
- normalize engine-specific command generation
- return consistent result payloads for:
  - capability discovery
  - estimates
  - slice execution

### 2. Extend the printing API

Add endpoints or extend existing routes in `services/edison_core/app.py`:

- `GET /printing/slicer/capabilities`
  - installed engine
  - supported defaults/presets
  - whether CuraEngine is available

- `POST /printing/slice/estimate`
  - accept model path + structured slicing options
  - return estimated duration/material/cost
  - if the selected engine cannot estimate directly, provide a safe heuristic fallback

- extend `POST /printing/slice`
  - keep backward compatibility with existing `profile` input
  - also accept structured slicing options such as:
    - `quality`
    - `layer_height`
    - `material`
    - `infill`
    - `supports`
    - `adhesion`
    - `nozzle`
    - `speed_profile`

- extend `POST /printing/slice-and-send`
  - same structured options support
  - preserve old request shape compatibility

### 3. Redesign the printing page

Replace the basic slicing panel in `web/printing.html` with a Cura-like workbench layout:

- printer column
  - selected printer
  - live status
  - quick connect/save controls

- model/slicing column
  - model path
  - quality preset
  - layer height
  - material
  - infill percent
  - supports toggle
  - adhesion type
  - nozzle size
  - speed preset

- estimate/output column
  - slicer engine status
  - estimated time
  - estimated filament usage
  - estimated cost
  - generated gcode path
  - JSON/raw job details

### 4. Add tests

Add focused tests for the slicer service and route behavior, likely under `tests/`.

Suggested coverage:

- slicer capability detection
- CuraEngine-first fallback order
- backward-compatible `POST /printing/slice` behavior
- structured slicing option parsing
- estimate endpoint response shape
- slice-and-send integration using a mocked slicer

## Practical Constraint

Using the full Ultimaker Cura desktop application directly inside this app is not the best integration path.

Recommended approach:

- integrate with `CuraEngine` CLI semantics and Cura-like presets/workflow
- do not attempt to embed the Cura desktop app UI into Edison

This preserves compatibility with the existing local-first FastAPI/web architecture.

## Next Steps To Resume

When resuming, start in this order:

1. Create `services/edison_core/slicing.py`.
2. Wire a slicer service instance into `services/edison_core/app.py` startup/init area.
3. Add `GET /printing/slicer/capabilities`.
4. Add `POST /printing/slice/estimate`.
5. Extend `POST /printing/slice` and `POST /printing/slice-and-send` to accept structured options.
6. Replace the Step 3 section in `web/printing.html` with a Cura-style workbench UI.
7. Add tests.
8. Run targeted validation.

## Validation To Run After Implementation

- `pytest tests/test_printer_manager.py`
- targeted API tests for printing routes if present
- manual smoke test of `/printing.html`
- verify fallback behavior when `CuraEngine` is not installed

## Notes

- No repository files were modified before this handoff file was created.
- Existing behavior should remain backward-compatible.
- The printing routes are embedded in a large `app.py`, so patch carefully and avoid unrelated refactors.