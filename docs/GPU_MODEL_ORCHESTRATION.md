# GPU and Model Orchestration Notes

Edison treats each GPU as a separate device. The RTX 3090 24GB, RTX 5060 Ti 16GB, and RTX 4060 Ti 16GB should not be represented as a single 56GB GPU.

This upgrade adds or strengthens:

- a richer `/models/status` response with per-slot configured model names and loaded/shared state
- OpenAI-compatible `/v1/models` listing with aliases and multimodal metadata
- global JobStore progress, stage, title, summary, timestamps, and log fields
- Persona Video registration into the unified jobs center
- cooling diagnostics that heavy jobs can use before long GPU work
- ComfyUI template capability metadata for future GPU-aware dispatch

Heavy jobs should coordinate through a shared workload governor rather than each subsystem unloading/reloading models independently. Persona Video Exclusive GPU Render Mode already snapshots Edison-controlled models/services, unloads what it can, waits for VRAM thresholds, records GPU state, and restores intended services afterward.

