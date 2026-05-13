# GPU and Model Orchestration Notes

Edison treats each GPU as a separate device. The RTX 3090 24GB, RTX 5060 Ti 16GB, and RTX 4060 Ti 16GB should not be represented as a single 56GB GPU.

On the live Edison host, NVIDIA reports the PCI order as:

- GPU 0: RTX 5060 Ti 16GB
- GPU 1: RTX 4060 Ti 16GB
- GPU 2: RTX 3090 24GB

The live Edison host was upgraded to PyTorch `2.11.0+cu130` for both Edison
core and ComfyUI. CUDA allocation has been validated on all three cards,
including the RTX 5060 Ti `sm_120`.

Edison should still avoid describing the system as pooled VRAM. The production
path is per-GPU worker dispatch:

- the primary ComfyUI worker runs on the RTX 3090 at `127.0.0.1:8188`
- the RTX 5060 Ti worker runs at `127.0.0.1:8189`
- the RTX 4060 Ti worker runs at `127.0.0.1:8190`

See `docs/COMFYUI_GPU_WORKERS.md` for installation and troubleshooting.

This upgrade adds or strengthens:

- a richer `/models/status` response with per-slot configured model names and loaded/shared state
- OpenAI-compatible `/v1/models` listing with aliases and multimodal metadata
- global JobStore progress, stage, title, summary, timestamps, and log fields
- Persona Video registration into the unified jobs center
- cooling diagnostics that heavy jobs can use before long GPU work
- ComfyUI template capability metadata and GPU-aware worker dispatch

Heavy jobs should coordinate through a shared workload governor rather than each subsystem unloading/reloading models independently. Persona Video Exclusive GPU Render Mode already snapshots Edison-controlled models/services, unloads what it can, waits for VRAM thresholds, records GPU state, and restores intended services afterward.
