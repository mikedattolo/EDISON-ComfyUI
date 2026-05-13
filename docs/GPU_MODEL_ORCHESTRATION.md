# GPU and Model Orchestration Notes

Edison treats each GPU as a separate device. The RTX 3090 24GB, RTX 5060 Ti 16GB, and RTX 4060 Ti 16GB should not be represented as a single 56GB GPU.

On the live Edison host, NVIDIA reports the PCI order as:

- GPU 0: RTX 5060 Ti 16GB
- GPU 1: RTX 4060 Ti 16GB
- GPU 2: RTX 3090 24GB

Current PyTorch `2.6.0+cu124` supports the RTX 3090 and RTX 4060 Ti, but not the RTX 5060 Ti `sm_120` compute capability. Until a PyTorch/CUDA 13 family build that supports `sm_120` is installed and validated, Edison and ComfyUI service environments should mask PyTorch-visible GPUs to the 3090 + 4060 Ti while diagnostics continue to show all three cards through `nvidia-smi`.

This upgrade adds or strengthens:

- a richer `/models/status` response with per-slot configured model names and loaded/shared state
- OpenAI-compatible `/v1/models` listing with aliases and multimodal metadata
- global JobStore progress, stage, title, summary, timestamps, and log fields
- Persona Video registration into the unified jobs center
- cooling diagnostics that heavy jobs can use before long GPU work
- ComfyUI template capability metadata for future GPU-aware dispatch

Heavy jobs should coordinate through a shared workload governor rather than each subsystem unloading/reloading models independently. Persona Video Exclusive GPU Render Mode already snapshots Edison-controlled models/services, unloads what it can, waits for VRAM thresholds, records GPU state, and restores intended services afterward.
