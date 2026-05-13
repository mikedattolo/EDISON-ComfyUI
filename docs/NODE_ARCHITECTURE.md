# Node and Remote Worker Architecture

Edison nodes are task workers. They do not pool VRAM across machines or make one logical GPU.

The node registry now supports richer capability metadata:

- OS, CPU, RAM
- one or more GPUs with VRAM metadata
- installed apps such as Rhino, SolidWorks, Blender, or ComfyUI
- allowed tools
- accepted job types
- heartbeat health payloads

Node task dispatch remains queue-first and HTTP-direct where available. `NodeManager.build_dispatch_request()` creates a stable envelope containing task payload, expected result shape, and node capabilities so remote agents can implement progress and artifact return consistently.

Use nodes for:

- CAD/Rhino/SolidWorks offload
- storage/indexing helpers
- background file work
- future GPU render workers with their own local VRAM

Do not use node mode to assume distributed VRAM pooling.

