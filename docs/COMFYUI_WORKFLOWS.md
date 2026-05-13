# ComfyUI Workflow Integration

Edison includes a shared ComfyUI workflow layer in `services/edison_core/comfyui_integration.py`.

GPU-aware worker routing lives in `services/edison_core/comfyui_workers.py`.
When `edison.comfyui.workers_enabled` is true, Edison chooses among separate
ComfyUI processes instead of sending every prompt to `8188`.

It supports:

- JSON workflow template discovery
- Edison metadata blocks under `edison`, `_edison`, or `metadata`
- validation of workflow object shape
- required node/model metadata
- `${placeholder}` discovery and variable injection
- prompt submit to `/prompt`
- queue/history polling
- prompt cancellation through `/interrupt`
- user-readable error translation
- per-GPU worker selection for image, mesh, and Persona Video jobs

Persona Video Studio looks for templates in:

```text
config/persona_video/comfyui_workflows/
```

Recommended template metadata:

```json
{
  "edison": {
    "template_id": "persona-basic",
    "name": "Persona Basic",
    "version": "1.0",
    "required_nodes": ["VideoHelperSuite"],
    "required_models": ["checkpoint.safetensors"],
    "parameter_schema": {},
    "capabilities": {
      "supports_parallel_segment_processing": true
    }
  },
  "workflow": {
    "1": {
      "class_type": "LoadVideo",
      "inputs": {
        "path": "${source_segment_path}"
      }
    }
  }
}
```

Supported Edison placeholders include `source_segment_path`, `source_frame_path`, `persona_paths`, `persona_pack_path`, `output_path`, `quality_preset`, `transformation_scope`, `segment_id`, `gpu_index`, and `gpu_name`.

Workflow metadata can include `capabilities.estimated_vram_mb`; Persona Video
uses that estimate when selecting a ComfyUI worker for a segment.

If templates, custom nodes, or models are missing, Edison reports setup-required details rather than claiming the backend is ready.
