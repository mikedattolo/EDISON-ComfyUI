# Persona Video Studio ComfyUI Workflows

Place curated Persona Video Studio ComfyUI workflow JSON templates in `comfyui_workflows/`.

The backend adapter expects future templates to support variable injection for:

- source segment or frame sequence path
- persona reference image/model/LoRA paths
- selected transformation scope
- output segment path
- optional GPU/device hint when the installed ComfyUI stack supports it

Core Persona Video Studio job handling does not depend on these templates. If no templates are installed, Edison exposes the ComfyUI adapter as setup-required and falls back to metadata-only pipeline validation.
