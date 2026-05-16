# Edison Model Lab

Model Lab is the workspace for planning larger local models and tool additions
against the actual Edison hardware profile. It is designed for the 128 GB RAM,
RTX 3090 24 GB, RTX 5060 Ti 16 GB, and RTX 4060 Ti 16 GB class machine.

## What It Does

- Scans installed `.gguf` files under the configured Edison model directory.
- Reads system RAM and NVIDIA GPU telemetry when available.
- Recommends model profiles that fit the machine.
- Produces copyable `huggingface-cli download` commands.
- Lists useful GitHub/tool additions with cautions.

## Guardrail Policy

Model Lab supports advanced local models and bring-your-own GGUF profiles, but it
does not create a global "no safeguards" bypass. Edison still keeps filesystem
path checks, node dispatch rules, tool confirmations, and GPU/job guardrails
around model output.

This matters because model behavior and system permissions are separate layers:
a more permissive model should not automatically get unrestricted file, shell,
browser, or remote-node access.

## Recommended Model Roles

- General/deep assistant: `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF`
- Code mode: `Qwen/Qwen2.5-Coder-32B-Instruct-GGUF`
- Reasoning: `unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF`
- RAG quality: `cstr/bge-reranker-base-GGUF`

Use the RTX 3090 as the primary large-model GPU. The 16 GB cards are best used
for ComfyUI workers, smaller parallel jobs, vision/code helpers, or background
tasks. Do not treat the three GPUs as one pooled 56 GB VRAM device.

## Useful Tool Additions

- ComfyUI Manager: custom-node discovery and dependency visibility.
- browser-use / Playwright tooling: stronger visible browser-agent workflows.
- MCP-style tool servers: useful only when scoped with strict allowlists.

## Deployment

After pulling the branch on Edison, restart `edison-core.service` and
`edison-web.service`. The workspace is available at:

```text
https://<edison-host>:8080/model-lab
```

The backend summary endpoint is:

```text
GET /model-lab/summary
```

The install-plan endpoint is:

```text
GET /model-lab/install-plan/{profile_id}
```
