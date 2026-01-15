# EDISON (Offline) — Build Spec for GitHub Copilot

## Goal
Create a local/offline AI system called EDISON that:
- Runs on Ubuntu Server
- Uses ComfyUI as the UI/orchestrator (port 8188)
- Runs a local LLM service (edison-core) via FastAPI (port 8811)
- Runs a local Coral TPU service (edison-coral) via FastAPI (port 8808)
- Boots automatically using systemd services
- Exposes a ComfyUI custom node "EDISON Chat" that calls edison-core /chat

## Hardware
- i5-12600K, 64GB RAM
- RTX 3090 24GB, RTX 5060 Ti 16GB, RTX 3060 12GB
- Coral M.2 Edge TPU (B+M Key)
- Ubuntu Server latest
- Offline operation: no internet required after setup

## edison-core requirements
- Python FastAPI server with:
  - /health endpoint
  - /chat endpoint that supports mode: auto|chat|reasoning|agent|code
- Uses llama-cpp-python to load GGUF models:
  - fast model: qwen2.5-14b-instruct Q4_K_M
  - deep model: qwen2.5-72b-instruct Q4_K_M
- Uses Qdrant local storage as vector DB and sentence-transformers for embeddings
- RAG: if mode is reasoning/agent/code, retrieve top_k chunks and inject into prompt
- Agent mode (v1): produce a plan, optionally run python tool, then finalize
- Memory: append (user+assistant) into vector DB unless remember=false
- Must be robust and not crash if models are missing (provide helpful errors)

## edison-coral requirements
- Python FastAPI server with:
  - /health endpoint that reports TPU availability and model presence
  - /intent endpoint that returns intent label + confidence
- V1 can use heuristic intent classification
- V2 should allow EdgeTPU TFLite intent classifier if present (optional)
- Never attempt to run LLMs on TPU

## ComfyUI custom node requirements
- Node: "EDISON Chat" with inputs:
  - text (multiline)
  - mode enum (auto/chat/reasoning/agent/code)
  - remember boolean
- Node calls http://127.0.0.1:8811/chat and returns reply string

## scripts requirements
- setup_ubuntu.sh: installs python3-venv, pip, git; creates venv; installs requirements; clones ComfyUI and ComfyUI-Manager; creates folders
- install_coral.sh: installs Coral Edge TPU runtime (gasket-dkms + libedgetpu1-std)
- enable_services.sh: installs systemd unit files into /etc/systemd/system, enables and starts them, sets repo path /opt/edison, user edison, ownership
- download_models.py: stub that prints where to place GGUF models (do not redistribute)

## systemd units requirements
- edison-coral.service: runs edison-coral on 127.0.0.1:8808
- edison-core.service: runs edison-core on 127.0.0.1:8811 with CUDA_VISIBLE_DEVICES=0,1,2
- edison-comfyui.service: runs ComfyUI on 0.0.0.0:8188
- All services restart automatically and start at boot

## Constraints
- Offline: no web search tools
- Must run headless on Ubuntu Server
- Keep code simple, readable, and production-ish (logging, timeouts)
