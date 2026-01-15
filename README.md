# EDISON-ComfyUI

Fully offline AI system combining local LLMs with ComfyUI for image generation workflows.

## Architecture

- **edison-core** (port 8811) - FastAPI service running local LLMs via llama-cpp-python with RAG
- **edison-coral** (port 8808) - FastAPI service for intent classification (optional Edge TPU support)
- **ComfyUI** (port 8188) - Web UI for image generation with EDISON Chat custom node

## Hardware Requirements

- Ubuntu Server (tested on 24.04 LTS)
- NVIDIA GPU (3x GPUs recommended: RTX 3090, RTX 5060 Ti, RTX 3060)
- 64GB RAM recommended
- Optional: Google Coral Edge TPU M.2 module

## Installation Runbook

### 1. Initial Setup

Clone the repository:
```bash
git clone https://github.com/mikedattolo/EDISON-ComfyUI.git
cd EDISON-ComfyUI
```

Run the setup script:
```bash
bash scripts/setup_ubuntu.sh
```

This will:
- Install system dependencies
- Create Python virtual environment at `.venv`
- Install Python packages from `requirements.txt`
- Clone ComfyUI and ComfyUI-Manager
- Create model directories

### 2. Install LLM Models

Download GGUF models and place them in `models/llm/`:

```bash
# Example using wget (you'll need to find the actual model URLs)
# Fast model (14B, ~8GB)
wget -O models/llm/qwen2.5-14b-instruct-q4_k_m.gguf [MODEL_URL]

# Deep model (72B, ~40GB)
wget -O models/llm/qwen2.5-72b-instruct-q4_k_m.gguf [MODEL_URL]
```

Recommended sources:
- Hugging Face: https://huggingface.co/models?search=qwen2.5+gguf
- Look for Q4_K_M quantization for best quality/size tradeoff

### 3. Optional: Install Coral Edge TPU

If you have a Coral Edge TPU:
```bash
bash scripts/install_coral.sh
sudo reboot
```

After reboot, verify:
```bash
ls -l /dev/apex_0
python3 -c 'from pycoral.utils import edgetpu; print(edgetpu.list_edge_tpus())'
```

### 4. Enable Systemd Services

Install and start services:
```bash
sudo bash scripts/enable_services.sh
```

This will:
- Copy repo to `/opt/edison`
- Create `edison` system user
- Install systemd unit files
- Enable and start all three services

### 5. Verify Installation

Run the doctor script:
```bash
bash scripts/doctor.sh
```

Check service status:
```bash
sudo systemctl status edison-coral
sudo systemctl status edison-core
sudo systemctl status edison-comfyui
```

Test health endpoints:
```bash
curl http://127.0.0.1:8808/health  # Coral
curl http://127.0.0.1:8811/health  # Core
curl http://localhost:8188         # ComfyUI
```

## Usage

### Access ComfyUI

Open browser to: `http://[server-ip]:8188`

### Use EDISON Chat Node

1. In ComfyUI, add a new node (Right-click → Add Node)
2. Navigate to: **EDISON → EDISON Chat**
3. Configure inputs:
   - **text**: Your message/prompt
   - **mode**: auto, chat, reasoning, agent, or code
   - **remember**: Store conversation in memory (true/false)
   - **timeout_seconds**: Request timeout (default 120)
4. Connect output to other nodes or view directly

### Test Chat API Directly

```bash
curl -X POST http://127.0.0.1:8811/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain how transformers work",
    "mode": "reasoning",
    "remember": true
  }'
```

### Test Intent Classification

```bash
curl -X POST http://127.0.0.1:8808/intent \
  -H "Content-Type: application/json" \
  -d '{"text": "Generate an image of a sunset"}'
```

## Viewing Logs

```bash
# Real-time logs
journalctl -u edison-coral -f
journalctl -u edison-core -f
journalctl -u edison-comfyui -f

# Last 50 lines
journalctl -u edison-core -n 50

# All EDISON services
journalctl -u "edison-*" -f
```

## Service Management

```bash
# Restart services
sudo systemctl restart edison-coral
sudo systemctl restart edison-core
sudo systemctl restart edison-comfyui

# Stop services
sudo systemctl stop edison-coral
sudo systemctl stop edison-core
sudo systemctl stop edison-comfyui

# Disable autostart
sudo systemctl disable edison-coral
sudo systemctl disable edison-core
sudo systemctl disable edison-comfyui

# Check status
sudo systemctl status edison-core
```

## Troubleshooting

### Models Not Loading

Check logs:
```bash
journalctl -u edison-core -n 100
```

Verify model files exist:
```bash
ls -lh /opt/edison/models/llm/
```

### Service Won't Start

Check for port conflicts:
```bash
sudo netstat -tlnp | grep -E '8808|8811|8188'
```

Check permissions:
```bash
ls -la /opt/edison
sudo chown -R edison:edison /opt/edison
```

### GPU Not Detected

Check NVIDIA drivers:
```bash
nvidia-smi
```

Check CUDA environment in service:
```bash
sudo systemctl edit edison-core
# Add: Environment="CUDA_VISIBLE_DEVICES=0,1,2"
```

### Coral TPU Not Working

Check device:
```bash
ls -l /dev/apex_0
dmesg | grep apex
lsmod | grep apex
```

EDISON will work without Coral, using heuristic intent classification.

## Development Mode

To run locally without systemd:

```bash
# Activate venv
source .venv/bin/activate

# Terminal 1 - Coral service
python -m uvicorn services.edison_coral.service:app --host 127.0.0.1 --port 8808

# Terminal 2 - Core service
python -m uvicorn services.edison_core.app:app --host 127.0.0.1 --port 8811

# Terminal 3 - ComfyUI
cd ComfyUI
python main.py --listen 0.0.0.0 --port 8188
```

## Configuration

Edit `/opt/edison/config/edison.yaml`:

```yaml
edison:
  core:
    host: "127.0.0.1"
    port: 8811
    models_path: "models/llm"
    fast_model: "qwen2.5-14b-instruct-q4_k_m.gguf"
    deep_model: "qwen2.5-72b-instruct-q4_k_m.gguf"
  
  coral:
    host: "127.0.0.1"
    port: 8808
```

Restart services after changes:
```bash
sudo systemctl restart edison-core edison-coral
```

## Project Structure

```
EDISON-ComfyUI/
├── config/              # Configuration files
├── services/
│   ├── edison_core/     # LLM service (FastAPI)
│   ├── edison_coral/    # Intent service (FastAPI)
│   └── systemd/         # Service unit files
├── scripts/             # Setup and maintenance scripts
├── models/              # Model storage
│   ├── llm/            # GGUF models
│   ├── qdrant/         # Vector DB
│   └── coral/          # Edge TPU models
└── ComfyUI/
    └── custom_nodes/
        └── edison_nodes/  # EDISON custom node
```

## License

See LICENSE file.

## Support

For issues, please check:
1. Run `bash scripts/doctor.sh`
2. Check service logs
3. Review COPILOT_SPEC.md for detailed requirements