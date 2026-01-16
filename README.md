# EDISON-ComfyUI

<div align="center">

**Fully offline AI system with modern web UI, local LLMs, and ComfyUI integration**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Ubuntu 22.04+](https://img.shields.io/badge/ubuntu-22.04%2B-orange.svg)](https://ubuntu.com/)

[Features](#features) • [Installation](#quick-start) • [Documentation](#documentation) • [Troubleshooting](#troubleshooting)

</div>

---

## Features

### 🚀 **Modern Web UI**
- Clean, responsive interface inspired by Claude/ChatGPT
- Real-time chat with conversation history
- Multiple AI modes: Chat, Deep Thinking, Code Assistant, Agent
- Auto-intent detection
- Settings management and health monitoring

### 🧠 **Powerful AI Models**
- **Fast mode**: Qwen 2.5 14B for quick responses
- **Deep mode**: Qwen 2.5 72B for detailed analysis
- **Code mode**: Specialized for programming tasks
- **Agent mode**: Tool-using capabilities
- Full GPU acceleration with CUDA

### 🔌 **Easy Deployment**
- One-command installation
- Systemd services for auto-start
- Network-accessible from any device
- Production-ready configuration

### 🛠️ **Advanced Features**
- RAG (Retrieval Augmented Generation) with Qdrant
- Intent classification with optional Coral TPU
- ComfyUI integration for image generation
- Multi-GPU support
- Comprehensive logging and monitoring

---

## Quick Start

```bash
# 1. Check system requirements
git clone https://github.com/mikedattolo/EDISON-ComfyUI.git
cd EDISON-ComfyUI
./scripts/check_system.sh

# 2. Run installation
./scripts/setup_ubuntu.sh

# 3. Download AI models
python3 scripts/download_models.py

# 4. Install as system services
sudo ./scripts/enable_services.sh

# 5. Access web UI
# Open browser: http://YOUR_IP:8080
```

**That's it!** EDISON is now running with:
- 🌐 Web UI on port **8080**
- 🤖 API on port **8811**  
- 🎨 ComfyUI on port **8188**
- 🧭 Intent API on port **8808**

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Web Browser                    │
│            http://YOUR_IP:8080                  │
└────────────────┬────────────────────────────────┘
                 │
       ┌─────────┴──────────┐
       │                    │
┌──────▼───────┐   ┌────────▼────────┐
│  EDISON Web  │   │    ComfyUI      │
│   (8080)     │   │    (8188)       │
└──────┬───────┘   └────────┬────────┘
       │                    │
       │           ┌────────▼────────┐
       │           │ EDISON Nodes    │
       │           │ (Custom Nodes)  │
       │           └────────┬────────┘
       │                    │
       ├────────────────────┘
       │
┌──────▼──────────────────────────────┐
│       EDISON Core (8811)            │
│   ┌─────────────────────────────┐   │
│   │   LLM Inference Engine      │   │
│   │  - Qwen 2.5 14B (Fast)      │   │
│   │  - Qwen 2.5 72B (Deep)      │   │
│   │  - llama-cpp-python         │   │
│   └───────────┬─────────────────┘   │
│               │                     │
│   ┌───────────▼─────────────────┐   │
│   │   RAG System (Qdrant)       │   │
│   │  - Vector embeddings        │   │
│   │  - Conversation memory      │   │
│   └─────────────────────────────┘   │
└──────┬──────────────────────────────┘
       │
┌──────▼───────────────────────────────┐
│   EDISON Coral (8808)                │
│   Intent Classification              │
│   - Heuristic patterns (default)    │
│   - Coral TPU acceleration (opt)    │
└──────────────────────────────────────┘
```

---

## System Requirements

### Minimum Configuration
| Component | Requirement |
|-----------|-------------|
| **OS** | Ubuntu 22.04 or 24.04 LTS |
| **CPU** | 8+ cores |
| **RAM** | 32GB |
| **Storage** | 50GB+ free |
| **Python** | 3.9+ |

### Recommended Configuration
| Component | Requirement |
|-----------|-------------|
| **RAM** | 64GB+ |
| **Storage** | 200GB+ NVMe SSD |
| **GPU** | NVIDIA RTX 3090/4080/4090 (8GB+ VRAM) |
| **CUDA** | 12.0+ |
| **Optional** | Google Coral M.2 Edge TPU |

---

## Installation

### Automated Installation (Recommended)

```bash
# Clone repository
git clone https://github.com/mikedattolo/EDISON-ComfyUI.git
cd EDISON-ComfyUI

# Run system check
./scripts/check_system.sh

# Install everything
./scripts/setup_ubuntu.sh

# Download models
python3 scripts/download_models.py

# Deploy as services
sudo ./scripts/enable_services.sh
```

### Services Installed

After installation, the following services will be running:

| Service | Port | Description |
|---------|------|-------------|
| `edison-web` | 8080 | Modern web UI for chat interface |
| `edison-core` | 8811 | Main AI service with LLM inference |
| `edison-coral` | 8808 | Intent classification service |
| `edison-comfyui` | 8188 | ComfyUI for image generation |

All services auto-start on boot and are accessible from network.

---

## Usage

### Web UI

Open your browser to **http://YOUR_IP:8080**

Features:
- 💬 **Chat modes**: Auto, Chat (14B), Deep (72B), Code, Agent
- 📝 **Conversation history** saved in browser
- ⚙️ **Settings panel** for configuration
- 🔍 **System status** monitoring
- 🎨 **Modern dark theme**

### API Usage

#### Chat Endpoint

```bash
curl -X POST http://YOUR_IP:8811/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain quantum computing",
    "mode": "deep",
    "remember": true
  }'
```

#### Intent Classification

```bash
curl -X POST http://YOUR_IP:8808/intent \
  -H "Content-Type: application/json" \
  -d '{"text": "Generate an image of a sunset"}'
```

#### Health Check

```bash
curl http://YOUR_IP:8811/health
curl http://YOUR_IP:8080/health
```

### ComfyUI Integration

1. Open ComfyUI at **http://YOUR_IP:8188**
2. Right-click → **Add Node** → **EDISON** → **EDISON Chat**
3. Configure:
   - **text**: Your message
   - **mode**: auto, chat, deep, code, or agent
   - **remember**: true/false
   - **timeout_seconds**: 120 (default)

### Modes Explained

| Mode | Model | Use Case |
|------|-------|----------|
| **Auto** | Detected | Automatically selects best mode based on intent |
| **Chat** | 14B | Quick responses, casual conversation |
| **Deep** | 72B | Detailed analysis, complex reasoning |
| **Code** | 72B | Programming assistance, debugging |
| **Agent** | 72B | Tool-using, multi-step tasks |

---

## Service Management

```bash
# Check status
sudo systemctl status edison-core edison-web

# Restart services
sudo systemctl restart edison-core

# View logs
sudo journalctl -u edison-core -f

# Stop services
sudo systemctl stop edison-core edison-web

# Disable auto-start
sudo systemctl disable edison-core
```

---

## Documentation

📚 **Comprehensive guides available:**

- **[INSTALL.md](INSTALL.md)** - Detailed installation guide with all options
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Solutions to common issues
- **[COPILOT_SPEC.md](COPILOT_SPEC.md)** - Original specification
- **[IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)** - Implementation notes

---

## Troubleshooting

### Common Issues

**❌ Disk space full**
```bash
# Expand LVM partition
sudo lvextend -l +100%FREE /dev/ubuntu-vg/ubuntu-lv
sudo resize2fs /dev/ubuntu-vg/ubuntu-lv
```

**❌ Python 3.12 + Coral TPU**
- Coral `pycoral` library incompatible with Python 3.12
- System automatically uses heuristic classification instead
- Or use separate Python 3.11 venv for coral service

**❌ Service won't start**
```bash
# Check logs for specific error
sudo journalctl -u edison-core -n 50

# Common fix: Ensure models are downloaded
ls -lh /opt/edison/models/llm/
```

**❌ Can't access from network**
```bash
# Check firewall
sudo ufw allow 8080/tcp
sudo ufw allow 8811/tcp
sudo ufw allow 8188/tcp
```

**❌ Coral TPU kernel module errors**
```bash
# Automated fix for kernel 6.8+ API changes
sudo ./scripts/fix_coral_tpu.sh
sudo reboot
```

**❌ GPU not working**
```bash
# Verify CUDA
nvidia-smi
nvcc --version

# Check PyTorch GPU support
python3 -c "import torch; print(torch.cuda.is_available())"
```

See **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** for detailed solutions.

---

## Updating

```bash
cd /workspaces/EDISON-ComfyUI
git pull
source .venv/bin/activate
pip install -r requirements.txt --upgrade

# If services installed:
sudo ./scripts/enable_services.sh
sudo systemctl daemon-reload
sudo systemctl restart edison-core edison-web edison-comfyui
```

---

## Uninstallation

```bash
# Stop and remove services
sudo systemctl stop edison-coral edison-core edison-web edison-comfyui
sudo systemctl disable edison-coral edison-core edison-web edison-comfyui
sudo rm /etc/systemd/system/edison-*.service
sudo systemctl daemon-reload

# Remove installation
sudo rm -rf /opt/edison

# Remove user (optional)
sudo userdel edison
```

---

## Development

### Local Development Mode

Run services locally without systemd:

```bash
# Terminal 1: Coral service
source .venv/bin/activate
python -m uvicorn services.edison_coral.service:app --host 0.0.0.0 --port 8808

# Terminal 2: Core service
source .venv/bin/activate
python -m uvicorn services.edison_core.app:app --host 0.0.0.0 --port 8811

# Terminal 3: Web UI
source .venv/bin/activate
python -m uvicorn services.edison_web.service:app --host 0.0.0.0 --port 8080

# Terminal 4: ComfyUI
source .venv/bin/activate
cd ComfyUI
python main.py --listen 0.0.0.0 --port 8188
```

### Project Structure

```
EDISON-ComfyUI/
├── services/
│   ├── edison_core/          # Main AI service
│   │   ├── app.py           # FastAPI app with LLM inference
│   │   ├── prompts.py       # System prompts
│   │   ├── rag.py           # RAG with Qdrant
│   │   └── tools.py         # Agent tools
│   ├── edison_coral/        # Intent classification
│   │   └── service.py
│   ├── edison_web/          # Web UI service
│   │   └── service.py
│   └── systemd/             # Service files
├── web/                     # Web UI frontend
│   ├── index.html          # Main HTML
│   ├── styles.css          # Modern styling
│   └── app.js              # Client-side logic
├── ComfyUI/
│   └── custom_nodes/
│       └── edison_nodes/    # EDISON custom nodes
├── scripts/                 # Installation scripts
│   ├── check_system.sh     # System requirements checker
│   ├── setup_ubuntu.sh     # Main setup script
│   ├── fix_coral_tpu.sh    # Coral TPU auto-fixer
│   ├── download_models.py  # Model downloader
│   └── enable_services.sh  # Service installer
├── config/
│   ├── edison.yaml         # Main configuration
│   └── gpu_map.yaml        # GPU assignments
└── models/                  # Model storage
    ├── llm/                # GGUF models
    ├── qdrant/             # Vector database
    └── embeddings/         # Sentence transformers
```

---

## Performance Tips

### For 14B Model Only
If you only have 32GB RAM, use only the fast model:
```yaml
# config/edison.yaml
edison:
  core:
    fast_model: "qwen2.5-14b-instruct-q4_k_m.gguf"
    deep_model: "qwen2.5-14b-instruct-q4_k_m.gguf"  # Same as fast
```

### GPU Optimization
```python
# services/edison_core/app.py
llm_fast = Llama(
    model_path=str(fast_model_path),
    n_ctx=4096,
    n_gpu_layers=-1,      # -1 = offload all layers to GPU
    n_threads=8,          # CPU threads for non-GPU work
    verbose=False
)
```

### Multiple GPUs
```bash
# Set in systemd service
Environment="CUDA_VISIBLE_DEVICES=0,1,2"
```

---

## Security Considerations

### Network Exposure
By default, all services bind to `0.0.0.0` for network access. For localhost-only:

```bash
# Edit service files
sudo nano /etc/systemd/system/edison-web.service
# Change: --host 0.0.0.0 to --host 127.0.0.1
sudo systemctl daemon-reload
sudo systemctl restart edison-web
```

### Firewall
```bash
# Allow specific IPs only
sudo ufw default deny incoming
sudo ufw allow from 192.168.1.0/24 to any port 8080
sudo ufw allow from 192.168.1.0/24 to any port 8811
sudo ufw enable
```

---

## Known Issues

### Python 3.12 + Coral TPU
- **Issue**: `pycoral` library not compatible with Python 3.12
- **Status**: System automatically uses heuristic classification
- **Workaround**: Use separate Python 3.11 venv for coral service
- **Tracking**: Waiting for pycoral Python 3.12 support

### RTX 5060 Ti (Blackwell)
- **Issue**: PyTorch 2.5.1 doesn't support compute capability 12.0
- **Status**: GPU falls back to CPU
- **Workaround**: Use other GPUs or wait for PyTorch 2.6+
- **Tracking**: PyTorch 2.6 will add Blackwell support

### Coral TPU Kernel 6.8+
- **Issue**: gasket driver incompatible with kernel 6.8+ API changes
- **Status**: ✅ Fixed with automated patcher
- **Solution**: Run `sudo ./scripts/fix_coral_tpu.sh`

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

Areas for contribution:
- Additional model support
- UI improvements
- Documentation
- Bug fixes
- Performance optimizations

---

## License

MIT License - see [LICENSE](LICENSE) file for details

---

## Acknowledgments

- **llama-cpp-python** - Fast LLM inference
- **ComfyUI** - Node-based image generation
- **Qwen** - Excellent open-source models
- **Qdrant** - Vector database
- **FastAPI** - Modern web framework
- **Google Coral** - Edge TPU acceleration

---

## Support

- 📖 [Installation Guide](INSTALL.md)
- 🔧 [Troubleshooting Guide](TROUBLESHOOTING.md)
- 🐛 [Issue Tracker](https://github.com/mikedattolo/EDISON-ComfyUI/issues)
- 💬 [Discussions](https://github.com/mikedattolo/EDISON-ComfyUI/discussions)

---

<div align="center">

**Made with ❤️ for the open-source AI community**

[⬆ Back to Top](#edison-comfyui)

</div>

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