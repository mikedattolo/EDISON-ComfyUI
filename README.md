# EDISON-ComfyUI

<div align="center">

<img src="logo.png" alt="EDISON Logo" width="200"/>

**Your Complete Offline AI Platform**

*Enterprise-grade local LLM system with modern web UI, vision capabilities, RAG memory, and ComfyUI integration*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Ubuntu 22.04+](https://img.shields.io/badge/ubuntu-22.04%2B-orange.svg)](https://ubuntu.com/)
[![CUDA 12+](https://img.shields.io/badge/CUDA-12%2B-green.svg)](https://developer.nvidia.com/cuda-downloads)

[Features](#features) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Troubleshooting](#troubleshooting)

</div>

---

## 🎯 Why EDISON?

**EDISON** (Enhanced Distributed Intelligence System for Offline Networks) is a production-ready, fully offline AI platform that brings enterprise-level capabilities to your local infrastructure:

- ✅ **100% Private** - All processing happens on your hardware
- ✅ **No API Costs** - One-time setup, unlimited usage
- ✅ **Multi-Modal** - Text, images, code, and vision understanding
- ✅ **Production Ready** - Systemd services, logging, monitoring
- ✅ **Multi-GPU Support** - Efficient tensor splitting across GPUs
- ✅ **Memory System** - RAG-powered long-term context retention

---

## ✨ Features

### 🚀 **Modern Web Interface**
- **Claude/ChatGPT-inspired UI** with real-time streaming responses
- **Conversation Management** - Save, load, and organize chat histories
- **File Upload Support** - Attach documents and images to conversations
- **Multiple AI Modes** - Chat, Deep Thinking, Reasoning, Code, Agent
- **Auto-Intent Detection** - Automatically selects optimal model and mode
- **Hardware Monitoring** - Real-time GPU, CPU, and memory stats
- **Work Mode** - Dedicated interface for complex multi-step tasks
- **Settings Panel** - Configure models, memory, and behavior

### 🧠 **Powerful AI Models**
- **Fast Mode (Qwen 2.5 14B)** - Instant responses for quick questions
- **Deep Mode (Qwen 2.5 72B)** - Comprehensive analysis and reasoning
- **Vision Model (LLaVA 1.6)** - Image understanding and description
- **Code Assistant** - Specialized for programming and technical tasks
- **Agent Mode** - Web search, tool use, and multi-step problem solving
- **Full GPU Acceleration** - CUDA-optimized inference with tensor splitting

### 🎨 **Image Generation**
- **ComfyUI Integration** - Professional node-based workflow system
- **Custom EDISON Nodes** - Direct chat interface within ComfyUI
- **Workflow Library** - Pre-configured templates for common tasks
- **Real-time Generation** - Streaming progress updates

### 🧩 **Advanced Capabilities**
- **RAG Memory System** - Vector-based long-term context with Qdrant
- **Web Search** - Agent mode with DuckDuckGo integration
- **File Processing** - Upload and analyze documents, images, and code
- **Intent Classification** - Optional Google Coral TPU acceleration
- **Multi-GPU Support** - Automatic load balancing across devices
- **Conversation Memory** - Remember facts, preferences, and context
- **Smart Prompting** - Context-aware system prompts per mode

### 🔌 **Enterprise Deployment**
- **One-Command Install** - Automated setup with dependency management
- **Systemd Services** - Auto-start on boot with crash recovery
- **Production Logging** - Structured logs with journalctl integration
- **Health Monitoring** - Endpoints for uptime and status checks
- **Network Access** - Secure LAN-accessible from any device
- **Zero Configuration** - Intelligent defaults with optional customization

---

## 🚀 Quick Start

### Prerequisites
- Ubuntu 22.04+ or Debian-based Linux
- NVIDIA GPU with 12GB+ VRAM (16GB+ recommended)
- CUDA 12.0+ drivers installed
- 100GB+ free disk space for models

### Installation (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/mikedattolo/EDISON-ComfyUI.git
cd EDISON-ComfyUI

# 2. Check system requirements
./scripts/check_system.sh

# 3. Run automated installation (installs all dependencies)
./scripts/setup_ubuntu.sh

# 4. Download AI models (~50GB)
python3 scripts/download_models.py

# 5. Install as system services (auto-start on boot)
sudo ./scripts/enable_services.sh

# 6. Access web interface
# Open browser: http://YOUR_SERVER_IP:8080
```

### What Gets Installed

| Service | Port | Description | Auto-Start |
|---------|------|-------------|------------|
| **edison-web** | 8080 | Modern web UI for chat | ✅ Yes |
| **edison-core** | 8811 | LLM inference API | ✅ Yes |
| **edison-coral** | 8808 | Intent classification | ✅ Yes |
| **edison-comfyui** | 8188 | ComfyUI image generation | ✅ Yes |

### First Steps

1. **Open Web UI** - Navigate to `http://YOUR_IP:8080`
2. **Try Chat Mode** - Ask: "What can you help me with?"
3. **Test Deep Mode** - Ask: "Explain quantum computing in detail"
4. **Upload Files** - Click 📎 to attach documents or images
5. **Try Vision** - Upload an image and ask "What's in this image?"
6. **Enable Memory** - Toggle "Remember" to store conversations

---

## 📊 EDISON vs Cloud AI

| Feature | EDISON | ChatGPT/Claude | Local Alternatives |
|---------|--------|----------------|-------------------|
| **Privacy** | 100% offline | Cloud-based | Varies |
| **Cost** | One-time hardware | $20-200/month | Varies |
| **Latency** | <1s local | 1-5s network | <1s local |
| **Customization** | Full control | Limited | Limited |
| **Multi-modal** | Text + Vision + Image Gen | Text + Vision | Often separate |
| **Memory/RAG** | Built-in Qdrant | Token-limited | Often missing |
| **Production Ready** | Systemd services | N/A | Manual setup |
| **Multi-GPU** | Tensor splitting | N/A | Rarely supported |
| **Web Interface** | Modern, responsive | Excellent | Basic |
| **API Access** | OpenAI-compatible | Yes | Varies |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Web Browser (Any Device)           │
│            http://YOUR_IP:8080                  │
└────────────────┬────────────────────────────────┘
                 │
       ┌─────────┴──────────┐
       │                    │
┌──────▼───────┐   ┌────────▼────────┐
│  EDISON Web  │   │    ComfyUI      │
│   (8080)     │   │    (8188)       │
│  - Chat UI   │   │  - Node Editor  │
│  - File Up   │   │  - Workflows    │
│  - Settings  │   │  - Generation   │
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
│   │  - LLaVA 1.6 (Vision)       │   │
│   │  - llama-cpp-python + CUDA  │   │
│   └───────────┬─────────────────┘   │
│               │                     │
│   ┌───────────▼─────────────────┐   │
│   │   RAG System (Qdrant)       │   │
│   │  - Vector embeddings        │   │
│   │  - Conversation memory      │   │
│   │  - Document storage         │   │
│   └───────────┬─────────────────┘   │
│               │                     │
│   ┌───────────▼─────────────────┐   │
│   │   Tool System               │   │
│   │  - Web Search (DuckDuckGo)  │   │
│   │  - Code execution           │   │
│   └─────────────────────────────┘   │
└──────┬──────────────────────────────┘
       │
┌──────▼───────────────────────────────┐
│   EDISON Coral (8808)                │
│   Intent Classification              │
│   - Auto mode detection             │
│   - Coral TPU acceleration (opt)    │
└──────────────────────────────────────┘
```

---

## 💻 System Requirements

### Minimum Configuration
| Component | Requirement | Notes |
|-----------|-------------|-------|
| **OS** | Ubuntu 22.04+ | Debian-based Linux |
| **CPU** | 8+ cores | Intel/AMD x86_64 |
| **RAM** | 32GB | For 14B models |
| **Storage** | 100GB free | For models + system |
| **GPU** | 12GB VRAM | RTX 3060 12GB minimum |
| **Python** | 3.9+ | Installed by setup script |

### Recommended Configuration
| Component | Requirement | Notes |
|-----------|-------------|-------|
| **RAM** | 64GB+ | For 72B deep model |
| **Storage** | 200GB+ NVMe | Faster model loading |
| **GPU** | 16GB+ VRAM | RTX 4080/4090/A5000 |
| **CUDA** | 12.0+ | Latest drivers |
| **Network** | Gigabit LAN | For remote access |
| **Optional** | Coral M.2 TPU | Intent classification |

### Multi-GPU Support
EDISON automatically splits models across multiple GPUs:
- **3x GPU Setup**: 50% / 25% / 25% tensor split
- **2x GPU Setup**: 60% / 40% tensor split  
- **1x GPU**: Full model on single GPU

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