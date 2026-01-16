# EDISON Installation Guide

Complete guide for installing EDISON on Ubuntu 22.04/24.04 LTS.

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/mikedattolo/EDISON-ComfyUI.git
cd EDISON-ComfyUI

# 2. Check system requirements
./scripts/check_system.sh

# 3. Run setup
./scripts/setup_ubuntu.sh

# 4. Download models
python3 scripts/download_models.py

# 5. Install as system services
sudo ./scripts/enable_services.sh

# 6. Access web UI
# Open browser to: http://YOUR_IP:8080
```

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 22.04 or 24.04 LTS
- **CPU**: 8+ cores
- **RAM**: 32GB (64GB+ recommended for 72B model)
- **Storage**: 50GB+ free (100GB+ recommended)
- **Python**: 3.9 or higher
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but highly recommended)

### Recommended Configuration
- **RAM**: 64GB+
- **Storage**: 200GB+ on fast NVMe SSD
- **GPU**: NVIDIA RTX 3090/4080/4090 or better
- **CUDA**: 12.0+
- **Optional**: Google Coral M.2 Edge TPU for intent classification

## Pre-Installation Steps

### 1. Expand Disk Space (if needed)

If you're using LVM and have a small root partition:

```bash
# Check current space
df -h /

# If root partition is small but you have a large disk, expand it:
sudo lvextend -l +100%FREE /dev/ubuntu-vg/ubuntu-lv
sudo resize2fs /dev/ubuntu-vg/ubuntu-lv

# Verify
df -h /
```

### 2. Install NVIDIA Drivers and CUDA (if you have NVIDIA GPU)

```bash
# Install NVIDIA driver
sudo ubuntu-drivers install

# Reboot
sudo reboot

# Verify
nvidia-smi

# Install CUDA Toolkit (if not already installed)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6
```

### 3. Install Google Coral TPU Drivers (Optional)

If you have a Google Coral Edge TPU:

```bash
# Automated installation with kernel 6.8+ patches
sudo ./scripts/fix_coral_tpu.sh

# Reboot after installation
sudo reboot

# Verify
ls -l /dev/apex_0
lsmod | grep apex
```

## Installation

### Step 1: System Requirements Check

```bash
./scripts/check_system.sh
```

This will verify:
- OS version
- Python version
- Disk space
- RAM
- NVIDIA GPU
- CUDA installation
- Build tools
- Coral TPU (if present)
- Network connectivity

### Step 2: Run Setup Script

```bash
./scripts/setup_ubuntu.sh
```

This will:
- Install system dependencies
- Create Python virtual environment
- Install Python packages
- Clone ComfyUI
- Set up directory structure
- Install EDISON custom nodes

### Step 3: Download Models

```bash
# Activate virtual environment
source .venv/bin/activate

# Download models interactively
python3 scripts/download_models.py
```

Models to download:
- **Fast model** (required): qwen2.5-14b-instruct-q4_k_m.gguf (~9GB)
- **Deep model** (optional): qwen2.5-72b-instruct-q4_k_m.gguf (~42GB)

Alternative: Download manually from Hugging Face:
```bash
# Install huggingface-cli
pip install huggingface-hub[cli]

# Download models
huggingface-cli download TheBloke/Qwen2.5-14B-Instruct-GGUF qwen2.5-14b-instruct-q4_k_m.gguf --local-dir models/llm/
huggingface-cli download TheBloke/Qwen2.5-72B-Instruct-GGUF qwen2.5-72b-instruct-q4_k_m.gguf --local-dir models/llm/
```

### Step 4: Configure Services

Edit configuration if needed:

```bash
nano config/edison.yaml
```

Key settings:
- `models_path`: Path to model directory
- `fast_model`: Filename of 14B model
- `deep_model`: Filename of 72B model
- Service ports (coral: 8808, core: 8811, web: 8080, comfyui: 8188)

### Step 5: Install as System Services

```bash
sudo ./scripts/enable_services.sh
```

This will:
- Copy EDISON to `/opt/edison`
- Install systemd service files
- Enable auto-start on boot
- Start all services

Services installed:
- `edison-coral.service` - Intent classification (port 8808)
- `edison-core.service` - Main AI service (port 8811)
- `edison-web.service` - Web UI (port 8080)
- `edison-comfyui.service` - ComfyUI (port 8188)

### Step 6: Verify Installation

```bash
# Check service status
sudo systemctl status edison-coral
sudo systemctl status edison-core
sudo systemctl status edison-web
sudo systemctl status edison-comfyui

# Check logs
sudo journalctl -u edison-core -f

# Test API
curl http://localhost:8811/health
curl http://localhost:8080/health

# Open web UI
# Browser: http://YOUR_IP:8080
```

## Network Access

All services are configured to bind to `0.0.0.0` by default, allowing access from other devices on your network.

Access URLs:
- **Web UI**: http://YOUR_IP:8080
- **Core API**: http://YOUR_IP:8811
- **ComfyUI**: http://YOUR_IP:8188
- **Coral API**: http://YOUR_IP:8808

To restrict to localhost only, edit the systemd service files:
```bash
sudo nano /opt/edison/services/systemd/edison-web.service
# Change --host 0.0.0.0 to --host 127.0.0.1
sudo systemctl daemon-reload
sudo systemctl restart edison-web
```

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions to common issues.

### Quick Fixes

**Service won't start:**
```bash
sudo journalctl -u edison-core -n 50
```

**Models not loading:**
```bash
ls -lh /opt/edison/models/llm/
# Ensure .gguf files are present and readable
```

**GPU not detected:**
```bash
nvidia-smi
# If no output, reinstall drivers:
sudo ubuntu-drivers install
sudo reboot
```

**Web UI not accessible:**
```bash
# Check if service is running
sudo systemctl status edison-web
# Check firewall
sudo ufw status
sudo ufw allow 8080/tcp
```

## Updating

```bash
cd /workspaces/EDISON-ComfyUI
git pull
source .venv/bin/activate
pip install -r requirements.txt --upgrade

# If system services are installed:
sudo ./scripts/enable_services.sh
sudo systemctl daemon-reload
sudo systemctl restart edison-core edison-web edison-comfyui edison-coral
```

## Uninstallation

```bash
# Stop and disable services
sudo systemctl stop edison-coral edison-core edison-web edison-comfyui
sudo systemctl disable edison-coral edison-core edison-web edison-comfyui

# Remove service files
sudo rm /etc/systemd/system/edison-*.service
sudo systemctl daemon-reload

# Remove installation (optional)
sudo rm -rf /opt/edison

# Remove user (optional)
sudo userdel edison
```

## Next Steps

- See [USER_GUIDE.md](USER_GUIDE.md) for usage instructions
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- See [DEVELOPMENT.md](DEVELOPMENT.md) for development guide
