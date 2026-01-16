#!/bin/bash
# EDISON-ComfyUI Ubuntu Setup Script
# Production-ready and idempotent setup for EDISON system

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== EDISON-ComfyUI Setup ==="
echo "Repository root: $REPO_ROOT"
echo ""

# Update system
echo "[1/8] Updating system packages..."
sudo apt-get update
sudo apt-get install -y \
    git \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    ninja-build \
    curl \
    wget \
    rsync

echo ""
echo "[2/8] Creating Python virtual environment..."
if [ ! -d "$REPO_ROOT/.venv" ]; then
    python3 -m venv "$REPO_ROOT/.venv"
    echo "✓ Virtual environment created at $REPO_ROOT/.venv"
else
    echo "✓ Virtual environment already exists"
fi

echo ""
echo "[3/8] Installing Python requirements..."
source "$REPO_ROOT/.venv/bin/activate"
pip install --upgrade pip setuptools wheel
pip install -r "$REPO_ROOT/requirements.txt"
echo "✓ Python packages installed"

echo ""
echo "[4/8] Setting up ComfyUI..."
if [ ! -d "$REPO_ROOT/ComfyUI" ] || [ ! -f "$REPO_ROOT/ComfyUI/main.py" ]; then
    echo "Cloning ComfyUI..."
    cd "$REPO_ROOT"
    rm -rf ComfyUI  # Remove if exists but incomplete
    git clone https://github.com/comfyanonymous/ComfyUI.git
    echo "✓ ComfyUI cloned"
    
    # Install ComfyUI requirements
    if [ -f "$REPO_ROOT/ComfyUI/requirements.txt" ]; then
        echo "Installing ComfyUI requirements..."
        pip install -r "$REPO_ROOT/ComfyUI/requirements.txt"
        echo "✓ ComfyUI requirements installed"
    fi
else
    echo "✓ ComfyUI already exists"
fi

echo ""
echo "[5/8] Setting up ComfyUI-Manager..."
MANAGER_PATH="$REPO_ROOT/ComfyUI/custom_nodes/ComfyUI-Manager"
if [ ! -d "$MANAGER_PATH" ]; then
    echo "Cloning ComfyUI-Manager..."
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git "$MANAGER_PATH"
    echo "✓ ComfyUI-Manager cloned"
else
    echo "✓ ComfyUI-Manager already exists"
fi

echo ""
echo "[6/8] Creating model directories..."
mkdir -p "$REPO_ROOT/models/llm"
mkdir -p "$REPO_ROOT/models/qdrant"
mkdir -p "$REPO_ROOT/models/embeddings"
mkdir -p "$REPO_ROOT/models/coral"
echo "✓ Model directories created"

echo ""
echo "[7/8] Installing EDISON custom nodes in ComfyUI..."
EDISON_NODES_SRC="$REPO_ROOT/ComfyUI/custom_nodes/edison_nodes"
if [ -d "$EDISON_NODES_SRC" ]; then
    echo "✓ EDISON nodes already in place"
else
    echo "ERROR: EDISON nodes not found at $EDISON_NODES_SRC"
    echo "This should have been created during repo setup"
    exit 1
fi

echo ""
echo "[8/8] Verifying installation..."
python3 -c "import fastapi, uvicorn, pydantic" && echo "✓ FastAPI packages OK" || echo "✗ FastAPI packages FAILED"
python3 -c "import qdrant_client" && echo "✓ Qdrant client OK" || echo "✗ Qdrant client FAILED"
python3 -c "import sentence_transformers" && echo "✓ Sentence transformers OK" || echo "✗ Sentence transformers FAILED"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "NEXT STEPS:"
echo "1. Place GGUF models in $REPO_ROOT/models/llm/"
echo "   - qwen2.5-14b-instruct-q4_k_m.gguf (fast model)"
echo "   - qwen2.5-72b-instruct-q4_k_m.gguf (deep model)"
echo ""
echo "2. If you have a Coral Edge TPU, run:"
echo "   bash $REPO_ROOT/scripts/install_coral.sh"
echo "   (Note: Coral requires Python <3.12, auto-detected during install)"
echo ""
echo "3. To enable systemd services (requires sudo):"
echo "   bash $REPO_ROOT/scripts/enable_services.sh"
echo ""
echo "4. To test locally without systemd:"
echo "   source $REPO_ROOT/.venv/bin/activate"
echo "   cd $REPO_ROOT"
echo "   python -m uvicorn services.edison_coral.service:app --port 8808 &"
echo "   python -m uvicorn services.edison_core.app:app --port 8811 &"
echo "   cd ComfyUI && python main.py --port 8188"
echo ""
