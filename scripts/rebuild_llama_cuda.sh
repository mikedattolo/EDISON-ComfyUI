#!/bin/bash
# Rebuild llama-cpp-python with CUDA support for EDISON

set -e

echo "=========================================="
echo "EDISON - Rebuild llama-cpp-python with CUDA"
echo "=========================================="
echo ""

# Check if running in venv
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠ Not in virtual environment"
    echo "Activating /opt/edison/.venv..."
    source /opt/edison/.venv/bin/activate
fi

echo "Current environment: $VIRTUAL_ENV"
echo ""

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "✗ nvidia-smi not found!"
    echo "Please install NVIDIA drivers first"
    exit 1
fi

echo "✓ NVIDIA drivers found"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Check CUDA toolkit
if [ ! -d "/usr/local/cuda" ] && [ ! -d "/usr/lib/cuda" ]; then
    echo "⚠ CUDA toolkit not found in standard locations"
    echo "llama-cpp-python will try to find CUDA automatically"
fi

echo "Uninstalling current llama-cpp-python..."
pip uninstall llama-cpp-python -y
echo ""

echo "Rebuilding llama-cpp-python with CUDA support..."
echo "This will take 5-10 minutes..."
echo ""

# Build with CUDA
CMAKE_ARGS="-DLLAMA_CUDA=on" \
FORCE_CMAKE=1 \
pip install llama-cpp-python --force-reinstall --no-cache-dir --verbose

echo ""
echo "=========================================="
echo "✓ llama-cpp-python rebuilt with CUDA"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Restart edison-core service:"
echo "   sudo systemctl restart edison-core"
echo ""
echo "2. Check logs for GPU usage:"
echo "   sudo journalctl -u edison-core -f"
echo ""
echo "3. Look for messages like:"
echo "   - 'ggml_cuda_init: found X CUDA devices'"
echo "   - 'llm_load_tensors: using CUDA'"
echo ""
