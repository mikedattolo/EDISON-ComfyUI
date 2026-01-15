#!/bin/bash
# EDISON System Health Check
# Diagnostic script to verify all components

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== EDISON System Doctor ==="
echo "Checking system health and configuration..."
echo ""

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

check_pass() {
    echo "✓ PASS: $1"
    ((PASS_COUNT++))
}

check_fail() {
    echo "✗ FAIL: $1"
    echo "  Fix: $2"
    ((FAIL_COUNT++))
}

check_warn() {
    echo "⚠ WARN: $1"
    echo "  Note: $2"
    ((WARN_COUNT++))
}

# Check 1: NVIDIA GPU
echo "[1/12] NVIDIA GPU"
if command -v nvidia-smi &>/dev/null; then
    if nvidia-smi &>/dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        check_pass "nvidia-smi working ($GPU_COUNT GPUs detected)"
    else
        check_fail "nvidia-smi command failed" "Install NVIDIA drivers or check GPU connection"
    fi
else
    check_fail "nvidia-smi not found" "Install NVIDIA drivers: sudo apt install nvidia-driver-<version>"
fi

# Check 2: Coral TPU
echo "[2/12] Coral Edge TPU"
if [ -e /dev/apex_0 ]; then
    check_pass "Coral TPU device found at /dev/apex_0"
else
    check_warn "Coral TPU not found at /dev/apex_0" "Run scripts/install_coral.sh and reboot if you have a Coral TPU"
fi

# Check 3: Python version
echo "[3/12] Python version"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"; then
    check_pass "Python $PYTHON_VERSION (>= 3.10)"
else
    check_warn "Python $PYTHON_VERSION" "Python 3.10+ recommended"
fi

# Check 4: Virtual environment
echo "[4/12] Virtual environment"
if [ -d "$REPO_ROOT/.venv" ]; then
    check_pass "Virtual environment exists at $REPO_ROOT/.venv"
else
    check_fail "No virtual environment found" "Run: bash scripts/setup_ubuntu.sh"
fi

# Check 5: Python packages
echo "[5/12] Python packages"
if [ -d "$REPO_ROOT/.venv" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
    
    MISSING_PACKAGES=()
    python3 -c "import fastapi" 2>/dev/null || MISSING_PACKAGES+=("fastapi")
    python3 -c "import uvicorn" 2>/dev/null || MISSING_PACKAGES+=("uvicorn")
    python3 -c "import llama_cpp" 2>/dev/null || MISSING_PACKAGES+=("llama-cpp-python")
    python3 -c "import qdrant_client" 2>/dev/null || MISSING_PACKAGES+=("qdrant-client")
    python3 -c "import sentence_transformers" 2>/dev/null || MISSING_PACKAGES+=("sentence-transformers")
    
    if [ ${#MISSING_PACKAGES[@]} -eq 0 ]; then
        check_pass "All required Python packages installed"
    else
        check_fail "Missing packages: ${MISSING_PACKAGES[*]}" "Activate venv and run: pip install -r requirements.txt"
    fi
    
    deactivate 2>/dev/null || true
fi

# Check 6: LLM Models
echo "[6/12] LLM Models"
MODELS_FOUND=0
[ -f "$REPO_ROOT/models/llm/qwen2.5-14b-instruct-q4_k_m.gguf" ] && ((MODELS_FOUND++))
[ -f "$REPO_ROOT/models/llm/qwen2.5-72b-instruct-q4_k_m.gguf" ] && ((MODELS_FOUND++))

if [ $MODELS_FOUND -eq 2 ]; then
    check_pass "Both LLM models found (fast + deep)"
elif [ $MODELS_FOUND -eq 1 ]; then
    check_warn "Only 1 LLM model found" "Place GGUF models in $REPO_ROOT/models/llm/"
else
    check_warn "No LLM models found" "Download GGUF models to $REPO_ROOT/models/llm/"
fi

# Check 7: ComfyUI
echo "[7/12] ComfyUI"
if [ -d "$REPO_ROOT/ComfyUI" ]; then
    if [ -f "$REPO_ROOT/ComfyUI/main.py" ]; then
        check_pass "ComfyUI installed"
    else
        check_fail "ComfyUI directory incomplete" "Re-run scripts/setup_ubuntu.sh"
    fi
else
    check_fail "ComfyUI not found" "Run: bash scripts/setup_ubuntu.sh"
fi

# Check 8: EDISON Custom Node
echo "[8/12] EDISON Custom Node"
if [ -f "$REPO_ROOT/ComfyUI/custom_nodes/edison_nodes/edison_chat_node.py" ]; then
    check_pass "EDISON custom node installed"
else
    check_fail "EDISON custom node missing" "Ensure edison_nodes is in ComfyUI/custom_nodes/"
fi

# Check 9-11: Service ports
echo "[9/12] Port 8808 (edison-coral)"
if nc -z 127.0.0.1 8808 2>/dev/null; then
    check_pass "Port 8808 responding"
else
    check_warn "Port 8808 not responding" "Start service: sudo systemctl start edison-coral"
fi

echo "[10/12] Port 8811 (edison-core)"
if nc -z 127.0.0.1 8811 2>/dev/null; then
    check_pass "Port 8811 responding"
else
    check_warn "Port 8811 not responding" "Start service: sudo systemctl start edison-core"
fi

echo "[11/12] Port 8188 (ComfyUI)"
if nc -z 127.0.0.1 8188 2>/dev/null; then
    check_pass "Port 8188 responding"
else
    check_warn "Port 8188 not responding" "Start service: sudo systemctl start edison-comfyui"
fi

# Check 12: Systemd services
echo "[12/12] Systemd services"
if systemctl list-unit-files | grep -q "edison-"; then
    ENABLED=0
    systemctl is-enabled edison-coral &>/dev/null && ((ENABLED++))
    systemctl is-enabled edison-core &>/dev/null && ((ENABLED++))
    systemctl is-enabled edison-comfyui &>/dev/null && ((ENABLED++))
    
    if [ $ENABLED -eq 3 ]; then
        check_pass "All 3 services enabled"
    else
        check_warn "$ENABLED/3 services enabled" "Run: sudo bash scripts/enable_services.sh"
    fi
else
    check_fail "No EDISON systemd services found" "Run: sudo bash scripts/enable_services.sh"
fi

# Summary
echo ""
echo "=== Summary ==="
echo "✓ Passed: $PASS_COUNT"
echo "⚠ Warnings: $WARN_COUNT"
echo "✗ Failed: $FAIL_COUNT"
echo ""

if [ $FAIL_COUNT -eq 0 ] && [ $WARN_COUNT -eq 0 ]; then
    echo "🎉 All checks passed! EDISON is ready."
    exit 0
elif [ $FAIL_COUNT -eq 0 ]; then
    echo "⚠ System mostly ready, but some warnings exist."
    exit 0
else
    echo "✗ System has failures that need attention."
    exit 1
fi
