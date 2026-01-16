#!/bin/bash
# EDISON System Requirements Checker
# Validates system configuration before installation

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== EDISON System Requirements Checker ==="
echo ""

# Check overall pass/fail
ALL_PASSED=true

# 1. Check OS
echo -n "[1/10] Checking OS... "
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" == "ubuntu" ]] && [[ "${VERSION_ID}" == "24.04" || "${VERSION_ID}" == "22.04" ]]; then
        echo -e "${GREEN}✓ Ubuntu ${VERSION_ID}${NC}"
    else
        echo -e "${YELLOW}⚠ ${PRETTY_NAME} (Ubuntu 22.04/24.04 recommended)${NC}"
    fi
else
    echo -e "${RED}✗ Unknown OS${NC}"
    ALL_PASSED=false
fi

# 2. Check Python version
echo -n "[2/10] Checking Python version... "
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
        echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"
        
        # Check if Python 3.12 (Coral incompatibility)
        if [ "$PYTHON_MINOR" -eq 12 ]; then
            echo -e "   ${YELLOW}⚠ Python 3.12 detected: Coral TPU pycoral library incompatible${NC}"
            echo -e "   ${YELLOW}   System will use heuristic intent classification instead${NC}"
        fi
    else
        echo -e "${RED}✗ Python $PYTHON_VERSION (3.9+ required)${NC}"
        ALL_PASSED=false
    fi
else
    echo -e "${RED}✗ Python 3 not found${NC}"
    ALL_PASSED=false
fi

# 3. Check disk space
echo -n "[3/10] Checking disk space... "
AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -ge 100 ]; then
    echo -e "${GREEN}✓ ${AVAILABLE_GB}GB available${NC}"
elif [ "$AVAILABLE_GB" -ge 50 ]; then
    echo -e "${YELLOW}⚠ ${AVAILABLE_GB}GB available (100GB+ recommended)${NC}"
    echo -e "   ${YELLOW}   Minimum 50GB for 14B model, 100GB+ for 72B model${NC}"
else
    echo -e "${RED}✗ Only ${AVAILABLE_GB}GB available (50GB minimum)${NC}"
    echo -e "   ${YELLOW}   Run: sudo lvextend -l +100%FREE /dev/ubuntu-vg/ubuntu-lv && sudo resize2fs /dev/ubuntu-vg/ubuntu-lv${NC}"
    ALL_PASSED=false
fi

# 4. Check RAM
echo -n "[4/10] Checking RAM... "
TOTAL_RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
if [ "$TOTAL_RAM_GB" -ge 64 ]; then
    echo -e "${GREEN}✓ ${TOTAL_RAM_GB}GB RAM${NC}"
elif [ "$TOTAL_RAM_GB" -ge 32 ]; then
    echo -e "${YELLOW}⚠ ${TOTAL_RAM_GB}GB RAM (64GB+ recommended for 72B model)${NC}"
else
    echo -e "${YELLOW}⚠ ${TOTAL_RAM_GB}GB RAM (32GB minimum, 64GB+ recommended)${NC}"
fi

# 5. Check NVIDIA GPU
echo -n "[5/10] Checking NVIDIA GPU... "
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo -e "${GREEN}✓ $GPU_COUNT NVIDIA GPU(s) detected${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
        echo -e "   - $line"
    done
else
    echo -e "${YELLOW}⚠ nvidia-smi not found (GPU acceleration disabled)${NC}"
    echo -e "   ${YELLOW}   Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads${NC}"
fi

# 6. Check CUDA
echo -n "[6/10] Checking CUDA... "
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    echo -e "${GREEN}✓ CUDA $CUDA_VERSION${NC}"
else
    echo -e "${YELLOW}⚠ CUDA not found (GPU acceleration may not work)${NC}"
fi

# 7. Check git
echo -n "[7/10] Checking git... "
if command -v git &> /dev/null; then
    GIT_VERSION=$(git --version | awk '{print $3}')
    echo -e "${GREEN}✓ git $GIT_VERSION${NC}"
else
    echo -e "${RED}✗ git not found (required)${NC}"
    echo -e "   ${YELLOW}   Install: sudo apt-get install git${NC}"
    ALL_PASSED=false
fi

# 8. Check build tools
echo -n "[8/10] Checking build tools... "
if command -v gcc &> /dev/null && command -v make &> /dev/null; then
    GCC_VERSION=$(gcc --version | head -1 | awk '{print $3}')
    echo -e "${GREEN}✓ build-essential installed (gcc $GCC_VERSION)${NC}"
else
    echo -e "${RED}✗ build-essential not found${NC}"
    echo -e "   ${YELLOW}   Install: sudo apt-get install build-essential${NC}"
    ALL_PASSED=false
fi

# 9. Check for Coral TPU
echo -n "[9/10] Checking Google Coral TPU... "
if [ -e /dev/apex_0 ]; then
    echo -e "${GREEN}✓ Coral TPU detected at /dev/apex_0${NC}"
    
    # Check kernel modules
    if lsmod | grep -q "^apex"; then
        echo -e "   ${GREEN}✓ apex kernel module loaded${NC}"
    else
        echo -e "   ${YELLOW}⚠ apex kernel module not loaded${NC}"
        echo -e "   ${YELLOW}   Run: sudo modprobe apex${NC}"
    fi
    
    if lsmod | grep -q "^gasket"; then
        echo -e "   ${GREEN}✓ gasket kernel module loaded${NC}"
    else
        echo -e "   ${YELLOW}⚠ gasket kernel module not loaded${NC}"
        echo -e "   ${YELLOW}   Run: sudo modprobe gasket${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Coral TPU not detected (optional - will use heuristic intent classification)${NC}"
fi

# 10. Check network connectivity
echo -n "[10/10] Checking network... "
if ping -c 1 -W 2 github.com &> /dev/null; then
    echo -e "${GREEN}✓ Network connectivity OK${NC}"
else
    echo -e "${YELLOW}⚠ Cannot reach github.com (may affect setup)${NC}"
fi

echo ""
echo "=== Summary ==="
if [ "$ALL_PASSED" = true ]; then
    echo -e "${GREEN}✓ All critical checks passed! System ready for EDISON installation.${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Run: ./scripts/setup_ubuntu.sh"
    echo "2. Download models: ./scripts/download_models.py"
    echo "3. Install services: sudo ./scripts/enable_services.sh"
    exit 0
else
    echo -e "${RED}✗ Some critical checks failed. Please address the issues above before installation.${NC}"
    exit 1
fi
