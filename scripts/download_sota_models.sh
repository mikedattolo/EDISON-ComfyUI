#!/bin/bash
# Download SOTA models for EDISON
# Run this from EDISON root: ./scripts/download_sota_models.sh

set -e

# Activate venv
cd /workspaces/EDISON-ComfyUI
source venv/bin/activate

# Target directory
TARGET_DIR="/mnt/models/llm"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "🚀 Downloading SOTA models to $TARGET_DIR"
echo ""

# Clean up failed download
if [ -f "DeepSeek-V3-Q4_K_M.gguf" ] && [ ! -s "DeepSeek-V3-Q4_K_M.gguf" ]; then
    echo "Removing failed DeepSeek-V3 download..."
    rm -f DeepSeek-V3-Q4_K_M.gguf
fi

# ===== Option 1: Qwen2.5-72B (Proven, works now) =====
echo "📦 Option 1: Qwen2.5-72B-Instruct (44GB, excellent reasoning)"
echo "   Same model you're currently using, just Q4_K_M quantization"
read -p "Download Qwen2.5-72B? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    huggingface-cli download bartowski/Qwen2.5-72B-Instruct-GGUF \
        Qwen2.5-72B-Instruct-Q4_K_M.gguf \
        --local-dir . --local-dir-use-symlinks False
    echo "✅ Downloaded: Qwen2.5-72B-Instruct-Q4_K_M.gguf"
fi
echo ""

# ===== Option 2: Qwen2.5-Coder-32B (Better coding) =====
echo "📦 Option 2: Qwen2.5-Coder-32B (20GB, best for coding)"
echo "   Superior code generation and understanding"
read -p "Download Qwen2.5-Coder-32B? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    huggingface-cli download bartowski/Qwen2.5-Coder-32B-Instruct-GGUF \
        Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf \
        --local-dir . --local-dir-use-symlinks False
    echo "✅ Downloaded: Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf"
fi
echo ""

# ===== Option 3: Qwen2-VL-7B (Better vision) =====
echo "📦 Option 3: Qwen2-VL-7B (5GB, much better vision than LLaVA)"
echo "   Improved OCR, detail recognition, multi-image"
read -p "Download Qwen2-VL-7B? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    huggingface-cli download bartowski/Qwen2-VL-7B-Instruct-GGUF \
        Qwen2-VL-7B-Instruct-Q4_K_M.gguf \
        mmproj-model-f16.gguf \
        --local-dir . --local-dir-use-symlinks False
    echo "✅ Downloaded: Qwen2-VL-7B-Instruct-Q4_K_M.gguf + mmproj"
fi
echo ""

# ===== Option 4: Llama-3.3-70B (OpenAI-style) =====
echo "📦 Option 4: Llama-3.3-70B-Instruct (42GB, ChatGPT-like)"
echo "   Meta's latest, similar to GPT-4 quality"
read -p "Download Llama-3.3-70B? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    huggingface-cli download bartowski/Llama-3.3-70B-Instruct-GGUF \
        Llama-3.3-70B-Instruct-Q4_K_M.gguf \
        --local-dir . --local-dir-use-symlinks False
    echo "✅ Downloaded: Llama-3.3-70B-Instruct-Q4_K_M.gguf"
fi
echo ""

# ===== Option 5: Mistral-Large-2 (Strong alternative) =====
echo "📦 Option 5: Mistral-Large-Instruct-2407 (73GB, very capable)"
echo "   Excellent reasoning and coding"
read -p "Download Mistral-Large-2? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    huggingface-cli download bartowski/Mistral-Large-Instruct-2407-GGUF \
        Mistral-Large-Instruct-2407-Q4_K_M.gguf \
        --local-dir . --local-dir-use-symlinks False
    echo "✅ Downloaded: Mistral-Large-Instruct-2407-Q4_K_M.gguf"
fi
echo ""

echo "✅ Download complete!"
echo ""
echo "📝 Next steps:"
echo "1. Update config/edison.yaml with new model paths"
echo "2. Restart EDISON: docker-compose restart edison-core"
echo "3. Test: curl http://localhost:8000/models/list"
echo ""
echo "💡 Recommended config for best quality:"
echo "   fast_model: qwen2.5-coder-32b-instruct-q4_k_m.gguf"
echo "   medium_model: qwen2.5-72b-instruct-q4_k_m.gguf"
echo "   deep_model: llama-3.3-70b-instruct-q4_k_m.gguf"
echo "   vision_model: qwen2-vl-7b-instruct-q4_k_m.gguf"
