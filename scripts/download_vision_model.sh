#!/bin/bash
# Download LLaVA vision model for EDISON

echo "===================================="
echo "EDISON Vision Model Downloader"
echo "===================================="
echo ""

MODELS_DIR="/opt/edison/models/llm"
VENV_DIR="/opt/edison/.venv"

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

echo "📥 Downloading LLaVA-v1.6-Mistral-7B vision model..."
echo "This will download ~4GB of files"
echo ""

# Activate virtual environment if it exists
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    echo "✓ Using virtual environment at $VENV_DIR"
elif [ -f "/opt/edison/venv/bin/activate" ]; then
    source "/opt/edison/venv/bin/activate"
    echo "✓ Using virtual environment at /opt/edison/venv"
else
    echo "⚠️  No virtual environment found, using system Python"
fi

# Check if huggingface-cli is available
if command -v huggingface-cli &> /dev/null; then
    echo "✓ Using huggingface-cli for download..."
    echo ""
    
    # Download main model
    if [ ! -f "$MODELS_DIR/llava-v1.6-mistral-7b-q4_k_m.gguf" ]; then
        echo "Downloading main model (~3.8GB)..."
        huggingface-cli download bartowski/llava-v1.6-mistral-7b-GGUF \
            llava-v1.6-mistral-7b-Q4_K_M.gguf \
            --local-dir "$MODELS_DIR" \
            --local-dir-use-symlinks False
        mv "$MODELS_DIR/llava-v1.6-mistral-7b-Q4_K_M.gguf" "$MODELS_DIR/llava-v1.6-mistral-7b-q4_k_m.gguf"
        echo "✅ Main model downloaded"
    else
        echo "✅ Main model already exists"
    fi
    
    # Download CLIP projector
    if [ ! -f "$MODELS_DIR/llava-v1.6-mistral-7b-mmproj-q4_0.gguf" ]; then
        echo "Downloading CLIP projector (~634MB)..."
        huggingface-cli download bartowski/llava-v1.6-mistral-7b-GGUF \
            mmproj-mistral7b-f16.gguf \
            --local-dir "$MODELS_DIR" \
            --local-dir-use-symlinks False
        mv "$MODELS_DIR/mmproj-mistral7b-f16.gguf" "$MODELS_DIR/llava-v1.6-mistral-7b-mmproj-q4_0.gguf"
        echo "✅ CLIP projector downloaded"
    else
        echo "✅ CLIP projector already exists"
    fi
else
    echo "⚠️  huggingface-cli not found. Installing in virtual environment..."
    pip install -q -U "huggingface_hub[cli]"
    
    if command -v huggingface-cli &> /dev/null; then
        echo "✅ huggingface-cli installed successfully"
        echo ""
        echo "Rerunning download..."
        exec bash "$0"
    else
        echo ""
        echo "❌ Failed to install huggingface-cli"
        echo ""
        echo "📝 Manual download instructions:"
        echo ""
        echo "Option 1 - Install manually and retry:"
        echo "  source $VENV_DIR/bin/activate"
        echo "  pip install -U 'huggingface_hub[cli]'"
        echo "  bash scripts/download_vision_model.sh"
        echo ""
        echo "Option 2 - Manual browser download:"
        echo "  1. Visit: https://huggingface.co/bartowski/llava-v1.6-mistral-7b-GGUF/tree/main"
        echo "  2. Download these files (click the ↓ icon):"
        echo "     - llava-v1.6-mistral-7b-Q4_K_M.gguf (~3.8GB)"
        echo "     - mmproj-mistral7b-f16.gguf (~634MB)"
        echo "  3. Copy/rename them to:"
        echo "     - $MODELS_DIR/llava-v1.6-mistral-7b-q4_k_m.gguf"
        echo "     - $MODELS_DIR/llava-v1.6-mistral-7b-mmproj-q4_0.gguf"
        echo ""
        exit 1
    fi
fi

echo ""
echo "===================================="
echo "✅ Vision model setup complete!"
echo "===================================="
echo ""
echo "Files installed in: $MODELS_DIR"
echo ""
echo "Next steps:"
echo "1. Restart edison-core service:"
echo "   sudo systemctl restart edison-core"
echo ""
echo "2. Check logs to verify vision model loaded:"
echo "   sudo journalctl -u edison-core -f"
echo ""
echo "3. Test image understanding in the web UI!"
echo ""
