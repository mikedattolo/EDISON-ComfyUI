#!/bin/bash
# Download LLaVA vision model for EDISON

echo "===================================="
echo "EDISON Vision Model Downloader"
echo "===================================="
echo ""

MODELS_DIR="/opt/edison/models/llm"

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

echo "📥 Downloading LLaVA-v1.6-Mistral-7B vision model..."
echo "This will download ~4GB of files"
echo ""

# Download the main model
if [ ! -f "$MODELS_DIR/llava-v1.6-mistral-7b-q4_k_m.gguf" ]; then
    echo "Downloading main model (~3.8GB)..."
    cd "$MODELS_DIR"
    if wget --no-verbose --show-progress -c \
        "https://huggingface.co/bartowski/llava-v1.6-mistral-7b-GGUF/resolve/main/llava-v1.6-mistral-7b-Q4_K_M.gguf" \
        -O llava-v1.6-mistral-7b-q4_k_m.gguf; then
        echo "✅ Main model downloaded successfully"
    else
        echo "❌ Failed to download main model"
        echo "You can manually download from:"
        echo "https://huggingface.co/bartowski/llava-v1.6-mistral-7b-GGUF/blob/main/llava-v1.6-mistral-7b-Q4_K_M.gguf"
        exit 1
    fi
else
    echo "✅ Main model already exists"
fi

# Download the CLIP projector (mmproj)
if [ ! -f "$MODELS_DIR/llava-v1.6-mistral-7b-mmproj-q4_0.gguf" ]; then
    echo "Downloading CLIP projector (~634MB)..."
    cd "$MODELS_DIR"
    if wget --no-verbose --show-progress -c \
        "https://huggingface.co/bartowski/llava-v1.6-mistral-7b-GGUF/resolve/main/mmproj-mistral7b-f16.gguf" \
        -O llava-v1.6-mistral-7b-mmproj-q4_0.gguf; then
        echo "✅ CLIP projector downloaded successfully"
    else
        echo "❌ Failed to download CLIP projector"
        echo "You can manually download from:"
        echo "https://huggingface.co/bartowski/llava-v1.6-mistral-7b-GGUF/blob/main/mmproj-mistral7b-f16.gguf"
        exit 1
    fi
else
    echo "✅ CLIP projector already exists"
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
