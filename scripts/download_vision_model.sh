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
if [ ! -f "$MODELS_DIR/llava-v1.6-mistral-7b-q4_k_m.gguf" ] || [ ! -s "$MODELS_DIR/llava-v1.6-mistral-7b-q4_k_m.gguf" ]; then
    echo "Downloading main model (~3.8GB)..."
    cd "$MODELS_DIR"
    
    # Try direct CDN link first
    if wget --no-verbose --show-progress -c \
        "https://huggingface.co/bartowski/llava-v1.6-mistral-7b-GGUF/resolve/main/llava-v1.6-mistral-7b-Q4_K_M.gguf?download=true" \
        -O llava-v1.6-mistral-7b-q4_k_m.gguf.tmp; then
        mv llava-v1.6-mistral-7b-q4_k_m.gguf.tmp llava-v1.6-mistral-7b-q4_k_m.gguf
        
        # Verify it's a valid GGUF file (should start with GGUF magic bytes)
        if head -c 4 llava-v1.6-mistral-7b-q4_k_m.gguf | grep -q "GGUF"; then
            echo "✅ Main model downloaded successfully"
        else
            echo "❌ Downloaded file is not a valid GGUF file (likely an error page)"
            rm -f llava-v1.6-mistral-7b-q4_k_m.gguf
            echo ""
            echo "Manual download required:"
            echo "1. Visit: https://huggingface.co/bartowski/llava-v1.6-mistral-7b-GGUF/tree/main"
            echo "2. Download: llava-v1.6-mistral-7b-Q4_K_M.gguf"
            echo "3. Copy to: $MODELS_DIR/llava-v1.6-mistral-7b-q4_k_m.gguf"
            exit 1
        fi
    else
        rm -f llava-v1.6-mistral-7b-q4_k_m.gguf.tmp
        echo "❌ Failed to download main model"
        echo ""
        echo "Manual download required:"
        echo "1. Visit: https://huggingface.co/bartowski/llava-v1.6-mistral-7b-GGUF/tree/main"
        echo "2. Download: llava-v1.6-mistral-7b-Q4_K_M.gguf"
        echo "3. Copy to: $MODELS_DIR/llava-v1.6-mistral-7b-q4_k_m.gguf"
        exit 1
    fi
else
    echo "✅ Main model already exists"
fi

# Download the CLIP projector (mmproj)
if [ ! -f "$MODELS_DIR/llava-v1.6-mistral-7b-mmproj-q4_0.gguf" ] || [ ! -s "$MODELS_DIR/llava-v1.6-mistral-7b-mmproj-q4_0.gguf" ]; then
    echo "Downloading CLIP projector (~634MB)..."
    cd "$MODELS_DIR"
    
    if wget --no-verbose --show-progress -c \
        "https://huggingface.co/bartowski/llava-v1.6-mistral-7b-GGUF/resolve/main/mmproj-mistral7b-f16.gguf?download=true" \
        -O llava-v1.6-mistral-7b-mmproj-q4_0.gguf.tmp; then
        mv llava-v1.6-mistral-7b-mmproj-q4_0.gguf.tmp llava-v1.6-mistral-7b-mmproj-q4_0.gguf
        
        # Verify it's a valid GGUF file
        if head -c 4 llava-v1.6-mistral-7b-mmproj-q4_0.gguf | grep -q "GGUF"; then
            echo "✅ CLIP projector downloaded successfully"
        else
            echo "❌ Downloaded file is not a valid GGUF file (likely an error page)"
            rm -f llava-v1.6-mistral-7b-mmproj-q4_0.gguf
            echo ""
            echo "Manual download required:"
            echo "1. Visit: https://huggingface.co/bartowski/llava-v1.6-mistral-7b-GGUF/tree/main"
            echo "2. Download: mmproj-mistral7b-f16.gguf"
            echo "3. Copy to: $MODELS_DIR/llava-v1.6-mistral-7b-mmproj-q4_0.gguf"
            exit 1
        fi
    else
        rm -f llava-v1.6-mistral-7b-mmproj-q4_0.gguf.tmp
        echo "❌ Failed to download CLIP projector"
        echo ""
        echo "Manual download required:"
        echo "1. Visit: https://huggingface.co/bartowski/llava-v1.6-mistral-7b-GGUF/tree/main"
        echo "2. Download: mmproj-mistral7b-f16.gguf"
        echo "3. Copy to: $MODELS_DIR/llava-v1.6-mistral-7b-mmproj-q4_0.gguf"
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
