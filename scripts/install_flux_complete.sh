#!/bin/bash
set -e

# Complete FLUX.1 Installation Script
# Downloads all required models with HuggingFace authentication

echo "======================================"
echo "FLUX.1 Complete Installation"
echo "======================================"
echo ""
echo "This will download all required FLUX models:"
echo "  1. FLUX.1 Dev: ~23GB (main model)"
echo "  2. VAE: ~160MB"
echo "  3. CLIP L: ~950MB"
echo "  4. T5 XXL (FP8): ~9GB"
echo "  Total: ~33GB"
echo ""

# Get HuggingFace token
read -p "Enter your HuggingFace token: " HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: Token required"
    exit 1
fi

# Setup directories
MODELS_DIR="/opt/edison/models"
FLUX_DIR="$MODELS_DIR/flux"
CLIP_DIR="$MODELS_DIR/clip"
COMFYUI_CHECKPOINTS="/opt/edison/ComfyUI/models/checkpoints"
COMFYUI_VAE="/opt/edison/ComfyUI/models/vae"
COMFYUI_CLIP="/opt/edison/ComfyUI/models/clip"

echo ""
echo "Creating directories..."
sudo mkdir -p "$FLUX_DIR" "$CLIP_DIR"
sudo mkdir -p "$COMFYUI_VAE" "$COMFYUI_CLIP"

# Fix permissions
sudo chown -R $USER:$USER "$MODELS_DIR"
sudo chown -R $USER:$USER /opt/edison/ComfyUI/models/

echo "✓ Directories ready"

# Function to download with progress
download_model() {
    local url="$1"
    local output="$2"
    local name="$3"
    local use_auth="$4"
    
    if [ -f "$output" ]; then
        local size=$(du -h "$output" | cut -f1)
        echo "✓ $name already exists ($size), skipping..."
        return 0
    fi
    
    echo ""
    echo "⬇️  Downloading $name..."
    
    if [ "$use_auth" = "true" ]; then
        wget --continue --progress=bar:force \
            --header="Authorization: Bearer $HF_TOKEN" \
            -O "$output" "$url"
    else
        wget --continue --progress=bar:force \
            -O "$output" "$url"
    fi
    
    if [ $? -eq 0 ]; then
        local size=$(du -h "$output" | cut -f1)
        echo "✓ Downloaded $name ($size)"
    else
        echo "❌ Failed to download $name"
        exit 1
    fi
}

# 1. Download FLUX.1 Dev (if not already done)
download_model \
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors" \
    "$COMFYUI_CHECKPOINTS/flux1-dev.safetensors" \
    "FLUX.1 Dev (Main Model)" \
    "true"

# 2. Download VAE
download_model \
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors" \
    "$FLUX_DIR/ae.safetensors" \
    "FLUX VAE" \
    "true"

# 3. Download CLIP L
download_model \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors" \
    "$CLIP_DIR/clip_l.safetensors" \
    "CLIP L Text Encoder" \
    "false"

# 4. Download T5 XXL (FP8)
download_model \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors" \
    "$CLIP_DIR/t5xxl_fp8_e4m3fn.safetensors" \
    "T5 XXL FP8 Text Encoder" \
    "false"

echo ""
echo "======================================"
echo "Creating ComfyUI symlinks..."
echo "======================================"

# Create symlinks so ComfyUI can find the models
ln -sf "$FLUX_DIR/ae.safetensors" "$COMFYUI_VAE/ae.safetensors" 2>/dev/null || true
ln -sf "$CLIP_DIR/clip_l.safetensors" "$COMFYUI_CLIP/clip_l.safetensors" 2>/dev/null || true
ln -sf "$CLIP_DIR/t5xxl_fp8_e4m3fn.safetensors" "$COMFYUI_CLIP/t5xxl_fp8_e4m3fn.safetensors" 2>/dev/null || true

echo "✓ Symlinks created"

# Install ComfyUI FLUX custom nodes
echo ""
echo "======================================"
echo "Installing ComfyUI FLUX nodes..."
echo "======================================"

cd /opt/edison/ComfyUI/custom_nodes

if [ -d "ComfyUI-FLUX-BFL" ]; then
    echo "✓ FLUX nodes already installed"
else
    echo "Cloning FLUX custom nodes..."
    # Clone without credentials prompt (public repo)
    GIT_TERMINAL_PROMPT=0 git clone https://github.com/kijai/ComfyUI-FLUX-BFL.git
    if [ $? -eq 0 ]; then
        cd ComfyUI-FLUX-BFL
        pip install -r requirements.txt
        echo "✓ FLUX nodes installed"
    else
        echo "⚠️  Failed to clone FLUX nodes. Installing manually..."
        echo "   Run: cd /opt/edison/ComfyUI/custom_nodes && git clone https://github.com/kijai/ComfyUI-FLUX-BFL.git"
    fi
fi

echo ""
echo "======================================"
echo "✅ FLUX.1 Installation Complete!"
echo "======================================"
echo ""
echo "Installed models:"
echo "  Main model: $COMFYUI_CHECKPOINTS/flux1-dev.safetensors"
echo "  VAE:        $FLUX_DIR/ae.safetensors"
echo "  CLIP L:     $CLIP_DIR/clip_l.safetensors"
echo "  T5 XXL:     $CLIP_DIR/t5xxl_fp8_e4m3fn.safetensors"
echo ""
echo "Disk usage:"
du -sh "$COMFYUI_CHECKPOINTS"/flux1-dev.safetensors 2>/dev/null || echo "  FLUX model: not found"
du -sh "$FLUX_DIR" 2>/dev/null || echo "  FLUX dir: 0"
du -sh "$CLIP_DIR" 2>/dev/null || echo "  CLIP dir: 0"
echo ""
echo "Next steps:"
echo "1. Restart ComfyUI: sudo systemctl restart comfyui"
echo "2. Test in EDISON: 'generate an image of a cyberpunk city'"
echo "3. Images will auto-save to gallery!"
echo ""
echo "FLUX.1 is ready! 🚀"
echo "Quality: Surpasses DALL-E 3"
echo "Speed: 4-8 steps vs 30-50 for SDXL"
echo ""
