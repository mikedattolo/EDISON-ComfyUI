#!/bin/bash
set -e

echo "Downloading FLUX Required Models"
echo "================================="
echo ""
echo "FLUX requires additional models:"
echo "  - VAE (ae.safetensors) ~335MB"
echo "  - CLIP models (t5xxl and clip_l) ~10GB total"
echo ""

read -p "Enter your HuggingFace token: " HF_TOKEN

if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: Token cannot be empty"
    exit 1
fi

cd /opt/edison/ComfyUI

# Download VAE
echo ""
echo "📥 Downloading VAE model (335MB)..."
mkdir -p models/vae
cd models/vae
if [ ! -f "ae.safetensors" ]; then
    wget -c --header="Authorization: Bearer $HF_TOKEN" \
        https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors
    echo "✅ VAE downloaded"
else
    echo "✓ VAE already exists"
fi

# Download CLIP models
echo ""
echo "📥 Downloading CLIP models (~10GB total, this will take time)..."
cd /opt/edison/ComfyUI
mkdir -p models/clip
cd models/clip

if [ ! -f "t5xxl_fp16.safetensors" ]; then
    echo "Downloading T5-XXL (9.5GB)..."
    wget -c --header="Authorization: Bearer $HF_TOKEN" \
        https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors
    echo "✅ T5-XXL downloaded"
else
    echo "✓ T5-XXL already exists"
fi

if [ ! -f "clip_l.safetensors" ]; then
    echo "Downloading CLIP-L (246MB)..."
    wget -c --header="Authorization: Bearer $HF_TOKEN" \
        https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors
    echo "✅ CLIP-L downloaded"
else
    echo "✓ CLIP-L already exists"
fi

echo ""
echo "✅ All FLUX dependencies downloaded!"
echo ""
echo "Installed models:"
echo "  FLUX Model: $(ls -lh /opt/edison/ComfyUI/models/checkpoints/flux*.safetensors 2>/dev/null | awk '{print $9, $5}')"
echo "  VAE: $(ls -lh /opt/edison/ComfyUI/models/vae/ae.safetensors 2>/dev/null | awk '{print $9, $5}')"
echo "  CLIP T5: $(ls -lh /opt/edison/ComfyUI/models/clip/t5xxl_fp16.safetensors 2>/dev/null | awk '{print $9, $5}')"
echo "  CLIP-L: $(ls -lh /opt/edison/ComfyUI/models/clip/clip_l.safetensors 2>/dev/null | awk '{print $9, $5}')"
echo ""
echo "Next step: Restart EDISON services"
echo "  sudo systemctl restart edison-comfyui"
echo "  sudo systemctl restart edison-core"
