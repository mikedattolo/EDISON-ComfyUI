#!/bin/bash
set -e

echo "FLUX Model Download for ComfyUI"
echo "================================"
echo ""
echo "Both FLUX models now require HuggingFace authentication."
echo ""
echo "Setup steps:"
echo "1. Create account at https://huggingface.co/join (if you don't have one)"
echo "2. Get your token at https://huggingface.co/settings/tokens"
echo "3. Accept the model license:"
echo "   - FLUX.1-schnell: https://huggingface.co/black-forest-labs/FLUX.1-schnell"
echo "   - FLUX.1-dev: https://huggingface.co/black-forest-labs/FLUX.1-dev"
echo ""
read -p "Enter your HuggingFace token: " HF_TOKEN

if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: Token cannot be empty"
    exit 1
fi

echo ""
echo "Choose a model:"
echo "1. FLUX.1-schnell (Recommended - Faster, 4-step generation, 23GB)"
echo "2. FLUX.1-dev (Better quality, more steps required, 23GB)"
echo ""
read -p "Enter choice (1 or 2): " choice

cd /opt/edison/ComfyUI/models/checkpoints

case $choice in
    1)
        echo ""
        echo "Downloading FLUX.1-schnell..."
        echo "This is the faster, distilled version (4-step generation)."
        echo ""
        wget -c --header="Authorization: Bearer $HF_TOKEN" \
            https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors
        echo ""
        echo "✅ FLUX.1-schnell downloaded successfully!"
        ;;
    2)
        echo ""
        echo "Downloading FLUX.1-dev..."
        echo "This version produces higher quality but requires more steps."
        echo ""
        wget -c --header="Authorization: Bearer $HF_TOKEN" \
            https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors
        echo ""
        echo "✅ FLUX.1-dev downloaded successfully!"
        ;;
    *)
        echo "❌ Invalid choice. Please run script again and choose 1 or 2."
        exit 1
        ;;
esac

echo ""
echo "Model location: $(pwd)"
ls -lh flux*.safetensors 2>/dev/null || true
echo ""
echo "Next step: Start ComfyUI service"
echo "  sudo systemctl start edison-comfyui"
echo "  sudo systemctl status edison-comfyui"
