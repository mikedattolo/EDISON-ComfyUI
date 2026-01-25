#!/bin/bash
set -e

echo "FLUX Model Download Options"
echo "============================"
echo ""
echo "Choose a model:"
echo "1. FLUX.1-schnell (Recommended - No auth required, faster, 23GB)"
echo "2. FLUX.1-dev (Requires HuggingFace token, better quality, 23GB)"
echo ""
read -p "Enter choice (1 or 2): " choice

cd /opt/edison/ComfyUI/models/checkpoints

case $choice in
    1)
        echo ""
        echo "Downloading FLUX.1-schnell..."
        echo "This is the faster, distilled version - no authentication needed."
        echo ""
        wget -c https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors
        echo ""
        echo "✅ FLUX.1-schnell downloaded successfully!"
        ;;
    2)
        echo ""
        echo "FLUX.1-dev requires a HuggingFace account and token."
        echo ""
        echo "Setup steps:"
        echo "1. Create account at https://huggingface.co/join"
        echo "2. Accept FLUX.1-dev license at https://huggingface.co/black-forest-labs/FLUX.1-dev"
        echo "3. Get token at https://huggingface.co/settings/tokens"
        echo ""
        read -p "Enter your HuggingFace token: " HF_TOKEN
        
        if [ -z "$HF_TOKEN" ]; then
            echo "❌ Error: Token cannot be empty"
            exit 1
        fi
        
        echo ""
        echo "Downloading FLUX.1-dev with authentication..."
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
