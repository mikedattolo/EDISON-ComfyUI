#!/bin/bash

echo "Checking ComfyUI Installation"
echo "=============================="
echo ""

# Check if ComfyUI is running
if curl -s http://localhost:8188 > /dev/null 2>&1; then
    echo "✅ ComfyUI is running on port 8188"
else
    echo "❌ ComfyUI is NOT running"
    echo ""
    echo "Start it with:"
    echo "  sudo systemctl start edison-comfyui"
    echo "  sudo systemctl status edison-comfyui"
    exit 1
fi

echo ""
echo "Checking model files..."
echo ""

# Check FLUX model
if [ -f "/opt/edison/ComfyUI/models/checkpoints/flux1-schnell.safetensors" ]; then
    SIZE=$(ls -lh /opt/edison/ComfyUI/models/checkpoints/flux1-schnell.safetensors | awk '{print $5}')
    echo "✅ FLUX model: $SIZE"
else
    echo "❌ FLUX model NOT FOUND"
fi

# Check VAE
if [ -f "/opt/edison/ComfyUI/models/vae/ae.safetensors" ]; then
    SIZE=$(ls -lh /opt/edison/ComfyUI/models/vae/ae.safetensors | awk '{print $5}')
    echo "✅ VAE model: $SIZE"
else
    echo "❌ VAE model NOT FOUND"
fi

# Check CLIP models
if [ -f "/opt/edison/ComfyUI/models/clip/t5xxl_fp16.safetensors" ]; then
    SIZE=$(ls -lh /opt/edison/ComfyUI/models/clip/t5xxl_fp16.safetensors | awk '{print $5}')
    echo "✅ T5-XXL CLIP: $SIZE"
else
    echo "❌ T5-XXL CLIP NOT FOUND"
fi

if [ -f "/opt/edison/ComfyUI/models/clip/clip_l.safetensors" ]; then
    SIZE=$(ls -lh /opt/edison/ComfyUI/models/clip/clip_l.safetensors | awk '{print $5}')
    echo "✅ CLIP-L model: $SIZE"
else
    echo "❌ CLIP-L model NOT FOUND"
fi

echo ""
echo "Checking ComfyUI logs for errors..."
echo ""
sudo journalctl -u edison-comfyui -n 50 --no-pager | grep -i "error\|warning\|failed" || echo "No errors found in recent logs"

echo ""
echo "Testing ComfyUI API..."
curl -s http://localhost:8188/history | head -c 200
echo ""
