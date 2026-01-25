#!/bin/bash
set -e

echo "Installing ComfyUI for EDISON..."
echo "======================================"

# Navigate to EDISON directory
cd /opt/edison

# Check if ComfyUI directory exists and is empty
if [ -d "ComfyUI" ]; then
    # Count files (excluding . and ..)
    file_count=$(ls -A ComfyUI | wc -l)
    
    if [ "$file_count" -gt 2 ]; then
        echo "⚠️  ComfyUI directory not empty. Backing up custom_nodes..."
        if [ -d "ComfyUI/custom_nodes" ]; then
            cp -r ComfyUI/custom_nodes /tmp/custom_nodes_backup
        fi
        rm -rf ComfyUI
    else
        # Clean up the empty directory
        rm -rf ComfyUI
    fi
fi

# Clone ComfyUI
echo "📥 Cloning ComfyUI repository..."
git clone https://github.com/comfyanonymous/ComfyUI.git

# Change ownership to edison user
sudo chown -R edison:edison ComfyUI

cd ComfyUI

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment and install dependencies
echo "📦 Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Create models directory structure
echo "📁 Creating model directories..."
mkdir -p models/checkpoints
mkdir -p models/vae
mkdir -p models/loras
mkdir -p models/clip

# Restore custom_nodes if backed up
if [ -d "/tmp/custom_nodes_backup" ]; then
    echo "♻️  Restoring custom_nodes backup..."
    cp -r /tmp/custom_nodes_backup custom_nodes
    rm -rf /tmp/custom_nodes_backup
fi

# Create startup script
cat > start_comfyui.sh << 'EOF'
#!/bin/bash
cd /opt/edison/ComfyUI
source venv/bin/activate
python main.py --listen 0.0.0.0 --port 8188
EOF

chmod +x start_comfyui.sh

echo ""
echo "✅ ComfyUI installed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Download FLUX model:"
echo "   cd /opt/edison/ComfyUI/models/checkpoints"
echo "   wget https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors"
echo ""
echo "2. Start ComfyUI:"
echo "   cd /opt/edison/ComfyUI"
echo "   ./start_comfyui.sh"
echo ""
echo "   Or in the background:"
echo "   nohup ./start_comfyui.sh > comfyui.log 2>&1 &"
echo ""
echo "3. Verify it's running:"
echo "   curl http://localhost:8188"
echo ""
