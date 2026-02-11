#!/bin/bash
# Download and install ComfyUI video generation nodes (AnimateDiff, Video Helper Suite)
# This enables text-to-video generation in EDISON

set -e

COMFYUI_DIR="${1:-$(dirname "$0")/../ComfyUI}"
CUSTOM_NODES="$COMFYUI_DIR/custom_nodes"
MODELS_DIR="$COMFYUI_DIR/models"

echo "============================================"
echo "  EDISON Video Generation Setup"
echo "============================================"
echo ""

# 1. Install AnimateDiff Evolved (text-to-video via ComfyUI)
echo "📦 Installing AnimateDiff Evolved..."
if [ -d "$CUSTOM_NODES/ComfyUI-AnimateDiff-Evolved" ]; then
    echo "  → Already installed, pulling latest..."
    cd "$CUSTOM_NODES/ComfyUI-AnimateDiff-Evolved" && git pull
else
    cd "$CUSTOM_NODES"
    git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git
    echo "  ✓ AnimateDiff Evolved cloned"
fi

# 2. Install Video Helper Suite (video encoding/decoding)
echo "📦 Installing Video Helper Suite..."
if [ -d "$CUSTOM_NODES/ComfyUI-VideoHelperSuite" ]; then
    echo "  → Already installed, pulling latest..."
    cd "$CUSTOM_NODES/ComfyUI-VideoHelperSuite" && git pull
else
    cd "$CUSTOM_NODES"
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
    echo "  ✓ Video Helper Suite cloned"
fi

# 3. Install pip dependencies for video nodes
echo "📦 Installing Python dependencies..."
pip install -q imageio imageio-ffmpeg opencv-python-headless 2>/dev/null || true

# 4. Download AnimateDiff motion model
MOTION_DIR="$MODELS_DIR/animatediff_models"
mkdir -p "$MOTION_DIR"

if [ ! -f "$MOTION_DIR/v3_sd15_mm.ckpt" ]; then
    echo "📥 Downloading AnimateDiff v3 motion model (~1.8GB)..."
    wget -q --show-progress -O "$MOTION_DIR/v3_sd15_mm.ckpt" \
        "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt" || {
        echo "  ⚠ Download failed. You can manually download from:"
        echo "    https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt"
        echo "    Place in: $MOTION_DIR/"
    }
else
    echo "  ✓ AnimateDiff motion model already downloaded"
fi

# 5. Ensure ffmpeg is installed (for video stitching)
echo "📦 Checking ffmpeg..."
if command -v ffmpeg &>/dev/null; then
    echo "  ✓ ffmpeg is installed: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "  ⚠ ffmpeg not found. Installing..."
    apt-get update -qq && apt-get install -y -qq ffmpeg || {
        echo "  ⚠ Could not install ffmpeg. Please install manually."
    }
fi

echo ""
echo "============================================"
echo "  ✅ Video Generation Setup Complete"
echo "============================================"
echo ""
echo "Available backends (auto-detected at runtime):"
echo "  • AnimateDiff: Text-to-video with motion models"
echo "  • Frame-based: Generate image frames + stitch to video"
echo ""
echo "Optional extras:"
echo "  • CogVideoX: pip install cogvideo-diffusers (requires 16GB+ VRAM)"
echo "  • Wan2.1: Available as ComfyUI custom node"
echo ""
echo "Usage: Ask EDISON to 'make a video of ...' or use /generate-video endpoint"
