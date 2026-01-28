#!/bin/bash

# Quick Test Script for Image Gallery
# This will verify the gallery system is working

API_URL="http://192.168.1.26:8811"

echo "======================================"
echo "EDISON Image Gallery Quick Test"
echo "======================================"
echo ""

# Test 1: Check API is running
echo "Test 1: Checking EDISON API..."
if curl -s "$API_URL/health" > /dev/null 2>&1; then
    echo "✅ API is running"
else
    echo "❌ API is not running. Start it with: sudo systemctl start edison-core"
    exit 1
fi

# Test 2: Check gallery endpoint
echo ""
echo "Test 2: Checking gallery endpoint..."
response=$(curl -s "$API_URL/gallery/list")
if echo "$response" | grep -q "images"; then
    echo "✅ Gallery endpoint working"
    image_count=$(echo "$response" | grep -o '"id"' | wc -l)
    echo "   Found $image_count images in gallery"
else
    echo "❌ Gallery endpoint failed"
    echo "   Response: $response"
    exit 1
fi

# Test 3: Check gallery directory
echo ""
echo "Test 3: Checking gallery directory..."
if [ -d "/workspaces/EDISON-ComfyUI/gallery" ]; then
    echo "✅ Gallery directory exists"
    file_count=$(find /workspaces/EDISON-ComfyUI/gallery -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
    echo "   Found $file_count image files"
else
    echo "⚠️  Gallery directory doesn't exist yet (will be created on first save)"
fi

# Test 4: Check web files
echo ""
echo "Test 4: Checking web UI files..."
if [ -f "/workspaces/EDISON-ComfyUI/web/gallery.js" ]; then
    echo "✅ gallery.js exists"
else
    echo "❌ gallery.js missing"
fi

if grep -q "gallery-panel" /workspaces/EDISON-ComfyUI/web/index.html; then
    echo "✅ Gallery UI added to index.html"
else
    echo "❌ Gallery UI not found in index.html"
fi

if grep -q "gallery-btn" /workspaces/EDISON-ComfyUI/web/styles.css; then
    echo "✅ Gallery styles added to styles.css"
else
    echo "❌ Gallery styles not found in styles.css"
fi

# Test 5: Check ComfyUI
echo ""
echo "Test 5: Checking ComfyUI..."
COMFYUI_URL="http://192.168.1.26:8188"
if curl -s "$COMFYUI_URL/system_stats" > /dev/null 2>&1; then
    echo "✅ ComfyUI is running"
else
    echo "⚠️  ComfyUI is not running (needed for image generation)"
    echo "   Start it with: sudo systemctl start comfyui"
fi

echo ""
echo "======================================"
echo "✅ Gallery System Status"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Open web UI: http://192.168.1.26:8810"
echo "2. Generate an image: 'generate an image of a sunset'"
echo "3. Click the Gallery button in the sidebar"
echo "4. View, download, or delete your generated images"
echo ""
echo "To upgrade to FLUX.1 (better quality):"
echo "  ./scripts/download_flux_model.sh"
echo ""
echo "See GALLERY_AND_UPGRADES.md for full upgrade guide!"
echo ""
