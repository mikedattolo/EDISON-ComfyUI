#!/bin/bash
set -e

echo "Enabling ComfyUI as a system service..."
echo "========================================"

# Check if ComfyUI is installed
if [ ! -f "/opt/edison/ComfyUI/main.py" ]; then
    echo "❌ Error: ComfyUI not found at /opt/edison/ComfyUI"
    echo "Please run scripts/install_comfyui.sh first"
    exit 1
fi

# Check if venv exists
if [ ! -d "/opt/edison/ComfyUI/venv" ]; then
    echo "❌ Error: ComfyUI virtual environment not found"
    echo "Please run scripts/install_comfyui.sh first"
    exit 1
fi

# Copy service file
echo "Installing systemd service..."
sudo cp /opt/edison/services/systemd/edison-comfyui.service /etc/systemd/system/

# Reload systemd
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

# Enable service
echo "Enabling edison-comfyui service..."
sudo systemctl enable edison-comfyui

# Start service
echo "Starting edison-comfyui service..."
sudo systemctl start edison-comfyui

# Wait a moment for startup
sleep 3

# Check status
echo ""
echo "Service status:"
sudo systemctl status edison-comfyui --no-pager -l

echo ""
echo "✅ ComfyUI service enabled and started!"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status edison-comfyui    # Check status"
echo "  sudo systemctl restart edison-comfyui   # Restart service"
echo "  sudo systemctl stop edison-comfyui      # Stop service"
echo "  sudo journalctl -u edison-comfyui -f    # View logs"
echo ""
echo "Test connection:"
echo "  curl http://localhost:8188"
