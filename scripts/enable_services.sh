#!/bin/bash
# Enable EDISON systemd services
# Production-ready with safety checks

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_DIR="/opt/edison"

echo "=== Enabling EDISON Services ==="
echo "Source: $REPO_ROOT"
echo "Target: $TARGET_DIR"
echo ""

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: This script must be run with sudo"
    echo "Usage: sudo bash $0"
    exit 1
fi

# Ensure target directory exists and sync repo
echo "[1/7] Syncing repository to $TARGET_DIR..."
if [ "$REPO_ROOT" != "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
    rsync -av --exclude='.git' --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' "$REPO_ROOT/" "$TARGET_DIR/"
    echo "✓ Repository synced to $TARGET_DIR"
    
    # If .venv exists in source, copy it or instruct to recreate
    if [ -d "$REPO_ROOT/.venv" ]; then
        echo "⚠ Virtual environment detected in source"
        echo "  Copying to target (may need to be recreated if paths differ)..."
        cp -r "$REPO_ROOT/.venv" "$TARGET_DIR/.venv" || true
    fi
    
    if [ ! -d "$TARGET_DIR/.venv" ]; then
        echo ""
        echo "WARNING: No virtual environment at $TARGET_DIR/.venv"
        echo "You should run setup_ubuntu.sh from $TARGET_DIR first:"
        echo "  cd $TARGET_DIR && bash scripts/setup_ubuntu.sh"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "✓ Already at target location"
fi

# Create edison user if it doesn't exist
echo ""
echo "[2/7] Creating edison system user..."
if id "edison" &>/dev/null; then
    echo "✓ User 'edison' already exists"
else
    useradd -r -s /bin/bash -d /opt/edison -m edison
    echo "✓ User 'edison' created"
fi

# Set ownership
echo ""
echo "[3/7] Setting ownership..."
chown -R edison:edison "$TARGET_DIR"
echo "✓ Ownership set to edison:edison"

# Copy systemd unit files
echo ""
echo "[4/7] Installing systemd unit files..."
cp "$TARGET_DIR/services/systemd/edison-coral.service" /etc/systemd/system/
cp "$TARGET_DIR/services/systemd/edison-core.service" /etc/systemd/system/
cp "$TARGET_DIR/services/systemd/edison-comfyui.service" /etc/systemd/system/
echo "✓ Unit files copied to /etc/systemd/system/"

# Reload systemd
echo ""
echo "[5/7] Reloading systemd daemon..."
systemctl daemon-reload
echo "✓ Systemd daemon reloaded"

# Enable and start services
echo ""
echo "[6/7] Enabling and starting services..."
systemctl enable edison-coral.service
systemctl enable edison-core.service
systemctl enable edison-comfyui.service

echo "Starting edison-coral..."
systemctl restart edison-coral.service
sleep 2

echo "Starting edison-core..."
systemctl restart edison-core.service
sleep 2

echo "Starting edison-comfyui..."
systemctl restart edison-comfyui.service
sleep 3

echo "✓ Services enabled and started"

# Check service status
echo ""
echo "[7/7] Checking service status..."
echo ""
echo "=== edison-coral.service ==="
if systemctl is-active --quiet edison-coral.service; then
    echo "✓ RUNNING"
    systemctl status edison-coral.service --no-pager -l | head -n 10
else
    echo "✗ FAILED"
    echo "Last 20 lines of logs:"
    journalctl -u edison-coral.service -n 20 --no-pager
fi

echo ""
echo "=== edison-core.service ==="
if systemctl is-active --quiet edison-core.service; then
    echo "✓ RUNNING"
    systemctl status edison-core.service --no-pager -l | head -n 10
else
    echo "✗ FAILED"
    echo "Last 20 lines of logs:"
    journalctl -u edison-core.service -n 20 --no-pager
fi

echo ""
echo "=== edison-comfyui.service ==="
if systemctl is-active --quiet edison-comfyui.service; then
    echo "✓ RUNNING"
    systemctl status edison-comfyui.service --no-pager -l | head -n 10
else
    echo "✗ FAILED"
    echo "Last 20 lines of logs:"
    journalctl -u edison-comfyui.service -n 20 --no-pager
fi

echo ""
echo "=== Services Enabled ==="
echo ""
echo "Service URLs (from this machine):"
echo "  - Coral Intent:  http://127.0.0.1:8808/health"
echo "  - Core Chat:     http://127.0.0.1:8811/health"
echo "  - ComfyUI:       http://localhost:8188"
echo ""
echo "To test health endpoints:"
echo "  curl http://127.0.0.1:8808/health"
echo "  curl http://127.0.0.1:8811/health"
echo ""
echo "To view logs:"
echo "  journalctl -u edison-coral -f"
echo "  journalctl -u edison-core -f"
echo "  journalctl -u edison-comfyui -f"
echo ""
echo "To restart a service:"
echo "  sudo systemctl restart edison-coral"
echo "  sudo systemctl restart edison-core"
echo "  sudo systemctl restart edison-comfyui"
echo ""
