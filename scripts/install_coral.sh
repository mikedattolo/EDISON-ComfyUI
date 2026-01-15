#!/bin/bash
# Install Google Coral Edge TPU Runtime
# Production-ready and idempotent

set -euo pipefail

echo "=== Installing Coral Edge TPU Runtime ==="
echo ""

# Check if already installed
if dpkg -l | grep -q libedgetpu1-std; then
    echo "✓ Coral packages already installed"
    echo ""
    if [ -e /dev/apex_0 ]; then
        echo "✓ Coral TPU device detected at /dev/apex_0"
    else
        echo "⚠ Coral packages installed but device not found at /dev/apex_0"
        echo "  You may need to reboot or check hardware connection"
    fi
    exit 0
fi

# Add Coral repository if not present
CORAL_LIST="/etc/apt/sources.list.d/coral-edgetpu.list"
if [ ! -f "$CORAL_LIST" ]; then
    echo "[1/4] Adding Coral repository..."
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee "$CORAL_LIST"
    echo "✓ Repository added"
else
    echo "[1/4] Coral repository already present"
fi

# Add GPG key
echo ""
echo "[2/4] Adding repository GPG key..."
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/coral-edgetpu-archive-keyring.gpg 2>/dev/null || true
# Update the list file to use the signed-by key
echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu-archive-keyring.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee "$CORAL_LIST"
echo "✓ GPG key added"

# Update package lists
echo ""
echo "[3/4] Updating package lists..."
sudo apt-get update
echo "✓ Package lists updated"

# Install Coral packages
echo ""
echo "[4/4] Installing Coral packages..."
sudo apt-get install -y gasket-dkms libedgetpu1-std
echo "✓ Coral packages installed"

echo ""
echo "=== Coral Installation Complete ==="
echo ""
echo "IMPORTANT: A REBOOT IS REQUIRED for the Coral TPU driver to load"
echo ""
echo "After reboot, verify installation with:"
echo "  ls -l /dev/apex_0"
echo ""
echo "Expected output:"
echo "  crw-rw---- 1 root apex 120, 0 <date> /dev/apex_0"
echo ""
echo "If the device is not present after reboot:"
echo "  - Check that the Coral module is properly inserted"
echo "  - Run: sudo dmesg | grep apex"
echo "  - Run: lsmod | grep apex"
echo ""
echo "To test the TPU after reboot:"
echo "  python3 -c 'from pycoral.utils import edgetpu; print(edgetpu.list_edge_tpus())'"
echo ""
