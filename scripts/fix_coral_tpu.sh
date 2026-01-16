#!/bin/bash
# Automated Coral TPU Kernel Module Installer/Fixer
# Handles gasket/apex driver installation with kernel 6.8+ compatibility patches

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=== EDISON Coral TPU Kernel Module Installer ==="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}ERROR: This script must be run as root${NC}"
    echo "Usage: sudo $0"
    exit 1
fi

# Check if Coral TPU hardware exists
if [ ! -e /dev/apex_0 ]; then
    echo -e "${YELLOW}WARNING: No Coral TPU detected at /dev/apex_0${NC}"
    echo "If you have a Coral TPU installed, please:"
    echo "1. Check hardware connection (M.2 or USB)"
    echo "2. Reboot the system"
    echo "3. Run this script again"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Detect kernel version
KERNEL_VERSION=$(uname -r)
KERNEL_MAJOR=$(echo $KERNEL_VERSION | cut -d. -f1)
KERNEL_MINOR=$(echo $KERNEL_VERSION | cut -d. -f2)

echo "[1/6] Kernel version: $KERNEL_VERSION"

# Check if already installed and working
if lsmod | grep -q "^apex" && lsmod | grep -q "^gasket"; then
    echo -e "${GREEN}✓ Coral TPU kernel modules already loaded${NC}"
    echo ""
    echo "Module status:"
    lsmod | grep -E "^(apex|gasket)"
    echo ""
    echo "Device status:"
    ls -l /dev/apex_0 2>/dev/null || echo "No /dev/apex_0 found"
    exit 0
fi

# Install dependencies
echo ""
echo "[2/6] Installing dependencies..."
apt-get update
apt-get install -y \
    debhelper \
    dkms \
    devscripts \
    dh-dkms \
    build-essential \
    linux-headers-$(uname -r)

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Clone gasket-driver
WORK_DIR="/tmp/coral-install-$$"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo ""
echo "[3/6] Downloading gasket-driver..."
git clone https://github.com/google/gasket-driver.git
cd gasket-driver

echo -e "${GREEN}✓ gasket-driver downloaded${NC}"

# Apply patches for kernel 6.8+
if [ "$KERNEL_MAJOR" -ge 6 ] && [ "$KERNEL_MINOR" -ge 8 ]; then
    echo ""
    echo "[4/6] Applying kernel 6.8+ compatibility patches..."
    
    # Patch 1: Fix eventfd_signal API change
    echo "Patching apex_driver.c for eventfd_signal..."
    sed -i 's/eventfd_signal(ctx->eventfd, 1);/eventfd_signal(ctx->eventfd);/g' src/apex_driver.c
    
    # Patch 2: Fix class_create API change
    echo "Patching gasket_core.c for class_create..."
    sed -i 's/class_create(THIS_MODULE, class_name)/class_create(class_name)/g' src/gasket_core.c
    
    echo -e "${GREEN}✓ Patches applied${NC}"
else
    echo ""
    echo "[4/6] Kernel version < 6.8, no patches needed"
fi

# Build and install
echo ""
echo "[5/6] Building and installing kernel modules..."

# Try DKMS first
echo "Attempting DKMS installation..."
debuild -us -uc -tc -b 2>&1 | tee build.log || BUILD_FAILED=true

if [ "${BUILD_FAILED:-false}" = "true" ]; then
    echo -e "${YELLOW}⚠ DKMS build failed, trying manual build...${NC}"
    
    # Manual build
    cd src
    make clean || true
    make
    
    if [ $? -eq 0 ]; then
        echo "Installing kernel modules manually..."
        cp gasket.ko /lib/modules/$(uname -r)/kernel/drivers/char/
        cp apex.ko /lib/modules/$(uname -r)/kernel/drivers/char/
        depmod -a
        echo -e "${GREEN}✓ Manual installation successful${NC}"
    else
        echo -e "${RED}✗ Build failed${NC}"
        echo ""
        echo "Build log:"
        tail -50 ../build.log 2>/dev/null || tail -50 build.log 2>/dev/null || echo "No log available"
        exit 1
    fi
else
    # DKMS succeeded, install the package
    dpkg -i ../gasket-dkms_*.deb 2>/dev/null || true
fi

# Load modules
echo ""
echo "[6/6] Loading kernel modules..."
modprobe gasket
modprobe apex

# Verify
if lsmod | grep -q "^apex" && lsmod | grep -q "^gasket"; then
    echo -e "${GREEN}✓ Kernel modules loaded successfully${NC}"
else
    echo -e "${RED}✗ Failed to load kernel modules${NC}"
    exit 1
fi

# Setup device permissions
echo ""
echo "Setting up device permissions..."
groupadd -f apex
usermod -aG apex edison 2>/dev/null || true

# Create udev rule
cat > /etc/udev/rules.d/65-apex.rules << 'EOF'
SUBSYSTEM=="apex", MODE="0660", GROUP="apex"
EOF

udevadm control --reload-rules
udevadm trigger

# Verify device
if [ -e /dev/apex_0 ]; then
    echo -e "${GREEN}✓ Coral TPU device available at /dev/apex_0${NC}"
    ls -l /dev/apex_0
else
    echo -e "${YELLOW}⚠ /dev/apex_0 not found after module load${NC}"
    echo "   Try rebooting the system"
fi

# Cleanup
cd /
rm -rf "$WORK_DIR"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Kernel modules loaded:"
lsmod | grep -E "^(apex|gasket)"
echo ""
echo "Next steps:"
echo "1. Reboot the system for changes to take effect"
echo "2. Verify with: ls -l /dev/apex_0"
echo "3. Install pycoral if using Python 3.9-3.11"
echo ""

# Auto-load on boot
echo "Setting up auto-load on boot..."
echo "gasket" >> /etc/modules-load.d/coral.conf 2>/dev/null || true
echo "apex" >> /etc/modules-load.d/coral.conf 2>/dev/null || true
echo -e "${GREEN}✓ Modules will auto-load on boot${NC}"
