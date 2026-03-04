#!/bin/bash
# ============================================================
# EDISON Setup Script
# Fixes venv, installs all dependencies, configures Playwright
# Run as: sudo bash /opt/edison/setup_edison.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "═══════════════════════════════════════════"
echo "  EDISON Setup — $SCRIPT_DIR"
echo "═══════════════════════════════════════════"

# ── Step 1: Find Python ──────────────────────
PYBIN=""
for py in python3.12 python3.11 python3.10 python3; do
    if command -v "$py" &>/dev/null; then
        PYBIN="$(command -v $py)"
        break
    fi
done

if [ -z "$PYBIN" ]; then
    echo "❌ No python3 found! Install python3 first:"
    echo "   sudo apt-get install python3 python3-venv python3-pip"
    exit 1
fi

echo "✅ Found Python: $PYBIN ($($PYBIN --version 2>&1))"

# ── Step 2: Fix/recreate venv ────────────────
# Check if venv python is actually working
if [ -f "$VENV_DIR/bin/python" ] && "$VENV_DIR/bin/python" --version &>/dev/null; then
    echo "✅ Existing venv OK"
else
    echo "⚙️  Rebuilding venv (broken or missing python)..."
    
    # Ensure python3-venv is installed
    if ! $PYBIN -m ensurepip --version &>/dev/null 2>&1; then
        echo "   Installing python3-venv..."
        apt-get install -y python3-venv python3-pip 2>/dev/null || true
    fi
    
    # Save the existing site-packages if present (don't lose torch, etc.)
    SITE_PKGS="$VENV_DIR/lib"
    BACKUP=""
    if [ -d "$SITE_PKGS" ]; then
        BACKUP="/tmp/edison_venv_lib_backup_$$"
        echo "   Backing up existing packages..."
        cp -a "$SITE_PKGS" "$BACKUP"
    fi
    
    # Remove broken venv bin (keep lib if backed up)
    rm -rf "$VENV_DIR/bin" "$VENV_DIR/include" "$VENV_DIR/pyvenv.cfg"
    
    # Create fresh venv skeleton
    $PYBIN -m venv "$VENV_DIR" --without-pip 2>/dev/null || \
    $PYBIN -m venv --copies "$VENV_DIR" --without-pip 2>/dev/null || {
        # Nuclear option: manually create the structure
        echo "   Manual venv creation..."
        mkdir -p "$VENV_DIR/bin"
        cp "$PYBIN" "$VENV_DIR/bin/python3"
        ln -sf python3 "$VENV_DIR/bin/python"
        cat > "$VENV_DIR/pyvenv.cfg" <<PYCFG
home = $(dirname $PYBIN)
include-system-site-packages = false
version = $($PYBIN --version 2>&1 | awk '{print $2}')
PYCFG
    }
    
    # Restore backed up packages
    if [ -n "$BACKUP" ] && [ -d "$BACKUP" ]; then
        echo "   Restoring existing packages..."
        rm -rf "$VENV_DIR/lib"
        cp -a "$BACKUP" "$VENV_DIR/lib"
        rm -rf "$BACKUP"
    fi
    
    # Now bootstrap pip
    echo "   Installing pip into venv..."
    "$VENV_DIR/bin/python" -m ensurepip --upgrade 2>/dev/null || \
    "$VENV_DIR/bin/python" -c "
import urllib.request, os, tempfile
url = 'https://bootstrap.pypa.io/get-pip.py'
f = os.path.join(tempfile.gettempdir(), 'get-pip.py')
urllib.request.urlretrieve(url, f)
" && "$VENV_DIR/bin/python" /tmp/get-pip.py 2>/dev/null || {
        echo "❌ Could not install pip. Try: apt-get install python3-pip"
        exit 1
    }
    
    echo "✅ Venv rebuilt successfully"
fi

# Verify venv python works
echo "   Python: $("$VENV_DIR/bin/python" --version 2>&1)"

# ── Step 3: Upgrade pip and install requirements ──
echo ""
echo "⚙️  Installing Python packages..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel -q

if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    "$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt" -q
    echo "✅ Python packages installed"
else
    echo "⚠️  No requirements.txt found at $SCRIPT_DIR"
fi

# ── Step 4: Install Playwright + Chromium ─────
echo ""
echo "⚙️  Installing Playwright + Chromium..."
"$VENV_DIR/bin/python" -m pip install playwright -q

# Playwright install might need to run as the user who owns the cache
"$VENV_DIR/bin/python" -m playwright install chromium 2>&1 || {
    echo "⚠️  Playwright chromium install had issues, trying with --with-deps..."
    "$VENV_DIR/bin/python" -m playwright install --with-deps chromium 2>&1 || true
}

# ── Step 5: Install system libraries for Chromium ──
echo ""
echo "⚙️  Installing system libraries for headless Chromium..."
apt-get install -y \
    libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libpango-1.0-0 libcairo2 \
    libxkbcommon0 libnspr4 libnss3 libx11-xcb1 \
    libxshmfence1 libdbus-1-3 \
    libasound2t64 2>/dev/null || \
apt-get install -y \
    libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libpango-1.0-0 libcairo2 \
    libxkbcommon0 libnspr4 libnss3 libx11-xcb1 \
    libxshmfence1 libdbus-1-3 \
    libasound2 2>/dev/null || true

echo "✅ System libraries installed"

# ── Step 6: Fix permissions ───────────────────
echo ""
echo "⚙️  Fixing permissions..."
# Detect the user who owns the services directory
OWNER=$(stat -c '%U' "$SCRIPT_DIR/services" 2>/dev/null || echo "edison")
chown -R "$OWNER:$OWNER" "$VENV_DIR" 2>/dev/null || true
echo "✅ Permissions set to $OWNER"

# ── Step 7: Verify ───────────────────────────
echo ""
echo "═══════════════════════════════════════════"
echo "  Verification"
echo "═══════════════════════════════════════════"
echo -n "  Python:     " && "$VENV_DIR/bin/python" --version 2>&1
echo -n "  pip:        " && "$VENV_DIR/bin/python" -m pip --version 2>&1 | awk '{print $2}'
echo -n "  uvicorn:    " && "$VENV_DIR/bin/python" -c "import uvicorn; print(uvicorn.__version__)" 2>&1 || echo "MISSING"
echo -n "  playwright: " && "$VENV_DIR/bin/python" -c "import playwright; print('OK')" 2>&1 || echo "MISSING"

# Quick Playwright browser test
echo -n "  chromium:   "
"$VENV_DIR/bin/python" -c "
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(args=['--no-sandbox','--disable-dev-shm-usage'])
    pg = b.new_page()
    pg.goto('https://example.com', timeout=10000)
    print(f'OK — {pg.title()}')
    b.close()
" 2>&1 || echo "FAILED (browser may still work at runtime)"

echo ""
echo "═══════════════════════════════════════════"
echo "  ✅ Setup complete! Restart EDISON with:"
echo "     cd $SCRIPT_DIR && sudo ./restart_edison.sh"
echo "═══════════════════════════════════════════"
