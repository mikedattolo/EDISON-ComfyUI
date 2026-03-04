#!/bin/bash
# Restart EDISON services

echo "🔄 Restarting EDISON..."

# Detect install location — works in both dev container and production
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

# Stop any running instances
pkill -f "python.*edison_core" || true
pkill -f "uvicorn.*edison" || true

# Wait a moment
sleep 2

cd "$SCRIPT_DIR"

# Find working Python: prefer venv, fall back to system
PYTHON=""
if [ -f "$VENV_DIR/bin/python" ] && "$VENV_DIR/bin/python" --version &>/dev/null; then
    PYTHON="$VENV_DIR/bin/python"
    echo "   Using venv Python: $PYTHON"
elif [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    PYTHON="python"
    echo "   Using activated venv"
else
    # Fall back to system Python
    for py in python3.12 python3.11 python3.10 python3; do
        if command -v "$py" &>/dev/null; then
            PYTHON="$py"
            break
        fi
    done
    echo "⚠️  No working venv — using system $PYTHON"
    echo "   Run: sudo bash $SCRIPT_DIR/setup_edison.sh"
fi

if [ -z "$PYTHON" ]; then
    echo "❌ No Python found! Install python3 first."
    exit 1
fi

echo "✅ Starting EDISON core..."
cd "$SCRIPT_DIR/services/edison_core"
$PYTHON -m uvicorn app:app --host 0.0.0.0 --port 8811 > /tmp/edison.log 2>&1 &

echo "✅ EDISON restarted!"
echo "📝 Logs: tail -f /tmp/edison.log"
echo "🌐 API: http://localhost:8811"
echo ""
echo "💡 To test code execution:"
echo '   curl -X POST http://localhost:8811/chat -H "Content-Type: application/json" -d '"'"'{"message":"Write Python code to calculate 2+2","mode":"auto"}'"'"''
