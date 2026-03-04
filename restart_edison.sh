#!/bin/bash
# Restart EDISON services

echo "🔄 Restarting EDISON..."

# Detect install location — works in both dev container and production
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stop any running instances
pkill -f "python.*edison_core" || true
pkill -f "uvicorn.*edison" || true

# Wait a moment
sleep 2

# Start EDISON core in background
cd "$SCRIPT_DIR"

# Activate venv if present, otherwise use system python
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
else
    echo "⚠️  No venv found at $SCRIPT_DIR/venv — using system python"
fi

echo "✅ Starting EDISON core..."
cd "$SCRIPT_DIR/services/edison_core"
python -m uvicorn app:app --host 0.0.0.0 --port 8811 > /tmp/edison.log 2>&1 &

echo "✅ EDISON restarted!"
echo "📝 Logs: tail -f /tmp/edison.log"
echo "🌐 API: http://localhost:8811"
echo ""
echo "💡 To test code execution:"
echo '   curl -X POST http://localhost:8811/chat -H "Content-Type: application/json" -d '"'"'{"message":"Write Python code to calculate 2+2","mode":"auto"}'"'"''
