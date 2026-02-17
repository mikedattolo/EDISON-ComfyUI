#!/usr/bin/env bash
# EDISON Repo Health Check
# Runs static analysis and basic validations
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

FAIL=0

echo "=== EDISON Health Check ==="
echo ""

# 1. Python syntax check
echo "▸ Checking Python syntax (compileall)..."
if python -m compileall services/ scripts/ -q 2>&1 | grep -i "error"; then
  echo "  ✗ Syntax errors found"
  FAIL=1
else
  echo "  ✓ No syntax errors"
fi

# 2. Ruff lint (critical errors only)
echo "▸ Running ruff (critical checks)..."
if command -v ruff &>/dev/null; then
  if ruff check services/ --select E9,F63,F7,F82 --no-fix 2>&1; then
    echo "  ✓ No critical lint errors"
  else
    echo "  ✗ Critical lint errors found"
    FAIL=1
  fi
else
  echo "  ⚠ ruff not installed (pip install ruff)"
fi

# 3. Merge conflict markers
echo "▸ Checking for merge conflict markers..."
if grep -rn '<<<<<<<\|>>>>>>>' services/ config/ scripts/ web/ --include='*.py' --include='*.yaml' --include='*.js' --include='*.html' 2>/dev/null; then
  echo "  ✗ Merge conflict markers found"
  FAIL=1
else
  echo "  ✓ No merge conflict markers"
fi

# 4. Import sanity (can the main app module be imported?)
echo "▸ Checking main module imports..."
IMPORT_OUT=$(python -c "
import sys, os
sys.path.insert(0, os.path.join('$REPO_ROOT', 'services'))
# Just compile, don't actually run (avoids needing GPU/models)
import py_compile
py_compile.compile('services/edison_core/app.py', doraise=True)
print('OK')
" 2>&1)
if echo "$IMPORT_OUT" | grep -q "OK"; then
  echo "  ✓ Main module compiles cleanly"
else
  echo "$IMPORT_OUT"
  echo "  ✗ Main module has issues"
  FAIL=1
fi

echo ""
if [ "$FAIL" -eq 0 ]; then
  echo "✅ All health checks passed"
else
  echo "❌ Some checks failed — see above"
  exit 1
fi
