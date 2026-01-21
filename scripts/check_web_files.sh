#!/bin/bash
# Diagnostic script to check web files

echo "=== Checking EDISON Web Files ==="
echo ""
echo "1. Repository location:"
pwd

echo ""
echo "2. Web directory exists:"
ls -la web/ 2>&1 | head -5

echo ""
echo "3. app_features.js details:"
ls -lh web/app_features.js

echo ""
echo "4. File content sample (first 5 lines):"
head -5 web/app_features.js

echo ""
echo "5. Git status of web directory:"
git status web/

echo ""
echo "6. Edison-web service status:"
systemctl status edison-web --no-pager | head -15

echo ""
echo "7. Check if file is readable:"
test -r web/app_features.js && echo "✓ File is readable" || echo "✗ File is NOT readable"

echo ""
echo "=== Diagnosis Complete ==="
