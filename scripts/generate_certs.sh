#!/bin/bash
# Generate self-signed TLS certificates for EDISON Web UI
# This enables HTTPS, which is required for:
#   - Voice input (Web Speech API requires secure context)
#   - getUserMedia (microphone access)
#
# Usage: bash scripts/generate_certs.sh
# Certs are stored in /opt/edison/certs/ (or repo_root/certs/)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CERT_DIR="${REPO_ROOT}/certs"

mkdir -p "$CERT_DIR"

# Get the machine's hostname and IP for SAN
HOSTNAME=$(hostname 2>/dev/null || echo "edison")
# Try to get LAN IP
LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "")

echo "🔐 Generating self-signed TLS certificate for EDISON..."
echo "   Hostname: ${HOSTNAME}"
echo "   LAN IP:   ${LAN_IP:-unknown}"
echo "   Output:   ${CERT_DIR}/"

# Build SAN (Subject Alternative Names) for the cert
SAN="DNS:${HOSTNAME},DNS:localhost,IP:127.0.0.1"
if [ -n "$LAN_IP" ]; then
    SAN="${SAN},IP:${LAN_IP}"
fi

openssl req -x509 \
    -newkey rsa:2048 \
    -keyout "${CERT_DIR}/key.pem" \
    -out "${CERT_DIR}/cert.pem" \
    -days 365 \
    -nodes \
    -subj "/CN=${HOSTNAME}/O=EDISON AI/OU=Self-Signed" \
    -addext "subjectAltName=${SAN}" \
    2>/dev/null

echo ""
echo "✅ Certificates generated:"
echo "   ${CERT_DIR}/cert.pem"
echo "   ${CERT_DIR}/key.pem"
echo ""
echo "📋 Next steps:"
echo "   1. Restart the web UI:  sudo systemctl restart edison-web"
echo "   2. Open https://${LAN_IP:-<your-ip>}:8080 in your browser"
echo "   3. Accept the self-signed certificate warning (it's your own server)"
echo ""
echo "💡 To trust the cert system-wide (optional):"
echo "   sudo cp ${CERT_DIR}/cert.pem /usr/local/share/ca-certificates/edison.crt"
echo "   sudo update-ca-certificates"
