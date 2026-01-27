#!/bin/bash
# Generate self-signed SSL certificate for EDISON Web UI
# This enables HTTPS for voice mode microphone access

CERT_DIR="/opt/edison/ssl"
DAYS_VALID=3650  # 10 years

echo "🔐 Generating SSL certificate for EDISON Web UI..."

# Create SSL directory
sudo mkdir -p "$CERT_DIR"
sudo chmod 755 "$CERT_DIR"

# Generate self-signed certificate
sudo openssl req -x509 -newkey rsa:4096 -nodes \
    -keyout "$CERT_DIR/key.pem" \
    -out "$CERT_DIR/cert.pem" \
    -days $DAYS_VALID \
    -subj "/C=US/ST=State/L=City/O=EDISON/CN=edison-ai" \
    -addext "subjectAltName=DNS:localhost,DNS:*.local,IP:127.0.0.1,IP:192.168.1.26"

# Set permissions - readable by edison user
sudo chmod 644 "$CERT_DIR/cert.pem"
sudo chmod 640 "$CERT_DIR/key.pem"
sudo chown -R edison:edison "$CERT_DIR"

echo "✅ SSL certificate generated at $CERT_DIR"
echo "   Owner: edison:edison"
echo "   Permissions: cert.pem (644), key.pem (640)"
echo ""
echo "📋 Certificate info:"
openssl x509 -in "$CERT_DIR/cert.pem" -noout -text | grep -A2 "Subject:"
echo ""
echo "⚠️  Note: This is a self-signed certificate. Your browser will show a security warning."
echo "    Click 'Advanced' and 'Proceed' to accept it."
echo ""
echo "🎙️  Voice mode will now work with HTTPS!"
