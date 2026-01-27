# HTTPS/SSL Setup for Voice Mode

## Why HTTPS is Required

Web browsers require HTTPS (secure connection) to access the microphone for security reasons. This enables voice mode to work like ChatGPT's voice feature.

## Quick Setup (Self-Signed Certificate)

### 1. Generate SSL Certificate

```bash
# Run the certificate generation script
cd /opt/edison
sudo bash scripts/generate_ssl_cert.sh
```

This creates:
- `/opt/edison/ssl/cert.pem` (certificate)
- `/opt/edison/ssl/key.pem` (private key)

### 2. Update and Restart Services

```bash
# Copy updated service file
sudo cp services/systemd/edison-web.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Restart web service
sudo systemctl restart edison-web.service

# Check status
sudo systemctl status edison-web.service
```

### 3. Access via HTTPS

Browse to: `https://YOUR-IP:8080`

**Important:** Your browser will show a security warning because the certificate is self-signed. This is normal and safe for private use.

Click:
1. **Advanced** or **More Information**
2. **Proceed to site** or **Accept the Risk and Continue**

### 4. Test Voice Mode

The microphone button (🎙️) should now appear in the UI. Click it to start voice mode!

## Access Methods

### Option 1: HTTPS (Recommended)
```
https://192.168.1.26:8080
```
- ✅ Voice mode works
- ⚠️ Requires accepting self-signed certificate warning

### Option 2: SSH Tunnel (Alternative)
```bash
ssh -L 8080:localhost:8080 -L 8811:localhost:8811 user@192.168.1.26
```
Then browse to: `http://localhost:8080`
- ✅ Voice mode works (localhost is considered secure)
- ✅ No certificate warnings

## Troubleshooting

### Voice button doesn't appear
Check browser console for warnings. Make sure you're using HTTPS or localhost.

### Certificate warning persists
This is normal for self-signed certificates. The connection is still encrypted, you just need to accept it once per browser.

### Service won't start
Check logs:
```bash
sudo journalctl -u edison-web.service -f
```

Verify certificates exist:
```bash
ls -la /opt/edison/ssl/
```

### Generate new certificate
```bash
sudo rm -rf /opt/edison/ssl/
sudo bash scripts/generate_ssl_cert.sh
sudo systemctl restart edison-web.service
```

## Production Setup (Optional)

For a production environment with a real domain name, you can use Let's Encrypt:

```bash
# Install certbot
sudo apt install certbot

# Generate real certificate
sudo certbot certonly --standalone -d your-domain.com

# Update systemd service to use Let's Encrypt certs
# Certificate: /etc/letsencrypt/live/your-domain.com/fullchain.pem
# Key: /etc/letsencrypt/live/your-domain.com/privkey.pem
```

## Features Enabled by HTTPS

- 🎙️ **Voice Mode**: Real-time speech-to-text input
- 🔒 **Secure Connection**: Encrypted communication
- 📱 **Mobile Support**: Works on mobile browsers
- 🌐 **Remote Access**: Secure access from any network

## Security Notes

- Self-signed certificates are fine for private/local use
- The connection is still encrypted
- Browser warnings are just about certificate trust, not security
- For internet-facing deployments, use a real certificate from Let's Encrypt
