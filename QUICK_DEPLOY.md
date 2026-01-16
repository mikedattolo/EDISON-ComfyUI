# EDISON Quick Deployment Guide for AI PC

## On Your AI PC (192.168.1.26)

### Update the Repository

```bash
cd /opt/edison
sudo -u edison git pull origin main
```

### Copy New Files

```bash
# Copy web UI files
sudo -u edison mkdir -p /opt/edison/web
sudo -u edison cp -r web/* /opt/edison/web/

# Install new service
sudo cp /opt/edison/services/systemd/edison-web.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload
```

### Enable and Start Web Service

```bash
# Enable auto-start on boot
sudo systemctl enable edison-web

# Start the service
sudo systemctl start edison-web

# Check status
sudo systemctl status edison-web
```

### Restart Core Service (for CORS support)

```bash
sudo systemctl restart edison-core
```

### Verify All Services

```bash
# Check all EDISON services
sudo systemctl status edison-coral
sudo systemctl status edison-core  
sudo systemctl status edison-web
sudo systemctl status edison-comfyui

# Test health endpoints
curl http://localhost:8080/health  # Web UI
curl http://localhost:8811/health  # Core API
curl http://localhost:8808/health  # Coral API
curl http://localhost:8188         # ComfyUI
```

## Access the Web UI

### From AI PC Browser
```
http://localhost:8080
```

### From Your Main PC
```
http://192.168.1.26:8080
```

## Configure Web UI Settings

1. Click the **Settings** button (⚙️) in the bottom left
2. Set **API Endpoint** to: `http://192.168.1.26:8811`
3. Choose **Default Mode** (recommended: Auto)
4. Click **Save Settings**
5. Verify **System Status** shows "Connected"

## Test Chat Functionality

1. Type a message in the input box
2. Select a mode (or use Auto)
3. Toggle "Remember conversation context" if desired
4. Click Send (or press Enter)
5. Watch for response from EDISON

## Troubleshooting Quick Fixes

### Web UI Won't Load
```bash
# Check service
sudo systemctl status edison-web
sudo journalctl -u edison-web -n 50

# Check if port is in use
sudo netstat -tulpn | grep 8080

# Restart service
sudo systemctl restart edison-web
```

### Shows "Offline" in Settings
```bash
# Check core service
sudo systemctl status edison-core

# Test API directly
curl http://localhost:8811/health

# Check CORS (should see CORSMiddleware in code)
grep -A5 "CORSMiddleware" /opt/edison/services/edison_core/app.py

# Restart core service
sudo systemctl restart edison-core
```

### Can't Access from Main PC
```bash
# Check firewall
sudo ufw status
sudo ufw allow 8080/tcp
sudo ufw allow 8811/tcp

# Verify binding
sudo netstat -tulpn | grep 8080
# Should show: 0.0.0.0:8080

# Check services are using 0.0.0.0
grep "host" /etc/systemd/system/edison-web.service
# Should show: --host 0.0.0.0
```

### Chat Not Responding
```bash
# Check core service logs
sudo journalctl -u edison-core -f

# Verify models are loaded
curl http://localhost:8811/health
# Should show: "fast_model": true, "deep_model": true

# Check model files exist
ls -lh /opt/edison/models/llm/
```

## Port Reference

| Service | Port | URL |
|---------|------|-----|
| Web UI | 8080 | http://192.168.1.26:8080 |
| Core API | 8811 | http://192.168.1.26:8811 |
| ComfyUI | 8188 | http://192.168.1.26:8188 |
| Coral API | 8808 | http://192.168.1.26:8808 |

## Service Logs

```bash
# View all EDISON service logs
sudo journalctl -u "edison-*" -f

# View specific service
sudo journalctl -u edison-web -f
sudo journalctl -u edison-core -f

# Last 100 lines
sudo journalctl -u edison-core -n 100
```

## Useful Commands

```bash
# Restart all services
sudo systemctl restart edison-coral edison-core edison-web edison-comfyui

# Stop all services
sudo systemctl stop edison-coral edison-core edison-web edison-comfyui

# Check permissions
ls -la /opt/edison
# Should be owned by: edison:edison

# Fix permissions if needed
sudo chown -R edison:edison /opt/edison
```

## What's New

✨ **Modern Web UI**
- Claude/ChatGPT-style interface
- Dark theme, responsive design
- Conversation history
- Multiple AI modes
- System status monitoring

🔧 **Installation Improvements**
- Pre-flight system checker
- Automated Coral TPU fixer
- Comprehensive documentation
- All known issues documented

🌐 **Network Access**
- CORS enabled
- All services on 0.0.0.0
- Accessible from any device

## Next Time: Fresh Installation

For future installations or setting up on another machine, use:

```bash
# 1. Check system
./scripts/check_system.sh

# 2. Install
./scripts/setup_ubuntu.sh

# 3. Optional: Fix Coral TPU
sudo ./scripts/fix_coral_tpu.sh

# 4. Download models
python3 scripts/download_models.py

# 5. Deploy services
sudo ./scripts/enable_services.sh

# 6. Access web UI
# http://YOUR_IP:8080
```

## Documentation

- **README.md** - Overview and quick start
- **INSTALL.md** - Complete installation guide  
- **TROUBLESHOOTING.md** - All known issues and solutions
- **DEPLOYMENT_SUMMARY.md** - What was built and why

---

**Your AI PC is ready! Open http://192.168.1.26:8080 and start chatting with EDISON!** 🚀
