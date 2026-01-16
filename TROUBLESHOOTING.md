# EDISON Troubleshooting Guide

Solutions to common issues encountered during installation and operation.

## Installation Issues

### Issue: Disk Space - Root Partition Full

**Symptoms:**
```
No space left on device
df -h shows 100% usage on /
```

**Solution:**

If you have a large physical disk but small LVM partition:

```bash
# Check physical disk size
lsblk

# Check LVM configuration
sudo pvdisplay
sudo vgdisplay
sudo lvdisplay

# Extend logical volume to use all available space
sudo lvextend -l +100%FREE /dev/ubuntu-vg/ubuntu-lv

# Resize filesystem
sudo resize2fs /dev/ubuntu-vg/ubuntu-lv

# Verify
df -h /
```

**Expected result:** Root partition should now use most of the physical disk space (e.g., 935GB on a 1TB disk).

---

### Issue: Python 3.12 Incompatibility with Coral TPU

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement pycoral
No matching distribution found for pycoral
```

**Root cause:** `pycoral` library only supports Python 3.9-3.11. It has not been updated for Python 3.12.

**Solution 1: Use Heuristic Classification (Recommended)**

EDISON automatically falls back to heuristic intent classification when `pycoral` is unavailable:

```bash
# pycoral is already removed from requirements.txt
# System will work without Coral TPU library
```

**Solution 2: Use Separate Python 3.11 Environment for Coral**

```bash
# Install Python 3.11
sudo apt-get install python3.11 python3.11-venv

# Create separate environment for coral service
python3.11 -m venv /opt/edison/.venv-coral
source /opt/edison/.venv-coral/bin/activate
pip install pycoral tflite-runtime

# Update edison-coral.service to use this venv
sudo nano /etc/systemd/system/edison-coral.service
# Change: Environment="PATH=/opt/edison/.venv-coral/bin:..."
```

---

### Issue: ComfyUI Directory Empty or Missing main.py

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: '.../ComfyUI/main.py'
```

**Solution:**

```bash
cd /opt/edison  # or your EDISON directory
rm -rf ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
```

This is now handled automatically by the updated `setup_ubuntu.sh` script.

---

### Issue: Coral TPU Kernel Module Build Failure

**Symptoms:**
```
ERROR: DKMS build failed
eventfd_signal: too many arguments
class_create: wrong number of arguments
```

**Root cause:** Kernel 6.8+ changed internal APIs that gasket driver uses.

**Solution:**

Use the automated fixer script:

```bash
sudo ./scripts/fix_coral_tpu.sh
```

This script automatically:
- Patches `eventfd_signal` calls
- Patches `class_create` calls
- Builds and installs kernel modules
- Sets up auto-load on boot

Manual alternative:
```bash
cd /tmp
git clone https://github.com/google/gasket-driver.git
cd gasket-driver

# Apply patches
sed -i 's/eventfd_signal(ctx->eventfd, 1);/eventfd_signal(ctx->eventfd);/g' src/apex_driver.c
sed -i 's/class_create(THIS_MODULE, class_name)/class_create(class_name)/g' src/gasket_core.c

# Build
cd src
make
sudo make install
sudo modprobe gasket
sudo modprobe apex
```

---

### Issue: RTX 5060 Ti (Blackwell Architecture) Not Supported

**Symptoms:**
```
CUDA error: no kernel image is available for execution on the device (Compute Capability: 12.0)
```

**Root cause:** RTX 5060 Ti uses compute capability `sm_120` (Blackwell). PyTorch 2.5.1 only supports up to `sm_90` (Hopper).

**Solution:**

**Option 1: Wait for PyTorch 2.6+ (Recommended)**
```bash
# PyTorch 2.6 will add Blackwell support
# For now, GPU will use CPU fallback
```

**Option 2: Try PyTorch Nightly**
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu126
```

**Option 3: Use Other GPUs**

The RTX 4080 Super and RTX 4070 (sm_89) work fine. Configure CUDA_VISIBLE_DEVICES:

```bash
# Edit service file
sudo nano /etc/systemd/system/edison-core.service

# Change to use only sm_89 GPUs:
Environment="CUDA_VISIBLE_DEVICES=0,1"  # Exclude GPU 2 (5060 Ti)

sudo systemctl daemon-reload
sudo systemctl restart edison-core
```

---

## Service Issues

### Issue: Service Won't Start

**Diagnosis:**

```bash
# Check status
sudo systemctl status edison-core

# View logs
sudo journalctl -u edison-core -n 50

# View live logs
sudo journalctl -u edison-core -f
```

**Common causes:**

1. **Missing models:**
   ```bash
   ls -lh /opt/edison/models/llm/
   # Should show .gguf files
   ```
   Solution: Download models with `./scripts/download_models.py`

2. **Permission issues:**
   ```bash
   sudo chown -R edison:edison /opt/edison
   ```

3. **Port already in use:**
   ```bash
   sudo netstat -tulpn | grep 8811
   ```
   Solution: Kill conflicting process or change port in config

4. **Python package missing:**
   ```bash
   /opt/edison/.venv/bin/python -c "import fastapi, llama_cpp"
   ```
   Solution: Reinstall requirements

---

### Issue: Network Access Blocked

**Symptoms:**
```
Connection refused when accessing from another device
curl: (7) Failed to connect to 192.168.1.26 port 8080
```

**Solution:**

1. **Check service binding:**
   ```bash
   sudo grep "host" /etc/systemd/system/edison-web.service
   # Should show: --host 0.0.0.0
   ```

2. **Check firewall:**
   ```bash
   sudo ufw status
   sudo ufw allow 8080/tcp
   sudo ufw allow 8811/tcp
   sudo ufw allow 8188/tcp
   sudo ufw allow 8808/tcp
   ```

3. **Verify service is listening:**
   ```bash
   sudo netstat -tulpn | grep 8080
   # Should show: 0.0.0.0:8080
   ```

---

### Issue: API Returns Wrong Field Error

**Symptoms:**
```
422 Unprocessable Entity
Field required: message
```

**Root cause:** API expects `message` field, not `text`.

**Solution:**

Correct API format:

```bash
curl -X POST http://localhost:8811/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello EDISON",
    "mode": "chat",
    "remember": false
  }'
```

Wrong format (do not use):
```bash
# ✗ Wrong:
"text": "Hello EDISON"

# ✓ Correct:
"message": "Hello EDISON"
```

---

### Issue: Health Endpoint Validation Error

**Symptoms:**
```
Internal Server Error
Validation error: repo_root field required
```

**Solution:**

This is fixed in the current version. Update your installation:

```bash
cd /workspaces/EDISON-ComfyUI
git pull
sudo cp -r services/edison_core /opt/edison/services/
sudo systemctl restart edison-core
```

---

## Model Issues

### Issue: Model Loading Extremely Slow

**Symptoms:**
- 72B model takes 60+ seconds to load
- High swap usage

**Diagnosis:**
```bash
free -h
# Check swap usage
```

**Solutions:**

1. **Add more RAM** (recommended for 72B model: 64GB+)

2. **Use only 14B model:**
   ```bash
   # Edit config
   nano /opt/edison/config/edison.yaml
   # Remove deep_model or set to same as fast_model
   ```

3. **Reduce context window:**
   ```python
   # In services/edison_core/app.py
   llm_deep = Llama(
       model_path=str(deep_model_path),
       n_ctx=2048,  # Reduce from 4096
       n_gpu_layers=-1,
       verbose=False
   )
   ```

---

### Issue: GPU Out of Memory

**Symptoms:**
```
CUDA error: out of memory
```

**Solutions:**

1. **Reduce GPU layers:**
   ```python
   # In services/edison_core/app.py
   llm_fast = Llama(
       model_path=str(fast_model_path),
       n_ctx=4096,
       n_gpu_layers=35,  # Reduce from -1 (all layers)
       verbose=False
   )
   ```

2. **Use CPU offloading:**
   ```python
   n_gpu_layers=0  # Use CPU only
   ```

3. **Use smaller quantization:**
   - Switch from Q4_K_M to Q3_K_M or Q2_K models

---

## Web UI Issues

### Issue: Web UI Shows "Offline" Status

**Symptoms:**
- Status indicator shows red "Offline"
- Settings modal shows "Cannot reach API"

**Diagnosis:**
```bash
# Test API directly
curl http://localhost:8811/health
```

**Solutions:**

1. **Check API endpoint in settings:**
   - Click Settings in web UI
   - Verify API Endpoint matches your setup
   - Default: `http://192.168.1.26:8811`
   - For local access: `http://localhost:8811`

2. **Check CORS:**
   - Ensure `CORSMiddleware` is enabled in `services/edison_core/app.py`
   - This is now added by default

3. **Check service is running:**
   ```bash
   sudo systemctl status edison-core
   ```

---

### Issue: Chat Messages Not Sending

**Symptoms:**
- Send button disabled
- No response after clicking send

**Check browser console:**
```
F12 → Console tab
Look for errors
```

**Common issues:**

1. **Empty message:** Type a message first
2. **API endpoint wrong:** Check settings
3. **CORS blocked:** Check browser console for CORS errors
4. **Network error:** Verify services are running

---

## Performance Issues

### Issue: Slow Response Time

**Diagnosis:**

Check which model is being used:
```bash
sudo journalctl -u edison-core -n 100 | grep "Generating response"
```

**Solutions:**

1. **Force fast model:**
   - Use "Chat" mode instead of "Auto" or "Deep"

2. **Reduce max_tokens:**
   ```python
   # In services/edison_core/app.py
   response = llm(
       full_prompt,
       max_tokens=1024,  # Reduce from 2048
       ...
   )
   ```

3. **Check GPU utilization:**
   ```bash
   nvidia-smi -l 1
   # Should show GPU usage during generation
   ```

---

## Data Issues

### Issue: Qdrant Vector Store Errors

**Symptoms:**
```
Qdrant connection error
Failed to store conversation
```

**Solution:**

```bash
# Check Qdrant directory
ls -lh /opt/edison/models/qdrant/

# Reset Qdrant (will lose stored conversations)
rm -rf /opt/edison/models/qdrant/collections
sudo systemctl restart edison-core
```

---

### Issue: Chat History Not Persisting

**Symptoms:**
- Chat history disappears after browser refresh

**Root cause:** Chat history is stored in browser's localStorage.

**Solution:**

Chat history is browser-specific and device-specific. To preserve:
1. Use same browser on same device
2. Don't clear browser data
3. Export important conversations manually (feature coming soon)

---

## Getting More Help

### Enable Debug Logging

```bash
# Edit service
sudo nano /etc/systemd/system/edison-core.service

# Add debug logging
Environment="LOG_LEVEL=DEBUG"

sudo systemctl daemon-reload
sudo systemctl restart edison-core

# View debug logs
sudo journalctl -u edison-core -f
```

### Collect System Info

```bash
# Run system check
./scripts/check_system.sh > system-info.txt

# Collect logs
sudo journalctl -u edison-core -n 200 > edison-core.log
sudo journalctl -u edison-web -n 200 > edison-web.log

# Collect config
cat /opt/edison/config/edison.yaml > config.txt
```

### Report Issues

Open an issue on GitHub with:
1. Output of `./scripts/check_system.sh`
2. Relevant log files
3. Steps to reproduce
4. Expected vs actual behavior

GitHub: https://github.com/mikedattolo/EDISON-ComfyUI/issues
