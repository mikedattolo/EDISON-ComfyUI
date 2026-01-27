# API Recovery Guide

## Current Issue: Port 8811 In Use

The error `[Errno 98] address already in use on 0.0.0.0:8811` indicates an old Python process is still holding the port.

### Quick Fix

On your production server (`/opt/edison` or wherever it's deployed), run:

```bash
# Kill any Python processes
sudo killall python python3

# Wait a moment for cleanup
sleep 2

# Restart the services
sudo systemctl restart edison-core.service
sudo systemctl restart edison-web.service

# Verify services are running
sudo systemctl status edison-core.service
sudo systemctl status edison-web.service
```

### Verify API is Working

Check that the API responds:
```bash
curl http://localhost:8811/health
# Should return: {"status":"online",...}
```

### Verify Web UI is Working

Open your browser to:
```
http://localhost:8080
```

You should see the chat interface without the microphone icon.

### Test Chat

Type in the chat: `hi`

Expected response: EDISON should respond with a greeting.

### Test Image Generation

Type: `generate an image of a cat`

Expected: Image should be generated and displayed in the interface.

## If Services Still Won't Start

Check logs for detailed errors:
```bash
# For edison-core service
journalctl -u edison-core -f --no-pager

# For edison-web service  
journalctl -u edison-web -f --no-pager
```

## Alternative: Manual Start (if systemd not working)

```bash
cd /opt/edison

# Start core service in background
python -m services.edison_core.app &

# Start web service in background
python -m services.edison_web.service &
```

Then test with curl or browser.

## What Changed

- ✅ Voice mode completely removed
- ✅ All UI references cleaned up
- ✅ Configuration simplified
- ✅ Dependencies reduced

The API should now be cleaner and faster without voice processing overhead.
