# Quick Test Guide - Model Selection Feature

## Testing the Implementation

### 1. Start/Restart EDISON

```bash
# If running as systemd service
sudo systemctl restart edison

# Or manually
cd /workspaces/EDISON-ComfyUI
python -m services.edison_core.app
```

### 2. Open Web UI

Open browser and navigate to:
```
http://localhost:8000
```

### 3. Test Model Selector Visibility

**Test Case 1: Auto Mode (Default)**
- Expected: Model selector is **hidden**
- Action: Check that the model dropdown is not visible

**Test Case 2: Switch to Chat Mode**
- Action: Click "💬 Chat" mode button
- Expected: Model selector **appears** below mode buttons

**Test Case 3: Switch Back to Auto**
- Action: Click "🤖 Auto" mode button
- Expected: Model selector **disappears**

### 4. Test Model Loading

**Check Browser Console:**
```
1. Open DevTools (F12)
2. Go to Console tab
3. Look for: "Loaded X available models"
```

**Expected Output:**
```
Loaded 3 available models
```

### 5. Test Model Selection

**Test Case: Select a Model**
1. Switch to "Chat" or "Deep" mode
2. Click the model dropdown
3. Verify available models are listed:
   - "Auto (System Default)"
   - List of GGUF models from your system

4. Select a model
5. Check console for: `Model changed to: <model_path>`

### 6. Test API Request

**Test Case: Send Message with Selected Model**

1. Select "Deep" mode
2. Choose a specific model from dropdown
3. Type a message: "Hello, what model are you?"
4. Send the message

**Check Network Tab:**
1. Open DevTools → Network tab
2. Find the POST request to `/chat/stream` or `/chat`
3. Check Request Payload:

```json
{
  "message": "Hello, what model are you?",
  "mode": "chat",
  "selected_model": "/path/to/your/selected/model.gguf"
}
```

**Expected:** `selected_model` field should be present

### 7. Check Backend Logs

```bash
# View real-time logs
journalctl -u edison -f

# Or if running manually, check console output
```

**Look for:**
```
INFO: User-selected model override: /path/to/model.gguf
```

### 8. Test Auto Mode (No Override)

1. Switch to "Auto" mode
2. Send a message
3. Check logs - should use automatic routing (no override message)

## Troubleshooting

### Model selector not appearing

**Check:**
```javascript
// In browser console
document.getElementById('modelSelector')
// Should return: <div class="model-selector" ...>
```

**Fix:** Refresh page (Ctrl+F5) to clear cache

### No models in dropdown

**Check API endpoint:**
```bash
curl http://localhost:8000/models/list
```

**Expected Response:**
```json
{
  "models": [
    {
      "name": "qwen2.5-14b-instruct-q4_k_m.gguf",
      "path": "/opt/edison/models/llm/qwen2.5-14b-instruct-q4_k_m.gguf",
      "size": "8GB"
    }
  ]
}
```

**If empty:**
- Check models exist in `/opt/edison/models/llm/`
- Check permissions: `ls -la /opt/edison/models/llm/`
- Restart EDISON

### selected_model not in API request

**Check browser console:**
```javascript
// Test if selectedModel is set
window.edisonChat.selectedModel
// Should return: "auto" or a model path
```

**Check code:**
- Verify `this.selectedModel` is initialized
- Verify change event listener is attached
- Check API call includes `selected_model` parameter

### Backend not using selected model

**Check logs for:**
```
INFO: User-selected model override: <model_path>
```

**If not present:**
- Verify `ChatRequest` has `selected_model` field
- Check both `/chat` and `/chat/stream` endpoints handle it
- Restart EDISON after code changes

## Expected Behavior Summary

| Action | Expected Result |
|--------|----------------|
| Load page | Model selector hidden (Auto mode) |
| Switch to Chat | Model selector appears |
| Switch to Auto | Model selector hides |
| Click dropdown | Shows all available models |
| Select model | Console logs change |
| Send message | API request includes selected_model |
| Backend receives | Logs "User-selected model override" |

## Manual API Test

Test the API directly:

```bash
# Test with model selection
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello!",
    "mode": "chat",
    "selected_model": "/opt/edison/models/llm/qwen2.5-14b-instruct-q4_k_m.gguf"
  }'

# Test without model selection (auto routing)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello!",
    "mode": "chat"
  }'
```

## Success Criteria

✅ Model selector shows/hides based on mode  
✅ Models load from `/models/list` API  
✅ Dropdown populates with available models  
✅ Selecting model updates `selectedModel` variable  
✅ API requests include `selected_model` parameter  
✅ Backend logs model override when selected  
✅ Backend uses selected model (or falls back to auto)  

## Quick Debug Commands

```bash
# Check if models exist
ls -lh /opt/edison/models/llm/*.gguf
ls -lh /mnt/models/llm/*.gguf  # If external drive set up

# Check EDISON is running
sudo systemctl status edison

# View recent logs
journalctl -u edison -n 50

# Test /models/list endpoint
curl http://localhost:8000/models/list | jq

# Check config
cat /workspaces/EDISON-ComfyUI/config/edison.yaml | grep -A5 "large_model_path"
```

## Next Steps After Successful Test

1. Download additional models (see `MODEL_DOWNLOAD_GUIDE.md`)
2. Set up external drive if needed (see `EXTERNAL_DRIVE_SETUP.md`)
3. Test with multiple models
4. Experiment with different modes and model combinations
5. Monitor VRAM usage with large models

## Documentation Reference

- [MODEL_SELECTION_COMPLETE.md](MODEL_SELECTION_COMPLETE.md) - Full implementation details
- [EXTERNAL_DRIVE_SETUP.md](EXTERNAL_DRIVE_SETUP.md) - External drive configuration
- [MODEL_DOWNLOAD_GUIDE.md](MODEL_DOWNLOAD_GUIDE.md) - Download commands for large models
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - General troubleshooting guide
