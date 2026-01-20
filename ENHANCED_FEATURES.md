# EDISON Enhanced Features - Deployment Guide

## 🎉 New Features Added (Commit 9b6214b)

### 1. **File Upload & Document Reading** 📄
- **UI**: + button next to message input for attaching files
- **Supported**: .txt, .pdf, .doc, .docx, .md, .json, .csv, .py, .js, .html, .css
- **Limit**: 10MB per file
- **Backend**: `/upload-document` endpoint stores files in RAG for context

### 2. **Hardware Monitoring Widget** 📊
- **Access**: Monitor button in sidebar footer
- **Shows**: 
  - CPU usage %
  - GPU usage % (NVIDIA only)
  - RAM usage (GB)
  - CPU temperature
- **Updates**: Every 2 seconds
- **Backend**: `/system/stats` endpoint using psutil

### 3. **Work Mode with AI Desktop** 🖥️
- **Mode Button**: New "Work" mode alongside Chat/Deep/Code/Agent
- **Features**:
  - Current task display
  - Live search results visualization
  - Uploaded documents panel
  - Thinking process log
- **Use Case**: Visual workflow for complex multi-step tasks

### 4. **Chat History Search** 🔍
- **Location**: Search bar above chat history in sidebar
- **Function**: Real-time filtering of conversations
- **Performance**: Instant client-side search

### 5. **Dynamic Chat Naming** 🏷️
- **Auto-Generated**: AI creates descriptive titles for chats
- **Backend**: `/generate-title` endpoint uses LLM
- **Fallback**: Smart extraction from first message if AI unavailable
- **Example**: "New Chat" → "Python sorting algorithms"

## 🚀 Deployment Instructions

### On Edison Server:

```bash
cd /opt/edison

# Pull latest code
sudo git pull origin main

# Install new dependency
source .venv/bin/activate
pip install psutil
deactivate

# Restart services
sudo systemctl restart edison-core
sudo systemctl restart edison-web

# Verify
sudo journalctl -u edison-core -f
```

### Expected Logs:
```
✓ Fast model loaded successfully
✓ RAG system initialized
✓ Web search tool initialized (ddgs)
EDISON Core Service ready
```

### Access Web UI:
```
http://192.168.1.26:8080
```

## 🎨 UI Features Overview

### New Buttons & Controls:
1. **Attach Button** (+): Upload files - left side of message input
2. **Monitor Button** (🖥️): Toggle hardware stats - sidebar footer
3. **Work Mode** (🖥️): Task visualization - mode selector
4. **Chat Search**: Filter conversations - top of sidebar

### Visual Enhancements:
- File chips show attached documents
- Hardware stats with gradient bars
- Work desktop with 4-panel layout
- Search highlights in sidebar

## 🔧 Configuration

### Environment Variables (optional):
```yaml
# config/edison.yaml
edison:
  features:
    file_upload_max_size: 10485760  # 10MB
    hardware_monitor_interval: 2000  # ms
    chat_title_max_length: 50
```

### Browser Requirements:
- Modern browser (Chrome 90+, Firefox 88+, Safari 14+)
- JavaScript enabled
- Local storage enabled (for chat history)

## 📝 Usage Examples

### 1. File Upload Workflow:
```
1. Click + button
2. Select document(s)
3. Files appear as chips below input
4. Type question about the files
5. AI has access to file contents via RAG
```

### 2. Work Mode:
```
1. Select "Work" mode
2. Desktop view opens
3. Type complex task
4. Watch AI's workflow:
   - Task breakdown
   - Search progress
   - Document references
   - Thinking steps
```

### 3. Chat Search:
```
1. Type keywords in search bar
2. Only matching chats show
3. Clear to see all again
```

## 🐛 Troubleshooting

### Issue: Hardware monitor shows 0%
**Fix**: 
```bash
sudo pip install psutil pynvml
sudo systemctl restart edison-core
```

### Issue: File upload fails
**Check**: 
- File size < 10MB
- Supported file type
- Browser console for errors

### Issue: Work mode doesn't open
**Check**:
- JavaScript errors in console
- app_features.js loaded
- Browser compatibility

### Issue: Chat search not working
**Check**:
- Chat history has items
- Search input focused
- Try clearing and retyping

## 📊 API Endpoints Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/upload-document` | POST | Store uploaded files in RAG |
| `/generate-title` | POST | Create smart chat titles |
| `/system/stats` | GET | Hardware monitoring data |
| `/chat` | POST | Main chat (includes file context) |
| `/search` | POST | Web search (used in work mode) |

## 🔐 Security Notes

- File uploads stored in memory/RAG only
- No persistent file storage on disk
- 10MB size limit enforced
- Supported file types whitelisted
- CORS enabled for local network only

## 🎯 Next Steps

### Potential Enhancements:
1. PDF text extraction (pdfplumber)
2. Image file support with OCR
3. Audio transcription (whisper)
4. Export work mode logs
5. Hardware alerts/notifications
6. Multi-file batch upload
7. Drag-and-drop file support

### Performance Optimization:
- Lazy load hardware stats
- Cache chat titles
- Debounce search input
- Virtual scrolling for large chat histories

## 📚 Dependencies Added

```
psutil>=5.9.0  # System monitoring
```

All other features use existing dependencies.

## ✅ Testing Checklist

- [ ] File upload works for all supported types
- [ ] Hardware monitor shows live stats
- [ ] Work mode opens and displays tasks
- [ ] Chat search filters correctly
- [ ] Chat titles auto-generate
- [ ] All modes still work (chat/deep/agent/code)
- [ ] Web search still functional
- [ ] RAG remembers context
- [ ] Sidebar toggle works
- [ ] Mobile responsive

---

**Version**: 1.1.0  
**Commit**: 9b6214b  
**Date**: January 20, 2026  
**Status**: ✅ Ready for deployment
