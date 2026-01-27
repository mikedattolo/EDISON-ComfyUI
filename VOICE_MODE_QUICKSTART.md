# Voice Mode - Quick Start

## Installation

### 1. Install Dependencies

```bash
# On production machine
cd /opt/edison
sudo git pull origin main

# Install voice dependencies
pip install faster-whisper>=0.10.0
pip install TTS>=0.22.0
pip install numpy>=1.24.0
pip install soundfile>=0.12.1
pip install webrtcvad>=2.0.10
```

### 2. Restart Services

```bash
# Restart both core and web services
sudo systemctl restart edison-core.service
sudo systemctl restart edison-web.service

# Check status
sudo systemctl status edison-core.service
sudo systemctl status edison-web.service
```

### 3. First Run - Model Download

On first use, models will be downloaded automatically (~2GB total):
- Silero VAD: ~1.5MB
- Whisper base: ~140MB  
- XTTS v2: ~1.8GB

This happens when you first click the voice button. Wait for download to complete.

## Quick Test

### 1. Open EDISON Web UI

Navigate to: `http://YOUR_IP:8080`

### 2. Find the Voice Button

Look for the microphone icon 🎙️ next to the attachment button in the input area.

### 3. Test Push-to-Talk

1. **Click and hold** the mic button (or press and hold Space bar)
2. **Speak**: "Hello EDISON, what's the weather like?"
3. **Release** the button when done speaking
4. Watch the transcript appear
5. Listen to EDISON's response

### 4. Verify Features

- ✅ Partial transcript shows while speaking
- ✅ Final transcript appears when you stop
- ✅ EDISON responds with text streaming
- ✅ Audio plays automatically
- ✅ Status indicator changes (listening → thinking → speaking)

## Troubleshooting

### Voice button not appearing

```bash
# Check if app_voice.js is loaded
curl http://localhost:8080/app_voice.js

# Should return JavaScript code
# If 404, restart web service
sudo systemctl restart edison-web.service
```

### "Voice mode not available" error

```bash
# Check dependencies
pip list | grep -E "(whisper|TTS|numpy)"

# Should show:
# faster-whisper    0.10.x
# TTS              0.22.x
# numpy            1.24.x

# If missing, reinstall
pip install -r requirements.txt
```

### Models not downloading

```bash
# Check internet connection
ping github.com

# Check disk space
df -h

# Models cache locations:
ls -lh ~/.cache/torch/hub/          # Silero VAD
ls -lh ~/.cache/whisper/            # Whisper models
ls -lh ~/.cache/tts/                # TTS models
```

### High latency or timeouts

```bash
# Check GPU availability
nvidia-smi

# Should show GPU with free VRAM
# If no GPU, voice mode will use CPU (slower)

# For CPU-only, use smaller models in config/edison.yaml:
# stt_model: "tiny"  # Instead of "base"
```

### Microphone permission denied

- Grant microphone access in browser
- Chrome: Settings → Privacy → Site Settings → Microphone
- Firefox: Preferences → Privacy & Security → Permissions → Microphone

### No audio playback

- Check browser audio settings
- Verify volume is not muted
- Try different browser (Chrome recommended)
- Check Web Audio API support: https://caniuse.com/audio-api

## Configuration Quick Tweaks

### For faster response (lower quality)

Edit `config/edison.yaml`:

```yaml
voice:
  stt_model: "tiny"                  # Faster transcription
  tts_chunk_threshold: 30            # Start TTS sooner
  silence_duration_ms: 500           # Detect end sooner
```

### For better quality (slower)

```yaml
voice:
  stt_model: "small"                 # Better accuracy
  tts_sample_rate: 24000             # Higher quality audio
  vad_threshold: 0.6                 # More confident VAD
```

### For quieter environments

```yaml
voice:
  vad_threshold: 0.3                 # More sensitive
  min_speech_duration_ms: 150        # Faster trigger
```

## Logs and Debugging

### Check logs

```bash
# Real-time log watching
sudo journalctl -u edison-core.service -f | grep -i voice

# Last 100 lines
sudo journalctl -u edison-core.service -n 100 | grep -i voice

# Check for errors
sudo journalctl -u edison-core.service --since "5 minutes ago" | grep -i error
```

### Common log messages

**Good signs:**
```
Voice mode configured successfully
Voice session created: abc-123-def
WebSocket connected for session: abc-123-def
Loading faster-whisper model: base...
Whisper model loaded on cuda
XTTS v2 loaded on cuda
Speech started
End of utterance detected (700ms silence)
Final transcript: hello edison
```

**Issues:**
```
Voice module not available: No module named 'faster_whisper'
  → Run: pip install faster-whisper

Failed to load Whisper: CUDA out of memory
  → Use smaller model: stt_model: "tiny"

TTS synthesis error: ...
  → Check TTS installation: pip install TTS
```

## Performance Monitoring

### Check resource usage

```bash
# GPU usage (if available)
watch -n 1 nvidia-smi

# CPU and memory
htop

# Service resource usage
systemctl status edison-core.service
```

### Expected resource usage (with GPU):
- **VRAM**: 2-4GB (base Whisper + XTTS)
- **RAM**: 1-2GB
- **CPU**: 10-30% (during voice processing)

## Next Steps

Once basic voice mode is working:

1. **Customize voice**: See [VOICE_MODE_GUIDE.md](VOICE_MODE_GUIDE.md#voice-cloning)
2. **Tune latency**: Adjust VAD and TTS thresholds
3. **Test on mobile**: Voice mode is mobile-friendly
4. **Try different models**: Experiment with model sizes

## Getting Help

- Check [VOICE_MODE_GUIDE.md](VOICE_MODE_GUIDE.md) for detailed documentation
- Review logs: `sudo journalctl -u edison-core.service`
- Test basic chat first to isolate voice-specific issues
- Verify microphone works in browser (test on https://mictests.com)

## Known Limitations

- First voice session takes longer (model loading)
- Requires microphone permissions
- Some browsers require HTTPS for microphone access
- Background noise can affect VAD accuracy
- Large models require significant VRAM

## Success Criteria

Your voice mode is working if:
- ✅ Mic button appears and responds to clicks
- ✅ Partial transcript shows while speaking
- ✅ Final transcript appears within 1-2 seconds
- ✅ EDISON responds with relevant text
- ✅ Audio plays automatically
- ✅ Total latency < 3 seconds (end of speech to audio start)
