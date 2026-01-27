# Voice Mode Setup - Localhost Implementation

## Overview

Edison's voice mode uses external binaries (whisper.cpp for STT and Piper for TTS) to work on localhost without requiring SSL or Python downgrade.

**Features:**
- 🎤 Push-to-talk interface
- 📝 Live transcription display
- 🗣️ Natural TTS voice responses
- ⚡ Barge-in support (interrupt while speaking)
- 🔒 Works on http://localhost:8080 (no HTTPS needed)

## Installation

### 1. Install Voice Binaries

On your server, run the setup script:

```bash
cd /opt/edison
sudo bash scripts/setup_voice_binaries.sh
```

This will:
- Clone and compile whisper.cpp
- Download the base.en Whisper model (~140MB)
- Download Piper TTS binary and en_US voice (~60MB)
- Install to `/opt/edison/voice_binaries/`

**Time:** ~5-10 minutes depending on server

### 2. Restart Edison Services

```bash
sudo systemctl restart edison-core.service
sudo journalctl -u edison-core.service -f
```

Look for: `Voice mode available`

### 3. Access via Localhost

#### Option A: SSH Tunnel (recommended)

```bash
ssh -L 8080:localhost:8080 -L 8811:localhost:8811 mike@100.67.221.112
```

Then browse to: **http://localhost:8080**

#### Option B: Direct on Server

If running a browser directly on the server machine, just browse to:
**http://localhost:8080**

## Usage

### Push-to-Talk

1. Click and **hold** the microphone button (🎙️)
2. Speak your message
3. Release the button when done
4. Edison will:
   - Show your transcript
   - Think about the response
   - Speak the answer back

### Barge-In

If Edison is speaking and you want to interrupt:
1. Just click the mic button again
2. Edison will stop speaking immediately
3. Start speaking your new message

### Visual Feedback

- **Blue pulsing button**: Recording your voice
- **"Listening..." overlay**: Capturing audio
- **"Processing..."**: Transcribing speech
- **"Thinking..."**: Generating response
- **"Edison is speaking..."**: Playing audio response

## Technical Details

### Audio Pipeline

```
User Speech
  ↓ Web Audio API (16kHz PCM16)
  ↓ WebSocket (ws://localhost:8811)
  ↓ whisper.cpp (STT)
  ↓ Edison LLM (voice-optimized prompt)
  ↓ Piper TTS (synthesis)
  ↓ WebSocket (audio chunks)
  ↓ Web Audio API (playback)
User Hears Response
```

### Binaries Used

**Whisper.cpp:**
- Location: `/opt/edison/voice_binaries/whisper.cpp/main`
- Model: `ggml-base.en.bin` (English-only, ~140MB)
- Speed: ~1-2 seconds for 5-second audio
- Quality: Good accuracy for conversational speech

**Piper TTS:**
- Location: `/opt/edison/voice_binaries/piper`
- Voice: `en_US-lessac-medium` (clear American English)
- Speed: Real-time (~50ms/sentence)
- Quality: Natural-sounding neural TTS

### Protocol

**Client → Server:**
```json
{"type":"audio_pcm16","data_b64":"...","sample_rate":16000}
{"type":"stop_listening"}
{"type":"barge_in"}
```

**Server → Client:**
```json
{"type":"state","value":"listening|thinking|speaking|ready"}
{"type":"stt_final","text":"user transcript"}
{"type":"llm_text","text":"edison response"}
{"type":"tts_audio","data_b64":"...","sample_rate":22050}
{"type":"done"}
```

### Voice-Optimized Prompts

Edison uses shorter, conversational responses in voice mode:

```
System: You are Edison, a helpful AI assistant. 
Keep responses concise (1-3 sentences) unless asked for detail. 
Speak naturally and conversationally.
```

## Troubleshooting

### Voice button not appearing

Check console logs:
```
"⚠️ Voice mode requires localhost"
```

**Solution:** Access via localhost (SSH tunnel or local browser)

### "Voice mode not available"

```bash
# Check if binaries exist
ls -la /opt/edison/voice_binaries/whisper.cpp/main
ls -la /opt/edison/voice_binaries/piper

# Re-run setup if missing
cd /opt/edison
sudo bash scripts/setup_voice_binaries.sh
```

### Microphone not working

**Chrome/Edge:** Should work automatically on localhost  
**Firefox:** May need to explicitly allow microphone  
**Safari:** Not tested, may have issues

Check browser console for errors.

### Slow transcription

The base.en model is optimized for speed. If too slow:

```bash
# Use tiny.en model (faster but less accurate)
cd /opt/edison/voice_binaries/whisper.cpp
bash ./models/download-ggml-model.sh tiny.en

# Update voice.py to use tiny model:
# WHISPER_MODEL = VOICE_BIN_DIR / "whisper.cpp" / "models" / "ggml-tiny.en.bin"
```

### Audio playback glitchy

1. Check network - audio is streamed in real-time
2. Verify sample rate matches (22050 Hz for Piper)
3. Look for WebSocket errors in browser console

### Memory issues

Whisper.cpp uses ~500MB RAM when processing. If server is low on memory:

1. Close other applications
2. Use tiny model instead of base
3. Consider adding swap space

## Development

### Test STT only

```bash
# Record a test
arecord -d 5 -f S16_LE -r 16000 -c 1 test.wav

# Transcribe
/opt/edison/voice_binaries/whisper.cpp/main \
  -m /opt/edison/voice_binaries/whisper.cpp/models/ggml-base.en.bin \
  -f test.wav
```

### Test TTS only

```bash
echo "Hello, this is a test." | /opt/edison/voice_binaries/piper \
  --model /opt/edison/voice_binaries/voices/en_US-lessac-medium.onnx \
  --output_file test.wav

# Play
aplay test.wav
```

### WebSocket testing

```javascript
// Browser console
const ws = new WebSocket('ws://localhost:8811/voice/ws/<session_id>');
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

## Performance

**Typical latency (on decent hardware):**
- Microphone → Server: <50ms
- STT (5s audio): 1-2 seconds
- LLM response: 2-3 seconds  
- TTS synthesis: 0.5-1 second
- Audio playback: Real-time streaming

**Total turn latency: 4-7 seconds**

**Server requirements:**
- CPU: 4+ cores (whisper uses multiple threads)
- RAM: 2GB free (models + buffers)
- Disk: 500MB for models

## Comparison to ChatGPT Voice

**Advantages:**
- ✅ Fully offline/local
- ✅ No external API costs
- ✅ Privacy (no audio leaves your server)
- ✅ Works on localhost without SSL hassle

**Limitations:**
- ❌ Slightly higher latency (5-7s vs 2-3s)
- ❌ Push-to-talk only (no VAD auto-detection yet)
- ❌ No streaming TTS (generates full audio first)
- ❌ Requires localhost access

## Future Improvements

Potential enhancements:
1. Add silero-vad for automatic end-of-utterance detection
2. Implement streaming TTS (sentence-by-sentence)
3. Add voice activity visualization
4. Support conversation history in voice mode
5. Add different voice options
6. Implement wake word ("Hey Edison")

## Credits

- **Whisper**: OpenAI (via whisper.cpp by ggerganov)
- **Piper TTS**: Rhasspy project
- **Edison**: Your awesome local AI!
