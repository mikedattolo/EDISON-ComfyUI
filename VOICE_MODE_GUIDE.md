# EDISON Voice Mode - Complete Guide

## Overview

EDISON Voice Mode provides real-time voice interaction similar to ChatGPT's voice feature. Users can speak to EDISON, which listens, transcribes, generates responses, and speaks them back using natural-sounding text-to-speech.

## Features

### Core Capabilities
- **Push-to-Talk**: Hold space bar or click/hold the microphone button
- **Real-time Transcription**: See your words as you speak (partial transcripts)
- **Voice Activity Detection (VAD)**: Automatically detects when you stop speaking
- **Streaming TTS**: Audio playback starts before full response completes
- **Barge-in**: Interrupt EDISON's response to ask something new
- **Concise Responses**: Optimized for voice with 1-3 sentence replies

### Technical Features
- Low latency: < 1.5s from end of speech to response start
- GPU acceleration for STT and TTS (when available)
- Configurable models and parameters
- WebSocket bidirectional streaming
- Robust error handling and session management

## Architecture

### Backend Components

1. **Voice Module** (`services/edison_core/voice.py`)
   - `VoiceActivityDetector`: Silero VAD for speech detection
   - `SpeechToText`: faster-whisper for transcription
   - `TextToSpeech`: Coqui XTTS v2 for natural voice synthesis
   - `VoiceSession`: Manages state for each interaction

2. **FastAPI Endpoints** (`services/edison_core/app.py`)
   - `POST /voice/session`: Create new voice session
   - `WebSocket /voice/ws/{session_id}`: Bidirectional audio/text streaming
   - `POST /voice/cancel/{session_id}`: Cancel ongoing tasks

3. **Configuration** (`config/edison.yaml`)
   - VAD settings (silence detection, thresholds)
   - STT model selection (tiny to large)
   - TTS model and voice settings
   - Audio streaming parameters

### Frontend Components

1. **Voice UI** (`web/index.html`)
   - Microphone button with visual feedback
   - Voice panel showing state and transcript
   - Animated indicators for listening/thinking/speaking

2. **Voice Manager** (`web/app_voice.js`)
   - Audio capture and PCM encoding
   - WebSocket communication
   - Audio playback queue
   - State management

3. **Styling** (`web/styles.css`)
   - Voice button animations
   - Status indicators with pulse effects
   - Responsive mobile layout

## Usage

### Basic Usage

1. **Start Voice Mode**:
   - Click and hold the microphone button 🎙️
   - Or press and hold the Space bar (when not typing)

2. **Speak Your Query**:
   - Speak clearly into your microphone
   - Watch the partial transcript appear in real-time
   - Release the button when done speaking

3. **Listen to Response**:
   - EDISON processes your speech
   - Text response streams in
   - Audio playback begins automatically

4. **Barge-in** (interrupt):
   - Start speaking while EDISON is talking
   - Audio stops immediately
   - New query begins

### Keyboard Shortcuts

- **Space**: Push-to-talk (hold to speak, release to stop)
- **Click Cancel**: Stop current voice session

### Visual Feedback

- **🎙️ Ready**: System ready for input
- **🎙️ Listening** (pulsing): Capturing audio
- **🎙️ Thinking** (spinning): Processing transcript
- **🎙️ Speaking** (green pulse): Playing response

## Configuration

### Basic Settings (`config/edison.yaml`)

```yaml
voice:
  # VAD (Voice Activity Detection)
  silence_duration_ms: 700      # Silence before ending utterance
  min_speech_duration_ms: 250   # Minimum speech to trigger
  vad_threshold: 0.5            # Speech confidence (0.0-1.0)
  
  # STT (Speech-to-Text)
  stt_model: "base"             # tiny, base, small, medium, large
  stt_language: "en"            # Language code
  stt_device: "cuda"            # cuda or cpu
  
  # TTS (Text-to-Speech)
  tts_model: "xtts"             # xtts, piper, bark
  tts_voice: "default"          # Speaker name
  tts_sample_rate: 24000        # Output sample rate
  
  # Streaming
  partial_transcript_interval_ms: 500  # Partial update frequency
  tts_chunk_threshold: 60       # Chars before starting TTS
```

### Advanced Configuration

#### STT Model Selection

- **tiny**: Fastest, less accurate (use for testing)
- **base**: Good balance (recommended)
- **small**: Better accuracy, slower
- **medium**: High accuracy, requires more VRAM
- **large**: Best accuracy, slowest

#### TTS Options

**XTTS v2** (default):
- Most natural sounding
- Supports voice cloning with reference audio
- GPU accelerated
- ~24kHz output

**Piper** (coming soon):
- Faster than XTTS
- Lower quality
- CPU friendly
- Good for resource-constrained systems

**Bark** (coming soon):
- Highest quality
- Slowest
- Supports multiple languages
- Natural prosody and emotions

#### Voice Cloning

To use your own voice with XTTS:

1. Record 6-30 seconds of clear speech
2. Save as `my_voice.wav`
3. Update config:
```yaml
tts_speaker_wav: "/path/to/my_voice.wav"
```

## WebSocket Protocol

### Client → Server Messages

```json
// Audio chunk
{
  "type": "audio",
  "pcm16_base64": "base64_encoded_pcm",
  "sample_rate": 16000
}

// Barge-in (interrupt)
{
  "type": "barge_in"
}

// Manual end of utterance
{
  "type": "end_utterance"
}
```

### Server → Client Messages

```json
// State update
{
  "type": "state",
  "value": "listening|thinking|speaking|ready"
}

// Partial transcript
{
  "type": "stt_partial",
  "text": "Hello worl..."
}

// Final transcript
{
  "type": "stt_final",
  "text": "Hello world"
}

// LLM token
{
  "type": "llm_token",
  "text": "The"
}

// Audio chunk
{
  "type": "tts_audio",
  "pcm16_base64": "base64_encoded_audio",
  "sample_rate": 24000
}

// TTS complete
{
  "type": "tts_end"
}

// Error
{
  "type": "error",
  "message": "Error description"
}
```

## Installation

### Dependencies

```bash
# Install voice dependencies
pip install faster-whisper>=0.10.0
pip install TTS>=0.22.0
pip install numpy>=1.24.0
pip install soundfile>=0.12.1
pip install webrtcvad>=2.0.10  # Optional alternative VAD
```

### First Run

On first use, models will be downloaded automatically:
- **Silero VAD**: ~1.5MB (via torch.hub)
- **Whisper base**: ~140MB
- **XTTS v2**: ~1.8GB

Models are cached in:
- `~/.cache/torch/hub` (Silero VAD)
- `~/.cache/whisper` (Whisper models)
- `~/.cache/tts` (TTS models)

### GPU Requirements

**Minimum**:
- 4GB VRAM for base Whisper + XTTS
- 8GB VRAM for better quality models

**Recommended**:
- 8GB+ VRAM
- CUDA 11.8+
- cuDNN 8.x

**CPU Fallback**:
- Works without GPU
- Expect 2-5x slower processing
- Use smaller models (tiny/base)

## Troubleshooting

### Common Issues

**"Voice mode not available"**
- Check dependencies are installed
- Verify models downloaded successfully
- Check logs: `tail -f logs/edison.log`

**"Permission denied" (microphone)**
- Grant browser microphone access
- Check system audio settings
- Try HTTPS (required for some browsers)

**High latency**
- Use smaller STT model (base or tiny)
- Enable GPU acceleration
- Reduce `tts_chunk_threshold`
- Check CPU/GPU usage

**Poor audio quality**
- Increase `tts_sample_rate` (24000 or 22050)
- Use XTTS instead of Piper
- Check microphone quality
- Enable noise suppression in browser

**VAD not detecting speech**
- Lower `vad_threshold` (try 0.3)
- Reduce `min_speech_duration_ms`
- Speak louder/clearer
- Check microphone levels

### Debug Mode

Enable debug logging in `config/edison.yaml`:

```yaml
logging:
  level: "DEBUG"
```

Check logs:
```bash
tail -f logs/edison.log | grep voice
```

### Performance Monitoring

Check system resources:
```bash
# GPU usage
nvidia-smi -l 1

# CPU/Memory
htop

# Service status
systemctl status edison-core.service
```

## Performance Benchmarks

### Typical Latency (with GPU)

| Component | Time |
|-----------|------|
| VAD detection | ~100ms |
| STT (base model) | ~300-500ms |
| LLM response | ~500-1000ms |
| TTS first chunk | ~200-400ms |
| **Total to first audio** | **~1.1-2.0s** |

### Model Comparison

| STT Model | VRAM | Speed | WER |
|-----------|------|-------|-----|
| tiny | 1GB | 10x RT | 8-10% |
| base | 2GB | 7x RT | 5-7% |
| small | 3GB | 5x RT | 4-5% |
| medium | 5GB | 3x RT | 3-4% |
| large | 8GB | 2x RT | 2-3% |

*RT = Real-time (e.g., 10x RT = 10 seconds of audio processed in 1 second)*

## API Reference

### VoiceSession

```python
from services.edison_core.voice import VoiceSession, VoiceConfig

config = VoiceConfig(
    stt_model="base",
    tts_model="xtts",
    vad_threshold=0.5
)

session = VoiceSession("session_id", config)
await session.initialize()

# Process audio
result = session.process_audio_chunk(pcm_data)

# Get transcript
transcript = session.finalize_transcript()

# Cancel
session.cancel()
```

### VoiceConfig

```python
@dataclass
class VoiceConfig:
    # VAD settings
    vad_mode: str = "silero"
    silence_duration_ms: int = 700
    min_speech_duration_ms: int = 250
    vad_threshold: float = 0.5
    
    # STT settings
    stt_model: str = "base"
    stt_language: str = "en"
    stt_device: str = "cuda"
    
    # TTS settings
    tts_model: str = "xtts"
    tts_voice: str = "default"
    tts_speaker_wav: Optional[str] = None
    tts_sample_rate: int = 24000
    
    # Streaming settings
    partial_transcript_interval_ms: int = 500
    tts_chunk_threshold: int = 60
    
    # Voice mode prompt
    voice_system_prompt: str = "..."
```

## Future Enhancements

- [ ] Toggle mode (click once to start, click again to stop)
- [ ] Voice activity visualization (waveform)
- [ ] Multiple voice profiles
- [ ] Multilingual support
- [ ] Background noise suppression
- [ ] Echo cancellation
- [ ] Voice commands (e.g., "EDISON, stop")
- [ ] Conversation history replay
- [ ] Voice-to-voice translation
- [ ] Custom wake words

## Contributing

Voice mode improvements welcome! Areas for contribution:

1. **Model Integration**: Add Piper, Bark, or other TTS/STT models
2. **Performance**: Optimize streaming, reduce latency
3. **Quality**: Better VAD tuning, noise handling
4. **UI**: Enhanced visualizations, accessibility
5. **Testing**: Cross-browser, mobile, edge cases

## License

Same as EDISON project (see main README).

## Credits

Built with:
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Fast STT
- [Coqui TTS](https://github.com/coqui-ai/TTS) - XTTS v2
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice detection
- [FastAPI](https://fastapi.tiangolo.com/) - Backend framework
- [Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API) - Browser audio

## Support

- GitHub Issues: https://github.com/mikedattolo/EDISON-ComfyUI/issues
- Documentation: See `/docs` folder
- Logs: `logs/edison.log`
