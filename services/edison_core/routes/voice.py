"""
Voice assistant backend — Server-side STT & TTS.

STT: faster-whisper  (GPU-accelerated Whisper via CTranslate2)
TTS: edge-tts        (Microsoft Neural voices — natural, expressive, high quality)

Endpoints:
  GET  /voice/config   — frontend discovers server STT/TTS capabilities
  POST /voice/stt      — speech-to-text  (upload audio → JSON {text, language, …})
  POST /voice/tts      — text-to-speech  (JSON {text, voice?} → audio/mpeg stream)
  GET  /voice/voices   — list available TTS voices
"""

import asyncio
import logging
import re
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])

# ── Lazy-loaded STT model ─────────────────────────────────────────────
_stt_model = None
_stt_checked = False


def _get_voice_config() -> dict:
    """Read voice config from edison.yaml."""
    try:
        from services.edison_core.app import config as app_config
    except ImportError:
        try:
            from ..app import config as app_config
        except Exception:
            app_config = None
    cfg = {}
    if app_config:
        cfg = app_config.get("edison", {}).get("voice", {})
    return cfg


def _ensure_stt():
    """Lazy-load the faster-whisper model on first STT request."""
    global _stt_model, _stt_checked
    if _stt_checked:
        return _stt_model
    _stt_checked = True

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.warning(
            "faster-whisper not installed — server STT disabled.  "
            "Install with:  pip install faster-whisper"
        )
        return None

    cfg = _get_voice_config()
    model_size = cfg.get("stt_model", "base")
    device = cfg.get("stt_device", "auto")

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    compute_type = "float16" if device == "cuda" else "int8"

    try:
        logger.info(f"Loading faster-whisper '{model_size}' on {device} ({compute_type}) ...")
        _stt_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info(f"STT ready: faster-whisper {model_size} ({device})")
        return _stt_model
    except Exception as e:
        logger.error(f"Failed to load STT model: {e}")
        return None


def _check_tts_available() -> bool:
    """Return True if edge-tts is importable."""
    try:
        import edge_tts  # noqa: F401
        return True
    except ImportError:
        return False


# ── Configuration endpoint ────────────────────────────────────────────

@router.get("/config")
async def voice_config():
    """Tell the frontend what the server supports so it can choose mode."""
    stt_ok = False
    try:
        from faster_whisper import WhisperModel  # noqa: F401
        stt_ok = True
    except ImportError:
        pass

    tts_ok = _check_tts_available()
    cfg = _get_voice_config()

    return {
        "voice_enabled": True,
        "server_stt": stt_ok,
        "server_tts": tts_ok,
        "recommended_mode": "server" if (stt_ok or tts_ok) else "web_speech_api",
        "stt_endpoint": "/voice/stt" if stt_ok else None,
        "tts_endpoint": "/voice/tts" if tts_ok else None,
        "tts_voice": cfg.get("tts_voice", "en-US-GuyNeural"),
        "stt_model": cfg.get("stt_model", "base"),
    }


# ── Speech-to-Text ───────────────────────────────────────────────────

@router.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    """
    Transcribe uploaded audio to text using faster-whisper.
    Accepts webm, wav, mp3, ogg, flac — anything ffmpeg can decode.
    """
    model = _ensure_stt()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Server STT not available. Install: pip install faster-whisper",
        )

    audio_data = await file.read()
    if not audio_data:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Save to temp file (faster-whisper needs a file path)
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()

        def _transcribe():
            segs, info = model.transcribe(
                tmp_path,
                beam_size=5,
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            # segments is a generator — consume it here
            text = " ".join(s.text.strip() for s in segs).strip()
            return text, info

        text, info = await loop.run_in_executor(None, _transcribe)

        return {
            "text": text,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 2),
        }
    except Exception as e:
        logger.error(f"STT transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass


# ── Text-to-Speech ───────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None


@router.post("/tts")
async def text_to_speech(req: TTSRequest):
    """
    Generate natural speech from text using edge-tts (Microsoft Neural voices).
    Returns streaming audio/mpeg for immediate playback.

    Popular voices:
        en-US-GuyNeural     — male, natural, warm (default EDISON voice)
        en-US-DavisNeural   — male, casual, friendly
        en-US-JennyNeural   — female, natural
        en-US-AriaNeural    — female, expressive
        en-GB-RyanNeural    — British male
    """
    if not _check_tts_available():
        raise HTTPException(
            status_code=503,
            detail="Server TTS not available. Install: pip install edge-tts",
        )

    import edge_tts

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    # Clean text for speech (strip markdown / code blocks)
    clean = text
    clean = re.sub(r"```[\s\S]*?```", " code block ", clean)
    clean = re.sub(r"`[^`]+`", "", clean)
    clean = re.sub(r"[#*_~]", "", clean)
    clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)
    clean = re.sub(r"\n+", ". ", clean)
    clean = clean[:3000]

    cfg = _get_voice_config()
    voice = req.voice or cfg.get("tts_voice", "en-US-GuyNeural")

    try:
        communicate = edge_tts.Communicate(clean, voice)

        async def audio_stream():
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]

        return StreamingResponse(
            audio_stream(),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline", "Cache-Control": "no-cache"},
        )
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


# ── Voice listing ────────────────────────────────────────────────────

@router.get("/voices")
async def list_voices():
    """List curated English neural voices for the voice selector UI."""
    voices = [
        {"id": "en-US-GuyNeural", "name": "Guy", "lang": "en-US", "gender": "Male", "style": "Natural, warm"},
        {"id": "en-US-DavisNeural", "name": "Davis", "lang": "en-US", "gender": "Male", "style": "Casual, friendly"},
        {"id": "en-US-JasonNeural", "name": "Jason", "lang": "en-US", "gender": "Male", "style": "Professional"},
        {"id": "en-US-TonyNeural", "name": "Tony", "lang": "en-US", "gender": "Male", "style": "Friendly, approachable"},
        {"id": "en-US-JennyNeural", "name": "Jenny", "lang": "en-US", "gender": "Female", "style": "Natural"},
        {"id": "en-US-AriaNeural", "name": "Aria", "lang": "en-US", "gender": "Female", "style": "Expressive"},
        {"id": "en-US-NancyNeural", "name": "Nancy", "lang": "en-US", "gender": "Female", "style": "Warm, mature"},
        {"id": "en-US-SaraNeural", "name": "Sara", "lang": "en-US", "gender": "Female", "style": "Young, energetic"},
        {"id": "en-GB-RyanNeural", "name": "Ryan", "lang": "en-GB", "gender": "Male", "style": "British"},
        {"id": "en-GB-SoniaNeural", "name": "Sonia", "lang": "en-GB", "gender": "Female", "style": "British"},
        {"id": "en-AU-WilliamNeural", "name": "William", "lang": "en-AU", "gender": "Male", "style": "Australian"},
        {"id": "en-IN-PrabhatNeural", "name": "Prabhat", "lang": "en-IN", "gender": "Male", "style": "Indian English"},
    ]

    cfg = _get_voice_config()
    current = cfg.get("tts_voice", "en-US-GuyNeural")

    return {"voices": voices, "current": current}
