"""
Voice assistant backend routes.

Endpoints:
  GET  /voice/config   — frontend discovers server capabilities
  POST /voice/stt      — optional fallback speech-to-text (stub → 501)
"""

import logging
from fastapi import APIRouter, HTTPException, UploadFile, File

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])


# ── Configuration endpoint ───────────────────────────────────────────────

@router.get("/config")
async def voice_config():
    """
    Tell the frontend what the server supports so it can pick
    Web Speech API (default) vs. server-side STT/TTS.
    """
    # Import config lazily to avoid circular imports
    try:
        from services.edison_core.app import config as app_config
    except ImportError:
        app_config = None

    voice_cfg = {}
    if app_config:
        voice_cfg = app_config.get("edison", {}).get("voice", {})

    server_stt = bool(voice_cfg.get("server_stt", False))
    server_tts = bool(voice_cfg.get("server_tts", False))
    voice_enabled = bool(voice_cfg.get("enabled", False))

    return {
        "voice_enabled": voice_enabled,
        "server_stt": server_stt,
        "server_tts": server_tts,
        "recommended_mode": "web_speech_api",
        "stt_endpoint": "/voice/stt" if server_stt else None,
        "tts_endpoint": None,  # TTS is handled client-side via Web Speech API
    }


# ── Speech-to-text fallback ─────────────────────────────────────────────

@router.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    """
    Fallback STT endpoint.  Accepts audio (webm/wav) and returns text.

    Currently returns 501 — implement with Whisper / faster-whisper when ready.
    """
    # Check if server STT is enabled
    try:
        from services.edison_core.app import config as app_config
    except ImportError:
        app_config = None

    voice_cfg = {}
    if app_config:
        voice_cfg = app_config.get("edison", {}).get("voice", {})

    if not voice_cfg.get("server_stt", False):
        raise HTTPException(
            status_code=501,
            detail={
                "message": "Server-side STT not configured. Use Web Speech API on the client.",
                "hint": "Set edison.voice.server_stt: true in config/edison.yaml and install whisper.",
            },
        )

    # ── Actual STT implementation (placeholder) ─────────────────────
    # When a model is available, read the audio and transcribe:
    #
    # audio_bytes = await file.read()
    # transcript = whisper_model.transcribe(audio_bytes)
    # return {"text": transcript, "language": "en"}

    raise HTTPException(
        status_code=501,
        detail="STT model not loaded. Install faster-whisper and enable in config.",
    )
