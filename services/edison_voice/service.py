"""Voice service for speech-to-text and text-to-speech (optional)."""

import base64
import logging
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException, Body

logger = logging.getLogger(__name__)

app = FastAPI()

try:
    import whisper
except Exception:  # pragma: no cover
    whisper = None

try:
    import pyttsx3
except Exception:  # pragma: no cover
    pyttsx3 = None

_whisper_model = None
_tts_engine = None

@app.on_event("startup")
def startup():
    global _whisper_model, _tts_engine
    if whisper:
        try:
            _whisper_model = whisper.load_model("base")
            logger.info("Loaded Whisper base model")
        except Exception as e:
            logger.warning(f"Failed to load Whisper model: {e}")
    if pyttsx3:
        try:
            _tts_engine = pyttsx3.init()
        except Exception as e:
            logger.warning(f"Failed to init TTS engine: {e}")

@app.post("/voice-to-text")
async def voice_to_text(audio: UploadFile = File(...)):
    if _whisper_model is None:
        raise HTTPException(status_code=503, detail="Whisper model not available")

    data = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    result = _whisper_model.transcribe(tmp_path)
    return {"text": result.get("text", "").strip()}

@app.post("/text-to-voice")
async def text_to_voice(text: str = Body(..., embed=True)):
    if _tts_engine is None:
        raise HTTPException(status_code=503, detail="TTS engine not available")
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        out_path = tmp.name
    _tts_engine.save_to_file(text, out_path)
    _tts_engine.runAndWait()
    with open(out_path, "rb") as f:
        audio_data = f.read()
    return {"audio": base64.b64encode(audio_data).decode("ascii")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8809)
