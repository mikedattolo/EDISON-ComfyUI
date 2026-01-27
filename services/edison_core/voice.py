"""
EDISON Voice Mode - Real-time voice interaction with VAD, STT, and TTS
"""

import asyncio
import base64
import io
import logging
import numpy as np
import torch
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Optional, Callable
import yaml

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_whisper_model = None
_tts_model = None
_vad_model = None


@dataclass
class VoiceConfig:
    """Voice mode configuration"""
    # VAD settings
    vad_mode: str = "silero"  # silero or webrtc
    silence_duration_ms: int = 700
    min_speech_duration_ms: int = 250
    vad_threshold: float = 0.5
    
    # STT settings
    stt_model: str = "base"  # tiny, base, small, medium, large
    stt_language: str = "en"
    stt_device: str = "cuda"  # cuda or cpu
    
    # TTS settings
    tts_model: str = "xtts"  # xtts, piper, bark
    tts_voice: str = "default"
    tts_speaker_wav: Optional[str] = None  # for XTTS voice cloning
    tts_sample_rate: int = 24000
    
    # Audio settings
    input_sample_rate: int = 16000
    chunk_duration_ms: int = 30
    
    # Streaming settings
    partial_transcript_interval_ms: int = 500
    tts_chunk_threshold: int = 60  # chars before starting TTS
    
    # Voice mode prompt
    voice_system_prompt: str = "You are EDISON in voice mode. Respond concisely in 1-3 sentences unless asked for detail. Be conversational and natural."


class VoiceActivityDetector:
    """Voice Activity Detection using Silero VAD"""
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.model = None
        self.sample_rate = config.input_sample_rate
        self.window_size = int(config.silence_duration_ms * self.sample_rate / 1000)
        self.speech_buffer = deque(maxlen=self.window_size)
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = 0
        
    def load_model(self):
        """Lazy load VAD model"""
        if self.model is None:
            global _vad_model
            if _vad_model is None:
                logger.info("Loading Silero VAD model...")
                try:
                    import torch
                    _vad_model, utils = torch.hub.load(
                        repo_or_dir='snakers4/silero-vad',
                        model='silero_vad',
                        force_reload=False,
                        onnx=False
                    )
                    logger.info("Silero VAD model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load Silero VAD: {e}")
                    raise
            self.model = _vad_model
            
    def process_audio(self, audio_chunk: np.ndarray) -> dict:
        """
        Process audio chunk and detect voice activity
        
        Returns:
            dict with keys: is_speech, confidence, end_of_utterance
        """
        if self.model is None:
            self.load_model()
        
        # Ensure audio is float32 and normalized
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32) / 32768.0
        
        # Silero VAD expects specific chunk sizes (512, 1024, 1536 samples for 16kHz)
        # We'll use 512 samples (32ms at 16kHz)
        chunk_size = 512
        if len(audio_chunk) < chunk_size:
            # Pad if too short
            audio_chunk = np.pad(audio_chunk, (0, chunk_size - len(audio_chunk)))
        
        # Process in chunks
        confidences = []
        for i in range(0, len(audio_chunk), chunk_size):
            chunk = audio_chunk[i:i+chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(chunk)
            
            # Get VAD prediction
            with torch.no_grad():
                confidence = self.model(audio_tensor, self.sample_rate).item()
            confidences.append(confidence)
        
        # Average confidence for this chunk
        avg_confidence = np.mean(confidences)
        is_speech = avg_confidence > self.config.vad_threshold
        
        # Update state
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
            if not self.is_speaking:
                # Check if we have enough speech to start
                if self.speech_frames * 32 >= self.config.min_speech_duration_ms:
                    self.is_speaking = True
                    logger.debug("Speech started")
        else:
            self.silence_frames += 1
            if self.is_speaking:
                # Check if silence is long enough to end utterance
                silence_ms = self.silence_frames * 32
                if silence_ms >= self.config.silence_duration_ms:
                    logger.debug(f"End of utterance detected ({silence_ms}ms silence)")
                    return {
                        "is_speech": False,
                        "confidence": avg_confidence,
                        "end_of_utterance": True
                    }
        
        return {
            "is_speech": is_speech,
            "confidence": avg_confidence,
            "end_of_utterance": False
        }
    
    def reset(self):
        """Reset VAD state"""
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.speech_buffer.clear()


class SpeechToText:
    """Speech-to-text using faster-whisper"""
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.model = None
        self.audio_buffer = []
        
    def load_model(self):
        """Lazy load Whisper model"""
        if self.model is None:
            global _whisper_model
            if _whisper_model is None:
                logger.info(f"Loading faster-whisper model: {self.config.stt_model}...")
                try:
                    from faster_whisper import WhisperModel
                    device = self.config.stt_device if torch.cuda.is_available() else "cpu"
                    compute_type = "float16" if device == "cuda" else "int8"
                    
                    _whisper_model = WhisperModel(
                        self.config.stt_model,
                        device=device,
                        compute_type=compute_type,
                        download_root=str(Path.home() / ".cache" / "whisper")
                    )
                    logger.info(f"Whisper model loaded on {device}")
                except Exception as e:
                    logger.error(f"Failed to load Whisper: {e}")
                    raise
            self.model = _whisper_model
    
    def add_audio(self, audio_chunk: np.ndarray):
        """Add audio chunk to buffer"""
        self.audio_buffer.append(audio_chunk)
    
    def transcribe_partial(self) -> Optional[str]:
        """Generate partial transcript from current buffer"""
        if not self.audio_buffer or self.model is None:
            return None
        
        try:
            # Concatenate all audio chunks
            audio = np.concatenate(self.audio_buffer)
            
            # Ensure float32 and normalized
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32) / 32768.0
            
            # Transcribe with beam_size=1 for speed
            segments, info = self.model.transcribe(
                audio,
                language=self.config.stt_language,
                beam_size=1,
                vad_filter=False,
                word_timestamps=False
            )
            
            # Collect segments
            text = " ".join([segment.text.strip() for segment in segments])
            return text if text else None
            
        except Exception as e:
            logger.error(f"Partial transcription error: {e}")
            return None
    
    def transcribe_final(self) -> str:
        """Generate final transcript and clear buffer"""
        if self.model is None:
            self.load_model()
        
        if not self.audio_buffer:
            return ""
        
        try:
            # Concatenate all audio chunks
            audio = np.concatenate(self.audio_buffer)
            
            # Ensure float32 and normalized
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32) / 32768.0
            
            # Transcribe with better quality settings
            segments, info = self.model.transcribe(
                audio,
                language=self.config.stt_language,
                beam_size=5,
                vad_filter=True,
                word_timestamps=False
            )
            
            # Collect segments
            text = " ".join([segment.text.strip() for segment in segments])
            logger.info(f"Final transcript: {text}")
            
            # Clear buffer
            self.audio_buffer = []
            
            return text
            
        except Exception as e:
            logger.error(f"Final transcription error: {e}")
            self.audio_buffer = []
            return ""
    
    def reset(self):
        """Reset STT state"""
        self.audio_buffer = []


class TextToSpeech:
    """Text-to-speech using Coqui XTTS v2"""
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.model = None
        
    def load_model(self):
        """Lazy load TTS model"""
        if self.model is None:
            global _tts_model
            if _tts_model is None:
                logger.info(f"Loading TTS model: {self.config.tts_model}...")
                try:
                    if self.config.tts_model == "xtts":
                        from TTS.api import TTS
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        _tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
                        logger.info(f"XTTS v2 loaded on {device}")
                    elif self.config.tts_model == "piper":
                        # Piper implementation would go here
                        raise NotImplementedError("Piper TTS not yet implemented")
                    elif self.config.tts_model == "bark":
                        # Bark implementation would go here
                        raise NotImplementedError("Bark TTS not yet implemented")
                    else:
                        raise ValueError(f"Unknown TTS model: {self.config.tts_model}")
                except Exception as e:
                    logger.error(f"Failed to load TTS: {e}")
                    raise
            self.model = _tts_model
    
    async def synthesize_streaming(
        self,
        text_generator: AsyncGenerator[str, None],
        chunk_callback: Callable[[bytes], None]
    ):
        """
        Synthesize speech from streaming text tokens
        
        Args:
            text_generator: Async generator yielding text tokens
            chunk_callback: Callback to send audio chunks
        """
        if self.model is None:
            self.load_model()
        
        text_buffer = ""
        sentence_endings = {'.', '!', '?', '\n'}
        
        async for token in text_generator:
            text_buffer += token
            
            # Check if we have a complete sentence or enough text
            should_synthesize = (
                any(text_buffer.endswith(end) for end in sentence_endings) or
                len(text_buffer) >= self.config.tts_chunk_threshold
            )
            
            if should_synthesize and text_buffer.strip():
                # Synthesize this chunk
                try:
                    audio_chunk = await self._synthesize_chunk(text_buffer.strip())
                    if audio_chunk:
                        chunk_callback(audio_chunk)
                    text_buffer = ""
                except Exception as e:
                    logger.error(f"TTS synthesis error: {e}")
        
        # Synthesize any remaining text
        if text_buffer.strip():
            try:
                audio_chunk = await self._synthesize_chunk(text_buffer.strip())
                if audio_chunk:
                    chunk_callback(audio_chunk)
            except Exception as e:
                logger.error(f"TTS final chunk error: {e}")
    
    async def _synthesize_chunk(self, text: str) -> Optional[bytes]:
        """Synthesize a single text chunk to audio"""
        try:
            # Run TTS in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            wav = await loop.run_in_executor(
                None,
                self._tts_sync,
                text
            )
            
            if wav is None:
                return None
            
            # Convert to PCM16 bytes
            audio_array = (wav * 32767).astype(np.int16)
            return audio_array.tobytes()
            
        except Exception as e:
            logger.error(f"Chunk synthesis error: {e}")
            return None
    
    def _tts_sync(self, text: str) -> Optional[np.ndarray]:
        """Synchronous TTS call"""
        try:
            if self.config.tts_model == "xtts":
                # Use speaker wav if provided, otherwise use default voice
                if self.config.tts_speaker_wav and Path(self.config.tts_speaker_wav).exists():
                    wav = self.model.tts(
                        text=text,
                        speaker_wav=self.config.tts_speaker_wav,
                        language=self.config.stt_language
                    )
                else:
                    # Use built-in speaker
                    wav = self.model.tts(
                        text=text,
                        speaker="Claribel Dervla",  # Default XTTS speaker
                        language=self.config.stt_language
                    )
                return np.array(wav)
            return None
        except Exception as e:
            logger.error(f"TTS sync error: {e}")
            return None


class VoiceSession:
    """Manages a single voice interaction session"""
    
    def __init__(self, session_id: str, config: VoiceConfig):
        self.session_id = session_id
        self.config = config
        
        # Components
        self.vad = VoiceActivityDetector(config)
        self.stt = SpeechToText(config)
        self.tts = TextToSpeech(config)
        
        # State
        self.state = "idle"  # idle, listening, thinking, speaking
        self.tasks: dict[str, asyncio.Task] = {}
        self.cancelled = False
        
        # Timestamps
        self.last_partial_time = 0
        
    async def initialize(self):
        """Initialize models (can be slow)"""
        logger.info(f"Initializing voice session {self.session_id}...")
        await asyncio.gather(
            asyncio.to_thread(self.vad.load_model),
            asyncio.to_thread(self.stt.load_model),
            asyncio.to_thread(self.tts.load_model)
        )
        logger.info(f"Voice session {self.session_id} initialized")
    
    def process_audio_chunk(self, pcm_data: bytes) -> dict:
        """
        Process incoming audio chunk
        
        Returns:
            dict with VAD results and optional partial transcript
        """
        # Convert PCM bytes to numpy array
        audio_chunk = np.frombuffer(pcm_data, dtype=np.int16)
        
        # Add to STT buffer
        self.stt.add_audio(audio_chunk)
        
        # Run VAD
        vad_result = self.vad.process_audio(audio_chunk)
        
        result = {
            "vad": vad_result,
            "partial_transcript": None
        }
        
        # Generate partial transcript periodically
        import time
        current_time = time.time() * 1000
        if current_time - self.last_partial_time >= self.config.partial_transcript_interval_ms:
            partial = self.stt.transcribe_partial()
            if partial:
                result["partial_transcript"] = partial
            self.last_partial_time = current_time
        
        return result
    
    def finalize_transcript(self) -> str:
        """Get final transcript and reset"""
        transcript = self.stt.transcribe_final()
        self.vad.reset()
        self.stt.reset()
        return transcript
    
    def cancel(self):
        """Cancel all ongoing tasks"""
        logger.info(f"Cancelling voice session {self.session_id}")
        self.cancelled = True
        for task_name, task in self.tasks.items():
            if not task.done():
                logger.debug(f"Cancelling task: {task_name}")
                task.cancel()
        self.tasks.clear()
    
    def reset(self):
        """Reset session state"""
        self.vad.reset()
        self.stt.reset()
        self.state = "idle"
        self.cancelled = False
        self.last_partial_time = 0


def load_voice_config(config_path: str = "config/edison.yaml") -> VoiceConfig:
    """Load voice configuration from YAML file"""
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return VoiceConfig()
    
    try:
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)
        
        voice_config = data.get('voice', {})
        return VoiceConfig(**voice_config)
    except Exception as e:
        logger.error(f"Error loading voice config: {e}")
        return VoiceConfig()
