"""
Voice Mode for Edison - Localhost implementation
Uses external binaries: whisper.cpp (STT) and Piper (TTS)
Works on http://localhost without SSL
"""

import asyncio
import base64
import wave
import tempfile
import subprocess
import os
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Paths to voice binaries
VOICE_BIN_DIR = Path("/opt/edison/voice_binaries")
WHISPER_BIN = VOICE_BIN_DIR / "whisper.cpp" / "main"
WHISPER_MODEL = VOICE_BIN_DIR / "whisper.cpp" / "models" / "ggml-base.en.bin"
PIPER_BIN = VOICE_BIN_DIR / "piper"
PIPER_VOICE = VOICE_BIN_DIR / "voices" / "en_US-lessac-medium.onnx"


class VoiceSession:
    """Manages a single voice conversation session"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.audio_buffer = []
        self.is_listening = False
        self.is_speaking = False
        self.cancel_flag = False
        
        # Process handles
        self.whisper_proc: Optional[subprocess.Popen] = None
        self.piper_proc: Optional[subprocess.Popen] = None
        
        # Tasks
        self.stt_task: Optional[asyncio.Task] = None
        self.llm_task: Optional[asyncio.Task] = None
        self.tts_task: Optional[asyncio.Task] = None
        
        logger.info(f"Voice session created: {session_id}")
    
    def add_audio(self, pcm_data: bytes):
        """Add audio chunk to buffer"""
        self.audio_buffer.append(pcm_data)
    
    def clear_audio_buffer(self):
        """Clear the audio buffer"""
        self.audio_buffer = []
    
    def cancel(self):
        """Cancel all ongoing operations"""
        logger.info(f"Cancelling session {self.session_id}")
        self.cancel_flag = True
        
        # Kill subprocesses
        if self.whisper_proc and self.whisper_proc.poll() is None:
            self.whisper_proc.kill()
        if self.piper_proc and self.piper_proc.poll() is None:
            self.piper_proc.kill()
        
        # Cancel tasks
        for task in [self.stt_task, self.llm_task, self.tts_task]:
            if task and not task.done():
                task.cancel()
    
    async def transcribe_audio(self) -> str:
        """Transcribe buffered audio using whisper.cpp"""
        if not self.audio_buffer:
            return ""
        
        try:
            # Create temp WAV file
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            
            with wave.open(temp_wav.name, 'wb') as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)
                
                for pcm_data in self.audio_buffer:
                    wav_file.writeframes(pcm_data)
            
            logger.info(f"Transcribing {len(self.audio_buffer)} audio chunks")
            
            # Run whisper.cpp
            cmd = [
                str(WHISPER_BIN),
                "-m", str(WHISPER_MODEL),
                "-f", temp_wav.name,
                "-t", "4",
                "-nt",
            ]
            
            self.whisper_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = self.whisper_proc.communicate(timeout=30)
            
            os.unlink(temp_wav.name)
            self.whisper_proc = None
            
            if self.cancel_flag:
                return ""
            
            # Parse transcript
            lines = stdout.strip().split('\n')
            transcript = ""
            for line in lines:
                line = line.strip()
                if line and not line.startswith('[') and not line.startswith('whisper_'):
                    transcript += line + " "
            
            transcript = transcript.strip()
            logger.info(f"Transcription: {transcript}")
            return transcript
            
        except Exception as e:
            logger.error(f"STT error: {e}")
            return ""
    
    async def synthesize_speech(self, text: str) -> bytes:
        """Synthesize speech using Piper TTS"""
        try:
            cmd = [
                str(PIPER_BIN),
                "--model", str(PIPER_VOICE),
                "--output_raw"
            ]
            
            logger.info(f"Synthesizing: {text[:50]}...")
            
            self.piper_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            stdout, stderr = self.piper_proc.communicate(
                input=text.encode('utf-8'),
                timeout=30
            )
            
            self.piper_proc = None
            
            if self.cancel_flag:
                return b""
            
            logger.info(f"Generated {len(stdout)} bytes of audio")
            return stdout
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return b""


class VoiceManager:
    """Manages all voice sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, VoiceSession] = {}
        self._check_binaries()
    
    def _check_binaries(self):
        """Check if voice binaries are installed"""
        if not WHISPER_BIN.exists():
            logger.warning(f"Whisper binary not found: {WHISPER_BIN}")
        if not PIPER_BIN.exists():
            logger.warning(f"Piper binary not found: {PIPER_BIN}")
    
    def create_session(self, session_id: str) -> VoiceSession:
        """Create a new voice session"""
        session = VoiceSession(session_id)
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[VoiceSession]:
        """Get existing session"""
        return self.sessions.get(session_id)
    
    def remove_session(self, session_id: str):
        """Remove and cleanup session"""
        if session_id in self.sessions:
            self.sessions[session_id].cancel()
            del self.sessions[session_id]


voice_manager = VoiceManager()
