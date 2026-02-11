"""
Music Generation Service for EDISON
Generates music from text prompts, lyrics, and style descriptions
using Meta's MusicGen (via audiocraft) or fallback to API-based services.
"""

import logging
import json
import uuid
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
MUSIC_OUTPUT_DIR = REPO_ROOT / "outputs" / "music"
MUSIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class MusicGenerationService:
    """
    Music generation using Meta's MusicGen (audiocraft library).
    Supports text-to-music with genre, mood, instruments, tempo, and lyrics.
    """

    def __init__(self, config: Dict[str, Any]):
        music_cfg = config.get("edison", {}).get("music", {})
        self.model_size = music_cfg.get("model_size", "medium")  # small, medium, large, melody
        self.default_duration = music_cfg.get("default_duration", 15)  # seconds
        self.max_duration = music_cfg.get("max_duration", 60)
        self.sample_rate = 32000
        self._model = None
        self._model_loaded = False
        self._loading = False
        logger.info(f"✓ Music generation service initialized (model: musicgen-{self.model_size})")

    # ── Model management ─────────────────────────────────────────────────

    def _load_model(self):
        """Lazy-load MusicGen model on first use."""
        if self._model_loaded and self._model is not None:
            return True

        if self._loading:
            # Wait for another thread to finish loading
            for _ in range(60):
                if self._model_loaded:
                    return True
                time.sleep(1)
            return False

        self._loading = True
        try:
            from audiocraft.models import MusicGen
            model_name = f"facebook/musicgen-{self.model_size}"
            logger.info(f"Loading MusicGen model: {model_name}...")
            self._model = MusicGen.get_pretrained(model_name)
            self._model.set_generation_params(duration=self.default_duration)
            self._model_loaded = True
            logger.info(f"✓ MusicGen model loaded: {model_name}")
            return True
        except ImportError:
            logger.error(
                "❌ audiocraft not installed. Install with: pip install audiocraft"
            )
            self._loading = False
            return False
        except Exception as e:
            logger.error(f"❌ Failed to load MusicGen: {e}")
            self._loading = False
            return False

    def _unload_model(self):
        """Unload model to free VRAM."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_loaded = False
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("✓ MusicGen model unloaded")

    # ── Prompt builder ───────────────────────────────────────────────────

    @staticmethod
    def build_music_prompt(
        description: str = "",
        genre: str = "",
        mood: str = "",
        instruments: str = "",
        tempo: str = "",
        style: str = "",
        lyrics: str = "",
        reference_artist: str = "",
    ) -> str:
        """
        Build an optimized MusicGen prompt from structured fields.
        MusicGen responds well to descriptive natural-language prompts.
        """
        parts = []

        if genre:
            parts.append(genre)
        if mood:
            parts.append(f"{mood} mood")
        if style:
            parts.append(f"{style} style")
        if tempo:
            parts.append(f"{tempo} tempo")
        if instruments:
            parts.append(f"featuring {instruments}")
        if reference_artist:
            parts.append(f"inspired by {reference_artist}")
        if description:
            parts.append(description)

        prompt = ", ".join(parts) if parts else "upbeat electronic music"

        # MusicGen doesn't handle lyrics directly, but we add them
        # as a textual hint for mood/theme
        if lyrics:
            # Extract theme from lyrics for prompt augmentation
            theme_hint = lyrics[:200].replace("\n", " ").strip()
            prompt += f". Theme from lyrics: {theme_hint}"

        return prompt

    # ── Generation ───────────────────────────────────────────────────────

    def generate_music(
        self,
        prompt: str = "",
        description: str = "",
        genre: str = "",
        mood: str = "",
        instruments: str = "",
        tempo: str = "",
        style: str = "",
        lyrics: str = "",
        reference_artist: str = "",
        duration: int = 15,
        melody_audio_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate music from a text prompt or structured parameters.

        Args:
            prompt:            Free-form text prompt (highest priority)
            description:       Additional description
            genre:             e.g. "rock", "hip-hop", "classical", "electronic"
            mood:              e.g. "happy", "melancholic", "energetic", "chill"
            instruments:       e.g. "guitar, drums, piano"
            tempo:             e.g. "fast", "slow", "120 BPM"
            style:             e.g. "lo-fi", "orchestral", "80s synth"
            lyrics:            Song lyrics (used for theme extraction)
            reference_artist:  e.g. "inspired by Daft Punk"
            duration:          Length in seconds (max 60)
            melody_audio_path: Path to a melody/audio file to condition on
        """
        duration = min(duration, self.max_duration)

        # Build/refine the prompt
        if not prompt:
            prompt = self.build_music_prompt(
                description=description,
                genre=genre,
                mood=mood,
                instruments=instruments,
                tempo=tempo,
                style=style,
                lyrics=lyrics,
                reference_artist=reference_artist,
            )

        logger.info(f"Generating music: '{prompt}' ({duration}s)")

        # Try audiocraft (local GPU generation)
        result = self._generate_with_audiocraft(prompt, duration, melody_audio_path)
        if result.get("ok"):
            return result

        # If audiocraft fails, return instructions for manual setup
        return {
            "ok": False,
            "error": (
                "MusicGen model not available. To enable music generation:\n"
                "1. Install audiocraft: pip install audiocraft\n"
                "2. Ensure you have a CUDA GPU with sufficient VRAM (4GB+ for small, 8GB+ for medium)\n"
                "3. The model will be auto-downloaded on first use (~3.3GB for medium)"
            ),
            "prompt_used": prompt,
            "duration": duration,
        }

    def _generate_with_audiocraft(
        self, prompt: str, duration: int, melody_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate music using Meta's audiocraft MusicGen."""
        try:
            if not self._load_model():
                return {"ok": False, "error": "MusicGen model not loaded"}

            import torch
            import torchaudio

            self._model.set_generation_params(duration=duration)

            output_id = uuid.uuid4().hex[:12]

            if melody_path and Path(melody_path).exists() and self.model_size == "melody":
                # Melody-conditioned generation
                melody_waveform, sr = torchaudio.load(melody_path)
                # Resample if needed
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    melody_waveform = resampler(melody_waveform)
                # Generate conditioned on melody
                wav = self._model.generate_with_chroma(
                    descriptions=[prompt],
                    melody_wavs=melody_waveform.unsqueeze(0),
                    melody_sample_rate=self.sample_rate,
                    progress=True,
                )
            else:
                # Standard text-to-music
                wav = self._model.generate(
                    descriptions=[prompt],
                    progress=True,
                )

            # Save output
            output_path = MUSIC_OUTPUT_DIR / f"EDISON_music_{output_id}.wav"
            # wav shape: (batch, channels, samples)
            torchaudio.save(str(output_path), wav[0].cpu(), self.sample_rate)

            # Also save as MP3 if ffmpeg available
            mp3_path = None
            try:
                import subprocess
                mp3_file = MUSIC_OUTPUT_DIR / f"EDISON_music_{output_id}.mp3"
                result = subprocess.run(
                    ["ffmpeg", "-y", "-i", str(output_path), "-b:a", "192k", str(mp3_file)],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    mp3_path = str(mp3_file)
            except Exception:
                pass

            file_size = output_path.stat().st_size
            duration_actual = wav.shape[-1] / self.sample_rate

            return {
                "ok": True,
                "data": {
                    "wav_path": str(output_path),
                    "mp3_path": mp3_path,
                    "filename": output_path.name,
                    "duration_seconds": round(duration_actual, 1),
                    "sample_rate": self.sample_rate,
                    "file_size_bytes": file_size,
                    "prompt_used": prompt,
                    "model": f"musicgen-{self.model_size}",
                },
            }

        except ImportError:
            return {"ok": False, "error": "audiocraft library not installed"}
        except Exception as e:
            logger.error(f"MusicGen generation failed: {e}")
            return {"ok": False, "error": f"Music generation failed: {str(e)}"}

    # ── Utility ──────────────────────────────────────────────────────────

    def list_generated_music(self) -> List[Dict[str, Any]]:
        """List all generated music files."""
        files = []
        for f in sorted(MUSIC_OUTPUT_DIR.glob("EDISON_music_*"), reverse=True):
            files.append({
                "filename": f.name,
                "path": str(f),
                "size_bytes": f.stat().st_size,
                "created": f.stat().st_mtime,
                "format": f.suffix.lstrip("."),
            })
        return files

    def get_available_models(self) -> Dict[str, Any]:
        """Return info about available MusicGen models."""
        return {
            "models": [
                {
                    "name": "musicgen-small",
                    "size": "~300M params",
                    "vram": "~4GB",
                    "quality": "Good for short clips",
                    "description": "Fast generation, lower quality",
                },
                {
                    "name": "musicgen-medium",
                    "size": "~1.5B params",
                    "vram": "~8GB",
                    "quality": "Balanced quality and speed",
                    "description": "Recommended default model",
                },
                {
                    "name": "musicgen-large",
                    "size": "~3.3B params",
                    "vram": "~16GB",
                    "quality": "Highest text-to-music quality",
                    "description": "Best quality, requires more VRAM",
                },
                {
                    "name": "musicgen-melody",
                    "size": "~1.5B params",
                    "vram": "~8GB",
                    "quality": "Melody-conditioned generation",
                    "description": "Can take an audio melody as input to guide generation",
                },
            ],
            "current_model": f"musicgen-{self.model_size}",
            "audiocraft_installed": self._check_audiocraft(),
        }

    @staticmethod
    def _check_audiocraft() -> bool:
        try:
            import audiocraft
            return True
        except ImportError:
            return False
