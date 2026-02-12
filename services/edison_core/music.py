"""
Music Generation Service for EDISON
Generates music from text prompts, lyrics, and style descriptions
using Meta's MusicGen via Hugging Face transformers.
"""

import logging
import json
import uuid
import time
import shutil
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

def _find_ffmpeg() -> str:
    """Find ffmpeg binary, checking common paths if not on PATH."""
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    for p in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/snap/bin/ffmpeg"]:
        if Path(p).exists():
            return p
    return "ffmpeg"

FFMPEG_BIN = _find_ffmpeg()

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
MUSIC_OUTPUT_DIR = REPO_ROOT / "outputs" / "music"
MUSIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class MusicGenerationService:
    """
    Music generation using Meta's MusicGen via Hugging Face transformers.
    Supports text-to-music with genre, mood, instruments, tempo, and lyrics.
    """

    def __init__(self, config: Dict[str, Any]):
        music_cfg = config.get("edison", {}).get("music", {})
        self.model_size = music_cfg.get("model_size", "small")  # small, medium, large
        self.default_duration = music_cfg.get("default_duration", 15)  # seconds
        self.max_duration = music_cfg.get("max_duration", 60)
        self.sample_rate = 32000
        self._model = None
        self._processor = None
        self._model_loaded = False
        self._loading = False
        logger.info(f"✓ Music generation service initialized (model: musicgen-{self.model_size})")

    # ── Model management ─────────────────────────────────────────────────

    def _load_model(self):
        """Lazy-load MusicGen model on first use via transformers."""
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
            import torch
            from transformers import MusicgenForConditionalGeneration, AutoProcessor

            model_name = f"facebook/musicgen-{self.model_size}"
            logger.info(f"Loading MusicGen model via transformers: {model_name}...")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_dtype = torch.float16 if device == "cuda" else torch.float32

            logger.info(f"Downloading/loading processor for {model_name}...")
            self._processor = AutoProcessor.from_pretrained(model_name)
            
            logger.info(f"Downloading/loading model weights for {model_name} (this may take several minutes on first use)...")
            # Try 'dtype' first (newer transformers), fall back to 'torch_dtype'
            try:
                self._model = MusicgenForConditionalGeneration.from_pretrained(
                    model_name, torch_dtype=compute_dtype
                ).to(device)
            except TypeError:
                self._model = MusicgenForConditionalGeneration.from_pretrained(
                    model_name
                ).to(device)

            self.sample_rate = self._model.config.audio_encoder.sampling_rate
            self._model_loaded = True
            self._loading = False
            logger.info(f"✓ MusicGen model loaded: {model_name} on {device} (sr={self.sample_rate})")
            return True
        except ImportError as e:
            logger.error(
                f"❌ transformers MusicGen not available: {e}. "
                "Install with: pip install transformers torch torchaudio"
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
            del self._processor
            self._model = None
            self._processor = None
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

        # Generate via transformers MusicGen
        result = self._generate_with_transformers(prompt, duration)
        if result.get("ok"):
            return result

        # If generation fails, return instructions for setup
        return {
            "ok": False,
            "error": (
                "MusicGen model not available. To enable music generation:\n"
                "1. Install: pip install transformers torch torchaudio scipy\n"
                "2. Ensure you have a CUDA GPU with sufficient VRAM (4GB+ for small, 8GB+ for medium)\n"
                "3. The model will be auto-downloaded on first use (~3.3GB for medium)"
            ),
            "prompt_used": prompt,
            "duration": duration,
        }

    def _generate_with_transformers(
        self, prompt: str, duration: int
    ) -> Dict[str, Any]:
        """Generate music using MusicGen via Hugging Face transformers."""
        try:
            if not self._load_model():
                return {"ok": False, "error": "MusicGen model not loaded"}

            import torch
            import scipy.io.wavfile

            output_id = uuid.uuid4().hex[:12]

            # Calculate max_new_tokens from duration
            # MusicGen generates at ~50 tokens/sec for the audio codec
            tokens_per_second = 50
            max_new_tokens = int(duration * tokens_per_second)

            # Tokenize the prompt
            inputs = self._processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            ).to(self._model.device)

            # Generate audio
            with torch.no_grad():
                audio_values = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                )

            # audio_values shape: (batch, 1, samples)
            audio_data = audio_values[0, 0].cpu().float().numpy()

            # Normalize and convert to int16 for proper WAV playback
            import numpy as np
            audio_max = np.abs(audio_data).max()
            if audio_max > 0:
                audio_data = audio_data / audio_max
            audio_int16 = (audio_data * 32767).astype(np.int16)

            # Save as WAV
            output_path = MUSIC_OUTPUT_DIR / f"EDISON_music_{output_id}.wav"
            scipy.io.wavfile.write(
                str(output_path), self.sample_rate, audio_int16
            )

            # Also save as MP3 if ffmpeg available
            mp3_path = None
            try:
                import subprocess
                mp3_file = MUSIC_OUTPUT_DIR / f"EDISON_music_{output_id}.mp3"
                result = subprocess.run(
                    [FFMPEG_BIN, "-y", "-i", str(output_path), "-b:a", "192k", str(mp3_file)],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    mp3_path = str(mp3_file)
            except Exception:
                pass

            file_size = output_path.stat().st_size
            duration_actual = len(audio_data) / self.sample_rate

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

        except ImportError as e:
            return {"ok": False, "error": f"transformers MusicGen not available: {e}"}
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
            ],
            "current_model": f"musicgen-{self.model_size}",
            "musicgen_available": self._check_musicgen(),
        }

    @staticmethod
    def _check_musicgen() -> bool:
        try:
            from transformers import MusicgenForConditionalGeneration
            return True
        except ImportError:
            return False
