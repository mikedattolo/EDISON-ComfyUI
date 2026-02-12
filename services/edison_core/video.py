"""
Video Generation Service for EDISON
Primary: CogVideoX-5B via HuggingFace diffusers (real AI video generation)
Fallback: ComfyUI-based AnimateDiff / framebased stitching
"""

import gc
import logging
import json
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

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

# ── Repo root resolution ────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
UPLOAD_DIR = REPO_ROOT / "uploads" / "audio"
VIDEO_OUTPUT_DIR = REPO_ROOT / "outputs" / "videos"


class VideoGenerationService:
    """
    AI video generation service.

    Primary backend: CogVideoX-5B via HuggingFace diffusers — produces
    temporally-coherent 6-second clips (49 frames @ 8 fps, 720×480).
    Falls back to CogVideoX-2B if VRAM is tight.
    Last-resort fallback: ComfyUI framebased pipeline.
    """

    def __init__(self, config: Dict[str, Any]):
        video_cfg = config.get("edison", {}).get("video", {})
        comfyui_cfg = config.get("edison", {}).get("comfyui", {})

        # CogVideoX settings
        self.model_id: str = video_cfg.get("cogvideox_model", "THUDM/CogVideoX-5b")
        self.num_frames: int = video_cfg.get("num_frames", 49)
        self.width: int = video_cfg.get("width", 720)
        self.height: int = video_cfg.get("height", 480)
        self.fps: int = video_cfg.get("fps", 8)
        self.guidance_scale: float = video_cfg.get("guidance_scale", 6.0)
        self.num_inference_steps: int = video_cfg.get("num_inference_steps", 50)

        # ComfyUI fallback
        host = comfyui_cfg.get("host", "127.0.0.1")
        if host == "0.0.0.0":
            host = "127.0.0.1"
        port = comfyui_cfg.get("port", 8188)
        self.comfyui_url = f"http://{host}:{port}"

        # Internal job tracking (replaces ComfyUI polling for diffusers jobs)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._jobs_lock = threading.Lock()

        # Pipeline (lazy-loaded)
        self._pipe = None
        self._pipe_lock = threading.Lock()

        # Ensure directories exist
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"✓ Video generation service initialized "
            f"(model={self.model_id}, {self.width}×{self.height}, "
            f"{self.num_frames}f@{self.fps}fps)"
        )

    # ── Pipeline management ──────────────────────────────────────────────

    def _load_pipeline(self):
        """Lazy-load the CogVideoX diffusers pipeline."""
        if self._pipe is not None:
            return

        with self._pipe_lock:
            if self._pipe is not None:
                return

            import torch

            t0 = time.time()

            # Build list of models to try (primary + fallback)
            models_to_try = [self.model_id]
            if "5b" in self.model_id.lower():
                models_to_try.append("THUDM/CogVideoX-2b")

            last_error = None
            for model_id in models_to_try:
                for strategy in ("specific_fp16", "specific_bf16", "auto_fp16"):
                    try:
                        logger.info(
                            f"⏳ Loading CogVideoX pipeline: {model_id} "
                            f"(strategy={strategy})..."
                        )
                        pipe = self._try_load(model_id, strategy, torch)
                        pipe.enable_model_cpu_offload()
                        pipe.vae.enable_slicing()
                        pipe.vae.enable_tiling()

                        self._pipe = pipe
                        self.model_id = model_id
                        elapsed = time.time() - t0
                        logger.info(
                            f"✓ CogVideoX pipeline loaded in {elapsed:.1f}s "
                            f"(model={model_id}, strategy={strategy})"
                        )
                        return
                    except Exception as e:
                        logger.warning(
                            f"Strategy {strategy} failed for {model_id}: {e}"
                        )
                        last_error = e
                        # Clean up partial state
                        gc.collect()
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

            raise RuntimeError(
                f"Could not load any CogVideoX model. Last error: {last_error}"
            )

    @staticmethod
    def _try_load(model_id: str, strategy: str, torch):
        """Attempt to load a pipeline with a specific strategy."""
        if strategy == "specific_fp16":
            from diffusers import CogVideoXPipeline
            return CogVideoXPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16,
            )
        elif strategy == "specific_bf16":
            from diffusers import CogVideoXPipeline
            return CogVideoXPipeline.from_pretrained(
                model_id, torch_dtype=torch.bfloat16,
            )
        else:  # "auto_fp16" — use DiffusionPipeline auto-detect
            from diffusers import DiffusionPipeline
            return DiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16,
            )

    def _unload_pipeline(self):
        """Free the pipeline and reclaim VRAM."""
        with self._pipe_lock:
            if self._pipe is not None:
                logger.info("⏳ Unloading CogVideoX pipeline to free VRAM...")
                del self._pipe
                self._pipe = None
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception:
                    pass
                gc.collect()
                logger.info("✓ CogVideoX pipeline unloaded")

    # ── Core generation (runs in background thread) ──────────────────────

    def _generate_worker(self, job_id: str, prompt: str,
                         negative_prompt: str, audio_path: Optional[str]):
        """Background thread: load model → generate → export mp4 → unload."""
        try:
            self._update_job(job_id, status="loading_model",
                             message="Loading CogVideoX model...")

            self._load_pipeline()

            self._update_job(job_id, status="generating",
                             message="Generating video frames (this takes 2-5 minutes)...")

            import torch

            neg = negative_prompt or "nsfw, worst quality, low quality, blurry"

            with torch.no_grad():
                output = self._pipe(
                    prompt=prompt,
                    negative_prompt=neg,
                    num_frames=self.num_frames,
                    width=self.width,
                    height=self.height,
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=self.num_inference_steps,
                    generator=torch.Generator(device="cpu").manual_seed(
                        int(time.time()) % (2**32)
                    ),
                )

            frames = output.frames[0]  # list of PIL Images

            self._update_job(job_id, status="encoding",
                             message="Encoding video to MP4...")

            # Export frames → mp4 via imageio (or PIL fallback)
            output_file = VIDEO_OUTPUT_DIR / f"EDISON_video_{job_id}.mp4"
            self._export_frames_to_mp4(frames, output_file)

            # Optionally mux audio
            if audio_path and Path(audio_path).exists():
                muxed = VIDEO_OUTPUT_DIR / f"EDISON_video_{job_id}_audio.mp4"
                self._mux_audio(output_file, Path(audio_path), muxed)
                if muxed.exists():
                    output_file = muxed

            self._update_job(
                job_id,
                status="complete",
                message="Video generated successfully!",
                filename=output_file.name,
                video_path=str(output_file),
            )
            logger.info(f"✓ Video generation complete: {output_file.name}")

        except Exception as e:
            logger.error(f"❌ Video generation failed for job {job_id}: {e}",
                         exc_info=True)
            self._update_job(job_id, status="error", message=str(e))
        finally:
            # Always free model VRAM so LLMs can reload
            self._unload_pipeline()

    def _export_frames_to_mp4(self, frames, output_path: Path):
        """Convert a list of PIL Images to an MP4 file."""
        import numpy as np

        # Try imageio first (cleanest approach)
        try:
            import imageio.v3 as iio

            frame_arrays = [np.array(f) for f in frames]
            iio.imwrite(
                str(output_path),
                frame_arrays,
                fps=self.fps,
                codec="libx264",
                plugin="pyav",
            )
            logger.info(f"Exported {len(frames)} frames via imageio → {output_path.name}")
            return
        except Exception as e:
            logger.warning(f"imageio export failed ({e}), falling back to ffmpeg pipe")

        # Fallback: pipe raw frames to ffmpeg
        frame_arrays = [np.array(f) for f in frames]
        h, w = frame_arrays[0].shape[:2]
        cmd = [
            FFMPEG_BIN, "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "rgb24",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "medium",
            str(output_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for arr in frame_arrays:
            proc.stdin.write(arr.astype(np.uint8).tobytes())
        proc.stdin.close()
        proc.wait(timeout=120)
        if proc.returncode != 0:
            err = proc.stderr.read().decode()[:500]
            raise RuntimeError(f"ffmpeg encode failed: {err}")
        logger.info(f"Exported {len(frames)} frames via ffmpeg pipe → {output_path.name}")

    @staticmethod
    def _mux_audio(video: Path, audio: Path, output: Path):
        """Mux audio onto a video file."""
        cmd = [
            FFMPEG_BIN, "-y",
            "-i", str(video),
            "-i", str(audio),
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            str(output),
        ]
        subprocess.run(cmd, capture_output=True, timeout=120)

    # ── Job tracking helpers ─────────────────────────────────────────────

    def _update_job(self, job_id: str, **kwargs):
        with self._jobs_lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(kwargs)

    # ── Public API ───────────────────────────────────────────────────────

    def submit_video_generation(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        frames: Optional[int] = None,
        fps: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: float = 6.0,
        audio_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a video generation job.
        Uses CogVideoX diffusers pipeline (primary) with internal status tracking.
        """
        job_id = uuid.uuid4().hex[:12]

        # Allow per-request overrides
        if width:
            self.width = width
        if height:
            self.height = height
        if frames:
            self.num_frames = frames
        if fps:
            self.fps = fps
        if steps:
            self.num_inference_steps = steps
        if guidance_scale != 6.0:
            self.guidance_scale = guidance_scale

        # Register the job
        with self._jobs_lock:
            self._jobs[job_id] = {
                "status": "queued",
                "prompt": prompt,
                "backend": "cogvideox-diffusers",
                "audio_path": audio_path,
                "created": time.time(),
                "message": "Video generation queued...",
            }

        # Launch background thread
        thread = threading.Thread(
            target=self._generate_worker,
            args=(job_id, prompt, negative_prompt, audio_path),
            daemon=True,
        )
        thread.start()

        return {
            "ok": True,
            "data": {
                "prompt_id": job_id,
                "backend": "cogvideox-diffusers",
                "frames": self.num_frames,
                "fps": self.fps,
                "audio_path": audio_path,
                "message": (
                    f"Video generation started using CogVideoX "
                    f"({self.width}×{self.height}, {self.num_frames} frames). "
                    f"This takes 2-5 minutes."
                ),
            },
        }

    def check_video_status(self, prompt_id: str) -> Dict[str, Any]:
        """Check video generation status."""
        with self._jobs_lock:
            job = self._jobs.get(prompt_id)

        if job is None:
            # Unknown job — possibly a stale id
            return {"ok": True, "data": {"status": "unknown", "prompt_id": prompt_id}}

        status = job.get("status", "unknown")

        if status == "complete":
            return {
                "ok": True,
                "data": {
                    "status": "complete",
                    "prompt_id": prompt_id,
                    "videos": [{
                        "filename": job.get("filename", ""),
                        "video_path": job.get("video_path", ""),
                    }],
                    "message": job.get("message", ""),
                },
            }

        if status == "error":
            return {
                "ok": False,
                "error": job.get("message", "Video generation failed"),
            }

        # Still in progress (queued / loading_model / generating / encoding)
        return {
            "ok": True,
            "data": {
                "status": "generating",
                "prompt_id": prompt_id,
                "message": job.get("message", "Generating..."),
            },
        }

    def mux_audio_to_video(self, video_path: str, audio_path: str) -> Dict[str, Any]:
        """Mux an audio file onto a generated video using ffmpeg."""
        video_p = Path(video_path)
        audio_p = Path(audio_path)
        if not video_p.exists():
            return {"ok": False, "error": f"Video not found: {video_path}"}
        if not audio_p.exists():
            return {"ok": False, "error": f"Audio not found: {audio_path}"}

        output_file = VIDEO_OUTPUT_DIR / f"EDISON_musicvideo_{uuid.uuid4().hex[:8]}.mp4"
        self._mux_audio(video_p, audio_p, output_file)

        if not output_file.exists():
            return {"ok": False, "error": "ffmpeg mux failed"}

        return {
            "ok": True,
            "data": {
                "video_path": str(output_file),
                "filename": output_file.name,
                "has_audio": True,
            },
        }

    # Kept for backwards compat — no longer used by primary flow
    def stitch_frames_to_video(
        self, prompt_id: str, fps: int = 8, audio_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Legacy: stitch ComfyUI-generated frames into MP4."""
        try:
            comfyui_output = REPO_ROOT / "ComfyUI" / "output"
            meta_path = VIDEO_OUTPUT_DIR / f"{prompt_id}_meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                fps = meta.get("fps", fps)
                audio_path = audio_path or meta.get("audio_path")

            frames = sorted(comfyui_output.glob("EDISON_frames_*.png"))
            if not frames:
                return {"ok": False, "error": "No frames found to stitch"}

            output_file = VIDEO_OUTPUT_DIR / f"EDISON_video_{uuid.uuid4().hex[:8]}.mp4"
            target_fps = max(fps * 3, 24)
            cmd = [
                FFMPEG_BIN, "-y",
                "-framerate", str(fps),
                "-pattern_type", "glob",
                "-i", str(comfyui_output / "EDISON_frames_*.png"),
                "-vf", f"minterpolate=fps={target_fps}:mi_mode=blend",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "20", "-preset", "medium",
            ]
            if audio_path and Path(audio_path).exists():
                cmd.extend(["-i", audio_path, "-c:a", "aac", "-shortest"])
            cmd.append(str(output_file))

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                # Simple fallback
                cmd2 = [
                    FFMPEG_BIN, "-y",
                    "-framerate", str(fps), "-pattern_type", "glob",
                    "-i", str(comfyui_output / "EDISON_frames_*.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                ]
                if audio_path and Path(audio_path).exists():
                    cmd2.extend(["-i", audio_path, "-c:a", "aac", "-shortest"])
                cmd2.append(str(output_file))
                result = subprocess.run(cmd2, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                return {"ok": False, "error": f"ffmpeg failed: {result.stderr[:500]}"}

            video_info = {
                "video_path": str(output_file),
                "filename": output_file.name,
                "fps": fps,
                "has_audio": bool(audio_path),
            }
            return {"ok": True, "data": video_info}
        except Exception as e:
            logger.error(f"Frame stitching failed: {e}")
            return {"ok": False, "error": str(e)}

    @staticmethod
    def save_uploaded_audio(file_bytes: bytes, original_filename: str) -> str:
        """Save an uploaded audio file and return the path."""
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        ext = Path(original_filename).suffix or ".mp3"
        safe_name = f"audio_{uuid.uuid4().hex[:8]}{ext}"
        dest = UPLOAD_DIR / safe_name
        dest.write_bytes(file_bytes)
        logger.info(f"Saved uploaded audio: {dest}")
        return str(dest)
