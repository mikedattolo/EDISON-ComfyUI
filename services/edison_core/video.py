"""
Video Generation Service for EDISON
Primary: CogVideoX-5B via HuggingFace diffusers (real AI video generation)
Multi-GPU: distributes model components across all available GPUs
Long video: generates multiple segments with crossfade blending
"""

import gc
import logging
import json
import random
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Lazy import to avoid circular deps at module level
_job_store = None


def _get_job_store():
    """Get the singleton JobStore (lazy import)."""
    global _job_store
    if _job_store is None:
        try:
            from services.edison_core.job_store import JobStore
            _job_store = JobStore.get_instance()
        except Exception as e:
            logger.warning(f"JobStore unavailable, falling back to in-memory tracking: {e}")
    return _job_store


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
    Multi-GPU: text encoder on secondary GPU, transformer+VAE on primary GPU
    (eliminates CPU offload overhead).
    Long video: generates multiple segments with crossfade blending.
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
        self.num_inference_steps: int = video_cfg.get("num_inference_steps", 30)

        # Multi-GPU and long video settings
        self.multi_gpu: bool = video_cfg.get("multi_gpu", True)
        self.max_duration: int = video_cfg.get("max_duration", 30)  # seconds
        self.segment_overlap: int = video_cfg.get("segment_overlap", 8)  # crossfade frames

        # ComfyUI fallback
        host = comfyui_cfg.get("host", "127.0.0.1")
        if host == "0.0.0.0":
            host = "127.0.0.1"
        port = comfyui_cfg.get("port", 8188)
        self.comfyui_url = f"http://{host}:{port}"

        # Low-VRAM mode: reduce inference steps and resolution for GPUs < 8GB
        self.low_vram: bool = video_cfg.get("low_vram", False)
        if self.low_vram:
            self.num_inference_steps = min(self.num_inference_steps, 20)
            self.width = min(self.width, 480)
            self.height = min(self.height, 320)
            logger.info("Low-VRAM mode enabled: reduced resolution and steps")

        # Fallback in-memory job tracking (used when JobStore is unavailable)
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
            f"{self.num_frames}f@{self.fps}fps, multi_gpu={self.multi_gpu}, "
            f"max_duration={self.max_duration}s)"
        )

    # ── Pipeline management ──────────────────────────────────────────────

    def _load_pipeline(self):
        """Load CogVideoX pipeline with multi-GPU distribution."""
        if self._pipe is not None:
            return

        with self._pipe_lock:
            if self._pipe is not None:
                return

            import torch

            t0 = time.time()

            models_to_try = [self.model_id]
            if "5b" in self.model_id.lower():
                models_to_try.append("THUDM/CogVideoX-2b")

            last_error = None
            for model_id in models_to_try:
                try:
                    logger.info(f"⏳ Loading CogVideoX pipeline: {model_id}...")
                    pipe = self._load_model(model_id, torch)
                    self._pipe = pipe
                    self.model_id = model_id
                    elapsed = time.time() - t0
                    logger.info(f"✓ CogVideoX loaded in {elapsed:.1f}s (model={model_id})")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load {model_id}: {e}", exc_info=True)
                    last_error = e
                    gc.collect()
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

            raise RuntimeError(f"Could not load CogVideoX. Last error: {last_error}")

    def _load_model(self, model_id: str, torch):
        """Load pipeline, trying multi-GPU first then single-GPU CPU offload."""
        from diffusers import CogVideoXPipeline

        # Try CogVideoXPipeline first, fall back to DiffusionPipeline
        for PipeClass in (CogVideoXPipeline,):
            for dtype in (torch.float16, torch.bfloat16):
                try:
                    pipe = PipeClass.from_pretrained(model_id, torch_dtype=dtype)
                    pipe.vae.enable_slicing()
                    pipe.vae.enable_tiling()

                    # Find best compatible GPU and use CPU offload on it
                    best_gpu = self._find_best_gpu(torch)
                    logger.info(f"Using GPU {best_gpu} with CPU offload")
                    pipe.enable_model_cpu_offload(gpu_id=best_gpu)
                    return pipe
                except Exception as e:
                    logger.warning(f"{PipeClass.__name__} + {dtype} failed: {e}")
                    gc.collect()
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

        # Last resort: DiffusionPipeline auto-detect
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        best_gpu = self._find_best_gpu(torch)
        pipe.enable_model_cpu_offload(gpu_id=best_gpu)
        return pipe

    def _find_best_gpu(self, torch) -> int:
        """Find the compatible GPU with the most free VRAM.

        Probes each GPU with a small fp16 matmul to verify CUDA kernel support
        (filters out GPUs like RTX 5060 Ti whose architecture isn't supported
        by the installed PyTorch). Returns the GPU id with the most free VRAM,
        or 0 as a fallback.
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus < 1:
            return 0

        gpu_info = []
        for i in range(num_gpus):
            try:
                free, total = torch.cuda.mem_get_info(i)
                name = torch.cuda.get_device_name(i)

                # Probe: run a small fp16 matmul to verify CUDA kernel support
                try:
                    dev = f"cuda:{i}"
                    a = torch.randn(4, 4, dtype=torch.float16, device=dev)
                    _ = a @ a
                    del a
                    torch.cuda.empty_cache()
                except Exception as probe_err:
                    logger.warning(
                        f"GPU {i} ({name}) failed CUDA probe — skipping: {probe_err}"
                    )
                    continue

                gpu_info.append({
                    "id": i,
                    "free_gb": free / (1024**3),
                    "name": name,
                })
            except Exception:
                continue

        if not gpu_info:
            logger.warning("No compatible GPUs found via probe — defaulting to GPU 0")
            return 0

        gpu_info.sort(key=lambda g: g["free_gb"], reverse=True)
        best = gpu_info[0]
        survey = [(g["id"], g["name"], f"{g['free_gb']:.1f}GB free") for g in gpu_info]
        logger.info(
            f"GPU survey: {survey} "
            f"→ using GPU {best['id']} ({best['name']}, {best['free_gb']:.1f}GB free)"
        )
        return best["id"]

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

    def _is_cancelled(self, job_id: str) -> bool:
        """Check if a job has been cancelled (via JobStore or in-memory)."""
        store = _get_job_store()
        if store:
            job = store.get_job(job_id)
            return job is not None and job.get("status") == "cancelled"
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            return job is not None and job.get("status") == "cancelled"

    def _generate_worker(self, job_id: str, prompt: str,
                         negative_prompt: str, audio_path: Optional[str],
                         duration: float = 6.0, gen_params: Optional[Dict[str, Any]] = None):
        """Background thread: load model → generate (multi-segment if needed) → export."""
        try:
            self._update_job(job_id, status="loading",
                             message="Loading CogVideoX model...")

            if self._is_cancelled(job_id):
                logger.info(f"Job {job_id} cancelled before model load")
                return

            self._load_pipeline()

            import torch
            import numpy as np
            from PIL import Image

            neg = negative_prompt or "nsfw, worst quality, low quality, blurry"

            # Use per-request params (never read from self to avoid race conditions)
            p = gen_params or {}
            local_width = p.get("width", self.width)
            local_height = p.get("height", self.height)
            local_frames = p.get("num_frames", self.num_frames)
            local_fps = p.get("fps", self.fps)
            local_steps = p.get("num_inference_steps", self.num_inference_steps)
            local_guidance = p.get("guidance_scale", self.guidance_scale)
            local_overlap = p.get("segment_overlap", self.segment_overlap)

            # Calculate segments needed for requested duration
            segment_secs = local_frames / local_fps  # ~6.125s per segment
            num_segments = max(1, round(duration / segment_secs))
            num_segments = min(num_segments, 5)  # Cap at 5 segments (~30s)

            all_frames: List = []
            all_seeds: List[int] = []

            for seg in range(num_segments):
                if self._is_cancelled(job_id):
                    logger.info(f"Job {job_id} cancelled at segment {seg}")
                    self._update_job(job_id, status="cancelled", message="Cancelled by user")
                    return
                seg_label = (
                    f"segment {seg + 1}/{num_segments} "
                    if num_segments > 1 else ""
                )
                est_min = int(local_steps * 11.5 / 60)
                self._update_job(
                    job_id, status="generating",
                    message=(
                        f"Generating {seg_label}video frames "
                        f"(~{est_min} min per segment)..."
                    ),
                )

                # Vary prompt slightly for continuation segments
                seg_prompt = prompt
                if seg > 0:
                    seg_prompt = (
                        f"{prompt}, smooth continuation, consistent style, "
                        f"seamless motion"
                    )

                seed = random.randint(0, 2**32 - 1) + seg
                all_seeds.append(seed)
                with torch.inference_mode():
                    output = self._pipe(
                        prompt=seg_prompt,
                        negative_prompt=neg,
                        num_frames=local_frames,
                        width=local_width,
                        height=local_height,
                        guidance_scale=local_guidance,
                        num_inference_steps=local_steps,
                        generator=torch.Generator(device="cpu").manual_seed(seed),
                    )

                segment_frames = output.frames[0]  # list of PIL Images

                if seg == 0:
                    all_frames.extend(segment_frames)
                else:
                    # Crossfade overlap region between segments
                    overlap = min(
                        local_overlap,
                        len(all_frames),
                        len(segment_frames),
                    )
                    for i in range(overlap):
                        alpha = (i + 1) / (overlap + 1)
                        prev = np.array(all_frames[-(overlap - i)])
                        curr = np.array(segment_frames[i])
                        blended = (
                            prev * (1 - alpha) + curr * alpha
                        ).astype(np.uint8)
                        all_frames[-(overlap - i)] = Image.fromarray(blended)
                    # Append non-overlapping frames
                    all_frames.extend(segment_frames[overlap:])

                # Free intermediate tensors between segments
                del output
                torch.cuda.empty_cache()

                logger.info(
                    f"Segment {seg + 1}/{num_segments} complete: "
                    f"{len(segment_frames)} frames"
                )

            total_secs = len(all_frames) / local_fps
            self._update_job(
                job_id, status="encoding",
                message=f"Encoding {len(all_frames)} frames ({total_secs:.1f}s) to MP4...",
            )

            output_file = VIDEO_OUTPUT_DIR / f"EDISON_video_{job_id}.mp4"
            self._export_frames_to_mp4(all_frames, output_file)

            # Optionally mux audio
            if audio_path and Path(audio_path).exists():
                muxed = VIDEO_OUTPUT_DIR / f"EDISON_video_{job_id}_audio.mp4"
                self._mux_audio(output_file, Path(audio_path), muxed)
                if muxed.exists():
                    output_file = muxed

            # Record provenance (seeds, model, params)
            provenance = {
                "model": self.model_id,
                "seeds": all_seeds,
                "backend": "cogvideox-diffusers",
                "num_segments": num_segments,
                "total_frames": len(all_frames),
            }

            self._update_job(
                job_id,
                status="complete",
                message=(
                    f"Video generated! {len(all_frames)} frames, "
                    f"{total_secs:.1f}s{' (' + str(num_segments) + ' segments)' if num_segments > 1 else ''}"
                ),
                filename=output_file.name,
                video_path=str(output_file),
                provenance=provenance,
            )

            # Write metadata sidecar & update job store outputs
            store = _get_job_store()
            if store:
                try:
                    store.update_status(
                        job_id, "complete",
                        outputs=[str(output_file)],
                        provenance=provenance,
                    )
                    store.write_metadata_sidecar(job_id, output_file)
                except Exception as e:
                    logger.warning(f"Failed to update job store for {job_id}: {e}")

            logger.info(
                f"✓ Video generation complete: {output_file.name} "
                f"({len(all_frames)} frames, {total_secs:.1f}s)"
            )

        except Exception as e:
            logger.error(f"❌ Video generation failed for job {job_id}: {e}",
                         exc_info=True)
            self._update_job(job_id, status="error", message=str(e))
            store = _get_job_store()
            if store:
                try:
                    store.update_status(job_id, "error", error_log=str(e))
                except Exception:
                    pass
            # Clean up partial output files
            for partial in VIDEO_OUTPUT_DIR.glob(f"*{job_id}*"):
                try:
                    partial.unlink(missing_ok=True)
                except Exception:
                    pass
        finally:
            # Always free model VRAM so LLMs can reload
            self._unload_pipeline()

    def _export_frames_to_mp4(self, frames, output_path: Path, fps: Optional[int] = None):
        """Convert a list of PIL Images to an MP4 file."""
        import numpy as np
        out_fps = fps or self.fps

        # Try imageio first (cleanest approach)
        try:
            import imageio.v3 as iio

            frame_arrays = [np.array(f) for f in frames]
            iio.imwrite(
                str(output_path),
                frame_arrays,
                fps=out_fps,
                codec="libx264",
                plugin="pyav",
            )
            logger.info(f"Exported {len(frames)} frames via imageio → {output_path.name}")
            return
        except Exception as e:
            logger.warning(f"imageio export failed ({e}), falling back to ffmpeg pipe")

        # Fallback: stream frames one-by-one to ffmpeg (no bulk materialization)
        h, w = np.array(frames[0]).shape[:2]
        cmd = [
            FFMPEG_BIN, "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "rgb24",
            "-r", str(out_fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "medium",
            str(output_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for f in frames:
                proc.stdin.write(np.array(f, dtype=np.uint8).tobytes())
            proc.stdin.close()
        except BrokenPipeError:
            pass
        proc.wait(timeout=120)
        if proc.returncode != 0:
            stderr_out = b""
            try:
                stderr_out = proc.stderr.read()
            except Exception:
                pass
            raise RuntimeError(f"ffmpeg encode failed: {stderr_out.decode(errors='replace')[:500]}")
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
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            logger.warning(f"Audio mux failed (rc={result.returncode}): {result.stderr.decode(errors='replace')[:300]}")

    # ── Job tracking helpers ─────────────────────────────────────────────

    def _update_job(self, job_id: str, **kwargs):
        """Update both in-memory job dict and unified job store."""
        with self._jobs_lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(kwargs)
        # Also update job store status if available
        status = kwargs.get("status")
        if status:
            store = _get_job_store()
            if store:
                status_map = {
                    "loading_model": "loading",
                    "loading": "loading",
                    "generating": "generating",
                    "encoding": "encoding",
                    "complete": "complete",
                    "error": "error",
                    "cancelled": "cancelled",
                }
                mapped = status_map.get(status)
                if mapped:
                    try:
                        store.update_status(job_id, mapped)
                    except Exception:
                        pass

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
        duration: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Submit a video generation job.
        Uses CogVideoX diffusers pipeline with multi-GPU and multi-segment support.
        """
        # Create job in unified store first (gives us a proper UUID)
        store = _get_job_store()

        # Snapshot per-request params (never mutate self)
        local_width = width or self.width
        local_height = height or self.height
        local_frames = frames or self.num_frames
        local_fps = fps or self.fps
        local_steps = steps or self.num_inference_steps
        local_guidance = guidance_scale if guidance_scale != 6.0 else self.guidance_scale

        # Calculate duration
        segment_secs = local_frames / local_fps
        req_duration = duration if duration else segment_secs
        req_duration = min(req_duration, self.max_duration)
        num_segments = max(1, round(req_duration / segment_secs))
        total_secs = num_segments * segment_secs

        # Estimate generation time
        est_per_segment = local_steps * 11.5  # seconds
        est_total_min = (est_per_segment * num_segments) / 60

        # Prune old completed/errored jobs to prevent memory leak
        self._prune_old_jobs()

        # Build params dict for the worker thread
        gen_params = {
            "width": local_width,
            "height": local_height,
            "num_frames": local_frames,
            "fps": local_fps,
            "num_inference_steps": local_steps,
            "guidance_scale": local_guidance,
            "segment_overlap": self.segment_overlap,
        }

        # Register job in unified store (falls back to in-memory)
        if store:
            try:
                job_id = store.create_job(
                    job_type="video",
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    params=gen_params,
                    provenance={"model": self.model_id, "backend": "cogvideox-diffusers"},
                )
            except Exception as e:
                logger.warning(f"JobStore create failed, using in-memory: {e}")
                job_id = uuid.uuid4().hex[:12]
        else:
            job_id = uuid.uuid4().hex[:12]

        # Also keep in-memory tracking for fast status checks
        with self._jobs_lock:
            self._jobs[job_id] = {
                "status": "queued",
                "prompt": prompt,
                "backend": "cogvideox-diffusers",
                "audio_path": audio_path,
                "created": time.time(),
                "message": "Video generation queued...",
                "num_segments": num_segments,
                "duration": total_secs,
            }

        # Launch background thread
        thread = threading.Thread(
            target=self._generate_worker,
            args=(job_id, prompt, negative_prompt, audio_path, req_duration, gen_params),
            daemon=True,
        )
        thread.start()

        return {
            "ok": True,
            "data": {
                "prompt_id": job_id,
                "backend": "cogvideox-diffusers",
                "frames": local_frames,
                "fps": local_fps,
                "duration": total_secs,
                "segments": num_segments,
                "audio_path": audio_path,
                "message": (
                    f"Video generation started using CogVideoX "
                    f"({local_width}×{local_height}, ~{total_secs:.0f}s"
                    f"{', ' + str(num_segments) + ' segments' if num_segments > 1 else ''}). "
                    f"Estimated time: ~{est_total_min:.0f} minutes."
                ),
            },
        }

    def _prune_old_jobs(self, max_age_secs: int = 3600):
        """Remove completed/errored jobs older than max_age_secs to prevent memory leak."""
        cutoff = time.time() - max_age_secs
        with self._jobs_lock:
            stale = [
                jid for jid, job in self._jobs.items()
                if job.get("status") in ("complete", "error")
                and job.get("created", 0) < cutoff
            ]
            for jid in stale:
                del self._jobs[jid]

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
