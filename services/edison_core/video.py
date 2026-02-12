"""
Video Generation Service for EDISON
Supports text-to-video and music-video generation via ComfyUI workflows
with AnimateDiff/CogVideoX/Wan2.1 backends.
"""

import logging
import json
import random
import shutil
import time
import uuid
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
    return "ffmpeg"  # fallback, let subprocess raise a clear error

FFMPEG_BIN = _find_ffmpeg()

# ── Repo root resolution ────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
UPLOAD_DIR = REPO_ROOT / "uploads" / "audio"
VIDEO_OUTPUT_DIR = REPO_ROOT / "outputs" / "videos"


class VideoGenerationService:
    """Manages video generation via ComfyUI pipelines."""

    def __init__(self, config: Dict[str, Any]):
        comfyui_cfg = config.get("edison", {}).get("comfyui", {})
        host = comfyui_cfg.get("host", "127.0.0.1")
        if host == "0.0.0.0":
            host = "127.0.0.1"
        port = comfyui_cfg.get("port", 8188)
        self.comfyui_url = f"http://{host}:{port}"

        # Video generation defaults
        video_cfg = config.get("edison", {}).get("video", {})
        self.default_frames = video_cfg.get("default_frames", 16)
        self.default_fps = video_cfg.get("default_fps", 8)
        self.default_width = video_cfg.get("default_width", 512)
        self.default_height = video_cfg.get("default_height", 512)
        self.default_steps = video_cfg.get("default_steps", 20)

        # Ensure directories exist
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(f"✓ Video generation service initialized (ComfyUI: {self.comfyui_url})")

    # ── Available backend detection ──────────────────────────────────────

    def detect_available_backend(self) -> str:
        """
        Query ComfyUI /object_info to discover which video nodes are installed.
        Returns the best available backend name.
        Priority: CogVideoX > Wan2.1 > AnimateDiff > fallback (frame-by-frame)
        """
        import requests

        try:
            resp = requests.get(f"{self.comfyui_url}/object_info", timeout=5)
            if resp.ok:
                nodes = resp.json()
                if "CogVideoXSampler" in nodes or "CogVideoXLoader" in nodes:
                    return "cogvideox"
                if "Wan21VideoSampler" in nodes or "WanVideoSampler" in nodes:
                    return "wan21"
                if "ADE_AnimateDiffLoaderWithContext" in nodes or "AnimateDiffLoaderV1" in nodes:
                    return "animatediff"
        except Exception as e:
            logger.warning(f"Could not query ComfyUI nodes: {e}")

        return "framebased"  # fallback: generate frames then stitch

    # ── Workflow builders ────────────────────────────────────────────────

    def create_animatediff_workflow(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        frames: int = 16,
        fps: int = 8,
        steps: int = 20,
        guidance_scale: float = 7.5,
    ) -> Dict[str, Any]:
        """Build an AnimateDiff ComfyUI workflow."""
        seed = random.randint(0, 2**32 - 1)
        neg = negative_prompt or "nsfw, nude, worst quality, low quality, blurry, static"
        prefix = f"EDISON_video_{uuid.uuid4().hex[:8]}"

        return {
            "1": {
                "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
                "class_type": "CheckpointLoaderSimple",
            },
            "2": {
                "inputs": {
                    "model_name": "v3_sd15_mm.ckpt",
                    "beta_schedule": "sqrt_linear (AnimateDiff)",
                    "model": ["1", 0],
                },
                "class_type": "ADE_AnimateDiffLoaderWithContext",
            },
            "3": {
                "inputs": {"text": prompt, "clip": ["1", 1]},
                "class_type": "CLIPTextEncode",
            },
            "4": {
                "inputs": {"text": neg, "clip": ["1", 1]},
                "class_type": "CLIPTextEncode",
            },
            "5": {
                "inputs": {"width": width, "height": height, "batch_size": frames},
                "class_type": "EmptyLatentImage",
            },
            "6": {
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": guidance_scale,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["2", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0],
                },
                "class_type": "KSampler",
            },
            "7": {
                "inputs": {"samples": ["6", 0], "vae": ["1", 2]},
                "class_type": "VAEDecode",
            },
            "8": {
                "inputs": {
                    "frame_rate": fps,
                    "loop_count": 0,
                    "filename_prefix": prefix,
                    "format": "video/h264-mp4",
                    "images": ["7", 0],
                },
                "class_type": "VHS_VideoCombine",
            },
        }

    def create_cogvideox_workflow(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 720,
        height: int = 480,
        frames: int = 49,
        fps: int = 8,
        steps: int = 50,
        guidance_scale: float = 6.0,
    ) -> Dict[str, Any]:
        """Build a CogVideoX ComfyUI workflow."""
        seed = random.randint(0, 2**32 - 1)
        neg = negative_prompt or "nsfw, worst quality, low quality"
        prefix = f"EDISON_video_{uuid.uuid4().hex[:8]}"

        return {
            "1": {
                "inputs": {"model_name": "CogVideoX-5b"},
                "class_type": "CogVideoXLoader",
            },
            "2": {
                "inputs": {"prompt": prompt, "model": ["1", 0]},
                "class_type": "CogVideoXTextEncode",
            },
            "3": {
                "inputs": {"prompt": neg, "model": ["1", 0]},
                "class_type": "CogVideoXTextEncode",
            },
            "4": {
                "inputs": {
                    "width": width,
                    "height": height,
                    "num_frames": frames,
                    "seed": seed,
                    "steps": steps,
                    "cfg": guidance_scale,
                    "scheduler": "DPM++",
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                },
                "class_type": "CogVideoXSampler",
            },
            "5": {
                "inputs": {"vae": ["1", 1], "samples": ["4", 0]},
                "class_type": "CogVideoXDecode",
            },
            "6": {
                "inputs": {
                    "frame_rate": fps,
                    "loop_count": 0,
                    "filename_prefix": prefix,
                    "format": "video/h264-mp4",
                    "images": ["5", 0],
                },
                "class_type": "VHS_VideoCombine",
            },
        }

    def create_framebased_workflow(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        frames: int = 16,
        steps: int = 20,
        guidance_scale: float = 7.5,
    ) -> Dict[str, Any]:
        """
        Fallback: generate a batch of images as frames using standard SDXL,
        then stitch them into a video server-side.
        """
        seed = random.randint(0, 2**32 - 1)
        neg = negative_prompt or "nsfw, worst quality, low quality, blurry"
        prefix = f"EDISON_frames_{uuid.uuid4().hex[:8]}"

        return {
            "3": {
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": guidance_scale,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                },
                "class_type": "KSampler",
            },
            "4": {
                "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
                "class_type": "CheckpointLoaderSimple",
            },
            "5": {
                "inputs": {"width": width, "height": height, "batch_size": frames},
                "class_type": "EmptyLatentImage",
            },
            "6": {
                "inputs": {"text": prompt, "clip": ["4", 1]},
                "class_type": "CLIPTextEncode",
            },
            "7": {
                "inputs": {"text": neg, "clip": ["4", 1]},
                "class_type": "CLIPTextEncode",
            },
            "8": {
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
                "class_type": "VAEDecode",
            },
            "9": {
                "inputs": {
                    "filename_prefix": prefix,
                    "images": ["8", 0],
                },
                "class_type": "SaveImage",
            },
        }

    # ── Submit generation ────────────────────────────────────────────────

    def submit_video_generation(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        frames: Optional[int] = None,
        fps: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: float = 7.5,
        audio_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a video generation job to ComfyUI.
        Auto-detects the best available backend.
        """
        import requests

        width = width or self.default_width
        height = height or self.default_height
        frames = frames or self.default_frames
        fps = fps or self.default_fps
        steps = steps or self.default_steps

        backend = self.detect_available_backend()
        logger.info(f"Video generation backend: {backend}")

        if backend == "cogvideox":
            workflow = self.create_cogvideox_workflow(
                prompt, negative_prompt, width, height, frames, fps, steps, guidance_scale
            )
        elif backend == "animatediff":
            workflow = self.create_animatediff_workflow(
                prompt, negative_prompt, width, height, frames, fps, steps, guidance_scale
            )
        else:
            workflow = self.create_framebased_workflow(
                prompt, negative_prompt, width, height, frames, steps, guidance_scale
            )

        try:
            resp = requests.post(
                f"{self.comfyui_url}/prompt",
                json={"prompt": workflow},
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()
            prompt_id = result.get("prompt_id")

            if not prompt_id:
                return {"ok": False, "error": "No prompt_id from ComfyUI"}

            # Track audio path for music-video post-processing
            meta = {
                "prompt_id": prompt_id,
                "backend": backend,
                "audio_path": audio_path,
                "fps": fps,
                "frames": frames,
            }
            meta_path = VIDEO_OUTPUT_DIR / f"{prompt_id}_meta.json"
            meta_path.write_text(json.dumps(meta))

            return {
                "ok": True,
                "data": {
                    "prompt_id": prompt_id,
                    "backend": backend,
                    "frames": frames,
                    "fps": fps,
                    "audio_path": audio_path,
                    "message": f"Video generation started using {backend} backend.",
                },
            }

        except Exception as e:
            logger.error(f"Video generation submission failed: {e}")
            return {"ok": False, "error": str(e)}

    # ── Status & post-processing ─────────────────────────────────────────

    def check_video_status(self, prompt_id: str) -> Dict[str, Any]:
        """Check video generation status from ComfyUI history."""
        import requests

        try:
            resp = requests.get(f"{self.comfyui_url}/history/{prompt_id}", timeout=5)
            if not resp.ok:
                return {"ok": True, "data": {"status": "generating", "prompt_id": prompt_id}}

            history = resp.json()
            if prompt_id not in history:
                return {"ok": True, "data": {"status": "queued", "prompt_id": prompt_id}}

            entry = history[prompt_id]
            status = entry.get("status", {})

            if status.get("status_str") == "error":
                return {
                    "ok": False,
                    "error": status.get("messages", [["error"]])[0][-1] if status.get("messages") else "Generation failed",
                }

            # Check for completed outputs
            outputs = entry.get("outputs", {})
            video_files = []
            image_files = []

            for node_id, node_out in outputs.items():
                # Video files (from VHS_VideoCombine)
                if "gifs" in node_out:
                    for gif in node_out["gifs"]:
                        video_files.append({
                            "filename": gif.get("filename", ""),
                            "subfolder": gif.get("subfolder", ""),
                            "type": gif.get("type", "output"),
                        })
                # Image frames (from SaveImage)
                if "images" in node_out:
                    for img in node_out["images"]:
                        image_files.append({
                            "filename": img.get("filename", ""),
                            "subfolder": img.get("subfolder", ""),
                            "type": img.get("type", "output"),
                        })

            if video_files:
                # Check if we need to mux audio
                meta_path = VIDEO_OUTPUT_DIR / f"{prompt_id}_meta.json"
                audio_path = None
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    audio_path = meta.get("audio_path")

                result = {
                    "status": "complete",
                    "prompt_id": prompt_id,
                    "videos": video_files,
                }

                if audio_path:
                    result["audio_path"] = audio_path
                    result["message"] = "Video generated. Audio can be muxed with /mux-video-audio endpoint."

                return {"ok": True, "data": result}

            if image_files:
                return {
                    "ok": True,
                    "data": {
                        "status": "complete_frames",
                        "prompt_id": prompt_id,
                        "frames": image_files,
                        "message": "Frames generated. Use /stitch-frames endpoint to create video.",
                    },
                }

            return {"ok": True, "data": {"status": "generating", "prompt_id": prompt_id}}

        except Exception as e:
            logger.error(f"Video status check failed: {e}")
            return {"ok": False, "error": str(e)}

    def stitch_frames_to_video(
        self, prompt_id: str, fps: int = 8, audio_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Stitch generated image frames into an MP4 video using ffmpeg.
        Optionally mux with audio for music videos.
        """
        try:
            import subprocess

            # Find frames from ComfyUI output
            comfyui_output = REPO_ROOT / "ComfyUI" / "output"
            frame_pattern = None

            # Look for frame files with the prompt_id prefix
            meta_path = VIDEO_OUTPUT_DIR / f"{prompt_id}_meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                fps = meta.get("fps", fps)
                audio_path = audio_path or meta.get("audio_path")

            # Search for matching frame files
            frames = sorted(comfyui_output.glob(f"EDISON_frames_*"))
            if not frames:
                frames = sorted(comfyui_output.glob("EDISON_*.png"))

            if not frames:
                return {"ok": False, "error": "No frames found to stitch"}

            output_file = VIDEO_OUTPUT_DIR / f"EDISON_video_{uuid.uuid4().hex[:8]}.mp4"

            # Use ffmpeg to create video
            cmd = [
                FFMPEG_BIN, "-y",
                "-framerate", str(fps),
                "-pattern_type", "glob",
                "-i", str(comfyui_output / "EDISON_frames_*.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "23",
            ]

            if audio_path and Path(audio_path).exists():
                cmd.extend(["-i", audio_path, "-c:a", "aac", "-shortest"])

            cmd.append(str(output_file))

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                logger.error(f"ffmpeg error: {result.stderr}")
                return {"ok": False, "error": f"ffmpeg failed: {result.stderr[:500]}"}

            return {
                "ok": True,
                "data": {
                    "video_path": str(output_file),
                    "filename": output_file.name,
                    "fps": fps,
                    "has_audio": bool(audio_path),
                },
            }

        except Exception as e:
            logger.error(f"Frame stitching failed: {e}")
            return {"ok": False, "error": str(e)}

    def mux_audio_to_video(self, video_path: str, audio_path: str) -> Dict[str, Any]:
        """Mux an audio file onto a generated video using ffmpeg."""
        try:
            import subprocess

            video_p = Path(video_path)
            audio_p = Path(audio_path)
            if not video_p.exists():
                return {"ok": False, "error": f"Video not found: {video_path}"}
            if not audio_p.exists():
                return {"ok": False, "error": f"Audio not found: {audio_path}"}

            output_file = VIDEO_OUTPUT_DIR / f"EDISON_musicvideo_{uuid.uuid4().hex[:8]}.mp4"

            cmd = [
                FFMPEG_BIN, "-y",
                "-i", str(video_p),
                "-i", str(audio_p),
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                str(output_file),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                return {"ok": False, "error": f"ffmpeg mux failed: {result.stderr[:500]}"}

            return {
                "ok": True,
                "data": {
                    "video_path": str(output_file),
                    "filename": output_file.name,
                    "has_audio": True,
                },
            }

        except Exception as e:
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
