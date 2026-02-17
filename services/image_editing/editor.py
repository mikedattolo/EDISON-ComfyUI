"""
Edison Image Editing Pipeline.

Supports:
- Basic transforms (crop, resize, rotate, flip)
- Color adjustments (brightness, contrast, saturation, recolor)
- ComfyUI inpainting/img2img pipeline delegation
- Version history tracking
"""

import io
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
EDITS_DIR = OUTPUTS_DIR / "edits"


# ── Data Classes ─────────────────────────────────────────────────────────

@dataclass
class EditRecord:
    """Record of a single image edit operation."""
    edit_id: str
    source_image_id: Optional[str]  # file_store file_id or output path
    source_path: str
    edit_type: str  # crop, resize, rotate, brightness, inpaint, img2img, etc.
    edit_prompt: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    output_path: str = ""
    timestamp: float = 0
    model_used: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ── PIL Helpers ──────────────────────────────────────────────────────────

def _ensure_pil():
    """Import PIL and return Image module, or raise RuntimeError."""
    try:
        from PIL import Image, ImageEnhance, ImageFilter
        return Image, ImageEnhance, ImageFilter
    except ImportError:
        raise RuntimeError("Pillow is not installed. Install with: pip install Pillow")


def _load_image(path: str):
    """Load an image from path, returning a PIL Image."""
    Image, _, _ = _ensure_pil()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(p).convert("RGBA")


def _save_image(img, output_path: str, fmt: str = "PNG") -> str:
    """Save a PIL Image and return the path."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if fmt.upper() == "JPEG" and img.mode == "RGBA":
        img = img.convert("RGB")
    img.save(output_path, format=fmt)
    return output_path


# ── Image Editor ─────────────────────────────────────────────────────────

class ImageEditor:
    """
    Image editing pipeline with version history.

    All operations return an EditRecord with the output path.
    ComfyUI-based edits (inpaint, img2img) are delegated to the ComfyUI
    service when available.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self._edits_dir = Path(output_dir) if output_dir else EDITS_DIR
        self._edits_dir.mkdir(parents=True, exist_ok=True)
        self._history: List[EditRecord] = []

    def _output_path(self, edit_id: str, ext: str = ".png") -> str:
        return str(self._edits_dir / f"{edit_id}{ext}")

    def _record(self, source_path: str, edit_type: str,
                output_path: str, source_id: Optional[str] = None,
                prompt: Optional[str] = None,
                params: Optional[dict] = None,
                model: Optional[str] = None) -> EditRecord:
        rec = EditRecord(
            edit_id=str(uuid.uuid4()),
            source_image_id=source_id,
            source_path=source_path,
            edit_type=edit_type,
            edit_prompt=prompt,
            parameters=params or {},
            output_path=output_path,
            timestamp=time.time(),
            model_used=model,
        )
        self._history.append(rec)
        # Save provenance sidecar
        sidecar = Path(output_path + ".meta.json")
        try:
            sidecar.write_text(json.dumps(rec.to_dict(), indent=2))
        except Exception as e:
            logger.debug(f"Failed to write edit sidecar: {e}")
        return rec

    # ── Basic Transforms ─────────────────────────────────────────────

    def crop(self, source_path: str, box: Tuple[int, int, int, int],
             source_id: Optional[str] = None) -> EditRecord:
        """Crop image. box = (left, top, right, bottom)."""
        img = _load_image(source_path)
        cropped = img.crop(box)
        edit_id = str(uuid.uuid4())
        out = self._output_path(edit_id)
        _save_image(cropped, out)
        return self._record(source_path, "crop", out, source_id,
                            params={"box": list(box)})

    def resize(self, source_path: str, width: int, height: int,
               source_id: Optional[str] = None) -> EditRecord:
        """Resize image to exact dimensions."""
        Image, _, _ = _ensure_pil()
        img = _load_image(source_path)
        resized = img.resize((width, height), Image.LANCZOS)
        edit_id = str(uuid.uuid4())
        out = self._output_path(edit_id)
        _save_image(resized, out)
        return self._record(source_path, "resize", out, source_id,
                            params={"width": width, "height": height})

    def rotate(self, source_path: str, angle: float,
               source_id: Optional[str] = None) -> EditRecord:
        """Rotate image by angle degrees (counterclockwise)."""
        img = _load_image(source_path)
        rotated = img.rotate(angle, expand=True)
        edit_id = str(uuid.uuid4())
        out = self._output_path(edit_id)
        _save_image(rotated, out)
        return self._record(source_path, "rotate", out, source_id,
                            params={"angle": angle})

    def flip(self, source_path: str, direction: str = "horizontal",
             source_id: Optional[str] = None) -> EditRecord:
        """Flip image horizontally or vertically."""
        Image, _, _ = _ensure_pil()
        img = _load_image(source_path)
        if direction == "horizontal":
            flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif direction == "vertical":
            flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            raise ValueError(f"Invalid flip direction: {direction}")
        edit_id = str(uuid.uuid4())
        out = self._output_path(edit_id)
        _save_image(flipped, out)
        return self._record(source_path, "flip", out, source_id,
                            params={"direction": direction})

    # ── Color Adjustments ────────────────────────────────────────────

    def adjust_brightness(self, source_path: str, factor: float,
                          source_id: Optional[str] = None) -> EditRecord:
        """Adjust brightness. factor: 0.0=black, 1.0=original, 2.0=2x bright."""
        _, ImageEnhance, _ = _ensure_pil()
        img = _load_image(source_path)
        enhanced = ImageEnhance.Brightness(img).enhance(factor)
        edit_id = str(uuid.uuid4())
        out = self._output_path(edit_id)
        _save_image(enhanced, out)
        return self._record(source_path, "brightness", out, source_id,
                            params={"factor": factor})

    def adjust_contrast(self, source_path: str, factor: float,
                        source_id: Optional[str] = None) -> EditRecord:
        """Adjust contrast. factor: 0.0=grey, 1.0=original, 2.0=2x contrast."""
        _, ImageEnhance, _ = _ensure_pil()
        img = _load_image(source_path)
        enhanced = ImageEnhance.Contrast(img).enhance(factor)
        edit_id = str(uuid.uuid4())
        out = self._output_path(edit_id)
        _save_image(enhanced, out)
        return self._record(source_path, "contrast", out, source_id,
                            params={"factor": factor})

    def adjust_saturation(self, source_path: str, factor: float,
                          source_id: Optional[str] = None) -> EditRecord:
        """Adjust color saturation."""
        _, ImageEnhance, _ = _ensure_pil()
        img = _load_image(source_path)
        enhanced = ImageEnhance.Color(img).enhance(factor)
        edit_id = str(uuid.uuid4())
        out = self._output_path(edit_id)
        _save_image(enhanced, out)
        return self._record(source_path, "saturation", out, source_id,
                            params={"factor": factor})

    def blur(self, source_path: str, radius: float = 2.0,
             source_id: Optional[str] = None) -> EditRecord:
        """Apply Gaussian blur."""
        _, _, ImageFilter = _ensure_pil()
        img = _load_image(source_path)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
        edit_id = str(uuid.uuid4())
        out = self._output_path(edit_id)
        _save_image(blurred, out)
        return self._record(source_path, "blur", out, source_id,
                            params={"radius": radius})

    def sharpen(self, source_path: str, factor: float = 2.0,
                source_id: Optional[str] = None) -> EditRecord:
        """Sharpen image."""
        _, ImageEnhance, _ = _ensure_pil()
        img = _load_image(source_path)
        enhanced = ImageEnhance.Sharpness(img).enhance(factor)
        edit_id = str(uuid.uuid4())
        out = self._output_path(edit_id)
        _save_image(enhanced, out)
        return self._record(source_path, "sharpen", out, source_id,
                            params={"factor": factor})

    # ── ComfyUI-backed edits ─────────────────────────────────────────

    def img2img(self, source_path: str, prompt: str,
                denoise: float = 0.65, steps: int = 20,
                source_id: Optional[str] = None,
                comfyui_url: str = "http://127.0.0.1:8188") -> EditRecord:
        """
        Send image to ComfyUI for img2img transformation.
        Falls back to a simple overlay if ComfyUI is unavailable.
        """
        import requests

        edit_id = str(uuid.uuid4())
        out = self._output_path(edit_id)
        params = {"prompt": prompt, "denoise": denoise, "steps": steps}

        try:
            # Upload image to ComfyUI
            with open(source_path, "rb") as f:
                upload_resp = requests.post(
                    f"{comfyui_url}/upload/image",
                    files={"image": (Path(source_path).name, f, "image/png")},
                    timeout=10,
                )
            if upload_resp.status_code != 200:
                raise RuntimeError(f"ComfyUI upload failed: {upload_resp.status_code}")

            uploaded = upload_resp.json()
            image_name = uploaded.get("name", Path(source_path).name)

            # Build img2img workflow
            workflow = self._build_img2img_workflow(
                image_name=image_name,
                prompt=prompt,
                denoise=denoise,
                steps=steps,
            )

            # Queue the workflow
            queue_resp = requests.post(
                f"{comfyui_url}/prompt",
                json={"prompt": workflow},
                timeout=10,
            )
            if queue_resp.status_code != 200:
                raise RuntimeError(f"ComfyUI queue failed: {queue_resp.status_code}")

            prompt_id = queue_resp.json().get("prompt_id")

            # Poll for completion (max 120s)
            for _ in range(120):
                time.sleep(1)
                hist_resp = requests.get(f"{comfyui_url}/history/{prompt_id}", timeout=5)
                if hist_resp.status_code == 200:
                    hist = hist_resp.json()
                    if prompt_id in hist:
                        outputs = hist[prompt_id].get("outputs", {})
                        for node_id, node_out in outputs.items():
                            images = node_out.get("images", [])
                            if images:
                                img_info = images[0]
                                img_url = f"{comfyui_url}/view?filename={img_info['filename']}&subfolder={img_info.get('subfolder', '')}&type={img_info.get('type', 'output')}"
                                img_resp = requests.get(img_url, timeout=10)
                                if img_resp.status_code == 200:
                                    with open(out, "wb") as f:
                                        f.write(img_resp.content)
                                    return self._record(
                                        source_path, "img2img", out, source_id,
                                        prompt=prompt, params=params,
                                        model="comfyui",
                                    )
                        break

            raise RuntimeError("ComfyUI img2img timed out or produced no output")

        except Exception as e:
            logger.warning(f"ComfyUI img2img failed, falling back to local edit: {e}")
            # Fallback: just copy the source and note the failure
            import shutil
            shutil.copy2(source_path, out)
            return self._record(
                source_path, "img2img_fallback", out, source_id,
                prompt=prompt,
                params={**params, "fallback_reason": str(e)},
            )

    def _build_img2img_workflow(self, image_name: str, prompt: str,
                                denoise: float, steps: int) -> dict:
        """Build a simple ComfyUI img2img workflow."""
        import random
        seed = random.randint(0, 2**32 - 1)
        return {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": image_name},
            },
            "2": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["2", 1]},
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "", "clip": ["2", 1]},
            },
            "5": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["1", 0], "vae": ["2", 2]},
            },
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": 7.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": denoise,
                    "model": ["2", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0],
                },
            },
            "7": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["6", 0], "vae": ["2", 2]},
            },
            "8": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "edison_edit", "images": ["7", 0]},
            },
        }

    # ── History ──────────────────────────────────────────────────────

    def get_history(self, limit: int = 50) -> List[Dict]:
        """Get recent edit history."""
        return [r.to_dict() for r in self._history[-limit:]]

    def get_edit(self, edit_id: str) -> Optional[EditRecord]:
        """Get a specific edit record."""
        for r in self._history:
            if r.edit_id == edit_id:
                return r
        return None


# ── Singleton ────────────────────────────────────────────────────────────

_editor = None

def get_image_editor() -> ImageEditor:
    global _editor
    if _editor is None:
        _editor = ImageEditor()
    return _editor
