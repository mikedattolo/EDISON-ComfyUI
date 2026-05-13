"""Vision payload validation, tracing, and confidence heuristics."""

from __future__ import annotations

import base64
import hashlib
import io
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


SUPPORTED_IMAGE_MIME = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif", "image/bmp"}


@dataclass
class VisionImageInfo:
    ok: bool
    error: str = ""
    data_uri: Optional[str] = None
    mime_type: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    byte_length: int = 0
    sha256_12: str = ""
    resized: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("data_uri", None)
        return data


def normalize_data_uri(raw_image: str) -> Optional[Tuple[str, str]]:
    """Return (mime_type, base64_payload) for a data URI or raw base64 string."""

    if not isinstance(raw_image, str) or not raw_image.strip():
        return None
    value = raw_image.strip()
    if value.startswith("data:") and "," in value:
        header, payload = value.split(",", 1)
        match = re.match(r"^data:([^;,]+);base64$", header.strip(), re.I)
        if not match:
            return None
        return match.group(1).lower(), payload.strip()
    return "image/png", value


def prepare_vision_image(raw_image: str, max_dim: int = 1024) -> VisionImageInfo:
    """Validate, resize, and re-encode a vision image payload.

    Broken base64, empty payloads, unsupported formats, and non-image bytes are
    rejected.  The returned data URI is safe to send to llama-cpp vision chat
    handlers.
    """

    normalized = normalize_data_uri(raw_image)
    if not normalized:
        return VisionImageInfo(ok=False, error="image payload is empty or not a base64 data URI")
    mime_type, payload = normalized
    if mime_type not in SUPPORTED_IMAGE_MIME:
        return VisionImageInfo(ok=False, error=f"unsupported image MIME type: {mime_type}", mime_type=mime_type)
    try:
        raw_bytes = base64.b64decode(payload, validate=True)
    except Exception:
        return VisionImageInfo(ok=False, error="image payload is not valid base64", mime_type=mime_type)
    if not raw_bytes:
        return VisionImageInfo(ok=False, error="image payload decoded to zero bytes", mime_type=mime_type)
    sha = hashlib.sha256(raw_bytes).hexdigest()[:12]

    try:
        from PIL import Image
    except Exception:
        return VisionImageInfo(
            ok=True,
            data_uri=f"data:{mime_type};base64,{base64.b64encode(raw_bytes).decode('ascii')}",
            mime_type=mime_type,
            byte_length=len(raw_bytes),
            sha256_12=sha,
            warnings=["Pillow is not installed; image bytes could not be dimension-validated or resized."],
        )

    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.verify()
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        return VisionImageInfo(ok=False, error="payload decodes but is not a readable image", mime_type=mime_type, byte_length=len(raw_bytes), sha256_12=sha)

    width, height = img.size
    if width <= 0 or height <= 0:
        return VisionImageInfo(ok=False, error="image has invalid dimensions", mime_type=mime_type, byte_length=len(raw_bytes), sha256_12=sha)
    resized = False
    if max(width, height) > max_dim:
        scale = max_dim / max(width, height)
        img = img.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS)
        resized = True

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85, optimize=True)
    out_bytes = out.getvalue()
    return VisionImageInfo(
        ok=True,
        data_uri="data:image/jpeg;base64," + base64.b64encode(out_bytes).decode("ascii"),
        mime_type="image/jpeg",
        width=img.size[0],
        height=img.size[1],
        byte_length=len(out_bytes),
        sha256_12=hashlib.sha256(out_bytes).hexdigest()[:12],
        resized=resized,
    )


def vision_grounding_system_prompt() -> str:
    return (
        "You are Edison Vision. Answer only from visible evidence in the provided image(s). "
        "Be literal and grounded: mention concrete visual details, spatial relationships, text you can read, "
        "and uncertainty when something is unclear. Do not invent objects, people, brands, scenes, or events "
        "that are not visible. If the image is missing, corrupted, too small, or ambiguous, say so directly."
    )


GENERIC_LOW_CONFIDENCE_PATTERNS = [
    r"\bas an ai\b",
    r"\bi cannot (view|see|access) (the )?image\b",
    r"\bwithout (seeing|viewing) the image\b",
    r"\bit appears to be an image\b",
    r"\bthe image likely\b",
    r"\bthere may be\b",
    r"\bi don't have enough information\b",
]


def assess_vision_response_confidence(text: str, *, image_count: int = 0) -> Dict[str, Any]:
    body = (text or "").strip()
    lowered = body.lower()
    flags: List[str] = []
    if image_count <= 0:
        flags.append("no_image_present")
    if len(body.split()) < 8:
        flags.append("too_short")
    for pattern in GENERIC_LOW_CONFIDENCE_PATTERNS:
        if re.search(pattern, lowered):
            flags.append("generic_or_nonvisual_response")
            break
    visible_detail_terms = re.findall(r"\b(color|left|right|top|bottom|foreground|background|text|person|object|screen|photo|image|visible)\b", lowered)
    if image_count > 0 and len(visible_detail_terms) == 0 and len(body.split()) > 12:
        flags.append("lacks_visible_evidence_terms")
    confidence = "low" if flags else "normal"
    return {"confidence": confidence, "flags": sorted(set(flags)), "image_count": image_count}


def new_vision_trace_id() -> str:
    return f"vis_{uuid.uuid4().hex[:12]}"
