"""
Video timeline, shot list, storyboard, and export-preset helpers.

Phase 3 goal: turn EDISON's video tab from a single-operation tool into a
production workspace. This module is the data layer behind that —
front-end clip-sequence and storyboard panels consume these structures.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Export presets ───────────────────────────────────────────────────

@dataclass(frozen=True)
class ExportPreset:
    name: str
    width: int
    height: int
    aspect: str
    fps: int
    max_duration_s: int
    container: str
    video_codec: str
    audio_codec: str
    bitrate_kbps: int
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


EXPORT_PRESETS: Dict[str, ExportPreset] = {
    "instagram_reel": ExportPreset(
        name="Instagram Reel",
        width=1080, height=1920, aspect="9:16",
        fps=30, max_duration_s=90,
        container="mp4", video_codec="h264", audio_codec="aac",
        bitrate_kbps=10000,
        notes="Vertical short-form. Keep key action in the center safe area.",
    ),
    "instagram_feed": ExportPreset(
        name="Instagram Feed",
        width=1080, height=1080, aspect="1:1",
        fps=30, max_duration_s=60,
        container="mp4", video_codec="h264", audio_codec="aac",
        bitrate_kbps=8000,
    ),
    "tiktok": ExportPreset(
        name="TikTok",
        width=1080, height=1920, aspect="9:16",
        fps=30, max_duration_s=180,
        container="mp4", video_codec="h264", audio_codec="aac",
        bitrate_kbps=10000,
    ),
    "youtube_shorts": ExportPreset(
        name="YouTube Shorts",
        width=1080, height=1920, aspect="9:16",
        fps=30, max_duration_s=60,
        container="mp4", video_codec="h264", audio_codec="aac",
        bitrate_kbps=10000,
    ),
    "youtube_landscape": ExportPreset(
        name="YouTube Landscape",
        width=1920, height=1080, aspect="16:9",
        fps=30, max_duration_s=600,
        container="mp4", video_codec="h264", audio_codec="aac",
        bitrate_kbps=12000,
    ),
    "ad_landscape": ExportPreset(
        name="Ad Landscape",
        width=1920, height=1080, aspect="16:9",
        fps=30, max_duration_s=60,
        container="mp4", video_codec="h264", audio_codec="aac",
        bitrate_kbps=15000,
        notes="High-bitrate cut for paid placements.",
    ),
}


def list_presets() -> List[Dict[str, Any]]:
    return [p.to_dict() for p in EXPORT_PRESETS.values()]


def get_preset(name: str) -> Optional[ExportPreset]:
    return EXPORT_PRESETS.get(name)


# ── Shot list / storyboard ───────────────────────────────────────────

@dataclass
class Shot:
    id: int
    description: str
    duration_s: float
    shot_type: str  # "wide", "medium", "close", "insert", "overhead", etc.
    camera_motion: str = "static"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShotList:
    topic: str
    target_duration_s: float
    preset: Optional[str]
    shots: List[Shot] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "target_duration_s": self.target_duration_s,
            "preset": self.preset,
            "shots": [s.to_dict() for s in self.shots],
            "total_duration_s": sum(s.duration_s for s in self.shots),
        }


# Heuristic storyboard generator. We deliberately keep this rule-based so
# it works without an LLM call; chat-driven generation can replace this
# later by feeding a richer plan into ``ShotList``.
_DEFAULT_BEATS = [
    ("Hook / opening visual", "wide", "slow push-in"),
    ("Subject reveal",        "medium", "static"),
    ("Detail / texture",      "close", "static"),
    ("Action / use",          "medium", "handheld"),
    ("Insert / supporting",   "insert", "static"),
    ("Branding moment",       "close", "static"),
    ("Closing call to action", "medium", "static"),
]


def generate_shot_list(
    topic: str,
    *,
    preset: str = "instagram_reel",
    target_duration_s: Optional[float] = None,
    beat_count: Optional[int] = None,
) -> ShotList:
    """Generate a heuristic shot list/storyboard for ``topic``.

    ``target_duration_s`` defaults to 80% of the chosen preset's max
    duration so creators have room to breathe. The beats are evenly
    distributed across the duration with a 1.5s minimum per shot.
    """
    p = get_preset(preset)
    if p is None:
        raise ValueError(f"unknown export preset: {preset}")

    if target_duration_s is None:
        target_duration_s = max(8.0, p.max_duration_s * 0.5)
    target_duration_s = float(target_duration_s)

    beats = list(_DEFAULT_BEATS)
    if beat_count:
        beats = beats[: max(2, min(beat_count, len(beats)))]
    # Scale beats proportionally
    per_shot = max(1.5, target_duration_s / len(beats))
    shots: List[Shot] = []
    for idx, (desc, shot_type, motion) in enumerate(beats, start=1):
        shots.append(Shot(
            id=idx,
            description=f"{desc} — {topic}",
            duration_s=round(per_shot, 2),
            shot_type=shot_type,
            camera_motion=motion,
        ))
    return ShotList(
        topic=topic,
        target_duration_s=target_duration_s,
        preset=preset,
        shots=shots,
    )


# ── Clip sequence ────────────────────────────────────────────────────

@dataclass
class Clip:
    id: str
    source: str          # path or url
    in_s: float = 0.0
    out_s: Optional[float] = None
    transition_in: Optional[str] = None   # e.g. "fade"
    transition_out: Optional[str] = None
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def duration_s(self) -> Optional[float]:
        if self.out_s is None:
            return None
        return max(0.0, self.out_s - self.in_s)


class ClipSequence:
    """Ordered sequence of clips with rough duration tracking.

    Persistence is JSON-on-disk under ``data/video_sequences/`` so the
    front-end and chat both observe the same state.
    """

    REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
    DEFAULT_ROOT = REPO_ROOT / "data" / "video_sequences"

    def __init__(
        self,
        sequence_id: Optional[str] = None,
        *,
        title: Optional[str] = None,
        preset: Optional[str] = None,
        root: Optional[Path] = None,
    ) -> None:
        self.sequence_id = sequence_id or f"seq_{uuid.uuid4().hex[:10]}"
        self.title = title
        self.preset = preset
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.clips: List[Clip] = []
        self.root = (root or self.DEFAULT_ROOT).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ── persistence ────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        return self.root / f"{self.sequence_id}.json"

    def save(self) -> Path:
        self.updated_at = time.time()
        payload = {
            "sequence_id": self.sequence_id,
            "title": self.title,
            "preset": self.preset,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "clips": [c.to_dict() for c in self.clips],
        }
        self.path.write_text(json.dumps(payload, indent=2, default=str))
        return self.path

    @classmethod
    def load(cls, sequence_id: str, *, root: Optional[Path] = None) -> "ClipSequence":
        seq = cls(sequence_id, root=root)
        if not seq.path.exists():
            raise FileNotFoundError(f"sequence not found: {sequence_id}")
        payload = json.loads(seq.path.read_text())
        seq.title = payload.get("title")
        seq.preset = payload.get("preset")
        seq.created_at = float(payload.get("created_at") or time.time())
        seq.updated_at = float(payload.get("updated_at") or time.time())
        seq.clips = [Clip(**c) for c in payload.get("clips", [])]
        return seq

    @classmethod
    def list_sequences(cls, *, root: Optional[Path] = None) -> List[Dict[str, Any]]:
        base = (root or cls.DEFAULT_ROOT).resolve()
        if not base.exists():
            return []
        out: List[Dict[str, Any]] = []
        for path in sorted(base.glob("*.json")):
            try:
                p = json.loads(path.read_text())
                out.append({
                    "sequence_id": p.get("sequence_id", path.stem),
                    "title": p.get("title"),
                    "preset": p.get("preset"),
                    "clip_count": len(p.get("clips", [])),
                    "updated_at": p.get("updated_at"),
                })
            except Exception:
                continue
        return out

    # ── editing ────────────────────────────────────────────────────

    def add_clip(self, clip: Clip) -> Clip:
        self.clips.append(clip)
        return clip

    def remove_clip(self, clip_id: str) -> bool:
        before = len(self.clips)
        self.clips = [c for c in self.clips if c.id != clip_id]
        return len(self.clips) < before

    def move_clip(self, clip_id: str, new_index: int) -> bool:
        for i, c in enumerate(self.clips):
            if c.id == clip_id:
                clip = self.clips.pop(i)
                self.clips.insert(max(0, min(new_index, len(self.clips))), clip)
                return True
        return False

    @property
    def total_duration_s(self) -> float:
        total = 0.0
        for c in self.clips:
            if c.duration_s is not None:
                total += c.duration_s
        return round(total, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "title": self.title,
            "preset": self.preset,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "clips": [c.to_dict() for c in self.clips],
            "total_duration_s": self.total_duration_s,
        }

    # ── export plan ────────────────────────────────────────────────

    def export_plan(self, preset_name: Optional[str] = None) -> Dict[str, Any]:
        """Return a structured plan describing what an export would do.

        This does *not* invoke ffmpeg — it returns the parameters that the
        existing video pipeline can consume, plus warnings if the chosen
        preset can't accommodate the sequence.
        """
        chosen = preset_name or self.preset or "instagram_reel"
        preset = get_preset(chosen)
        if preset is None:
            raise ValueError(f"unknown preset: {chosen}")
        warnings: List[str] = []
        total = self.total_duration_s
        if total > preset.max_duration_s:
            warnings.append(
                f"sequence is {total:.1f}s but preset max is {preset.max_duration_s}s — "
                f"will need to be trimmed"
            )
        if not self.clips:
            warnings.append("sequence has no clips")
        return {
            "sequence_id": self.sequence_id,
            "preset": preset.to_dict(),
            "clip_count": len(self.clips),
            "total_duration_s": total,
            "warnings": warnings,
        }
