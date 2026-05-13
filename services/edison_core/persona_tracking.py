"""Target performer tracking interfaces for Persona Video Studio."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class PersonaTrack:
    track_id: str
    subject_label: str
    per_segment_coverage: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    representative_thumbnails: List[str] = field(default_factory=list)
    boxes: Dict[str, List[Dict[str, float]]] = field(default_factory=dict)
    masks: Dict[str, str] = field(default_factory=dict)
    provider: str = "metadata_only"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrackerProvider:
    provider_id = "base"

    def is_available(self) -> bool:
        return True

    def describe(self) -> Dict[str, Any]:
        return {"provider_id": self.provider_id, "available": self.is_available()}

    def track(self, job: Dict[str, Any], segments: List[Dict[str, Any]], workspace: Path) -> List[PersonaTrack]:
        raise NotImplementedError


class MetadataOnlyTrackerProvider(TrackerProvider):
    provider_id = "metadata_only"

    def describe(self) -> Dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "available": True,
            "automatic_detection": False,
            "summary": "No detector/tracker dependency is configured; Edison records target metadata and per-segment coverage placeholders.",
        }

    def track(self, job: Dict[str, Any], segments: List[Dict[str, Any]], workspace: Path) -> List[PersonaTrack]:
        target = job.get("target_selection") or {}
        mode = target.get("mode") or "auto_detect_primary_subject"
        label = target.get("subject_label") or ("primary subject" if mode == "auto_detect_primary_subject" else "selected subject")
        track_id = target.get("track_id") or f"track_{uuid.uuid4().hex[:8]}"
        coverage = {str(seg.get("segment_id")): 1.0 for seg in segments}
        confidence = 0.58 if mode == "auto_detect_primary_subject" else 0.76
        return [
            PersonaTrack(
                track_id=track_id,
                subject_label=label,
                per_segment_coverage=coverage,
                confidence=confidence,
                provider=self.provider_id,
                notes="Metadata-only target track. Install a detector/tracker provider for boxes, masks, and stronger candidate selection.",
            )
        ]


class OptionalDetectorTrackerProvider(TrackerProvider):
    """Optional placeholder for detector-backed tracking.

    The provider advertises availability only when OpenCV is importable.  It
    still returns metadata-only tracks because no face/body model is bundled in
    this repository; the explicit provider boundary lets future installations
    add a real detector without changing PersonaVideoService.
    """

    provider_id = "opencv_detector_placeholder"

    def is_available(self) -> bool:
        try:
            import cv2  # noqa: F401
            return True
        except Exception:
            return False

    def describe(self) -> Dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "available": self.is_available(),
            "automatic_detection": False,
            "summary": "OpenCV is present, but no bundled identity/body detector model is configured.",
            "setup_required": ["Configure a detector model/provider before expecting automatic subject boxes or masks."],
        }

    def track(self, job: Dict[str, Any], segments: List[Dict[str, Any]], workspace: Path) -> List[PersonaTrack]:
        return MetadataOnlyTrackerProvider().track(job, segments, workspace)


class TrackingService:
    def __init__(self, providers: Optional[Iterable[TrackerProvider]] = None) -> None:
        self.providers = list(providers or [OptionalDetectorTrackerProvider(), MetadataOnlyTrackerProvider()])

    def capabilities(self) -> List[Dict[str, Any]]:
        return [provider.describe() for provider in self.providers]

    def choose_provider(self, requested: str = "auto") -> TrackerProvider:
        requested = (requested or "auto").strip()
        if requested and requested != "auto":
            for provider in self.providers:
                if provider.provider_id == requested:
                    return provider
        for provider in self.providers:
            if provider.is_available() and provider.provider_id != "metadata_only":
                return provider
        return MetadataOnlyTrackerProvider()

    def track(self, job: Dict[str, Any], segments: List[Dict[str, Any]], workspace: Path) -> Dict[str, Any]:
        provider = self.choose_provider((job.get("settings") or {}).get("tracker_provider", "auto"))
        tracks = provider.track(job, segments, workspace)
        return {
            "provider": provider.describe(),
            "tracks": [track.to_dict() for track in tracks],
            "candidate_count": len(tracks),
            "fallback": provider.provider_id == "metadata_only" or not provider.describe().get("automatic_detection", False),
        }
