"""Quality-control helpers for Persona Video Studio segments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SegmentQCResult:
    status: str
    score: float
    checks: Dict[str, Any] = field(default_factory=dict)
    warning_flags: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    needs_review: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _probe(path: Path) -> Dict[str, Any]:
    try:
        from .persona_video import probe_video

        return probe_video(path)
    except Exception as exc:
        return {"probe_backend": "unavailable", "warning": str(exc)}


def basic_segment_qc(
    *,
    output_path: Path,
    source_probe: Optional[Dict[str, Any]] = None,
    segment: Optional[Dict[str, Any]] = None,
    backend_result: Optional[Dict[str, Any]] = None,
    expect_audio: bool = False,
    duration_tolerance_s: float = 0.75,
) -> SegmentQCResult:
    checks: Dict[str, Any] = {}
    warnings: List[str] = []
    errors: List[str] = []
    score = 1.0

    checks["exists"] = output_path.exists()
    if not output_path.exists():
        return SegmentQCResult(
            status="failed",
            score=0.0,
            checks=checks,
            warning_flags=["missing_output_file"],
            errors=["Segment output file is missing."],
            needs_review=True,
        )
    size = output_path.stat().st_size
    checks["size_bytes"] = size
    if size <= 0:
        return SegmentQCResult(
            status="failed",
            score=0.0,
            checks=checks,
            warning_flags=["empty_output_file"],
            errors=["Segment output file is empty."],
            needs_review=True,
        )

    if backend_result and backend_result.get("ok") is False:
        score -= 0.35
        warnings.append("failed_backend_response")
        errors.append(str(backend_result.get("error") or "Backend reported failure."))

    probe = _probe(output_path)
    checks["probe"] = probe
    expected_duration = None
    if segment is not None:
        expected_duration = segment.get("duration_s")
    if expected_duration is None and source_probe:
        expected_duration = source_probe.get("duration_s")
    actual_duration = probe.get("duration_s")
    if expected_duration and actual_duration:
        diff = abs(float(actual_duration) - float(expected_duration))
        checks["duration_diff_s"] = round(diff, 3)
        if diff > duration_tolerance_s:
            score -= 0.2
            warnings.append("duration_mismatch")

    expected_fps = (source_probe or {}).get("fps")
    actual_fps = probe.get("fps")
    if expected_fps and actual_fps and abs(float(expected_fps) - float(actual_fps)) > 1.0:
        score -= 0.08
        warnings.append("frame_rate_mismatch")

    if expect_audio and not probe.get("audio_present"):
        score -= 0.15
        warnings.append("expected_audio_missing")

    status = "pass"
    needs_review = False
    if errors:
        status = "failed"
        needs_review = True
    elif score < 0.6:
        status = "needs_review"
        needs_review = True
    elif warnings:
        status = "warning"

    return SegmentQCResult(
        status=status,
        score=round(max(0.0, min(1.0, score)), 3),
        checks=checks,
        warning_flags=warnings,
        errors=errors,
        needs_review=needs_review,
    )


def merge_backend_qc(backend_qc: Dict[str, Any], basic_qc: SegmentQCResult) -> Dict[str, Any]:
    payload = dict(backend_qc or {})
    base = basic_qc.to_dict()
    backend_score = float(payload.get("score") or 0.0)
    combined = round((backend_score * 0.55) + (basic_qc.score * 0.45), 3) if payload else basic_qc.score
    flags = list(payload.get("warning_flags") or [])
    flags.extend(base.get("warning_flags") or [])
    payload.update(
        {
            "basic_qc": base,
            "warning_flags": sorted(set(flags)),
            "score": combined,
            "status": base["status"] if base["status"] in {"failed", "needs_review"} else payload.get("status", base["status"]),
            "needs_review": bool(payload.get("needs_review")) or bool(base.get("needs_review")),
        }
    )
    return payload
