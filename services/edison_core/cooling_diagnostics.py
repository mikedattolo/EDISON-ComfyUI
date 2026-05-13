"""Reusable GPU cooling diagnostics.

The fan controller can read telemetry from several sources, and not every
NVIDIA card exposes the same fan fields on Linux.  This module keeps the
classification logic pure and testable so the API, diagnostics dashboard, and
heavy-job governors can all agree on what a reading means.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


COOLING_STATES = {
    "normal",
    "zero_rpm_idle_probably_normal",
    "telemetry_unavailable",
    "manual_control_unavailable",
    "suspect_fan_not_spinning",
    "over_temp_warning",
}


@dataclass
class CoolingPolicy:
    """Thresholds used to classify GPU cooling telemetry."""

    zero_rpm_idle_temp_c: float = 45.0
    elevated_temp_c: float = 55.0
    critical_temp_c: float = 84.0
    active_utilization_percent: float = 25.0
    elevated_duration_s: float = 60.0
    fan_zero_percent_threshold: float = 1.0

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "CoolingPolicy":
        cfg = dict(config or {})
        return cls(
            zero_rpm_idle_temp_c=float(cfg.get("zero_rpm_idle_temp_c", cfg.get("zero_rpm_idle_max_temp_c", 45.0))),
            elevated_temp_c=float(cfg.get("elevated_temp_c", cfg.get("gpu_temperature_warning_threshold_c", 55.0))),
            critical_temp_c=float(cfg.get("critical_temp_c", cfg.get("gpu_temperature_critical_threshold_c", 84.0))),
            active_utilization_percent=float(cfg.get("active_utilization_percent", 25.0)),
            elevated_duration_s=float(cfg.get("elevated_duration_s", 60.0)),
            fan_zero_percent_threshold=float(cfg.get("fan_zero_percent_threshold", 1.0)),
        )


@dataclass
class CoolingHealth:
    """Serializable cooling health state for one GPU."""

    state: str
    severity: str
    summary: str
    recommended_action: str
    temperature_c: Optional[float] = None
    utilization_percent: Optional[float] = None
    fan_speed_percent: Optional[float] = None
    fan_rpm: Optional[float] = None
    evidence: List[str] = field(default_factory=list)
    observed_for_s: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _num(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        text = str(value).strip().replace("%", "")
        if not text or text.lower() in {"n/a", "none", "nan", "[not supported]"}:
            return None
        return float(text)
    except Exception:
        return None


def classify_cooling_health(
    *,
    temperature_c: Any = None,
    utilization_percent: Any = None,
    fan_speed_percent: Any = None,
    fan_rpm: Any = None,
    telemetry_available: bool = True,
    manual_control_available: Optional[bool] = None,
    observed_for_s: Optional[float] = None,
    policy: Optional[CoolingPolicy] = None,
) -> CoolingHealth:
    """Classify one GPU cooling reading.

    The important distinction is that zero-RPM fan mode is normal on many
    cards when the GPU is cool.  A zero fan reading becomes suspicious only
    when the GPU is warm, loaded, or persistently elevated.
    """

    p = policy or CoolingPolicy()
    temp = _num(temperature_c)
    util = _num(utilization_percent)
    fan = _num(fan_speed_percent)
    rpm = _num(fan_rpm)
    evidence: List[str] = []

    if not telemetry_available or (temp is None and fan is None and rpm is None):
        return CoolingHealth(
            state="telemetry_unavailable",
            severity="unknown",
            summary="GPU cooling telemetry is not available from the driver or this execution environment.",
            recommended_action=(
                "Run diagnostics on the Edison host with nvidia-smi/NVML access. "
                "In containers, expose NVIDIA devices and driver capabilities."
            ),
            temperature_c=temp,
            utilization_percent=util,
            fan_speed_percent=fan,
            fan_rpm=rpm,
            evidence=["missing temperature/fan telemetry"],
            observed_for_s=observed_for_s,
        )

    fan_zero = False
    if fan is not None:
        fan_zero = fan <= p.fan_zero_percent_threshold
        evidence.append(f"fan_percent={fan:g}")
    elif rpm is not None:
        fan_zero = rpm <= 1
        evidence.append(f"fan_rpm={rpm:g}")
    else:
        evidence.append("fan telemetry unavailable")

    if temp is not None:
        evidence.append(f"temperature_c={temp:g}")
    if util is not None:
        evidence.append(f"utilization_percent={util:g}")

    if temp is not None and temp >= p.critical_temp_c:
        state = "over_temp_warning"
        if fan_zero:
            state = "suspect_fan_not_spinning"
            evidence.append("critical temperature with zero fan reading")
        return CoolingHealth(
            state=state,
            severity="critical",
            summary="GPU temperature is at or above the configured critical threshold.",
            recommended_action=(
                "Stop starting new heavy jobs, inspect physical fans and airflow, "
                "and verify fan-control permissions before continuing long renders."
            ),
            temperature_c=temp,
            utilization_percent=util,
            fan_speed_percent=fan,
            fan_rpm=rpm,
            evidence=evidence,
            observed_for_s=observed_for_s,
        )

    elevated = temp is not None and temp >= p.elevated_temp_c
    active = util is not None and util >= p.active_utilization_percent
    sustained = observed_for_s is not None and observed_for_s >= p.elevated_duration_s

    if fan_zero and temp is not None and temp <= p.zero_rpm_idle_temp_c and not active:
        return CoolingHealth(
            state="zero_rpm_idle_probably_normal",
            severity="info",
            summary="Fan reports zero while the GPU is cool and not meaningfully loaded; this is usually normal zero-RPM idle behavior.",
            recommended_action="No repair needed unless the fan remains at zero after temperature or workload rises.",
            temperature_c=temp,
            utilization_percent=util,
            fan_speed_percent=fan,
            fan_rpm=rpm,
            evidence=evidence,
            observed_for_s=observed_for_s,
        )

    if fan_zero and (elevated or active or sustained):
        if elevated:
            evidence.append("temperature is elevated")
        if active:
            evidence.append("GPU utilization indicates active workload")
        if sustained:
            evidence.append("condition is sustained")
        return CoolingHealth(
            state="suspect_fan_not_spinning",
            severity="warning",
            summary="Fan is reporting zero while the GPU is warm, active, or persistently elevated.",
            recommended_action=(
                "Inspect the GPU fans, fan cable, shroud, and driver fan-control state. "
                "This may be normal only if telemetry is wrong or the card has a delayed fan curve."
            ),
            temperature_c=temp,
            utilization_percent=util,
            fan_speed_percent=fan,
            fan_rpm=rpm,
            evidence=evidence,
            observed_for_s=observed_for_s,
        )

    if manual_control_available is False:
        return CoolingHealth(
            state="manual_control_unavailable",
            severity="info",
            summary="Cooling telemetry is readable, but manual fan control is not available in this Linux/NVIDIA configuration.",
            recommended_action="Use diagnostics-only mode or configure Xorg/Coolbits/nvidia-settings on the host for fan writes.",
            temperature_c=temp,
            utilization_percent=util,
            fan_speed_percent=fan,
            fan_rpm=rpm,
            evidence=evidence,
            observed_for_s=observed_for_s,
        )

    return CoolingHealth(
        state="normal",
        severity="ok",
        summary="Cooling telemetry is within expected bounds.",
        recommended_action="No action required.",
        temperature_c=temp,
        utilization_percent=util,
        fan_speed_percent=fan,
        fan_rpm=rpm,
        evidence=evidence,
        observed_for_s=observed_for_s,
    )


def aggregate_cooling_health(items: Iterable[Dict[str, Any] | CoolingHealth]) -> Dict[str, Any]:
    """Return a compact dashboard summary for multiple GPUs."""

    serialized: List[Dict[str, Any]] = [item.to_dict() if isinstance(item, CoolingHealth) else dict(item) for item in items]
    severity_rank = {"ok": 0, "info": 1, "unknown": 2, "warning": 3, "critical": 4}
    worst = "ok"
    counts: Dict[str, int] = {}
    alerts: List[Dict[str, Any]] = []
    for item in serialized:
        state = item.get("state", "telemetry_unavailable")
        sev = item.get("severity", "unknown")
        counts[state] = counts.get(state, 0) + 1
        if severity_rank.get(sev, 2) > severity_rank.get(worst, 0):
            worst = sev
        if sev in {"warning", "critical", "unknown"}:
            alerts.append(item)
    return {
        "overall_severity": worst,
        "counts_by_state": counts,
        "alerts": alerts,
        "gpu_count": len(serialized),
    }
