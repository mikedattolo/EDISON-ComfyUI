"""Safe NVIDIA GPU fan control helpers for EDISON.

This module provides a conservative fan controller for three-card NVIDIA
workstations. It is designed to run on the host where the NVIDIA driver,
NVML/nvidia-smi, nvidia-settings, Xorg Coolbits, and /dev/nvidia* devices are
actually available. In an unprivileged dev container it will still return useful
diagnostics, but it will not be able to command fans.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .cooling_diagnostics import CoolingPolicy, aggregate_cooling_health, classify_cooling_health

logger = logging.getLogger(__name__)

DEFAULT_CURVE = [
    {"temp_c": 35, "fan_percent": 30},
    {"temp_c": 45, "fan_percent": 36},
    {"temp_c": 55, "fan_percent": 48},
    {"temp_c": 65, "fan_percent": 62},
    {"temp_c": 72, "fan_percent": 76},
    {"temp_c": 80, "fan_percent": 92},
    {"temp_c": 84, "fan_percent": 100},
]

DEFAULT_EXPECTED_GPUS = [
    {"index": 0, "name_contains": "3090", "label": "RTX 3090 24GB"},
    {"index": 1, "name_contains": "5060", "label": "RTX 5060 Ti 16GB"},
    {"index": 2, "name_contains": "4060", "label": "RTX 4060 Ti 16GB"},
]


@dataclass
class GpuFanReading:
    index: int
    name: str
    temperature_c: Optional[float] = None
    fan_speed_percent: Optional[float] = None
    fan_rpm: Optional[float] = None
    utilization_percent: Optional[float] = None
    power_watts: Optional[float] = None
    memory_used_mb: Optional[float] = None
    memory_total_mb: Optional[float] = None
    bus_id: Optional[str] = None
    uuid: Optional[str] = None
    source: str = "unknown"
    warnings: List[str] = field(default_factory=list)
    cooling_health: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FanDecision:
    index: int
    name: str
    temperature_c: Optional[float]
    current_fan_percent: Optional[float]
    target_fan_percent: Optional[int]
    reason: str
    apply_requested: bool
    applied: bool = False
    backend: str = "none"
    command: Optional[List[str]] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GpuFanControlError(RuntimeError):
    """Raised when fan control cannot be applied safely."""


def _find_binary(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    for directory in ("/usr/bin", "/usr/local/bin", "/usr/sbin", "/bin", "/snap/bin"):
        candidate = Path(directory) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        text = str(value).strip().replace("%", "").replace("W", "").replace("MiB", "")
        if not text or text.lower() in {"n/a", "nan", "none", "[not supported]"}:
            return None
        return float(text)
    except Exception:
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def interpolate_curve(temp_c: float, curve: Iterable[Dict[str, Any]]) -> int:
    """Return an interpolated fan percent for a temperature."""
    points = sorted(
        [
            (float(item.get("temp_c", 0)), float(item.get("fan_percent", 0)))
            for item in curve
            if item.get("temp_c") is not None and item.get("fan_percent") is not None
        ],
        key=lambda pair: pair[0],
    )
    if not points:
        points = [(item["temp_c"], item["fan_percent"]) for item in DEFAULT_CURVE]
    if temp_c <= points[0][0]:
        return int(round(points[0][1]))
    if temp_c >= points[-1][0]:
        return int(round(points[-1][1]))
    for (t0, f0), (t1, f1) in zip(points, points[1:]):
        if t0 <= temp_c <= t1:
            ratio = (temp_c - t0) / max(0.001, t1 - t0)
            return int(round(f0 + ((f1 - f0) * ratio)))
    return int(round(points[-1][1]))


def default_config() -> Dict[str, Any]:
    return {
        "enabled": True,
        "apply_enabled_default": False,
        "poll_interval_s": 5,
        "telemetry_preference": ["nvml", "nvidia-smi", "sysfs"],
        "control_preference": ["nvidia-settings", "nvml"],
        "display": os.environ.get("DISPLAY", ":0"),
        "xauthority": os.environ.get("XAUTHORITY"),
        "fan_index_map": {"0": 0, "1": 1, "2": 2},
        "deadband_percent": 3,
        "minimum_change_interval_s": 8,
        "min_fan_percent": 30,
        "max_fan_percent": 100,
        "critical_temp_c": 84,
        "critical_fan_percent": 100,
        "zero_rpm_idle_temp_c": 45,
        "elevated_temp_c": 55,
        "gpu_temperature_warning_threshold_c": 72,
        "gpu_temperature_critical_threshold_c": 84,
        "fan_anomaly_warnings_enabled": True,
        "dashboard_refresh_interval_s": 5,
        "stale_reading_timeout_s": 20,
        "curve": list(DEFAULT_CURVE),
        "expected_gpus": list(DEFAULT_EXPECTED_GPUS),
        "gpu_overrides": {
            "4060": {
                "min_fan_percent": 40,
                "spinup_if_fan_zero_above_c": 42,
                "spinup_percent": 60,
            }
        },
    }


def normalize_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    root = config or {}
    edison_cfg = root.get("edison", root)
    raw = dict(edison_cfg.get("gpu_fan_control", {}))
    merged = default_config()
    for key, value in raw.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


class GpuFanController:
    """Temperature-driven fan controller with dry-run diagnostics by default."""

    def __init__(self, repo_root: Path, config: Optional[Dict[str, Any]] = None) -> None:
        self.repo_root = repo_root.resolve()
        self.config = normalize_config(config)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._last_decisions: List[FanDecision] = []
        self._last_readings: List[GpuFanReading] = []
        self._last_apply_at: Dict[int, float] = {}
        self._last_target: Dict[int, int] = {}
        self._loop_apply = bool(self.config.get("apply_enabled_default", False))

    def diagnostics(self) -> Dict[str, Any]:
        dev_nodes = sorted(str(path) for path in Path("/dev").glob("nvidia*"))
        proc_driver = Path("/proc/driver/nvidia").exists()
        nvidia_smi = _find_binary("nvidia-smi")
        nvidia_settings = _find_binary("nvidia-settings")
        display = self.config.get("display") or os.environ.get("DISPLAY")
        xauthority = self.config.get("xauthority") or os.environ.get("XAUTHORITY")
        try_nvml = self._probe_nvml()
        checks = {
            "nvidia_smi_path": nvidia_smi,
            "nvidia_settings_path": nvidia_settings,
            "dev_nodes": dev_nodes,
            "proc_driver_present": proc_driver,
            "display": display,
            "xauthority": xauthority,
            "nvml": try_nvml,
            "container_or_host_has_gpu_access": bool(dev_nodes or proc_driver or try_nvml.get("ok")),
            "fan_control_ready": bool(nvidia_settings and display) or bool(try_nvml.get("fan_set_supported_hint")),
        }
        recommendations: List[str] = []
        if not nvidia_smi and not try_nvml.get("ok"):
            recommendations.append(
                "nvidia-smi/NVML are unavailable. On the host, install the NVIDIA driver and nvidia-utils; in Docker/devcontainer run with NVIDIA Container Toolkit and --gpus all."
            )
        if not dev_nodes and not proc_driver:
            recommendations.append(
                "No /dev/nvidia* devices or /proc/driver/nvidia found here. Fan control must run on the host or a privileged GPU-enabled container."
            )
        if not nvidia_settings:
            recommendations.append(
                "Install nvidia-settings on the host for fan writes, or provide NVML fan-set support."
            )
        if not display:
            recommendations.append(
                "nvidia-settings fan control requires an X display such as DISPLAY=:0 plus working XAUTHORITY."
            )
        recommendations.append(
            "Enable NVIDIA Coolbits fan control on the host Xorg config, then restart the display manager before running the apply service."
        )
        recommendations.append(
            "A 4060 Ti fan at 0% can be normal zero-RPM behavior at low temperature; this controller forces a spin-up target when it is above the configured threshold."
        )
        checks["recommendations"] = recommendations
        return checks

    def cooling_policy(self) -> CoolingPolicy:
        return CoolingPolicy.from_config(self.config)

    def read_gpus(self) -> Tuple[List[GpuFanReading], List[str]]:
        errors: List[str] = []
        for source in self.config.get("telemetry_preference", ["nvml", "nvidia-smi", "sysfs"]):
            try:
                if source == "nvml":
                    readings = self._read_nvml()
                elif source == "nvidia-smi":
                    readings = self._read_nvidia_smi()
                elif source == "sysfs":
                    readings = self._read_sysfs()
                else:
                    continue
                if readings:
                    self._annotate_cooling_health(readings)
                    with self._lock:
                        self._last_readings = readings
                    return readings, errors
                errors.append(f"{source}: no GPU readings returned")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{source}: {exc}")
        with self._lock:
            return list(self._last_readings), errors

    def _annotate_cooling_health(self, readings: List[GpuFanReading]) -> None:
        diagnostics = self.diagnostics()
        manual_ready = bool(diagnostics.get("fan_control_ready"))
        policy = self.cooling_policy()
        for reading in readings:
            health = classify_cooling_health(
                temperature_c=reading.temperature_c,
                utilization_percent=reading.utilization_percent,
                fan_speed_percent=reading.fan_speed_percent,
                fan_rpm=reading.fan_rpm,
                telemetry_available=True,
                manual_control_available=manual_ready,
                policy=policy,
            )
            reading.cooling_health = health.to_dict()
            if bool(self.config.get("fan_anomaly_warnings_enabled", True)) and health.severity in {"warning", "critical"}:
                reading.warnings.append(health.summary)

    def evaluate_once(self, apply: Optional[bool] = None) -> Dict[str, Any]:
        readings, telemetry_errors = self.read_gpus()
        apply_requested = self._should_apply(apply)
        decisions = [self.decide(reading, apply_requested) for reading in readings]
        applied: List[FanDecision] = []
        for decision in decisions:
            if decision.target_fan_percent is not None and apply_requested:
                applied.append(self._apply_with_safety(decision))
            else:
                decision.applied = False
                decision.backend = "dry_run" if not apply_requested else "none"
                if not apply_requested:
                    decision.warnings.append("Fan write skipped because apply mode is disabled. Use API start/apply-once with apply=true or CLI --apply on the host.")
                applied.append(decision)
        with self._lock:
            self._last_decisions = applied
        return {
            "ok": True,
            "enabled": bool(self.config.get("enabled", True)),
            "apply_requested": apply_requested,
            "running": self.is_running(),
            "readings": [item.to_dict() for item in readings],
            "decisions": [item.to_dict() for item in applied],
            "cooling": aggregate_cooling_health([item.cooling_health for item in readings if item.cooling_health]),
            "telemetry_errors": telemetry_errors,
            "diagnostics": self.diagnostics(),
            "timestamp": time.time(),
        }

    def decide(self, reading: GpuFanReading, apply_requested: bool = False) -> FanDecision:
        warnings = list(reading.warnings)
        if reading.temperature_c is None:
            return FanDecision(
                index=reading.index,
                name=reading.name,
                temperature_c=None,
                current_fan_percent=reading.fan_speed_percent,
                target_fan_percent=None,
                reason="No temperature reading available; refusing to set fan blindly.",
                apply_requested=apply_requested,
                warnings=warnings,
            )
        cfg = self._policy_for_gpu(reading)
        min_fan = float(cfg.get("min_fan_percent", self.config.get("min_fan_percent", 30)))
        max_fan = float(cfg.get("max_fan_percent", self.config.get("max_fan_percent", 100)))
        target = interpolate_curve(float(reading.temperature_c), cfg.get("curve", self.config.get("curve", DEFAULT_CURVE)))
        reason = "temperature curve"
        if float(reading.temperature_c) >= float(cfg.get("critical_temp_c", self.config.get("critical_temp_c", 84))):
            target = int(cfg.get("critical_fan_percent", self.config.get("critical_fan_percent", 100)))
            reason = "critical temperature fail-safe"
        zero_spin_temp = cfg.get("spinup_if_fan_zero_above_c")
        if zero_spin_temp is not None and float(reading.temperature_c) >= float(zero_spin_temp):
            if reading.fan_speed_percent is None or float(reading.fan_speed_percent) <= 1.0:
                target = max(target, int(cfg.get("spinup_percent", 60)))
                reason = "zero-RPM/high-temp spin-up guard"
                warnings.append("Fan reports 0% above spin-up threshold; forcing a safe spin-up target.")
        target = int(round(_clamp(target, min_fan, max_fan)))
        last_target = self._last_target.get(reading.index)
        deadband = int(self.config.get("deadband_percent", 3))
        if last_target is not None and abs(target - last_target) < deadband:
            target = last_target
            reason += "; held by deadband"
        return FanDecision(
            index=reading.index,
            name=reading.name,
            temperature_c=reading.temperature_c,
            current_fan_percent=reading.fan_speed_percent,
            target_fan_percent=target,
            reason=reason,
            apply_requested=apply_requested,
            warnings=warnings,
        )

    def start(self, apply: Optional[bool] = None, poll_interval_s: Optional[float] = None) -> Dict[str, Any]:
        with self._lock:
            self._loop_apply = self._should_apply(apply)
            if poll_interval_s is not None:
                self.config["poll_interval_s"] = float(poll_interval_s)
            if self._thread and self._thread.is_alive():
                return {"ok": True, "running": True, "apply": self._loop_apply, "message": "GPU fan controller already running"}
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True, name="edison-gpu-fan-controller")
            self._thread.start()
            return {"ok": True, "running": True, "apply": self._loop_apply, "poll_interval_s": self.config.get("poll_interval_s")}

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=3.0)
        return {"ok": True, "running": self.is_running()}

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self) -> Dict[str, Any]:
        with self._lock:
            readings = [item.to_dict() for item in self._last_readings]
            decisions = [item.to_dict() for item in self._last_decisions]
        return {
            "ok": True,
            "running": self.is_running(),
            "apply": self._loop_apply,
            "config": self.public_config(),
            "readings": readings,
            "decisions": decisions,
            "cooling": aggregate_cooling_health([item.get("cooling_health", {}) for item in readings if item.get("cooling_health")]),
            "diagnostics": self.diagnostics(),
        }

    def public_config(self) -> Dict[str, Any]:
        public = dict(self.config)
        return public

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.evaluate_once(apply=self._loop_apply)
            except Exception as exc:  # noqa: BLE001
                logger.exception("GPU fan controller loop failed: %s", exc)
            self._stop.wait(float(self.config.get("poll_interval_s", 5)))

    def _should_apply(self, apply: Optional[bool]) -> bool:
        if not bool(self.config.get("enabled", True)):
            return False
        if apply is None:
            return bool(self.config.get("apply_enabled_default", False))
        return bool(apply)

    def _policy_for_gpu(self, reading: GpuFanReading) -> Dict[str, Any]:
        policy = dict(self.config)
        name_l = (reading.name or "").lower()
        for needle, override in (self.config.get("gpu_overrides") or {}).items():
            if str(needle).lower() in name_l:
                policy.update(override or {})
        return policy

    def _apply_with_safety(self, decision: FanDecision) -> FanDecision:
        now = time.time()
        last_apply = self._last_apply_at.get(decision.index, 0.0)
        if now - last_apply < float(self.config.get("minimum_change_interval_s", 8)):
            decision.applied = False
            decision.backend = "rate_limited"
            decision.warnings.append("Skipped fan write because minimum_change_interval_s has not elapsed.")
            return decision
        if decision.target_fan_percent is None:
            return decision
        for backend in self.config.get("control_preference", ["nvidia-settings", "nvml"]):
            try:
                if backend == "nvidia-settings":
                    command = self._apply_nvidia_settings(decision.index, decision.target_fan_percent)
                    decision.applied = True
                    decision.backend = backend
                    decision.command = command
                    self._last_apply_at[decision.index] = now
                    self._last_target[decision.index] = decision.target_fan_percent
                    return decision
                if backend == "nvml":
                    self._apply_nvml(decision.index, decision.target_fan_percent)
                    decision.applied = True
                    decision.backend = backend
                    self._last_apply_at[decision.index] = now
                    self._last_target[decision.index] = decision.target_fan_percent
                    return decision
            except Exception as exc:  # noqa: BLE001
                decision.error = str(exc)
                decision.warnings.append(f"{backend} fan write failed: {exc}")
        decision.applied = False
        decision.backend = "unavailable"
        if not decision.error:
            decision.error = "No configured fan-control backend was available."
        return decision

    def _apply_nvidia_settings(self, gpu_index: int, fan_percent: int) -> List[str]:
        binary = _find_binary("nvidia-settings")
        if not binary:
            raise GpuFanControlError("nvidia-settings not found")
        display = self.config.get("display") or os.environ.get("DISPLAY")
        if not display:
            raise GpuFanControlError("DISPLAY is not set; nvidia-settings fan writes require Xorg")
        fan_map = self.config.get("fan_index_map") or {}
        fan_index = _safe_int(fan_map.get(str(gpu_index), fan_map.get(gpu_index, gpu_index)), gpu_index)
        command = [
            binary,
            "-c",
            str(display),
            "-a",
            f"[gpu:{gpu_index}]/GPUFanControlState=1",
            "-a",
            f"[fan:{fan_index}]/GPUTargetFanSpeed={int(fan_percent)}",
        ]
        env = os.environ.copy()
        env["DISPLAY"] = str(display)
        xauthority = self.config.get("xauthority") or os.environ.get("XAUTHORITY")
        if xauthority:
            env["XAUTHORITY"] = str(xauthority)
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, env=env)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "nvidia-settings failed").strip()
            raise GpuFanControlError(detail[:800])
        return command

    def _apply_nvml(self, gpu_index: int, fan_percent: int) -> None:
        try:
            import pynvml  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise GpuFanControlError("pynvml/nvidia-ml-py is not installed") from exc
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
            if hasattr(pynvml, "nvmlDeviceSetFanSpeed_v2"):
                pynvml.nvmlDeviceSetFanSpeed_v2(handle, 0, int(fan_percent))
            elif hasattr(pynvml, "nvmlDeviceSetFanSpeed"):
                pynvml.nvmlDeviceSetFanSpeed(handle, int(fan_percent))
            else:
                raise GpuFanControlError("NVML fan write function is not exposed by this pynvml build")
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _read_nvidia_smi(self) -> List[GpuFanReading]:
        binary = _find_binary("nvidia-smi")
        if not binary:
            raise GpuFanControlError("nvidia-smi not found")
        result = subprocess.run(
            [
                binary,
                "--query-gpu=index,name,pci.bus_id,uuid,temperature.gpu,fan.speed,utilization.gpu,power.draw,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise GpuFanControlError((result.stderr or result.stdout or "nvidia-smi failed").strip()[:800])
        readings: List[GpuFanReading] = []
        for line in result.stdout.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 8:
                continue
            readings.append(
                GpuFanReading(
                    index=_safe_int(parts[0]),
                    name=parts[1],
                    bus_id=parts[2],
                    uuid=parts[3],
                    temperature_c=_safe_float(parts[4]),
                    fan_speed_percent=_safe_float(parts[5]),
                    utilization_percent=_safe_float(parts[6]) if len(parts) > 6 else None,
                    power_watts=_safe_float(parts[7]) if len(parts) > 7 else None,
                    memory_used_mb=_safe_float(parts[8]) if len(parts) > 8 else None,
                    memory_total_mb=_safe_float(parts[9]) if len(parts) > 9 else None,
                    source="nvidia-smi",
                )
            )
        return readings

    def _read_nvml(self) -> List[GpuFanReading]:
        try:
            import pynvml  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise GpuFanControlError("pynvml/nvidia-ml-py is not installed") from exc
        readings: List[GpuFanReading] = []
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            for index in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                raw_name = pynvml.nvmlDeviceGetName(handle)
                name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
                temp = None
                fan = None
                util = None
                power = None
                mem_used = None
                mem_total = None
                fan_rpm = None
                bus_id = None
                uuid = None
                warnings: List[str] = []
                try:
                    temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"temperature unavailable: {exc}")
                try:
                    fan = float(pynvml.nvmlDeviceGetFanSpeed(handle))
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"fan speed unavailable: {exc}")
                try:
                    if hasattr(pynvml, "nvmlDeviceGetFanSpeedRPM"):
                        fan_rpm = float(pynvml.nvmlDeviceGetFanSpeedRPM(handle))
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"fan RPM unavailable: {exc}")
                try:
                    u = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    util = float(u.gpu)
                except Exception:
                    pass
                try:
                    power = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
                except Exception:
                    pass
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used = round(float(mem.used) / (1024**2), 1)
                    mem_total = round(float(mem.total) / (1024**2), 1)
                except Exception:
                    pass
                try:
                    pci = pynvml.nvmlDeviceGetPciInfo(handle)
                    raw_bus = getattr(pci, "busId", None)
                    bus_id = raw_bus.decode("utf-8") if isinstance(raw_bus, bytes) else str(raw_bus) if raw_bus else None
                except Exception:
                    pass
                try:
                    raw_uuid = pynvml.nvmlDeviceGetUUID(handle)
                    uuid = raw_uuid.decode("utf-8") if isinstance(raw_uuid, bytes) else str(raw_uuid)
                except Exception:
                    pass
                readings.append(
                    GpuFanReading(
                        index=index,
                        name=name,
                        temperature_c=temp,
                        fan_speed_percent=fan,
                        fan_rpm=fan_rpm,
                        utilization_percent=util,
                        power_watts=power,
                        memory_used_mb=mem_used,
                        memory_total_mb=mem_total,
                        bus_id=bus_id,
                        uuid=uuid,
                        source="nvml",
                        warnings=warnings,
                    )
                )
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        return readings

    def _read_sysfs(self) -> List[GpuFanReading]:
        readings: List[GpuFanReading] = []
        candidates = sorted(Path("/sys/class/drm").glob("card*/device/hwmon/hwmon*"))
        for fallback_index, hwmon in enumerate(candidates):
            temp_file = hwmon / "temp1_input"
            if not temp_file.exists():
                continue
            try:
                temp = float(temp_file.read_text().strip()) / 1000.0
            except Exception:
                continue
            name_file = hwmon / "name"
            name = name_file.read_text().strip() if name_file.exists() else f"GPU {fallback_index}"
            fan = None
            pwm_file = hwmon / "pwm1"
            if pwm_file.exists():
                try:
                    fan = round(float(pwm_file.read_text().strip()) / 255.0 * 100.0, 1)
                except Exception:
                    fan = None
            readings.append(
                GpuFanReading(
                    index=fallback_index,
                    name=name,
                    temperature_c=temp,
                    fan_speed_percent=fan,
                    source="sysfs",
                    warnings=["sysfs fallback may not preserve NVIDIA GPU ordering"],
                )
            )
        return readings

    def _probe_nvml(self) -> Dict[str, Any]:
        try:
            import pynvml  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"pynvml import failed: {exc}", "fan_set_supported_hint": False}
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            return {
                "ok": True,
                "device_count": count,
                "fan_set_supported_hint": bool(
                    hasattr(pynvml, "nvmlDeviceSetFanSpeed_v2") or hasattr(pynvml, "nvmlDeviceSetFanSpeed")
                ),
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc), "fan_set_supported_hint": False}
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        import yaml  # type: ignore
        return yaml.safe_load(config_path.read_text()) or {}
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to read {config_path}: {exc}") from exc


def build_controller(repo_root: Optional[Path] = None, config_path: Optional[Path] = None) -> GpuFanController:
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    path = config_path or (root / "config" / "edison.yaml")
    return GpuFanController(root, load_yaml_config(path))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="EDISON safe GPU fan controller")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--config", default=None)
    parser.add_argument("--apply", action="store_true", help="Actually write fan speeds. Without this, runs as dry-run diagnostics.")
    parser.add_argument("--once", action="store_true", help="Run one evaluation and exit.")
    parser.add_argument("--loop", action="store_true", help="Run continuously until interrupted.")
    parser.add_argument("--interval", type=float, default=None, help="Override polling interval seconds.")
    parser.add_argument("--diagnose", action="store_true", help="Print tool/device diagnostics and exit.")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve() if args.config else repo_root / "config" / "edison.yaml"
    controller = build_controller(repo_root, config_path)
    if args.interval is not None:
        controller.config["poll_interval_s"] = args.interval

    if args.diagnose:
        print(json.dumps(controller.diagnostics(), indent=2))
        return 0

    if args.once or not args.loop:
        print(json.dumps(controller.evaluate_once(apply=args.apply), indent=2))
        return 0

    logger.info("Starting EDISON GPU fan controller loop apply=%s interval=%s", args.apply, controller.config.get("poll_interval_s"))
    try:
        while True:
            print(json.dumps(controller.evaluate_once(apply=args.apply), indent=2), flush=True)
            time.sleep(float(controller.config.get("poll_interval_s", 5)))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
