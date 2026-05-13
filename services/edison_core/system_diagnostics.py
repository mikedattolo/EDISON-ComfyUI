"""System diagnostics and doctor report helpers for EDISON."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from .comfyui_integration import discover_workflow_templates, summarize_template_library
from .cooling_diagnostics import aggregate_cooling_health, classify_cooling_health
from .gpu_fan_control import GpuFanController, load_yaml_config


@dataclass
class DiagnosticCheck:
    key: str
    title: str
    status: str
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommended_fix: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _run(cmd: List[str], timeout_s: float = 10.0) -> Dict[str, Any]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": (result.stdout or "").strip()[:4000],
            "stderr": (result.stderr or "").strip()[:4000],
            "cmd": cmd,
        }
    except FileNotFoundError:
        return {"ok": False, "error": f"{cmd[0]} not found", "cmd": cmd}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "cmd": cmd}


def _pkg_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def _find_binary(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    for directory in ("/usr/bin", "/usr/local/bin", "/usr/sbin", "/bin", "/snap/bin"):
        candidate = Path(directory) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _torch_arch_supported(arch: str, supported_arches: Iterable[str]) -> Optional[bool]:
    supported = {str(item).lower() for item in supported_arches if item}
    if not arch or not supported:
        return None
    arch = arch.lower()
    if arch in supported:
        return True
    # PyTorch may report PTX support such as compute_90. Treat exact compute_NN
    # support as usable when a device reports sm_NN.
    if arch.startswith("sm_"):
        return f"compute_{arch[3:]}" in supported
    return False


def collect_nvidia_smi() -> DiagnosticCheck:
    binary = _find_binary("nvidia-smi")
    if not binary:
        return DiagnosticCheck(
            key="nvidia_smi",
            title="NVIDIA driver CLI",
            status="fail",
            summary="nvidia-smi is not available in this environment.",
            details={"path": None},
            recommended_fix="Install/repair NVIDIA driver utilities on the host or run Edison with GPU device access.",
        )
    query = _run(
        [
            binary,
            "--query-gpu=index,name,uuid,pci.bus_id,driver_version,temperature.gpu,utilization.gpu,memory.total,memory.used,memory.free,power.draw,fan.speed",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=8,
    )
    if not query.get("ok"):
        return DiagnosticCheck(
            key="nvidia_smi",
            title="NVIDIA driver CLI",
            status="fail",
            summary="nvidia-smi is installed but failed to query GPUs.",
            details={"path": binary, "query": query},
            recommended_fix="Run nvidia-smi on the Edison host, check driver/kernel module state, and reboot after driver repair if needed.",
        )
    gpus: List[Dict[str, Any]] = []
    for line in (query.get("stdout") or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 10:
            continue
        gpus.append(
            {
                "index": _maybe_int(parts[0]),
                "name": parts[1],
                "uuid": parts[2],
                "bus_id": parts[3],
                "driver_version": parts[4],
                "temperature_c": _maybe_float(parts[5]),
                "utilization_percent": _maybe_float(parts[6]),
                "memory_total_mb": _maybe_float(parts[7]),
                "memory_used_mb": _maybe_float(parts[8]),
                "memory_free_mb": _maybe_float(parts[9]),
                "power_watts": _maybe_float(parts[10]) if len(parts) > 10 else None,
                "fan_speed_percent": _maybe_float(parts[11]) if len(parts) > 11 else None,
            }
        )
    expected = {"3090", "5060", "4060"}
    names = " ".join(gpu.get("name", "") for gpu in gpus).lower()
    missing = [needle for needle in expected if needle not in names]
    status = "pass" if gpus and not missing else "warn" if gpus else "fail"
    return DiagnosticCheck(
        key="nvidia_smi",
        title="NVIDIA driver CLI",
        status=status,
        summary=f"Detected {len(gpus)} NVIDIA GPU(s)." if gpus else "No GPUs returned by nvidia-smi.",
        details={"path": binary, "gpus": gpus, "missing_expected_cards": missing, "raw": query},
        recommended_fix=(
            "Confirm all three expected cards are visible by bus ID and reseat/check power for any missing card."
            if missing
            else ""
        ),
    )


def collect_cuda_toolkit() -> DiagnosticCheck:
    nvcc = _find_binary("nvcc")
    symlink = Path("/usr/local/cuda")
    details = {
        "nvcc_path": nvcc,
        "usr_local_cuda_exists": symlink.exists(),
        "usr_local_cuda_target": str(symlink.resolve(strict=False)) if symlink.exists() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if not nvcc:
        return DiagnosticCheck(
            key="cuda_toolkit",
            title="CUDA toolkit",
            status="warn",
            summary="nvcc is not installed or not on PATH. PyTorch can still use CUDA runtime wheels without nvcc.",
            details=details,
            recommended_fix="Install the CUDA toolkit only if you compile CUDA extensions; otherwise keep using matching PyTorch CUDA wheels.",
        )
    details["nvcc_version"] = _run([nvcc, "--version"], timeout_s=5)
    return DiagnosticCheck(
        key="cuda_toolkit",
        title="CUDA toolkit",
        status="pass",
        summary="CUDA toolkit compiler is visible.",
        details=details,
    )


def collect_python_packages() -> DiagnosticCheck:
    packages = {
        "torch": _pkg_version("torch"),
        "torchvision": _pkg_version("torchvision"),
        "torchaudio": _pkg_version("torchaudio"),
        "xformers": _pkg_version("xformers"),
        "triton": _pkg_version("triton"),
        "flash_attn": _pkg_version("flash-attn"),
        "bitsandbytes": _pkg_version("bitsandbytes"),
        "accelerate": _pkg_version("accelerate"),
        "numpy": _pkg_version("numpy"),
        "llama_cpp_python": _pkg_version("llama-cpp-python"),
    }
    missing_optional = [name for name in ("xformers", "flash_attn", "bitsandbytes") if packages.get(name) is None]
    return DiagnosticCheck(
        key="python_packages",
        title="Python AI packages",
        status="pass",
        summary="Python package versions collected.",
        details={"python": sys.executable, "version": sys.version, "packages": packages, "missing_optional": missing_optional},
    )


def collect_torch_cuda(run_allocation: bool = False) -> DiagnosticCheck:
    details: Dict[str, Any] = {"python": sys.executable}
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        return DiagnosticCheck(
            key="torch_cuda",
            title="PyTorch CUDA",
            status="fail",
            summary="torch could not be imported.",
            details={"error": str(exc), **details},
            recommended_fix="Install a PyTorch wheel matching the Edison Python environment and NVIDIA driver.",
        )
    details.update(
        {
            "torch_version": getattr(torch, "__version__", None),
            "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "devices": [],
        }
    )
    warnings: List[str] = []
    try:
        supported_arches = list(torch.cuda.get_arch_list())
    except Exception:
        supported_arches = []
    details["supported_cuda_arches"] = supported_arches
    if not details["cuda_available"]:
        return DiagnosticCheck(
            key="torch_cuda",
            title="PyTorch CUDA",
            status="fail",
            summary="PyTorch reports CUDA is not available.",
            details=details,
            recommended_fix=(
                "Verify this is not a CPU-only torch wheel. Reinstall torch/torchvision/torchaudio with the CUDA wheel family "
                "that matches the driver-supported runtime."
            ),
        )
    for idx in range(details["device_count"]):
        device: Dict[str, Any] = {"index": idx}
        try:
            props = torch.cuda.get_device_properties(idx)
            arch = f"sm_{props.major}{props.minor}"
            arch_supported = _torch_arch_supported(arch, supported_arches)
            device.update(
                {
                    "name": props.name,
                    "total_memory_mb": round(props.total_memory / (1024**2), 1),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "torch_arch": arch,
                    "torch_arch_supported": arch_supported,
                }
            )
        except Exception as exc:
            device["properties_error"] = str(exc)
        if run_allocation:
            try:
                with torch.cuda.device(idx):
                    tensor = torch.ones((8, 8), device=f"cuda:{idx}")
                    device["allocation_test"] = float(tensor.sum().item())
                    del tensor
            except Exception as exc:
                device["allocation_error"] = str(exc)
                warnings.append(f"cuda:{idx} allocation failed: {exc}")
        if device.get("torch_arch_supported") is False:
            if device.get("allocation_test") is not None:
                device["torch_arch_note"] = "Architecture is not listed by torch, but a CUDA allocation test succeeded."
            else:
                warnings.append(
                    f"cuda:{idx} {device.get('name', 'GPU')} reports {device.get('torch_arch')}, "
                    "which is not in this PyTorch build's supported CUDA architectures."
                )
        details["devices"].append(device)
    expected_names = " ".join(str(d.get("name", "")) for d in details["devices"]).lower()
    for expected in ("3090", "5060", "4060"):
        if expected not in expected_names:
            warnings.append(f"Expected RTX {expected} class GPU was not reported by torch.")
    return DiagnosticCheck(
        key="torch_cuda",
        title="PyTorch CUDA",
        status="warn" if warnings else "pass",
        summary=f"PyTorch CUDA is available with {details['device_count']} device(s).",
        details={**details, "warnings": warnings},
        recommended_fix=(
            "Check CUDA_VISIBLE_DEVICES, driver support, and PyTorch wheel family. For newer GPUs, use a PyTorch/CUDA build "
            "that includes the device compute capability, or temporarily mask that GPU from PyTorch services."
            if warnings
            else ""
        ),
    )


def collect_binary_tools() -> DiagnosticCheck:
    tools = {name: _find_binary(name) for name in ("ffmpeg", "ffprobe", "whisper", "nvidia-settings")}
    missing_required = [name for name in ("ffmpeg", "ffprobe") if not tools.get(name)]
    return DiagnosticCheck(
        key="binary_tools",
        title="External media/system tools",
        status="warn" if missing_required else "pass",
        summary="Required media tools are present." if not missing_required else "Some required media tools are missing.",
        details={"tools": tools},
        recommended_fix="Install FFmpeg on Ubuntu with: sudo apt install ffmpeg" if missing_required else "",
    )


def collect_comfyui(repo_root: Path, config: Dict[str, Any]) -> DiagnosticCheck:
    comfy = ((config.get("edison") or {}).get("comfyui") or {})
    host = comfy.get("host", "127.0.0.1")
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    port = int(comfy.get("port", 8188))
    base_url = f"http://{host}:{port}"
    reachable = False
    detail: Dict[str, Any] = {"base_url": base_url}
    try:
        response = requests.get(f"{base_url}/queue", timeout=2)
        detail["queue_status_code"] = response.status_code
        reachable = bool(response.ok)
    except Exception as exc:
        detail["error"] = str(exc)
    workflow_dir = repo_root / "config" / "persona_video" / "comfyui_workflows"
    templates = discover_workflow_templates(workflow_dir)
    detail["workflow_library"] = summarize_template_library(templates)
    status = "pass" if reachable else "warn"
    return DiagnosticCheck(
        key="comfyui",
        title="ComfyUI",
        status=status,
        summary="ComfyUI queue endpoint is reachable." if reachable else "ComfyUI is not reachable from Edison.",
        details=detail,
        recommended_fix="Start ComfyUI and verify edison.comfyui host/port. Install required custom nodes/models for workflow templates."
        if not reachable or not templates
        else "",
    )


def collect_paths(repo_root: Path, config: Dict[str, Any]) -> DiagnosticCheck:
    paths = {
        "repo_root": repo_root,
        "outputs": repo_root / "outputs",
        "uploads": repo_root / "uploads",
        "config": repo_root / "config",
        "persona_workflows": repo_root / "config" / "persona_video" / "comfyui_workflows",
    }
    rows: Dict[str, Any] = {}
    warnings: List[str] = []
    for key, path in paths.items():
        p = Path(path)
        writable = False
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".edison_diag_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            writable = True
        except Exception as exc:
            warnings.append(f"{key} not writable: {exc}")
        rows[key] = {"path": str(p), "exists": p.exists(), "writable": writable}
    return DiagnosticCheck(
        key="paths",
        title="Workspace paths",
        status="warn" if warnings else "pass",
        summary="Workspace paths checked.",
        details={"paths": rows, "warnings": warnings},
        recommended_fix="Fix ownership/permissions or volume mounts for any unwritable directory." if warnings else "",
    )


def collect_cooling(repo_root: Path, config: Dict[str, Any]) -> DiagnosticCheck:
    controller = GpuFanController(repo_root, config)
    status = controller.evaluate_once(apply=False)
    readings = status.get("readings") or []
    health = [reading.get("cooling_health") for reading in readings if reading.get("cooling_health")]
    if not health and not readings:
        health = [
            classify_cooling_health(telemetry_available=False).to_dict()
        ]
    summary = aggregate_cooling_health(health)
    diag_status = "pass"
    if summary.get("overall_severity") in {"warning", "critical"}:
        diag_status = "warn"
    elif summary.get("overall_severity") == "unknown":
        diag_status = "warn"
    return DiagnosticCheck(
        key="cooling",
        title="GPU cooling and fan health",
        status=diag_status,
        summary=f"Cooling severity: {summary.get('overall_severity')}.",
        details={"fan_status": status, "cooling_summary": summary},
        recommended_fix="Inspect any GPU flagged as suspect_fan_not_spinning or over_temp_warning before long renders.",
    )


def build_system_diagnostic_report(
    repo_root: Path,
    config: Optional[Dict[str, Any]] = None,
    *,
    run_cuda_allocation: bool = False,
) -> Dict[str, Any]:
    cfg = config if config is not None else load_yaml_config(repo_root / "config" / "edison.yaml")
    checks = [
        collect_nvidia_smi(),
        collect_cuda_toolkit(),
        collect_python_packages(),
        collect_torch_cuda(run_allocation=run_cuda_allocation),
        collect_binary_tools(),
        collect_comfyui(repo_root, cfg),
        collect_paths(repo_root, cfg),
        collect_cooling(repo_root, cfg),
    ]
    counts: Dict[str, int] = {}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    overall = "pass"
    if counts.get("fail"):
        overall = "fail"
    elif counts.get("warn"):
        overall = "warn"
    return {
        "schema": "edison.system_diagnostics.v1",
        "generated_at": int(time.time()),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.executable,
            "python_version": sys.version,
        },
        "overall": overall,
        "counts": counts,
        "checks": [check.to_dict() for check in checks],
    }


def report_as_text(report: Dict[str, Any]) -> str:
    lines = [
        "EDISON System Diagnostics",
        f"Overall: {report.get('overall')}",
        f"Generated at: {report.get('generated_at')}",
        "",
    ]
    for check in report.get("checks", []):
        lines.append(f"[{str(check.get('status', '')).upper()}] {check.get('title')}: {check.get('summary')}")
        fix = check.get("recommended_fix")
        if fix:
            lines.append(f"  Fix: {fix}")
    return "\n".join(lines).strip() + "\n"


def _maybe_float(value: Any) -> Optional[float]:
    try:
        text = str(value).strip().replace("%", "").replace("W", "")
        if not text or text.lower() in {"n/a", "none", "[not supported]"}:
            return None
        return float(text)
    except Exception:
        return None


def _maybe_int(value: Any) -> Optional[int]:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None
