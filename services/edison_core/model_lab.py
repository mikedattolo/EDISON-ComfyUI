"""Hardware-aware local model recommendations for Edison Model Lab.

The lab intentionally supports advanced/custom models without creating a
"disable every safety check" switch. Edison still keeps its filesystem,
node-dispatch, GPU, and confirmation guardrails around model output.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ModelLabProfile:
    id: str
    name: str
    provider: str
    repo: str
    role: str
    target_slot: str
    source_url: str
    install_glob: str
    recommended_quant: str
    estimated_size_gb: float
    min_system_ram_gb: int
    preferred_vram_gb: int
    command_note: str
    notes: List[str] = field(default_factory=list)
    safety_note: str = "Runs locally, but Edison route/file/tool guardrails remain active."


MODEL_LAB_PROFILES: List[ModelLabProfile] = [
    ModelLabProfile(
        id="qwen3-30b-a3b-instruct",
        name="Qwen3 30B-A3B Instruct",
        provider="Hugging Face",
        repo="unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
        role="general_agent",
        target_slot="deep",
        source_url="https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
        install_glob="*UD-Q4_K_XL*.gguf",
        recommended_quant="UD-Q4_K_XL",
        estimated_size_gb=18.0,
        min_system_ram_gb=64,
        preferred_vram_gb=16,
        command_note="Good default upgrade candidate: MoE-style speed, strong general/tool following, and long-context headroom.",
        notes=[
            "Use as a deep/general model candidate.",
            "128 GB RAM makes larger context/KV cache spillover less painful.",
        ],
    ),
    ModelLabProfile(
        id="qwen2-5-coder-32b",
        name="Qwen2.5 Coder 32B Instruct",
        provider="Hugging Face",
        repo="Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        role="code",
        target_slot="reasoning",
        source_url="https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        install_glob="*q5_k_m*.gguf",
        recommended_quant="Q5_K_M",
        estimated_size_gb=23.3,
        min_system_ram_gb=96,
        preferred_vram_gb=24,
        command_note="Best fit for Edison code mode: code generation, debugging, refactors, and repo Q&A.",
        notes=[
            "Use on the RTX 3090 when possible.",
            "Q4_K_M is the lower-VRAM fallback; Q5_K_M benefits from your new RAM.",
        ],
    ),
    ModelLabProfile(
        id="deepseek-r1-distill-qwen-32b",
        name="DeepSeek R1 Distill Qwen 32B",
        provider="Hugging Face",
        repo="unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        role="reasoning",
        target_slot="reasoning",
        source_url="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        install_glob="*Q4_K_M*.gguf",
        recommended_quant="Q4_K_M",
        estimated_size_gb=19.0,
        min_system_ram_gb=64,
        preferred_vram_gb=24,
        command_note="Good separate reasoning model for deliberate problem solving and math/code planning.",
        notes=[
            "Keep temperature conservative for deterministic work.",
            "Expect longer answers; pair with response guard already in Edison.",
        ],
    ),
    ModelLabProfile(
        id="bge-reranker-gguf",
        name="BGE Reranker GGUF",
        provider="Hugging Face",
        repo="cstr/bge-reranker-base-GGUF",
        role="rag_reranker",
        target_slot="rag",
        source_url="https://huggingface.co/cstr/bge-reranker-base-GGUF",
        install_glob="bge-reranker-base-q4_k.gguf",
        recommended_quant="Q4_K",
        estimated_size_gb=0.3,
        min_system_ram_gb=16,
        preferred_vram_gb=4,
        command_note="Small but high-value: improves memory/RAG result ordering before answers are generated.",
        notes=[
            "This is not a chat model.",
            "Best used as a retrieval quality upgrade.",
        ],
    ),
]


GITHUB_TOOL_RECOMMENDATIONS = [
    {
        "id": "comfyui-manager",
        "name": "ComfyUI Manager",
        "source_url": "https://github.com/Comfy-Org/ComfyUI-Manager",
        "category": "comfyui",
        "why": "Manage custom nodes, missing nodes, and ComfyUI extension health from the ComfyUI side.",
        "caution": "Install custom nodes deliberately; they run Python code inside the ComfyUI environment.",
    },
    {
        "id": "browser-use",
        "name": "browser-use / Playwright browser agent tooling",
        "source_url": "https://github.com/browser-use/browser-use",
        "category": "agent",
        "why": "Useful foundation for a stronger visible browser/Agent Mode workflow.",
        "caution": "Keep state-changing actions behind Edison confirmations.",
    },
    {
        "id": "mcp-servers",
        "name": "Model Context Protocol server catalog",
        "source_url": "https://github.com/modelcontextprotocol/servers",
        "category": "tools",
        "why": "Filesystem, Git, GitHub, database, and browser tool bridges can extend Edison cleanly.",
        "caution": "Use strict allowlists and sandboxing; do not expose broad shell/filesystem access by default.",
    },
]


def detect_system_ram_gb() -> Optional[float]:
    """Return total system RAM in GB when available."""
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            text = meminfo.read_text(encoding="utf-8", errors="ignore")
            match = re.search(r"^MemTotal:\s+(\d+)\s+kB", text, flags=re.MULTILINE)
            if match:
                return round(int(match.group(1)) / (1024 * 1024), 1)
        except OSError:
            pass
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        return None


def detect_gpus() -> List[Dict[str, Any]]:
    """Read lightweight NVIDIA GPU info without failing when nvidia-smi is absent."""
    if not shutil.which("nvidia-smi"):
        return []
    query = "index,name,memory.total,memory.used,temperature.gpu,fan.speed,power.draw"
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    rows: List[Dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        rows.append(
            {
                "index": _to_int(parts[0]),
                "name": parts[1],
                "vram_total_mb": _to_int(parts[2]),
                "vram_used_mb": _to_int(parts[3]),
                "temperature_c": _to_int(parts[4]),
                "fan_speed_percent": _to_int(parts[5]),
                "power_draw_w": _to_float(parts[6]),
            }
        )
    return rows


def scan_installed_gguf(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        if not path or not path.exists():
            continue
        for file_path in sorted(path.glob("*.gguf")):
            key = str(file_path.resolve())
            if key in seen:
                continue
            seen.add(key)
            models.append(
                {
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size_gb": round(file_path.stat().st_size / (1024**3), 2),
                }
            )
    return models


def build_download_command(profile: ModelLabProfile, models_dir: Path) -> str:
    safe_dir = str(models_dir)
    return (
        f"huggingface-cli download {profile.repo} "
        f"--include \"{profile.install_glob}\" "
        f"--local-dir \"{safe_dir}\" "
        "--local-dir-use-symlinks False"
    )


def summarize_model_lab(repo_root: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    core_cfg = (config.get("edison", {}) or {}).get("core", {}) or {}
    models_rel = core_cfg.get("models_path", "models/llm")
    models_dir = (repo_root / models_rel).resolve()
    large_raw = (config.get("llm", {}) or {}).get("large_model_path")
    large_dir = Path(large_raw).expanduser().resolve() if large_raw else None
    scan_paths = [models_dir] + ([large_dir] if large_dir else [])

    ram_gb = detect_system_ram_gb()
    gpus = detect_gpus()
    largest_vram_gb = max((g.get("vram_total_mb", 0) or 0 for g in gpus), default=0) / 1024
    installed = scan_installed_gguf([p for p in scan_paths if p])
    installed_names = {m["filename"].lower() for m in installed}

    recommendations = []
    for profile in MODEL_LAB_PROFILES:
        ram_ok = ram_gb is None or ram_gb >= profile.min_system_ram_gb
        vram_ok = not gpus or largest_vram_gb >= profile.preferred_vram_gb
        installed_match = any(_profile_matches_installed(profile, name) for name in installed_names)
        fit = "ready" if installed_match else ("recommended" if ram_ok and vram_ok else "possible_with_tradeoffs")
        recommendations.append(
            {
                **profile.__dict__,
                "fit": fit,
                "installed": installed_match,
                "download_command": build_download_command(profile, models_dir),
                "hardware_notes": _hardware_notes(profile, ram_gb, largest_vram_gb),
            }
        )

    return {
        "ok": True,
        "policy": {
            "advanced_model_support": True,
            "no_safeguard_bypass": False,
            "message": (
                "Model Lab can run advanced local models, but Edison keeps its "
                "path, node-dispatch, tool, and confirmation guardrails active."
            ),
        },
        "hardware": {
            "system_ram_gb": ram_gb,
            "gpus": gpus,
            "largest_vram_gb": round(largest_vram_gb, 1),
            "ram_upgrade_detected": bool(ram_gb and ram_gb >= 96),
        },
        "paths": {
            "models_dir": str(models_dir),
            "large_models_dir": str(large_dir) if large_dir else None,
        },
        "installed_models": installed,
        "recommendations": recommendations,
        "github_tools": GITHUB_TOOL_RECOMMENDATIONS,
        "next_moves": _next_moves(ram_gb, gpus),
    }


def install_plan(profile_id: str, repo_root: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    profile = next((p for p in MODEL_LAB_PROFILES if p.id == profile_id), None)
    if profile is None:
        raise KeyError(profile_id)
    core_cfg = (config.get("edison", {}) or {}).get("core", {}) or {}
    models_dir = (repo_root / core_cfg.get("models_path", "models/llm")).resolve()
    return {
        "profile": profile.__dict__,
        "commands": [
            f"mkdir -p \"{models_dir}\"",
            build_download_command(profile, models_dir),
        ],
        "config_hint": {
            "slot": profile.target_slot,
            "set_in_config": f"edison.core.{profile.target_slot}_model",
            "filename_pattern": profile.install_glob,
        },
        "restart_hint": "Restart edison-core after changing the active configured model slot.",
    }


def _profile_matches_installed(profile: ModelLabProfile, installed_name: str) -> bool:
    repo_tail = profile.repo.split("/")[-1].replace("-GGUF", "").lower()
    tokens = [t for t in re.split(r"[-_]+", repo_tail) if t and t not in {"gguf", "instruct"}]
    return all(token in installed_name for token in tokens[:3])


def _hardware_notes(profile: ModelLabProfile, ram_gb: Optional[float], largest_vram_gb: float) -> List[str]:
    notes = []
    if ram_gb is not None and ram_gb < profile.min_system_ram_gb:
        notes.append(f"Recommended system RAM is {profile.min_system_ram_gb} GB; detected {ram_gb} GB.")
    if largest_vram_gb and largest_vram_gb < profile.preferred_vram_gb:
        notes.append(f"Preferred single-GPU VRAM is {profile.preferred_vram_gb} GB; largest detected is {largest_vram_gb:.1f} GB.")
    if not notes:
        notes.append("Hardware fit looks reasonable for this profile.")
    return notes


def _next_moves(ram_gb: Optional[float], gpus: List[Dict[str, Any]]) -> List[str]:
    moves = [
        "Use the 3090 as the primary large-model GPU and keep the 16 GB cards for ComfyUI workers or smaller parallel jobs.",
        "Raise context/KV-cache budgets carefully; 128 GB RAM helps, but VRAM is still per-card and not pooled.",
        "Add a reranker before expanding RAG corpus size; retrieval quality usually beats raw context stuffing.",
    ]
    if ram_gb and ram_gb >= 96:
        moves.append("Consider Q5_K_M/Q6_K quantizations for key 32B models when latency is acceptable.")
    if len(gpus) >= 3:
        moves.append("Keep per-GPU ComfyUI workers enabled; avoid pretending the GPUs are one 56 GB card.")
    return moves


def _to_int(value: str) -> Optional[int]:
    try:
        return int(float(str(value).replace("[Not Supported]", "").strip()))
    except Exception:
        return None


def _to_float(value: str) -> Optional[float]:
    try:
        return float(str(value).replace("[Not Supported]", "").strip())
    except Exception:
        return None
