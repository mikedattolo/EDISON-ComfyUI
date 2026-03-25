"""Unified model and workflow catalog for EDISON media and local AI capabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import os
import re


COMFYUI_MODEL_SUBDIRS = {
    "checkpoints": "checkpoint",
    "loras": "lora",
    "vae": "vae",
    "controlnet": "controlnet",
    "upscale_models": "upscaler",
    "clip_vision": "clip_vision",
    "ipadapter": "ipadapter",
}


KNOWN_3D_MODEL_HINTS = [
    "3d",
    "mesh",
    "triposr",
    "hunyuan3d",
    "stable-fast-3d",
    "sf3d",
    "instantmesh",
    "crm",
    "shap-e",
    "shapee",
    "wonder3d",
]


KNOWN_3D_NODE_HINTS = [
    "triposr",
    "hunyuan3d",
    "stable-fast-3d",
    "sf3d",
    "instantmesh",
    "crm",
    "wonder3d",
    "mesh",
]


SUPPORTED_MESH_WORKFLOWS = [
    {
        "workflow": "text_to_3d",
        "endpoint": "/3d/generate",
        "best_for": ["concept meshes", "signage prototypes", "promo products", "basic product forms"],
        "needs": ["ComfyUI running", "3D custom nodes", "compatible 3D weights"],
    },
    {
        "workflow": "image_to_3d",
        "endpoint": "/generate-3d",
        "best_for": ["logo relief concepts", "reference-based mockups", "product rough-outs"],
        "needs": ["ComfyUI running", "3D custom nodes", "reference image support"],
    },
]


SUPPORTED_IMAGE_WORKFLOWS = [
    {
        "workflow": "text_to_image",
        "endpoint": "/generate-image",
        "best_for": ["general image generation", "marketing art", "concepts", "product hero shots"],
        "needs": ["checkpoint"],
    },
    {
        "workflow": "image_to_image",
        "endpoint": "/images/edit",
        "route": "img2img",
        "best_for": ["product variations", "restyling", "photo enhancement", "brand adaptation"],
        "needs": ["checkpoint"],
    },
    {
        "workflow": "inpaint",
        "endpoint": "/images/edit",
        "route": "inpaint",
        "best_for": ["background cleanup", "object removal", "replacement edits", "product cleanup"],
        "needs": ["checkpoint", "mask or auto-mask"],
    },
]


USE_CASE_GUIDE = [
    {
        "task": "product_images",
        "label": "Product images and product hero shots",
        "recommended_workflows": ["text_to_image", "image_to_image", "inpaint"],
        "preferred_image_families": ["flux", "sdxl", "realism", "photo"],
        "style_presets": ["photo", "cinematic"],
        "notes": "Use text-to-image for fresh concepts, img2img for product variations, and inpaint for cleanup or background replacement.",
    },
    {
        "task": "logo_design",
        "label": "Logo, brandmark, and visual identity ideation",
        "recommended_workflows": ["text_to_image", "image_to_image"],
        "preferred_image_families": ["flux", "sdxl", "illustration", "clean design"],
        "style_presets": ["logo", "illustration"],
        "notes": "Use a clean checkpoint for ideation, then refine with img2img or export to vector manually for production branding.",
    },
    {
        "task": "social_ads",
        "label": "Social ads, posters, and campaign graphics",
        "recommended_workflows": ["text_to_image", "image_to_image"],
        "preferred_image_families": ["flux", "sdxl", "photo", "illustration"],
        "style_presets": ["photo", "cinematic", "illustration"],
        "notes": "Use text-to-image for first-pass creatives and img2img to adapt approved art into campaign variants.",
    },
    {
        "task": "promo_video",
        "label": "Promo video and motion concepts",
        "recommended_workflows": ["text_to_video"],
        "preferred_video_models": ["CogVideoX-5b", "CogVideoX-2b"],
        "notes": "Use CogVideoX-5B for quality when VRAM allows, and keep 2B as the fallback for lower-memory systems.",
    },
    {
        "task": "music_bed",
        "label": "Ad music, mood beds, and sonic branding drafts",
        "recommended_workflows": ["text_to_music"],
        "preferred_audio_models": ["musicgen-medium", "musicgen-large"],
        "notes": "Use medium as the default balance of quality and speed, and large when you want the strongest text-to-music fidelity.",
    },
    {
        "task": "voice_transcription",
        "label": "Transcription and voice features",
        "recommended_workflows": ["speech_to_text", "text_to_speech"],
        "preferred_voice_models": ["faster-whisper", "edge-tts"],
        "notes": "Whisper-size selection matters for STT quality; TTS is service-based and does not require local voice-weight downloads.",
    },
    {
        "task": "mesh_generation",
        "label": "3D mesh generation and fabrication concepts",
        "recommended_workflows": ["text_to_3d", "image_to_3d"],
        "preferred_mesh_families": ["triposr", "hunyuan3d", "stable-fast-3d", "instantmesh"],
        "notes": "Use text-to-3D for rough concept meshes and image-to-3D when you have a logo, sketch, or product reference image. Final fabrication assets usually still need cleanup before printing.",
    },
]


def build_model_catalog(repo_root: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    repo_root = repo_root.resolve()
    edison_cfg = config.get("edison", config)
    llm = _discover_llm_models(repo_root, edison_cfg)
    image = _discover_image_models(repo_root, edison_cfg)
    video = _discover_video_models(edison_cfg)
    music = _discover_music_models(edison_cfg)
    voice = _discover_voice_models(edison_cfg)
    mesh = _discover_mesh_models(repo_root, edison_cfg)
    recommendations = _build_recommendations(image, video, music, voice, mesh)

    summary = {
        "llm_installed": len(llm["models"]),
        "image_checkpoints_installed": len(image["checkpoints"]),
        "image_loras_installed": len(image["loras"]),
        "video_models_configured": len(video["supported_models"]),
        "music_models_supported": len(music["models"]),
        "voice_stt_options": len(voice["stt_models"]),
        "mesh_model_candidates": len(mesh["model_candidates"]),
        "mesh_custom_nodes": len(mesh["custom_nodes"]),
    }

    return {
        "summary": summary,
        "llm": llm,
        "image": image,
        "video": video,
        "music": music,
        "voice": voice,
        "mesh": mesh,
        "use_case_guide": recommendations,
    }


def recommend_models_for_task(task: str, catalog: Dict[str, Any]) -> Dict[str, Any]:
    lowered = (task or "").strip().lower()
    matches = []
    for guide in catalog.get("use_case_guide", []):
        haystack = " ".join([
            guide.get("task", ""),
            guide.get("label", ""),
            guide.get("notes", ""),
            " ".join(guide.get("recommended_workflows", [])),
        ]).lower()
        if lowered and lowered in haystack:
            matches.append((100, guide))
            continue
        keywords = {
            "product": "product_images",
            "hero": "product_images",
            "logo": "logo_design",
            "brand": "logo_design",
            "social": "social_ads",
            "ad": "social_ads",
            "video": "promo_video",
            "promo": "promo_video",
            "music": "music_bed",
            "audio": "music_bed",
            "voice": "voice_transcription",
            "transcription": "voice_transcription",
            "whisper": "voice_transcription",
            "3d": "mesh_generation",
            "mesh": "mesh_generation",
            "stl": "mesh_generation",
            "glb": "mesh_generation",
            "fabrication": "mesh_generation",
        }
        score = 0
        for word, target in keywords.items():
            if word in lowered and guide.get("task") == target:
                score += 3 if target == "mesh_generation" else 1
        if score:
            matches.append((score, guide))

    deduped = []
    seen = set()
    for _score, match in sorted(matches, key=lambda item: item[0], reverse=True):
        task_name = match.get("task")
        if task_name in seen:
            continue
        seen.add(task_name)
        deduped.append(match)

    return {
        "task": task,
        "matches": deduped or catalog.get("use_case_guide", [])[:3],
    }


def _discover_llm_models(repo_root: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    core_cfg = config.get("core", {})
    models_rel_path = core_cfg.get("models_path", "models/llm")
    models_dir = (repo_root / models_rel_path).resolve()
    installed = []
    if models_dir.exists():
        for model_file in sorted(models_dir.glob("*.gguf")):
            size_gb = model_file.stat().st_size / (1024 ** 3)
            name = model_file.name
            installed.append({
                "name": model_file.stem,
                "filename": name,
                "path": str(model_file),
                "size_gb": round(size_gb, 2),
                "roles": _infer_llm_roles(name),
            })

    defaults = {
        "fast": core_cfg.get("fast_model"),
        "medium": core_cfg.get("medium_model"),
        "deep": core_cfg.get("deep_model"),
        "reasoning": core_cfg.get("reasoning_model"),
        "vision": core_cfg.get("vision_model"),
        "vision_clip": core_cfg.get("vision_clip"),
        "vision_code": core_cfg.get("vision_code_model"),
        "vision_code_clip": core_cfg.get("vision_code_clip"),
    }
    return {
        "models_dir": _relative_or_absolute(models_dir, repo_root),
        "models": installed,
        "defaults": defaults,
    }


def _discover_image_models(repo_root: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    comfyui_cfg = config.get("comfyui", {})
    roots = _comfyui_model_roots(repo_root)
    selected_root = next((root for root in roots if root.exists()), roots[0]) if roots else repo_root / "ComfyUI" / "models"

    groups = {key: [] for key in COMFYUI_MODEL_SUBDIRS}
    for dirname, asset_type in COMFYUI_MODEL_SUBDIRS.items():
        directory = selected_root / dirname
        if not directory.exists():
            continue
        for entry in sorted(_iter_model_files(directory)):
            groups[dirname].append({
                "name": entry.stem,
                "filename": entry.name,
                "path": _relative_or_absolute(entry, repo_root),
                "asset_type": asset_type,
                "capabilities": _infer_image_capabilities(entry.name, asset_type),
            })

    return {
        "comfyui": {
            "host": comfyui_cfg.get("host", "127.0.0.1"),
            "port": comfyui_cfg.get("port", 8188),
            "models_root": _relative_or_absolute(selected_root, repo_root),
            "candidate_roots": [_relative_or_absolute(root, repo_root) for root in roots],
        },
        "checkpoints": groups["checkpoints"],
        "loras": groups["loras"],
        "vae": groups["vae"],
        "controlnet": groups["controlnet"],
        "upscalers": groups["upscale_models"],
        "clip_vision": groups["clip_vision"],
        "ipadapter": groups["ipadapter"],
        "workflows": SUPPORTED_IMAGE_WORKFLOWS,
    }


def _discover_video_models(config: Dict[str, Any]) -> Dict[str, Any]:
    video_cfg = config.get("video", {})
    current = video_cfg.get("cogvideox_model", "THUDM/CogVideoX-5b")
    supported = [current]
    if current != "THUDM/CogVideoX-2b":
        supported.append("THUDM/CogVideoX-2b")
    return {
        "backend": "cogvideox-diffusers",
        "endpoint": "/generate-video",
        "configured_model": current,
        "supported_models": [
            {
                "name": model_name,
                "task": "text_to_video",
                "quality_profile": "higher_quality" if "5b" in model_name.lower() else "lower_vram_fallback",
            }
            for model_name in supported
        ],
        "defaults": {
            "width": video_cfg.get("width", 720),
            "height": video_cfg.get("height", 480),
            "fps": video_cfg.get("fps", 8),
            "steps": video_cfg.get("num_inference_steps", 30),
        },
    }


def _discover_music_models(config: Dict[str, Any]) -> Dict[str, Any]:
    music_cfg = config.get("music", {})
    current = f"musicgen-{music_cfg.get('model_size', 'medium')}"
    return {
        "endpoint": "/generate-music",
        "current_model": current,
        "models": [
            {
                "name": "musicgen-small",
                "task": "text_to_music",
                "vram": "~4GB",
                "best_for": ["quick drafts", "short ad ideas"],
            },
            {
                "name": "musicgen-medium",
                "task": "text_to_music",
                "vram": "~8GB",
                "best_for": ["balanced quality", "marketing beds", "default use"],
            },
            {
                "name": "musicgen-large",
                "task": "text_to_music",
                "vram": "~16GB",
                "best_for": ["highest quality", "detailed prompt adherence"],
            },
        ],
    }


def _discover_voice_models(config: Dict[str, Any]) -> Dict[str, Any]:
    voice_cfg = config.get("voice", {})
    configured_stt = voice_cfg.get("stt_model", "base")
    return {
        "stt_endpoint": "/voice/stt",
        "tts_endpoint": "/voice/tts",
        "stt_models": [
            {"name": size, "family": "whisper", "task": "speech_to_text"}
            for size in ["tiny", "base", "small", "medium", "large-v3"]
        ],
        "configured_stt_model": configured_stt,
        "tts_engine": "edge-tts",
        "tts_voice": voice_cfg.get("tts_voice", "en-US-GuyNeural"),
    }


def _discover_mesh_models(repo_root: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    comfyui_cfg = config.get("comfyui", {})
    comfy_root = repo_root / "ComfyUI"
    custom_nodes_dir = comfy_root / "custom_nodes"
    model_roots = _comfyui_model_roots(repo_root)

    custom_nodes = []
    if custom_nodes_dir.exists():
        for entry in sorted(custom_nodes_dir.iterdir()):
            if not entry.is_dir():
                continue
            lowered = entry.name.lower()
            hints = [hint for hint in KNOWN_3D_NODE_HINTS if hint in lowered]
            custom_nodes.append({
                "name": entry.name,
                "path": _relative_or_absolute(entry, repo_root),
                "likely_3d": bool(hints),
                "hints": sorted(set(hints)),
            })

    model_candidates = []
    seen = set()
    for root in model_roots:
        if not root.exists():
            continue
        for entry in root.rglob("*"):
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".onnx"}:
                continue
            lowered = entry.name.lower()
            hints = [hint for hint in KNOWN_3D_MODEL_HINTS if hint in lowered]
            if not hints:
                continue
            key = str(entry.resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            model_candidates.append({
                "name": entry.stem,
                "filename": entry.name,
                "path": _relative_or_absolute(entry, repo_root),
                "hints": sorted(set(hints)),
            })

    likely_nodes = [item for item in custom_nodes if item["likely_3d"]]
    readiness = {
        "comfyui_configured": bool(comfyui_cfg),
        "comfyui_root_present": comfy_root.exists(),
        "custom_nodes_present": bool(custom_nodes),
        "likely_3d_nodes_present": bool(likely_nodes),
        "model_candidates_present": bool(model_candidates),
        "usable": bool(likely_nodes and model_candidates),
        "status": "ready" if (likely_nodes and model_candidates) else "setup_required",
    }

    return {
        "backend": "comfyui_mesh",
        "endpoints": ["/3d/generate", "/generate-3d", "/3d/status/{job_id}", "/3d/result/{job_id}"],
        "workflows": SUPPORTED_MESH_WORKFLOWS,
        "custom_nodes_root": _relative_or_absolute(custom_nodes_dir, repo_root),
        "custom_nodes": custom_nodes,
        "model_candidates": model_candidates,
        "readiness": readiness,
        "download_if_empty": "Install a ComfyUI 3D workflow pack plus compatible model weights such as TripoSR, Hunyuan3D, Stable Fast 3D, or InstantMesh before relying on text-to-3D.",
    }


def _build_recommendations(image: Dict[str, Any], video: Dict[str, Any], music: Dict[str, Any], voice: Dict[str, Any], mesh: Dict[str, Any]) -> List[Dict[str, Any]]:
    checkpoint_names = [item["filename"] for item in image.get("checkpoints", [])]
    photo_candidates = _select_matching_models(checkpoint_names, ["flux", "photo", "real", "juggernaut", "xl", "sdxl"])
    clean_candidates = _select_matching_models(checkpoint_names, ["flux", "xl", "sdxl", "design", "graphic"])

    recommendations = []
    for guide in USE_CASE_GUIDE:
        entry = dict(guide)
        if guide["task"] == "product_images":
            entry["suggested_models"] = photo_candidates[:4]
            entry["download_if_empty"] = "Install at least one strong SDXL or FLUX checkpoint for product-photo and marketing-image work."
        elif guide["task"] == "logo_design":
            entry["suggested_models"] = clean_candidates[:4]
            entry["download_if_empty"] = "Install a clean general-purpose FLUX or SDXL checkpoint; logos do not need a dedicated photoreal model."
        elif guide["task"] == "promo_video":
            entry["suggested_models"] = [item["name"] for item in video.get("supported_models", [])]
        elif guide["task"] == "music_bed":
            entry["suggested_models"] = [item["name"] for item in music.get("models", [])]
        elif guide["task"] == "voice_transcription":
            entry["suggested_models"] = [item["name"] for item in voice.get("stt_models", [])]
        elif guide["task"] == "mesh_generation":
            entry["suggested_models"] = [item["filename"] for item in mesh.get("model_candidates", [])[:4]]
            entry["download_if_empty"] = mesh.get("download_if_empty")
            entry["readiness"] = mesh.get("readiness", {})
        else:
            entry["suggested_models"] = clean_candidates[:4] or photo_candidates[:4]
        recommendations.append(entry)
    return recommendations


def _infer_llm_roles(name: str) -> List[str]:
    lowered = name.lower()
    roles = []
    if "coder" in lowered:
        roles.append("coding")
    if any(token in lowered for token in ["vision", "llava", "vl", "minicpm"]):
        roles.append("vision")
    if any(token in lowered for token in ["14b", "tinyllama", "1.1b"]):
        roles.append("fast_or_light")
    if any(token in lowered for token in ["72b", "32b"]):
        roles.append("deep_reasoning")
    return roles or ["general_chat"]


def _infer_image_capabilities(name: str, asset_type: str) -> List[str]:
    lowered = name.lower()
    capabilities = []
    if asset_type == "checkpoint":
        capabilities.extend(["text_to_image", "image_to_image", "inpaint"])
    if asset_type == "lora":
        capabilities.extend(["style_transfer", "subject_consistency"])
    if asset_type == "controlnet":
        capabilities.extend(["structure_guidance", "pose_or_edge_guidance"])
    if asset_type == "ipadapter":
        capabilities.extend(["reference_image_guidance"])
    if asset_type == "upscaler":
        capabilities.extend(["upscaling", "detail_recovery"])
    if any(token in lowered for token in ["flux", "sdxl", "xl"]):
        capabilities.append("general_high_quality")
    if any(token in lowered for token in ["real", "photo", "juggernaut"]):
        capabilities.append("product_and_photo")
    if any(token in lowered for token in ["turbo", "lightning"]):
        capabilities.append("fast_iteration")
    if any(token in lowered for token in ["anime", "pony"]):
        capabilities.append("illustration")
    if "inpaint" in lowered:
        capabilities.append("specialized_inpaint")
    return sorted(set(capabilities))


def _comfyui_model_roots(repo_root: Path) -> List[Path]:
    candidates = []
    env_root = os.environ.get("EDISON_COMFYUI_MODELS_DIR")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend([
        repo_root / "ComfyUI" / "models",
        repo_root / "comfyui" / "models",
        repo_root / "models" / "comfyui",
    ])
    deduped = []
    seen = set()
    for path in candidates:
        key = str(path.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path.resolve(strict=False))
    return deduped


def _iter_model_files(directory: Path) -> Iterable[Path]:
    allowed_suffixes = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf"}
    for entry in directory.iterdir():
        if entry.is_file() and entry.suffix.lower() in allowed_suffixes:
            yield entry


def _select_matching_models(names: List[str], patterns: List[str]) -> List[str]:
    ranked = []
    for name in names:
        lowered = name.lower()
        score = sum(1 for pattern in patterns if pattern in lowered)
        ranked.append((score, name))
    ranked.sort(key=lambda item: (item[0], item[1].lower()), reverse=True)
    return [name for score, name in ranked if score > 0] or names[:4]


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(repo_root.resolve(strict=False)))
    except Exception:
        return str(path.resolve(strict=False))