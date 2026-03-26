from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import os
import shutil
import subprocess


SUPPORTED_MODEL_EXTENSIONS = {".stl", ".3mf", ".obj"}


@dataclass
class SlicingOptions:
    profile: str = "0.2mm"
    quality: str = "standard"
    layer_height: Optional[float] = None
    material: str = "PLA"
    infill: int = 20
    supports: bool = False
    adhesion: str = "skirt"
    nozzle: str = "0.4"
    speed_profile: str = "balanced"

    @classmethod
    def from_request(cls, payload: Optional[dict]) -> "SlicingOptions":
        payload = payload or {}

        def _as_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)

        def _as_int(value: Any, default: int) -> int:
            try:
                return max(0, min(100, int(value)))
            except (TypeError, ValueError):
                return default

        def _as_float(value: Any) -> Optional[float]:
            if value in (None, ""):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        return cls(
            profile=str(payload.get("profile") or "0.2mm").strip() or "0.2mm",
            quality=str(payload.get("quality") or "standard").strip() or "standard",
            layer_height=_as_float(payload.get("layer_height")),
            material=str(payload.get("material") or "PLA").strip() or "PLA",
            infill=_as_int(payload.get("infill"), 20),
            supports=_as_bool(payload.get("supports")),
            adhesion=str(payload.get("adhesion") or "skirt").strip() or "skirt",
            nozzle=str(payload.get("nozzle") or "0.4").strip() or "0.4",
            speed_profile=str(payload.get("speed_profile") or "balanced").strip() or "balanced",
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SlicerService:
    def __init__(self, workspace_root: Path, config: Optional[dict] = None):
        self.workspace_root = Path(workspace_root)
        self.config = config or {}

    def _slicer_config(self) -> dict:
        return self.config.get("edison", {}).get("printing", {}).get("slicer", {})

    def _resolve_config_path(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (self.workspace_root / candidate).resolve()
        return str(candidate) if candidate.exists() else None

    def _cura_material_diameter(self) -> float:
        try:
            return float(self._slicer_config().get("material_diameter", 1.75))
        except (TypeError, ValueError):
            return 1.75

    def _currency(self) -> str:
        return str(self._slicer_config().get("currency", "USD")).upper()

    def _material_cost_per_kg(self, material: str) -> float:
        overrides = self._slicer_config().get("material_cost_per_kg", {})
        key = (material or "pla").lower()
        if isinstance(overrides, dict) and key in overrides:
            try:
                return float(overrides[key])
            except (TypeError, ValueError):
                pass
        defaults = {
            "pla": 25.0,
            "petg": 30.0,
            "abs": 32.0,
            "tpu": 40.0,
        }
        return defaults.get(key, 30.0)

    def _engine_specs(self) -> list[dict[str, Any]]:
        slicer_config = self._slicer_config()
        cura_definition = self._resolve_config_path(slicer_config.get("cura_definition") or slicer_config.get("definition_file"))
        cura_profile = self._resolve_config_path(slicer_config.get("cura_profile") or slicer_config.get("profile_file"))
        return [
            {
                "key": "curaengine",
                "label": "CuraEngine",
                "binary": shutil.which("CuraEngine"),
                "supports_structured_options": True,
                "requires_definition": True,
                "definition_file": cura_definition,
                "profile_file": cura_profile,
            },
            {
                "key": "prusa-slicer",
                "label": "PrusaSlicer",
                "binary": shutil.which("prusa-slicer"),
                "supports_structured_options": True,
                "requires_definition": False,
            },
            {
                "key": "orca-slicer",
                "label": "OrcaSlicer",
                "binary": shutil.which("orca-slicer"),
                "supports_structured_options": True,
                "requires_definition": False,
            },
            {
                "key": "slic3r",
                "label": "Slic3r",
                "binary": shutil.which("slic3r"),
                "supports_structured_options": False,
                "requires_definition": False,
            },
        ]

    def _engine_payload(self, spec: dict[str, Any]) -> dict[str, Any]:
        available = bool(spec.get("binary"))
        has_definition = bool(spec.get("definition_file")) if spec.get("requires_definition") else True
        usable = available and has_definition
        status = "ready" if usable else "needs_setup" if available else "missing"
        payload = {
            "key": spec["key"],
            "label": spec["label"],
            "available": available,
            "usable": usable,
            "status": status,
            "binary": spec.get("binary"),
            "supports_structured_options": spec.get("supports_structured_options", False),
            "requires_definition": spec.get("requires_definition", False),
        }
        if spec.get("requires_definition"):
            payload["definition_file"] = spec.get("definition_file")
            payload["profile_file"] = spec.get("profile_file")
            payload["definition_configured"] = bool(spec.get("definition_file"))
        return payload

    def get_capabilities(self) -> dict[str, Any]:
        engines = [self._engine_payload(spec) for spec in self._engine_specs()]
        preferred = next((engine for engine in engines if engine["available"]), None)
        active = next((engine for engine in engines if engine["usable"]), None)
        return {
            "engines": engines,
            "preferred_engine": preferred["key"] if preferred else None,
            "active_engine": active["key"] if active else None,
            "supports_estimates": True,
            "supported_formats": sorted(SUPPORTED_MODEL_EXTENSIONS),
            "quality_presets": ["draft", "standard", "fine"],
            "materials": ["PLA", "PETG", "ABS", "TPU"],
            "adhesion_types": ["none", "skirt", "brim", "raft"],
            "speed_profiles": ["slow", "balanced", "fast"],
            "currency": self._currency(),
            "cura_config": {
                "definition_file": engines[0].get("definition_file"),
                "profile_file": engines[0].get("profile_file"),
                "configured": bool(engines[0].get("definition_file")),
            },
            "default_options": SlicingOptions().to_dict(),
        }

    def estimate(self, model_path: Path, options: SlicingOptions) -> dict[str, Any]:
        stat = model_path.stat()
        size_mb = max(stat.st_size / (1024 * 1024), 0.01)
        layer_height = options.layer_height if options.layer_height is not None else self._profile_to_layer_height(options.profile)
        density_factor = 0.75 + (options.infill / 100.0)
        support_factor = 1.18 if options.supports else 1.0
        quality_factor = max(0.4, 0.28 / max(layer_height, 0.08))

        duration_hours = size_mb * 1.7 * density_factor * support_factor * quality_factor
        material_grams = size_mb * 28.0 * density_factor * support_factor
        material_cost_per_kg = self._material_cost_per_kg(options.material)
        cost_per_gram = material_cost_per_kg / 1000.0
        cost = material_grams * cost_per_gram
        spool_grams = 1000.0
        spool_fraction = min(1.0, material_grams / spool_grams)

        return {
            "duration_minutes": max(1, int(duration_hours * 60)),
            "material_grams": round(material_grams, 2),
            "material_meters": round(material_grams / 3.0, 2),
            "estimated_cost": round(cost, 2),
            "currency": self._currency(),
            "spool_fraction": round(spool_fraction, 3),
            "cost_breakdown": {
                "material_cost_per_kg": round(material_cost_per_kg, 2),
                "cost_per_gram": round(cost_per_gram, 4),
                "spool_size_grams": spool_grams,
            },
            "assumptions": {
                "layer_height": layer_height,
                "material": options.material,
                "infill": options.infill,
                "supports": options.supports,
                "adhesion": options.adhesion,
                "nozzle": options.nozzle,
                "speed_profile": options.speed_profile,
            },
        }

    def slice(self, model_path: Path, output_path: Path, options: SlicingOptions) -> dict[str, Any]:
        engine_spec = self._pick_engine_for_execution()
        if engine_spec is None:
            raise RuntimeError("No usable slicer found (install CuraEngine with definitions, prusa-slicer, orca-slicer, or slic3r)")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = self._build_command(engine_spec, model_path, output_path, options)
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if result.returncode != 0 or not output_path.exists():
            message = (result.stderr or result.stdout or "Slicing failed")[:1200]
            raise RuntimeError(message)

        return {
            "engine": engine_spec["key"],
            "command": command,
            "output_path": str(output_path),
            "options": options.to_dict(),
        }

    def _pick_engine_for_execution(self) -> Optional[dict[str, Any]]:
        for spec in self._engine_specs():
            payload = self._engine_payload(spec)
            if payload["usable"]:
                return spec
        return None

    def _build_command(self, spec: dict[str, Any], model_path: Path, output_path: Path, options: SlicingOptions) -> list[str]:
        binary = spec["binary"]
        key = spec["key"]
        if key == "curaengine":
            definition_file = spec.get("definition_file")
            if not definition_file:
                raise RuntimeError("CuraEngine is available but no definition file is configured")
            command = [binary, "slice", "-j", str(definition_file), "-l", str(model_path), "-o", str(output_path)]
            if spec.get("profile_file"):
                command.extend(["-j", str(spec["profile_file"])])
            command.extend(self._cura_settings_args(options))
            return command

        command = [binary, "--export-gcode", str(model_path), "--output", str(output_path)]
        command.extend(self._generic_settings_args(key, options))
        return command

    def _cura_settings_args(self, options: SlicingOptions) -> list[str]:
        layer_height = options.layer_height if options.layer_height is not None else self._profile_to_layer_height(options.profile)
        args = [
            "-s", f"layer_height={layer_height}",
            "-s", f"infill_sparse_density={options.infill}",
            "-s", f"material_diameter={self._cura_material_diameter()}",
            "-s", f"adhesion_type={options.adhesion}",
            "-s", f"support_enable={'true' if options.supports else 'false'}",
            "-s", f"machine_nozzle_size={options.nozzle}",
        ]
        return args

    def _generic_settings_args(self, key: str, options: SlicingOptions) -> list[str]:
        args: list[str] = []
        layer_height = options.layer_height if options.layer_height is not None else self._profile_to_layer_height(options.profile)
        if key in {"prusa-slicer", "orca-slicer", "slic3r"}:
            args.extend(["--layer-height", str(layer_height)])
            args.extend(["--fill-density", f"{options.infill}%"])
            args.extend(["--support-material", "1" if options.supports else "0"])
            if options.adhesion == "brim":
                args.extend(["--brim-width", "3"])
            elif options.adhesion == "raft":
                args.extend(["--raft-layers", "1"])
            args.extend(["--nozzle-diameter", str(options.nozzle)])
        return args

    def _profile_to_layer_height(self, profile: str) -> float:
        lowered = (profile or "").lower()
        if "0.1" in lowered or "fine" in lowered:
            return 0.1
        if "0.28" in lowered or "draft" in lowered:
            return 0.28
        if "0.3" in lowered:
            return 0.3
        return 0.2

def validate_model_path(model_path: Path) -> None:
    if model_path.suffix.lower() not in SUPPORTED_MODEL_EXTENSIONS:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")
    if not model_path.exists() or not model_path.is_file():
        raise ValueError("Model file not found")
    if os.path.getsize(model_path) <= 0:
        raise ValueError("Model file is empty")