"""ComfyUI workflow discovery and execution helpers.

Edison has several features that depend on ComfyUI, but optional custom nodes
and model files vary a lot from machine to machine.  This module provides a
small, honest contract:

* discover versioned workflow templates
* validate Edison metadata and required placeholders
* inject variables without hard-coding one workflow shape
* submit/poll/cancel through ComfyUI's HTTP API
* translate common connection and workflow errors into Edison-friendly payloads
"""

from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

logger = logging.getLogger(__name__)


DEFAULT_PLACEHOLDERS = {
    "source_segment_path",
    "source_frame_path",
    "persona_paths",
    "persona_pack_path",
    "output_path",
    "quality_preset",
    "transformation_scope",
    "segment_id",
    "gpu_index",
    "gpu_name",
}


@dataclass
class WorkflowValidation:
    ok: bool
    template_id: str = ""
    path: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    required_nodes: List[str] = field(default_factory=list)
    required_models: List[str] = field(default_factory=list)
    placeholders: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowTemplate:
    template_id: str
    name: str
    version: str
    path: Path
    workflow: Dict[str, Any]
    description: str = ""
    required_nodes: List[str] = field(default_factory=list)
    required_models: List[str] = field(default_factory=list)
    parameter_schema: Dict[str, Any] = field(default_factory=dict)
    capability_metadata: Dict[str, Any] = field(default_factory=dict)
    validation: WorkflowValidation = field(default_factory=lambda: WorkflowValidation(ok=False))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "version": self.version,
            "path": str(self.path),
            "description": self.description,
            "required_nodes": self.required_nodes,
            "required_models": self.required_models,
            "parameter_schema": self.parameter_schema,
            "capability_metadata": self.capability_metadata,
            "validation": self.validation.to_dict(),
        }


def _meta_from_workflow(data: Dict[str, Any]) -> Dict[str, Any]:
    meta = data.get("edison") or data.get("_edison") or data.get("metadata") or {}
    return meta if isinstance(meta, dict) else {}


def _workflow_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    payload = data.get("workflow") or data.get("prompt") or data
    if not isinstance(payload, dict):
        return {}
    payload = copy.deepcopy(payload)
    for key in ("edison", "_edison", "metadata"):
        payload.pop(key, None)
    return payload


def discover_workflow_templates(directory: Path) -> List[WorkflowTemplate]:
    """Load all JSON ComfyUI workflow templates in a directory."""

    root = directory.resolve(strict=False)
    if not root.exists():
        return []
    templates: List[WorkflowTemplate] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            validation = WorkflowValidation(ok=False, path=str(path), errors=[f"invalid JSON: {exc}"])
            templates.append(
                WorkflowTemplate(
                    template_id=path.stem,
                    name=path.stem,
                    version="unknown",
                    path=path,
                    workflow={},
                    validation=validation,
                )
            )
            continue
        meta = _meta_from_workflow(data)
        template_id = str(meta.get("template_id") or meta.get("id") or path.stem)
        workflow = _workflow_payload(data)
        validation = validate_workflow_template(workflow, meta=meta, template_id=template_id, path=path)
        templates.append(
            WorkflowTemplate(
                template_id=template_id,
                name=str(meta.get("name") or template_id),
                version=str(meta.get("version") or "1"),
                path=path,
                workflow=workflow,
                description=str(meta.get("description") or ""),
                required_nodes=[str(x) for x in meta.get("required_nodes", [])],
                required_models=[str(x) for x in meta.get("required_models", [])],
                parameter_schema=dict(meta.get("parameter_schema") or {}),
                capability_metadata=dict(meta.get("capabilities") or {}),
                validation=validation,
            )
        )
    return templates


def validate_workflow_template(
    workflow: Dict[str, Any],
    *,
    meta: Optional[Dict[str, Any]] = None,
    template_id: str = "",
    path: Optional[Path] = None,
) -> WorkflowValidation:
    """Validate the basic Edison/ComfyUI shape without requiring ComfyUI."""

    meta = meta or {}
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(workflow, dict) or not workflow:
        errors.append("workflow payload is empty or not an object")
    node_like = [key for key, value in workflow.items() if isinstance(value, dict) and ("class_type" in value or "inputs" in value)]
    if workflow and not node_like:
        warnings.append("workflow does not look like a ComfyUI API prompt; expected node objects with class_type/inputs")
    placeholders = sorted(find_placeholders(workflow))
    if not placeholders:
        warnings.append("no ${...} placeholders found; template may not receive Edison source/persona/output variables")
    unknown = [ph for ph in placeholders if ph not in DEFAULT_PLACEHOLDERS]
    if unknown:
        warnings.append("unknown placeholders: " + ", ".join(unknown))
    required_nodes = [str(x) for x in meta.get("required_nodes", [])]
    required_models = [str(x) for x in meta.get("required_models", [])]
    return WorkflowValidation(
        ok=not errors,
        template_id=template_id,
        path=str(path) if path else "",
        errors=errors,
        warnings=warnings,
        required_nodes=required_nodes,
        required_models=required_models,
        placeholders=placeholders,
    )


def find_placeholders(value: Any) -> set[str]:
    placeholders: set[str] = set()
    if isinstance(value, str):
        start = 0
        while True:
            pos = value.find("${", start)
            if pos < 0:
                break
            end = value.find("}", pos + 2)
            if end < 0:
                break
            key = value[pos + 2 : end].strip()
            if key:
                placeholders.add(key)
            start = end + 1
    elif isinstance(value, dict):
        for item in value.values():
            placeholders.update(find_placeholders(item))
    elif isinstance(value, list):
        for item in value:
            placeholders.update(find_placeholders(item))
    return placeholders


def inject_workflow_variables(value: Any, variables: Dict[str, Any]) -> Any:
    """Recursively replace ${var} placeholders inside a workflow payload."""

    if isinstance(value, str):
        out = value
        for key, replacement in variables.items():
            token = "${" + key + "}"
            if token not in out:
                continue
            if isinstance(replacement, (list, dict)):
                out = out.replace(token, json.dumps(replacement))
            else:
                out = out.replace(token, str(replacement))
        return out
    if isinstance(value, dict):
        return {key: inject_workflow_variables(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [inject_workflow_variables(item, variables) for item in value]
    return value


class ComfyUIExecutionService:
    """Small HTTP client for ComfyUI prompt execution."""

    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout_s: float = 10.0) -> None:
        self.base_url = str(base_url or "http://127.0.0.1:8188").rstrip("/")
        self.timeout_s = float(timeout_s)
        self.client_id = f"edison-{uuid.uuid4().hex[:10]}"

    def health(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.base_url}/queue", timeout=self.timeout_s)
            return {"ok": bool(response.ok), "status_code": response.status_code, "base_url": self.base_url}
        except Exception as exc:
            return {"ok": False, "base_url": self.base_url, "error": translate_comfyui_error(exc)}

    def submit(self, workflow: Dict[str, Any], extra_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"prompt": workflow, "client_id": self.client_id}
        if extra_data:
            payload["extra_data"] = extra_data
        try:
            response = requests.post(f"{self.base_url}/prompt", json=payload, timeout=self.timeout_s)
            if not response.ok:
                return {
                    "ok": False,
                    "status_code": response.status_code,
                    "error": translate_comfyui_error(response.text),
                    "base_url": self.base_url,
                }
            data = response.json()
            return {"ok": True, "prompt_id": data.get("prompt_id"), "number": data.get("number"), "base_url": self.base_url}
        except Exception as exc:
            return {"ok": False, "error": translate_comfyui_error(exc), "base_url": self.base_url}

    def history(self, prompt_id: str) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=self.timeout_s)
            if not response.ok:
                return {"ok": False, "status_code": response.status_code, "error": translate_comfyui_error(response.text)}
            return {"ok": True, "history": response.json()}
        except Exception as exc:
            return {"ok": False, "error": translate_comfyui_error(exc)}

    def poll_until_complete(self, prompt_id: str, timeout_s: float = 900.0, poll_interval_s: float = 2.0) -> Dict[str, Any]:
        deadline = time.time() + max(1.0, timeout_s)
        last: Dict[str, Any] = {}
        while time.time() < deadline:
            last = self.history(prompt_id)
            if not last.get("ok"):
                time.sleep(max(0.25, poll_interval_s))
                continue
            history = last.get("history") or {}
            item = history.get(prompt_id) if isinstance(history, dict) else None
            if item:
                status = item.get("status") or {}
                if status.get("completed") is True:
                    return {"ok": True, "prompt_id": prompt_id, "history": item}
                messages = status.get("messages") or []
                if any(str(m).lower().find("exception") >= 0 for m in messages):
                    return {"ok": False, "prompt_id": prompt_id, "history": item, "error": "ComfyUI reported an exception"}
            time.sleep(max(0.25, poll_interval_s))
        return {"ok": False, "prompt_id": prompt_id, "error": "Timed out waiting for ComfyUI workflow", "last": last}

    def cancel(self, prompt_id: str) -> Dict[str, Any]:
        try:
            response = requests.post(f"{self.base_url}/queue", json={"delete": [prompt_id]}, timeout=self.timeout_s)
            return {"ok": bool(response.ok), "status_code": response.status_code}
        except Exception as exc:
            return {"ok": False, "error": translate_comfyui_error(exc)}


def translate_comfyui_error(error: Any) -> str:
    text = str(error or "").strip()
    lowered = text.lower()
    if "connection refused" in lowered or "failed to establish" in lowered:
        return "ComfyUI is not reachable at the configured host/port."
    if "no such file" in lowered or "filenotfound" in lowered:
        return "ComfyUI workflow references a missing model, input, or output file."
    if "class_type" in lowered and "not found" in lowered:
        return "ComfyUI is missing a custom node required by this workflow."
    if "cuda out of memory" in lowered or "outofmemory" in lowered:
        return "ComfyUI ran out of GPU memory while executing the workflow."
    return text[:1000] or "Unknown ComfyUI error"


def summarize_template_library(templates: Iterable[WorkflowTemplate]) -> Dict[str, Any]:
    rows = [template.to_dict() for template in templates]
    available = [row for row in rows if (row.get("validation") or {}).get("ok")]
    setup_required: List[str] = []
    for row in rows:
        validation = row.get("validation") or {}
        setup_required.extend(validation.get("errors") or [])
        setup_required.extend(validation.get("warnings") or [])
    return {
        "template_count": len(rows),
        "available_count": len(available),
        "templates": rows,
        "setup_required": sorted(set(setup_required)),
    }
