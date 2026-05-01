"""
Rule-based business action interpreter for chat-triggered client, project, and branding workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import re

from .branding_ops import BrandingWorkflowService
from .branding_store import BrandingClientStore
from .contracts import BrandingGenerationRequest, MarketingCopyRequest, ProjectCreateRequest
from .model_catalog import build_model_catalog, recommend_models_for_task
from .projects import ProjectWorkspaceManager
from .system_awareness import build_capability_map


_MODEL_REQUEST_VERBS = ("generate", "create", "make", "design", "model", "build", "sculpt")
_MODEL_REQUEST_HINTS = ("3d", "cad", "rhino", "blender", "solidworks", "mesh", "prototype")
_PROCEDURAL_MODEL_ALIASES = {
    "vase": ("vase",),
    "planter": ("planter", "plant pot", "flower pot", "pot"),
    "bowl": ("bowl",),
    "cup": ("cup", "mug", "glass"),
    "box": ("box", "cube"),
    "cylinder": ("cylinder",),
    "cone": ("cone",),
    "sphere": ("sphere", "ball"),
}


def execute_business_action(
    message: str,
    repo_root: Path,
    config: Dict[str, Any],
    branding_store: BrandingClientStore,
    project_manager: ProjectWorkspaceManager,
    node_manager: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    text = (message or "").strip()
    lowered = text.lower()
    workflow_service = BrandingWorkflowService(repo_root, branding_store, project_manager)

    if any(phrase in lowered for phrase in ["business overview", "show overview", "system capabilities", "capability map", "what pages", "what routes"]):
        capability_map = build_capability_map(repo_root, config)
        printer_routes = [route["path"] for route in capability_map.get("routes", []) if "printer" in route["path"] or "printing" in route["path"]]
        project_routes = [route["path"] for route in capability_map.get("routes", []) if "project" in route["path"]][:6]
        pages = ", ".join(page["route"] for page in capability_map.get("pages", [])[:8])
        response = (
            f"EDISON currently exposes {capability_map['summary'].get('route_count', 0)} routes across "
            f"{capability_map['summary'].get('page_count', 0)} pages. "
            f"Key pages: {pages}. "
            f"Project routes include {', '.join(project_routes) or 'none yet'}. "
            f"Printing routes include {', '.join(printer_routes[:6]) or 'none detected'}."
        )
        return {"response": response, "mode_used": "business", "business_action": {"type": "capabilities", "summary": capability_map["summary"]}}

    node_model_action = _maybe_execute_node_model_action(text, lowered, node_manager)
    if node_model_action is not None:
        return node_model_action

    if any(phrase in lowered for phrase in ["what models", "available models", "model catalog", "which model should", "what model should", "which mesh model", "what mesh model", "product image model", "img2img model", "image to image model", "video model", "music model", "text to 3d model", "image to 3d model"]):
        catalog = build_model_catalog(repo_root, config)
        recommendation = recommend_models_for_task(text, catalog)
        summary = catalog.get("summary", {})
        first_match = (recommendation.get("matches") or [{}])[0]
        suggested = ", ".join(first_match.get("suggested_models", [])[:4]) or "no installed match yet"
        response = (
            f"EDISON currently sees {summary.get('llm_installed', 0)} local LLMs, "
            f"{summary.get('image_checkpoints_installed', 0)} ComfyUI image checkpoints, "
            f"{summary.get('image_loras_installed', 0)} LoRAs, and "
            f"{summary.get('mesh_model_candidates', 0)} 3D model candidates. "
            f"Best match for this request: {first_match.get('label', 'general media work')}. "
            f"Recommended workflows: {', '.join(first_match.get('recommended_workflows', [])) or 'text_to_image'}. "
            f"Installed model suggestions: {suggested}."
        )
        return {
            "response": response,
            "mode_used": "business",
            "business_action": {"type": "model_catalog", "catalog": catalog, "recommendation": recommendation},
        }

    client_match = re.search(r"(?:create|add|make|set up)\s+(?:a\s+)?(?:branding\s+)?client(?:\s+folder)?\s+(?:for\s+)?(.+)$", text, re.IGNORECASE)
    if client_match:
        client_name = client_match.group(1).strip(" .")
        result = branding_store.create_client({"business_name": client_name})
        client = result["client"]
        action = "Created" if result["created"] else "Found existing"
        return {
            "response": f"{action} client {client['business_name']} with workspace at {client['paths']['base']}.",
            "mode_used": "business",
            "business_action": {"type": "create_client", "client": client, "created": result["created"]},
        }

    project_name = ""
    client_name = ""
    for pattern in [
        r"(?:create|add|make|start|build)\s+(?:a\s+)?project\s+for\s+(?P<client>.+?)\s+(?:called|named)\s+(?P<name>.+)$",
        r"(?:create|add|make|start|build)\s+(?:a\s+)?project\s+(?:called|named)\s+(?P<name>.+)$",
        r"(?:create|add|make|start|build)\s+(?:a\s+)?project\s+for\s+(?P<client>.+?)\s+(?P<name>.+)$",
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            client_name = (match.groupdict().get("client") or "").strip(" .")
            project_name = (match.groupdict().get("name") or "").strip(" .")
            break

    if project_name and "project" in lowered and not project_name.lower().startswith("for "):
        client = branding_store.find_client_by_name(client_name) if client_name else None
        project = project_manager.create_project(ProjectCreateRequest(
            name=project_name,
            client_id=client.get("id") if client else None,
            service_types=_infer_service_types(lowered),
            notes=f"Created from chat request: {text}",
        ))
        client_label = client.get("business_name") if client else "no linked client"
        return {
            "response": f"Created project {project.name} for {client_label}. Workspace ready at {project.root_path}.",
            "mode_used": "business",
            "business_action": {"type": "create_project", "project": _model_dump(project)},
        }

    if any(term in lowered for term in ["branding package", "logo concept", "logo concepts", "brand voice", "style guide", "tagline", "slogan", "palette", "moodboard"]):
        client = _find_client_from_message(text, branding_store)
        business_name = client.get("business_name") if client else _extract_business_name_from_request(text)
        if business_name:
            project = _find_project_from_message(text, project_manager)
            result = workflow_service.generate_brand_package(BrandingGenerationRequest(
                business_name=business_name,
                client_id=client.get("id") if client else None,
                project_id=project.project_id if project else None,
                prompt=text,
                industry=client.get("industry", "") if client else "",
                style_keywords=_extract_keywords(text),
                include_moodboard=True,
            ))
            location = result["workspace"]
            return {
                "response": f"Generated a branding package for {business_name} and saved it to {location}.",
                "mode_used": "business",
                "business_action": {"type": "branding_package", "result": result},
            }

    if any(term in lowered for term in ["social caption", "social captions", "ad copy", "business description", "website hero", "email campaign", "product copy"]):
        client = _find_client_from_message(text, branding_store)
        business_name = client.get("business_name") if client else _extract_business_name_from_request(text)
        if business_name:
            project = _find_project_from_message(text, project_manager)
            result = workflow_service.generate_marketing_copy(MarketingCopyRequest(
                business_name=business_name,
                client_id=client.get("id") if client else None,
                project_id=project.project_id if project else None,
                prompt=text,
                industry=client.get("industry", "") if client else "",
                copy_types=_infer_copy_types(lowered),
            ))
            return {
                "response": f"Generated marketing copy for {business_name} and saved it to {result['workspace']}.",
                "mode_used": "business",
                "business_action": {"type": "marketing_copy", "result": result},
            }

    return None


def _extract_business_name_from_request(text: str) -> str:
    match = re.search(r"(?:for|about)\s+([A-Z][A-Za-z0-9&'\- ]+)$", text)
    if match:
        return match.group(1).strip(" .")
    return ""


def _extract_keywords(text: str) -> list[str]:
    keywords = []
    for word in re.findall(r"[A-Za-z]{4,}", text or ""):
        lowered = word.lower()
        if lowered in {"create", "brand", "branding", "package", "logo", "concepts", "slogan", "style", "guide", "with", "that", "from", "this"}:
            continue
        if lowered not in keywords:
            keywords.append(lowered)
    return keywords[:6]


def _find_client_from_message(text: str, branding_store: BrandingClientStore) -> Optional[Dict[str, Any]]:
    for client in branding_store.list_clients():
        business_name = str(client.get("business_name") or "")
        if business_name and business_name.lower() in text.lower():
            return client
    extracted = _extract_business_name_from_request(text)
    return branding_store.find_client_by_name(extracted) if extracted else None


def _find_project_from_message(text: str, project_manager: ProjectWorkspaceManager):
    normalized_message = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    message_tokens = {token for token in normalized_message.split() if len(token) > 2}
    for project in project_manager.list_projects():
        normalized_name = re.sub(r"[^a-z0-9]+", " ", project.name.lower()).strip()
        if normalized_name and normalized_name in normalized_message:
            return project
        name_tokens = {token for token in normalized_name.split() if len(token) > 2}
        if name_tokens and name_tokens.issubset(message_tokens):
            return project
    return None


def _infer_service_types(lowered: str) -> list[str]:
    service_types = []
    for service in ["branding", "marketing", "printing", "video"]:
        if service in lowered:
            service_types.append(service)
    return service_types or ["mixed"]


def _infer_copy_types(lowered: str) -> list[str]:
    mapping = {
        "ad copy": "ad_copy",
        "social caption": "social_captions",
        "social captions": "social_captions",
        "email campaign": "email_campaign",
        "business description": "business_description",
        "product copy": "product_copy",
        "website hero": "website_hero_text",
    }
    copy_types = []
    for phrase, copy_type in mapping.items():
        if phrase in lowered and copy_type not in copy_types:
            copy_types.append(copy_type)
    return copy_types or ["ad_copy", "social_captions", "business_description", "website_hero_text"]


def _model_dump(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _maybe_execute_node_model_action(
    text: str,
    lowered: str,
    node_manager: Optional[Any],
) -> Optional[Dict[str, Any]]:
    model_request = _parse_model_request(lowered)
    if model_request is None:
        return None

    if model_request["shape"] is None:
        supported = ", ".join(sorted(_PROCEDURAL_MODEL_ALIASES.keys()))
        return {
            "response": (
                "I can auto-run simple CAD requests on a Rhino node right now, but I only have procedural "
                f"templates for: {supported}."
            ),
            "mode_used": "business",
            "business_action": {"type": "node_model_request", "ok": False, "reason": "unsupported_shape"},
        }

    if node_manager is None:
        return {
            "response": "A CAD request was detected, but the node manager is not available on this server.",
            "mode_used": "business",
            "business_action": {"type": "node_model_request", "ok": False, "reason": "node_manager_unavailable"},
        }

    target_node, node_error = _resolve_model_node(text, lowered, node_manager)
    if target_node is None:
        return {
            "response": node_error,
            "mode_used": "business",
            "business_action": {"type": "node_model_request", "ok": False, "reason": "node_unavailable"},
        }

    filename = f"{model_request['shape']}-{_slugify_for_filename(text)[:24] or model_request['shape']}.3dm"
    output_hint = f"~/.edison/generated_models/{filename}"
    payload = {
        "script_content": _build_rhino_script(model_request["shape"], filename),
        "output_paths": [output_hint],
    }

    command_result = _dispatch_node_model_request(
        text=text,
        node_manager=node_manager,
        target_node=target_node,
        payload=payload,
        preferred_software=[software for software in ["rhino", "grasshopper"] if software in lowered] or ["rhino", "grasshopper"],
    )

    if command_result.get("status") == "queued":
        queued_task = command_result.get("task") or {}
        return {
            "response": (
                f"Sent the {model_request['shape']} model request to {target_node.get('name', target_node['id'])}. "
                f"The node will pick it up from the queue on its next heartbeat. Task id: {queued_task.get('id', 'pending')}. "
                f"Output hint: {output_hint}."
            ),
            "mode_used": "business",
            "business_action": {
                "type": "node_model_request",
                "ok": True,
                "status": "queued",
                "shape": model_request["shape"],
                "node": {"id": target_node["id"], "name": target_node.get("name")},
                "task": queued_task,
                "output_paths": [output_hint],
            },
        }

    if not command_result.get("ok"):
        error_text = command_result.get("error") or "unknown transport error"
        return {
            "response": f"I found {target_node.get('name', target_node['id'])}, but the command could not reach the node: {error_text}.",
            "mode_used": "business",
            "business_action": {
                "type": "node_model_request",
                "ok": False,
                "reason": "dispatch_failed",
                "node": {"id": target_node["id"], "name": target_node.get("name")},
            },
        }

    node_response = command_result.get("response") or {}
    if not node_response.get("ok"):
        error_text = node_response.get("error") or "unknown Rhino error"
        return {
            "response": (
                f"I sent the {model_request['shape']} request to {target_node.get('name', target_node['id'])}, "
                f"but Rhino reported: {error_text}."
            ),
            "mode_used": "business",
            "business_action": {
                "type": "node_model_request",
                "ok": False,
                "reason": "execution_failed",
                "node": {"id": target_node["id"], "name": target_node.get("name")},
                "node_response": node_response,
            },
        }

    output_paths = node_response.get("output_paths") or [output_hint]
    return {
        "response": (
            f"Sent the {model_request['shape']} model request to {target_node.get('name', target_node['id'])} "
            f"and Rhino executed it. Output hint: {', '.join(output_paths)}."
        ),
        "mode_used": "business",
        "business_action": {
            "type": "node_model_request",
            "ok": True,
            "shape": model_request["shape"],
            "node": {"id": target_node["id"], "name": target_node.get("name")},
            "output_paths": output_paths,
            "node_response": node_response,
        },
    }


def _parse_model_request(lowered: str) -> Optional[Dict[str, Optional[str]]]:
    if not any(verb in lowered for verb in _MODEL_REQUEST_VERBS):
        return None
    if not any(hint in lowered for hint in _MODEL_REQUEST_HINTS):
        return None
    for shape, aliases in _PROCEDURAL_MODEL_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                return {"shape": shape, "matched_alias": alias}
    return {"shape": None, "matched_alias": None}


def _resolve_model_node(text: str, lowered: str, node_manager: Any) -> tuple[Optional[Dict[str, Any]], str]:
    if hasattr(node_manager, "mark_stale"):
        node_manager.mark_stale()
    nodes_payload = node_manager.list_nodes() if hasattr(node_manager, "list_nodes") else {"nodes": []}
    nodes = nodes_payload.get("nodes", []) if isinstance(nodes_payload, dict) else []

    named_node = _find_named_node(text, nodes)
    if named_node is not None:
        if named_node.get("status") != "online":
            return None, f"The requested node {named_node.get('name', named_node.get('id', 'unknown'))} is registered but not online."
        if not _node_supports_rhino(named_node):
            return None, f"{named_node.get('name', named_node.get('id', 'That node'))} is online, but it is not advertising Rhino support yet."
        return named_node, ""

    preferred_software = [software for software in ["rhino", "grasshopper", "solidworks", "blender"] if software in lowered]
    if not preferred_software:
        preferred_software = ["rhino", "grasshopper"]
    best_node = node_manager.find_best_node_for_task(
        text,
        required_capabilities=["cad", "3d-modeling"],
        preferred_software=preferred_software,
    )
    if best_node is None:
        return None, "No online CAD node is available right now. Start the node agent on the workstation and make sure it has checked in."
    if not _node_supports_rhino(best_node):
        return None, f"{best_node.get('name', best_node.get('id', 'The best CAD node'))} is online, but this automatic modeling flow currently targets Rhino-capable nodes only."
    return best_node, ""


def _dispatch_node_model_request(
    text: str,
    node_manager: Any,
    target_node: Dict[str, Any],
    payload: Dict[str, Any],
    preferred_software: list[str],
) -> Dict[str, Any]:
    if hasattr(node_manager, "delegate_task"):
        queued_result = node_manager.delegate_task(
            task_description=text,
            task_type="rhino_script",
            payload=payload,
            required_capabilities=["cad", "3d-modeling"],
            preferred_software=preferred_software,
            node_id=target_node["id"],
        )
        if queued_result.get("ok"):
            queued_result["status"] = "queued"
            return queued_result

    if hasattr(node_manager, "submit_task"):
        try:
            task = node_manager.submit_task(target_node["id"], "rhino_script", payload)
            return {
                "ok": True,
                "status": "queued",
                "node": {
                    "id": target_node["id"],
                    "name": target_node.get("name"),
                },
                "task": task,
            }
        except Exception as exc:
            queued_result = {"ok": False, "error": str(exc)}

    if hasattr(node_manager, "send_command"):
        try:
            direct_result = node_manager.send_command(target_node["id"], "rhino_script", payload)
        except Exception as exc:
            direct_result = {"ok": False, "error": str(exc)}
        if direct_result.get("ok"):
            direct_result["status"] = "completed"
            return direct_result

    direct_error = (direct_result.get("error") if 'direct_result' in locals() else "") or "unknown transport error"
    queued_error = (queued_result.get("error") if 'queued_result' in locals() else "") or "unknown queue error"
    return {
        "ok": False,
        "error": f"Queued dispatch failed ({queued_error}) and direct dispatch failed ({direct_error})",
    }


def _find_named_node(text: str, nodes: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    normalized_message = _normalize_lookup_value(text)
    message_tokens = {token for token in re.sub(r"[^a-z0-9]+", " ", text.lower()).split() if len(token) > 2}
    for node in nodes:
        name = str(node.get("name") or "")
        if not name:
            continue
        normalized_name = _normalize_lookup_value(name)
        if normalized_name and normalized_name in normalized_message:
            return node
        node_tokens = {token for token in re.sub(r"[^a-z0-9]+", " ", name.lower()).split() if len(token) > 2}
        if node_tokens and node_tokens.issubset(message_tokens):
            return node
    return None


def _normalize_lookup_value(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _node_supports_rhino(node: Dict[str, Any]) -> bool:
    capabilities = {str(item).lower() for item in node.get("capabilities", [])}
    software = {str(item).lower() for item in (node.get("software") or {}).keys()}
    combined = capabilities | software
    return any(item in combined for item in ["rhino", "grasshopper"])


def _slugify_for_filename(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return re.sub(r"-+", "-", lowered)


def _build_rhino_script(shape: str, filename: str) -> str:
    shape_body = _build_rhino_shape_body(shape)
    layer_name = f"EDISON_{filename.rsplit('.', 1)[0]}"
    return (
        "import os\n"
        "import rhinoscriptsyntax as rs\n\n"
        f"output_file = os.path.join(os.path.expanduser('~'), '.edison', 'generated_models', '{filename}')\n"
        "output_dir = os.path.dirname(output_file)\n"
        "if not os.path.isdir(output_dir):\n"
        "    os.makedirs(output_dir)\n\n"
        "previous_layer = rs.CurrentLayer()\n"
        f"layer_name = '{layer_name}'\n"
        "if not rs.IsLayer(layer_name):\n"
        "    rs.AddLayer(layer_name)\n"
        "rs.CurrentLayer(layer_name)\n"
        "created_ids = []\n\n"
        f"{shape_body}\n\n"
        "if not created_ids:\n"
        "    raise Exception('EDISON did not create any geometry.')\n"
        "rs.UnselectAllObjects()\n"
        "for object_id in created_ids:\n"
        "    rs.SelectObject(object_id)\n"
        "rs.Command('-_Export \"{}\" _Enter'.format(output_file), False)\n"
        "rs.UnselectAllObjects()\n"
        "if previous_layer and rs.IsLayer(previous_layer):\n"
        "    rs.CurrentLayer(previous_layer)\n"
    )


def _build_rhino_shape_body(shape: str) -> str:
    if shape == "vase":
        return (
            "axis = rs.AddLine((0, 0, 0), (0, 0, 180))\n"
            "profile = rs.AddInterpCurve([(0, 0, 0), (32, 0, 0), (42, 0, 18), (54, 0, 70), (48, 0, 128), (26, 0, 176), (0, 0, 180)], degree=3)\n"
            "if not axis or not profile:\n"
            "    raise Exception('Failed to build the vase profile.')\n"
            "surface = rs.AddRevSrf(profile, axis)\n"
            "if surface:\n"
            "    created_ids.append(surface)\n"
            "rs.DeleteObject(profile)\n"
            "rs.DeleteObject(axis)"
        )
    if shape == "planter":
        return (
            "axis = rs.AddLine((0, 0, 0), (0, 0, 110))\n"
            "profile = rs.AddInterpCurve([(0, 0, 0), (50, 0, 0), (56, 0, 16), (60, 0, 70), (52, 0, 108), (0, 0, 110)], degree=3)\n"
            "if not axis or not profile:\n"
            "    raise Exception('Failed to build the planter profile.')\n"
            "surface = rs.AddRevSrf(profile, axis)\n"
            "if surface:\n"
            "    created_ids.append(surface)\n"
            "rs.DeleteObject(profile)\n"
            "rs.DeleteObject(axis)"
        )
    if shape == "bowl":
        return (
            "axis = rs.AddLine((0, 0, 0), (0, 0, 80))\n"
            "profile = rs.AddInterpCurve([(0, 0, 0), (64, 0, 0), (76, 0, 18), (86, 0, 42), (72, 0, 76), (0, 0, 80)], degree=3)\n"
            "if not axis or not profile:\n"
            "    raise Exception('Failed to build the bowl profile.')\n"
            "surface = rs.AddRevSrf(profile, axis)\n"
            "if surface:\n"
            "    created_ids.append(surface)\n"
            "rs.DeleteObject(profile)\n"
            "rs.DeleteObject(axis)"
        )
    if shape == "cup":
        return (
            "axis = rs.AddLine((0, 0, 0), (0, 0, 115))\n"
            "profile = rs.AddInterpCurve([(0, 0, 0), (34, 0, 0), (42, 0, 20), (46, 0, 88), (38, 0, 114), (0, 0, 115)], degree=3)\n"
            "if not axis or not profile:\n"
            "    raise Exception('Failed to build the cup profile.')\n"
            "surface = rs.AddRevSrf(profile, axis)\n"
            "if surface:\n"
            "    created_ids.append(surface)\n"
            "rs.DeleteObject(profile)\n"
            "rs.DeleteObject(axis)"
        )
    if shape == "box":
        return (
            "box = rs.AddBox([(0, 0, 0), (80, 0, 0), (80, 60, 0), (0, 60, 0), (0, 0, 50), (80, 0, 50), (80, 60, 50), (0, 60, 50)])\n"
            "if box:\n"
            "    created_ids.append(box)"
        )
    if shape == "cylinder":
        return (
            "cylinder = rs.AddCylinder((0, 0, 0), 120, 40)\n"
            "if cylinder:\n"
            "    created_ids.append(cylinder)"
        )
    if shape == "cone":
        return (
            "cone = rs.AddCone((0, 0, 0), 120, 48)\n"
            "if cone:\n"
            "    created_ids.append(cone)"
        )
    if shape == "sphere":
        return (
            "sphere = rs.AddSphere((0, 0, 55), 55)\n"
            "if sphere:\n"
            "    created_ids.append(sphere)"
        )
    raise ValueError(f"Unsupported Rhino template: {shape}")