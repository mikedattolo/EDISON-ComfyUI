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


def execute_business_action(
    message: str,
    repo_root: Path,
    config: Dict[str, Any],
    branding_store: BrandingClientStore,
    project_manager: ProjectWorkspaceManager,
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

    if any(phrase in lowered for phrase in ["what models", "available models", "model catalog", "which model should", "what model should", "product image model", "img2img model", "image to image model", "video model", "music model", "3d model", "mesh model", "text to 3d", "image to 3d"]):
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