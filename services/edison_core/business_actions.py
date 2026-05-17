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
_MODEL_REQUEST_HINTS = (
    "3d", "cad", "rhino", "blender", "solidworks", "mesh", "prototype",
    "model", "shape", "figure", "solid", "stl", "object", "sculpture",
    "printable", "3d print",
)
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

# Words in an extracted shape description that suggest a non-physical, non-3D intent.
_NON_PHYSICAL_NOUNS = frozenset([
    "plan", "list", "summary", "report", "document", "email", "message", "note",
    "text", "post", "caption", "copy", "flyer", "logo", "image", "photo", "video",
    "campaign", "strategy", "analysis", "invoice", "proposal", "description",
    "script", "story", "blog", "tweet", "ad", "slogan", "tagline",
])

# Module-level injectable LLM reference used for free-form Rhino script codegen.
_rhino_llm_ref: list = []


def set_rhino_llm(llm: Any) -> None:
    """Inject the active LLM so EDISON can generate Rhino Python for arbitrary shape requests."""
    _rhino_llm_ref.clear()
    if llm is not None:
        _rhino_llm_ref.append(llm)


_RHINO_CODEGEN_SYSTEM_PROMPT = (
    "You are a Rhinoceros 3D Python scripting expert targeting IronPython 2.7 inside Rhino 6/7/8.\n"
    "Generate ONLY the body statements of a rhinoscriptsyntax Python script that creates the requested 3D model.\n\n"
    "STRICT RULES:\n"
    "- Use ONLY `rhinoscriptsyntax` (already imported as `rs`). No other imports.\n"
    "- Do NOT add import statements, function definitions, try/except blocks, or class definitions.\n"
    "- Do NOT use list comprehensions, f-strings, walrus operator, or any Python 3-only syntax.\n"
    "- Use only plain for-loops and string concatenation for compatibility with IronPython 2.7.\n"
    "- Append every FINAL geometry object ID to the list named `created_ids`.\n"
    "- Append all temporary/construction geometry to the list named `construction_ids`.\n"
    "- Call `_show_progress('EDISON: <step>', <weight>)` for each major step. Total weights ~1.0.\n"
    "- Units are millimetres. Keep the model between 80mm and 250mm in its largest dimension.\n"
    "- Prefer rs.AddBox, rs.AddCylinder, rs.AddSphere, rs.AddTorus for primitive parts.\n"
    "- For revolution profiles: start X must be >= 3 (never on the axis). \n"
    "  Always call rs.CapPlanarHoles(srf) after rs.AddRevSrf.\n"
    "- Build compound objects part by part; store each part id then join with rs.BooleanUnion.\n"
    "  IMPORTANT: after BooleanUnion delete the source parts from created_ids; add the union result.\n"
    "- ALWAYS check that each rs.Add* call succeeded (result is not None/False) before using it.\n"
    "  If a call fails, skip that part silently (do not raise).\n"
    "- Keep it simple: 3-6 primitives max. Do not attempt fine surface modelling.\n"
    "- Output ONLY plain Python statements. No markdown fences, no prose, no inline comments.\n"
)


def _extract_shape_description(lowered: str, original: str) -> str:
    """Return the noun phrase describing the 3D shape, preserving original casing."""
    verb_alt = "|".join(re.escape(v) for v in _MODEL_REQUEST_VERBS)
    # Pattern: <verb> [article/me/us] <noun_phrase> [stop words]
    stop = r"(?:\s+(?:in rhino|in 3d|in cad|in blender|as a model|as a 3d|for me|for us|on the|with rhino)\b|$)"
    m = re.search(
        rf"\b(?:{verb_alt})\b\s+(?:(?:a|an|the|me|us)\s+)?(.+?){stop}",
        lowered,
    )
    raw = (m.group(1) if m else re.sub(rf".*\b(?:{verb_alt})\b\s*", "", lowered, count=1)).strip()
    # Restore original casing
    idx = lowered.find(raw)
    return original[idx: idx + len(raw)].strip() if idx >= 0 and raw else raw


def _strip_to_valid_python(code: str) -> str:
    """Remove trailing lines that make the code unparseable (e.g. LLM truncation mid-statement)."""
    lines = code.splitlines()
    # Walk backwards removing lines until the remainder compiles
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end])
        try:
            compile(candidate, "<generated>", "exec")
            return candidate
        except SyntaxError:
            continue
    return ""


def _fstr_body_to_concat(quote: str, body: str) -> str:
    """Turn the interior of an f-string into IronPython 2.7 str-concat."""
    parts = re.split(r"\{([^}]+)\}", body)
    out = []
    for idx, part in enumerate(parts):
        if idx % 2 == 0:
            if part:
                out.append(quote + part + quote)
        else:
            out.append("str(" + part.strip() + ")")
    return " + ".join(out) if out else quote + quote


def _py2_sanitize(code: str) -> str:
    """Convert Python 3-only syntax to IronPython 2.7-compatible equivalents."""
    # f"..."  double-quoted f-strings
    code = re.sub(
        r'f"((?:[^\\"]|\\.)*)"',
        lambda m: _fstr_body_to_concat('"', m.group(1)),
        code,
    )
    # f'...'  single-quoted f-strings
    code = re.sub(
        r"f'((?:[^\\']|\\.)*)'",
        lambda m: _fstr_body_to_concat("'", m.group(1)),
        code,
    )
    # Walrus operator  x := expr  ->  x = expr
    code = re.sub(r"(\w+)\s*:=", r"\1 =", code)
    # Bare type annotations  var: SomeType = ...  ->  var = ...
    code = re.sub(
        r"^(\s*\w+)\s*:[A-Za-z_][\w\[\], ]*\s*=",
        r"\1 =",
        code,
        flags=re.MULTILINE,
    )
    return code

def _generate_rhino_body_via_llm(description: str) -> Optional[str]:
    """Ask the injected LLM to write rhinoscriptsyntax Python for the given description."""
    if not _rhino_llm_ref:
        return None
    llm = _rhino_llm_ref[0]
    prompt = (
        f"{_RHINO_CODEGEN_SYSTEM_PROMPT}\n"
        f"Create a 3D Rhino model of: {description}\n\n"
        "Code:\n"
    )
    try:
        result = llm(
            prompt,
            max_tokens=3000,
            temperature=0.15,
            stop=["```", "\nimport ", "\nclass "],
            echo=False,
        )
        code = result["choices"][0]["text"].strip()
        # Strip markdown fences if the model added them anyway
        code = re.sub(r"^```[a-zA-Z]*\n?", "", code)
        code = re.sub(r"\n?```$", "", code).strip()
        if not code:
            return None
        # Remove any try/except/finally wrappers the LLM may have added
        code = re.sub(r"^try:\s*\n", "", code)
        code = re.sub(r"\nexcept[^\n]*:\n(?: +[^\n]*\n)*", "\n", code)
        code = re.sub(r"\nfinally:[^\n]*\n(?: +[^\n]*\n)*", "\n", code)
        # Convert Python 3-only syntax to IronPython 2.7 equivalents (f-strings, walrus, annotations)
        code = _py2_sanitize(code)
        # Strip trailing incomplete lines caused by token-limit truncation
        code = _strip_to_valid_python(code)
        return code or None
    except Exception:
        return None


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
    model_request = _parse_model_request(lowered, text, node_manager is not None)
    if model_request is None:
        return None

    shape = model_request["shape"]
    needs_llm = model_request.get("needs_llm", False)

    if needs_llm:
        # Free-form LLM codegen path
        if not _rhino_llm_ref:
            supported = ", ".join(sorted(_PROCEDURAL_MODEL_ALIASES.keys()))
            return {
                "response": (
                    f"I understand you want a 3D model of \"{shape}\", but the AI code generator "
                    f"is not available yet (LLM not loaded). I have built-in templates for: {supported}."
                ),
                "mode_used": "business",
                "business_action": {"type": "node_model_request", "ok": False, "reason": "llm_unavailable"},
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

        custom_body = _generate_rhino_body_via_llm(shape)
        if not custom_body:
            return {
                "response": (
                    f"I tried to generate Rhino code for \"{shape}\" but the model produced no output. "
                    f"Try rephrasing or ask for a simpler shape."
                ),
                "mode_used": "business",
                "business_action": {"type": "node_model_request", "ok": False, "reason": "codegen_failed"},
            }

        safe_name = _slugify_for_filename(shape)[:32] or "custom-model"
        import time as _time
        _ts = str(int(_time.time()))[-6:]
        filename = f"{safe_name}-{_ts}.3dm"
        output_hint = f"~/.edison/generated_models/{filename}"
        result_hint = f"~/.edison/generated_models/{Path(filename).stem}.result.json"
        full_script = _build_rhino_script(safe_name, filename, shape_body=custom_body)
        # Validate the assembled script compiles before sending to the node
        try:
            compile(full_script, "<validation>", "exec")
        except SyntaxError as _se:
            return {
                "response": (
                    f"The AI generated a Rhino script for \"{shape}\" but it contained a syntax error "
                    f"({_se}). Try asking again — the model may produce a cleaner result on a retry."
                ),
                "mode_used": "business",
                "business_action": {"type": "node_model_request", "ok": False, "reason": "codegen_syntax_error"},
            }
        payload = {
            "script_content": full_script,
            "output_paths": [output_hint],
            "result_paths": [result_hint],
            "timeout_seconds": 180,
        }

        command_result = _dispatch_node_model_request(
            text=text,
            node_manager=node_manager,
            target_node=target_node,
            payload=payload,
            preferred_software=["rhino", "grasshopper"],
        )

        if command_result.get("status") == "queued":
            queued_task = command_result.get("task") or {}
            return {
                "response": (
                    f"Generated custom Rhino script for \"{shape}\" and sent it to "
                    f"{target_node.get('name', target_node['id'])}. "
                    f"Task id: {queued_task.get('id', 'pending')}. Output: {output_hint}."
                ),
                "mode_used": "business",
                "business_action": {
                    "type": "node_model_request",
                    "ok": True,
                    "status": "queued",
                    "shape": shape,
                    "node": {"id": target_node["id"], "name": target_node.get("name")},
                    "task": queued_task,
                    "output_paths": [output_hint],
                    "result_paths": [result_hint],
                },
            }

        if not command_result.get("ok"):
            error_text = command_result.get("error") or "unknown transport error"
            return {
                "response": f"Script generated but could not reach the node: {error_text}.",
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
                    f"Sent the \"{shape}\" script to {target_node.get('name', target_node['id'])}, "
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
                f"Rhino executed the custom \"{shape}\" model on "
                f"{target_node.get('name', target_node['id'])}. Output: {', '.join(output_paths)}."
            ),
            "mode_used": "business",
            "business_action": {
                "type": "node_model_request",
                "ok": True,
                "shape": shape,
                "node": {"id": target_node["id"], "name": target_node.get("name")},
                "output_paths": output_paths,
                "node_response": node_response,
            },
        }

    # ─── Preset/template path ──────────────────────────────────────────────────
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

    import time as _t2; _ts2 = str(int(_t2.time()))[-6:]
    filename = f"{model_request['shape']}-{_slugify_for_filename(text)[:24] or model_request['shape']}-{_ts2}.3dm"
    output_hint = f"~/.edison/generated_models/{filename}"
    result_hint = f"~/.edison/generated_models/{Path(filename).stem}.result.json"
    payload = {
        "script_content": _build_rhino_script(model_request["shape"], filename),
        "output_paths": [output_hint],
        "result_paths": [result_hint],
        "timeout_seconds": 120,
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
                "result_paths": [result_hint],
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


def _parse_model_request(
    lowered: str,
    original: str = "",
    has_node_manager: bool = False,
) -> Optional[Dict[str, Any]]:
    if not any(verb in lowered for verb in _MODEL_REQUEST_VERBS):
        return None
    # Check preset aliases first — works regardless of hint keywords
    for shape, aliases in _PROCEDURAL_MODEL_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                return {"shape": shape, "matched_alias": alias, "needs_llm": False}
    # Require a 3D hint OR an active node manager for free-form codegen
    has_hint = any(hint in lowered for hint in _MODEL_REQUEST_HINTS)
    if not has_hint and not has_node_manager:
        return None
    # Extract the noun phrase the user wants modelled
    description = _extract_shape_description(lowered, original or lowered)
    if not description:
        return None
    # Reject clearly non-physical intents
    if any(skip in description.lower().split() for skip in _NON_PHYSICAL_NOUNS):
        return None
    return {"shape": description, "matched_alias": None, "needs_llm": True}


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


def _build_rhino_script(shape: str, filename: str, shape_body: Optional[str] = None) -> str:
    if shape_body is None:
        shape_body = _build_rhino_shape_body(shape).replace("\n", "\n    ")
    else:
        shape_body = shape_body.replace("\n", "\n    ")
    layer_name = f"EDISON_{filename.rsplit('.', 1)[0]}"
    result_filename = f"{Path(filename).stem}.result.json"
    return (
        "import json\n"
        "import os\n"
        "import time\n"
        "import traceback\n"
        "import rhinoscriptsyntax as rs\n\n"
        f"output_file = os.path.join(os.path.expanduser('~'), '.edison', 'generated_models', '{filename}')\n"
        f"result_file = os.path.join(os.path.expanduser('~'), '.edison', 'generated_models', '{result_filename}')\n"
        "output_dir = os.path.dirname(output_file)\n"
        "if not os.path.isdir(output_dir):\n"
        "    os.makedirs(output_dir)\n"
        "for artifact_path in [output_file, result_file]:\n"
        "    try:\n"
        "        if os.path.exists(artifact_path):\n"
        "            os.remove(artifact_path)\n"
        "    except Exception:\n"
        "        pass\n\n"
        "def _write_result(ok, message, obj_ids=None):\n"
        "    id_list = []\n"
        "    for oid in (obj_ids or []):\n"
        "        id_list.append(str(oid))\n"
        "    payload = {\n"
        "        'ok': bool(ok),\n"
        "        'message': message,\n"
        "        'output_file': output_file,\n"
        "        'result_file': result_file,\n"
        "        'created_ids': id_list,\n"
        "    }\n"
        "    with open(result_file, 'w') as handle:\n"
        "        json.dump(payload, handle, indent=2)\n\n"
        "def _show_progress(msg, pause=0.35):\n"
        "    try:\n"
        "        print(msg)\n"
        "    except Exception:\n"
        "        pass\n"
        "    try:\n"
        "        rs.Redraw()\n"
        "    except Exception:\n"
        "        pass\n"
        "    try:\n"
        "        time.sleep(pause)\n"
        "    except Exception:\n"
        "        pass\n\n"
        "def _focus_model():\n"
        "    try:\n"
        "        rs.ViewDisplayMode(rs.CurrentView(), 'Shaded')\n"
        "    except Exception:\n"
        "        pass\n"
        "    try:\n"
        "        rs.ZoomSelected()\n"
        "    except Exception:\n"
        "        try:\n"
        "            rs.ZoomExtents()\n"
        "        except Exception:\n"
        "            pass\n"
        "    try:\n"
        "        rs.Redraw()\n"
        "    except Exception:\n"
        "        pass\n\n"
        "previous_layer = None\n"
        "created_ids = []\n"
        "construction_ids = []\n"
        "try:\n"
        "    rs.EnableRedraw(True)\n"
        "    try:\n"
        "        rs.CurrentView('Perspective')\n"
        "    except Exception:\n"
        "        pass\n"
        "    previous_layer = rs.CurrentLayer()\n"
        f"    layer_name = '{layer_name}'\n"
        "    if not rs.IsLayer(layer_name):\n"
        "        rs.AddLayer(layer_name)\n"
        "    rs.CurrentLayer(layer_name)\n"
        "    _show_progress('EDISON: preparing Rhino workspace', 0.2)\n"
        f"    {shape_body}\n"
        "    if not created_ids:\n"
        "        raise Exception('EDISON did not create any geometry.')\n"
        "    for object_id in list(construction_ids):\n"
        "        try:\n"
        "            if rs.IsObject(object_id):\n"
        "                rs.DeleteObject(object_id)\n"
        "        except Exception:\n"
        "            pass\n"
        "    _show_progress('EDISON: cleaned construction geometry', 0.2)\n"
        "    rs.UnselectAllObjects()\n"
        "    for object_id in created_ids:\n"
        "        try:\n"
        "            rs.SelectObject(object_id)\n"
        "        except Exception:\n"
        "            pass\n"
        "    _focus_model()\n"
        "    _show_progress('EDISON: focused generated model', 0.45)\n"
        "    export_ok = rs.Command('-_Export \"{}\" _Enter'.format(output_file), False)\n"
        "    if not export_ok:\n"
        "        raise Exception('Rhino export command failed.')\n"
        "    deadline = time.time() + 15.0\n"
        "    while time.time() < deadline and not os.path.exists(output_file):\n"
        "        time.sleep(0.25)\n"
        "    if not os.path.exists(output_file):\n"
        "        raise Exception('Rhino export did not produce the expected file.')\n"
        "    _show_progress('EDISON: export complete', 0.2)\n"
        "    _write_result(True, 'Rhino model created successfully.', created_ids)\n"
        "except Exception:\n"
        "    _write_result(False, traceback.format_exc(), created_ids)\n"
        "    raise\n"
        "finally:\n"
        "    try:\n"
        "        rs.UnselectAllObjects()\n"
        "        if previous_layer and rs.IsLayer(previous_layer):\n"
        "            rs.CurrentLayer(previous_layer)\n"
        "        rs.Redraw()\n"
        "    except Exception:\n"
        "        pass\n"
    )


def _build_rhino_shape_body(shape: str) -> str:
    if shape == "vase":
        return (
            "axis = rs.AddLine((0, 0, 0), (0, 0, 180))\n"
            "if not axis:\n"
            "    raise Exception('Failed to build the vase axis.')\n"
            "construction_ids.append(axis)\n"
            "_show_progress('EDISON: created revolve axis', 0.25)\n"
            "profile = rs.AddInterpCurve([(3, 0, 0), (32, 0, 0), (42, 0, 18), (54, 0, 70), (48, 0, 128), (26, 0, 176), (20, 0, 180)], degree=3)\n"
            "if not profile:\n"
            "    raise Exception('Failed to build the vase profile.')\n"
            "construction_ids.append(profile)\n"
            "_show_progress('EDISON: drew vase profile', 0.35)\n"
            "surface = rs.AddRevSrf(profile, axis)\n"
            "if not surface:\n"
            "    raise Exception('Failed to revolve the vase surface.')\n"
            "rs.CapPlanarHoles(surface)\n"
            "created_ids.append(surface)\n"
            "_show_progress('EDISON: revolved vase surface', 0.55)"
        )
    if shape == "planter":
        return (
            "axis = rs.AddLine((0, 0, 0), (0, 0, 110))\n"
            "if not axis:\n"
            "    raise Exception('Failed to build the planter axis.')\n"
            "construction_ids.append(axis)\n"
            "_show_progress('EDISON: created planter axis', 0.25)\n"
            "profile = rs.AddInterpCurve([(4, 0, 0), (50, 0, 0), (56, 0, 16), (60, 0, 70), (52, 0, 108), (40, 0, 110)], degree=3)\n"
            "if not profile:\n"
            "    raise Exception('Failed to build the planter profile.')\n"
            "construction_ids.append(profile)\n"
            "_show_progress('EDISON: drew planter profile', 0.35)\n"
            "surface = rs.AddRevSrf(profile, axis)\n"
            "if not surface:\n"
            "    raise Exception('Failed to revolve the planter surface.')\n"
            "rs.CapPlanarHoles(surface)\n"
            "created_ids.append(surface)\n"
            "_show_progress('EDISON: revolved planter surface', 0.55)"
        )
    if shape == "bowl":
        return (
            "axis = rs.AddLine((0, 0, 0), (0, 0, 80))\n"
            "if not axis:\n"
            "    raise Exception('Failed to build the bowl axis.')\n"
            "construction_ids.append(axis)\n"
            "_show_progress('EDISON: created bowl axis', 0.25)\n"
            "profile = rs.AddInterpCurve([(4, 0, 0), (64, 0, 0), (76, 0, 18), (86, 0, 42), (72, 0, 76), (50, 0, 80)], degree=3)\n"
            "if not profile:\n"
            "    raise Exception('Failed to build the bowl profile.')\n"
            "construction_ids.append(profile)\n"
            "_show_progress('EDISON: drew bowl profile', 0.35)\n"
            "surface = rs.AddRevSrf(profile, axis)\n"
            "if not surface:\n"
            "    raise Exception('Failed to revolve the bowl surface.')\n"
            "rs.CapPlanarHoles(surface)\n"
            "created_ids.append(surface)\n"
            "_show_progress('EDISON: revolved bowl surface', 0.55)"
        )
    if shape == "cup":
        return (
            "axis = rs.AddLine((0, 0, 0), (0, 0, 115))\n"
            "if not axis:\n"
            "    raise Exception('Failed to build the cup axis.')\n"
            "construction_ids.append(axis)\n"
            "_show_progress('EDISON: created cup axis', 0.25)\n"
            "profile = rs.AddInterpCurve([(4, 0, 0), (34, 0, 0), (42, 0, 20), (46, 0, 88), (38, 0, 114), (30, 0, 115)], degree=3)\n"
            "if not profile:\n"
            "    raise Exception('Failed to build the cup profile.')\n"
            "construction_ids.append(profile)\n"
            "_show_progress('EDISON: drew cup profile', 0.35)\n"
            "surface = rs.AddRevSrf(profile, axis)\n"
            "if not surface:\n"
            "    raise Exception('Failed to revolve the cup surface.')\n"
            "rs.CapPlanarHoles(surface)\n"
            "created_ids.append(surface)\n"
            "_show_progress('EDISON: revolved cup surface', 0.55)"
        )
    if shape == "box":
        return (
            "box = rs.AddBox([(0, 0, 0), (80, 0, 0), (80, 60, 0), (0, 60, 0), (0, 0, 50), (80, 0, 50), (80, 60, 50), (0, 60, 50)])\n"
            "if not box:\n"
            "    raise Exception('Failed to create the box.')\n"
            "created_ids.append(box)\n"
            "_show_progress('EDISON: created box solid', 0.45)"
        )
    if shape == "cylinder":
        return (
            "cylinder = rs.AddCylinder((0, 0, 0), 120, 40)\n"
            "if not cylinder:\n"
            "    raise Exception('Failed to create the cylinder.')\n"
            "created_ids.append(cylinder)\n"
            "_show_progress('EDISON: created cylinder solid', 0.45)"
        )
    if shape == "cone":
        return (
            "cone = rs.AddCone((0, 0, 0), 120, 48)\n"
            "if not cone:\n"
            "    raise Exception('Failed to create the cone.')\n"
            "created_ids.append(cone)\n"
            "_show_progress('EDISON: created cone solid', 0.45)"
        )
    if shape == "sphere":
        return (
            "sphere = rs.AddSphere((0, 0, 55), 55)\n"
            "if not sphere:\n"
            "    raise Exception('Failed to create the sphere.')\n"
            "created_ids.append(sphere)\n"
            "_show_progress('EDISON: created sphere solid', 0.45)"
        )
    raise ValueError(f"Unsupported Rhino template: {shape}")