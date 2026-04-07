"""
tool_runtime.py — Tool registry, validation, execution, and structured tool loop.

Extracts the TOOL_REGISTRY, validation helpers, and run_structured_tool_loop()
from app.py into a standalone, testable module.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Awaitable

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────
TOOL_LOOP_MAX_STEPS = 12          # bumped from 8 to allow deeper multi-step tasks
TOOL_CALL_TIMEOUT_SEC = 45
TOOL_RESULT_CHAR_LIMIT = 2000


# ── Tool registry ────────────────────────────────────────────────────
# The registry is module-level so it can be extended at startup by plugins.
TOOL_REGISTRY: Dict[str, dict] = {
    "web_search": {
        "args": {
            "query": {"type": str, "required": True},
            "max_results": {"type": int, "required": False, "default": 5},
        }
    },
    "rag_search": {
        "args": {
            "query": {"type": str, "required": True},
            "limit": {"type": int, "required": False, "default": 3},
            "global": {"type": bool, "required": False, "default": False},
        }
    },
    "knowledge_search": {
        "args": {
            "query": {"type": str, "required": True},
            "limit": {"type": int, "required": False, "default": 4},
            "include_web_search": {"type": bool, "required": False, "default": False},
        }
    },
    "generate_image": {
        "args": {
            "prompt": {"type": str, "required": True},
            "width": {"type": int, "required": False, "default": 1024},
            "height": {"type": int, "required": False, "default": 1024},
            "steps": {"type": int, "required": False, "default": 20},
            "guidance_scale": {"type": float, "required": False, "default": 3.5},
        }
    },
    "system_stats": {"args": {}},
    "execute_python": {
        "args": {
            "code": {"type": str, "required": True},
            "packages": {"type": str, "required": False, "default": ""},
            "description": {"type": str, "required": False, "default": ""},
        }
    },
    "read_file": {
        "args": {"path": {"type": str, "required": True}},
    },
    "list_files": {
        "args": {
            "directory": {"type": str, "required": False, "default": "/opt/edison/gallery"},
        }
    },
    "analyze_csv": {
        "args": {
            "file_path": {"type": str, "required": True},
            "operation": {"type": str, "required": True},
        }
    },
    "get_current_time": {
        "args": {"timezone": {"type": str, "required": False, "default": "local"}},
    },
    "get_weather": {
        "args": {"location": {"type": str, "required": True}},
    },
    "get_news": {
        "args": {
            "topic": {"type": str, "required": False, "default": "top news today"},
            "max_results": {"type": int, "required": False, "default": 8},
        }
    },
    "generate_music": {
        "args": {
            "prompt": {"type": str, "required": True},
            "genre": {"type": str, "required": False, "default": ""},
            "mood": {"type": str, "required": False, "default": ""},
            "duration": {"type": int, "required": False, "default": 15},
        }
    },
    "call_external_api": {
        "args": {
            "connector": {"type": str, "required": True},
            "path": {"type": str, "required": False, "default": "/"},
            "method": {"type": str, "required": False, "default": "GET"},
            "body": {"type": str, "required": False, "default": ""},
        }
    },
    "open_sandbox_browser": {
        "args": {"url": {"type": str, "required": True}},
    },
    "browser.observe": {
        "args": {"session_id": {"type": str, "required": True}},
    },
    "browser.get_text": {
        "args": {"session_id": {"type": str, "required": True}},
    },
    "browser.navigate": {
        "args": {
            "session_id": {"type": str, "required": True},
            "url": {"type": str, "required": True},
        }
    },
    "browser.click": {
        "args": {
            "session_id": {"type": str, "required": True},
            "x": {"type": int, "required": True},
            "y": {"type": int, "required": True},
            "button": {"type": str, "required": False, "default": "left"},
            "click_count": {"type": int, "required": False, "default": 1},
        }
    },
    "browser.type": {
        "args": {
            "session_id": {"type": str, "required": True},
            "text": {"type": str, "required": True},
            "delay_ms": {"type": int, "required": False, "default": 10},
        }
    },
    "browser.find_element": {
        "args": {
            "session_id": {"type": str, "required": True},
            "selector": {"type": str, "required": True},
        }
    },
    "browser.click_by_text": {
        "args": {
            "session_id": {"type": str, "required": True},
            "text": {"type": str, "required": True},
        }
    },
    "browser.fill_form": {
        "args": {
            "session_id": {"type": str, "required": True},
            "fields": {"type": dict, "required": True},
        }
    },
    "browser.press": {
        "args": {
            "session_id": {"type": str, "required": True},
            "key": {"type": str, "required": True},
        }
    },
    "browser.scroll": {
        "args": {
            "session_id": {"type": str, "required": True},
            "delta_x": {"type": int, "required": False, "default": 0},
            "delta_y": {"type": int, "required": True},
        }
    },
    "browser.create_session": {
        "args": {
            "url": {"type": str, "required": True},
            "width": {"type": int, "required": False, "default": 1280},
            "height": {"type": int, "required": False, "default": 800},
        }
    },
    "write_file": {
        "args": {
            "path": {"type": str, "required": True},
            "content": {"type": str, "required": True},
        }
    },
    "summarize_url": {
        "args": {"url": {"type": str, "required": True}},
    },
}

# Additional printer tools (dynamically registered if printing is enabled)
PRINTER_TOOLS = {
    "list_printers": {"args": {}},
    "get_printer_status": {
        "args": {"printer_id": {"type": str, "required": True}},
    },
    "send_3d_print": {
        "args": {
            "printer_id": {"type": str, "required": True},
            "file_path": {"type": str, "required": True},
        }
    },
}

# Task management tools (always available)
TASK_TOOLS = {
    "create_task": {
        "args": {
            "objective": {"type": str, "required": True},
            "chat_id": {"type": str, "required": False, "default": ""},
        }
    },
    "list_tasks": {
        "args": {
            "status": {"type": str, "required": False, "default": ""},
        }
    },
    "complete_task": {
        "args": {
            "task_id": {"type": str, "required": True},
        }
    },
}

# ── Domain / business tools ───────────────────────────────────────────

BRANDING_TOOLS = {
    "generate_brand_package": {
        "args": {
            "business_name": {"type": str, "required": True},
            "industry": {"type": str, "required": False, "default": ""},
            "audience": {"type": str, "required": False, "default": ""},
            "tone": {"type": str, "required": False, "default": "confident"},
            "client_id": {"type": str, "required": False, "default": ""},
            "project_id": {"type": str, "required": False, "default": ""},
        }
    },
    "generate_marketing_copy": {
        "args": {
            "business_name": {"type": str, "required": True},
            "industry": {"type": str, "required": False, "default": ""},
            "audience": {"type": str, "required": False, "default": ""},
            "tone": {"type": str, "required": False, "default": "confident"},
            "copy_types": {"type": str, "required": False, "default": "ad_copy,social_captions"},
            "client_id": {"type": str, "required": False, "default": ""},
            "project_id": {"type": str, "required": False, "default": ""},
        }
    },
    "create_branding_client": {
        "args": {
            "name": {"type": str, "required": True},
            "industry": {"type": str, "required": False, "default": ""},
            "contact_person": {"type": str, "required": False, "default": ""},
            "email": {"type": str, "required": False, "default": ""},
            "phone": {"type": str, "required": False, "default": ""},
            "website": {"type": str, "required": False, "default": ""},
            "notes": {"type": str, "required": False, "default": ""},
        }
    },
    "list_branding_clients": {
        "args": {}
    },
}

VIDEO_TOOLS = {
    "generate_video": {
        "args": {
            "prompt": {"type": str, "required": True},
            "duration": {"type": int, "required": False, "default": 6},
            "width": {"type": int, "required": False, "default": 720},
            "height": {"type": int, "required": False, "default": 480},
        }
    },
}

PROJECT_TOOLS = {
    "create_project": {
        "args": {
            "name": {"type": str, "required": True},
            "description": {"type": str, "required": False, "default": ""},
            "client_id": {"type": str, "required": False, "default": ""},
            "service_types": {"type": str, "required": False, "default": ""},
        }
    },
    "list_projects": {
        "args": {
            "status": {"type": str, "required": False, "default": ""},
        }
    },
}

FABRICATION_TOOLS = {
    "slice_model": {
        "args": {
            "file_path": {"type": str, "required": True},
            "layer_height": {"type": float, "required": False, "default": 0.2},
            "infill": {"type": int, "required": False, "default": 20},
            "supports": {"type": bool, "required": False, "default": False},
        }
    },
}

SOCIAL_TOOLS = {
    "create_social_post": {
        "args": {
            "platform": {"type": str, "required": True},
            "caption": {"type": str, "required": True},
            "post_type": {"type": str, "required": False, "default": "image"},
            "hashtags": {"type": list, "required": False, "default": []},
            "campaign_name": {"type": str, "required": False},
        }
    },
    "schedule_social_post": {
        "args": {
            "post_id": {"type": str, "required": True},
            "scheduled_at": {"type": str, "required": True},
            "timezone": {"type": str, "required": False, "default": "UTC"},
        }
    },
    "list_social_posts": {
        "args": {
            "platform": {"type": str, "required": False},
            "status": {"type": str, "required": False},
            "campaign": {"type": str, "required": False},
        }
    },
}

# Auto-register all domain tools
TOOL_REGISTRY.update(TASK_TOOLS)
TOOL_REGISTRY.update(BRANDING_TOOLS)
TOOL_REGISTRY.update(VIDEO_TOOLS)
TOOL_REGISTRY.update(PROJECT_TOOLS)
TOOL_REGISTRY.update(FABRICATION_TOOLS)
TOOL_REGISTRY.update(SOCIAL_TOOLS)


def register_tool(name: str, schema: dict) -> None:
    """Register a new tool in the global registry."""
    TOOL_REGISTRY[name] = schema


def register_printer_tools() -> None:
    """Register printer tools into the global registry."""
    TOOL_REGISTRY.update(PRINTER_TOOLS)


# ── Validation ───────────────────────────────────────────────────────
def _coerce_int(val) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def validate_and_normalize_tool_call(
    payload: dict,
    registry: Optional[Dict[str, dict]] = None,
) -> tuple:
    """
    Validate tool call JSON strictly against registry schema.
    Returns (ok: bool, error: str|None, tool_name: str|None, normalized_args: dict|None).
    """
    reg = registry or TOOL_REGISTRY
    if not isinstance(payload, dict):
        return False, "Payload must be an object", None, None
    if set(payload.keys()) != {"tool", "args"}:
        return False, "Payload must contain exactly 'tool' and 'args' keys", None, None

    tool_name = payload.get("tool")
    args = payload.get("args")

    if not isinstance(tool_name, str):
        return False, "'tool' must be a string", None, None
    if tool_name not in reg:
        return False, f"Unknown tool '{tool_name}'", tool_name, None
    if not isinstance(args, dict):
        return False, "'args' must be an object", tool_name, None

    schema = reg[tool_name]["args"]
    normalized = {}
    for arg_name, meta in schema.items():
        if arg_name in args:
            value = args[arg_name]
            expected_type = meta["type"]
            if expected_type is int and not _coerce_int(value):
                return False, f"{arg_name} must be int", tool_name, None
            if expected_type is float and not isinstance(value, (int, float)):
                return False, f"{arg_name} must be float", tool_name, None
            if expected_type is bool and not isinstance(value, bool):
                return False, f"{arg_name} must be bool", tool_name, None
            if expected_type is str and not isinstance(value, str):
                return False, f"{arg_name} must be string", tool_name, None
            if expected_type is dict and not isinstance(value, dict):
                return False, f"{arg_name} must be object", tool_name, None
            if expected_type is list and not isinstance(value, list):
                return False, f"{arg_name} must be array", tool_name, None
            normalized[arg_name] = value
        else:
            if meta.get("required"):
                return False, f"Missing required arg '{arg_name}'", tool_name, None
            if "default" in meta:
                normalized[arg_name] = meta["default"]

    return True, None, tool_name, normalized


def extract_tool_payload_from_text(raw_output: str) -> Optional[dict]:
    """
    Extract a JSON object for tool calls from noisy LLM output.
    Handles pure JSON, markdown fences, prose-wrapped JSON, trailing commas,
    and single-quoted JSON (common LLM quirks).
    """
    if not isinstance(raw_output, str):
        return None
    text = raw_output.strip()
    if not text:
        return None

    def _try_parse(s: str) -> Optional[dict]:
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
        cleaned = re.sub(r",\s*([}\]])", r"\1", s)
        if cleaned != s:
            try:
                obj = json.loads(cleaned)
                return obj if isinstance(obj, dict) else None
            except Exception:
                pass
        try:
            import ast
            obj = ast.literal_eval(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
        return None

    result = _try_parse(text)
    if result:
        return result

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if fenced:
        result = _try_parse(fenced.group(1))
        if result:
            return result

    for start in (i for i, ch in enumerate(text) if ch == "{"):
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    result = _try_parse(candidate)
                    if result:
                        return result
                    break
    return None


# ── Tool event dataclass ─────────────────────────────────────────────
@dataclass
class ToolEvent:
    """One step in a tool loop execution."""
    step: int
    tool_name: str
    args: dict
    result_summary: str
    ok: bool
    elapsed_sec: float = 0.0
    raw_result: Optional[dict] = None


@dataclass
class ToolLoopResult:
    """Final output of a structured tool loop run."""
    final_answer: str
    events: List[ToolEvent] = field(default_factory=list)
    steps_used: int = 0
    total_elapsed_sec: float = 0.0
    tools_used: List[str] = field(default_factory=list)
    cancelled: bool = False


# ── Tool loop ────────────────────────────────────────────────────────
async def run_tool_loop(
    *,
    call_llm: Callable[[str], Awaitable[str]],
    execute_tool: Callable[[str, dict, Optional[str]], Awaitable[dict]],
    summarize_tool_result: Callable[[str, dict], str],
    user_message: str,
    context_note: str,
    model_name: str,
    chat_id: Optional[str] = None,
    request_id: Optional[str] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    emit_status: Optional[Callable[[str], Any]] = None,
    max_steps: int = TOOL_LOOP_MAX_STEPS,
    registry: Optional[Dict[str, dict]] = None,
) -> ToolLoopResult:
    """
    Shared structured tool loop.

    This is a pure orchestration function. It does NOT directly access
    any LLM or tool implementation — callers inject those via callbacks.
    This makes it testable and decoupled from app.py globals.
    """
    reg = registry or TOOL_REGISTRY
    result = ToolLoopResult()
    loop_start = time.time()

    tool_history: list = []
    final_answer: Optional[str] = None
    step = 0

    tool_descriptions = "\n".join(
        f"- {name}: args={json.dumps({k: v['type'].__name__ for k, v in spec['args'].items()})}"
        for name, spec in reg.items()
    )

    system_instructions = (
        "You are EDISON, an AI assistant with tool access.\n"
        "When you need to call a tool, respond with ONLY a JSON object:\n"
        '{"tool": "<name>", "args": {<arguments>}}\n'
        "When you have the final answer for the user, respond with plain text (no JSON).\n"
        f"\nAvailable tools:\n{tool_descriptions}\n"
    )

    while step < max_steps and final_answer is None:
        if is_cancelled and is_cancelled():
            result.cancelled = True
            result.final_answer = "Request was cancelled."
            break

        step += 1
        if emit_status:
            emit_status(f"Thinking… (step {step}/{max_steps})")

        # Build prompt with history
        history_block = ""
        if tool_history:
            history_block = "\n\nPrevious tool results:\n" + "\n".join(tool_history)

        prompt = (
            f"{system_instructions}\n"
            f"{context_note}\n"
            f"User request: {user_message}\n"
            f"{history_block}\n\n"
            f"Step {step} of {max_steps}. Provide JSON to call a tool or final answer now.\n"
        )

        raw_output = await call_llm(prompt)
        if not raw_output or not raw_output.strip():
            continue

        # Try to extract tool call
        payload = extract_tool_payload_from_text(raw_output)

        if payload is None:
            # No valid tool call found → treat as final answer
            final_answer = raw_output.strip()
            break

        ok, err, tool_name, normalized_args = validate_and_normalize_tool_call(payload, reg)

        if not ok:
            # Give the LLM one chance to correct
            correction_prompt = (
                f"{system_instructions}\n"
                f"Your previous response had an error: {err}\n"
                f"Available tools: {', '.join(reg.keys())}. "
                f"Please try again with valid JSON or provide a final answer.\n"
            )
            raw_output = await call_llm(correction_prompt)
            payload = extract_tool_payload_from_text(raw_output)
            if payload is None:
                final_answer = raw_output.strip()
                break
            ok, err, tool_name, normalized_args = validate_and_normalize_tool_call(payload, reg)
            if not ok:
                final_answer = raw_output.strip()
                break

        # Execute the tool
        if emit_status:
            emit_status(f"Using {tool_name}…")

        t0 = time.time()
        try:
            tool_result = await asyncio.wait_for(
                execute_tool(tool_name, normalized_args, chat_id),
                timeout=TOOL_CALL_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            tool_result = {"ok": False, "error": f"Tool {tool_name} timed out after {TOOL_CALL_TIMEOUT_SEC}s"}
        except Exception as exc:
            tool_result = {"ok": False, "error": f"Tool {tool_name} error: {str(exc)[:200]}"}
        elapsed = time.time() - t0

        summary = summarize_tool_result(tool_name, tool_result)
        tool_history.append(f"[{tool_name}] → {summary[:TOOL_RESULT_CHAR_LIMIT]}")

        event = ToolEvent(
            step=step,
            tool_name=tool_name,
            args=normalized_args or {},
            result_summary=summary[:500],
            ok=tool_result.get("ok", False),
            elapsed_sec=round(elapsed, 2),
            raw_result=tool_result,
        )
        result.events.append(event)
        if tool_name not in result.tools_used:
            result.tools_used.append(tool_name)

    result.steps_used = step
    result.total_elapsed_sec = round(time.time() - loop_start, 2)
    result.final_answer = final_answer or "I wasn't able to produce a final answer within the allowed steps."
    return result
