"""
response_runtime.py — Response formatting, OpenAI conversion, and output assembly.

Handles the conversion between EDISON internal response format and
external formats (native chat, OpenAI-compatible, SSE streaming).
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ── Internal response model ──────────────────────────────────────────
@dataclass
class ChatPipelineResponse:
    """Internal response produced by the unified chat pipeline."""
    content: str = ""
    mode_used: str = "chat"
    model_used: str = "fast"
    tools_used: List[str] = field(default_factory=list)
    tool_events: List[dict] = field(default_factory=list)
    trust_signals: List[dict] = field(default_factory=list)
    artifacts_created: List[dict] = field(default_factory=list)
    search_results_count: int = 0
    context_chars_used: int = 0
    work_steps: Optional[List[str]] = None
    task_id: Optional[str] = None
    business_action: Optional[dict] = None
    automation: Optional[dict] = None
    image_generation: Optional[dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Native EDISON response ──────────────────────────────────────────
def format_native_response(pipeline_resp: ChatPipelineResponse) -> dict:
    """Format a pipeline response for the native /chat endpoint."""
    return {
        "response": pipeline_resp.content,
        "mode_used": pipeline_resp.mode_used,
        "model_used": pipeline_resp.model_used,
        "tools_used": pipeline_resp.tools_used or None,
        "search_results_count": pipeline_resp.search_results_count or None,
        "context_used": pipeline_resp.context_chars_used or None,
        "work_steps": pipeline_resp.work_steps,
        "business_action": pipeline_resp.business_action,
        "automation": pipeline_resp.automation,
        "image_generation": pipeline_resp.image_generation,
    }


# ── OpenAI-compatible response ──────────────────────────────────────

def format_openai_response(
    pipeline_resp: ChatPipelineResponse,
    request_model: str = "edison",
) -> dict:
    """Format a pipeline response as an OpenAI chat completion."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    
    # Approximate token counts (rough char/4 estimate)
    prompt_tokens = pipeline_resp.context_chars_used // 4
    completion_tokens = len(pipeline_resp.content) // 4

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": request_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": pipeline_resp.content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def format_openai_stream_chunk(
    content_delta: str,
    completion_id: str,
    model: str = "edison",
    created: int = 0,
    finish_reason: Optional[str] = None,
    role: Optional[str] = None,
) -> str:
    """Format a single SSE chunk in OpenAI streaming format."""
    if not created:
        created = int(time.time())
    
    delta: Dict[str, str] = {}
    if role:
        delta["role"] = role
    if content_delta:
        delta["content"] = content_delta

    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def format_openai_stream_done() -> str:
    """Format the final [DONE] sentinel for OpenAI streaming."""
    return "data: [DONE]\n\n"


# ── Native SSE streaming helpers ─────────────────────────────────────

def format_native_sse_token(token: str) -> str:
    """Format a single token for native EDISON SSE streaming."""
    return f"data: {json.dumps({'t': token})}\n\n"


def format_native_sse_status(request_id: str, stage: str) -> str:
    """Format a status event for native SSE streaming."""
    return f"event: status\ndata: {json.dumps({'request_id': request_id, 'stage': stage})}\n\n"


def format_native_sse_tool_event(event: dict) -> str:
    """Format a tool event for native SSE streaming."""
    return f"event: tool\ndata: {json.dumps(event)}\n\n"


def format_native_sse_done(
    pipeline_resp: ChatPipelineResponse,
    request_id: str = "",
) -> str:
    """Format the done event for native SSE streaming."""
    done_payload = {
        "ok": True,
        "response": pipeline_resp.content,
        "mode_used": pipeline_resp.mode_used,
        "model_used": pipeline_resp.model_used,
        "tools_used": pipeline_resp.tools_used or [],
        "trust_signals": pipeline_resp.trust_signals or [],
        "artifacts": pipeline_resp.artifacts_created or [],
        "task_id": pipeline_resp.task_id,
        "request_id": request_id,
    }
    return f"event: done\ndata: {json.dumps(done_payload)}\n\n"


# ── OpenAI message conversion ───────────────────────────────────────

def openai_messages_to_prompt(
    messages: List[dict],
    default_system: str = "You are EDISON, a helpful AI assistant.",
) -> tuple:
    """
    Convert OpenAI-style messages to an EDISON-compatible prompt.
    
    Preserves FULL message history, not just last system+user message.
    Returns (system_prompt, conversation_history, last_user_message, has_images).
    """
    system_parts = []
    conversation_history = []
    last_user_message = ""
    has_images = False

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Check for multimodal content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    has_images = True

        if role == "system":
            if isinstance(content, list):
                text_parts = [
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                system_parts.append(" ".join(text_parts))
            else:
                system_parts.append(str(content))
        elif role == "user":
            if isinstance(content, list):
                text_parts = [
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                last_user_message = " ".join(text_parts)
            else:
                last_user_message = str(content)
            conversation_history.append({"role": "user", "content": last_user_message})
        elif role == "assistant":
            flat = content
            if isinstance(content, list):
                flat = " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            conversation_history.append({"role": "assistant", "content": str(flat)})
        elif role == "tool":
            conversation_history.append({"role": "tool", "content": str(content)})

    system_prompt = "\n".join(system_parts) if system_parts else default_system
    return system_prompt, conversation_history, last_user_message, has_images


def flatten_openai_content(content) -> str:
    """Flatten multimodal content blocks to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "image_url":
                    text_parts.append("[image]")
            elif isinstance(block, str):
                text_parts.append(block)
        return " ".join(text_parts)
    return str(content)
