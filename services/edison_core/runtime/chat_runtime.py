"""
chat_runtime.py — Unified chat pipeline shared by all entry points.

ALL chat flows (normal chat, streaming, work mode, agent mode, swarm mode,
/v1/chat/completions) funnel through the same pipeline stages:

    1. Normalize incoming request
    2. Load session/task/project/workspace context
    3. Classify intent (routing)
    4. Decide whether tools are needed
    5. Retrieve memory and relevant knowledge
    6. Decide model target and reasoning level
    7. Optionally plan the response/tool sequence
    8. Execute tools if needed
    9. Synthesize response
   10. Run quality check
   11. Format response for UI/API
   12. Persist memory/task/artifact state

The pipeline produces a ChatPipelineResponse that callers can then
format for their specific output format (native JSON, OpenAI, SSE, etc.).
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Awaitable

from .routing_runtime import route, RoutingDecision
from .context_runtime import assemble_context, get_summary, update_summary, AssembledContext
from .task_runtime import get_active_task_for_chat, TaskState
from .artifact_runtime import artifact_refs_for_context
from .quality_runtime import check_response_quality, clean_response, format_trust_signals
from .response_runtime import ChatPipelineResponse
from .tool_runtime import run_tool_loop, ToolLoopResult

logger = logging.getLogger(__name__)


# ── Pipeline Input ───────────────────────────────────────────────────
@dataclass
class ChatPipelineInput:
    """Normalized input for the unified chat pipeline."""
    message: str = ""
    mode: str = "auto"
    images: Optional[List[str]] = None
    conversation_history: Optional[List[dict]] = None
    chat_id: str = ""
    request_id: str = ""
    workspace_id: str = "default"
    project_id: str = ""
    selected_model: Optional[str] = None
    assistant_profile: Optional[dict] = None
    remember: Optional[bool] = None
    global_memory_search: bool = False
    swarm_session_id: Optional[str] = None
    # Set by OpenAI-compat adapter
    system_prompt_override: Optional[str] = None
    openai_messages: Optional[List[dict]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())
        if not self.chat_id:
            self.chat_id = f"chat_{uuid.uuid4().hex[:8]}"


# ── Pipeline Callbacks ───────────────────────────────────────────────
# The pipeline is decoupled from app.py globals via callbacks.
@dataclass
class ChatPipelineCallbacks:
    """Injectable functions that connect the pipeline to the actual app services."""
    # Model inference
    call_llm: Optional[Callable[[str, str], Awaitable[str]]] = None  # (prompt, model_target) -> response
    # Tool execution
    execute_tool: Optional[Callable[[str, dict, Optional[str]], Awaitable[dict]]] = None
    summarize_tool_result: Optional[Callable[[str, dict], str]] = None
    # Memory
    retrieve_memory: Optional[Callable[[str, str, bool], List[str]]] = None  # (query, chat_id, global) -> facts
    store_memory: Optional[Callable[[str, str, str], None]] = None  # (chat_id, user_msg, assistant_msg)
    # Search / RAG
    retrieve_rag: Optional[Callable[[str, str], List[dict]]] = None  # (query, chat_id) -> chunks
    # Intent
    get_coral_intent: Optional[Callable[[str], Optional[str]]] = None
    # Status emitter (for streaming)
    emit_status: Optional[Callable[[str], Any]] = None
    # Cancellation checker
    is_cancelled: Optional[Callable[[], bool]] = None
    # Remember decision
    should_remember: Optional[Callable[[str], dict]] = None
    # Automation / business action pre-checks
    check_automation: Optional[Callable[[str], Optional[dict]]] = None
    check_business_action: Optional[Callable[[str], Optional[dict]]] = None


# ── System prompts ───────────────────────────────────────────────────
DEFAULT_SYSTEM_PROMPT = (
    "You are EDISON, a capable and thoughtful AI assistant. "
    "You help with a wide range of tasks including conversation, research, "
    "coding, creative work, product design, branding, and project management. "
    "Be clear, honest, and helpful. Provide well-structured answers. "
    "When you don't know something, say so. When you use tools or search, "
    "cite your sources."
)

MODE_SYSTEM_ADDENDA = {
    "code": "\nYou are in code-assistance mode. Focus on precise, well-structured code.",
    "reasoning": "\nThink carefully step by step. Show your reasoning.",
    "agent": "\nYou have access to tools. Use them when needed to answer accurately.",
    "work": "\nYou are in work mode. Break complex tasks into steps and execute them.",
    "swarm": "\nYou are coordinating a multi-agent collaboration.",
}


# ── Unified Pipeline ─────────────────────────────────────────────────

async def run_pipeline(
    inp: ChatPipelineInput,
    cb: ChatPipelineCallbacks,
) -> ChatPipelineResponse:
    """
    Run the unified chat pipeline.
    
    This is the single pipeline function that ALL entry points should use.
    It orchestrates the full flow from input to response.
    """
    resp = ChatPipelineResponse()
    t0 = time.time()

    # ── Stage 1: Normalize ───────────────────────────────────────────
    if not inp.message and inp.openai_messages:
        # Extract from OpenAI messages
        from .response_runtime import openai_messages_to_prompt
        sys_prompt, conv_hist, user_msg, has_img = openai_messages_to_prompt(inp.openai_messages)
        inp.message = user_msg
        inp.conversation_history = conv_hist
        if has_img:
            inp.images = inp.images or []
        if inp.system_prompt_override is None:
            inp.system_prompt_override = sys_prompt

    if not inp.message.strip():
        resp.content = "I didn't receive a message. How can I help you?"
        resp.mode_used = "chat"
        return resp

    # ── Stage 1.5: Pre-pipeline interceptors ─────────────────────────
    # Check automations
    if cb.check_automation:
        auto_result = cb.check_automation(inp.message)
        if auto_result:
            resp.content = auto_result.get("response", "")
            resp.mode_used = auto_result.get("mode_used", "automation")
            resp.automation = auto_result
            return resp

    # Check business actions
    if cb.check_business_action:
        biz_result = cb.check_business_action(inp.message)
        if biz_result:
            resp.content = biz_result.get("response", "")
            resp.mode_used = biz_result.get("mode_used", "business")
            resp.business_action = biz_result
            return resp

    # ── Stage 2: Load context ────────────────────────────────────────
    has_image = bool(inp.images)
    coral_intent = None
    if cb.get_coral_intent:
        try:
            coral_intent = cb.get_coral_intent(inp.message)
        except Exception:
            pass

    # ── Stage 3: Classify intent (routing) ───────────────────────────
    routing = route(
        user_message=inp.message,
        requested_mode=inp.mode,
        has_image=has_image,
        coral_intent=coral_intent,
        conversation_history=inp.conversation_history,
    )
    resp.mode_used = routing.mode
    resp.model_used = routing.model_target

    # Override model target if user explicitly selected one
    if inp.selected_model:
        resp.model_used = inp.selected_model

    # ── Stage 4-5: Retrieve memory and knowledge ─────────────────────
    memory_facts = []
    if cb.retrieve_memory:
        try:
            memory_facts = cb.retrieve_memory(
                inp.message, inp.chat_id, inp.global_memory_search
            )
        except Exception as e:
            logger.warning(f"Memory retrieval error: {e}")

    rag_results = []
    if cb.retrieve_rag and routing.search_needed:
        try:
            rag_results = cb.retrieve_rag(inp.message, inp.chat_id)
        except Exception as e:
            logger.warning(f"RAG retrieval error: {e}")

    # Get task state
    active_task = get_active_task_for_chat(inp.chat_id)
    task_context = active_task.context_dict() if active_task else None

    # Get artifact refs
    art_refs = artifact_refs_for_context(inp.chat_id, limit=5)

    # Get conversation summary
    conv_summary_obj = get_summary(inp.chat_id)
    conv_summary = conv_summary_obj.summary_text if conv_summary_obj else None

    # ── Stage 6: Assemble context ────────────────────────────────────
    system_prompt = inp.system_prompt_override or DEFAULT_SYSTEM_PROMPT
    mode_addendum = MODE_SYSTEM_ADDENDA.get(routing.mode, "")
    full_system = system_prompt + mode_addendum

    assembled = assemble_context(
        system_prompt=full_system,
        conversation_history=inp.conversation_history,
        conversation_summary=conv_summary,
        task_state=task_context,
        memory_facts=memory_facts,
        rag_results=rag_results,
        artifact_refs=art_refs,
        assistant_profile=inp.assistant_profile,
    )
    resp.context_chars_used = assembled.total_chars

    # ── Stage 7-8: Execute tools if needed ───────────────────────────
    if routing.tools_allowed and cb.execute_tool and cb.call_llm and cb.summarize_tool_result:
        if cb.emit_status:
            cb.emit_status("Using tools…")

        context_note = assembled.combined_context

        async def _tool_call_llm(prompt: str) -> str:
            return await cb.call_llm(prompt, routing.model_target)

        tool_result: ToolLoopResult = await run_tool_loop(
            call_llm=_tool_call_llm,
            execute_tool=cb.execute_tool,
            summarize_tool_result=cb.summarize_tool_result,
            user_message=inp.message,
            context_note=context_note,
            model_name=resp.model_used,
            chat_id=inp.chat_id,
            request_id=inp.request_id,
            is_cancelled=cb.is_cancelled,
            emit_status=cb.emit_status,
        )

        resp.content = tool_result.final_answer
        resp.tools_used = tool_result.tools_used
        resp.tool_events = [
            {
                "step": e.step,
                "tool": e.tool_name,
                "args": e.args,
                "ok": e.ok,
                "summary": e.result_summary,
                "elapsed": e.elapsed_sec,
            }
            for e in tool_result.events
        ]
        if tool_result.steps_used > 0:
            resp.search_results_count = sum(
                1 for e in tool_result.events
                if e.tool_name in ("web_search", "rag_search", "knowledge_search")
            )
    else:
        # ── Stage 9: Direct LLM inference ────────────────────────────
        if cb.call_llm:
            prompt = _build_inference_prompt(assembled, inp.message, routing)
            resp.content = await cb.call_llm(prompt, routing.model_target)
        else:
            resp.content = "No language model is available to process this request."

    # ── Stage 10: Quality check ──────────────────────────────────────
    qc = check_response_quality(
        response=resp.content,
        user_message=inp.message,
        mode=routing.mode,
        tools_used=resp.tools_used,
    )
    if not qc.passed:
        resp.content = clean_response(resp.content)
        logger.info(f"Quality check issues: {qc.issues}")

    # ── Stage 11: Trust signals ──────────────────────────────────────
    resp.trust_signals = format_trust_signals(
        tools_used=resp.tools_used,
        search_performed=bool(resp.search_results_count),
        memory_used=bool(memory_facts),
        browser_used=any(t.startswith("browser.") for t in (resp.tools_used or [])),
        artifact_created=bool(resp.artifacts_created),
        code_executed="execute_python" in (resp.tools_used or []),
        uncertain=routing.confidence < 0.5,
    )

    # ── Stage 12: Persist state ──────────────────────────────────────
    if cb.store_memory:
        try:
            remember_result = {"remember": True}
            if cb.should_remember:
                remember_result = cb.should_remember(inp.message)
            should_store = inp.remember if inp.remember is not None else remember_result.get("remember", False)
            if should_store:
                cb.store_memory(inp.chat_id, inp.message, resp.content)
        except Exception as e:
            logger.warning(f"Memory storage error: {e}")

    # Update conversation summary (lightweight)
    turn_count = len(inp.conversation_history) if inp.conversation_history else 0
    if turn_count > 2:
        # Simple summary: just keep track of topics
        topics = []
        if routing.mode != "chat":
            topics.append(routing.mode)
        update_summary(
            inp.chat_id,
            summary_text=f"Conversation with {turn_count} turns. Last mode: {routing.mode}.",
            turn_count=turn_count,
            key_topics=topics,
            active_task=active_task.task_id if active_task else None,
        )

    # Set task_id if active
    if active_task:
        resp.task_id = active_task.task_id

    elapsed = time.time() - t0
    logger.info(
        f"Pipeline completed: mode={resp.mode_used}, model={resp.model_used}, "
        f"tools={resp.tools_used}, elapsed={elapsed:.2f}s"
    )
    return resp


def _build_inference_prompt(
    assembled: AssembledContext,
    user_message: str,
    routing: RoutingDecision,
) -> str:
    """Build the final prompt string for direct LLM inference."""
    parts = [assembled.system_prompt]

    context = assembled.combined_context
    if context.strip():
        parts.append(f"\n{context}")

    parts.append(f"\nUser: {user_message}")
    parts.append("Assistant:")

    return "\n".join(parts)
