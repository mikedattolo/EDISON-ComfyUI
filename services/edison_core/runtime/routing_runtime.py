"""
routing_runtime.py — Intent classification and mode/model/tool decision making.

Extracts the routing logic from app.py's route_mode() and supporting functions
into a clean, testable module. The router produces a structured RoutingDecision
object instead of a raw dict.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Routing Decision ─────────────────────────────────────────────────
@dataclass
class RoutingDecision:
    """Structured output of the router. All downstream code should depend on this."""
    mode: str = "chat"
    model_target: str = "fast"
    tools_allowed: bool = False
    search_needed: bool = False
    browser_needed: bool = False
    reasoning_depth: str = "light"  # light | moderate | deep
    artifact_likely: bool = False
    is_followup: bool = False
    latency_priority: str = "normal"  # fast | normal | patient
    confidence: float = 0.5
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "model_target": self.model_target,
            "tools_allowed": self.tools_allowed,
            "search_needed": self.search_needed,
            "browser_needed": self.browser_needed,
            "reasoning_depth": self.reasoning_depth,
            "artifact_likely": self.artifact_likely,
            "is_followup": self.is_followup,
            "latency_priority": self.latency_priority,
            "confidence": self.confidence,
            "reasons": self.reasons,
        }

    # Backward-compat: old code does decision["mode"]
    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


# ── Pattern libraries ────────────────────────────────────────────────
WORK_PATTERNS = [
    "create a project plan", "project plan", "multi-step workflow",
    "step-by-step workflow", "break down this task", "organize this project",
    "workflow for", "execution plan", "task plan", "roadmap", "milestones",
    "deliverables",
]

CODE_PATTERNS = [
    "write a function", "write a script", "write a class", "write code",
    "build a component", "react component", "fastapi endpoint", "api route",
    "python", "javascript", "typescript", "html", "css", "sql", "bash",
    "regex", "algorithm", "method", "class", "function", "unit test",
    "debug", "fix this bug", "syntax error", "stack trace", "traceback",
    "refactor", "code review", "endpoint", "schema", "component",
    "implement", "binary tree", "data structure", "linked list",
]

AGENT_PATTERNS = [
    "search", "internet", "web", "find on", "lookup", "google",
    "current", "latest", "news about", "information on", "information about",
    "tell me about", "research", "browse", "what's happening",
    "recent", "today", "this week", "this month", "this year",
    "2025", "2026", "2027", "now", "currently", "look up",
    "find out", "check", "what is happening", "what happened",
    "who is", "where is", "when is", "show me", "get me",
    "look for", "search for", "find information",
]

REALTIME_PATTERNS = [
    "what time", "current time", "the time", "what's the time",
    "whats the time", "what date", "today's date", "todays date",
    "what day is it", "what is today", "the date",
    "weather in", "weather for", "forecast", "temperature in",
    "is it raining", "is it cold", "is it hot", "is it snowing",
    "today's news", "todays news", "latest news", "top news",
    "news today", "headlines", "breaking news", "what's in the news",
    "news about",
]

MUSIC_PATTERNS = [
    "make music", "create music", "generate music",
    "make a song", "create a song", "generate a song",
    "compose", "make a beat", "produce music",
    "music like", "song about", "write a song",
    "make me a song", "generate a beat", "music from",
    "make me music", "create a beat", "lo-fi", "lofi",
    "hip hop beat", "hip-hop beat", "edm", "make a track",
    "generate song", "generate beat", "play me",
    "sing me", "beat for", "instrumental",
    "background music", "soundtrack",
]

REASONING_PATTERNS = [
    "explain", "why", "how does", "what is", "analyze", "detail",
    "understand", "break down", "elaborate", "clarify", "reasoning",
    "think through", "step by step", "logic", "rationale",
]

BUSINESS_PATTERNS = [
    "list my project", "list project", "my project", "show project",
    "create a project", "create project", "new project",
    "create a client", "create client", "new client",
    "list client", "my client", "show client",
    "branding client", "brand package", "brand brief",
    "generate a brand", "generate brand", "marketing copy",
    "generate marketing", "branding for", "brand for",
    "logo concept", "logo for", "slogan for", "tagline for",
    "moodboard", "mood board", "style guide for",
    "palette for", "color palette", "typography for",
    "create a task", "list task", "my task", "complete task",
    "slice model", "slice the", "3d print", "print job",
    "fabricat", "keychain", "plaque", "nameplate",
    "generate video", "make a video", "create a video",
    "ad copy", "social caption", "email campaign",
    "business card", "flyer for", "menu for",
    "signage", "promo", "campaign",
    "social post", "post to instagram", "post to facebook", "post to tiktok",
    "post to linkedin", "schedule post", "schedule a post", "draft post",
    "social media", "social campaign", "list posts", "list social",
    "publish post", "content calendar",
]

BROWSER_PATTERNS = [
    "go to", "open the website", "visit the page", "navigate to",
    "browse to", "click on", "fill out the form", "log into",
    "sign into", "scrape", "extract from the page",
]

ARTIFACT_PATTERNS = [
    "write a report", "create a document", "generate a brief",
    "make a plan", "draft a", "build a template",
    "create a file", "produce a", "compile a",
    "brand brief", "style guide", "shot list",
    "bill of materials", "bom",
]


# ── Continuation / follow-up detection ───────────────────────────────
_REFERENTIAL_TERMS = [
    "that", "it", "this", "those", "these", "them", "there", "here",
    "the same", "again", "instead", "also", "too", "more like", "less like",
    "make it", "change it", "fix it", "do that", "do this", "use that",
    "now", "then", "continue", "go on", "try again", "one more",
]

_FOLLOWUP_OPENERS = [
    "and ", "also ", "now ", "then ", "instead ", "but ", "so ",
    "make ", "change ", "fix ", "use ", "add ", "remove ", "update ",
    "turn ", "convert ", "try ", "keep ", "expand ", "refine ",
]


def looks_like_followup(user_message: str, conversation_history: Optional[list] = None) -> bool:
    """Detect whether the user message is a follow-up to prior context."""
    if not conversation_history:
        return False
    msg = (user_message or "").strip().lower()
    if not msg:
        return False
    if any(term in msg for term in _REFERENTIAL_TERMS):
        return True
    if len(msg.split()) <= 12 and any(msg.startswith(p) for p in _FOLLOWUP_OPENERS):
        return True
    return False


def infer_contextual_mode(conversation_history: Optional[list]) -> Optional[str]:
    """Infer the most likely continuation mode from recent history."""
    if not conversation_history:
        return None
    recent_text = "\n".join(
        str(msg.get("content", ""))
        for msg in conversation_history[-4:]
        if isinstance(msg, dict)
    ).lower()
    if not recent_text.strip():
        return None
    code_signals = [
        "```", "def ", "function ", "class ", "import ", "from ",
        "python", "javascript", "typescript", "react", "fastapi", "html", "css",
        ".py", ".js", ".ts", ".tsx", ".html", ".css", "stack trace", "refactor",
        "component", "endpoint", "bug", "exception", "traceback",
    ]
    search_signals = [
        "search results", "source:", "looked up", "web search",
        "according to", "latest news",
    ]
    if any(s in recent_text for s in code_signals):
        return "code"
    if any(s in recent_text for s in search_signals):
        return "agent"
    return "reasoning"


def _match_any(msg_lower: str, patterns: list) -> bool:
    return any(p in msg_lower for p in patterns)


# ── Main router ──────────────────────────────────────────────────────
def route(
    user_message: str,
    requested_mode: str = "auto",
    has_image: bool = False,
    coral_intent: Optional[str] = None,
    conversation_history: Optional[list] = None,
) -> RoutingDecision:
    """
    Central routing function.  Replaces the old ``route_mode()`` dict output
    with a structured ``RoutingDecision``.

    The function is a pure classifier — it does not call models or I/O.
    """
    d = RoutingDecision()
    msg_lower = (user_message or "").strip().lower()

    # ── Rule 1: Image input ──────────────────────────────────────────
    if has_image:
        d.mode = "image"
        d.model_target = "vision"
        d.confidence = 0.95
        d.reasons.append("Image input detected → image mode")
        return d

    # ── Rule 2: Explicit mode requested ──────────────────────────────
    if requested_mode != "auto":
        d.mode = requested_mode
        d.reasons.append(f"User explicitly requested mode: {requested_mode}")

        mode_map = {
            "instant": ("chat", "fast", False, "light", "fast"),
            "thinking": ("reasoning", "reasoning", False, "deep", "patient"),
            "reasoning": ("reasoning", "reasoning", False, "deep", "patient"),
            "agent": ("agent", "medium", True, "moderate", "normal"),
            "work": ("work", "deep", True, "deep", "patient"),
            "swarm": ("swarm", "deep", True, "deep", "patient"),
            "code": ("code", "deep", False, "moderate", "normal"),
            "chat": ("chat", "fast", False, "light", "fast"),
        }
        if requested_mode in mode_map:
            m = mode_map[requested_mode]
            d.mode, d.model_target, d.tools_allowed = m[0], m[1], m[2]
            d.reasoning_depth, d.latency_priority = m[3], m[4]
        d.confidence = 0.9
        return d

    # ── Rule 3: Coral intent ─────────────────────────────────────────
    if coral_intent:
        intent_lower = coral_intent.lower()
        if intent_lower in ("generate_image", "text_to_image", "create_image"):
            d.mode, d.model_target = "image", "vision"
            d.confidence = 0.85
        elif intent_lower in ("generate_music", "text_to_music", "create_music", "make_music", "compose_music"):
            d.mode, d.tools_allowed = "agent", True
            d.model_target = "medium"
            d.confidence = 0.8
        elif intent_lower in ("code", "write", "implement", "debug"):
            d.mode, d.model_target = "code", "deep"
            d.confidence = 0.8
        elif intent_lower in ("agent", "search", "web", "research"):
            d.mode, d.tools_allowed = "agent", True
            d.model_target = "medium"
            d.confidence = 0.8
        else:
            d.mode = "chat"
            d.confidence = 0.6
        d.reasons.append(f"Coral intent '{coral_intent}' → {d.mode} mode")
        return d

    # ── Rule 4: Heuristic pattern matching ───────────────────────────
    is_fu = looks_like_followup(user_message, conversation_history)
    d.is_followup = is_fu
    sticky_mode = infer_contextual_mode(conversation_history) if is_fu else None

    has_agent = _match_any(msg_lower, AGENT_PATTERNS)
    has_realtime = _match_any(msg_lower, REALTIME_PATTERNS)
    has_music = _match_any(msg_lower, MUSIC_PATTERNS)
    has_code = _match_any(msg_lower, CODE_PATTERNS)
    has_work = _match_any(msg_lower, WORK_PATTERNS)
    has_browser = _match_any(msg_lower, BROWSER_PATTERNS)
    has_artifact = _match_any(msg_lower, ARTIFACT_PATTERNS)
    has_business = _match_any(msg_lower, BUSINESS_PATTERNS)

    d.browser_needed = has_browser
    d.artifact_likely = has_artifact
    d.search_needed = has_agent or has_realtime

    # Sticky follow-ups
    if is_fu and sticky_mode and not any([has_realtime, has_music, has_agent, has_code, has_work]):
        d.mode = sticky_mode
        d.tools_allowed = sticky_mode == "agent"
        d.confidence = 0.65
        d.reasons.append(f"Follow-up detected → preserving {sticky_mode} mode")
    elif has_realtime:
        d.mode, d.tools_allowed = "agent", True
        d.model_target = "medium"
        d.confidence = 0.85
        d.reasons.append("Real-time data query → agent mode")
    elif has_business:
        d.mode, d.tools_allowed = "agent", True
        d.model_target = "medium"
        d.confidence = 0.85
        d.reasons.append("Business/project/branding request → agent mode")
    elif has_music:
        d.mode, d.tools_allowed = "agent", True
        d.model_target = "medium"
        d.confidence = 0.8
        d.reasons.append("Music request → agent mode")
    elif has_work:
        d.mode, d.tools_allowed = "work", True
        d.model_target = "deep"
        d.reasoning_depth = "deep"
        d.confidence = 0.75
        d.reasons.append("Work patterns → work mode")
    elif has_code:
        d.mode, d.model_target = "code", "deep"
        d.reasoning_depth = "moderate"
        d.confidence = 0.75
        d.reasons.append("Code patterns → code mode")
    elif has_agent:
        d.mode, d.tools_allowed = "agent", True
        d.model_target = "medium"
        d.confidence = 0.7
        d.reasons.append("Agent patterns → agent mode")
    elif _match_any(msg_lower, REASONING_PATTERNS):
        d.mode, d.model_target = "reasoning", "reasoning"
        d.reasoning_depth = "deep"
        d.latency_priority = "patient"
        d.confidence = 0.65
        if has_agent:
            d.tools_allowed = True
        d.reasons.append("Reasoning patterns → reasoning mode")
    else:
        d.mode = "chat"
        words = len(msg_lower.split())
        has_question = "?" in user_message
        if words > 15 or has_question:
            d.mode, d.model_target = "reasoning", "reasoning"
            d.reasoning_depth = "moderate"
            d.confidence = 0.5
            d.reasons.append("Complex/question message → reasoning mode")
        else:
            d.confidence = 0.4
            d.latency_priority = "fast"
            d.reasons.append("Default → chat mode")

    # ── Finalize model target ────────────────────────────────────────
    if d.model_target != "vision":
        target_map = {
            "reasoning": "reasoning",
            "work": "deep",
            "swarm": "deep",
            "code": "deep",
            "agent": "medium",
        }
        expected = target_map.get(d.mode)
        if expected and d.model_target != expected:
            d.model_target = expected
            d.reasons.append(f"Mode '{d.mode}' → {expected} model")
        elif d.mode in ("chat", "instant") and d.model_target not in ("fast",):
            d.model_target = "fast"

    logger.info(
        f"ROUTING: mode={d.mode}, model={d.model_target}, tools={d.tools_allowed}, "
        f"conf={d.confidence:.2f}, reasons={d.reasons}"
    )
    return d


# ── Legacy compatibility wrapper ─────────────────────────────────────
def route_mode(
    user_message: str,
    requested_mode: str,
    has_image: bool,
    coral_intent: Optional[str] = None,
    conversation_history: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Drop-in replacement for the old ``route_mode()`` in app.py.
    Returns a plain dict for backward compatibility while using the new router.
    """
    decision = route(
        user_message=user_message,
        requested_mode=requested_mode,
        has_image=has_image,
        coral_intent=coral_intent,
        conversation_history=conversation_history,
    )
    return {
        "mode": decision.mode,
        "tools_allowed": decision.tools_allowed,
        "model_target": decision.model_target,
        "reasons": decision.reasons,
    }
