"""
quality_runtime.py — Lightweight response quality/review pass.

Runs a quick check before sending final responses. Modular post-processing
that doesn't destroy latency on simple answers.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QualityCheck:
    """Result of a response quality pass."""
    passed: bool = True
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    response_modified: bool = False
    original_length: int = 0
    final_length: int = 0


def check_response_quality(
    response: str,
    user_message: str = "",
    mode: str = "chat",
    tools_used: Optional[List[str]] = None,
    skip_for_simple: bool = True,
) -> QualityCheck:
    """
    Run a lightweight quality check on a response before sending it.
    
    Designed to be fast (<1ms) for simple responses, with deeper checks
    only for longer or tool-augmented responses.
    """
    qc = QualityCheck(original_length=len(response), final_length=len(response))

    if not response or not response.strip():
        qc.passed = False
        qc.issues.append("Empty response")
        return qc

    # Skip detailed checks for very short simple responses
    if skip_for_simple and len(response) < 200 and mode == "chat":
        return qc

    stripped = response.strip()

    # Check: response is just the user's message echoed back
    if stripped.lower() == (user_message or "").strip().lower():
        qc.issues.append("Response mirrors user input")
        qc.passed = False

    # Check: response is too robotic / template-only
    robotic_starts = [
        "I'd be happy to help",
        "Sure, I can help with that",
        "As an AI language model",
        "As a large language model",
        "I don't have personal",
    ]
    for phrase in robotic_starts:
        if stripped.lower().startswith(phrase.lower()):
            qc.suggestions.append(f"Response starts with cliché: '{phrase}'")

    # Check: tool results mentioned but no actual tool used
    if tools_used is not None and not tools_used:
        tool_claim_patterns = [
            "according to my search",
            "i found that",
            "search results show",
            "after searching",
            "browsing the web",
        ]
        for pattern in tool_claim_patterns:
            if pattern in stripped.lower():
                qc.issues.append(f"Claims tool use ('{pattern}') but no tools were actually called")

    # Check: response contains raw JSON tool calls that leaked
    if '{"tool":' in stripped and '"args":' in stripped:
        qc.issues.append("Raw tool call JSON leaked into response")

    # Check: response length vs mode expectations
    if mode in ("chat", "instant") and len(stripped) > 4000:
        qc.suggestions.append("Response seems long for chat mode; consider summarizing")
    if mode in ("work", "agent") and len(stripped) < 20:
        qc.suggestions.append("Response seems too brief for work/agent mode")

    # Check: unclosed code blocks
    code_fence_count = stripped.count("```")
    if code_fence_count % 2 != 0:
        qc.issues.append("Unclosed code block (odd number of ``` fences)")

    # Check: planning text mixed with final answer (common LLM issue)
    planning_leaks = [
        "let me think about",
        "i'll start by",
        "my plan is to",
        "step 1:", 
    ]
    if mode not in ("work", "agent", "reasoning"):
        for pattern in planning_leaks:
            if pattern in stripped.lower()[:200]:
                qc.suggestions.append(f"Planning text may be leaking: '{pattern}'")

    if qc.issues:
        qc.passed = False

    return qc


def clean_response(response: str) -> str:
    """
    Apply light cleanup to a response:
    - Remove leaked tool JSON
    - Close unclosed code blocks
    - Strip trailing whitespace
    """
    cleaned = response.strip()

    # Remove leaked tool call JSON at the start
    if cleaned.startswith('{"tool":'):
        # Try to find actual text after the JSON
        brace_depth = 0
        for i, ch in enumerate(cleaned):
            if ch == '{':
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth == 0:
                    remainder = cleaned[i + 1:].strip()
                    if remainder:
                        cleaned = remainder
                    break

    # Close unclosed code blocks
    if cleaned.count("```") % 2 != 0:
        cleaned += "\n```"

    return cleaned


def format_trust_signals(
    tools_used: Optional[List[str]] = None,
    search_performed: bool = False,
    memory_used: bool = False,
    browser_used: bool = False,
    artifact_created: bool = False,
    code_executed: bool = False,
    uncertain: bool = False,
) -> List[dict]:
    """
    Produce trust signal metadata for the UI.
    Each signal can be rendered as a small badge/indicator.
    """
    signals = []
    if search_performed:
        signals.append({"type": "search", "label": "Searched the web", "icon": "🔍"})
    if memory_used:
        signals.append({"type": "memory", "label": "Used memory", "icon": "🧠"})
    if browser_used:
        signals.append({"type": "browser", "label": "Used browser", "icon": "🌐"})
    if artifact_created:
        signals.append({"type": "artifact", "label": "Created artifact", "icon": "📄"})
    if code_executed:
        signals.append({"type": "code", "label": "Ran code", "icon": "⚡"})
    if tools_used:
        for tool in tools_used:
            if tool.startswith("browser."):
                continue  # already covered
            if tool in ("web_search", "rag_search", "knowledge_search"):
                continue  # already covered
            signals.append({"type": "tool", "label": f"Used {tool}", "icon": "🔧"})
    if uncertain:
        signals.append({"type": "uncertain", "label": "Uncertain result", "icon": "⚠️"})
    return signals
