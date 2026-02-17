"""
Goal-Level Intent Detection and Continuation Detection for Edison.

Extends intent classification to output (intent, goal, confidence)
and detects whether the user is continuing/modifying a previous task
or starting something new.
"""

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Goal taxonomy ────────────────────────────────────────────────────────

class Goal(str, Enum):
    """High-level user goals that span multiple intents."""
    DEBUG_CODE = "debug_code"
    MODIFY_PREVIOUS_OUTPUT = "modify_previous_output"
    CONTINUE_PREVIOUS_TASK = "continue_previous_task"
    RESEARCH_TOPIC = "research_topic"
    GENERATE_NEW_ARTIFACT = "generate_new_artifact"
    EXPLAIN_CONCEPT = "explain_concept"
    CONFIGURE_SYSTEM = "configure_system"
    RECALL_MEMORY = "recall_memory"
    CASUAL_CHAT = "casual_chat"
    UNKNOWN = "unknown"


class ContinuationType(str, Enum):
    """Whether the user is continuing, modifying, or starting fresh."""
    CONTINUE_PREVIOUS = "continue_previous"
    MODIFY_PREVIOUS = "modify_previous"
    NEW_TASK = "new_task"


# ── Result dataclass ─────────────────────────────────────────────────────

@dataclass
class IntentResult:
    """Full intent classification result with goal and continuation info."""
    intent: str                          # raw intent label (e.g., "generate_image")
    goal: Goal                           # higher-level goal
    confidence: float                    # 0.0-1.0
    continuation: ContinuationType       # new / continue / modify
    reasoning: str = ""                  # audit trail

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "goal": self.goal.value,
            "confidence": self.confidence,
            "continuation": self.continuation.value,
            "reasoning": self.reasoning,
        }


# ── Goal detection ───────────────────────────────────────────────────────

# Intent → Goal mapping (for intents from Coral or heuristic)
_INTENT_GOAL_MAP: Dict[str, Goal] = {
    "generate_image": Goal.GENERATE_NEW_ARTIFACT,
    "text_to_image": Goal.GENERATE_NEW_ARTIFACT,
    "create_image": Goal.GENERATE_NEW_ARTIFACT,
    "generate_video": Goal.GENERATE_NEW_ARTIFACT,
    "text_to_video": Goal.GENERATE_NEW_ARTIFACT,
    "create_video": Goal.GENERATE_NEW_ARTIFACT,
    "make_video": Goal.GENERATE_NEW_ARTIFACT,
    "generate_music": Goal.GENERATE_NEW_ARTIFACT,
    "text_to_music": Goal.GENERATE_NEW_ARTIFACT,
    "create_music": Goal.GENERATE_NEW_ARTIFACT,
    "make_music": Goal.GENERATE_NEW_ARTIFACT,
    "compose_music": Goal.GENERATE_NEW_ARTIFACT,
    "generate_3d": Goal.GENERATE_NEW_ARTIFACT,
    "create_mesh": Goal.GENERATE_NEW_ARTIFACT,
    "code": Goal.DEBUG_CODE,
    "debug": Goal.DEBUG_CODE,
    "fix_code": Goal.DEBUG_CODE,
    "explain": Goal.EXPLAIN_CONCEPT,
    "define": Goal.EXPLAIN_CONCEPT,
    "research": Goal.RESEARCH_TOPIC,
    "search": Goal.RESEARCH_TOPIC,
    "web_search": Goal.RESEARCH_TOPIC,
    "get_time": Goal.CASUAL_CHAT,
    "get_weather": Goal.CASUAL_CHAT,
    "greeting": Goal.CASUAL_CHAT,
    "small_talk": Goal.CASUAL_CHAT,
    "recall": Goal.RECALL_MEMORY,
    "remember": Goal.RECALL_MEMORY,
    "system_status": Goal.CONFIGURE_SYSTEM,
    "configure": Goal.CONFIGURE_SYSTEM,
}

# Keyword patterns for goal detection (fallback when intent is vague)
_GOAL_PATTERNS: Dict[Goal, List[str]] = {
    Goal.DEBUG_CODE: [
        r"\b(debug|fix|error|bug|traceback|exception|crash|doesn'?t work|broken|issue)\b",
        r"\b(why is|what'?s wrong|not working)\b",
    ],
    Goal.MODIFY_PREVIOUS_OUTPUT: [
        r"\b(change|modify|update|edit|adjust|tweak|alter|revise)\b.{0,30}\b(that|it|this|previous|last|above)\b",
        r"\b(make it|can you)\b.{0,20}\b(more|less|bigger|smaller|different|brighter|darker)\b",
        r"\b(redo|undo|try again)\b",
    ],
    Goal.CONTINUE_PREVIOUS_TASK: [
        r"\b(continue|keep going|go on|next step|what'?s next|and then|also|now)\b",
        r"\b(what about|how about|add to)\b",
    ],
    Goal.RESEARCH_TOPIC: [
        r"\b(research|find out|look up|search for|what is|who is|tell me about)\b",
        r"\b(latest|recent|news about|updates on)\b",
    ],
    Goal.GENERATE_NEW_ARTIFACT: [
        r"\b(generate|create|make|build|write|draw|compose|produce|design)\b",
    ],
    Goal.EXPLAIN_CONCEPT: [
        r"\b(explain|describe|what does|how does|why does|define|meaning of)\b",
    ],
    Goal.CONFIGURE_SYSTEM: [
        r"\b(configure|settings|setup|install|enable|disable|restart)\b",
    ],
    Goal.RECALL_MEMORY: [
        r"\b(remember|recall|what did I|my name|do you know)\b",
    ],
}


def detect_goal(
    message: str,
    coral_intent: Optional[str] = None,
    last_intent: Optional[str] = None,
    last_goal: Optional[str] = None,
) -> Tuple[Goal, float]:
    """Detect the user's high-level goal.

    Returns (goal, confidence).  Uses coral intent mapping first, then
    falls back to keyword pattern matching.
    """
    # 1) Direct mapping from Coral/heuristic intent
    if coral_intent and coral_intent in _INTENT_GOAL_MAP:
        return _INTENT_GOAL_MAP[coral_intent], 0.85

    # 2) Keyword pattern matching
    msg_lower = message.lower()
    best_goal = Goal.UNKNOWN
    best_score = 0.0

    for goal, patterns in _GOAL_PATTERNS.items():
        hits = sum(1 for p in patterns if re.search(p, msg_lower, re.IGNORECASE))
        if hits > best_score:
            best_score = hits
            best_goal = goal

    if best_score > 0:
        confidence = min(0.5 + best_score * 0.15, 0.80)
        return best_goal, confidence

    # 3) Contextual fallback: if same domain as last turn, likely continuation
    if last_goal and last_goal != Goal.UNKNOWN.value:
        return Goal.CONTINUE_PREVIOUS_TASK, 0.35

    return Goal.CASUAL_CHAT, 0.30


# ── Continuation detection ───────────────────────────────────────────────

# Patterns that strongly indicate continuation or modification
_CONTINUE_PATTERNS = [
    r"^(ok|okay|yes|yeah|yep|sure|right|got it|and|also|then|next)\b",
    r"\b(continue|keep going|go on|next|proceed|what'?s next)\b",
    r"\b(another one|one more|again)\b",
]

_MODIFY_PATTERNS = [
    r"\b(change|modify|update|edit|adjust|tweak|make it|redo)\b",
    r"\b(but|instead|rather|actually|wait)\b.{0,30}\b(change|make|use|try)\b",
    r"\b(more|less|bigger|smaller|brighter|darker|louder|quieter|faster|slower)\b",
    r"\b(different|another|new)\b.{0,20}\b(style|color|tone|version|approach)\b",
]

_NEW_TASK_PATTERNS = [
    r"^(hey|hi|hello|yo|sup)\b",
    r"\b(new topic|something else|different question|unrelated)\b",
    r"\b(forget that|never ?mind|scratch that|start over)\b",
]


def detect_continuation(
    message: str,
    last_intent: Optional[str] = None,
    last_domain: Optional[str] = None,
    turn_count: int = 0,
) -> Tuple[ContinuationType, float]:
    """Determine whether a message continues, modifies, or starts a new task.

    Returns (continuation_type, confidence).
    """
    msg_lower = message.lower().strip()

    # First turn is always new
    if turn_count == 0 or not last_intent:
        return ContinuationType.NEW_TASK, 0.95

    # Check explicit new-task signals
    for pattern in _NEW_TASK_PATTERNS:
        if re.search(pattern, msg_lower, re.IGNORECASE):
            return ContinuationType.NEW_TASK, 0.80

    # Check modification signals
    modify_hits = sum(
        1 for p in _MODIFY_PATTERNS if re.search(p, msg_lower, re.IGNORECASE)
    )
    if modify_hits >= 1:
        return ContinuationType.MODIFY_PREVIOUS, min(0.55 + modify_hits * 0.15, 0.90)

    # Check continuation signals
    continue_hits = sum(
        1 for p in _CONTINUE_PATTERNS if re.search(p, msg_lower, re.IGNORECASE)
    )
    if continue_hits >= 1:
        return ContinuationType.CONTINUE_PREVIOUS, min(0.55 + continue_hits * 0.15, 0.90)

    # Short messages after user already has context → likely continuation
    if len(msg_lower.split()) <= 4 and last_intent:
        return ContinuationType.CONTINUE_PREVIOUS, 0.45

    # Default: new task
    return ContinuationType.NEW_TASK, 0.50


# ── Unified classifier ──────────────────────────────────────────────────

def classify_intent_with_goal(
    message: str,
    coral_intent: Optional[str] = None,
    last_intent: Optional[str] = None,
    last_goal: Optional[str] = None,
    last_domain: Optional[str] = None,
    turn_count: int = 0,
) -> IntentResult:
    """Full intent classification: intent + goal + continuation.

    Combines Coral intent, goal detection, and continuation analysis
    into a single result.  If confidence is below the threshold (0.40),
    the caller should fall back to LLM-based classification.
    """
    intent = coral_intent or "unknown"

    goal, goal_confidence = detect_goal(
        message, coral_intent, last_intent, last_goal
    )

    continuation, cont_confidence = detect_continuation(
        message, last_intent, last_domain, turn_count
    )

    # If modifying previous, inherit goal from last turn if not detected
    if continuation == ContinuationType.MODIFY_PREVIOUS and goal == Goal.UNKNOWN:
        if last_goal:
            try:
                goal = Goal(last_goal)
            except ValueError:
                pass

    # Aggregate confidence (weighted average)
    confidence = goal_confidence * 0.6 + cont_confidence * 0.4

    reasoning_parts = [
        f"intent={intent}",
        f"goal={goal.value}(conf={goal_confidence:.2f})",
        f"continuation={continuation.value}(conf={cont_confidence:.2f})",
    ]
    if coral_intent:
        reasoning_parts.insert(0, f"coral={coral_intent}")

    result = IntentResult(
        intent=intent,
        goal=goal,
        confidence=confidence,
        continuation=continuation,
        reasoning=", ".join(reasoning_parts),
    )

    logger.info(
        f"INTENT: {result.intent} | goal={result.goal.value} | "
        f"cont={result.continuation.value} | conf={result.confidence:.2f} | "
        f"{result.reasoning}"
    )
    return result
