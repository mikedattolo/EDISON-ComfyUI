"""
Structured Logging for Edison Awareness Layer.

Provides a consistent structured-log format for:
  - Intent detection decisions
  - Planner decisions
  - Routing decisions
  - State updates
  - Self-evaluation outcomes

Uses Python's built-in logging with JSON-formatted extra fields.
"""

import json
import logging
import time
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """Formatter that emits JSON-structured log records.

    Falls back to a readable text format for non-structured messages.
    """

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any extra structured data attached to the record
        extra = getattr(record, "_structured", None)
        if extra and isinstance(extra, dict):
            base.update(extra)
        # Include exception info if present
        if record.exc_info and record.exc_info[0]:
            base["exception"] = self.formatException(record.exc_info)
        return json.dumps(base, default=str)


def get_structured_logger(name: str, json_output: bool = False) -> logging.Logger:
    """Return a logger configured for structured output.

    Args:
        name: Logger name (typically __name__).
        json_output: If True, add a JSON formatter to the handler.
                     If False, use the default Edison console format.
    """
    _logger = logging.getLogger(name)
    if json_output and not any(
        isinstance(h.formatter, StructuredFormatter) for h in _logger.handlers
    ):
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        _logger.addHandler(handler)
    return _logger


def log_structured(
    logger_instance: logging.Logger,
    level: int,
    message: str,
    **fields: Any,
):
    """Emit a structured log message with arbitrary key-value fields.

    Example:
        log_structured(logger, logging.INFO, "Routing decision",
                       intent="generate_image", mode="image", confidence=0.92)
    """
    record = logger_instance.makeRecord(
        logger_instance.name,
        level,
        "(structured)",
        0,
        message,
        (),
        None,
    )
    record._structured = fields  # type: ignore[attr-defined]
    logger_instance.handle(record)


# ── Convenience wrappers ─────────────────────────────────────────────────

def log_intent_decision(
    logger_instance: logging.Logger,
    message: str,
    intent: str,
    goal: str,
    confidence: float,
    continuation: str,
    coral_intent: Optional[str] = None,
):
    """Log an intent classification decision."""
    log_structured(
        logger_instance, logging.INFO,
        f"Intent decision: {intent} → {goal}",
        event="intent_decision",
        intent=intent,
        goal=goal,
        confidence=confidence,
        continuation=continuation,
        coral_intent=coral_intent,
        user_message_preview=message[:100],
    )


def log_planner_decision(
    logger_instance: logging.Logger,
    plan_id: str,
    complexity: str,
    step_count: int,
    goal: str,
    actions: list,
):
    """Log a planner's plan creation."""
    log_structured(
        logger_instance, logging.INFO,
        f"Plan created: {plan_id} ({complexity}, {step_count} steps)",
        event="planner_decision",
        plan_id=plan_id,
        complexity=complexity,
        step_count=step_count,
        goal=goal,
        actions=actions,
    )


def log_routing_decision(
    logger_instance: logging.Logger,
    mode: str,
    model_target: str,
    tools_allowed: bool,
    reasons: list,
):
    """Log a routing decision."""
    log_structured(
        logger_instance, logging.INFO,
        f"Routing: mode={mode}, model={model_target}, tools={tools_allowed}",
        event="routing_decision",
        mode=mode,
        model_target=model_target,
        tools_allowed=tools_allowed,
        reasons=reasons,
    )


def log_state_update(
    logger_instance: logging.Logger,
    session_id: str,
    updates: Dict[str, Any],
):
    """Log a conversation state update."""
    log_structured(
        logger_instance, logging.DEBUG,
        f"State update [{session_id}]: {list(updates.keys())}",
        event="state_update",
        session_id=session_id,
        updates=updates,
    )


def log_eval_outcome(
    logger_instance: logging.Logger,
    action: str,
    success: bool,
    duration_s: float,
    error: Optional[str] = None,
):
    """Log a self-evaluation outcome."""
    log_structured(
        logger_instance, logging.INFO if success else logging.WARNING,
        f"Eval: {action} → {'success' if success else 'failure'} ({duration_s:.2f}s)",
        event="eval_outcome",
        action=action,
        success=success,
        duration_s=duration_s,
        error=error,
    )
