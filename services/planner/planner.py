"""
Planner Layer for Edison.

Sits between intent detection and routing.  Decides whether multi-step
execution is needed, determines tool order, and decides what to persist
to memory.

This is intentionally lightweight and rule-based.  A future iteration
can swap the rule engine for an LLM-based planner.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PlanComplexity(str, Enum):
    TRIVIAL = "trivial"       # single LLM call, no tools
    SIMPLE = "simple"         # single tool call + LLM
    MULTI_STEP = "multi_step" # 2+ ordered tool calls
    PARALLEL = "parallel"     # independent tool calls that can run concurrently


@dataclass
class PlanStep:
    """A single step in a plan."""
    step_id: int
    action: str              # tool name or "llm_respond" or "memory_write"
    description: str
    depends_on: List[int] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    result: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "description": self.description,
            "depends_on": self.depends_on,
            "completed": self.completed,
        }


@dataclass
class Plan:
    """An execution plan for a user request."""
    plan_id: str
    complexity: PlanComplexity
    steps: List[PlanStep]
    original_message: str
    created_at: float = field(default_factory=time.time)
    goal: str = ""
    should_remember: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "complexity": self.complexity.value,
            "steps": [s.to_dict() for s in self.steps],
            "goal": self.goal,
            "should_remember": self.should_remember,
        }

    @property
    def pending_steps(self) -> List[PlanStep]:
        return [s for s in self.steps if not s.completed]

    @property
    def is_complete(self) -> bool:
        return all(s.completed for s in self.steps)

    def next_steps(self) -> List[PlanStep]:
        """Return steps whose dependencies are all satisfied."""
        completed_ids = {s.step_id for s in self.steps if s.completed}
        return [
            s for s in self.steps
            if not s.completed and all(d in completed_ids for d in s.depends_on)
        ]


# ── Rule-based planner ───────────────────────────────────────────────────

class Planner:
    """Lightweight rule-based planner.

    Analyses the classified intent, goal, and conversation state to
    produce an execution plan.
    """

    def __init__(self):
        self._plan_counter = 0
        logger.info("Planner initialized")

    def _next_plan_id(self) -> str:
        self._plan_counter += 1
        return f"plan_{self._plan_counter}"

    def create_plan(
        self,
        message: str,
        intent: str = "unknown",
        goal: str = "unknown",
        continuation: str = "new_task",
        active_domain: str = "unknown",
        tools_allowed: bool = False,
        mode: str = "chat",
        has_image: bool = False,
    ) -> Plan:
        """Build a plan for the current user message.

        Returns a Plan with ordered steps.  The caller (router / chat endpoint)
        executes the plan steps in order.
        """
        steps: List[PlanStep] = []
        step_id = 0

        # ── Determine required steps ─────────────────────────────────────

        needs_retrieval = goal not in ("casual_chat", "configure_system")
        needs_tool = tools_allowed and goal in (
            "debug_code", "research_topic", "generate_new_artifact", "configure_system",
        )
        needs_web_search = mode in ("agent", "work") and goal == "research_topic"
        needs_generation = goal == "generate_new_artifact"
        needs_memory_write = goal not in ("casual_chat",)

        # Step: memory retrieval
        if needs_retrieval:
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="memory_retrieve",
                description="Retrieve relevant context from memory/RAG",
            ))

        # Step: web search (depends on retrieval)
        if needs_web_search:
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="web_search",
                description="Search the web for current information",
                depends_on=[1] if needs_retrieval else [],
            ))

        # Step: tool execution
        if needs_tool and not needs_generation:
            step_id += 1
            deps = [s.step_id for s in steps]  # depends on all prior steps
            steps.append(PlanStep(
                step_id=step_id,
                action="tool_execute",
                description=f"Execute relevant tool for {goal}",
                depends_on=deps,
            ))

        # Step: artifact generation
        if needs_generation:
            step_id += 1
            deps = [s.step_id for s in steps]
            gen_type = _infer_generation_type(intent, active_domain)
            steps.append(PlanStep(
                step_id=step_id,
                action=f"generate_{gen_type}",
                description=f"Generate {gen_type} artifact",
                depends_on=deps,
                params={"type": gen_type},
            ))

        # Step: LLM response (always present, depends on everything above)
        step_id += 1
        steps.append(PlanStep(
            step_id=step_id,
            action="llm_respond",
            description="Generate LLM response",
            depends_on=[s.step_id for s in steps],
        ))

        # Step: memory write (after response)
        if needs_memory_write:
            step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action="memory_write",
                description="Persist conversation to memory",
                depends_on=[step_id - 1],
            ))

        # Determine complexity
        if len(steps) <= 2:
            complexity = PlanComplexity.TRIVIAL
        elif needs_tool or needs_web_search:
            # Check if any steps can run in parallel
            has_parallel = any(
                len(s.depends_on) == 0 and s.step_id > 1
                for s in steps
            )
            complexity = PlanComplexity.PARALLEL if has_parallel else PlanComplexity.MULTI_STEP
        else:
            complexity = PlanComplexity.SIMPLE

        plan = Plan(
            plan_id=self._next_plan_id(),
            complexity=complexity,
            steps=steps,
            original_message=message,
            goal=goal,
            should_remember=needs_memory_write,
        )

        logger.info(
            f"PLAN [{plan.plan_id}]: complexity={complexity.value}, "
            f"steps={len(steps)}, goal={goal}, "
            f"actions=[{', '.join(s.action for s in steps)}]"
        )
        return plan


def _infer_generation_type(intent: str, domain: str) -> str:
    """Map intent/domain to a generation type."""
    if "image" in intent or domain == "image":
        return "image"
    if "video" in intent or domain == "video":
        return "video"
    if "music" in intent or domain == "music":
        return "music"
    if "mesh" in intent or "3d" in intent or domain == "mesh":
        return "mesh"
    return "text"


# ── Singleton ────────────────────────────────────────────────────────────

_planner: Optional[Planner] = None


def get_planner() -> Planner:
    global _planner
    if _planner is None:
        _planner = Planner()
    return _planner
