"""
Agent Controller Brain - orchestration scaffolding for multi-agent, multimodal flows.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import uuid

from .contracts import AgentMode, AgentTask, AgentPlanResponse


@dataclass
class AgentControllerBrain:
    config: Dict[str, Any]

    def plan(self, goal: str, mode: AgentMode, has_image: bool = False, project_id: Optional[str] = None,
             context: Optional[Dict[str, Any]] = None) -> AgentPlanResponse:
        context = context or {}
        tasks: List[AgentTask] = []
        rationale: List[str] = []
        parallel = False

        if mode == "instant":
            rationale.append("Instant mode: single fast LLM response")
            tasks.append(self._task("llm", "Quick response", {"goal": goal, "context": context}))
        elif mode == "thinking":
            rationale.append("Thinking mode: deeper reasoning and verification")
            tasks.append(self._task("llm", "Deep reasoning", {"goal": goal, "context": context}))
            tasks.append(self._task("tool", "Self-check and summarize", {"goal": goal}))
        elif mode == "agent":
            rationale.append("Agent mode: enable tools and external actions")
            tasks.append(self._task("llm", "Plan steps", {"goal": goal}))
            tasks.append(self._task("tool", "Execute tools", {"goal": goal, "project_id": project_id}))
            if has_image:
                tasks.append(self._task("vision", "Analyze image", {"goal": goal}))
        elif mode == "swarm":
            rationale.append("Swarm mode: parallel specialized agents")
            parallel = True
            tasks.append(self._task("llm", "Coordinator", {"goal": goal}))
            tasks.append(self._task("code", "Code specialist", {"goal": goal}))
            tasks.append(self._task("web", "Research specialist", {"goal": goal}))
            if has_image:
                tasks.append(self._task("vision", "Vision specialist", {"goal": goal}))
        else:
            rationale.append("Fallback: instant mode")
            tasks.append(self._task("llm", "Quick response", {"goal": goal, "context": context}))

        policy = {
            "project_id": project_id,
            "has_image": has_image,
            "tool_access": mode in {"agent", "swarm"}
        }

        return AgentPlanResponse(
            mode=mode,
            parallel=parallel,
            rationale=rationale,
            tasks=tasks,
            policy=policy
        )

    def _task(self, kind: str, description: str, inputs: Dict[str, Any]) -> AgentTask:
        return AgentTask(
            id=str(uuid.uuid4())[:8],
            kind=kind,
            description=description,
            inputs=inputs
        )
