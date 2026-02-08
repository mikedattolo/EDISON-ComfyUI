"""
Agent Controller Brain - orchestration for multi-agent, multimodal flows.
Includes work mode step planning with tool/search/artifact classification.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import uuid
import re

from .contracts import AgentMode, AgentTask, AgentPlanResponse, WorkStep, WorkPlanResponse


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
        elif mode == "work":
            rationale.append("Work mode: multi-step task with tools, search, and artifacts")
            parallel = False  # Steps are sequential in work mode
            tasks.append(self._task("llm", "Analyze and plan task breakdown", {"goal": goal}))
            if any(kw in goal.lower() for kw in ["search", "web", "find", "lookup", "research", "current", "latest"]):
                tasks.append(self._task("web", "Research and gather information", {"goal": goal}))
            if has_image:
                tasks.append(self._task("vision", "Analyze provided image", {"goal": goal}))
            tasks.append(self._task("llm", "Execute plan steps", {"goal": goal, "project_id": project_id}))
            if any(kw in goal.lower() for kw in ["document", "report", "file", "code", "repo", "schema", "website",
                                                    "spreadsheet", "slides", "presentation", "csv", "json"]):
                tasks.append(self._task("artifact", "Generate output files", {"goal": goal, "project_id": project_id}))
        elif mode == "agent":
            rationale.append("Agent mode: enable tools and external actions")
            tasks.append(self._task("llm", "Plan steps", {"goal": goal}))
            tasks.append(self._task("tool", "Execute tools", {"goal": goal, "project_id": project_id}))
            if has_image:
                tasks.append(self._task("vision", "Analyze image", {"goal": goal}))
            if any(token in goal.lower() for token in ["search", "web", "lookup", "find"]):
                tasks.append(self._task("web", "Web search", {"goal": goal}))
            if "image" in goal.lower() or "generate" in goal.lower():
                tasks.append(self._task("comfyui", "Generate image workflow", {"goal": goal}))
            if any(token in goal.lower() for token in ["document", "report", "slides", "presentation", "spreadsheet", "website", "repo", "repository", "schema"]):
                tasks.append(self._task("artifact", "Generate artifact output", {"goal": goal, "project_id": project_id, "kind": "document"}))
        elif mode == "swarm":
            rationale.append("Swarm mode: parallel specialized agents")
            parallel = True
            tasks.append(self._task("llm", "Coordinator", {"goal": goal}))
            tasks.append(self._task("code", "Code specialist", {"goal": goal}))
            tasks.append(self._task("web", "Research specialist", {"goal": goal}))
            if has_image:
                tasks.append(self._task("vision", "Vision specialist", {"goal": goal}))
            if "image" in goal.lower() or "generate" in goal.lower():
                tasks.append(self._task("comfyui", "Image workflow", {"goal": goal}))
            if any(token in goal.lower() for token in ["document", "report", "slides", "presentation", "spreadsheet", "website", "repo", "repository", "schema"]):
                tasks.append(self._task("artifact", "Generate artifact output", {"goal": goal, "project_id": project_id, "kind": "document"}))
        else:
            rationale.append("Fallback: instant mode")
            tasks.append(self._task("llm", "Quick response", {"goal": goal, "context": context}))

        policy = {
            "project_id": project_id,
            "has_image": has_image,
            "tool_access": mode in {"agent", "swarm", "work"}
        }

        return AgentPlanResponse(
            mode=mode,
            parallel=parallel,
            rationale=rationale,
            tasks=tasks,
            policy=policy
        )

    def plan_work_steps(self, goal: str, step_texts: List[str], has_image: bool = False,
                        project_id: Optional[str] = None) -> WorkPlanResponse:
        """
        Convert an LLM-generated step breakdown into typed WorkStep objects
        with auto-detected kinds (search, code, artifact, llm).
        """
        steps: List[WorkStep] = []
        goal_lower = goal.lower()

        for i, step_text in enumerate(step_texts):
            text_lower = step_text.lower()
            kind = self._classify_step_kind(text_lower, goal_lower, has_image)
            steps.append(WorkStep(
                id=i + 1,
                title=step_text.strip(),
                description="",
                kind=kind,
                status="pending"
            ))

        return WorkPlanResponse(
            task=goal,
            steps=steps,
            project_id=project_id,
            total_steps=len(steps),
            completed_steps=0,
            status="planning"
        )

    def _classify_step_kind(self, step_text: str, goal_text: str, has_image: bool) -> str:
        """Auto-detect the kind of work step based on keywords."""
        search_kws = ["search", "research", "look up", "find", "browse", "web", "internet",
                       "gather information", "sources", "latest", "current"]
        code_kws = ["code", "implement", "write", "script", "function", "program", "debug",
                     "test", "build", "develop", "refactor", "class", "method"]
        artifact_kws = ["create file", "generate file", "document", "report", "save",
                        "export", "output", "write to", "produce", "compile report",
                        "spreadsheet", "csv", "json", "schema", "presentation", "slides",
                        "website", "html", "pdf"]
        vision_kws = ["image", "photo", "picture", "screenshot", "visual", "analyze image",
                       "look at", "examine"]

        if has_image and any(kw in step_text for kw in vision_kws):
            return "vision"
        if any(kw in step_text for kw in search_kws):
            return "search"
        if any(kw in step_text for kw in artifact_kws):
            return "artifact"
        if any(kw in step_text for kw in code_kws):
            return "code"
        return "llm"

    def _task(self, kind: str, description: str, inputs: Dict[str, Any]) -> AgentTask:
        return AgentTask(
            id=str(uuid.uuid4())[:8],
            kind=kind,
            description=description,
            inputs=inputs
        )
