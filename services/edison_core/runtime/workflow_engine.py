"""
workflow_engine.py — Unified workflow orchestration for EDISON business operations.

Decomposes complex multi-step requests into typed action plans that coordinate
branding, marketing, fabrication, video, project management, and standard LLM steps.

Integrates with existing work-mode step execution while adding business-domain
awareness and automatic project/artifact linkage.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StepKind(str, Enum):
    LLM = "llm"
    SEARCH = "search"
    CODE = "code"
    ARTIFACT = "artifact"
    BRANDING = "branding"
    MARKETING = "marketing"
    FABRICATION = "fabrication"
    VIDEO = "video"
    PROJECT = "project"
    SOCIAL = "social"
    FILE_OP = "file_op"


# Keywords that signal each step kind
_KIND_SIGNALS: Dict[StepKind, List[str]] = {
    StepKind.BRANDING: [
        "brand", "logo", "palette", "typography", "moodboard", "style guide",
        "slogan", "tagline", "brand voice", "branding package", "brand identity",
    ],
    StepKind.MARKETING: [
        "marketing", "ad copy", "social caption", "email campaign", "promo",
        "campaign", "copy", "copywriting", "headline", "call to action",
        "marketing plan", "content calendar",
    ],
    StepKind.FABRICATION: [
        "3d print", "stl", "slice", "keychain", "plaque", "nameplate",
        "signage", "fabricat", "extrude", "emboss", "deboss", "cnc",
    ],
    StepKind.VIDEO: [
        "video", "storyboard", "shot list", "script", "promo video",
        "editing", "footage", "clip", "render video",
    ],
    StepKind.PROJECT: [
        "create project", "create client", "client folder", "project workspace",
        "organize assets", "link asset", "update project", "project status",
    ],
    StepKind.SOCIAL: [
        "social post", "post to instagram", "post to facebook", "post to tiktok",
        "post to linkedin", "schedule post", "draft post", "social media",
        "content calendar", "publish post", "social campaign",
    ],
    StepKind.FILE_OP: [
        "save file", "create file", "move file", "export", "download",
        "organize files", "upload",
    ],
    StepKind.SEARCH: [
        "search", "research", "look up", "find information", "browse", "web",
    ],
    StepKind.CODE: [
        "code", "implement", "write code", "script", "function", "program",
        "debug", "fix bug",
    ],
    StepKind.ARTIFACT: [
        "document", "report", "csv", "json", "template", "mockup",
        "business card", "flyer", "poster", "banner", "menu",
    ],
}

# Maps step kinds to the tool(s) they should use
_KIND_TOOL_MAP: Dict[StepKind, List[str]] = {
    StepKind.BRANDING: ["generate_brand_package", "create_branding_client", "list_branding_clients"],
    StepKind.MARKETING: ["generate_marketing_copy"],
    StepKind.FABRICATION: ["slice_model"],
    StepKind.VIDEO: ["generate_video"],
    StepKind.PROJECT: ["create_project", "list_projects"],
    StepKind.SOCIAL: ["create_social_post", "schedule_social_post", "list_social_posts"],
}


@dataclass
class WorkflowStep:
    id: int
    title: str
    kind: StepKind = StepKind.LLM
    description: str = ""
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[int] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed, skipped
    result: Optional[str] = None
    error: Optional[str] = None
    artifacts: List[Dict] = field(default_factory=list)
    elapsed_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "kind": self.kind.value if isinstance(self.kind, StepKind) else self.kind,
            "description": self.description,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "depends_on": self.depends_on,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "artifacts": self.artifacts,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class WorkflowPlan:
    workflow_id: str
    goal: str
    steps: List[WorkflowStep]
    project_id: Optional[str] = None
    client_id: Optional[str] = None
    created_at: float = 0.0
    status: str = "planned"  # planned, executing, completed, failed

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()
        if not self.workflow_id:
            self.workflow_id = str(uuid.uuid4())[:8]

    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "project_id": self.project_id,
            "client_id": self.client_id,
            "created_at": self.created_at,
            "status": self.status,
            "step_count": len(self.steps),
            "completed_count": sum(1 for s in self.steps if s.status == "completed"),
        }


def classify_step(text: str) -> StepKind:
    """Classify a step description into a StepKind using keyword matching."""
    text_lower = text.lower()
    scores: Dict[StepKind, int] = {}
    for kind, keywords in _KIND_SIGNALS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[kind] = score
    if not scores:
        return StepKind.LLM
    return max(scores, key=scores.get)


def select_tool_for_step(step: WorkflowStep) -> Optional[str]:
    """Pick the best tool name for a step based on its kind and title."""
    tools = _KIND_TOOL_MAP.get(step.kind, [])
    if not tools:
        return None
    title_lower = step.title.lower()

    # Heuristic: pick the most relevant tool
    if step.kind == StepKind.BRANDING:
        if "client" in title_lower:
            return "create_branding_client"
        if "list" in title_lower:
            return "list_branding_clients"
        return "generate_brand_package"
    if step.kind == StepKind.PROJECT:
        if "list" in title_lower or "status" in title_lower:
            return "list_projects"
        return "create_project"
    if step.kind == StepKind.SOCIAL:
        if "schedule" in title_lower:
            return "schedule_social_post"
        if "list" in title_lower:
            return "list_social_posts"
        return "create_social_post"
    return tools[0]


def plan_workflow(message: str, raw_steps: List[str],
                  project_id: Optional[str] = None,
                  client_id: Optional[str] = None) -> WorkflowPlan:
    """
    Build a WorkflowPlan from a user message and raw step descriptions.

    This is the business-aware counterpart to _plan_work_steps().
    It classifies each step, assigns tools, and builds dependency chains.
    """
    steps: List[WorkflowStep] = []
    for i, step_text in enumerate(raw_steps):
        kind = classify_step(step_text)
        ws = WorkflowStep(
            id=i + 1,
            title=step_text.strip(),
            kind=kind,
        )
        ws.tool_name = select_tool_for_step(ws)
        # Simple linear dependency: each step depends on the previous
        if i > 0:
            ws.depends_on = [i]
        steps.append(ws)

    plan = WorkflowPlan(
        workflow_id=str(uuid.uuid4())[:8],
        goal=message[:500],
        steps=steps,
        project_id=project_id,
        client_id=client_id,
    )
    logger.info(f"Workflow plan created: {plan.workflow_id} with {len(steps)} steps")
    return plan


def plan_from_message(message: str,
                      project_id: Optional[str] = None,
                      client_id: Optional[str] = None) -> WorkflowPlan:
    """
    Analyse a user message and build a workflow plan without LLM assistance.
    Uses pattern matching to decompose common business requests.

    For complex/ambiguous requests, returns a single-step plan that defers to the LLM.
    """
    msg_lower = message.lower().strip()

    # ── Multi-step compound patterns (check FIRST) ──────────────
    compound_steps: List[WorkflowStep] = []
    step_id = 0

    if re.search(r"brand(?:ing)?\s+package", msg_lower):
        step_id += 1
        compound_steps.append(WorkflowStep(
            id=step_id, title="Generate branding package",
            kind=StepKind.BRANDING, tool_name="generate_brand_package",
        ))

    if re.search(r"logo\s+concept", msg_lower):
        step_id += 1
        compound_steps.append(WorkflowStep(
            id=step_id, title="Generate logo concepts",
            kind=StepKind.BRANDING, tool_name="generate_brand_package",
            depends_on=[step_id - 1] if step_id > 1 else [],
        ))

    if re.search(r"marketing\s+(?:copy|plan|campaign)", msg_lower):
        step_id += 1
        compound_steps.append(WorkflowStep(
            id=step_id, title="Generate marketing copy",
            kind=StepKind.MARKETING, tool_name="generate_marketing_copy",
            depends_on=[step_id - 1] if step_id > 1 else [],
        ))

    if re.search(r"(?:storyboard|shot\s+list|video\s+plan)", msg_lower):
        step_id += 1
        compound_steps.append(WorkflowStep(
            id=step_id, title="Generate video production plan",
            kind=StepKind.VIDEO, tool_name="generate_video",
            depends_on=[step_id - 1] if step_id > 1 else [],
        ))

    if re.search(r"(?:keychain|plaque|nameplate|signage|stl)\b", msg_lower):
        step_id += 1
        compound_steps.append(WorkflowStep(
            id=step_id, title="Prepare fabrication output",
            kind=StepKind.FABRICATION, tool_name="slice_model",
            depends_on=[step_id - 1] if step_id > 1 else [],
        ))

    if re.search(r"(?:organiz|folder|workspace|save.*project)", msg_lower):
        step_id += 1
        compound_steps.append(WorkflowStep(
            id=step_id, title="Organize outputs into project",
            kind=StepKind.PROJECT, tool_name="create_project",
            depends_on=[step_id - 1] if step_id > 1 else [],
        ))

    if re.search(r"(?:social\s+post|post\s+to\s+(?:instagram|facebook|tiktok|linkedin)|draft\s+post|schedule\s+post)", msg_lower):
        step_id += 1
        compound_steps.append(WorkflowStep(
            id=step_id, title="Create social media post",
            kind=StepKind.SOCIAL, tool_name="create_social_post",
            depends_on=[step_id - 1] if step_id > 1 else [],
        ))

    if len(compound_steps) >= 2:
        return WorkflowPlan(
            workflow_id=str(uuid.uuid4())[:8],
            goal=message[:500],
            steps=compound_steps,
            project_id=project_id,
            client_id=client_id,
        )

    # ── Direct single-action patterns ─────────────────────────────
    single_action_patterns = [
        (r"create\s+(?:a\s+)?(?:branding\s+)?client\s+(?:called|named|for)\s+(.+)",
         StepKind.BRANDING, "create_branding_client"),
        (r"(?:generate|create|make)\s+(?:a\s+)?brand(?:ing)?\s+package\s+for\s+(.+)",
         StepKind.BRANDING, "generate_brand_package"),
        (r"(?:generate|write|create)\s+marketing\s+copy\s+for\s+(.+)",
         StepKind.MARKETING, "generate_marketing_copy"),
        (r"list\s+(?:my\s+)?(?:all\s+)?projects",
         StepKind.PROJECT, "list_projects"),
        (r"list\s+(?:my\s+)?(?:all\s+)?clients",
         StepKind.BRANDING, "list_branding_clients"),
        (r"create\s+(?:a\s+)?project\s+(?:called|named|for)\s+(.+)",
         StepKind.PROJECT, "create_project"),
        (r"slice\s+(?:the\s+)?model",
         StepKind.FABRICATION, "slice_model"),
        (r"(?:create|draft|write)\s+(?:a\s+)?(?:social\s+)?post\s+(?:for|on|to)\s+(\w+)",
         StepKind.SOCIAL, "create_social_post"),
        (r"list\s+(?:my\s+)?(?:all\s+)?(?:social\s+)?posts",
         StepKind.SOCIAL, "list_social_posts"),
        (r"schedule\s+(?:a\s+)?(?:social\s+)?post",
         StepKind.SOCIAL, "schedule_social_post"),
    ]

    for pattern, kind, tool_name in single_action_patterns:
        m = re.search(pattern, msg_lower)
        if m:
            step = WorkflowStep(
                id=1,
                title=message.strip(),
                kind=kind,
                tool_name=tool_name,
            )
            return WorkflowPlan(
                workflow_id=str(uuid.uuid4())[:8],
                goal=message[:500],
                steps=[step],
                project_id=project_id,
                client_id=client_id,
            )

    # ── Fallback: single LLM step (let the existing work-mode handle it) ───
    return WorkflowPlan(
        workflow_id=str(uuid.uuid4())[:8],
        goal=message[:500],
        steps=[WorkflowStep(id=1, title=message[:200], kind=StepKind.LLM)],
        project_id=project_id,
        client_id=client_id,
    )


def workflow_step_to_work_step(ws: WorkflowStep) -> dict:
    """Convert a WorkflowStep to the dict format expected by _execute_work_step."""
    kind_str = ws.kind.value if isinstance(ws.kind, StepKind) else ws.kind
    # Map business kinds to the base kinds that _execute_work_step handles
    base_kind_map = {
        "branding": "llm",
        "marketing": "llm",
        "fabrication": "llm",
        "video": "llm",
        "project": "llm",
        "social": "llm",
        "file_op": "artifact",
    }
    base_kind = base_kind_map.get(kind_str, kind_str)
    return {
        "id": ws.id,
        "title": ws.title,
        "description": ws.description,
        "kind": base_kind,
        "tool_name": ws.tool_name,
        "tool_args": ws.tool_args,
        "status": ws.status,
        "result": ws.result,
        "error": ws.error,
        "artifacts": ws.artifacts,
        "search_results": [],
        "elapsed_ms": ws.elapsed_ms,
    }


def summarize_workflow(plan: WorkflowPlan) -> str:
    """Return a human-readable summary of a completed workflow."""
    lines = [f"**Workflow: {plan.goal[:100]}**"]
    for step in plan.steps:
        icon = {"completed": "✅", "failed": "❌", "skipped": "⏭️", "running": "🔄"}.get(step.status, "⬜")
        kind_label = step.kind.value if isinstance(step.kind, StepKind) else step.kind
        timing = f" ({step.elapsed_ms:.0f}ms)" if step.elapsed_ms else ""
        lines.append(f"{icon} Step {step.id} [{kind_label}]: {step.title}{timing}")
        if step.error:
            lines.append(f"   Error: {step.error}")
    completed = sum(1 for s in plan.steps if s.status == "completed")
    lines.append(f"\n{completed}/{len(plan.steps)} steps completed.")
    return "\n".join(lines)
