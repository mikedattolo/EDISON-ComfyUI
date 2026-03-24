"""
API contracts for orchestration, projects, artifacts, and safety.
Scaffolding for agent-driven OS features.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

AgentMode = Literal["instant", "thinking", "agent", "swarm", "work"]


class WorkStep(BaseModel):
    """A single step in a work mode task plan."""
    id: int
    title: str
    description: str = ""
    kind: Literal["llm", "search", "code", "artifact", "vision", "tool", "comfyui"] = "llm"
    status: Literal["pending", "running", "completed", "failed", "skipped"] = "pending"
    result: Optional[str] = None
    error: Optional[str] = None
    artifacts: List[str] = Field(default_factory=list)
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    elapsed_ms: Optional[int] = None


class WorkPlanResponse(BaseModel):
    """Full work mode execution plan with steps."""
    task: str
    steps: List[WorkStep]
    project_id: Optional[str] = None
    total_steps: int = 0
    completed_steps: int = 0
    status: Literal["planning", "executing", "completed", "failed"] = "planning"


class AgentTask(BaseModel):
    id: str
    kind: Literal["llm", "vision", "code", "web", "comfyui", "tool", "artifact"]
    description: str
    inputs: Dict[str, Any] = Field(default_factory=dict)


class AgentTaskResult(BaseModel):
    id: str
    kind: str
    output: Optional[Any] = None
    error: Optional[str] = None


class AgentPlanRequest(BaseModel):
    goal: str
    mode: AgentMode = "instant"
    has_image: bool = False
    project_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class AgentPlanResponse(BaseModel):
    mode: AgentMode
    parallel: bool
    rationale: List[str]
    tasks: List[AgentTask]
    policy: Dict[str, Any] = Field(default_factory=dict)


class AgentExecuteRequest(BaseModel):
    plan: AgentPlanResponse
    project_id: Optional[str] = None
    dry_run: bool = True
    images: Optional[List[str]] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class AgentExecuteResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    message: str
    results: Optional[List[AgentTaskResult]] = None


class ProjectCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    template: Optional[str] = None

    client_id: Optional[str] = None
    service_types: List[Literal["branding", "printing", "video", "marketing", "mixed"]] = Field(default_factory=list)
    due_date: Optional[str] = None
    status: Literal["draft", "planned", "active", "in_review", "approved", "completed", "archived"] = "planned"
    notes: str = ""
    tags: List[str] = Field(default_factory=list)
    assets: List[Dict[str, Any]] = Field(default_factory=list)
    tasks: List[Dict[str, Any]] = Field(default_factory=list)
    approvals: List[Dict[str, Any]] = Field(default_factory=list)
    deliverables: List[Dict[str, Any]] = Field(default_factory=list)


class ProjectUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    template: Optional[str] = None
    client_id: Optional[str] = None
    service_types: Optional[List[Literal["branding", "printing", "video", "marketing", "mixed"]]] = None
    due_date: Optional[str] = None
    status: Optional[Literal["draft", "planned", "active", "in_review", "approved", "completed", "archived"]] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None
    assets: Optional[List[Dict[str, Any]]] = None
    tasks: Optional[List[Dict[str, Any]]] = None
    approvals: Optional[List[Dict[str, Any]]] = None
    deliverables: Optional[List[Dict[str, Any]]] = None


class ProjectResponse(BaseModel):
    project_id: str
    name: str
    slug: str = ""
    description: Optional[str] = None
    template: Optional[str] = None
    client_id: Optional[str] = None
    client_name: Optional[str] = None
    client_slug: Optional[str] = None
    service_types: List[str] = Field(default_factory=list)
    due_date: Optional[str] = None
    status: str = "planned"
    notes: str = ""
    tags: List[str] = Field(default_factory=list)
    assets: List[Dict[str, Any]] = Field(default_factory=list)
    tasks: List[Dict[str, Any]] = Field(default_factory=list)
    approvals: List[Dict[str, Any]] = Field(default_factory=list)
    deliverables: List[Dict[str, Any]] = Field(default_factory=list)
    workspace_paths: Dict[str, str] = Field(default_factory=dict)
    root_path: str
    created_at: str
    updated_at: Optional[str] = None
    config_path: str


class BrandingGenerationRequest(BaseModel):
    business_name: str
    client_id: Optional[str] = None
    project_id: Optional[str] = None
    industry: str = ""
    audience: str = ""
    prompt: str = ""
    tone: str = "confident"
    style_keywords: List[str] = Field(default_factory=list)
    deliverable_count: int = Field(default=5, ge=1, le=12)
    include_moodboard: bool = True


class MarketingCopyRequest(BaseModel):
    business_name: str
    client_id: Optional[str] = None
    project_id: Optional[str] = None
    industry: str = ""
    audience: str = ""
    prompt: str = ""
    tone: str = "confident"
    channels: List[str] = Field(default_factory=list)
    copy_types: List[Literal[
        "ad_copy",
        "social_captions",
        "email_campaign",
        "business_description",
        "product_copy",
        "website_hero_text",
    ]] = Field(default_factory=list)


class ArtifactGenerateRequest(BaseModel):
    project_id: str
    kind: Literal["document", "code", "schema", "ui", "presentation", "spreadsheet", "website"]
    prompt: str
    format: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ArtifactGenerateResponse(BaseModel):
    artifact_id: str
    kind: str
    output_path: str
    status: Literal["queued", "generated", "failed"]


class SafetyAssessmentRequest(BaseModel):
    action_type: Literal["filesystem", "network", "process", "system", "model"]
    risk_score: float = Field(ge=0, le=1)
    summary: str


class SafetyAssessmentResponse(BaseModel):
    allowed: bool
    requires_confirmation: bool
    rationale: str
