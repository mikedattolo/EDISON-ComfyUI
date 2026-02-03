"""
API contracts for orchestration, projects, artifacts, and safety.
Scaffolding for agent-driven OS features.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

AgentMode = Literal["instant", "thinking", "agent", "swarm"]


class AgentTask(BaseModel):
    id: str
    kind: Literal["llm", "vision", "code", "web", "comfyui", "tool"]
    description: str
    inputs: Dict[str, Any] = Field(default_factory=dict)


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


class AgentExecuteResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    message: str


class ProjectCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    template: Optional[str] = None


class ProjectResponse(BaseModel):
    project_id: str
    name: str
    description: Optional[str]
    root_path: str
    created_at: str
    config_path: str


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
