"""
EDISON Runtime Layer
====================

Shared runtime modules that extract business logic out of app.py route handlers
into reusable, composable, and testable components.

The runtime layer provides:
- chat_runtime: Unified chat pipeline shared by all entry points
- context_runtime: Layered context assembly (conversation, memory, task, RAG)
- routing_runtime: Intent classification and mode/model/tool decisions
- tool_runtime: Tool registry, validation, execution loop
- response_runtime: Response formatting and quality checks
- task_runtime: Persistent task state across turns
- artifact_runtime: Artifact registry and lifecycle
- workspace_runtime: Logical workspace/project scoping
- search_runtime: Multi-stage research pipeline
- browser_runtime: Browser session management and agent execution
- model_runtime: Task-aware model selection and fallback
- quality_runtime: Lightweight response quality/review pass
"""

__all__ = [
    "chat_runtime",
    "context_runtime",
    "routing_runtime",
    "tool_runtime",
    "response_runtime",
    "task_runtime",
    "artifact_runtime",
    "workspace_runtime",
    "search_runtime",
    "browser_runtime",
    "model_runtime",
    "quality_runtime",
]
