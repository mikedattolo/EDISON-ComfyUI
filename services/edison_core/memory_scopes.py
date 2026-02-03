"""
Scoped memory scaffolding for global vs project RAG namespaces.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MemoryScope:
    scope_id: str
    kind: str  # global | project | session


class MemoryScopeManager:
    def __init__(self, global_scope: str = "global"):
        self.global_scope = global_scope

    def get_scope(
        self,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        chat_id: Optional[str] = None
    ) -> MemoryScope:
        if project_id:
            return MemoryScope(scope_id=f"project:{project_id}", kind="project")
        if chat_id:
            return MemoryScope(scope_id=f"chat:{chat_id}", kind="chat")
        if session_id:
            return MemoryScope(scope_id=f"session:{session_id}", kind="session")
        return MemoryScope(scope_id=self.global_scope, kind="global")
