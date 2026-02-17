"""
Memory Models for EDISON
Three-tier memory: Profile, Episodic, Semantic
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time
import uuid


class MemoryType(str, Enum):
    PROFILE = "profile"      # Stable facts: preferences, hardware, projects
    EPISODIC = "episodic"    # Timestamped conversation/event summaries
    SEMANTIC = "semantic"    # Distilled reusable procedures/workflows


@dataclass
class MemoryEntry:
    """A single memory entry in the three-tier system."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: MemoryType = MemoryType.EPISODIC
    key: Optional[str] = None               # For profile items (e.g. "preferred_language")
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.8
    tags: List[str] = field(default_factory=list)
    source_conversation_id: Optional[str] = None
    pinned: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "memory_type": self.memory_type.value,
            "key": self.key,
            "content": self.content,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "tags": self.tags,
            "source_conversation_id": self.source_conversation_id,
            "pinned": self.pinned,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            memory_type=MemoryType(d.get("memory_type", "episodic")),
            key=d.get("key"),
            content=d.get("content", ""),
            timestamp=d.get("timestamp", time.time()),
            confidence=d.get("confidence", 0.8),
            tags=d.get("tags", []),
            source_conversation_id=d.get("source_conversation_id"),
            pinned=d.get("pinned", False),
            metadata=d.get("metadata", {}),
        )
