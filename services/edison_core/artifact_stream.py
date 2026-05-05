"""
Artifact streaming helpers.

Phase 1 goal: stream artifacts (documents, code patches, designs, etc.) to
the front-end *while they are being produced*, rather than only at final
completion. This module provides:

* :class:`ArtifactStream` — an asyncio-based stream that producers push
  partial deltas into and consumers iterate over.
* :func:`sse_lines` — convert a stream into Server-Sent Events lines so it
  can be returned directly from a FastAPI endpoint.
* Lightweight revision tracking so consumers can request "compare" or
  "restore" against a previous version.

This is purely additive; existing artifact code paths can keep returning
final blobs and adopt streaming incrementally.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional


@dataclass
class ArtifactRevision:
    revision_id: str
    created_at: float
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "created_at": self.created_at,
            "content": self.content,
            "metadata": self.metadata,
        }


class ArtifactStream:
    """Producer/consumer stream for partial artifact updates.

    Producers call :meth:`push` (deltas) and :meth:`commit` (final revision).
    Consumers iterate via ``async for event in stream``.
    """

    def __init__(
        self,
        *,
        artifact_id: Optional[str] = None,
        kind: str = "document",
        title: Optional[str] = None,
    ) -> None:
        self.artifact_id = artifact_id or f"art_{uuid.uuid4().hex[:10]}"
        self.kind = kind
        self.title = title
        self._queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._buffer: List[str] = []
        self._revisions: List[ArtifactRevision] = []
        self._closed = False
        self._created_at = time.time()

    @property
    def current_text(self) -> str:
        return "".join(self._buffer)

    @property
    def revisions(self) -> List[ArtifactRevision]:
        return list(self._revisions)

    async def push(self, delta: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Append a partial delta and notify consumers."""
        if self._closed:
            raise RuntimeError("ArtifactStream is closed")
        if not delta:
            return
        self._buffer.append(delta)
        await self._queue.put({
            "type": "artifact.delta",
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "delta": delta,
            "metadata": metadata or {},
            "ts": time.time(),
        })

    async def commit(self, *, metadata: Optional[Dict[str, Any]] = None) -> ArtifactRevision:
        """Finalize the current buffer as a new revision."""
        revision = ArtifactRevision(
            revision_id=f"rev_{uuid.uuid4().hex[:10]}",
            created_at=time.time(),
            content=self.current_text,
            metadata=metadata or {},
        )
        self._revisions.append(revision)
        await self._queue.put({
            "type": "artifact.revision",
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "revision": revision.to_dict(),
            "ts": time.time(),
        })
        return revision

    async def error(self, message: str) -> None:
        await self._queue.put({
            "type": "artifact.error",
            "artifact_id": self.artifact_id,
            "error": message,
            "ts": time.time(),
        })

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._queue.put({
            "type": "artifact.done",
            "artifact_id": self.artifact_id,
            "ts": time.time(),
        })

    def __aiter__(self) -> AsyncIterator[Dict[str, Any]]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[Dict[str, Any]]:
        while True:
            event = await self._queue.get()
            yield event
            if event.get("type") == "artifact.done":
                return

    # ── revision helpers ────────────────────────────────────────────

    def get_revision(self, revision_id: str) -> Optional[ArtifactRevision]:
        for rev in self._revisions:
            if rev.revision_id == revision_id:
                return rev
        return None

    def diff(self, a: str, b: str) -> Dict[str, Any]:
        """Return a coarse diff descriptor between two revision ids.

        We do not embed an actual diff library here to keep dependencies
        minimal; the front-end is expected to render the diff. We do report
        whether the two revisions exist and the line counts so the UI can
        warn if a revision was rolled away.
        """
        ra = self.get_revision(a)
        rb = self.get_revision(b)
        return {
            "a": a,
            "b": b,
            "a_exists": ra is not None,
            "b_exists": rb is not None,
            "a_lines": ra.content.count("\n") + 1 if ra else 0,
            "b_lines": rb.content.count("\n") + 1 if rb else 0,
        }


async def sse_lines(stream: ArtifactStream) -> AsyncIterator[str]:
    """Yield Server-Sent Events lines for a FastAPI ``StreamingResponse``."""
    async for event in stream:
        yield f"event: {event['type']}\n"
        yield f"data: {json.dumps(event, default=str)}\n\n"
    yield "data: [DONE]\n\n"
