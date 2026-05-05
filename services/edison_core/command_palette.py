"""
Command palette / intent registry.

Phase 2 goal: a unified command palette across all tabs. The registry
collects user-facing actions (with NL aliases, target lane, and a target
endpoint or handler key) so the front-end can render a single Cmd-K
palette and so chat intent routing can resolve "make me a logo" or "open
the video tab" against the same set of canonical actions.

This module is data only. The front-end and chat router consume it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Command:
    id: str
    title: str
    category: str          # "chat", "branding", "video", "printing", "project", "system"
    description: str = ""
    aliases: List[str] = field(default_factory=list)   # NL phrases that should match
    endpoint: Optional[str] = None                     # backend route the front-end can call
    intent: Optional[str] = None                       # chat router intent label
    params: List[str] = field(default_factory=list)    # required NL slots e.g. ["business_name"]
    lane: Optional[str] = None                         # scheduler lane hint

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "description": self.description,
            "aliases": list(self.aliases),
            "endpoint": self.endpoint,
            "intent": self.intent,
            "params": list(self.params),
            "lane": self.lane,
        }


# ── Default registry ──────────────────────────────────────────────────

_DEFAULT_COMMANDS: List[Command] = [
    # Chat
    Command(
        id="chat.new",
        title="New chat",
        category="chat",
        description="Start a new conversation.",
        aliases=["new chat", "start a new conversation", "fresh chat"],
        endpoint="/chat/new",
        intent="chat.new",
    ),
    Command(
        id="chat.search",
        title="Search conversations",
        category="chat",
        description="Search across all of your chat history.",
        aliases=["search chats", "find a conversation", "look up a chat"],
        endpoint="/api/phase2/conversations/search",
        intent="chat.search",
        params=["query"],
    ),
    # Branding
    Command(
        id="branding.logo",
        title="Generate logo concepts",
        category="branding",
        description="Produce several logo concepts for a brand.",
        aliases=["make a logo", "logo concepts", "design a logo for"],
        endpoint="/api/branding/logo",
        intent="branding.logo",
        params=["business_name"],
        lane="image",
    ),
    Command(
        id="branding.package",
        title="Build branding package",
        category="branding",
        description="Generate a full branding package for a client.",
        aliases=["branding package", "brand kit", "create a brand"],
        endpoint="/api/branding/package",
        intent="branding.package",
        params=["business_name"],
        lane="image",
    ),
    Command(
        id="marketing.copy",
        title="Write marketing copy",
        category="branding",
        description="Generate ad/social copy for a project.",
        aliases=["write ad copy", "marketing copy", "social caption"],
        endpoint="/api/marketing/copy",
        intent="marketing.copy",
        params=["topic"],
    ),
    # Video
    Command(
        id="video.shotlist",
        title="Generate shot list",
        category="video",
        description="Produce a shot list / storyboard for a video project.",
        aliases=["shot list", "storyboard", "shooting plan"],
        endpoint="/api/phase3/video/shotlist",
        intent="video.shotlist",
        params=["topic"],
        lane="video",
    ),
    Command(
        id="video.export",
        title="Export video preset",
        category="video",
        description="Export a clip with a preset for a target channel.",
        aliases=["export for instagram", "tiktok export", "ad export"],
        endpoint="/api/phase3/video/export",
        intent="video.export",
        params=["preset"],
        lane="video",
    ),
    # Fabrication / printing
    Command(
        id="printing.keychain",
        title="Generate keychain",
        category="printing",
        description="Turn a logo or image into a printable keychain STL.",
        aliases=["make a keychain", "keychain stl", "logo keychain"],
        endpoint="/api/fabrication/keychain",
        intent="printing.keychain",
        params=["asset"],
        lane="cad",
    ),
    Command(
        id="printing.qa",
        title="Run mesh QA",
        category="printing",
        description="Check a mesh for manifoldness, wall thickness, and bounding-box sanity.",
        aliases=["check mesh", "mesh qa", "geometry check"],
        endpoint="/api/phase3/cad/qa",
        intent="printing.qa",
        params=["mesh_path"],
        lane="cad",
    ),
    # Project
    Command(
        id="project.new",
        title="New project",
        category="project",
        description="Create a new client project workspace.",
        aliases=["new project", "create project", "start a project for"],
        endpoint="/api/projects",
        intent="project.new",
        params=["name"],
    ),
    Command(
        id="project.open",
        title="Open project",
        category="project",
        description="Switch the workspace context to a specific project.",
        aliases=["open project", "switch to project"],
        endpoint="/api/projects",
        intent="project.open",
        params=["name"],
    ),
    # System
    Command(
        id="system.jobs",
        title="Open jobs center",
        category="system",
        description="View all in-flight and recent jobs across modes.",
        aliases=["show jobs", "jobs center", "what's running"],
        endpoint="/api/phase2/jobs",
        intent="system.jobs",
    ),
    Command(
        id="system.scheduler",
        title="GPU scheduler telemetry",
        category="system",
        description="Inspect lane usage and queue depth.",
        aliases=["gpu scheduler", "lane telemetry", "queue depth"],
        endpoint="/api/phase1/scheduler/telemetry",
        intent="system.scheduler",
    ),
]


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class CommandPalette:
    """In-memory command registry with fuzzy NL matching."""

    def __init__(self, commands: Optional[List[Command]] = None) -> None:
        self.commands: Dict[str, Command] = {}
        for cmd in (commands or _DEFAULT_COMMANDS):
            self.register(cmd)

    def register(self, cmd: Command) -> None:
        self.commands[cmd.id] = cmd

    def all(self) -> List[Command]:
        return list(self.commands.values())

    def by_category(self, category: str) -> List[Command]:
        return [c for c in self.commands.values() if c.category == category]

    def search(self, query: str, *, limit: int = 8) -> List[Command]:
        """Score-based NL search across title, aliases, and description."""
        if not query.strip():
            return []
        q_tokens = set(_tokens(query))
        if not q_tokens:
            return []
        scored: List[tuple[int, Command]] = []
        for cmd in self.commands.values():
            haystacks = [cmd.title, cmd.description, *cmd.aliases]
            score = 0
            for h in haystacks:
                h_tokens = set(_tokens(h))
                overlap = len(q_tokens & h_tokens)
                if overlap:
                    score += overlap * (3 if h is cmd.title else 1)
                if h.lower() == query.lower():
                    score += 10
            if score:
                scored.append((score, cmd))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:limit]]


_palette: Optional[CommandPalette] = None


def get_palette() -> CommandPalette:
    global _palette
    if _palette is None:
        _palette = CommandPalette()
    return _palette
