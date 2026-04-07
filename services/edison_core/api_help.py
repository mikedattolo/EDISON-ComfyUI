"""
api_help.py — Help and documentation API for EDISON.

Provides a searchable help index that the assistant can query
to answer user questions about EDISON's capabilities.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(tags=["help"])

# In-memory help index — matches the topics in web/help.html
HELP_TOPICS = [
    {
        "id": "chat",
        "title": "Chat & Conversation",
        "category": "core",
        "keywords": ["chat", "conversation", "talk", "ask", "question", "help"],
        "summary": "EDISON is a conversational AI assistant. Type anything in the chat box. Supports follow-up questions and context across the conversation.",
        "examples": [
            "What are some marketing strategies for a pizza restaurant?",
            "Explain the difference between branding and marketing",
        ],
    },
    {
        "id": "modes",
        "title": "Chat Modes",
        "category": "core",
        "keywords": ["mode", "instant", "reasoning", "thinking", "code", "agent", "swarm", "work", "auto"],
        "summary": "Modes: auto (EDISON picks), instant (fast), chat (standard), reasoning (deep analysis), thinking (chain-of-thought), code (programming), agent (tool-using), work (multi-step tasks), swarm (multi-agent collaboration).",
        "examples": [],
    },
    {
        "id": "branding",
        "title": "Branding & Design",
        "category": "business",
        "keywords": ["brand", "logo", "palette", "typography", "slogan", "tagline", "moodboard", "style guide", "brand voice", "design"],
        "summary": "Create complete brand packages: logo concepts, color palettes, typography, slogans, moodboards, style guides, and brand voice definitions. Available via chat or /branding page.",
        "examples": [
            "Generate a branding package for Adoro Pizza",
            "Create 5 logo concepts for a tech startup called NovaByte",
        ],
    },
    {
        "id": "marketing",
        "title": "Marketing & Copywriting",
        "category": "business",
        "keywords": ["marketing", "copy", "ad", "caption", "email", "campaign", "headline", "product", "promo"],
        "summary": "Generate marketing copy: ad copy, social captions, email campaigns, business descriptions, website text, and product marketing.",
        "examples": [
            "Write marketing copy for a coffee shop grand opening",
            "Generate 10 Instagram captions for a fitness brand",
        ],
    },
    {
        "id": "social",
        "title": "Social Media Posting",
        "category": "business",
        "keywords": ["social", "instagram", "facebook", "tiktok", "linkedin", "post", "schedule", "publish", "campaign"],
        "summary": "Draft, schedule, and manage social media posts. Platforms: Instagram, Facebook, TikTok, LinkedIn, Google Business Profile. Track drafts, scheduled posts, and published content.",
        "examples": [
            "Create a social post for Instagram about our summer menu",
            "Schedule the post for next Friday at 2 PM",
        ],
    },
    {
        "id": "projects",
        "title": "Project Management",
        "category": "business",
        "keywords": ["project", "client", "task", "deliverable", "asset", "workspace", "organize"],
        "summary": "Organize work into clients and projects. Track tasks, assets, and deliverables. Service types: branding, printing, video, marketing, mixed. Visit /projects for the dashboard.",
        "examples": [
            "Create a project called Summer Campaign for Adoro Pizza",
            "List my projects",
        ],
    },
    {
        "id": "printing",
        "title": "3D Printing & Fabrication",
        "category": "fabrication",
        "keywords": ["print", "3d", "stl", "slice", "keychain", "plaque", "nameplate", "signage", "printer", "fabrication"],
        "summary": "Manage 3D printers, slice models with custom parameters, track print jobs. Supports logo-to-STL conversion, keychain/plaque/nameplate generators. Visit /printing.",
        "examples": [
            "Slice the model at 0.15mm layer height with 30% infill",
        ],
    },
    {
        "id": "video",
        "title": "Video Production",
        "category": "creative",
        "keywords": ["video", "storyboard", "shot list", "script", "footage", "clip", "render", "promo"],
        "summary": "Plan and produce video content: storyboards, shot lists, scripts, social video planning, promo campaigns. Visit /video-editor.",
        "examples": [
            "Create a storyboard for a 30-second pizza commercial",
        ],
    },
    {
        "id": "images",
        "title": "Image Generation",
        "category": "creative",
        "keywords": ["image", "generate", "picture", "photo", "illustration", "art"],
        "summary": "Generate images using AI models directly from chat. Describe what you want and EDISON generates it. Results appear in the gallery.",
        "examples": [
            "Generate an image of a cozy Italian restaurant interior",
        ],
    },
    {
        "id": "search",
        "title": "Web Search & Research",
        "category": "tools",
        "keywords": ["search", "web", "research", "browse", "url", "summarize", "deep search"],
        "summary": "Search the web for current information. Deep search for thorough research. Summarize websites by URL. Browser automation for interactive pages.",
        "examples": [
            "Search for the latest trends in restaurant marketing 2026",
            "Summarize https://example.com",
        ],
    },
    {
        "id": "connectors",
        "title": "Connectors & Integrations",
        "category": "system",
        "keywords": ["connector", "integration", "api", "github", "google", "slack", "notion", "oauth"],
        "summary": "Connect EDISON to external services: GitHub, Google, Slack, Notion, Dropbox, Discord, and custom APIs. Visit /connectors to manage.",
        "examples": [],
    },
    {
        "id": "workflow",
        "title": "Workflow Orchestration",
        "category": "advanced",
        "keywords": ["workflow", "orchestration", "multi-step", "plan", "decompose", "coordinate"],
        "summary": "Complex multi-step requests are automatically decomposed into typed steps combining branding, marketing, fabrication, video, and project management. Progress is tracked and outputs saved.",
        "examples": [
            "Make a branding package for Adoro Pizza and generate marketing copy for their social media campaign",
        ],
    },
    {
        "id": "auth",
        "title": "Authentication & Roles",
        "category": "system",
        "keywords": ["auth", "login", "register", "role", "admin", "designer", "user", "password"],
        "summary": "Optional multi-user auth with roles: admin, designer, fabricator, editor, client_viewer. Disabled by default — enable with EDISON_AUTH_ENABLED=true. First user gets admin.",
        "examples": [],
    },
    {
        "id": "system",
        "title": "System Diagnostics",
        "category": "system",
        "keywords": ["system", "status", "diagnostic", "readiness", "route", "endpoint", "config"],
        "summary": "EDISON can inspect its own system: readiness, loaded models, routes, configuration, storage, and service status. Use /api/system/capabilities and /api/system/readiness.",
        "examples": [
            "What's the current system status?",
            "Which endpoints handle printing?",
        ],
    },
]


@router.get("/api/help")
async def get_help(q: Optional[str] = None, category: Optional[str] = None):
    """
    Search help topics. Returns all topics if no query, or filtered by keyword/category.
    The assistant can use this to answer questions about EDISON's capabilities.
    """
    topics = HELP_TOPICS
    if category:
        topics = [t for t in topics if t["category"] == category.lower()]
    if q:
        q_lower = q.lower()
        scored = []
        for t in topics:
            score = 0
            if q_lower in t["title"].lower():
                score += 3
            if q_lower in t["summary"].lower():
                score += 2
            if any(q_lower in kw for kw in t["keywords"]):
                score += 2
            # partial word matching
            words = q_lower.split()
            for word in words:
                if any(word in kw for kw in t["keywords"]):
                    score += 1
                if word in t["summary"].lower():
                    score += 1
            if score > 0:
                scored.append((score, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        topics = [t for _, t in scored]

    return {
        "topics": topics,
        "total": len(topics),
        "categories": sorted(set(t["category"] for t in HELP_TOPICS)),
    }


@router.get("/api/help/{topic_id}")
async def get_help_topic(topic_id: str):
    """Get a single help topic by ID."""
    topic = next((t for t in HELP_TOPICS if t["id"] == topic_id), None)
    if not topic:
        return {"error": "Topic not found", "available": [t["id"] for t in HELP_TOPICS]}
    return {"topic": topic}
