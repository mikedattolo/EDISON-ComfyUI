"""
api_social.py — Social media connector and scheduling router for EDISON.

Provides endpoints for managing social media platform connections,
drafting posts, scheduling content, and tracking post status.

Platforms: Instagram, Facebook, TikTok, LinkedIn, Google Business Profile.

Storage: JSON file at config/integrations/social_posts.json
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["social"])

REPO_ROOT = Path(__file__).resolve().parents[2]
SOCIAL_DIR = REPO_ROOT / "config" / "integrations"
SOCIAL_POSTS_FILE = SOCIAL_DIR / "social_posts.json"
CONNECTORS_FILE = SOCIAL_DIR / "connectors.json"

# Supported social platforms
SOCIAL_PLATFORMS = {
    "instagram": {
        "name": "Instagram",
        "post_types": ["image", "carousel", "reel", "story"],
        "max_caption_length": 2200,
        "supports_scheduling": True,
    },
    "facebook": {
        "name": "Facebook",
        "post_types": ["text", "image", "video", "link", "carousel"],
        "max_caption_length": 63206,
        "supports_scheduling": True,
    },
    "tiktok": {
        "name": "TikTok",
        "post_types": ["video"],
        "max_caption_length": 2200,
        "supports_scheduling": True,
    },
    "linkedin": {
        "name": "LinkedIn",
        "post_types": ["text", "image", "video", "article", "document"],
        "max_caption_length": 3000,
        "supports_scheduling": True,
    },
    "google_business": {
        "name": "Google Business Profile",
        "post_types": ["update", "event", "offer"],
        "max_caption_length": 1500,
        "supports_scheduling": False,
    },
}


def _ensure_storage():
    SOCIAL_DIR.mkdir(parents=True, exist_ok=True)
    if not SOCIAL_POSTS_FILE.exists():
        SOCIAL_POSTS_FILE.write_text("[]", encoding="utf-8")


def _load_posts() -> list:
    _ensure_storage()
    try:
        data = json.loads(SOCIAL_POSTS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_posts(data: list):
    _ensure_storage()
    SOCIAL_POSTS_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _load_connectors() -> dict:
    if not CONNECTORS_FILE.exists():
        return {"connectors": []}
    try:
        data = json.loads(CONNECTORS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"connectors": []}
    except Exception:
        return {"connectors": []}


# ── Pydantic models ──────────────────────────────────────────────────────────

class SocialPostDraft(BaseModel):
    platform: str = Field(..., description="instagram, facebook, tiktok, linkedin, google_business")
    post_type: str = Field(default="image", description="Type of post for the platform")
    caption: str = Field(default="", max_length=63206)
    media_paths: List[str] = Field(default_factory=list, description="Paths to images/videos")
    hashtags: List[str] = Field(default_factory=list)
    link_url: Optional[str] = None
    project_id: Optional[str] = None
    client_id: Optional[str] = None
    campaign_name: Optional[str] = None
    notes: Optional[str] = None


class SocialPostSchedule(BaseModel):
    scheduled_at: str = Field(..., description="ISO timestamp or 'YYYY-MM-DD HH:MM' for scheduling")
    timezone: str = Field(default="UTC")


class SocialPostUpdate(BaseModel):
    caption: Optional[str] = None
    media_paths: Optional[List[str]] = None
    hashtags: Optional[List[str]] = None
    link_url: Optional[str] = None
    status: Optional[str] = None
    notes: Optional[str] = None


# ── Platform info ─────────────────────────────────────────────────────────────

@router.get("/api/social/platforms")
async def list_platforms():
    """List all supported social media platforms and their capabilities."""
    # Check which platforms have active connectors
    connectors = _load_connectors()
    connected_providers = set()
    for c in connectors.get("connectors", []):
        provider = (c.get("provider") or "").lower()
        if c.get("enabled", True) and provider in SOCIAL_PLATFORMS:
            connected_providers.add(provider)

    platforms = []
    for key, info in SOCIAL_PLATFORMS.items():
        platforms.append({
            **info,
            "key": key,
            "connected": key in connected_providers,
        })
    return {"platforms": platforms}


# ── Post CRUD ─────────────────────────────────────────────────────────────────

@router.get("/api/social/posts")
async def list_posts(
    platform: Optional[str] = None,
    status: Optional[str] = None,
    project_id: Optional[str] = None,
    campaign: Optional[str] = None,
):
    """List social media posts, optionally filtered."""
    posts = _load_posts()
    if platform:
        posts = [p for p in posts if p.get("platform") == platform.lower()]
    if status:
        posts = [p for p in posts if p.get("status") == status]
    if project_id:
        posts = [p for p in posts if p.get("project_id") == project_id]
    if campaign:
        posts = [p for p in posts if (p.get("campaign_name") or "").lower() == campaign.lower()]
    posts.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return {"posts": posts, "total": len(posts)}


@router.post("/api/social/posts")
async def create_post(body: SocialPostDraft):
    """Create a draft social media post."""
    platform = body.platform.lower()
    if platform not in SOCIAL_PLATFORMS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown platform '{body.platform}'. Supported: {', '.join(SOCIAL_PLATFORMS.keys())}"
        )

    platform_info = SOCIAL_PLATFORMS[platform]
    if body.post_type not in platform_info["post_types"]:
        raise HTTPException(
            status_code=400,
            detail=f"Post type '{body.post_type}' not supported on {platform}. Supported: {', '.join(platform_info['post_types'])}"
        )

    if len(body.caption) > platform_info["max_caption_length"]:
        raise HTTPException(
            status_code=400,
            detail=f"Caption too long for {platform}. Max: {platform_info['max_caption_length']} chars"
        )

    posts = _load_posts()
    post = {
        "id": str(uuid.uuid4()),
        "platform": platform,
        "post_type": body.post_type,
        "caption": body.caption,
        "hashtags": body.hashtags,
        "media_paths": body.media_paths,
        "link_url": body.link_url,
        "project_id": body.project_id,
        "client_id": body.client_id,
        "campaign_name": body.campaign_name,
        "notes": body.notes,
        "status": "draft",  # draft, scheduled, publishing, published, failed
        "scheduled_at": None,
        "published_at": None,
        "platform_post_id": None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    posts.append(post)
    _save_posts(posts)
    logger.info(f"Created social post draft: {post['id']} on {platform}")
    return {"ok": True, "post": post}


@router.get("/api/social/posts/{post_id}")
async def get_post(post_id: str):
    """Get a single social media post by ID."""
    posts = _load_posts()
    post = next((p for p in posts if p.get("id") == post_id), None)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return {"post": post}


@router.put("/api/social/posts/{post_id}")
async def update_post(post_id: str, body: SocialPostUpdate):
    """Update a draft post's content."""
    posts = _load_posts()
    idx = next((i for i, p in enumerate(posts) if p.get("id") == post_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="Post not found")

    if posts[idx].get("status") not in ("draft", "scheduled"):
        raise HTTPException(status_code=409, detail="Can only edit draft or scheduled posts")

    updates = body.dict(exclude_none=True)
    for key, val in updates.items():
        posts[idx][key] = val
    posts[idx]["updated_at"] = time.time()
    _save_posts(posts)
    return {"ok": True, "post": posts[idx]}


@router.delete("/api/social/posts/{post_id}")
async def delete_post(post_id: str):
    """Delete a social media post."""
    posts = _load_posts()
    before = len(posts)
    posts = [p for p in posts if p.get("id") != post_id]
    if len(posts) == before:
        raise HTTPException(status_code=404, detail="Post not found")
    _save_posts(posts)
    return {"ok": True, "deleted": True}


# ── Scheduling ────────────────────────────────────────────────────────────────

@router.post("/api/social/posts/{post_id}/schedule")
async def schedule_post(post_id: str, body: SocialPostSchedule):
    """Schedule a draft post for future publishing."""
    posts = _load_posts()
    idx = next((i for i, p in enumerate(posts) if p.get("id") == post_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="Post not found")

    post = posts[idx]
    if post.get("status") not in ("draft", "scheduled"):
        raise HTTPException(status_code=409, detail="Can only schedule draft or already-scheduled posts")

    platform = post.get("platform", "")
    if platform in SOCIAL_PLATFORMS and not SOCIAL_PLATFORMS[platform]["supports_scheduling"]:
        raise HTTPException(
            status_code=400,
            detail=f"Scheduling not supported for {platform}"
        )

    post["scheduled_at"] = body.scheduled_at
    post["timezone"] = body.timezone
    post["status"] = "scheduled"
    post["updated_at"] = time.time()
    _save_posts(posts)
    logger.info(f"Scheduled social post {post_id} for {body.scheduled_at} ({body.timezone})")
    return {"ok": True, "post": post}


@router.post("/api/social/posts/{post_id}/unschedule")
async def unschedule_post(post_id: str):
    """Revert a scheduled post back to draft status."""
    posts = _load_posts()
    idx = next((i for i, p in enumerate(posts) if p.get("id") == post_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="Post not found")

    if posts[idx].get("status") != "scheduled":
        raise HTTPException(status_code=409, detail="Only scheduled posts can be unscheduled")

    posts[idx]["status"] = "draft"
    posts[idx]["scheduled_at"] = None
    posts[idx]["updated_at"] = time.time()
    _save_posts(posts)
    return {"ok": True, "post": posts[idx]}


# ── Publishing (stub — real implementation needs platform API integration) ───

@router.post("/api/social/posts/{post_id}/publish")
async def publish_post(post_id: str):
    """
    Attempt to publish a post immediately through the connected platform API.

    Note: This requires the platform connector to be configured with valid
    OAuth tokens. Currently returns a simulated success for development.
    In production, this would call the platform's content publishing API.
    """
    posts = _load_posts()
    idx = next((i for i, p in enumerate(posts) if p.get("id") == post_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="Post not found")

    post = posts[idx]
    if post.get("status") == "published":
        raise HTTPException(status_code=409, detail="Post is already published")

    platform = post.get("platform", "")

    # Check if connector exists for this platform
    connectors = _load_connectors()
    platform_connector = None
    for c in connectors.get("connectors", []):
        if (c.get("provider") or "").lower() == platform and c.get("enabled", True):
            platform_connector = c
            break

    if not platform_connector:
        # Mark as pending — will need manual publishing or connector setup
        post["status"] = "pending_connector"
        post["updated_at"] = time.time()
        _save_posts(posts)
        return {
            "ok": False,
            "post": post,
            "detail": f"No active connector found for {platform}. Configure the {platform} connector first, or publish manually.",
        }

    # In production: call platform API here
    # For now, mark as published with a simulated platform post ID
    post["status"] = "published"
    post["published_at"] = time.time()
    post["platform_post_id"] = f"sim_{uuid.uuid4().hex[:12]}"
    post["updated_at"] = time.time()
    _save_posts(posts)
    logger.info(f"Published social post {post_id} to {platform}")
    return {"ok": True, "post": post}


# ── Campaign overview ─────────────────────────────────────────────────────────

@router.get("/api/social/campaigns")
async def list_campaigns():
    """List unique campaign names with post counts per platform."""
    posts = _load_posts()
    campaigns: Dict[str, Dict[str, Any]] = {}
    for p in posts:
        name = p.get("campaign_name") or "Uncategorized"
        if name not in campaigns:
            campaigns[name] = {"name": name, "posts": 0, "platforms": set(), "statuses": {}}
        campaigns[name]["posts"] += 1
        campaigns[name]["platforms"].add(p.get("platform", "unknown"))
        status = p.get("status", "draft")
        campaigns[name]["statuses"][status] = campaigns[name]["statuses"].get(status, 0) + 1

    result = []
    for c in campaigns.values():
        c["platforms"] = sorted(c["platforms"])
        result.append(c)
    result.sort(key=lambda x: x["posts"], reverse=True)
    return {"campaigns": result}


@router.get("/api/social/stats")
async def social_stats():
    """Overview stats for the social media module."""
    posts = _load_posts()
    by_status: Dict[str, int] = {}
    by_platform: Dict[str, int] = {}
    for p in posts:
        s = p.get("status", "draft")
        by_status[s] = by_status.get(s, 0) + 1
        plat = p.get("platform", "unknown")
        by_platform[plat] = by_platform.get(plat, 0) + 1
    return {
        "total_posts": len(posts),
        "by_status": by_status,
        "by_platform": by_platform,
    }
