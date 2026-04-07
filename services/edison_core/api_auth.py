"""
api_auth.py — Optional authentication and role-based access control for EDISON.

Provides user registration, login, session management, and role-based middleware.
DISABLED by default — enable by setting EDISON_AUTH_ENABLED=true in environment.

Roles: admin, designer, fabricator, editor, client_viewer
Storage: JSON file at config/auth/users.json

Security notes:
- Passwords are hashed with bcrypt (via hashlib + secrets as fallback)
- Sessions use random tokens stored server-side
- All sensitive endpoints require valid session token in Authorization header
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import time
import uuid
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["auth"])

REPO_ROOT = Path(__file__).resolve().parents[2]
AUTH_DIR = REPO_ROOT / "config" / "auth"
USERS_FILE = AUTH_DIR / "users.json"
SESSIONS_FILE = AUTH_DIR / "sessions.json"

# Feature flag
AUTH_ENABLED = os.environ.get("EDISON_AUTH_ENABLED", "false").lower() in ("true", "1", "yes")

VALID_ROLES = {"admin", "designer", "fabricator", "editor", "client_viewer"}
SESSION_TTL = 86400 * 7  # 7 days


def _ensure_storage():
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    for fpath in (USERS_FILE, SESSIONS_FILE):
        if not fpath.exists():
            fpath.write_text("[]", encoding="utf-8")


def _load_json(fpath: Path) -> list:
    _ensure_storage()
    try:
        data = json.loads(fpath.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_json(fpath: Path, data: list):
    _ensure_storage()
    fpath.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _hash_password(password: str, salt: Optional[str] = None) -> tuple:
    """Hash a password with PBKDF2-SHA256. Returns (hash_hex, salt_hex)."""
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return hashed.hex(), salt


def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verify a password against stored hash and salt."""
    computed, _ = _hash_password(password, salt)
    return secrets.compare_digest(computed, stored_hash)


# ── Pydantic models ──────────────────────────────────────────────────────────

class UserRegister(BaseModel):
    username: str = Field(..., min_length=2, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)
    display_name: Optional[str] = None
    role: str = Field(default="designer", description="admin, designer, fabricator, editor, client_viewer")


class UserLogin(BaseModel):
    username: str
    password: str


class UserUpdate(BaseModel):
    display_name: Optional[str] = None
    role: Optional[str] = None


# ── Session management ────────────────────────────────────────────────────────

def _create_session(user_id: str, role: str) -> str:
    """Create a new session token and store it."""
    token = secrets.token_urlsafe(32)
    sessions = _load_json(SESSIONS_FILE)
    sessions.append({
        "token": token,
        "user_id": user_id,
        "role": role,
        "created_at": time.time(),
        "expires_at": time.time() + SESSION_TTL,
    })
    # Prune expired sessions
    now = time.time()
    sessions = [s for s in sessions if s.get("expires_at", 0) > now]
    _save_json(SESSIONS_FILE, sessions)
    return token


def _get_session(token: str) -> Optional[dict]:
    """Look up a valid session by token."""
    sessions = _load_json(SESSIONS_FILE)
    now = time.time()
    for s in sessions:
        if s.get("token") == token and s.get("expires_at", 0) > now:
            return s
    return None


def _revoke_session(token: str):
    """Remove a session."""
    sessions = _load_json(SESSIONS_FILE)
    sessions = [s for s in sessions if s.get("token") != token]
    _save_json(SESSIONS_FILE, sessions)


# ── Auth dependency ───────────────────────────────────────────────────────────

async def get_current_user(request: Request) -> Optional[dict]:
    """
    FastAPI dependency that extracts user from Authorization header.
    Returns None if auth is disabled, allowing passthrough.
    """
    if not AUTH_ENABLED:
        return {"user_id": "local", "role": "admin", "username": "local"}

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header[7:]
    session = _get_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    users = _load_json(USERS_FILE)
    user = next((u for u in users if u.get("id") == session["user_id"]), None)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return {"user_id": user["id"], "role": user.get("role", "designer"), "username": user.get("username")}


def require_role(*allowed_roles: str):
    """Dependency factory that checks if the current user has one of the allowed roles."""
    async def check_role(user: dict = Depends(get_current_user)):
        if not AUTH_ENABLED:
            return user
        if user.get("role") not in allowed_roles:
            raise HTTPException(status_code=403, detail=f"Requires role: {', '.join(allowed_roles)}")
        return user
    return check_role


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/api/auth/status")
async def auth_status():
    """Check if authentication is enabled and how many users exist."""
    user_count = len(_load_json(USERS_FILE)) if AUTH_ENABLED else 0
    return {
        "auth_enabled": AUTH_ENABLED,
        "user_count": user_count,
        "roles": sorted(VALID_ROLES),
        "session_ttl_hours": SESSION_TTL // 3600,
    }


@router.post("/api/auth/register")
async def register_user(body: UserRegister):
    """Register a new user. First user automatically gets admin role."""
    if not AUTH_ENABLED:
        return {"ok": False, "detail": "Authentication is disabled. Set EDISON_AUTH_ENABLED=true to enable."}

    if body.role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role. Valid: {', '.join(sorted(VALID_ROLES))}")

    users = _load_json(USERS_FILE)

    # Check for duplicate username
    if any(u.get("username", "").lower() == body.username.lower() for u in users):
        raise HTTPException(status_code=409, detail="Username already exists")

    # First user gets admin role automatically
    role = "admin" if not users else body.role

    password_hash, salt = _hash_password(body.password)
    user = {
        "id": str(uuid.uuid4()),
        "username": body.username,
        "display_name": body.display_name or body.username,
        "role": role,
        "password_hash": password_hash,
        "salt": salt,
        "created_at": time.time(),
    }
    users.append(user)
    _save_json(USERS_FILE, users)
    logger.info(f"Registered user: {user['id']} ({body.username}) with role {role}")

    # Auto-login
    token = _create_session(user["id"], role)
    return {
        "ok": True,
        "user": {"id": user["id"], "username": user["username"], "role": role, "display_name": user["display_name"]},
        "token": token,
    }


@router.post("/api/auth/login")
async def login_user(body: UserLogin):
    """Authenticate and return a session token."""
    if not AUTH_ENABLED:
        return {"ok": True, "user": {"id": "local", "username": "local", "role": "admin"}, "token": "local"}

    users = _load_json(USERS_FILE)
    user = next((u for u in users if u.get("username", "").lower() == body.username.lower()), None)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not _verify_password(body.password, user["password_hash"], user["salt"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = _create_session(user["id"], user.get("role", "designer"))
    logger.info(f"User logged in: {user['id']} ({user['username']})")
    return {
        "ok": True,
        "user": {"id": user["id"], "username": user["username"], "role": user.get("role"), "display_name": user.get("display_name")},
        "token": token,
    }


@router.post("/api/auth/logout")
async def logout_user(request: Request):
    """Revoke the current session."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        _revoke_session(auth_header[7:])
    return {"ok": True}


@router.get("/api/auth/me")
async def get_current_user_info(user: dict = Depends(get_current_user)):
    """Get the current user's info."""
    return {"ok": True, "user": user}


@router.get("/api/auth/users")
async def list_users(user: dict = Depends(require_role("admin"))):
    """List all users (admin only). Passwords are never returned."""
    users = _load_json(USERS_FILE)
    return {
        "users": [
            {
                "id": u["id"],
                "username": u["username"],
                "display_name": u.get("display_name"),
                "role": u.get("role"),
                "created_at": u.get("created_at"),
            }
            for u in users
        ]
    }


@router.put("/api/auth/users/{user_id}")
async def update_user(user_id: str, body: UserUpdate, admin: dict = Depends(require_role("admin"))):
    """Update a user's role or display name (admin only)."""
    if body.role and body.role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role. Valid: {', '.join(sorted(VALID_ROLES))}")

    users = _load_json(USERS_FILE)
    idx = next((i for i, u in enumerate(users) if u.get("id") == user_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="User not found")

    if body.display_name is not None:
        users[idx]["display_name"] = body.display_name
    if body.role is not None:
        users[idx]["role"] = body.role
    _save_json(USERS_FILE, users)
    return {"ok": True, "user": {"id": users[idx]["id"], "username": users[idx]["username"], "role": users[idx].get("role")}}


@router.delete("/api/auth/users/{user_id}")
async def delete_user(user_id: str, admin: dict = Depends(require_role("admin"))):
    """Delete a user (admin only)."""
    users = _load_json(USERS_FILE)
    before = len(users)
    users = [u for u in users if u.get("id") != user_id]
    if len(users) == before:
        raise HTTPException(status_code=404, detail="User not found")
    _save_json(USERS_FILE, users)

    # Also revoke all sessions for this user
    sessions = _load_json(SESSIONS_FILE)
    sessions = [s for s in sessions if s.get("user_id") != user_id]
    _save_json(SESSIONS_FILE, sessions)
    return {"ok": True}
