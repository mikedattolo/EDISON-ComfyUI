"""
Branding client storage and asset management.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import re
import shutil
import time
import uuid


def _slugify_client_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return slug or f"client-{uuid.uuid4().hex[:8]}"


def _normalize_tags(raw_tags: Any) -> List[str]:
    if isinstance(raw_tags, list):
        vals = raw_tags
    elif isinstance(raw_tags, str):
        vals = [t.strip() for t in raw_tags.split(",")]
    else:
        vals = []
    out: List[str] = []
    seen = set()
    for tag in vals:
        cleaned = str(tag or "").strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(cleaned)
    return out


class BrandingClientStore:
    def __init__(
        self,
        repo_root: Path,
        branding_root: Path,
        branding_db_path: Path,
        media_roots: Optional[List[Path]] = None,
    ):
        self.repo_root = repo_root.resolve()
        self.branding_root = branding_root.resolve(strict=False)
        self.branding_db_path = branding_db_path.resolve(strict=False)
        self.media_roots = [root.resolve(strict=False) for root in (media_roots or [])]
        self.media_roots.append(self.branding_root)

    def list_clients(self) -> List[Dict[str, Any]]:
        db = self._load_branding()
        clients = [self._normalize_client_record(client) for client in db.get("clients", [])]
        return sorted(clients, key=lambda item: int(item.get("created_at") or 0), reverse=True)

    def get_client(self, client_id: str) -> Optional[Dict[str, Any]]:
        for client in self.list_clients():
            if client.get("id") == client_id:
                return client
        return None

    def find_client_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        lowered = (name or "").strip().lower()
        if not lowered:
            return None
        slug = _slugify_client_name(name)
        for client in self.list_clients():
            values = {
                str(client.get("name") or "").strip().lower(),
                str(client.get("business_name") or "").strip().lower(),
                str(client.get("slug") or "").strip().lower(),
            }
            if lowered in values or slug == client.get("slug"):
                return client
        return None

    def create_client(self, request: Dict[str, Any]) -> Dict[str, Any]:
        business_name = str(request.get("business_name") or request.get("name") or "").strip()
        if not business_name:
            raise ValueError("name is required")

        db = self._load_branding()
        clients = [self._normalize_client_record(client) for client in db.get("clients", [])]
        existing = self.find_client_by_name(business_name)
        if existing:
            return {"client": existing, "created": False}

        slug = self._unique_client_slug(business_name, clients)
        dirs = self._client_asset_dirs(slug)
        self._ensure_directory(self.branding_root)
        for path in dirs.values():
            self._ensure_directory(path)

        now = int(time.time())
        entry = self._normalize_client_record({
            "id": f"client_{uuid.uuid4().hex[:10]}",
            "name": business_name,
            "business_name": business_name,
            "contact_person": str(request.get("contact_person") or "").strip(),
            "email": str(request.get("email") or "").strip(),
            "phone": str(request.get("phone") or "").strip(),
            "website": str(request.get("website") or "").strip(),
            "industry": str(request.get("industry") or "").strip(),
            "notes": str(request.get("notes") or "").strip(),
            "tags": _normalize_tags(request.get("tags")),
            "slug": slug,
            "created_at": now,
            "updated_at": now,
            "paths": {
                "base": self._workspace_relative(dirs["base"]),
                "images": self._workspace_relative(dirs["images"]),
                "videos": self._workspace_relative(dirs["videos"]),
                "files": self._workspace_relative(dirs["files"]),
            },
        })
        clients.append(entry)
        db["clients"] = clients
        self._save_branding(db)
        return {"client": entry, "created": True}

    def update_client(self, client_id: str, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        db = self._load_branding()
        clients = [self._normalize_client_record(client) for client in db.get("clients", [])]
        target = next((client for client in clients if client.get("id") == client_id), None)
        if not target:
            return None

        for key in ["contact_person", "email", "phone", "website", "industry", "notes"]:
            if key in request:
                target[key] = str(request.get(key) or "").strip()
        if "tags" in request:
            target["tags"] = _normalize_tags(request.get("tags"))
        if "business_name" in request or "name" in request:
            next_name = str(request.get("business_name") or request.get("name") or target.get("business_name") or target.get("name") or "").strip()
            if next_name:
                target["name"] = next_name
                target["business_name"] = next_name
        target["updated_at"] = int(time.time())
        db["clients"] = clients
        self._save_branding(db)
        return target

    def list_assets(self, client_id: str) -> Dict[str, Any]:
        client = self.get_client(client_id)
        if not client:
            raise FileNotFoundError("Client not found")
        dirs = self._client_asset_dirs(client.get("slug", ""))

        def _scan(folder: Path) -> List[Dict[str, Any]]:
            if not folder.exists():
                return []
            out = []
            for fp in sorted(folder.rglob("*"), key=lambda item: item.stat().st_mtime if item.exists() else 0, reverse=True):
                if not fp.is_file():
                    continue
                try:
                    out.append({
                        "name": fp.name,
                        "path": self._workspace_relative(fp),
                        "size": fp.stat().st_size,
                        "mtime": int(fp.stat().st_mtime),
                    })
                except Exception:
                    continue
            return out[:200]

        return {
            "client": client,
            "assets": {
                "images": _scan(dirs["images"]),
                "videos": _scan(dirs["videos"]),
                "files": _scan(dirs["files"]),
            },
        }

    def add_existing_asset(self, client_id: str, source_path: str, asset_type: str = "", move_file: bool = False) -> Dict[str, Any]:
        client = self.get_client(client_id)
        if not client:
            raise FileNotFoundError("Client not found")
        safe_src = self._resolve_media_path(source_path)
        if not safe_src.is_file():
            raise FileNotFoundError(f"File not found: {source_path}")
        resolved_type = self._resolve_asset_type(safe_src.name, asset_type)
        dirs = self._client_asset_dirs(client.get("slug", ""))
        target_dir = self._ensure_directory(dirs[resolved_type])
        target = (target_dir / safe_src.name).resolve(strict=False)
        if target.exists():
            target = (target_dir / f"{target.stem}_{uuid.uuid4().hex[:6]}{target.suffix}").resolve(strict=False)
        if move_file:
            shutil.move(str(safe_src), str(target))
        else:
            shutil.copy2(str(safe_src), str(target))
        return {
            "client_id": client_id,
            "asset_type": resolved_type,
            "stored_path": self._workspace_relative(target),
            "moved": move_file,
        }

    def upload_asset(self, client_id: str, filename: str, content: bytes, asset_type: str = "") -> Dict[str, Any]:
        client = self.get_client(client_id)
        if not client:
            raise FileNotFoundError("Client not found")
        safe_name = re.sub(r"[^\w.\-]", "_", Path(filename or "upload").name)
        if not safe_name or safe_name.startswith("."):
            raise ValueError("Invalid filename")
        resolved_type = self._resolve_asset_type(safe_name, asset_type)
        dirs = self._client_asset_dirs(client.get("slug", ""))
        target_dir = self._ensure_directory(dirs[resolved_type])
        target = (target_dir / safe_name).resolve(strict=False)
        if target.exists():
            target = (target_dir / f"{Path(safe_name).stem}_{uuid.uuid4().hex[:6]}{Path(safe_name).suffix}").resolve(strict=False)
        target.write_bytes(content)
        return {
            "client_id": client_id,
            "asset_type": resolved_type,
            "filename": target.name,
            "stored_path": self._workspace_relative(target),
            "size": len(content),
        }

    def get_generation_dir(self, client_id: str, category: str) -> Path:
        client = self.get_client(client_id)
        if not client:
            raise FileNotFoundError("Client not found")
        dirs = self._client_asset_dirs(client.get("slug", ""))
        target = dirs["files"] / category
        return self._ensure_directory(target)

    def _load_branding(self) -> Dict[str, Any]:
        self._ensure_directory(self.branding_db_path.parent)
        try:
            return json.loads(self.branding_db_path.read_text())
        except Exception:
            return {"clients": []}

    def _save_branding(self, data: Dict[str, Any]) -> None:
        self._ensure_directory(self.branding_db_path.parent)
        self.branding_db_path.write_text(json.dumps(data, indent=2))

    def _normalize_client_record(self, client: Dict[str, Any]) -> Dict[str, Any]:
        business_name = str(client.get("business_name") or client.get("name") or "").strip()
        slug = str(client.get("slug") or _slugify_client_name(business_name)).strip()
        dirs = self._client_asset_dirs(slug)
        normalized = dict(client)
        normalized["name"] = business_name
        normalized["business_name"] = business_name
        normalized.setdefault("contact_person", "")
        normalized.setdefault("email", "")
        normalized.setdefault("phone", "")
        normalized.setdefault("website", "")
        normalized.setdefault("industry", "")
        normalized.setdefault("notes", "")
        normalized["tags"] = _normalize_tags(normalized.get("tags"))
        normalized["slug"] = slug
        normalized.setdefault("created_at", int(time.time()))
        normalized.setdefault("updated_at", normalized.get("created_at"))
        normalized["paths"] = {
            "base": self._workspace_relative(dirs["base"]),
            "images": self._workspace_relative(dirs["images"]),
            "videos": self._workspace_relative(dirs["videos"]),
            "files": self._workspace_relative(dirs["files"]),
        }
        return normalized

    def _unique_client_slug(self, name: str, clients: List[Dict[str, Any]]) -> str:
        base_slug = _slugify_client_name(name)
        used = {str((client or {}).get("slug") or "").strip() for client in clients}
        if base_slug not in used:
            return base_slug
        idx = 2
        while True:
            candidate = f"{base_slug}-{idx}"
            if candidate not in used:
                return candidate
            idx += 1

    def _client_asset_dirs(self, slug: str) -> Dict[str, Path]:
        base = (self.branding_root / slug).resolve(strict=False)
        return {
            "base": base,
            "images": base / "images",
            "videos": base / "videos",
            "files": base / "files",
        }

    def _ensure_directory(self, path: Path) -> Path:
        resolved = path.resolve(strict=False)
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def _workspace_relative(self, path: Path) -> str:
        try:
            return str(path.resolve(strict=False).relative_to(self.repo_root))
        except Exception:
            return str(path.resolve(strict=False))

    def _resolve_asset_type(self, filename: str, asset_type: str) -> str:
        resolved_type = (asset_type or "").strip().lower()
        ext = Path(filename).suffix.lower()
        if not resolved_type:
            if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}:
                resolved_type = "images"
            elif ext in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
                resolved_type = "videos"
            else:
                resolved_type = "files"
        if resolved_type not in {"images", "videos", "files"}:
            raise ValueError("asset_type must be one of images, videos, files")
        return resolved_type

    def _resolve_media_path(self, path_str: str) -> Path:
        candidate = Path(path_str)
        if not candidate.is_absolute():
            candidate = (self.repo_root / candidate).resolve(strict=False)
        else:
            candidate = candidate.resolve(strict=False)
        roots = self.media_roots or [self.repo_root]
        for root in roots:
            try:
                candidate.relative_to(root)
                return candidate
            except Exception:
                continue
        raise ValueError("Access denied: media path outside allowed roots")