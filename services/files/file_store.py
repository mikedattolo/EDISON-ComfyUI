"""
Edison File Upload & Management Service.

Handles file uploads, metadata tracking, retrieval, and deletion.
Files are stored in uploads/images/ and uploads/files/ with metadata sidecars.
"""

import hashlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
UPLOADS_ROOT = REPO_ROOT / "uploads"
UPLOADS_IMAGES = UPLOADS_ROOT / "images"
UPLOADS_FILES = UPLOADS_ROOT / "files"
METADATA_DIR = UPLOADS_ROOT / ".metadata"

MAX_FILE_SIZE_MB = 100
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".svg"}
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".yaml", ".yml", ".py", ".js", ".ts",
                   ".html", ".css", ".xml", ".csv", ".sh", ".bat", ".toml", ".ini",
                   ".cfg", ".conf", ".log", ".sql", ".rs", ".go", ".java", ".cpp",
                   ".c", ".h", ".rb", ".php"}


# ── Data Classes ─────────────────────────────────────────────────────────

@dataclass
class FileMetadata:
    """Metadata for an uploaded file."""
    file_id: str
    filename: str
    original_filename: str
    file_type: str  # "image" or "file"
    mime_type: str
    size_bytes: int
    sha256: str
    session_id: Optional[str] = None
    upload_time: float = 0
    tags: List[str] = field(default_factory=list)
    storage_path: str = ""  # relative to UPLOADS_ROOT
    versions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Helpers ──────────────────────────────────────────────────────────────

def _hash_file(path: Path) -> str:
    """SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _guess_mime(filename: str) -> str:
    """Best-effort MIME type guess."""
    ext = Path(filename).suffix.lower()
    mime_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
        ".svg": "image/svg+xml", ".tiff": "image/tiff",
        ".txt": "text/plain", ".md": "text/markdown", ".json": "application/json",
        ".yaml": "text/yaml", ".yml": "text/yaml", ".py": "text/x-python",
        ".js": "text/javascript", ".ts": "text/typescript",
        ".html": "text/html", ".css": "text/css", ".xml": "text/xml",
        ".csv": "text/csv", ".sh": "text/x-shellscript",
        ".pdf": "application/pdf", ".zip": "application/zip",
    }
    return mime_map.get(ext, "application/octet-stream")


def _sanitize_filename(filename: str) -> str:
    """Remove path traversal and dangerous characters."""
    # Strip directory components
    name = Path(filename).name
    # Remove leading dots (hidden files)
    name = name.lstrip(".")
    # Replace spaces with underscores
    name = name.replace(" ", "_")
    # Only keep safe characters
    safe = ""
    for c in name:
        if c.isalnum() or c in "._-":
            safe += c
    return safe or "unnamed"


def _is_path_safe(path: Path, root: Path) -> bool:
    """Ensure path doesn't escape the root directory."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


# ── File Store ───────────────────────────────────────────────────────────

class FileStore:
    """Thread-safe file upload and management store."""

    def __init__(self, base_dir: Optional[str] = None):
        self._lock = threading.Lock()
        self._metadata_cache: Dict[str, FileMetadata] = {}
        if base_dir:
            bd = Path(base_dir)
            self._uploads_root = bd
            self._uploads_images = bd / "images"
            self._uploads_files = bd / "files"
            self._metadata_dir = bd / ".metadata"
        else:
            self._uploads_root = UPLOADS_ROOT
            self._uploads_images = UPLOADS_IMAGES
            self._uploads_files = UPLOADS_FILES
            self._metadata_dir = METADATA_DIR
        self._ensure_dirs()
        self._load_metadata_cache()

    def _ensure_dirs(self):
        """Create upload directories if they don't exist."""
        for d in [self._uploads_images, self._uploads_files, self._metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _metadata_path(self, file_id: str) -> Path:
        return self._metadata_dir / f"{file_id}.json"

    def _save_metadata(self, meta: FileMetadata):
        """Persist metadata to disk."""
        path = self._metadata_path(meta.file_id)
        with open(path, "w") as f:
            json.dump(meta.to_dict(), f, indent=2)

    def _load_metadata_cache(self):
        """Load all metadata files into memory cache."""
        if not self._metadata_dir.exists():
            return
        for p in self._metadata_dir.glob("*.json"):
            try:
                with open(p) as f:
                    data = json.load(f)
                meta = FileMetadata(**{k: v for k, v in data.items()
                                       if k in FileMetadata.__dataclass_fields__})
                self._metadata_cache[meta.file_id] = meta
            except Exception as e:
                logger.warning(f"Failed to load metadata {p.name}: {e}")
        logger.info(f"Loaded {len(self._metadata_cache)} file metadata entries")

    # ── Public API ───────────────────────────────────────────────────

    def upload(self, filename: str, data: bytes,
               session_id: Optional[str] = None,
               tags: Optional[List[str]] = None) -> FileMetadata:
        """
        Store an uploaded file and return its metadata.

        Args:
            filename: Original filename from the client.
            data: Raw file bytes.
            session_id: Chat session ID.
            tags: Optional tags.

        Returns:
            FileMetadata for the stored file.

        Raises:
            ValueError: If file is too large or filename is invalid.
        """
        # Size check
        size_mb = len(data) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(f"File too large: {size_mb:.1f}MB (max {MAX_FILE_SIZE_MB}MB)")

        safe_name = _sanitize_filename(filename)
        ext = Path(safe_name).suffix.lower()
        file_id = str(uuid.uuid4())
        is_image = ext in IMAGE_EXTENSIONS

        # Determine storage directory
        dest_dir = self._uploads_images if is_image else self._uploads_files
        # Unique filename to prevent collisions
        stored_name = f"{file_id[:8]}_{safe_name}"
        dest_path = dest_dir / stored_name

        # Safety check
        if not _is_path_safe(dest_path, self._uploads_root):
            raise ValueError("Invalid filename (path traversal detected)")

        # Write file
        with open(dest_path, "wb") as f:
            f.write(data)

        # Compute hash
        file_hash = hashlib.sha256(data).hexdigest()

        # Build metadata
        meta = FileMetadata(
            file_id=file_id,
            filename=stored_name,
            original_filename=filename,
            file_type="image" if is_image else "file",
            mime_type=_guess_mime(filename),
            size_bytes=len(data),
            sha256=file_hash,
            session_id=session_id,
            upload_time=time.time(),
            tags=tags or [],
            storage_path=str(dest_path.relative_to(self._uploads_root)),
        )

        with self._lock:
            self._metadata_cache[file_id] = meta
            self._save_metadata(meta)

        logger.info(f"Uploaded file: {filename} → {stored_name} ({meta.file_type}, {len(data)} bytes)")
        return meta

    def get(self, file_id: str) -> Optional[FileMetadata]:
        """Get metadata for a file."""
        with self._lock:
            return self._metadata_cache.get(file_id)

    def get_path(self, file_id: str) -> Optional[Path]:
        """Get the absolute path for a stored file."""
        meta = self.get(file_id)
        if not meta:
            return None
        path = self._uploads_root / meta.storage_path
        if path.exists() and _is_path_safe(path, self._uploads_root):
            return path
        return None

    def read_file(self, file_id: str) -> Optional[bytes]:
        """Read file contents."""
        path = self.get_path(file_id)
        if path and path.exists():
            return path.read_bytes()
        return None

    def read_text(self, file_id: str) -> Optional[str]:
        """Read text file contents (returns None for binary files)."""
        meta = self.get(file_id)
        if not meta:
            return None
        ext = Path(meta.original_filename).suffix.lower()
        if ext not in TEXT_EXTENSIONS:
            return None
        path = self.get_path(file_id)
        if not path:
            return None
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return None

    def list_files(self, session_id: Optional[str] = None,
                   file_type: Optional[str] = None,
                   limit: int = 100, offset: int = 0) -> List[FileMetadata]:
        """List files with optional filtering."""
        with self._lock:
            files = list(self._metadata_cache.values())

        # Filter
        if session_id:
            files = [f for f in files if f.session_id == session_id]
        if file_type:
            files = [f for f in files if f.file_type == file_type]

        # Sort by upload time (newest first)
        files.sort(key=lambda f: f.upload_time, reverse=True)

        return files[offset:offset + limit]

    def delete(self, file_id: str) -> bool:
        """Delete a file and its metadata."""
        with self._lock:
            meta = self._metadata_cache.pop(file_id, None)
        if not meta:
            return False

        # Delete the actual file
        file_path = self._uploads_root / meta.storage_path
        if file_path.exists() and _is_path_safe(file_path, self._uploads_root):
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete file {file_path}: {e}")

        # Delete metadata sidecar
        md_path = self._metadata_path(file_id)
        if md_path.exists():
            try:
                md_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete metadata {md_path}: {e}")

        logger.info(f"Deleted file: {meta.original_filename} ({file_id})")
        return True

    def add_version(self, file_id: str, version_info: Dict[str, Any]):
        """Add a version record to a file's history."""
        with self._lock:
            meta = self._metadata_cache.get(file_id)
            if meta:
                meta.versions.append({
                    "timestamp": time.time(),
                    **version_info,
                })
                self._save_metadata(meta)


# ── Singleton ────────────────────────────────────────────────────────────

_store: Optional[FileStore] = None
_store_lock = threading.Lock()


def get_file_store() -> FileStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = FileStore()
    return _store
