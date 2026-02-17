# Files & Editing

## Overview

Edison supports file uploads, image editing, and text file editing through
chat. All operations maintain version history and provenance metadata.

---

## File Upload System

### Storage Layout

```
uploads/
├── images/      # Image files (.png, .jpg, etc.)
├── files/       # Text/code files (.py, .txt, .md, etc.)
└── .metadata/   # JSON metadata sidecars
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/files/upload` | Upload a file (multipart form) |
| `GET` | `/files/list` | List files (filter by session_id, file_type) |
| `GET` | `/files/{file_id}` | Get file metadata |
| `GET` | `/files/{file_id}/content` | Get file contents |
| `DELETE` | `/files/{file_id}` | Delete a file |

### Safety Features

- **100 MB size limit** per file
- **Path traversal prevention** — all filenames sanitized, resolved paths
  checked against the uploads root
- **SHA-256 deduplication** — identical files share a hash
- **Thread-safe** — all operations use a lock for the metadata cache

### Usage

```bash
# Upload a file
curl -X POST http://localhost:3456/files/upload \
  -F "file=@photo.png" \
  -F "session_id=chat_123"

# List files for a session
curl http://localhost:3456/files/list?session_id=chat_123

# Get file content
curl http://localhost:3456/files/{file_id}/content
```

---

## Image Editing

### Supported Operations

| Operation | Method | Parameters |
|-----------|--------|------------|
| Crop | `crop` | `left, top, right, bottom` |
| Resize | `resize` | `width, height` |
| Rotate | `rotate` | `angle` (degrees) |
| Flip | `flip` | `direction` (horizontal/vertical) |
| Brightness | `brightness` | `factor` (0.0-3.0) |
| Contrast | `contrast` | `factor` (0.0-3.0) |
| Saturation | `saturation` | `factor` (0.0-3.0) |
| Blur | `blur` | `radius` (pixels) |
| Sharpen | `sharpen` | — |
| img2img | `img2img` | `prompt, strength, steps` |

### API Endpoint

```bash
POST /images/edit
{
  "file_id": "abc-123",
  "edit_type": "crop",
  "parameters": {"left": 10, "top": 10, "right": 200, "bottom": 200}
}
```

### Version History

Every edit produces an `EditRecord` with:
- `edit_id` — unique identifier
- `source_image_id` — original file
- `edit_type` — operation performed
- `parameters` — exact parameters used
- `output_path` — path to the result
- `timestamp`
- Provenance `.meta.json` sidecar written next to the output

```bash
GET /images/edit/history
```

### img2img (ComfyUI)

When `edit_type=img2img`, the editor:
1. Uploads the source image to ComfyUI
2. Builds a workflow with the source image as input + user prompt
3. Polls ComfyUI for completion (120s timeout)
4. Downloads and saves the result

Falls back to PIL if ComfyUI is unavailable.

---

## Text File Editing

### Supported Operations

| Operation | Description |
|-----------|-------------|
| `replace_content` | Replace entire file content |
| `search_replace` | Find and replace text |
| `line_edit` | Edit a specific line by number |

### API Endpoints

```bash
# Apply an edit
POST /files/edit
{
  "file_id": "abc-123",
  "edit_type": "search_replace",
  "search": "old_function",
  "replace": "new_function"
}

# Get version history
GET /files/versions/{file_id}

# Get specific version content
GET /files/versions/{file_id}/{version_id}
```

### Version Tracking

Every edit creates a versioned snapshot in:
```
uploads/versions/{file_id}/v_{version_id}.ext
```

With a `.meta.json` sidecar containing:
- Version number
- Content hash
- Edit description
- Diff summary
- Timestamp
- Source (user_edit, llm_edit, transform)

### Safety Features

- **10 MB max** file size for editing
- **Extension whitelist** — only editable file types (.py, .js, .md, .txt, etc.)
- **Path traversal check** on all file paths
- **Automatic diff** generated for every edit
- **Revert** to any previous version

---

## Provenance Tracking

All generations and edits are tracked with provenance metadata:

```json
{
  "record_id": "uuid",
  "action": "image_generation",
  "model_used": "SDXL/FLUX",
  "parameters": {"prompt": "...", "steps": 20},
  "output_artifact": "output.png",
  "timestamp": 1234567890.0,
  "duration_seconds": 12.5
}
```

Records are stored in `data/provenance/` and accessible via:

```bash
GET /provenance/recent?limit=20
```

Sidecar `.provenance.json` files are written next to output artifacts when
`write_sidecar()` is called.
