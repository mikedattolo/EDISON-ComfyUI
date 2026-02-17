# Unified Generation System

EDISON now uses a **unified job store** backed by SQLite for tracking all
generation jobs — image, video, music, and 3D mesh.  Every generation gets
a persistent `job_id`, provenance metadata, and a `.meta.json` sidecar file
next to the output artifact.

## Architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Image Gen   │   │  Video Gen   │   │  Music Gen   │   │  3D Mesh Gen │
│  (FLUX/      │   │ (CogVideoX)  │   │ (MusicGen)   │   │  (ComfyUI)   │
│   ComfyUI)   │   │              │   │              │   │              │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │                  │
       └──────────────────┴──────────────────┴──────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Unified Job Store     │
                    │   (SQLite: data/jobs.db) │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   .meta.json sidecars   │
                    │   (next to each output) │
                    └─────────────────────────┘
```

## Job Lifecycle

```
queued → loading → generating → encoding → complete
                                         ↘ error
              (any active state)         → cancelled
```

## API Endpoints

### List Jobs
```
GET /generations?type=video&status=complete&limit=20
```

### Get Job Detail
```
GET /generations/{job_id}
```

### Cancel a Job
```
POST /generations/{job_id}/cancel
```

## Job Types

| Type | Backend | Output Format |
|------|---------|---------------|
| `image` | FLUX via ComfyUI | PNG |
| `video` | CogVideoX-5B diffusers | MP4 |
| `music` | MusicGen (small/medium/large) | WAV + MP3 |
| `mesh` | ComfyUI 3D workflow | GLB / STL |

## Provenance Tracking

Every completed job stores:
- **model** — exact model ID used
- **seed(s)** — deterministic seeds for reproducibility
- **backend** — which pipeline produced the output
- **params** — all generation parameters (steps, cfg, resolution, etc.)
- **duration_s** — wall-clock generation time

## Metadata Sidecars

When a job completes, a `<filename>.meta.json` file is written next to the
output artifact.  Example:

```json
{
  "job_id": "a1b2c3d4-...",
  "job_type": "video",
  "prompt": "a cat playing piano",
  "params": {
    "width": 720,
    "height": 480,
    "num_frames": 49,
    "fps": 8,
    "num_inference_steps": 30
  },
  "provenance": {
    "model": "THUDM/CogVideoX-5b",
    "seeds": [123456789],
    "backend": "cogvideox-diffusers"
  },
  "created_at": 1700000000.0,
  "completed_at": 1700000300.0,
  "duration_s": 300.0
}
```

## Video Generation

### Improvements
- **Unified job store** — jobs survive server restarts (SQLite-backed)
- **Cancellation** — `POST /generations/{job_id}/cancel` stops generation
  between segments
- **Seed tracking** — every segment's seed is recorded in provenance for
  reproducibility
- **Low-VRAM mode** — set `video.low_vram: true` in `config/edison.yaml`
  to auto-reduce resolution (480×320) and steps (≤20) for GPUs < 8GB
- **Metadata sidecars** — `.meta.json` written next to each MP4

### Configuration (`config/edison.yaml`)
```yaml
edison:
  video:
    cogvideox_model: THUDM/CogVideoX-5b
    width: 720
    height: 480
    num_frames: 49
    fps: 8
    guidance_scale: 6.0
    num_inference_steps: 30
    max_duration: 30
    low_vram: false        # Set true for GPUs < 8GB
```

## Music Generation

### Improvements
- **Unified job store** — music jobs tracked alongside image/video/mesh
- **Deterministic seeds** — every generation stores a seed for exact replay
- **Variation support** — pass `variation_of=<job_id>` to create a new
  track with a nearby seed (similar but not identical output)
- **Structured prompts** — genre, mood, instruments, tempo, style, lyrics,
  reference artist all compose into an optimized MusicGen prompt
- **Metadata sidecars** — `.meta.json` written next to each WAV file

### Variation Example
```python
# Generate original
result = music_service.generate_music(
    genre="electronic", mood="chill", duration=15
)
original_id = result["data"]["job_id"]

# Generate variation
variation = music_service.generate_music(
    genre="electronic", mood="chill", duration=15,
    variation_of=original_id,
)
```

## 3D Mesh Generation

### Features
- **ComfyUI backend** — uses installed 3D generation workflows
- **GLB default** / STL optional output format
- **Job store integration** — tracks status, provenance, outputs

### API
```
POST /3d/generate   {"prompt": "a low-poly tree", "format": "glb"}
GET  /3d/status/{job_id}
GET  /3d/models
```

## Style Profiles (Image Generation)

Four built-in style profiles enhance short image prompts:

| Profile | Key Traits |
|---------|-----------|
| `photorealistic` | 30 steps, CFG 4.0, adds "ultra-realistic, 8K" |
| `cinematic` | 35 steps, CFG 5.0, anamorphic, film grain |
| `anime` | 28 steps, CFG 7.0, clean lineart, vivid colors |
| `digital_art` | 30 steps, CFG 5.5, detailed illustration |

```
GET /style-profiles          # List all profiles
```

Prompt expansion automatically enriches short prompts (< 8 words) with
composition/lighting/material hints based on the selected style.
