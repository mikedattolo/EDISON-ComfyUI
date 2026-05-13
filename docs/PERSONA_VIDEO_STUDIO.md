# Persona Video Studio

Persona Video Studio is an additive Edison module for consent-gated, source-coherent persona video workflows. It is built for cleared, non-explicit production/performance footage where the creator owns or has licensed the source media and the persona identity is synthetic or explicitly authorized.

It does **not** implement an unauthorized real-person impersonation workflow, and it is not designed for sexual deepfake generation. Jobs require local rights metadata before they can start.

## Why this pipeline uses source transformation instead of pure video generation

Long-form pure video generation can drift: body motion, continuity, camera timing, scene physics, and audio sync often degrade across longer durations. Persona Video Studio keeps the source scene as the timing and motion authority:

1. Probe the selected video and audio streams.
2. Split the source into coherent segments.
3. Track the selected target subject as metadata.
4. Delegate persona rendering to a pluggable backend.
5. Run temporal/QC stages per segment.
6. Reassemble segments in source order.
7. Preserve, replace, or strip audio explicitly.
8. Emit sidecar reports and segment manifests.

This gives Edison production plumbing for future high-quality backends while preserving the fragile repository architecture.

## Rights and consent gate

Every job stores a local acknowledgement containing:

- every visible performer is a consenting adult
- the source footage is owned/licensed/cleared for AI transformation
- the persona identity is synthetic or explicitly authorized
- the footage is non-explicit production/performance material
- the material does not depict or simulate minors
- the workflow is not unauthorized real-person impersonation
- the user understands the acknowledgement is stored only as local job/report metadata

The acknowledgement is saved in the job record and final metadata report.

## Persona packs

Persona packs are registered in `config/integrations/persona_packs.json` and support:

- `name`
- `type`: `lora`, `reference_images`, `identity_model`, `adapter`, or `other`
- local `paths`
- notes
- preferred backend compatibility
- optional thumbnail/preview image

Register packs from the Persona Video Studio UI or `POST /api/persona-video/personas` through the web proxy.

## Processing stages

The job dashboard tracks these stages:

1. validating source and rights metadata
2. inspecting media streams
3. detecting shots/scenes
4. segmenting footage
5. tracking target performer
6. preparing persona backend/model
7. transforming segments
8. temporal stabilization / consistency cleanup
9. quality-control scoring
10. reprocessing failed segments if needed
11. stitching final video
12. restoring audio/remuxing
13. final export packaging

## Backends

The orchestration layer is backend-pluggable through `PersonaTransformBackend` in `services/edison_core/persona_video_backends.py`.

A backend must implement:

- `is_available()`
- `get_capabilities()`
- `prepare()`
- `transform_segment()`
- `temporal_stabilize()`
- `score_segment()`
- `cleanup()`

The included `metadata_only_passthrough` backend validates the pipeline without altering pixels. It is useful for smoke tests and report generation, but it is not a production identity transform.

The `comfyui_persona_workflow` adapter becomes available when curated workflow JSON templates are placed under `config/persona_video/comfyui_workflows/`. Future templates should support variable injection for source segment/frame paths, persona reference/model paths, output paths, and device hints where supported.

## Exclusive GPU Render Mode

Exclusive GPU Render Mode is a best-effort local render isolation mode:

1. Snapshot Edison-controlled loaded services and GPU memory.
2. Unload/park Edison-controlled LLM, video, or music resources where available.
3. Flush CUDA caches.
4. Query NVIDIA GPU memory per card.
5. Wait for configured free-VRAM thresholds.
6. Warn if unmanaged external processes still hold VRAM.
7. Restore previously loaded Edison services after completion/failure/cancellation when configured.

Edison does not kill unmanaged external processes.

## Multi-GPU strategy

Persona Video Studio treats GPUs as separate devices. The RTX 3090 24GB, RTX 5060 Ti 16GB, and RTX 4060 Ti 16GB do **not** become a simple pooled 56GB VRAM card.

Supported strategies:

- `Auto`: choose based on backend capabilities.
- `RTX 3090 Primary + Other GPUs for Auxiliary Tasks`: use the highest-memory/3090 card for transforms and reserve other cards for lighter work.
- `Parallel Segment Processing Across All GPUs`: assign different segments to different GPUs if the backend supports segment-level parallelism.
- `3090 Only`: reliability-focused single-card mode.
- `Advanced MultiGPU Backend Mode`: exposed only when the backend advertises safe advanced placement support.

Scheduling logs include entries like `Segment seg_001 → GPU 0 RTX 3090`.

## Configuration

The module is configured under `edison.persona_video` in `config/edison.yaml`:

- enabled/disabled
- default output directory
- temp working directory
- upload directory
- keep intermediates default
- default quality preset
- default GPU strategy
- Exclusive GPU Render Mode default
- auto restore suspended services
- per-GPU VRAM thresholds
- max concurrent segment workers
- failed segment rerender threshold
- QC thresholds
- backend selection
- ComfyUI workflow template directory

## API summary

Through the web reverse proxy, use `/api/persona-video/*`:

- `GET /api/persona-video/health`
- `GET /api/persona-video/settings`
- `GET /api/persona-video/gpus`
- `GET /api/persona-video/backends`
- `GET/POST/DELETE /api/persona-video/personas`
- `POST /api/persona-video/upload-source`
- `POST /api/persona-video/probe`
- `GET/POST /api/persona-video/jobs`
- `GET /api/persona-video/jobs/{job_id}`
- `POST /api/persona-video/jobs/{job_id}/cancel`
- `POST /api/persona-video/jobs/{job_id}/pause`
- `POST /api/persona-video/jobs/{job_id}/resume`
- `POST /api/persona-video/jobs/{job_id}/segments/{segment_id}/retry`
- `GET /api/persona-video/jobs/{job_id}/report`

Core service paths omit the `/api` proxy prefix.

## Setup

1. Confirm `ffmpeg` and `ffprobe` are installed and on `PATH`.
2. Start Edison Core and Web normally.
3. Open `/persona-video-studio` in the web UI.
4. Register a persona pack.
5. Upload or probe a local cleared source video.
6. Complete the rights/consent checklist.
7. Start with the `metadata_only_passthrough` backend for a dry run.
8. Install/register a production backend before expecting actual persona rendering.

## Test job

For a dry run:

1. Use any short, cleared, non-explicit local MP4.
2. Register a persona pack with type `other` and a note like `dry run`.
3. Select backend `Metadata-only passthrough / pipeline validation`.
4. Select transformation scope `Pipeline validation only`.
5. Keep audio preserve enabled.
6. Start the job and watch segment/QC metadata.

The output video should match the source pixels while still producing job metadata, QC summary, and a final sidecar JSON report.

## Troubleshooting

- **VRAM not freeing**: check unmanaged processes in the GPU snapshot; Edison only unloads Edison-controlled services.
- **Missing FFmpeg/FFprobe**: install FFmpeg. Without FFprobe, metadata is limited; without FFmpeg, multi-segment splitting/remuxing is unavailable.
- **Backend unavailable**: inspect `/api/persona-video/backends`; setup-required items list missing workflows/assets.
- **Unsupported persona pack**: verify paths exist and the selected backend supports the pack type.
- **Job fails midway**: inspect job logs and segment QC; retry failed segments if intermediates were retained.
- **Missing/desynced audio**: use preserve mode for source audio, replace mode with a known audio file, or strip mode for silent export.

## Roadmap

- Detector-backed subject tracks and mask-editing UI.
- Real ComfyUI workflow submission bridge with explicit variable mapping.
- ML-based identity, flicker, and temporal stability QC adapters.
- Segment-level resume from retained intermediate manifests.
- Project workspace linkage for deliverables and approvals.
