# Vision Reliability Troubleshooting

Edison now validates image payloads before routing them to vision models.

The OpenAI-compatible multimodal path and the native vision path now:

- reject empty, broken, unsupported, or non-image base64 payloads
- validate image dimensions with Pillow when available
- re-encode images to a bounded JPEG payload for llama-cpp vision handlers
- add a per-request vision trace ID
- log image count, image dimensions, image hash prefix, and selected model
- route only requests with valid image blocks to the vision model
- add a grounding system prompt that requires visible evidence and uncertainty
- flag generic or nonvisual responses as low confidence

If Edison says vision is unavailable, check:

- `config/edison.yaml` `vision_model` and `vision_clip`
- files exist under `models/llm`
- GPU VRAM guard settings such as `vision_min_free_vram_mb`
- System Doctor PyTorch/CUDA and model readiness checks

For OpenAI-compatible clients, use `/v1/models` to see the configured local model aliases and `/v1/chat/completions` with `content` blocks containing `image_url.url`.

