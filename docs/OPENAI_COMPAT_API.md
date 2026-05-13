# OpenAI-Compatible API Notes

Edison exposes OpenAI-style local endpoints for external clients:

- `GET /v1/models`
- `POST /v1/chat/completions`

Model aliases include `fast`, `medium`, `deep`, `reasoning`, `vision`, `vision_code`, `gpt-3.5-turbo`, `gpt-4`, and `gpt-4-vision-preview`, mapped to local configured models where available.

Multimodal requests must use OpenAI-style content blocks:

```json
{
  "role": "user",
  "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    {"type": "text", "text": "Describe only what is visible."}
  ]
}
```

Broken or unsupported image payloads return a `400` with a trace ID instead of silently routing to a text model. Streaming vision requests preserve the multimodal message list for the vision handler.

