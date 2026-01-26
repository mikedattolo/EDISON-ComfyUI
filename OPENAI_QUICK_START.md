# Quick Reference: OpenAI-Compatible EDISON

## Endpoint
```
POST http://localhost:8811/v1/chat/completions
```

## Basic Request
```json
{
  "model": "fast",
  "messages": [
    {"role": "user", "content": "What is 2+2?"}
  ]
}
```

## Python Example (Most Common)
```python
from openai import OpenAI

client = OpenAI(api_key="sk-not-needed", base_url="http://localhost:8811/v1")

response = client.chat.completions.create(
    model="fast",
    messages=[{"role": "user", "content": "Hi!"}]
)
print(response.choices[0].message.content)
```

## Model Aliases
| Name | Maps To | Speed | Quality |
|------|---------|-------|---------|
| `gpt-3.5-turbo` | fast | Fast | Good |
| `gpt-4` | deep | Slower | Better |
| `fast` | 14B model | Fast | Good |
| `medium` | 32B model | Medium | Better |
| `deep` | 72B model | Slow | Best |

## Streaming Example
```python
with client.chat.completions.create(
    model="fast",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

## cURL Examples

### Non-Streaming
```bash
curl http://localhost:8811/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fast",
    "messages": [{"role": "user", "content": "Hi"}]
  }'
```

### Streaming
```bash
curl http://localhost:8811/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fast",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": true
  }'
```

## Response Format
```json
{
  "id": "chatcmpl-xyz",
  "object": "chat.completion",
  "created": 1706284800,
  "model": "qwen2.5-fast",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "The answer is 4."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 4,
    "total_tokens": 14
  }
}
```

## All Supported Parameters
```python
client.chat.completions.create(
    model="fast",                      # Model: fast, medium, deep, or OpenAI names
    messages=[...],                    # Required: conversation messages
    temperature=0.7,                   # 0.0-2.0, default 0.7
    top_p=0.9,                        # 0.0-1.0, default 0.9
    max_tokens=2048,                  # Max generation length, default 2048
    stream=False,                      # Stream tokens if true
    tools=None,                        # Optional: tools/functions (parsed only)
    tool_choice=None                   # Optional: tool selection
)
```

## Status & Features
✅ Streaming with SSE
✅ Server-side cancellation (<1 second)
✅ Full OpenAI client compatibility
✅ Model routing & fallback
✅ Token usage tracking
✅ Concurrent request support
✅ Error handling

## Test Your Setup
```bash
python test_openai_compat.py
```

## Common Issues

**"No suitable model available"**
- Not enough VRAM for requested model
- Fallback to smaller model (fast instead of deep)

**Streaming chunks out of order**
- Normal for SSE format
- Client library handles reassembly

**Token count inaccurate**
- Using approximate word-based count (not exact)
- Good enough for most use cases

## Architecture
```
Your OpenAI Client
    ↓
/v1/chat/completions (EDISON)
    ↓
Model Router (gpt-3.5 → fast)
    ↓
EDISON Core (llama-cpp-python)
    ↓
Response (JSON or SSE)
```

## Next Steps
1. Update your client code to use: `http://localhost:8811/v1`
2. Choose model: `fast` (quick), `deep` (better)
3. Try streaming with `stream=True`
4. Enjoy drop-in OpenAI compatibility!

---

See `OPENAI_COMPAT.md` for full documentation
