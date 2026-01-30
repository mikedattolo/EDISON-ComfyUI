# LLM Model Upgrade Guide

## Current Issue
- **Qwen2.5-72B** causes OOM (out of memory) on your GPUs
- Need better models for **code, agent, deep, and work modes**

## Recommended Models (Pick One Strategy)

### Option 1: Use 32B Model (RECOMMENDED - Fits your VRAM)
**Qwen2.5-Coder-32B** - Excellent for code + general tasks
- Size: ~20GB GGUF Q4_K_M
- VRAM: Fits comfortably on 3x GPUs (24GB + 16GB + 12GB = 52GB total)
- Strengths: Best coding model, great reasoning, fast

```bash
# Download on edison server
cd /opt/edison/models/llm
sudo wget https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/qwen2.5-coder-32b-instruct-q4_k_m.gguf
```

Then update `/opt/edison/config/edison.yaml`:
```yaml
medium_model: "qwen2.5-coder-32b-instruct-q4_k_m.gguf"
```

### Option 2: DeepSeek-Coder-33B
Specialized for coding, good at general tasks
```bash
cd /opt/edison/models/llm
sudo wget https://huggingface.co/TheBloke/deepseek-coder-33B-instruct-GGUF/resolve/main/deepseek-coder-33b-instruct.Q4_K_M.gguf
```

### Option 3: Keep Current Setup But Use 32B for All
You already have `qwen2.5-32b-instruct-q4_k_m.gguf` - just make it the primary:
```yaml
# In edison.yaml, set both to use 32B:
medium_model: "qwen2.5-32b-instruct-q4_k_m.gguf"
deep_model: "qwen2.5-32b-instruct-q4_k_m.gguf"  # Disable 72B
```

## What Gets Fixed

### Before (Current Issues):
- ❌ 72B model OOM crashes
- ❌ Limited coding abilities in fast mode (14B)
- ❌ Agent mode underperforms

### After (With 32B Coder):
- ✅ Fits easily in VRAM (20GB across 3 GPUs)
- ✅ Excellent code generation and debugging
- ✅ Fast inference (~2-3 tokens/sec)
- ✅ Great for agent/work/deep modes
- ✅ Web search now returns recent results (today/this week)

## Quick Deploy

**Fastest option** - Use your existing 32B model:
```bash
# On edison server
cd /opt/edison
sudo nano config/edison.yaml
# Change: deep_model: "qwen2.5-32b-instruct-q4_k_m.gguf"
sudo systemctl restart edison-core
```

## Web Search Fix (Already Applied)
- Added automatic date filtering when you say "today", "latest", "recent", "news"
- Examples:
  - "What's the news today?" → filters to last 24 hours
  - "Recent developments in AI" → filters to last week
  - "What happened this week?" → filters to last 7 days

## Test After Upgrade

Try these prompts:
1. **Code Mode**: "Write a Python function to parse JSON with error handling"
2. **Agent Mode**: "Research the latest AI news from today"
3. **Deep Mode**: "Explain quantum computing in detail"
4. **Web Search**: "What's the tech news today?" (should show current day)

The 32B model should handle all of these much better than the 14B fast model!
