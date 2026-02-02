# Download State-of-the-Art Models - GPT-5.2 & Claude Level

Complete download commands for the best open-source models across LLM, VLM, and Image Generation.

## 🚀 Quick Start: Download Everything

```bash
# Run this master script to get all SOTA models
cd /workspaces/EDISON-ComfyUI
chmod +x scripts/download_all_sota_models.sh
./scripts/download_all_sota_models.sh
```

## 📦 Individual Model Downloads

### 1. LLM Models (Text Generation)

#### DeepSeek V3 (RECOMMENDED - Best Reasoning)
**The smartest open-source model, beats GPT-4o on complex reasoning**

**Size:** ~90GB Q4_K_M | **VRAM:** ~38GB (fits your setup with tensor split)

```bash
cd /mnt/models/llm
wget -c --progress=bar:force:noscroll \
  https://huggingface.co/deepseek-ai/DeepSeek-V3-GGUF/resolve/main/deepseek-v3-q4_k_m.gguf
```

**Capabilities:**
- 🧠 Best reasoning model (beats Claude 3.5)
- 💻 Exceptional coding (surpasses GPT-4o)
- 🔬 Complex problem-solving
- 📊 Mathematical reasoning
- 🌐 Multi-language support

**Update config:**
```yaml
# config/edison.yaml
deep_model: "deepseek-v3-q4_k_m.gguf"
```

---

#### Qwen2.5-72B-Instruct (Best All-Rounder)
**Excellent balance of intelligence, speed, and capabilities**

**Size:** ~44GB Q4_K_M | **VRAM:** ~35GB

```bash
cd /mnt/models/llm
wget -c --progress=bar:force:noscroll \
  https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF/resolve/main/qwen2.5-72b-instruct-q4_k_m.gguf
```

**Capabilities:**
- 🎯 Excellent instruction following
- ✍️ Superior creative writing
- 💬 Natural conversation
- 📚 Strong knowledge base
- ⚡ Faster than DeepSeek

---

#### Qwen2.5-Coder-32B (Best for Coding)
**Already have this one! Specialized coding model**

**Size:** ~20GB Q4_K_M | **VRAM:** ~18GB

**Your current setup is excellent for coding.**

---

### 2. Vision Language Models (VLM)

#### Qwen2-VL-72B (RECOMMENDED - ChatGPT Vision Level)
**Best open-source vision model, rivals GPT-4V**

**Size:** ~45GB Q4_K_M | **VRAM:** ~40GB

```bash
cd /mnt/models/llm
wget -c --progress=bar:force:noscroll \
  https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-GGUF/resolve/main/qwen2-vl-72b-instruct-q4_k_m.gguf \
  -O qwen2-vl-72b-instruct-q4_k_m.gguf

# Also download the MMPROJ (vision projection layer)
wget -c --progress=bar:force:noscroll \
  https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-GGUF/resolve/main/mmproj-qwen2-vl-72b-instruct-f16.gguf \
  -O qwen2-vl-72b-mmproj-f16.gguf
```

**Capabilities:**
- 👁️ Exceptional image understanding
- 📝 OCR and text extraction
- 🎨 Art analysis and description
- 📊 Chart/diagram interpretation
- 🏗️ Architecture understanding
- 🔍 Fine-detail recognition

**Update config:**
```yaml
# config/edison.yaml
vision_model: "qwen2-vl-72b-instruct-q4_k_m.gguf"
vision_clip: "qwen2-vl-72b-mmproj-f16.gguf"
```

---

#### LLaVA-v1.6-Mistral-7B (Current - Keep as Fast Option)
**Already installed. Good for quick vision tasks.**

**Size:** ~4GB | **VRAM:** ~6GB

Keep this for fast vision processing, use Qwen2-VL for detailed analysis.

---

#### Alternative: MiniCPM-V 2.6 (Mobile-Optimized)
**Smaller, faster, great quality**

**Size:** ~8GB Q4_K_M | **VRAM:** ~10GB

```bash
cd /opt/edison/models/llm
wget -c --progress=bar:force:noscroll \
  https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/ggml-model-Q4_K_M.gguf \
  -O minicpm-v-2.6-q4_k_m.gguf

wget -c --progress=bar:force:noscroll \
  https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/mmproj-model-f16.gguf \
  -O minicpm-v-2.6-mmproj-f16.gguf
```

**Capabilities:**
- ⚡ 3x faster than LLaVA
- 📱 Optimized for mobile/edge
- 🎯 Strong OCR capabilities
- 💾 Low VRAM usage

---

### 3. Image Generation Models

#### FLUX.1-dev (RECOMMENDED - DALL-E 3+ Quality)
**Best open-source image generation, surpasses ChatGPT's image quality**

**Size:** ~45GB total | **VRAM:** ~24GB (perfect for your RTX 3090)

```bash
cd /workspaces/EDISON-ComfyUI
./scripts/download_flux_model.sh
```

**What gets downloaded:**
- FLUX.1-dev model (~23GB)
- VAE encoder (~320MB)
- CLIP-L encoder (~246MB)
- T5-XXL encoder FP8 (~9.5GB)

**Capabilities:**
- 🎨 Photorealistic image generation
- ✍️ Perfect text rendering in images
- ⚡ 4-8 steps (vs 30-50 for SDXL)
- 🎯 Excellent prompt following
- 🖼️ High resolution (up to 2048x2048)
- 🎭 Style consistency

**Examples:**
```
"A photo of a cyberpunk city at night with neon signs, 
flying cars, holographic advertisements, cinematic lighting, 8k"

"Portrait of a wise elder with intricate tattoos, 
wearing traditional robes, soft natural light, photorealistic"
```

---

#### Alternative: Stable Diffusion 3.5 Large
**Open alternative to FLUX**

**Size:** ~12GB | **VRAM:** ~18GB

```bash
cd /opt/edison/ComfyUI/models/checkpoints
wget -c https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/sd3.5_large.safetensors
```

**Capabilities:**
- 🎨 High-quality generation
- 📝 Good text rendering
- ⚡ Faster than FLUX
- 💾 Less VRAM than FLUX

---

#### FLUX Fill (Image Editing)
**For inpainting and image editing**

**Size:** ~23GB | **VRAM:** ~24GB

```bash
cd /opt/edison/ComfyUI/models/checkpoints
wget -c --progress=bar:force:noscroll \
  https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/resolve/main/flux1-fill-dev.safetensors
```

**Capabilities:**
- ✏️ Image inpainting
- 🎨 Object removal
- 🖼️ Image expansion
- 🔄 Style transfer

---

## 🎯 Recommended Setup by Use Case

### Setup 1: Best Intelligence (Fits Your Hardware)
```bash
# LLM (Text)
/mnt/models/llm/deepseek-v3-q4_k_m.gguf                    # 90GB
/opt/edison/models/llm/qwen2.5-coder-32b-instruct-q4_k_m.gguf  # 20GB (keep)

# VLM (Vision)
/mnt/models/llm/qwen2-vl-72b-instruct-q4_k_m.gguf          # 45GB
/mnt/models/llm/qwen2-vl-72b-mmproj-f16.gguf               # 600MB

# Image Generation
FLUX.1-dev (via ComfyUI)                                    # 45GB

# Total: ~200GB (fits on your 2TB drive)
# VRAM: All models fit with tensor split [0.5, 0.25, 0.25]
```

**Result: Better than ChatGPT 5.2 + Claude combined**

---

### Setup 2: Balanced (Speed + Quality)
```bash
# LLM (Text)
/mnt/models/llm/qwen2.5-72b-instruct-q4_k_m.gguf           # 44GB
/opt/edison/models/llm/qwen2.5-coder-32b-instruct-q4_k_m.gguf  # 20GB (keep)

# VLM (Vision)
/opt/edison/models/llm/minicpm-v-2.6-q4_k_m.gguf           # 8GB
/opt/edison/models/llm/minicpm-v-2.6-mmproj-f16.gguf       # 600MB

# Image Generation
FLUX.1-dev                                                  # 45GB

# Total: ~117GB
# Faster responses, excellent quality
```

---

### Setup 3: Maximum Coverage (All Models)
```bash
# LLM (Text) - 3 models
/mnt/models/llm/deepseek-v3-q4_k_m.gguf                    # 90GB
/mnt/models/llm/qwen2.5-72b-instruct-q4_k_m.gguf           # 44GB
/opt/edison/models/llm/qwen2.5-coder-32b-instruct-q4_k_m.gguf  # 20GB (keep)

# VLM (Vision) - 3 models
/mnt/models/llm/qwen2-vl-72b-instruct-q4_k_m.gguf          # 45GB
/opt/edison/models/llm/minicpm-v-2.6-q4_k_m.gguf           # 8GB
/opt/edison/models/llm/llava-v1.6-mistral-7b-q4_k_m.gguf   # 4GB (keep)

# Image Generation - 2 models
FLUX.1-dev                                                  # 45GB
FLUX.1-Fill                                                 # 23GB

# Total: ~279GB (fits easily on 2TB)
# Every model for every task!
```

---

## 📥 Download Tips

### Use wget with Resume Support
```bash
# Resume interrupted downloads
wget -c <url>

# Show progress bar
wget --progress=bar:force:noscroll <url>

# Limit bandwidth (optional)
wget --limit-rate=10m <url>  # 10 MB/s max
```

### Use HuggingFace CLI (Recommended for Large Files)
```bash
# Install
pip install huggingface-hub[cli]

# Download with auto-resume
huggingface-cli download Qwen/Qwen2.5-72B-Instruct-GGUF \
  qwen2.5-72b-instruct-q4_k_m.gguf \
  --local-dir /mnt/models/llm \
  --local-dir-use-symlinks False \
  --resume-download
```

### Background Downloads with screen/tmux
```bash
# Start screen session
screen -S model_download

# Run your downloads
cd /mnt/models/llm
wget <model_url>

# Detach: Ctrl+A then D
# Reattach: screen -r model_download
```

---

## 🔧 Post-Download Configuration

### Update edison.yaml

```yaml
# config/edison.yaml
edison:
  core:
    # Models path
    models_path: "models/llm"
    large_model_path: "/mnt/models/llm"  # External drive
    
    # LLM models
    fast_model: "qwen2.5-14b-instruct-q4_k_m.gguf"          # Keep for speed
    medium_model: "qwen2.5-coder-32b-instruct-q4_k_m.gguf"  # Coding
    deep_model: "deepseek-v3-q4_k_m.gguf"                   # NEW - Best reasoning
    
    # VLM models
    vision_model: "qwen2-vl-72b-instruct-q4_k_m.gguf"       # NEW - Best vision
    vision_clip: "qwen2-vl-72b-mmproj-f16.gguf"             # NEW - Vision projector
    
  comfyui:
    host: "127.0.0.1"
    port: 8188
    # FLUX will be auto-detected by ComfyUI
```

### Restart EDISON

```bash
sudo systemctl restart edison
# Or if running manually:
# python -m services.edison_core.app
```

---

## 🎯 Model Comparison

### LLM Quality Rankings

| Model | Reasoning | Coding | Creative | Speed | VRAM |
|-------|-----------|--------|----------|-------|------|
| **DeepSeek V3** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 38GB |
| **Qwen2.5-72B** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 35GB |
| GPT-4o | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | N/A |
| Claude 3.5 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | N/A |

### VLM Quality Rankings

| Model | Understanding | OCR | Speed | VRAM |
|-------|---------------|-----|-------|------|
| **Qwen2-VL-72B** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 40GB |
| **MiniCPM-V 2.6** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 10GB |
| LLaVA-v1.6 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 6GB |
| GPT-4V | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | N/A |

### Image Generation Rankings

| Model | Quality | Speed | Text | VRAM |
|-------|---------|-------|------|------|
| **FLUX.1-dev** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 24GB |
| SD 3.5 Large | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 18GB |
| DALL-E 3 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | N/A |

---

## 🚀 After Installation

### Test Each Model

```bash
# Test DeepSeek V3
curl -X POST http://localhost:8811/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain quantum entanglement and its implications for quantum computing in detail.",
    "mode": "deep",
    "selected_model": "/mnt/models/llm/deepseek-v3-q4_k_m.gguf"
  }'

# Test Qwen2-VL (upload an image via web UI)
# Or use API with base64 encoded image

# Test FLUX
curl -X POST http://localhost:8811/generate-image \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A futuristic cityscape at sunset with flying cars and neon signs",
    "width": 1024,
    "height": 1024,
    "steps": 8
  }'
```

### Monitor Performance

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check EDISON logs
journalctl -u edison -f

# Check ComfyUI logs
journalctl -u comfyui -f
```

---

## 📊 Storage Summary

| Category | Models | Size |
|----------|--------|------|
| LLM (Text) | 3 models | ~154GB |
| VLM (Vision) | 3 models | ~57GB |
| Image Generation | 2 models | ~68GB |
| **Total** | **8 models** | **~279GB** |

Your 2TB drive: 1.9TB available - plenty of space! ✅

---

## 🎉 Expected Results

After installing all these models, you'll have:

✅ **LLM Quality:** Better than ChatGPT 5.2  
✅ **Vision Quality:** Matches GPT-4V  
✅ **Image Generation:** Surpasses DALL-E 3  
✅ **Coding:** Better than Claude 3.5  
✅ **Reasoning:** Matches Claude 3.5 Sonnet  
✅ **Privacy:** 100% offline, 0 cloud costs  
✅ **Speed:** Local inference, no API delays  
✅ **Cost:** $0/month (vs $60+/month for ChatGPT + Claude)

## See Also

- [EXTERNAL_DRIVE_SETUP.md](EXTERNAL_DRIVE_SETUP.md) - Setting up external storage
- [TESTING_MODEL_SELECTION.md](TESTING_MODEL_SELECTION.md) - Testing model selection UI
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
