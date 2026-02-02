# Large Model Download Commands

Quick reference for downloading state-of-the-art large language models to your external drive.

## Prerequisites

- External drive mounted at `/mnt/models` (see [EXTERNAL_DRIVE_SETUP.md](EXTERNAL_DRIVE_SETUP.md))
- At least 100GB free space for a single large model
- Fast internet connection (models are 40-90GB each)
- `wget` or `curl` installed

## Quick Commands

### Qwen2.5-72B-Instruct (Recommended)

The best all-around large model - excellent at reasoning, coding, creative writing, and instruction following.

**Size:** ~44GB (Q4_K_M quantization)

```bash
cd /mnt/models/llm
wget https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF/resolve/main/qwen2.5-72b-instruct-q4_k_m.gguf
```

### DeepSeek V3 (Most Intelligent)

The most capable reasoning model, excellent for complex problem-solving and deep analysis.

**Size:** ~90GB (Q4_K_M quantization)

```bash
cd /mnt/models/llm
wget https://huggingface.co/deepseek-ai/DeepSeek-V3-GGUF/resolve/main/deepseek-v3-q4_k_m.gguf
```

### Qwen2.5-Coder-32B (Best for Coding)

Specialized coding model with excellent code generation and understanding.

**Size:** ~20GB (Q4_K_M quantization)

```bash
cd /mnt/models/llm
wget https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/qwen2.5-coder-32b-instruct-q4_k_m.gguf
```

## Higher Quality Quantizations

For better quality at the cost of more disk space and VRAM:

### Qwen2.5-72B (Q5_K_M - Better Quality)

**Size:** ~53GB

```bash
cd /mnt/models/llm
wget https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF/resolve/main/qwen2.5-72b-instruct-q5_k_m.gguf
```

### DeepSeek V3 (Q5_K_M - Better Quality)

**Size:** ~110GB

```bash
cd /mnt/models/llm
wget https://huggingface.co/deepseek-ai/DeepSeek-V3-GGUF/resolve/main/deepseek-v3-q5_k_m.gguf
```

## Alternative: Using Hugging Face CLI

For more reliable downloads with resume support:

### Install Hugging Face CLI

```bash
pip install huggingface-hub[cli]
```

### Download Models

```bash
# Qwen2.5-72B
huggingface-cli download Qwen/Qwen2.5-72B-Instruct-GGUF \
  qwen2.5-72b-instruct-q4_k_m.gguf \
  --local-dir /mnt/models/llm \
  --local-dir-use-symlinks False

# DeepSeek V3
huggingface-cli download deepseek-ai/DeepSeek-V3-GGUF \
  deepseek-v3-q4_k_m.gguf \
  --local-dir /mnt/models/llm \
  --local-dir-use-symlinks False
```

## Download Progress Monitoring

### Using wget with progress bar

```bash
wget --progress=bar:force:noscroll -c \
  https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF/resolve/main/qwen2.5-72b-instruct-q4_k_m.gguf \
  -O /mnt/models/llm/qwen2.5-72b-instruct-q4_k_m.gguf
```

### Using curl with progress

```bash
curl -L --progress-bar -C - \
  https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF/resolve/main/qwen2.5-72b-instruct-q4_k_m.gguf \
  -o /mnt/models/llm/qwen2.5-72b-instruct-q4_k_m.gguf
```

## Download in Background

For long downloads, use `screen` or `tmux`:

### Using screen

```bash
# Start screen session
screen -S model_download

# Run download
cd /mnt/models/llm
wget https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF/resolve/main/qwen2.5-72b-instruct-q4_k_m.gguf

# Detach: Ctrl+A, then D
# Reattach later: screen -r model_download
```

### Using tmux

```bash
# Start tmux session
tmux new -s model_download

# Run download
cd /mnt/models/llm
wget https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF/resolve/main/qwen2.5-72b-instruct-q4_k_m.gguf

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t model_download
```

## Verify Downloaded Files

After downloading, verify the file integrity:

```bash
# Check file size
ls -lh /mnt/models/llm/*.gguf

# Check if file is complete (not truncated)
file /mnt/models/llm/*.gguf

# Expected output: "data" or similar
```

## Update EDISON Configuration

After downloading, update your configuration to use the new models:

```bash
nano /workspaces/EDISON-ComfyUI/config/edison.yaml
```

Add or update model paths:

```yaml
# Models
models:
  fast: "qwen2.5-14b-instruct-q4_k_m.gguf"
  medium: "qwen2.5-coder-32b-instruct-q4_k_m.gguf"
  deep: "qwen2.5-72b-instruct-q4_k_m.gguf"  # On external drive

# External drive models
large_model_path: "/mnt/models/llm"
```

## Restart EDISON

After downloading models and updating config:

```bash
# If running as systemd service
sudo systemctl restart edison

# Or if running manually
# Stop current instance (Ctrl+C)
# Then restart:
cd /workspaces/EDISON-ComfyUI
python -m services.edison_core.app
```

## Model Selection in UI

Once models are downloaded and EDISON is restarted:

1. Open the EDISON web UI
2. Select a mode (Chat, Deep, Code, Agent, or Work)
3. Click the **model selector dropdown** (visible in all modes except Auto)
4. Choose your desired model from the list
5. Start chatting!

The UI will show all available models from both your main drive (`/opt/edison/models/llm`) and external drive (`/mnt/models/llm`).

## Recommended Setup

For a balanced setup with 3 models:

| Model | Size | Use Case | Location |
|-------|------|----------|----------|
| Qwen2.5-14B-Instruct | ~8GB | Fast responses, casual chat | Main drive |
| Qwen2.5-Coder-32B | ~20GB | Code generation/review | Main drive |
| Qwen2.5-72B-Instruct | ~44GB | Deep thinking, complex tasks | External drive |

**Total:** ~72GB

With DeepSeek V3 as well:

| Model | Size | Use Case | Location |
|-------|------|----------|----------|
| Qwen2.5-14B-Instruct | ~8GB | Fast responses | Main drive |
| Qwen2.5-Coder-32B | ~20GB | Coding | Main drive |
| Qwen2.5-72B-Instruct | ~44GB | Complex tasks | External drive |
| DeepSeek V3 | ~90GB | Maximum intelligence | External drive |

**Total:** ~162GB

## Download Time Estimates

Approximate download times based on internet speed:

| Connection | 44GB (Qwen2.5-72B) | 90GB (DeepSeek V3) |
|------------|--------------------|--------------------|
| 100 Mbps | ~1 hour | ~2 hours |
| 500 Mbps | ~12 minutes | ~24 minutes |
| 1 Gbps | ~6 minutes | ~12 minutes |

## Troubleshooting

### Download interrupted

Use `-c` flag with wget to resume:

```bash
wget -c https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF/resolve/main/qwen2.5-72b-instruct-q4_k_m.gguf
```

### Out of space

Check available space:

```bash
df -h /mnt/models
```

Delete unnecessary models or use a larger external drive.

### Slow download speed

- Check your internet connection
- Try a different time of day (HuggingFace can be busy)
- Use the HuggingFace CLI with CDN acceleration
- Consider downloading from a mirror if available

### Permission denied

Fix permissions:

```bash
sudo chown -R $USER:$USER /mnt/models/llm
chmod 755 /mnt/models/llm
```

### Model not showing in UI

1. Check the model file is in the correct directory:
   ```bash
   ls -lh /mnt/models/llm/
   ```

2. Verify `large_model_path` in `config/edison.yaml`

3. Restart EDISON:
   ```bash
   sudo systemctl restart edison
   ```

4. Check logs:
   ```bash
   journalctl -u edison -f | grep -i "scanning"
   ```

## Alternative Download Locations

If HuggingFace is slow, try these mirrors:

### ModelScope (China mirror)

```bash
# Install ModelScope CLI
pip install modelscope

# Download models
modelscope download --model Qwen/Qwen2.5-72B-Instruct-GGUF \
  --local_dir /mnt/models/llm
```

### Direct from URLs

Some models may be available from alternative sources. Check the model's page for mirrors.

## Summary

You now know how to download and configure large language models for EDISON!

**Next steps:**
1. Choose your models based on your use case and available space
2. Download using the commands above
3. Update `config/edison.yaml` if needed
4. Restart EDISON
5. Select models in the UI and start chatting!

## See Also

- [EXTERNAL_DRIVE_SETUP.md](EXTERNAL_DRIVE_SETUP.md) - Setting up external storage
- [UPGRADE_TO_SOTA.md](UPGRADE_TO_SOTA.md) - Detailed model comparisons
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
