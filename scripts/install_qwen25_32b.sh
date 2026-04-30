#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models/llm}"
TARGET_FILE="qwen2.5-32b-instruct-q4_k_m.gguf"
TARGET_PATH="$MODELS_DIR/$TARGET_FILE"
SOURCE_URL="https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf"

mkdir -p "$MODELS_DIR"

if [[ -f "$TARGET_PATH" ]]; then
    echo "✓ Model already installed: $TARGET_PATH"
    exit 0
fi

TMP_PATH="$TARGET_PATH.partial"
AUTH_HEADER=()

if [[ -n "${HF_TOKEN:-}" ]]; then
    AUTH_HEADER=(-H "Authorization: Bearer $HF_TOKEN")
else
    echo "[INFO] HF_TOKEN not set. Downloading anonymously may be slower or rate-limited."
fi

echo "Downloading Qwen2.5 32B Instruct Q4_K_M to $TARGET_PATH"
echo "Source: $SOURCE_URL"

rm -f "$TMP_PATH"
curl -L --fail --progress-bar "${AUTH_HEADER[@]}" -o "$TMP_PATH" "$SOURCE_URL"
mv "$TMP_PATH" "$TARGET_PATH"

echo "✓ Installed: $TARGET_PATH"
echo "Next: restart EDISON so it picks up the new primary model."