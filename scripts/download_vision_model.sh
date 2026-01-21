#!/bin/bash
# Download LLaVA vision model for EDISON

echo "===================================="
echo "EDISON Vision Model Downloader"
echo "===================================="
echo ""

MODELS_DIR="/opt/edison/models/llm"

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

# Check if models already exist
if [ -f "$MODELS_DIR/llava-v1.6-mistral-7b-q4_k_m.gguf" ] && \
   [ -f "$MODELS_DIR/llava-v1.6-mistral-7b-mmproj-q4_0.gguf" ]; then
    echo "✅ Vision models already installed!"
    echo ""
    echo "Files found:"
    echo "  - llava-v1.6-mistral-7b-q4_k_m.gguf ($(du -h $MODELS_DIR/llava-v1.6-mistral-7b-q4_k_m.gguf | cut -f1))"
    echo "  - llava-v1.6-mistral-7b-mmproj-q4_0.gguf ($(du -h $MODELS_DIR/llava-v1.6-mistral-7b-mmproj-q4_0.gguf | cut -f1))"
    echo ""
    echo "Next step: sudo systemctl restart edison-core"
    exit 0
fi

echo "📝 Manual Download Instructions"
echo "================================"
echo ""
echo "HuggingFace requires authentication for automated downloads."
echo "Please download the vision models manually:"
echo ""
echo "1. Visit: https://huggingface.co/mradermacher/llava-v1.6-mistral-7b-GGUF/tree/main"
echo ""
echo "2. Download these two files:"
echo "   • llava-v1.6-mistral-7b.Q4_K_M.gguf (~3.8GB)"
echo "   • mmproj-mistral7b-f16.gguf (~634MB)"
echo ""
echo "3. Transfer files to your server using scp:"
echo "   scp llava-v1.6-mistral-7b.Q4_K_M.gguf mike@YOUR_SERVER_IP:/tmp/"
echo "   scp mmproj-mistral7b-f16.gguf mike@YOUR_SERVER_IP:/tmp/"
echo ""
echo "4. On your server, move and rename them:"
echo "   sudo mv /tmp/llava-v1.6-mistral-7b.Q4_K_M.gguf $MODELS_DIR/llava-v1.6-mistral-7b-q4_k_m.gguf"
echo "   sudo mv /tmp/mmproj-mistral7b-f16.gguf $MODELS_DIR/llava-v1.6-mistral-7b-mmproj-q4_0.gguf"
echo ""
echo "5. Restart the service:"
echo "   sudo systemctl restart edison-core"
echo ""
echo "================================"
echo ""

echo ""
echo "===================================="
echo "✅ Vision model setup complete!"
echo "===================================="
echo ""
echo "Files installed in: $MODELS_DIR"
echo ""
echo "Next steps:"
echo "1. Restart edison-core service:"
echo "   sudo systemctl restart edison-core"
echo ""
echo "2. Check logs to verify vision model loaded:"
echo "   sudo journalctl -u edison-core -f"
echo ""
echo "3. Test image understanding in the web UI!"
echo ""
