#!/bin/bash
# Setup whisper.cpp and Piper for voice mode (no Python dependencies)

set -e

EDISON_ROOT="/opt/edison"
BIN_DIR="$EDISON_ROOT/voice_binaries"

echo "🎙️ Setting up voice mode binaries..."

# Create binaries directory
mkdir -p "$BIN_DIR"
cd "$BIN_DIR"

# Install whisper.cpp
echo "📥 Installing whisper.cpp..."
if [ ! -d "whisper.cpp" ]; then
    git clone https://github.com/ggerganov/whisper.cpp.git
    cd whisper.cpp
    make
    
    # Download base model (good balance of speed/accuracy)
    bash ./models/download-ggml-model.sh base.en
    cd ..
fi

# Install Piper TTS
echo "📥 Installing Piper TTS..."
if [ ! -f "piper" ]; then
    # Download Piper release
    wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz
    tar -xzf piper_amd64.tar.gz
    rm piper_amd64.tar.gz
    
    # Download a voice model (en_US-lessac-medium)
    mkdir -p voices
    cd voices
    wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx
    wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
    cd ..
fi

# Set permissions
chmod +x piper
chmod +x whisper.cpp/main

echo "✅ Voice binaries installed:"
echo "   whisper.cpp: $BIN_DIR/whisper.cpp/main"
echo "   Piper: $BIN_DIR/piper"
echo "   Voice model: $BIN_DIR/voices/en_US-lessac-medium.onnx"
