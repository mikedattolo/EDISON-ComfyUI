#!/usr/bin/env python3
"""
Download required models for EDISON-ComfyUI
"""

import os
import urllib.request
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODELS = {
    "coral": {
        "mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite": 
            "https://github.com/google-coral/test_data/raw/master/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite",
        "inat_bird_labels.txt":
            "https://github.com/google-coral/test_data/raw/master/inat_bird_labels.txt"
    }
}

def download_file(url: str, dest: Path):
    """Download a file with progress indication"""
    print(f"Downloading {dest.name}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"✓ Downloaded {dest.name}")
    except Exception as e:
        print(f"✗ Error downloading {dest.name}: {e}")

def main():
    print("=== EDISON Model Download ===\n")
    
    # Download Coral models
    coral_dir = MODELS_DIR / "coral"
    coral_dir.mkdir(exist_ok=True)
    
    for filename, url in MODELS["coral"].items():
        dest = coral_dir / filename
        if dest.exists():
            print(f"⊙ {filename} already exists, skipping")
        else:
            download_file(url, dest)
    
    print("\n=== Download Complete ===")

if __name__ == "__main__":
    main()
