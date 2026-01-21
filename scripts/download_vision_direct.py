#!/usr/bin/env python3
"""
Direct download script for LLaVA vision models
Downloads files directly to the server without needing scp
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

MODELS_DIR = Path("/opt/edison/models/llm")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model URLs - using direct CDN links
MODELS = {
    "llava-v1.6-mistral-7b-q4_k_m.gguf": {
        "url": "https://huggingface.co/mradermacher/llava-v1.6-mistral-7b-GGUF/resolve/main/llava-v1.6-mistral-7b.Q4_K_M.gguf",
        "size": "3.8GB"
    },
    "llava-v1.6-mistral-7b-mmproj-q4_0.gguf": {
        "url": "https://huggingface.co/mradermacher/llava-v1.6-mistral-7b-GGUF/resolve/main/mmproj-mistral7b-f16.gguf",
        "size": "634MB"
    }
}

def download_file(url: str, dest: Path, desc: str):
    """Download a file with progress bar"""
    print(f"\n📥 Downloading {desc}...")
    print(f"   URL: {url}")
    print(f"   Destination: {dest}")
    
    # Check if file already exists
    if dest.exists():
        print(f"   ✅ File already exists ({dest.stat().st_size / (1024**3):.2f} GB)")
        return True
    
    try:
        # Stream download with progress bar
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
        
        print(f"   ✅ Downloaded successfully ({dest.stat().st_size / (1024**3):.2f} GB)")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to download: {e}")
        if dest.exists():
            dest.unlink()  # Remove partial file
        return False

def main():
    print("=" * 50)
    print("EDISON Vision Model Direct Downloader")
    print("=" * 50)
    print()
    
    # Check if both files already exist
    all_exist = all((MODELS_DIR / name).exists() for name in MODELS.keys())
    if all_exist:
        print("✅ All vision models already downloaded!")
        print()
        print("Files found:")
        for name in MODELS.keys():
            path = MODELS_DIR / name
            size_gb = path.stat().st_size / (1024**3)
            print(f"   • {name} ({size_gb:.2f} GB)")
        print()
        print("Next step: sudo systemctl restart edison-core")
        return 0
    
    print("📦 This will download ~4.4GB of files")
    print()
    
    # Download each model
    success = True
    for filename, info in MODELS.items():
        dest = MODELS_DIR / filename
        if not download_file(info["url"], dest, f"{filename} ({info['size']})"):
            success = False
            break
    
    print()
    if success:
        print("=" * 50)
        print("✅ Vision models downloaded successfully!")
        print("=" * 50)
        print()
        print("Next steps:")
        print("1. Restart edison-core service:")
        print("   sudo systemctl restart edison-core")
        print()
        print("2. Check logs to verify vision model loaded:")
        print("   sudo journalctl -u edison-core -f")
        print()
        print("3. Test image understanding in the web UI!")
        print()
        return 0
    else:
        print("=" * 50)
        print("❌ Download failed")
        print("=" * 50)
        print()
        print("This might be due to HuggingFace rate limiting.")
        print("Please try again in a few minutes, or:")
        print("1. Visit: https://huggingface.co/mradermacher/llava-v1.6-mistral-7b-GGUF")
        print("2. Download files manually from another computer")
        print("3. Use a USB drive to transfer to server")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
