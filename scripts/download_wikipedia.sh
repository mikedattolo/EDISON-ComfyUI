#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# Download and index Wikipedia data for EDISON's knowledge base
#
# Supports multiple Wikipedia data sources:
# 1. Simple Wikipedia (recommended - ~200MB, fast to index)
# 2. HuggingFace Wikipedia dataset (pre-processed, easy to use)
# 3. Wikipedia Kiwix ZIM files (full offline Wikipedia)
# 4. Raw Wikipedia XML dumps (largest, most complete)
#
# Usage:
#   ./scripts/download_wikipedia.sh              # Download Simple Wikipedia
#   ./scripts/download_wikipedia.sh --full        # Download full English Wikipedia
#   ./scripts/download_wikipedia.sh --kiwix       # Download Kiwix ZIM
#   ./scripts/download_wikipedia.sh --index-only  # Index existing download
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
KNOWLEDGE_DIR="${REPO_ROOT}/models/knowledge"
WIKI_DIR="${KNOWLEDGE_DIR}/wikipedia"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
step()  { echo -e "${BLUE}[STEP]${NC} $*"; }

# ── Parse arguments ──────────────────────────────────────────────────

MODE="simple"
INDEX_ONLY=false
MAX_ARTICLES=0
SKIP_INDEX=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --full)       MODE="full"; shift ;;
        --simple)     MODE="simple"; shift ;;
        --kiwix)      MODE="kiwix"; shift ;;
        --index-only) INDEX_ONLY=true; shift ;;
        --skip-index) SKIP_INDEX=true; shift ;;
        --max-articles) MAX_ARTICLES="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --simple        Download Simple Wikipedia (~200MB) [default]"
            echo "  --full          Download full English Wikipedia (~22GB)"
            echo "  --kiwix         Download Kiwix ZIM file (~10-90GB)"
            echo "  --index-only    Only index existing downloaded files"
            echo "  --skip-index    Only download, don't index"
            echo "  --max-articles N  Limit number of articles to index"
            echo "  -h, --help      Show this help"
            exit 0
            ;;
        *) error "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Setup ────────────────────────────────────────────────────────────

mkdir -p "$WIKI_DIR"

# ── Install dependencies ─────────────────────────────────────────────

install_deps() {
    step "Installing Python dependencies for Wikipedia loading..."
    pip install -q datasets pyarrow 2>/dev/null || true
    
    if [[ "$MODE" == "kiwix" ]]; then
        pip install -q libzim beautifulsoup4 2>/dev/null || true
    fi
    
    if [[ "$MODE" == "full" ]]; then
        pip install -q mwxml mwparserfromhell 2>/dev/null || true
    fi
    
    info "Dependencies installed"
}

# ── Download Simple Wikipedia (HuggingFace) ──────────────────────────

download_simple() {
    step "Downloading Simple Wikipedia via HuggingFace datasets..."
    
    python3 << 'PYEOF'
import os
import json
import sys

wiki_dir = os.environ.get("WIKI_DIR", "./models/knowledge/wikipedia")
os.makedirs(wiki_dir, exist_ok=True)

output_file = os.path.join(wiki_dir, "simple_wikipedia.jsonl")

if os.path.exists(output_file):
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"[INFO] Simple Wikipedia already downloaded ({size_mb:.1f} MB)")
    sys.exit(0)

try:
    from datasets import load_dataset
    print("[INFO] Loading Simple Wikipedia from HuggingFace...")
    
    # Simple Wikipedia - smaller, easier to process
    dataset = load_dataset("wikipedia", "20220301.simple", split="train", trust_remote_code=True)
    
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for article in dataset:
            title = article.get('title', '')
            text = article.get('text', '')
            
            if len(text) < 100:
                continue
            
            json.dump({
                'title': title,
                'text': text,
                'url': f"https://simple.wikipedia.org/wiki/{title.replace(' ', '_')}"
            }, f, ensure_ascii=False)
            f.write('\n')
            count += 1
            
            if count % 10000 == 0:
                print(f"  Processed {count} articles...")
    
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"[INFO] Downloaded {count} articles ({size_mb:.1f} MB)")

except ImportError:
    print("[ERROR] 'datasets' package not installed. Run: pip install datasets")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Failed to download: {e}")
    
    # Fallback: try direct download of pre-processed dump
    print("[INFO] Trying fallback: direct download of Wikipedia dump...")
    import urllib.request
    
    url = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
    dest = os.path.join(wiki_dir, "simplewiki-latest-pages-articles.xml.bz2")
    
    if not os.path.exists(dest):
        print(f"[INFO] Downloading from {url}...")
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"[INFO] Downloaded XML dump ({size_mb:.1f} MB)")
    else:
        print("[INFO] XML dump already exists")
PYEOF
    
    info "Simple Wikipedia download complete"
}

# ── Download Full English Wikipedia ──────────────────────────────────

download_full() {
    step "Downloading Full English Wikipedia via HuggingFace datasets..."
    warn "This is a large download (~22GB). Make sure you have enough disk space."
    
    python3 << 'PYEOF'
import os
import json
import sys

wiki_dir = os.environ.get("WIKI_DIR", "./models/knowledge/wikipedia")
os.makedirs(wiki_dir, exist_ok=True)

output_file = os.path.join(wiki_dir, "english_wikipedia.jsonl")

if os.path.exists(output_file):
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"[INFO] English Wikipedia already downloaded ({size_mb:.1f} MB)")
    sys.exit(0)

try:
    from datasets import load_dataset
    print("[INFO] Loading English Wikipedia from HuggingFace...")
    print("[INFO] This may take 30-60 minutes depending on your connection...")
    
    dataset = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True)
    
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for article in dataset:
            title = article.get('title', '')
            text = article.get('text', '')
            
            if len(text) < 200:
                continue
            
            json.dump({
                'title': title,
                'text': text,
                'url': f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            }, f, ensure_ascii=False)
            f.write('\n')
            count += 1
            
            if count % 50000 == 0:
                print(f"  Processed {count} articles...")
    
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"[INFO] Downloaded {count} articles ({size_mb:.1f} MB)")

except ImportError:
    print("[ERROR] 'datasets' package not installed. Run: pip install datasets")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Failed: {e}")
    sys.exit(1)
PYEOF
    
    info "Full Wikipedia download complete"
}

# ── Download Kiwix ZIM ───────────────────────────────────────────────

download_kiwix() {
    step "Downloading Wikipedia Kiwix ZIM file..."
    warn "ZIM files are large. Simple Wikipedia ZIM is ~300MB, English is ~90GB."
    
    ZIM_URL="https://download.kiwix.org/zim/wikipedia/wikipedia_en_simple_all_nopic_2024-04.zim"
    ZIM_FILE="${WIKI_DIR}/wikipedia_simple.zim"
    
    if [[ -f "$ZIM_FILE" ]]; then
        info "ZIM file already exists: $ZIM_FILE"
        return
    fi
    
    info "Downloading from: $ZIM_URL"
    wget -c -O "$ZIM_FILE" "$ZIM_URL" || {
        error "Download failed. You can manually download from https://wiki.kiwix.org/wiki/Content_in_all_languages"
        exit 1
    }
    
    info "ZIM download complete: $ZIM_FILE"
}

# ── Index downloaded data ────────────────────────────────────────────

index_wikipedia() {
    step "Indexing Wikipedia data into EDISON knowledge base..."
    
    python3 << PYEOF
import os
import sys
import glob

# Add repo root to path
repo_root = os.environ.get("REPO_ROOT", ".")
sys.path.insert(0, repo_root)

wiki_dir = os.environ.get("WIKI_DIR", "./models/knowledge/wikipedia")
max_articles = int(os.environ.get("MAX_ARTICLES", "0"))

# Find Wikipedia data files
data_files = (
    glob.glob(os.path.join(wiki_dir, "*.jsonl")) +
    glob.glob(os.path.join(wiki_dir, "*.json")) +
    glob.glob(os.path.join(wiki_dir, "*.xml.bz2")) +
    glob.glob(os.path.join(wiki_dir, "*.zim")) +
    glob.glob(os.path.join(wiki_dir, "*.parquet"))
)

if not data_files:
    print("[ERROR] No Wikipedia data files found in", wiki_dir)
    print("[INFO] Run this script without --index-only first to download data")
    sys.exit(1)

print(f"[INFO] Found {len(data_files)} data file(s)")

# Initialize knowledge base
try:
    from services.edison_core.knowledge_base import KnowledgeBase
    
    kb_path = os.path.join(repo_root, "models", "knowledge")
    qdrant_path = os.path.join(repo_root, "models", "qdrant_knowledge")
    
    kb = KnowledgeBase(storage_path=kb_path, qdrant_path=qdrant_path)
    
    if not kb.is_ready():
        print("[ERROR] Knowledge base failed to initialize")
        sys.exit(1)
    
    print("[INFO] Knowledge base initialized")
    
    # Load each data file
    total_stats = {"articles_loaded": 0, "chunks_stored": 0, "skipped": 0, "errors": 0}
    
    for data_file in data_files:
        print(f"\n[INFO] Loading: {os.path.basename(data_file)}")
        
        def progress(loaded, total):
            if total > 0:
                pct = (loaded / total) * 100
                print(f"  Progress: {loaded}/{total} articles ({pct:.1f}%)")
            else:
                print(f"  Progress: {loaded} articles loaded...")
        
        stats = kb.load_wikipedia_dump(
            dump_path=data_file,
            max_articles=max_articles,
            min_article_length=200,
            batch_size=500,
            progress_callback=progress
        )
        
        for key in total_stats:
            total_stats[key] += stats.get(key, 0)
        
        if "error" in stats:
            print(f"[WARN] Error during loading: {stats['error']}")
    
    print(f"\n{'='*60}")
    print(f"Wikipedia Indexing Complete!")
    print(f"{'='*60}")
    print(f"  Articles loaded: {total_stats['articles_loaded']}")
    print(f"  Chunks stored:   {total_stats['chunks_stored']}")
    print(f"  Skipped:         {total_stats['skipped']}")
    print(f"  Errors:          {total_stats['errors']}")
    print(f"{'='*60}")
    
    # Show knowledge base stats
    kb_stats = kb.get_stats()
    print(f"\nKnowledge Base Stats:")
    print(f"  Total knowledge points: {kb_stats.get('knowledge_points', 0)}")
    print(f"  Total sources: {kb_stats.get('total_sources', 0)}")

except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("[INFO] Make sure sentence-transformers and qdrant-client are installed")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Indexing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF
    
    info "Wikipedia indexing complete"
}

# ── Main ─────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  EDISON Wikipedia Knowledge Base Setup"
echo "════════════════════════════════════════════════════════════"
echo ""

export WIKI_DIR REPO_ROOT MAX_ARTICLES

if [[ "$INDEX_ONLY" == true ]]; then
    index_wikipedia
    exit 0
fi

install_deps

case "$MODE" in
    simple) download_simple ;;
    full)   download_full ;;
    kiwix)  download_kiwix ;;
esac

if [[ "$SKIP_INDEX" != true ]]; then
    index_wikipedia
fi

echo ""
info "Setup complete! EDISON now has Wikipedia knowledge."
info "To use: The knowledge base is automatically searched during conversations."
echo ""
