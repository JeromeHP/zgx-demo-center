#!/bin/bash
# =============================================================================
# Maritime Surveillance Intelligence Generator - Model Download
# Downloads Qwen3-VL-8B-Instruct (~16GB) from HuggingFace
# =============================================================================

set -e

echo ""
echo "============================================================"
echo "  Maritime Surveillance Model Download"
echo "============================================================"
echo ""
echo "  Model: Qwen/Qwen3-VL-8B-Instruct (~11GB)"
echo ""

MODEL_DIR="./models/Qwen3-VL-8B-Instruct-FP8"

# Check if already downloaded
if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "  ✅ Model already exists at $MODEL_DIR"
    echo "     To re-download, delete the directory first."
    echo ""
    exit 0
fi

mkdir -p ./models

# Create a temporary venv if not already in one
if [ -z "$VIRTUAL_ENV" ]; then
    echo "  Creating temporary Python environment..."
    python3 -m venv /tmp/hf-download-env
    source /tmp/hf-download-env/bin/activate
    pip install -q huggingface_hub
fi

echo "  Downloading Qwen3-VL-8B-Instruct..."
echo "  This may take 10-15 minutes depending on network speed."
echo ""

python3 << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-VL-8B-Instruct-FP8",
    local_dir="./models/Qwen3-VL-8B-Instruct-FP8"
)
EOF

if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR)" ]; then
    echo ""
    echo "  ✅ Download complete!"
    echo "     Location: $MODEL_DIR"
    echo ""
    echo "  Start the demo with: ./start.sh"
    echo ""
else
    echo ""
    echo "  ❌ Download failed!"
    exit 1
fi
