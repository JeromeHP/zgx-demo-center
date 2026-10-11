#!/bin/bash
# =============================================================================
# Construction Safety Assistant - Model Download
# Downloads ConstructionSafetyQA-1.2B-V1 (~2.4GB) from public HuggingFace.
# Credential-free: yasserrmd/ConstructionSafetyQA-1.2B-V1 is public + ungated,
# so a fresh CWC Nano needs no AWS creds and no HF token.
# =============================================================================

set -e

MODEL_DIR="backend/model/ConstructionSafetyQA-1.2B-V1"

echo ""
echo "============================================================"
echo "  Construction Safety Model Download"
echo "============================================================"
echo ""
echo "  Model: yasserrmd/ConstructionSafetyQA-1.2B-V1 (~2.4GB)"
echo ""

# Check if already downloaded — must verify the WEIGHTS exist, not just a
# non-empty dir. A stalled fetch can leave config/tokenizer behind with no
# model.safetensors, and a dir-only check would falsely skip re-download.
if [ -f "$MODEL_DIR/model.safetensors" ]; then
    echo "  ✅ Model already exists at $MODEL_DIR"
    echo "     To re-download, delete the directory first."
    echo ""
    exit 0
fi

mkdir -p "$MODEL_DIR"

# Create a temporary venv if not already in one
if [ -z "$VIRTUAL_ENV" ]; then
    echo "  Creating temporary Python environment..."
    python3 -m venv /tmp/hf-download-env
    source /tmp/hf-download-env/bin/activate
    pip install -q huggingface_hub
fi

echo "  Downloading ConstructionSafetyQA-1.2B-V1 from public HuggingFace..."
echo "  This may take a few minutes depending on network speed."
echo ""

python3 << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="yasserrmd/ConstructionSafetyQA-1.2B-V1",
    local_dir="backend/model/ConstructionSafetyQA-1.2B-V1",
)
EOF

if [ -d "$MODEL_DIR" ] && [ "$(ls -A "$MODEL_DIR")" ]; then
    echo ""
    echo "  ✅ Download complete!"
    echo "     Location: $MODEL_DIR"
    echo ""
else
    echo ""
    echo "  ❌ Download failed!"
    exit 1
fi
