#!/usr/bin/env bash
# Download the fine-tuned Mixtral Q4 GGUF into this demo's models/ directory.
# The install step then registers it with Ollama as 'medical-mixtral'.
#
# Self-locating: resolves its own directory, so it works regardless of the
# caller's cwd (no hardcoded home-dir assumption). Idempotent: skips the
# download if the target GGUF is already present.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODELS_DIR="$SCRIPT_DIR/models"
TARGET="$MODELS_DIR/medical-ft-mixtral-q4"   # matches Modelfile.medical `FROM ./models/...`

# Prefer the zgx env python (has huggingface_hub from the repo's requirements.txt);
# fall back to whatever python3 is on PATH.
PY="$HOME/miniforge3/envs/zgx/bin/python3"
[ -x "$PY" ] || PY="python3"

echo "================================================"
echo "Medical AI Model Download Script"
echo "================================================"

if [ -f "$TARGET" ]; then
    echo "✓ medical-ft-mixtral-q4 already present in models/ — skipping download"
    exit 0
fi

echo "Medical Fine-tuned Mixtral Q4 (~28GB) → $MODELS_DIR ..."
mkdir -p "$MODELS_DIR"

"$PY" - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(repo_id="curtburk/healthcare-polypharmacy-finetune", local_dir="$MODELS_DIR")
EOF

echo ""
echo "✅ Models downloaded to $MODELS_DIR/"
echo "   The installer will register it with Ollama as 'medical-mixtral'."
