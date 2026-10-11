#!/usr/bin/env bash
# Download the fine-tuned PII detector GGUF into this demo's models/ directory.
# The install step then registers it with Ollama as 'pii-detector'. The
# "base" comparison model (qwen2.5:32b) is a stock `ollama pull`, not part
# of this download — see install_plans.py.
#
# Self-locating (no hardcoded home-dir assumption). Delegates to _dl_models.py — the
# single source of truth for which GGUF to fetch — so this script and the
# orchestrator's install step download exactly the same file.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Prefer the zgx env python (has huggingface_hub); fall back to PATH python3.
PY="$HOME/miniforge3/envs/zgx/bin/python3"
[ -x "$PY" ] || PY="python3"

echo "================================================"
echo "PII Models Download Script"
echo "================================================"
echo "Fine-tuned PII Detector (~20GB)..."

"$PY" "$SCRIPT_DIR/_dl_models.py"

echo ""
echo "✅ Model downloaded to $SCRIPT_DIR/models/"
echo "   The installer will register it with Ollama (pii-detector) and separately pull qwen2.5:32b for the base side."
