#!/usr/bin/env bash
# Phase 2.1: download the Qwen3.6-35B-A3B GGUF. Idempotent: skips if the file
# already exists with the expected size.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models/qwen3.6-35b-a3b}"
MODEL_FILE="Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
EXPECTED_SIZE=22360456160
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

if [ -f "$MODEL_PATH" ]; then
  ACTUAL_SIZE="$(stat -c %s "$MODEL_PATH")"
  if [ "$ACTUAL_SIZE" -eq "$EXPECTED_SIZE" ]; then
    echo "Model already present and correct size ($ACTUAL_SIZE bytes): $MODEL_PATH"
    exit 0
  else
    echo "Model file exists but size mismatch (got $ACTUAL_SIZE, want $EXPECTED_SIZE); re-downloading."
  fi
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "Installing huggingface_hub[cli] (no sudo, user-space)..."
  uv tool install "huggingface_hub[cli]"
fi

mkdir -p "$MODEL_DIR"
hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  "$MODEL_FILE" \
  --local-dir "$MODEL_DIR"

ACTUAL_SIZE="$(stat -c %s "$MODEL_PATH")"
echo "Downloaded size: $ACTUAL_SIZE bytes (expected $EXPECTED_SIZE)"
if [ "$ACTUAL_SIZE" -ne "$EXPECTED_SIZE" ]; then
  echo "ERROR: size mismatch after download" >&2
  exit 1
fi
echo "OK: $MODEL_PATH"
