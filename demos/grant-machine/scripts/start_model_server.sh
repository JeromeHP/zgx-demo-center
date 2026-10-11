#!/usr/bin/env bash
set -euo pipefail
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:?set LLAMA_SERVER_BIN to the llama-server path}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the GGUF file}"
HOST="${LLM_HOST:-127.0.0.1}"
# 8080 to match config/default.yaml's llm.base_url. This defaulted to 8081 for the
# original dev machine, where an unrelated searxng container held 8080 -- a local
# accident that did not belong in a committed default. Override with LLM_PORT.
PORT="${LLM_PORT:-8080}"
CTX="${LLM_CTX:-32768}"
SLOTS="${LLM_SLOTS:-1}"
ALIAS="${LLM_ALIAS:-qwen3.6-35b-a3b}"

exec "$LLAMA_SERVER_BIN" \
  -m "$MODEL_PATH" \
  --alias "$ALIAS" \
  --host "$HOST" --port "$PORT" \
  -c "$CTX" -np "$SLOTS" \
  -ngl all \
  -fa auto \
  --jinja \
  --reasoning-format deepseek \
  --load-mode none \
  --metrics
