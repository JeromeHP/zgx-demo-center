#!/bin/bash
# =============================================================================
# Container Entrypoint
# 1. Start vLLM server in background (serves Qwen3-VL on internal port 8090)
# 2. Wait for vLLM to be ready
# 3. Start FastAPI application on port 8000
# =============================================================================

set -e

MODEL_PATH="${MODEL_PATH:-/models/Qwen3-VL-8B-Instruct-FP8}"
VLLM_PORT="${VLLM_PORT:-8090}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"

echo ""
echo "============================================================"
echo "  Maritime Surveillance Intelligence Generator"
echo "  Inference backend: ${INFERENCE_BACKEND:-vllm}"
echo "============================================================"
echo ""

# ── External inference backend (AMD / any non-CUDA box) ─────────────────────
#
# vLLM in this container is NVIDIA-only twice over: the image is
# nvcr.io/nvidia/vllm, and the shipped checkpoint is FP8, for which gfx1151 has
# no path. But backend/main.py only ever speaks OpenAI-compatible
# /chat/completions to VLLM_BASE_URL, so any server with that API and a vision
# model will do -- which is exactly what host Ollama already provides for the
# ballistics demo (qwen2.5vl:7b).
#
# With INFERENCE_BACKEND=external this container starts no vLLM and needs no
# GPU at all: it becomes a thin FastAPI frontend over host Ollama, the same
# shape the pharmacy, pii and ballistics demos already use.
if [ "${INFERENCE_BACKEND:-vllm}" = "external" ]; then
    echo "Using an external OpenAI-compatible endpoint; not starting vLLM."
    echo "  VLLM_BASE_URL: ${VLLM_BASE_URL}"
    echo "  VLLM_MODEL:    ${VLLM_MODEL}"
    echo ""
    echo "Checking the endpoint is reachable..."
    for i in $(seq 1 30); do
        if curl -sf "${VLLM_BASE_URL}/models" >/dev/null 2>&1; then
            echo "  endpoint OK"
            break
        fi
        [ "$i" = "30" ] && echo "  WARNING: ${VLLM_BASE_URL} did not answer after 30s — starting anyway; the UI will report errors per request."
        sleep 1
    done
    echo ""
    echo "Starting Maritime Surveillance API on port 8000..."
    if [ -n "${HOST_IP:-}" ]; then
        echo "  ➜  Demo:   http://${HOST_IP}:8000"
        echo "  ➜  Health: http://${HOST_IP}:8000/api/health"
    fi
    echo ""
    exec python3 /app/backend/main.py
fi

# ── Start vLLM server in background ─────────────────────────────────────────

echo "Starting vLLM server on port ${VLLM_PORT}..."
echo "  Model: ${MODEL_PATH}"
echo "  Max context: ${VLLM_MAX_MODEL_LEN}"
echo ""

python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --limit-mm-per-prompt '{"image": 1}' \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    2>&1 | sed 's/^/[vLLM] /' &

VLLM_PID=$!

# ── Wait for vLLM to be ready ───────────────────────────────────────────────

echo "Waiting for vLLM to load model (this may take 2-3 minutes)..."

MAX_WAIT=300
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -sf http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo ""
        echo "✅ vLLM server ready!"
        echo ""
        break
    fi

    # Check if vLLM process died
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "❌ vLLM process died unexpectedly"
        exit 1
    fi

    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  ... waiting (${ELAPSED}s / ${MAX_WAIT}s)"
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "❌ vLLM failed to start within ${MAX_WAIT}s"
    exit 1
fi

# ── Start FastAPI application ────────────────────────────────────────────────

HOST_IP="${HOST_IP:-}"
APP_PORT=8000

echo "Starting Maritime Surveillance API on port ${APP_PORT}..."
if [ -n "$HOST_IP" ]; then
    echo ""
    echo "  ➜  Demo:   http://${HOST_IP}:${APP_PORT}"
    echo "  ➜  Health: http://${HOST_IP}:${APP_PORT}/api/health"
fi
echo ""

exec python3 /app/backend/main.py
