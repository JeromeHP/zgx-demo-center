#!/bin/bash
# =============================================================================
# Maritime Surveillance Intelligence Generator - Start Script
# Detects host IP and launches the Docker container
# =============================================================================

set -e

# ── Which stack? Decided FIRST, because the pre-flight checks differ ────────
#
# The default stack runs vLLM in-container from nvcr.io/nvidia/vllm against an
# FP8 checkpoint -- NVIDIA-only twice over. On ROCm or a CPU-only box the AMD
# overlay swaps in a slim multi-arch image that talks to host Ollama, so this
# demo becomes a thin frontend like pharmacy/pii/ballistics and needs no GPU,
# and no FP8 checkpoint, inside the container at all.
#
# This block used to sit AFTER the model gate below. That was the bug: the
# gate is an NVIDIA-stack requirement, but it ran unconditionally, and the AMD
# install plan deliberately never downloads that checkpoint (install_plans.py
# says so). So on every AMD box the gate tripped, `set -e` exited 1, compose
# was never invoked, no container was ever created -- and /demo_log reported
# "(no container output — the container may never have been created)".
DETECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/zgx_hardware.py"
PROFILE="${ZGX_HW_PROFILE:-}"
if [ -z "$PROFILE" ] && [ -f "$DETECT" ]; then
    PROFILE="$(python3 -c "
import sys; sys.path.insert(0, '$(dirname "$DETECT")')
import zgx_hardware; print(zgx_hardware.detect()['profile'])" 2>/dev/null || echo "")"
fi

if [ -n "${MARITIME_COMPOSE:-}" ]; then
    COMPOSE_FILES="$MARITIME_COMPOSE"
    STACK="override"
elif [ "$PROFILE" = "nvidia-cuda" ]; then
    COMPOSE_FILES="-f docker-compose.yml"
    STACK="cuda"
elif [ "$PROFILE" = "amd-rocm" ] || [ "$PROFILE" = "cpu-only" ]; then
    # Matches install_plans.py's profile keys exactly. The old predicate here
    # was "anything that is not nvidia-cuda", which disagreed with the plan's
    # amd-rocm-only override: on a cpu-only box the installer built the NGC
    # image while this script ran the overlay.
    COMPOSE_FILES="-f docker-compose.yml -f docker-compose.amd.yml"
    STACK="external"
else
    # An unknown or undetectable profile used to fall through to the CUDA
    # stack in silence, which on an AMD box fails with an unrelated-looking
    # "could not select device driver". Say so instead of guessing.
    echo ""
    echo "❌ Could not determine this machine's accelerator profile."
    echo "   Detector: $DETECT"
    echo "   Set one explicitly, e.g.:  ZGX_HW_PROFILE=amd-rocm ./start.sh -d"
    echo "   (valid: nvidia-cuda, amd-rocm, cpu-only)"
    echo ""
    exit 1
fi

# ── Pre-flight checks ───────────────────────────────────────────────────────

# Preserve the previous attempt's logs before removing its container.
#
# This `docker rm -f` used to be the first thing the script did. With
# restart: "no", a failed container survives as Exited holding the only copy
# of the error -- and the next launch deleted it, turning a diagnosable
# failure into an empty log. Keep a copy first.
if docker ps -a --format '{{.Names}}' | grep -q "^maritime-surveillance-demo$"; then
    LOG_DIR="${LOG_DIR:-$HOME/demos/logs}"
    mkdir -p "$LOG_DIR" 2>/dev/null || true
    if docker logs maritime-surveillance-demo > "$LOG_DIR/maritime_previous.log" 2>&1; then
        echo "Saved the previous container's log to $LOG_DIR/maritime_previous.log"
    fi
    echo "Removing existing container..."
    docker rm -f maritime-surveillance-demo
fi

# The FP8 checkpoint is a requirement of the CUDA stack ONLY.
if [ "$STACK" = "cuda" ]; then
    MODEL_DIR="./models/Qwen3-VL-8B-Instruct-FP8"
    if [ ! -d "$MODEL_DIR" ]; then
        echo ""
        echo "❌ Model not found: $MODEL_DIR"
        echo ""
        echo "   Download the model first:"
        echo "     ./download_models.sh"
        echo ""
        exit 1
    fi
fi

# Check Docker is running
if ! docker info &>/dev/null; then
    echo ""
    echo "❌ Docker daemon is not running."
    echo "   Start it with: sudo systemctl start docker"
    echo ""
    exit 1
fi

# ── Detect host LAN IP ──────────────────────────────────────────────────────

if [ -z "$HOST_IP" ]; then
    HOST_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}')
fi
if [ -z "$HOST_IP" ]; then
    HOST_IP=$(hostname -I | awk '{print $1}')
fi

export HOST_IP

echo ""
echo "============================================================"
echo "  Maritime Surveillance Intelligence Generator"
echo "  Qwen3-VL-8B-Instruct | HP ZGX Nano | vLLM"
echo "============================================================"
echo "  Host IP: $HOST_IP"
echo ""
echo "  Click here for Demo 👉:   http://$HOST_IP:8000"
echo "  ➜  Health:  http://$HOST_IP:8000/api/health"
echo ""
echo "  Note: First startup takes 2-3 minutes for model loading."
echo "============================================================"
echo ""

echo "  Accelerator profile: ${PROFILE:-overridden} — $STACK stack"
if [ "$STACK" = "external" ]; then
    echo "  (host Ollama; no vLLM, no GPU needed inside the container)"
fi
echo ""

# Pass all arguments through (e.g., --build, -d)
docker compose $COMPOSE_FILES up "$@"
