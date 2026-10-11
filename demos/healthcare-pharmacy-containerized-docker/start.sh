#!/bin/bash
# =============================================================================
# Medical AI Drug Interaction Demo - Start Script
# Detects host IP and launches the Docker container
# =============================================================================

set -e

# ── Pre-flight checks ───────────────────────────────────────────────────────

if docker ps -a --format '{{.Names}}' | grep -q "^healthcare-drug-interaction-demo$"; then
    echo "Removing existing container..."
    docker rm -f healthcare-drug-interaction-demo
fi

# Check that model exists
MODEL_FILE="./models/medical-ft-mixtral-q4"
if [ ! -f "$MODEL_FILE" ]; then
    echo ""
    echo "❌ Model not found: $MODEL_FILE"
    echo ""
    echo "   Download the model first:"
    echo "     Option A (S3):  ./download_models.sh"
    echo "     Option B:       Place medical-ft-mixtral-q4 in ./models/"
    echo ""
    exit 1
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
echo "=============================================="
echo "  Medical AI Drug Interaction Demo"
echo "=============================================="
echo "  Host IP: $HOST_IP"
echo ""
echo "  Click here for Demo 👉:   http://$HOST_IP:8000"
echo "  ➜  Health:  http://$HOST_IP:8000/api/health"
echo "=============================================="
echo ""

# Pass all arguments through (e.g., --build, -d)
docker compose up "$@"
