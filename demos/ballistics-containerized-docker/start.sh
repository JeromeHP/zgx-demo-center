#!/bin/bash
# =============================================================================
# NAWSCL Aircraft Classification - Start Script
# Detects host IP and launches the Docker container
# =============================================================================

set -e

# ── Pre-flight checks ───────────────────────────────────────────────────────

# Remove existing container if present
if docker ps -a --format '{{.Names}}' | grep -q "^ballistics-classification-demo$"; then
    echo "Removing existing container..."
    docker rm -f ballistics-classification-demo
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
echo "  NAWSCL Aircraft Classification"
echo "  qwen2.5vl:7b | HP ZGX Nano | Ollama"
echo "============================================================"
echo "  Host IP: $HOST_IP"
echo ""
echo "  Click here for Demo 👉:   http://$HOST_IP:8093"
echo "  ➜  Health:  http://$HOST_IP:8093/api/health"
echo ""
echo "  Note: qwen2.5vl:7b must already be pulled (ollama pull qwen2.5vl:7b)."
echo "============================================================"
echo ""

# Pass all arguments through (e.g., --build, -d)
docker compose up "$@"
