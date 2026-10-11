#!/bin/bash
# =============================================================================
# Port-of-Entry Vehicle Inspection Analyst - Start Script
# Detects host IP and launches the Docker container on :8097.
#
# Non-interactive by design: the launcher runs this through an SSE stream, so
# a prompt would hang the launch. (Upstream asked "Continue anyway? [y/N]" when
# :8091 was busy -- on this box that is the Construction demo, not a vLLM.)
# =============================================================================

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

PORT=8097
NAME=vehicle-inspection-demo

# ── Pre-flight checks ───────────────────────────────────────────────────────

# Preserve the previous attempt's logs before removing its container: with
# restart: "no", a failed container holds the only copy of the error.
if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
    LOG_DIR="${LOG_DIR:-$HOME/demos/logs}"
    mkdir -p "$LOG_DIR" 2>/dev/null || true
    if docker logs "$NAME" > "$LOG_DIR/vehicle_previous.log" 2>&1; then
        echo "Saved the previous container's log to $LOG_DIR/vehicle_previous.log"
    fi
    echo "Removing existing container..."
    docker rm -f "$NAME"
fi

# Check Docker is running
if ! docker info &>/dev/null; then
    echo ""
    echo "❌ Docker daemon is not running."
    echo "   Start it with: sudo systemctl start docker"
    echo ""
    exit 1
fi

# Something other than our (just-removed) container on 8097 means the port map
# has been broken somewhere. Say which process, rather than fail inside uvicorn.
if ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
    echo ""
    echo "❌ Port ${PORT} is already in use by another process."
    echo "   8097 is reserved for this demo (RUNBOOK.md port map)."
    echo "   Find the owner with: sudo ss -ltnp | grep :${PORT}"
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
echo "  Port-of-Entry Vehicle Inspection Analyst"
echo "  mistral-small3.2 | Ollama"
echo "============================================================"
echo "  Host IP: $HOST_IP"
echo ""
echo "  Click here for Demo 👉:   http://$HOST_IP:${PORT}"
echo "  ➜  Health:  http://$HOST_IP:${PORT}/api/health"
echo ""
echo "  Note: mistral-small3.2 must already be pulled (ollama pull mistral-small3.2)."
echo "============================================================"
echo ""

# Pass all arguments through (e.g., --build, -d)
docker compose up "$@"
