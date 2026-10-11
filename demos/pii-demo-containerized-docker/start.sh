#!/bin/bash
# =============================================================================
# PII Masking Demo - Start Script
# Detects host IP and launches the Docker container
# =============================================================================

# Auto-detect host LAN IP (skip loopback and docker interfaces)
if [ -z "$HOST_IP" ]; then
    HOST_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}')
fi

# Fallback if detection failed
if [ -z "$HOST_IP" ]; then
    HOST_IP=$(hostname -I | awk '{print $1}')
fi

export HOST_IP

echo ""
echo "Detected host IP: $HOST_IP"
echo ""

# Pass all arguments through (e.g., --build, -d)
docker compose up "$@"