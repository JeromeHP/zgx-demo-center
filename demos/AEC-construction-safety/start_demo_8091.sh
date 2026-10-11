#!/bin/bash
# Construction Safety Demo — port 8091 (frontend) + 8191 (backend API)
# Wrapper created to avoid conflict with ZGX Performance Lab on 8080

FRONTEND_PORT=8091
BACKEND_PORT=8191
DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_IP=$(hostname -I | awk '{print $1}')

echo "🦺 Construction Safety Demo (port $FRONTEND_PORT)"

# Kill anything on our ports
fuser -k ${FRONTEND_PORT}/tcp 2>/dev/null
fuser -k ${BACKEND_PORT}/tcp 2>/dev/null
sleep 1

cd "$DEMO_DIR"

# Patch backend port
sed -i "s/port=8000/port=${BACKEND_PORT}/" backend/main.py 2>/dev/null
sed -i "s/port=8080/port=${BACKEND_PORT}/" backend/main.py 2>/dev/null

# Patch frontend API URL — use window.location.hostname so any IP/hostname works
sed -i "s|const API_URL = .*|const API_URL = 'http://' + window.location.hostname + ':${BACKEND_PORT}';|" frontend/index.html 2>/dev/null

# Start backend
cd "$DEMO_DIR/backend"
source ../venv/bin/activate
python3 main.py &
BACKEND_PID=$!
cd "$DEMO_DIR"

# Start frontend
cd "$DEMO_DIR/frontend"
python3 -m http.server $FRONTEND_PORT --bind 0.0.0.0 &
FRONTEND_PID=$!
cd "$DEMO_DIR"

echo "Waiting for services…"
sleep 5
echo "✅ Frontend: http://${SERVER_IP}:${FRONTEND_PORT}"
echo "✅ Backend:  http://${SERVER_IP}:${BACKEND_PORT}"

cleanup() {
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}
trap cleanup INT TERM
wait
