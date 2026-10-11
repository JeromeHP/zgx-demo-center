#!/bin/bash
# ──────────────────────────────────────────────────────────
# Consent Agent - Full Demo Launcher with Preflight Checks
# ──────────────────────────────────────────────────────────
#
# This script launches the full medical triage demo:
#   - AI API (Docker container on ZGX Nano)
#   - Live transcript dashboard (web server)
#   - Reachy Mini robot app (voice agent)
#
# It automatically detects the robot on the network and
# configures it to talk to this machine. No manual IP
# configuration needed.
#
# Prerequisites:
#   - Docker with NVIDIA GPU support
#   - AI models in ../models/
#   - Reachy Mini powered on and on the same WiFi network
#   - SSH key copied to robot: ssh-copy-id pollen@<robot_ip>
#
# ──────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
ROBOT_IP=""
ROBOT_USER="pollen"
API_PORT=8090
DASHBOARD_PORT=8080
APP_NAME="consent_agent_reachy"
MAIN_PY_PATH="/venvs/apps_venv/lib/python3.12/site-packages/consent_agent_reachy/main.py"

# Auto-detect this machine's IP (prefer route-based, fallback to hostname)
ZGX_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}')
if [ -z "$ZGX_IP" ]; then
    ZGX_IP=$(hostname -I | awk '{print $1}')
fi
if [ -z "$ZGX_IP" ]; then
    echo "  ❌ Could not detect this machine's IP address."
    echo "     This machine does not appear to be connected to a network."
    echo "     Connect to WiFi or Ethernet and try again."
    exit 1
fi

# ── Find the robot ──
find_robot() {
    # Try reachy-mini.local first (mDNS)
    if ping -c 1 -W 2 reachy-mini.local > /dev/null 2>&1; then
        ROBOT_IP=$(getent hosts reachy-mini.local | awk '{print $1}')
        if [ -z "$ROBOT_IP" ]; then
            ROBOT_IP="reachy-mini.local"
        fi
        return 0
    fi

    # mDNS didn't work - scan the local subnet for the Reachy daemon (port 8000)
    echo "         (mDNS unavailable, scanning subnet...)"
    local subnet
    subnet=$(echo "$ZGX_IP" | cut -d. -f1-3)
    for i in $(seq 1 254); do
        local candidate="${subnet}.${i}"
        [ "$candidate" = "$ZGX_IP" ] && continue
        if curl -sf --connect-timeout 0.3 "http://${candidate}:8000/api/daemon/status" > /dev/null 2>&1; then
            ROBOT_IP="$candidate"
            return 0
        fi
    done
    return 1
}

echo ""
echo "══════════════════════════════════════════════════════"
echo "  🩺 Medical Triage Demo"
echo "══════════════════════════════════════════════════════"
echo ""
echo "  ZGX Nano IP: $ZGX_IP"
echo ""

# ══════════════════════════════════════════════════════════
# PREFLIGHT CHECKS
# ══════════════════════════════════════════════════════════

echo "  [0/7] Preflight checks..."
PREFLIGHT_OK=true

# Check 1: Docker available
if ! docker info > /dev/null 2>&1; then
    echo "         ❌ Docker is not running."
    echo "            Start Docker and try again."
    PREFLIGHT_OK=false
else
    echo "         ✅ Docker running"
fi

# Check 2: Docker image exists
if ! docker image inspect consent-agent:latest > /dev/null 2>&1; then
    echo "         ❌ Docker image 'consent-agent:latest' not found."
    echo "            You need to build the AI API container first:"
    echo "              cd \$HOME/demos/Robotic-medical-AI-assistant/docker"
    echo "              docker build -t consent-agent:latest -f Dockerfile.api ."
    echo "            This only needs to be done once (takes ~5 min)."
    PREFLIGHT_OK=false
else
    echo "         ✅ Docker image exists"
fi

# Check 3: Models directory
if [ ! -d "$PROJECT_DIR/models" ]; then
    echo "         ❌ Models directory not found: $PROJECT_DIR/models"
    echo "            The AI models (Whisper, LLM, TTS) need to be downloaded."
    echo "            See README.md for model download instructions."
    PREFLIGHT_OK=false
else
    echo "         ✅ Models directory exists"
fi

# Check 4: Dashboard HTML
if [ ! -f "$PROJECT_DIR/index.html" ]; then
    echo "         ❌ Dashboard file not found: $PROJECT_DIR/index.html"
    echo "            The index.html file should be in the project root."
    PREFLIGHT_OK=false
else
    echo "         ✅ Dashboard HTML exists"
fi

# Check 5: Find robot
echo "         🔍 Looking for Reachy Mini on the network..."
if find_robot; then
    echo "         ✅ Robot found at $ROBOT_IP"
else
    echo "         ❌ Cannot find Reachy Mini on the network."
    echo ""
    echo "            The Reachy Mini robot was not detected. Check the following:"
    echo "              1. Is the robot powered on? (green LED on the base)"
    echo "              2. Is it connected to the same WiFi as this machine?"
    echo "                 - Open the Reachy dashboard at http://reachy-mini.local"
    echo "                   from a browser to check WiFi status."
    echo "                 - If that URL doesn't load, the robot may not be on"
    echo "                   the network yet. It takes ~60 seconds after power-on."
    echo "              3. Some networks block device-to-device communication"
    echo "                 (hotel WiFi, corporate networks with client isolation)."
    echo "                 Try a mobile hotspot or home network instead."
    echo ""
    echo "            If you know the robot's IP, you can set it manually:"
    echo "              export ROBOT_IP=<ip_address>"
    echo "              ./start_services.sh"
    PREFLIGHT_OK=false
fi

# Check 6: Robot SSH
if [ -n "$ROBOT_IP" ]; then
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3 -o BatchMode=yes ${ROBOT_USER}@${ROBOT_IP} "echo ok" > /dev/null 2>&1; then
        echo "         ✅ SSH to robot (passwordless)"
    else
        echo "         ⚠️  Cannot SSH to robot without a password."
        echo ""
        echo "            This script needs passwordless SSH to configure the robot."
        echo "            Run this once to set it up (password is 'root'):"
        echo "              ssh-copy-id ${ROBOT_USER}@${ROBOT_IP}"
        echo ""
        echo "            The Reachy Mini runs Linux internally. SSH lets this script"
        echo "            update the robot's configuration automatically when you"
        echo "            connect to a new network."
        PREFLIGHT_OK=false
    fi
fi

# Check 7: App installed on robot
if [ -n "$ROBOT_IP" ]; then
    APP_FOUND=$(curl -sf "http://${ROBOT_IP}:8000/api/apps/list-available" 2>/dev/null | python3 -c "import sys,json; apps=json.load(sys.stdin); print('yes' if any(a.get('name')=='consent_agent_reachy' for a in apps) else 'no')" 2>/dev/null || echo "no")
    if [ "$APP_FOUND" = "yes" ]; then
        echo "         ✅ App '${APP_NAME}' installed on robot"
    else
        echo "         ⚠️  App '${APP_NAME}' not found on robot. Installing..."
        echo ""
        echo "            The Reachy Mini runs apps from a specific directory."
        echo "            Copying the voice agent app to the robot now..."
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${ROBOT_USER}@${ROBOT_IP} \
            "mkdir -p /venvs/apps_venv/lib/python3.12/site-packages/consent_agent_reachy/" 2>/dev/null || true
        scp -o StrictHostKeyChecking=no "$PROJECT_DIR/hf-space/consent_agent_reachy/main.py" \
            ${ROBOT_USER}@${ROBOT_IP}:${MAIN_PY_PATH} 2>/dev/null || true
        touch /tmp/__init__.py
        scp -o StrictHostKeyChecking=no /tmp/__init__.py \
            ${ROBOT_USER}@${ROBOT_IP}:/venvs/apps_venv/lib/python3.12/site-packages/consent_agent_reachy/__init__.py 2>/dev/null || true

        # Verify it worked
        APP_FOUND=$(curl -sf "http://${ROBOT_IP}:8000/api/apps/list-available" 2>/dev/null | python3 -c "import sys,json; apps=json.load(sys.stdin); print('yes' if any(a.get('name')=='consent_agent_reachy' for a in apps) else 'no')" 2>/dev/null || echo "no")
        if [ "$APP_FOUND" = "yes" ]; then
            echo "         ✅ App installed successfully"
        else
            echo "         ❌ App installation failed."
            echo ""
            echo "            The robot's app directory may have different permissions"
            echo "            or the daemon may need a restart to detect new apps."
            echo "            Try: ssh ${ROBOT_USER}@${ROBOT_IP} 'sudo systemctl restart reachy-mini-daemon'"
            echo "            Then re-run this script."
            PREFLIGHT_OK=false
        fi
    fi
fi

if [ "$PREFLIGHT_OK" = false ]; then
    echo ""
    echo "  ❌ Preflight checks failed. Fix the issues above and re-run."
    exit 1
fi

echo ""

# ══════════════════════════════════════════════════════════
# Skip container restart if already running and healthy
if curl -sf http://localhost:8090/health | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('whisper') and d.get('llm') and d.get('tts') else 1)" 2>/dev/null; then
    echo "  [1-3] ✅ Container already running and healthy, skipping restart"
    echo ""
else
# STEP 1: Clean up
# ══════════════════════════════════════════════════════════

echo "  [1/7] Cleaning up old processes..."
lsof -ti:${DASHBOARD_PORT} 2>/dev/null | xargs kill -9 2>/dev/null || true
docker stop zgx-ai-api 2>/dev/null || true
docker rm zgx-ai-api 2>/dev/null || true
curl -sf -X POST "http://${ROBOT_IP}:8000/api/apps/stop-current-app" > /dev/null 2>&1 || true
sleep 2
echo "         ✅ Clean"
echo ""

# ══════════════════════════════════════════════════════════
# STEP 2: Start Docker API container
# ══════════════════════════════════════════════════════════

echo "  [2/7] Starting AI API container..."
docker run --rm -d --name zgx-ai-api --gpus all --network host \
    -v "$PROJECT_DIR/models:/models" \
    consent-agent:latest > /dev/null 2>&1
echo "         ✅ Container started"
echo ""

# ══════════════════════════════════════════════════════════
# STEP 3: Wait for API
# ══════════════════════════════════════════════════════════

echo "  [3/7] Waiting for models to load (this takes ~90 seconds)..."
SECONDS=0
while true; do
    if curl -sf http://localhost:${API_PORT}/health > /dev/null 2>&1; then
        HEALTH=$(curl -sf http://localhost:${API_PORT}/health)
        WHISPER_OK=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('whisper',False))" 2>/dev/null)
        LLM_OK=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('llm',False))" 2>/dev/null)
        TTS_OK=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tts',False))" 2>/dev/null)
        WHISPER_MODEL=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('whisper_model','?'))" 2>/dev/null)

        if [ "$LLM_OK" = "True" ] && [ "$WHISPER_OK" = "True" ] && [ "$TTS_OK" = "True" ]; then
            echo ""
            echo "         ✅ All systems ready (${SECONDS}s)"
            echo "            Whisper: ✓ ${STT_MODEL}"
            echo "            LLM:     ✓ online"
            echo "            TTS:     ✓ online"
            break
        else
            printf "\r         ⏳ API up but waiting for components... (%ds)" $SECONDS
        fi
    else
        printf "\r         ⏳ Loading models... (%ds)                        " $SECONDS
    fi

    if [ $SECONDS -gt 300 ]; then
        echo ""
        echo "         ❌ Timed out after 5 minutes."
        echo ""
        echo "            The AI API container started but the models did not load."
        echo "            This usually means the GPU ran out of memory."
        echo ""
        echo "            Debug steps:"
        echo "              1. Check container logs: docker logs zgx-ai-api -f"
        echo "              2. Check GPU memory: nvidia-smi"
        echo "              3. Make sure no other GPU processes are running"
        exit 1
    fi

    sleep 3
done
echo ""

fi
# ══════════════════════════════════════════════════════════
# STEP 4: Start dashboard
# ══════════════════════════════════════════════════════════

echo "  [4/7] Starting dashboard server..."
cd "$PROJECT_DIR"
python3 -m http.server ${DASHBOARD_PORT} > /dev/null 2>&1 &
DASHBOARD_PID=$!
sleep 1

if curl -sf -o /dev/null http://localhost:${DASHBOARD_PORT}/index.html; then
    echo "         ✅ Dashboard serving on port ${DASHBOARD_PORT}"
else
    echo "         ❌ Dashboard failed to start."
    echo "            Check that index.html exists in: $PROJECT_DIR"
    echo "            Also check if port ${DASHBOARD_PORT} is already in use:"
    echo "              lsof -i:${DASHBOARD_PORT}"
    exit 1
fi
echo ""

# ══════════════════════════════════════════════════════════
# STEP 5: Patch robot API URL
# ══════════════════════════════════════════════════════════

echo "  [5/7] Configuring robot to talk to this machine..."
echo "         Target API: http://${ZGX_IP}:${API_PORT}"

# The Reachy Mini runs our Python app, which needs the API URL.
# The GitHub version has API_URL = None (no fallback).
# The local version may have a previously patched IP.
# Either way, we replace the entire API_URL line with the correct IP.

CURRENT_LINE=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${ROBOT_USER}@${ROBOT_IP} \
    "grep 'API_URL' ${MAIN_PY_PATH} 2>/dev/null | head -1" 2>/dev/null || echo "unknown")
echo "         Robot currently has: $CURRENT_LINE"

EXPECTED_LINE="API_URL = \"http://${ZGX_IP}:${API_PORT}\""

if echo "$CURRENT_LINE" | grep -q "http://${ZGX_IP}:${API_PORT}"; then
    echo "         ✅ Already correct, no change needed"
else
    echo "         🔧 Updating robot to point to http://${ZGX_IP}:${API_PORT}..."
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${ROBOT_USER}@${ROBOT_IP} \
        "sed -i '/^API_URL\s*=/c\\API_URL = \"http://${ZGX_IP}:${API_PORT}\"' ${MAIN_PY_PATH} && \
         rm -rf /venvs/apps_venv/lib/python3.12/site-packages/consent_agent_reachy/__pycache__" 2>/dev/null

    # Verify the change took effect
    NEW_LINE=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${ROBOT_USER}@${ROBOT_IP} \
        "grep 'API_URL' ${MAIN_PY_PATH} 2>/dev/null | head -1" 2>/dev/null || echo "unknown")

    if echo "$NEW_LINE" | grep -q "http://${ZGX_IP}:${API_PORT}"; then
        echo "         ✅ Robot updated to: http://${ZGX_IP}:${API_PORT}"
    else
        echo "         ❌ FAILED to update the robot's API URL."
        echo ""
        echo "            This is the #1 cause of 'robot not responding' issues."
        echo "            The robot needs to know this machine's IP address to send"
        echo "            audio for processing. When you change networks, the IP changes."
        echo ""
        echo "            What the robot has: $NEW_LINE"
        echo "            What it should be:  API_URL = \"http://${ZGX_IP}:${API_PORT}\""
        echo ""
        echo "            Manual fix (run this on the ZGX):"
        echo "              ssh ${ROBOT_USER}@${ROBOT_IP} \"sed -i '/^API_URL/c\\\\API_URL = \\\"http://${ZGX_IP}:${API_PORT}\\\"' ${MAIN_PY_PATH}\""
        exit 1
    fi
fi
echo ""

# ══════════════════════════════════════════════════════════
# STEP 6: Verify robot can reach API
# ══════════════════════════════════════════════════════════

echo "  [6/7] Verifying robot can reach the AI API..."
ROBOT_HEALTH=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${ROBOT_USER}@${ROBOT_IP} \
    "curl -sf http://${ZGX_IP}:${API_PORT}/health 2>/dev/null" 2>/dev/null || echo "FAILED")

if echo "$ROBOT_HEALTH" | grep -q "whisper"; then
    echo "         ✅ Robot can reach API at http://${ZGX_IP}:${API_PORT}"
else
    echo "         ❌ The robot CANNOT reach the AI API on this machine."
    echo ""
    echo "            The robot (${ROBOT_IP}) tried to connect to the API"
    echo "            at http://${ZGX_IP}:${API_PORT} and failed."
    echo ""
    echo "            This means the two devices can't talk to each other,"
    echo "            even though they're both on the network."
    echo ""
    echo "            Common causes:"
    echo "              - The WiFi network has 'client isolation' enabled,"
    echo "                which blocks device-to-device traffic. This is"
    echo "                common on hotel, conference, and corporate WiFi."
    echo "                Fix: use a mobile hotspot or home router instead."
    echo "              - A firewall on this machine is blocking port ${API_PORT}."
    echo "                Fix: sudo ufw allow ${API_PORT}/tcp"
    echo "              - The devices are on different subnets."
    echo "                ZGX: ${ZGX_IP}  Robot: ${ROBOT_IP}"
    echo "                The first three octets should match."
    exit 1
fi
echo ""

# ══════════════════════════════════════════════════════════
# STEP 7: Start Reachy app
# ══════════════════════════════════════════════════════════

echo "  [7/7] Starting Reachy Mini app..."

# The Reachy Mini's motors need to be 'primed' - we start the app
# briefly, stop it, then start again. This ensures the daemon has
# handed motor control to the app process. Without this, the robot's
# head and antennas won't move on first boot.
echo "         Priming robot motors (start/stop cycle)..."
curl -sf -X POST "http://${ROBOT_IP}:8000/api/apps/start-app/${APP_NAME}" > /dev/null 2>&1 || true
sleep 5
curl -sf -X POST "http://${ROBOT_IP}:8000/api/apps/stop-current-app" > /dev/null 2>&1 || true
sleep 3

# Real launch
echo "         Starting app (this is the real launch)..."
curl -sf -X POST "http://${ROBOT_IP}:8000/api/apps/start-app/${APP_NAME}" > /dev/null 2>&1 || true
sleep 10

# Verify app is still running (not crashed)
APP_STATUS=$(curl -sf "http://${ROBOT_IP}:8000/api/apps/current-app-status" 2>/dev/null || echo "")
if echo "$APP_STATUS" | grep -q "running"; then
    echo "         ✅ App is running"
elif echo "$APP_STATUS" | grep -q "starting"; then
    echo "         ✅ App is starting (connecting to API)"
else
    echo "         ❌ The app started but immediately crashed."
    echo ""
    echo "            The Reachy Mini runs our voice agent as a Python app"
    echo "            managed by its internal daemon. The app crashed within"
    echo "            10 seconds of starting."
    echo ""
    echo "            Check the robot's logs for the actual error:"
    echo "              ssh ${ROBOT_USER}@${ROBOT_IP} \"sudo journalctl -u reachy-mini-daemon --since '30 sec ago'\" | grep -i \"consent\\|error\\|ERROR\\|Traceback\""
    echo ""
    echo "            Common causes:"
    echo "              - Wrong API URL in main.py. Step 5 should have fixed"
    echo "                this, but verify with:"
    echo "                  ssh ${ROBOT_USER}@${ROBOT_IP} \"grep 'API_URL\\|http://' ${MAIN_PY_PATH} | head -5\""
    echo "              - The ALSA audio device name changed. The robot's mic"
    echo "                is accessed via ALSA device 'reachymini_audio_src'."
    echo "                Check if it exists:"
    echo "                  ssh ${ROBOT_USER}@${ROBOT_IP} \"arecord -L | grep reachy\""
    echo "              - A Python import error. The app needs numpy, requests,"
    echo "                and scipy installed in the robot's venv."
    echo "                  ssh ${ROBOT_USER}@${ROBOT_IP} \"/venvs/apps_venv/bin/pip list | grep -i 'numpy\\|requests\\|scipy'\""
fi
echo ""

# ══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════

echo "══════════════════════════════════════════════════════"
echo ""
echo "  ✅ DEMO READY - Open the dashboard:"
echo ""
echo "     👉 http://${ZGX_IP}:${DASHBOARD_PORT}/index.html"
echo ""
echo "  Configuration:"
echo "     ZGX Nano:  ${ZGX_IP} (API: ${API_PORT}, Dashboard: ${DASHBOARD_PORT})"
echo "     Robot:     ${ROBOT_IP} (using API at http://${ZGX_IP}:${API_PORT})"
echo "     STT:      ${STT_MODEL}"
echo ""
echo "  Speak to the Reachy Mini. The transcript"
echo "  will appear on the dashboard in real time."
echo ""
echo "══════════════════════════════════════════════════════"
echo ""
echo "  Ctrl+C to stop the dashboard server."
echo "  To stop everything:"
echo "    docker stop zgx-ai-api"
echo "    curl -X POST http://${ROBOT_IP}:8000/api/apps/stop-current-app"
echo ""

# ── Cleanup on exit ──
cleanup() {
    echo ""
    echo "  Stopping dashboard server..."
    kill $DASHBOARD_PID 2>/dev/null || true
    echo "  Dashboard stopped."
    echo "  API container and Reachy app still running."
    exit 0
}

trap cleanup INT TERM
wait $DASHBOARD_PID
