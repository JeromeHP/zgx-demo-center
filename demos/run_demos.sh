#!/bin/bash
# HP ZGX Nano — Demo Launcher v2.0
#   8085  Demo Center HTML (file server)
#   8086  Install API (FastAPI/uvicorn, SSE)
# Separate from the ZGX Performance Lab on port 8080.

DEMO_PORT=8085
API_PORT=8086

# ~/demos must resolve to this checkout before anything else — the developer
# path skips install.sh, so this is the only place the link gets ensured when
# running the services directly. See ensure_demos_link.sh for why the path is
# a fixed convention rather than script-relative.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ensure_demos_link.sh
source "$SCRIPT_DIR/ensure_demos_link.sh"
ensure_demos_link "$SCRIPT_DIR" || exit 1

DEMO_DIR="$HOME/demos"
DEMO_FILE="zgx-demo-launcher.html"
API_FILE="demo_install_server.py"
LOG_DIR="$DEMO_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_HTML="$LOG_DIR/demo_launcher.log"
LOG_API="$LOG_DIR/demo_api.log"
# On-box access is localhost; remote is the tailnet address printed at startup.
# Both work with no forwarding because every server binds 0.0.0.0. See RUNBOOK.md
# "Access" (incl. why not to route demos through VS Code auto-forwarding).
HOST_IP="localhost"
# Resolve the tailnet address at runtime (never hard-code it). Empty if Tailscale
# is down — then only the on-box localhost URL is shown.
TS_IP=""; TS_DNS=""
if command -v tailscale >/dev/null 2>&1; then
    TS_IP="$(tailscale ip -4 2>/dev/null | head -1)"
    # Must be .Self.DNSName — grepping the first "DNSName" can hit a peer.
    # No jq => skip it; the IP URL below always works.
    if command -v jq >/dev/null 2>&1; then
        TS_DNS="$(tailscale status --json 2>/dev/null \
            | jq -r '.Self.DNSName // empty' | sed 's/\.$//')"
    fi
fi

# Pin the interpreter — a bare `python3` can resolve to a conda env and drift
# the whole stack onto the wrong Python. Fall back to PATH only if absent.
PYTHON_BIN="/usr/bin/python3.12"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="$(command -v python3)"

echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║      HP ZGX Nano — AI Demo Center            ║"
echo "  ║      Customer Demo Launcher v2.0             ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""

# Kill anything already on our ports
for PORT in $DEMO_PORT $API_PORT; do
    EXISTING=$(lsof -ti:$PORT 2>/dev/null)
    if [ -n "$EXISTING" ]; then
        echo "  ⚡ Stopping existing process on port $PORT..."
        kill -9 $EXISTING 2>/dev/null
        sleep 0.5
    fi
done

# Check launcher HTML exists
if [ ! -f "$DEMO_DIR/$DEMO_FILE" ]; then
    echo "  ✗  ERROR: $DEMO_FILE not found in $DEMO_DIR"
    exit 1
fi

# Auto-install FastAPI/uvicorn if needed
if [ -f "$DEMO_DIR/$API_FILE" ]; then
    "$PYTHON_BIN" -c "import fastapi, uvicorn, httpx" 2>/dev/null || {
        echo "  📦 Installing FastAPI + uvicorn + httpx (one-time setup)..."
        "$PYTHON_BIN" -m pip install fastapi "uvicorn[standard]" httpx --break-system-packages -q
    }
fi

# Start HTML file server on 8085
cd "$DEMO_DIR"
"$PYTHON_BIN" -m http.server $DEMO_PORT --bind ${ZGX_BIND:-0.0.0.0} > "$LOG_HTML" 2>&1 &
HTML_PID=$!
sleep 0.8

if ! kill -0 $HTML_PID 2>/dev/null; then
    echo "  ✗  File server failed to start. Check $LOG_HTML"; exit 1
fi
echo "  ✓  Demo Center    (PID $HTML_PID)  →  port $DEMO_PORT"

# Start Install API on 8086
API_PID=""
if [ -f "$DEMO_DIR/$API_FILE" ]; then
    "$PYTHON_BIN" "$DEMO_DIR/$API_FILE" > "$LOG_API" 2>&1 &
    API_PID=$!
    sleep 1.5
    if kill -0 $API_PID 2>/dev/null; then
        echo "  ✓  Install API    (PID $API_PID)  →  port $API_PORT"
    else
        echo "  ⚠  Install API failed — check $LOG_API"
        API_PID=""
    fi
else
    echo "  ⚠  demo_install_server.py not found — copy to $DEMO_DIR to enable Install buttons"
fi

echo ""
BAR=63
printf '  ┌'; printf '─%.0s' $(seq "$BAR"); printf '┐\n'
printf '  │  ▶  %-13s%-45s│\n' "Demo Center:" "http://$HOST_IP:$DEMO_PORT/$DEMO_FILE"
[ -n "$API_PID" ] && printf '  │  ▶  %-13s%-45s│\n' "Install API:" "http://$HOST_IP:$API_PORT/health"
printf '  │%-63s│\n' ""
printf '  │%-63s│\n' "  ZGX Performance Lab runs separately on port 8080"
printf '  └'; printf '─%.0s' $(seq "$BAR"); printf '┘\n'
echo ""
if [ -n "$TS_IP" ]; then
    echo "  Remote access (tailnet — from any device on the tailnet, no forwarding):"
    echo "     ▶  http://$TS_IP:$DEMO_PORT/$DEMO_FILE"
    [ -n "$TS_DNS" ] && echo "     (or via MagicDNS if enabled: http://$TS_DNS:$DEMO_PORT/$DEMO_FILE)"
    echo ""
else
    echo "  Remote access: Tailscale not detected — start it (sudo tailscale up) to"
    echo "  reach the launcher from other tailnet devices without VS Code forwarding."
    echo ""
fi
echo "  Press Ctrl+C to stop."
echo ""

cleanup(){ echo ""; echo "  Stopping..."; kill $HTML_PID 2>/dev/null; [ -n "$API_PID" ] && kill $API_PID 2>/dev/null; echo "  Done."; exit 0; }
trap cleanup INT TERM
wait $HTML_PID
