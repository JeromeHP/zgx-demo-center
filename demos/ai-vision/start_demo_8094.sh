#!/bin/bash
# HP ZGX Nano — AI Vision Demo launcher
# Port: 8094

DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$HOME/demos/logs"
mkdir -p "$LOG_DIR"

# Kill any existing instance on 8094
fuser -k 8094/tcp 2>/dev/null || true
pkill -f "vision_demo.py" 2>/dev/null || true
sleep 2

# Start AI Vision Demo
# The zgx env is the load-bearing interpreter for every native demo, and it is
# addressed by a fixed absolute path. Renaming the env (conda rename) leaves that
# path dangling and `nohup` then fails into the redirected log file, where the
# launcher never sees it — the demo silently never starts and the UI spins for its
# full poll budget. Fail here instead, on stdout, so the message reaches the
# install API's SSE stream and the operator sees the real cause immediately.
PY="${ZGX_PYTHON:-$HOME/miniforge3/envs/zgx/bin/python3}"
if [ ! -x "$PY" ]; then
    echo "ERROR: zgx env python not found at $PY"
    echo "ERROR: the Demo Center cannot start native demos. Check: conda env list"
    exit 1
fi

cd "$DEMO_DIR"
nohup "$PY" vision_demo.py \
    > "$LOG_DIR/vision_demo.log" 2>&1 &

echo "AI Vision Demo starting on http://$(hostname -I | awk '{print $1}'):8094"
echo "Log: $LOG_DIR/vision_demo.log"
