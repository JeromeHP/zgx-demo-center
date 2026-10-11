#!/bin/bash
DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$HOME/demos/logs"
mkdir -p "$LOG_DIR"
NANO_IP=$(hostname -I | awk '{print $1}')

# Kill any existing instance on 8088
fuser -k 8088/tcp 2>/dev/null || true
sleep 2

# Start Pipeline Pulse on port 8088
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
export PP_PORT=8088
nohup "$PY" pipeline_pulse_v6.py > "$LOG_DIR/pipeline_pulse_demo.log" 2>&1 &

echo "Pipeline Pulse (Demo Center) starting on http://${NANO_IP}:8088"
