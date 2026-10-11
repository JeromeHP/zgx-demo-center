#!/bin/bash
# HP ZGX Nano — Grant Machine (Agentic Grant Writing) launcher
# Port: 8095

DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$HOME/demos/logs"
mkdir -p "$LOG_DIR"
NANO_IP=$(hostname -I | awk '{print $1}')

# Kill any existing instance on 8095
fuser -k 8095/tcp 2>/dev/null || true
sleep 2

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

# Machine config is passed as GM_* env vars, NOT written into config/local.yaml.
# settings.py load order is default.yaml -> local.yaml -> GM_<SECTION>__<KEY>, so
# these win. Keeping them here rather than in a file means launching this demo
# never modifies a tracked source file (see KNOWN-ISSUES.md — construction and
# translation rewrite their own source at launch, and that is a bug, not a pattern).
#
# thinking_control=reasoning_effort is REQUIRED on Ollama: it ignores
# chat_template_kwargs.enable_thinking entirely (and `think` on the /v1 route),
# and honours only reasoning_effort, where "none" is the one value that actually
# suppresses thinking. Left on the llama.cpp default, the orchestrator's thinking
# would never be captured for the UI and the drafter/reviewer profiles would think
# when they must not — burning their max_tokens budget on reasoning and returning
# empty content. Verified per-machine by scripts/endpoint_check.py (E3/E4).
export GM_LLM__BASE_URL="http://127.0.0.1:11434/v1"
export GM_LLM__MODEL="qwen3.6:35b"
export GM_LLM__THINKING_CONTROL="reasoning_effort"
export GM_SERVER__PORT=8095
# Self-hosted SearXNG for the Discover mode's general web search. Optional: if the
# container is not running, search_web reports itself unavailable and the federal
# catalogue search (which needs no provider) carries the run. Bound to loopback, and
# reached via the egress guard's configured-endpoint allowance — the query terms are
# validated before they get here exactly as they are for any other destination.
export GM_EGRESS__SEARXNG_URL="http://127.0.0.1:8096"
# The device label shows in the UI header, the replay banner AND the generated
# PDF packet (app/packet.py:463,485) -- a packet handed to a grants director is a
# durable artifact, so naming the wrong machine there is worse than cosmetic.
# It used to be the literal string "HP ZGX Nano", which meant an AMD box would
# have to be remembered and hand-edited before every show. Ask the Demo Center's
# detector instead, and fall back to the literal only if it is unavailable (this
# demo is also runnable standalone, outside the Demo Center checkout).
DETECT="$(dirname "$DEMO_DIR")/zgx_hardware.py"
if [ -f "$DETECT" ]; then
  GM_DEMO__DEVICE_LABEL="$("$PY" -c "
import json,sys; sys.path.insert(0,'$(dirname "$DEMO_DIR")')
import zgx_hardware; print(zgx_hardware.detect()['device_label'])" 2>/dev/null)"
fi
export GM_DEMO__DEVICE_LABEL="${GM_DEMO__DEVICE_LABEL:-this AI Station}"
echo "  device label: $GM_DEMO__DEVICE_LABEL"

cd "$DEMO_DIR"
nohup "$PY" -m app.main > "$LOG_DIR/grant_machine_demo.log" 2>&1 &

echo "Grant Machine starting on http://${NANO_IP}:8095"
echo "Log: $LOG_DIR/grant_machine_demo.log"
