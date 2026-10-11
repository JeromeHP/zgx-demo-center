#!/bin/bash
# HP ZGX Demo Center — launch wrapper (Phase 7)
#
# Launched from the desktop icon (or over SSH). Brings the stack up if needed,
# then opens the launcher truly fullscreen in Chromium.
#
# Fullscreen recipe (X11 + GNOME/Mutter): Mutter ignores a fullscreen request
# at window-creation time but honors an EWMH request once the window exists. So
# we drop --kiosk, open a chrome-less --app window, wait for it to map, then
# force fullscreen via `wmctrl -b add,fullscreen`. Not a locked kiosk — the
# desktop is escapable; it just presents with no top bar, dock, or chrome.
#
# Admin exit (when fullscreen / chrome-less): Alt+F4 closes the window; the
# in-page "Exit Demo Center" button tears down the whole stack. F11 does NOT
# toggle fullscreen in --app mode (no browser UI to handle the key).
#
# A double-clicked icon has no terminal, so every action is logged to
# $HOME/demos/logs/demo_center_launch.log.
#
# Flags:
#   --check   Resolve chromium, ensure the stack is up, print the command that
#             WOULD run, then exit 0 — without launching. Verifiable over SSH.
set -u

# ── Config (mirrors install.py Phase 5 + run_demos.sh) ──────────────────────────
LAUNCHER_FILE="zgx-demo-launcher.html"
LAUNCHER_PORT=8085
INSTALL_API_PORT=8086
LAUNCHER_URL="http://localhost:${LAUNCHER_PORT}/${LAUNCHER_FILE}"
RUN_DEMOS="$HOME/demos/run_demos.sh"
LOG_DIR="$HOME/demos/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/demo_center_run.log"
LAUNCH_LOG="$LOG_DIR/demo_center_launch.log"
USER_DATA_DIR="$HOME/zgx-demo-center-chrome"
# The launcher's <title> — used to find THIS launch's window, never the desktop
# browser's.
WINDOW_TITLE_MATCH="ZGX Demo Center"
READY_TIMEOUT_S=30      # wait for the stack (:8085) to answer
WIN_TIMEOUT_S=30        # poll iterations (×0.5s) waiting for the window to map

CHECK_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=1 ;;
        *) ;;
    esac
done

# ── Logging — append to the launch log AND echo to stdout (for --check / SSH) ──
log() {
    local line
    line="$(date '+%Y-%m-%d %H:%M:%S') $*"
    echo "$line" | tee -a "$LAUNCH_LOG"
}

# ── Resolve the chromium binary at runtime ──────────────────────────────────────
# Order: PATH names first, then the snap absolute path, then google-chrome.
resolve_chromium() {
    local cand
    for cand in chromium chromium-browser google-chrome; do
        if command -v "$cand" >/dev/null 2>&1; then
            command -v "$cand"
            return 0
        fi
    done
    if [ -x /snap/bin/chromium ]; then
        echo /snap/bin/chromium
        return 0
    fi
    return 1
}

# ── HTTP 200 check (short timeout) ──────────────────────────────────────────────
http_ok() {
    local url="$1"
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$url" 2>/dev/null)"
    [ "$code" = "200" ]
}

# ── Ensure the stack is running (gate on the :8085 launcher) ─────────────────────
ensure_stack() {
    if http_ok "$LAUNCHER_URL"; then
        log "Stack already up (${LAUNCHER_URL})."
        return 0
    fi

    if [ ! -f "$RUN_DEMOS" ]; then
        log "ERROR: run_demos.sh not found at $RUN_DEMOS — cannot start the stack."
        return 1
    fi

    log "Stack not responding — starting via $RUN_DEMOS (UNCONFINED, detached)."
    # CRITICAL: launch the stack UNCONFINED (AppArmor).
    # From the desktop icon this script runs under the 'desktop-icons-ng' profile,
    # which is inherited by everything we spawn — including the install server's
    # detached "Exit" teardown. The kiosk is a chromium snap that only accepts
    # signals peer=unconfined/snap.*, so a confined teardown's SIGKILL is DENIED:
    # the python servers die but the snap window survives. `systemd-run --user`
    # runs the stack under the (unconfined) user systemd manager instead of as our
    # child, so its later teardown's kills are accepted. KillMode=process keeps
    # systemd from tearing down the cgroup (and aborting the detached teardown)
    # when run_demos exits mid-teardown. Falls back to setsid if systemd-run is
    # unavailable (degraded: stack comes up, but Exit's window-close is best-effort).
    local launched=0
    if command -v systemd-run >/dev/null 2>&1; then
        if systemd-run --user --quiet --collect -p KillMode=process \
                bash "$RUN_DEMOS" >>"$RUN_LOG" 2>&1; then
            log "Stack launched UNCONFINED via systemd-run (user manager)."
            launched=1
        else
            log "WARNING: 'systemd-run --user' failed (no user bus / denied?) — falling back to setsid; stack will be CONFINED and Exit may not close the kiosk window."
        fi
    fi
    if [ "$launched" -eq 0 ]; then
        if command -v setsid >/dev/null 2>&1; then
            setsid bash "$RUN_DEMOS" </dev/null >>"$RUN_LOG" 2>&1 &
        else
            nohup bash "$RUN_DEMOS" </dev/null >>"$RUN_LOG" 2>&1 &
        fi
    fi

    log "Waiting up to ${READY_TIMEOUT_S}s for :${LAUNCHER_PORT} to respond..."
    local i
    for ((i = 0; i < READY_TIMEOUT_S; i++)); do
        if http_ok "$LAUNCHER_URL"; then
            log "Stack is up (launcher :${LAUNCHER_PORT})."
            return 0
        fi
        sleep 1
    done
    log "ERROR: stack did not become ready within ${READY_TIMEOUT_S}s (see $RUN_LOG)."
    return 1
}

# ── Kill any prior kiosk chromium for THIS profile ──────────────────────────────
# Matches the kiosk's --user-data-dir flag. Safe: this script's argv is just
# `bash .../start_demo_center.sh` (never an inline `bash -c`), so the flag isn't
# in our own command line and pkill skips its own PID anyway.
kill_prior_kiosk() {
    pkill -f -- "--user-data-dir=${USER_DATA_DIR}" 2>/dev/null || true
    sleep 1
}

# ── Clear the crash flag so there's no "restore pages" bar ───────────────────────
# Sets profile.exited_cleanly=true / exit_type="Normal" in the profile Prefs,
# handling the first-run case where the file doesn't exist yet.
clear_crash_flag() {
    local prefs="$USER_DATA_DIR/Default/Preferences"
    local py
    py="$(command -v python3 || echo /usr/bin/python3)"
    mkdir -p "$USER_DATA_DIR/Default"
    "$py" - "$prefs" <<'PYEOF'
import json, os, sys
path = sys.argv[1]
try:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        data = {}
except Exception:
    data = {}                      # first run / corrupt — start clean
prof = data.get("profile")
if not isinstance(prof, dict):
    prof = {}
prof["exit_type"] = "Normal"
prof["exited_cleanly"] = True
data["profile"] = prof
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    json.dump(data, f)
PYEOF
    log "Cleared crash flag in $prefs (exit_type=Normal, exited_cleanly=true)."
}

# ── Wait for this launch's window to map, then force true EWMH fullscreen ────────
fullscreen_launcher() {
    if ! command -v wmctrl >/dev/null 2>&1; then
        log "WARNING: wmctrl not found — window opened but NOT forced fullscreen."
        return 1
    fi
    export DISPLAY="${DISPLAY:-:0}"

    local i wid=""
    for ((i = 0; i < WIN_TIMEOUT_S; i++)); do
        # A top-level window matching our title that wasn't present before launch
        # (so we never grab the desktop browser, which may have a same-named tab).
        while read -r id _rest; do
            case ",${WINS_BEFORE}," in
                *",${id},"*) continue ;;   # existed before launch — not ours
            esac
            wid="$id"
            break
        done < <(wmctrl -l 2>/dev/null | grep -F "$WINDOW_TITLE_MATCH")
        [ -n "$wid" ] && break
        sleep 0.5
    done

    if [ -z "$wid" ]; then
        log "WARNING: launcher window ('${WINDOW_TITLE_MATCH}') did not map within $((WIN_TIMEOUT_S / 2))s — not fullscreened."
        return 1
    fi

    log "Launcher window mapped: $wid — requesting EWMH fullscreen (post-map, as Mutter requires)."
    wmctrl -i -r "$wid" -b add,fullscreen 2>/dev/null || true
    sleep 0.3
    wmctrl -i -r "$wid" -b add,fullscreen 2>/dev/null || true   # re-assert once; Mutter occasionally needs a beat
    log "Fullscreen requested on $wid."
}

# ── Main ────────────────────────────────────────────────────────────────────────
log "──────── start_demo_center.sh invoked (check_only=${CHECK_ONLY}) ────────"

CHROMIUM="$(resolve_chromium || true)"
if [ -z "$CHROMIUM" ]; then
    log "ERROR: no chromium/chrome binary found (tried chromium, chromium-browser, /snap/bin/chromium, google-chrome)."
    log "       Install Chromium (e.g. 'sudo snap install chromium') or run install.py Phase 7."
    exit 1
fi
log "Resolved chromium binary: $CHROMIUM"

if ! ensure_stack; then
    exit 1
fi

# Cache-bust the launcher URL — Chromium's persistent profile otherwise serves a
# stale cached page after the HTML is redeployed. The per-launch query string
# forces a fresh load; http.server ignores the query and location.hostname / the
# window <title> are unaffected.
APP_URL="${LAUNCHER_URL}?v=$(date +%s)"

# The command we launch (and that --check prints). --app gives a chrome-less
# window; fullscreen is forced afterward via wmctrl (not --kiosk). --user-data-dir
# is REQUIRED: the in-page Exit teardown pkills the kiosk by exactly that flag.
LAUNCH_CMD=(
    "$CHROMIUM"
    "--app=${APP_URL}"
    "--no-first-run"
    "--no-default-browser-check"
    "--password-store=basic"
    "--disable-session-crashed-bubble"
    "--disable-infobars"
    "--user-data-dir=${USER_DATA_DIR}"
)

if [ "$CHECK_ONLY" -eq 1 ]; then
    log "--check: stack ensured, chromium resolved. Would launch:"
    log "  ${LAUNCH_CMD[*]}"
    echo ""
    echo "Resolved chromium binary: $CHROMIUM"
    echo "Command that WOULD run:"
    echo "  ${LAUNCH_CMD[*]}"
    exit 0
fi

kill_prior_kiosk
clear_crash_flag
mkdir -p "$USER_DATA_DIR"

# Snapshot existing windows so fullscreen_launcher can pick out the NEW one.
export DISPLAY="${DISPLAY:-:0}"
WINS_BEFORE="$(wmctrl -l 2>/dev/null | awk '{print $1}' | paste -sd, )"

log "Launching Demo Center (app mode, detached):"
log "  ${LAUNCH_CMD[*]}"
# Launch DETACHED (own session) so Chromium survives this script exiting. Don't
# exec/replace this process — we still have to fullscreen the window once it maps.
setsid "${LAUNCH_CMD[@]}" </dev/null >>"$LAUNCH_LOG" 2>&1 &
disown 2>/dev/null || true

fullscreen_launcher

log "start_demo_center.sh done."
exit 0
