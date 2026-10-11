#!/usr/bin/env python3
"""
HP ZGX Nano — Demo Install Server
Runs on port 8086 alongside the demo launcher (port 8085).
Provides SSE-streamed install endpoints for each demo.
"""

import asyncio
import os
import json
import shutil
import subprocess
from collections import deque
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from zgx_demo_lib import (
    CommandExit,
    sse_event,
    stream_command,
    check_ollama_model,
    check_port_alive,
    check_http_health,
    check_files_and_model,
)
from install_plans import INSTALL_PLANS
import zgx_gpu_telemetry
import zgx_hardware

app = FastAPI()

# ── Log/runtime data location ──────────────────────────────────────────────
# All Demo Center logs live under the repo's logs/ dir (~/demos/logs), never
# scattered in $HOME. Created on demand by the writers below.
LOG_DIR = Path.home() / "demos" / "logs"

# ── Load time tracking ─────────────────────────────────────────────────────
LOAD_TIMES_FILE = Path.home() / "demos" / ".load_times.json"
LAST_LOAD_TIMES: dict[str, float] = {}

def _load_persisted_times():
    global LAST_LOAD_TIMES
    try:
        if LOAD_TIMES_FILE.exists():
            with open(LOAD_TIMES_FILE) as f:
                LAST_LOAD_TIMES = {k: float(v) for k, v in json.load(f).items()}
    except Exception:
        LAST_LOAD_TIMES = {}

def _persist_times():
    try:
        LOAD_TIMES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOAD_TIMES_FILE, "w") as f:
            json.dump(LAST_LOAD_TIMES, f)
    except Exception:
        pass

_load_persisted_times()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1|100\.\d+\.\d+\.\d+|[A-Za-z0-9-]+|[A-Za-z0-9-]+\.tail33c034\.ts\.net)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)

DEMOS_DIR = Path.home() / "demos"


def resolve_plan(plan: dict) -> dict:
    """Apply this machine's hardware-profile overrides to a plan.

    install_plans.py stays pure data; the merge happens here. A plan's
    "profiles" entry for the active profile replaces the keys it names (today
    only "steps") -- e.g. on amd-rocm the vision demo installs onnxruntime
    instead of tensorrt-cu13 + pycuda, which do not exist for ROCm.

    Returns the plan unchanged when there is no override, so the NVIDIA path is
    byte-for-byte what it was.
    """
    overrides = (plan.get("profiles") or {}).get(zgx_hardware.detect()["profile"])
    if not overrides:
        return plan
    merged = dict(plan)
    merged.update(overrides)
    return merged


def plan_is_supported(plan: dict) -> bool:
    """False when this demo declares an accelerator this box does not have."""
    needs = (plan.get("requires") or {}).get("profiles")
    return not needs or zgx_hardware.detect()["profile"] in needs


@app.get("/demos")
async def demos():
    """The installable demos for THIS machine, in install order.

    The launcher's "Install All" hardcoded six ids and silently omitted five --
    grant, vision, spectrum, pipeline and ballistics -- so pressing it produced
    a Demo Center where those were never installed, with nothing saying so.
    That list was also blind to hardware, and on an AMD box would have tried the
    NVIDIA-only demos.

    Order matters and is preserved from INSTALL_PLANS: vision pip-installs the
    TensorRT wheel that spectrum and install.py's Phase 4b engine build depend on.
    (See KNOWN-ISSUES.md on the demo/port tables being maintained by hand in
    several places; this is the endpoint that entry asks for.)
    """
    profile = zgx_hardware.detect()["profile"]
    out = []
    for demo_id, plan in INSTALL_PLANS.items():
        needs = (plan.get("requires") or {}).get("profiles")
        out.append({
            "id": demo_id,
            "name": plan.get("name", demo_id),
            "port": plan.get("port"),
            "path": plan.get("port_path", "/"),
            "supported": not needs or profile in needs,
            "requires": needs or [],
            "steps": len(resolve_plan(plan).get("steps", [])),
        })
    return {"profile": profile, "demos": out}


@app.get("/status/{demo_id}")
async def get_status(demo_id: str):
    """Check if a demo is already installed."""
    plan = INSTALL_PLANS.get(demo_id)
    if not plan:
        return {"installed": False, "error": "Unknown demo"}

    # Resolve profile overrides here too, not just in /install. Without this the
    # AMD vision readiness check still demanded ~/yolov8s.trt -- a file Phase 4b
    # deliberately never builds on ROCm -- so the card read "Needs Setup"
    # forever, and re-installing could not possibly clear it.
    plan = resolve_plan(plan)
    rc = plan.get("readiness_check")
    if rc:
        return await check_files_and_model(
            rc["files"],
            rc["ollama_model"],
            rc.get("path"),
        )

    demo_path = DEMOS_DIR / plan["dir"]
    installed = demo_path.exists()

    model_ok = True
    if installed:
        models_dir = demo_path / "models"
        if models_dir.exists():
            model_ok = any(models_dir.iterdir())

    return {
        "installed": installed,
        "model_ok": model_ok,
        "path": str(demo_path),
    }


@app.get("/install/{demo_id}")
async def install_demo(demo_id: str):
    """Stream installation progress via SSE."""
    plan = INSTALL_PLANS.get(demo_id)
    if not plan:
        async def error_gen():
            yield sse_event({"type": "error", "msg": f"Unknown demo: {demo_id}"})
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    plan = resolve_plan(plan)
    demo_path = DEMOS_DIR / plan["dir"]
    DEMOS_DIR.mkdir(exist_ok=True)

    async def install_gen():
        if not plan_is_supported(plan):
            needs = "/".join(plan["requires"]["profiles"])
            yield sse_event({"type": "error", "msg":
                             f"{plan['name']} requires {needs}; this machine is "
                             f"{zgx_hardware.detect()['profile']}."})
            return
        yield sse_event({"type": "start", "name": plan["name"], "steps": len(plan["steps"])})

        for i, step in enumerate(plan["steps"]):
            label = step["label"]
            raw_cmd = step["cmd"]

            # Skip clone steps for vendored demos (no repo URL in plan).
            if "git clone" in raw_cmd and not plan.get("repo"):
                yield sse_event({"type": "step_skip", "step": i, "label": label, "msg": "Source vendored — no clone needed"})
                continue

            # Check if already done (repo already cloned)
            if "git clone" in raw_cmd and demo_path.exists():
                yield sse_event({"type": "step_skip", "step": i, "label": label, "msg": "Already cloned — skipping"})
                continue

            cmd = raw_cmd.format(
                repo=plan.get("repo", ""),
                dir=str(demo_path),
            )

            yield sse_event({"type": "step_start", "step": i, "label": label, "total": len(plan["steps"])})

            # Fix 3: per-step heartbeat. Download steps declare a `heartbeat`
            # ({"label","path","total_gb"?}) so a quiet multi-GB pull reports its
            # on-disk size every ~30s; the {dir} placeholder in `path` is expanded
            # the same way the command is. Steps without it still get the generic
            # elapsed-time heartbeat from stream_command (covers docker builds).
            hb = step.get("heartbeat")
            hb_arg = None
            if hb:
                hb_arg = dict(hb)
                hb_arg["path"] = hb["path"].format(dir=str(demo_path))

            # A step's exit code arrives as the generator's terminal item.
            # Report a failure as output (so it is visible in the console) but
            # keep going: several plans use `cmd-a || cmd-b` fallbacks and
            # tolerate a failing step, so aborting here would break them.
            step_rc = 0
            async for line in stream_command(cmd, DEMOS_DIR, heartbeat=hb_arg):
                if isinstance(line, CommandExit):
                    step_rc = line.code
                    continue
                yield sse_event({"type": "output", "line": line})

            if step_rc != 0:
                yield sse_event({"type": "output",
                                 "line": f"ERROR: step exited with code {step_rc} — {label}"})

            yield sse_event({"type": "step_done", "step": i, "label": label,
                             "exit_code": step_rc, "optional": bool(step.get("optional"))})

        if "note" in plan:
            yield sse_event({"type": "note", "msg": plan["note"]})

        yield sse_event({
            "type": "complete",
            "name": plan["name"],
            "port": plan.get("port"),
            "path": plan.get("port_path", "/"),
        })

    return StreamingResponse(
        install_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/launch/{demo_id}")
async def launch_demo(demo_id: str):
    """Start a demo's server in the background."""
    plan = INSTALL_PLANS.get(demo_id)
    if not plan:
        return {"ok": False, "error": "Unknown demo"}

    demo_path = DEMOS_DIR / plan["dir"]
    if not demo_path.exists():
        return {"ok": False, "error": "Not installed yet"}

    cmd = plan["launch_cmd"].format(dir=str(demo_path))

    subprocess.Popen(
        cmd, shell=True, cwd=demo_path,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True
    )

    return {"ok": True, "port": plan.get("port"), "path": plan.get("port_path", "/")}


@app.get("/port_check/{port}")
async def port_check(port: int):
    return {"alive": await check_port_alive(port)}

@app.get("/health_check/{demo_id}")
async def health_check(demo_id: str, path: str | None = None):
    """Check if a demo is fully ready by hitting its health endpoint."""
    plan = INSTALL_PLANS.get(demo_id)
    health_url = plan.get("health_url") if plan else None
    if not health_url:
        return {"ready": False, "error": "Unknown demo"}
    parts = urlsplit(health_url)
    base = f"{parts.scheme}://{parts.netloc}"
    default_path = parts.path or "/"
    url = base + (path if path else default_path)
    return await check_http_health(url)

@app.get("/robot_check")
async def robot_check():
    """Check if a Reachy Mini robot is reachable on the local network."""
    import httpx
    urls_to_try = [
        "http://reachy-mini.local:8000/api/apps/list-available",
        "http://reachy-mini.local:8000/",
    ]
    async with httpx.AsyncClient(timeout=3.0) as client:
        for url in urls_to_try:
            try:
                resp = await client.get(url)
                if resp.status_code in [200, 404]:
                    return {
                        "robot_found": True,
                        "url": url,
                        "status": resp.status_code,
                    }
            except Exception:
                continue
    return {"robot_found": False}

@app.get("/bench_status")
async def bench_status():
    """Check if the Performance Lab is running on port 8080."""
    import socket
    s = socket.socket()
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", 8080))
        s.close()
        running = True
    except:
        running = False
    return {"bench_running": running}

@app.post("/ollama_release")
async def ollama_release():
    """Release all Ollama models from GPU memory."""
    proc = await asyncio.create_subprocess_shell(
        "ollama stop llama3.1:70b 2>/dev/null; "
        "ollama stop medical-mixtral 2>/dev/null; "
        "ollama stop llama3.3:70b 2>/dev/null; "
        "ollama stop llama3.1:8b 2>/dev/null; "
        "ollama stop llama3.2:3b 2>/dev/null; "
        "sleep 2",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.communicate()
    return {"ok": True}


@app.post("/exit_kiosk")
async def exit_kiosk():
    """FULL Demo Center teardown — "Exit" means "release the Nano".

    Stops EVERY demo, releases ALL Ollama models, closes the kiosk browser,
    then stops the Demo Center servers themselves (:8085 HTML, then :8086 this
    install API — LAST). After this nothing Demo-Center-related is left running;
    relaunching the desktop icon rebuilds the whole stack in a few seconds.

    DETACHED-PROCESS REQUIREMENT: the final steps kill the :8086 server this
    endpoint runs on, so the whole teardown is launched as a DETACHED process
    (setsid / start_new_session=True). It lives in its own session, holds no
    port, and is therefore immune to the `fuser -k 8086/tcp` that kills us — so
    killing :8086 can't abort the script midway. We return 200 immediately and
    let the detached script run to completion. Every step is best-effort
    (|| true) so one failure never stops the rest.

    The teardown is BUILT FROM INSTALL_PLANS (mirroring `_teardown_plan` and
    `/ollama_release` in shell, since the detached process can't reuse those
    coroutines) so it stays complete if the demo lineup changes:
      - docker_compose demos → `docker compose down` in their dir
      - all demos            → `fuser -k` / `lsof` on their declared port(s)
      - process_pattern demos→ `pkill -f` their patterns
      - Ollama               → stop every currently-loaded model
    Order: SIGKILL browser FIRST (fire-and-forget) → stop demos → release ollama
    → THEN kill :8085 and :8086 (LAST), UNCONDITIONALLY. Browser-close runs first
    so Exit dismisses the UI promptly even when the demo/ollama cleanup is slow or
    wedged — e.g. a hung GPU vision demo that blocks `ollama ps`/`ollama stop`,
    which is exactly how Exit got stuck behind step 2 in the field. The demo and
    ollama steps are `timeout`-bounded for the same reason.

    GPU-WEDGE SURVIVAL (why every step is unconditional): rapid AI Vision switching
    wedges the GPU, leaving chrome's gpu/renderer procs in uninterruptible sleep
    (D-state) where SIGKILL cannot reap them immediately. A SIGKILL to a D-state
    process is NOT lost — it stays PENDING and is delivered the instant the GPU
    un-wedges, so the frozen window self-closes on recovery. We therefore send the
    kill and PROCEED — we do NOT verify "browser gone" or gate the server kills on
    it. Stopping the demos in step 2 drops GPU load, which helps the wedge clear so
    the pending chrome SIGKILL lands sooner (demo-stop doubles as un-wedging the
    browser). The OLD guard SKIPPED the :8085/:8086 kills unless the browser was
    confirmed gone; during a wedge that abort is precisely what made Exit appear to
    do nothing. There is no "live page on dead servers" case left to fear: a
    non-wedged chrome always dies on SIGKILL, and a wedged one shows a frozen page
    that the pending SIGKILL closes anyway.

    INSTRUMENTED: every step appends a timestamped line to ~/demos/logs/exit_kiosk.log
    (endpoint hit, browser-close patterns + matched PIDs before the SIGKILL, a
    "proceeding regardless" marker, and start/end of each later step incl. the
    unconditional :8085/:8086 kills) so a failed Exit is never silent. Pair with
    ~/watch_8086.sh to see if :8086 itself died/hung.
    """
    import datetime as _dt
    import shlex
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _exit_log = LOG_DIR / "exit_kiosk.log"

    def _log_py(msg: str):
        try:
            with open(_exit_log, "a") as f:
                f.write(f"{_dt.datetime.now():%Y-%m-%d %H:%M:%S.%f} | [py] {msg}\n")
        except Exception:
            pass

    _log_py("════════ /exit_kiosk ENDPOINT HIT — building detached teardown ════════")

    def _port_kill(p: int) -> str:
        # Same universal fallthrough _teardown_plan uses: fuser, then lsof —
        # both scoped to the LISTENER (-sTCP:LISTEN) so neither catches a process
        # that merely holds an OUTBOUND connection to the port (friendly fire).
        return f"fuser -k {p}/tcp 2>/dev/null || lsof -ti:{p} -sTCP:LISTEN | xargs -r kill -9 2>/dev/null || true"

    def _proc_pattern_kill(patterns: list[str]) -> list[str]:
        """Emit bash that SIGKILLs procs matching each pattern BY PID, skipping the
        teardown's own process group.

        SELF-MATCH FOOTGUN (same class that bit browser-close): a literal
        `pkill -f '<pat>'` here would match THIS detached `bash -c '<script>'` —
        the script TEXT contains the pattern, so the teardown's own cmdline (and
        every subshell it forks, including `$( … )` command substitutions) contains
        it too. pkill would then SIGKILL the teardown mid-run — during demo-stop,
        BEFORE the server kills — stranding :8085/:8086 UP, the exact regression we
        must avoid. The $PROFILE/$VAR trick used for the browser patterns does NOT
        help: a bare script name like 'vision_demo.py' is the same literal whether
        written inline or assigned to a var, so it still self-matches.

        Fix: resolve PIDs with `pgrep -f` and skip any whose PGID == ours. The
        teardown is launched start_new_session=True, so it leads its own process
        group (PGID==$$) shared by all its subshells; the real demo was launched
        new_session=True as well, so it lives in a DIFFERENT group and is killed.
        We kill by explicit PID (never by pattern), so the kill itself can't
        self-match either.
        """
        if not patterns:
            return []
        lines = ['TD_PGID="$(ps -o pgid= -p $$ | tr -d " ")"']
        for pat in patterns:
            q = shlex.quote(pat)
            lines += [
                f'for _pid in $(pgrep -f -- {q} 2>/dev/null); do',
                '    _pg="$(ps -o pgid= -p "$_pid" 2>/dev/null | tr -d " ")"',
                '    [ "$_pg" = "$TD_PGID" ] && continue   # our own teardown group — never kill',
                '    kill -9 "$_pid" 2>/dev/null || true',
                'done',
            ]
        return lines

    # ── 1. Stop every demo (registry-driven, mirrors _teardown_plan) ──────────
    demo_cmds: list[str] = []
    proc_patterns: list[str] = []
    kill_ports: set[int] = set()
    for plan in INSTALL_PLANS.values():
        strategy = plan.get("teardown_strategy") or {}
        stype = strategy.get("type")
        if stype == "docker_compose":
            ddir = DEMOS_DIR / plan["dir"]
            demo_cmds.append(
                f"(cd '{ddir}' && (timeout 25 docker compose down 2>/dev/null || timeout 25 docker-compose down 2>/dev/null)) || true"
            )
        elif stype == "process_pattern":
            # Collected, then killed via _proc_pattern_kill (PID-resolved + own-PGID
            # excluded) — NOT a literal `pkill -f`, which would self-match and kill
            # this teardown mid-run. See _proc_pattern_kill for the full rationale.
            proc_patterns.extend(strategy.get("patterns", []))
        port = plan.get("port")
        if port:
            kill_ports.add(int(port))
    # Native demos also bind a backend port (frontend+100) that isn't a separate
    # registry entry: construction 8091→8191, translation 8092→8192.
    kill_ports.update({8191, 8192})
    # process_pattern kills run FIRST (catch a still-loading proc that hasn't bound
    # its port yet — the Exit-during-load orphan), THEN port kills as fallthrough.
    proc_kill_cmds = _proc_pattern_kill(proc_patterns)
    # Port kills run AFTER the docker downs above so containers release :8000 first.
    for p in sorted(kill_ports):
        demo_cmds.append(_port_kill(p))

    # ── 2. Release ALL Ollama models (mirrors /ollama_release, but dynamic so it
    #       frees whatever is loaded: mixtral, nomic, llama3.1:70b,
    #       tinyllama-pii, pii-detector, …) ──────────────────────────────────
    # timeout-bounded: a wedged GPU (e.g. after a hung vision demo) can make the
    # ollama daemon — and therefore `ollama ps`/`ollama stop` — block indefinitely.
    # Browser-close already ran by the time we get here, but the bounds keep the
    # rest of the teardown (incl. the server kills) from hanging forever too.
    ollama_cmd = (
        "for m in $(timeout 10 ollama ps 2>/dev/null | awk 'NR>1{print $1}'); do "
        'timeout 10 ollama stop "$m" 2>/dev/null || true; done'
    )

    # ── 3. Close the kiosk browser — fire-and-forget SIGKILL (NO verify/gate) ──
    # Hold the profile path and app port in RUNTIME shell vars ($PROFILE/$KPORT)
    # rather than f-stringing the expanded values into the script. `pkill -f`
    # matches whole command lines, and this script's own cmdline is
    # `bash -c '<this script>'` — if an EXPANDED kiosk pattern appeared in the text,
    # the kiosk pkill/pgrep would match and KILL THIS TEARDOWN. With the vars, the
    # script text holds only "$PROFILE"/"$KPORT", so the real chromium (expanded
    # values in its argv) still matches but we don't suicide.
    #
    # SIGKILL (-9), root --app= process FIRST (its death tears down the child
    # zygote/gpu/renderer procs and stops them respawning), then the whole profile
    # group as a backstop. We do NOT verify here and do NOT gate the server kills on
    # it — that gate is what made Exit look dead during a wedge. Fire the kill and
    # march straight through the teardown.
    #
    # BUT a single SIGKILL is NOT sufficient on its own: observed in the field, a
    # chrome that was GPU-wedged at the instant of Exit survived this kill and left
    # NO pending signal (SigPnd=0), so the window just sat there on dead servers — an
    # identical pkill 2 min later killed it cleanly, proving it was a TRANSIENT miss,
    # not a pattern bug. So the reliable close is handled by the bounded REAPER at the
    # very END of the teardown (after the server kills), which retries this same kill
    # ~1×/s until the window is actually gone. This first shot just dismisses the UI
    # promptly in the common (non-wedged) case.
    browser_cmd = "\n".join([
        'KPORT="8085"',
        'PROFILE="$HOME/zgx-demo-center-chrome"',
        'PKILL_PAT="--user-data-dir=$PROFILE"',
        'ROOT_PAT="--app=http://localhost:$KPORT/"',
        r'MATCHED="$(pgrep -f -- "$PKILL_PAT" | tr "\n" " ")"',
        'log "browser-close: profile_pat=[$PKILL_PAT] root_pat=[$ROOT_PAT]"',
        'log "browser-close: pgrep matched -> count=$(echo $MATCHED | wc -w) pids=[${MATCHED:-<none>}]"',
        'pkill -9 -f -- "$ROOT_PAT"  2>/dev/null || true   # root --app= browser first',
        'pkill -9 -f -- "$PKILL_PAT" 2>/dev/null || true   # all kiosk-profile procs (backstop)',
        'log "browser-close: SIGKILL sent (stays PENDING if D-state) — proceeding regardless"',
    ])

    # ── 4/5. Stop the Demo Center servers — UNCONDITIONALLY (no browser gate).
    # The only way chrome outlives its SIGKILL is a D-state GPU wedge, where the page
    # is frozen (not a functional UI worth preserving) and the reaper below will close
    # the window as soon as the GPU recovers. So we always release the Nano: the
    # demo/ollama stops above have already dropped GPU load to help the wedge clear,
    # and we kill :8085 then :8086 (LAST — our own server) come what may.
    server_kill_cmd = "\n".join([
        'log "kill-8085 START"',
        _port_kill(8085) + '    # :8085 Demo Center HTML',
        'log "kill-8085 END"',
        'log "kill-8086 START (kills our own install server; detached script continues)"',
        _port_kill(8086) + '    # :8086 install API LAST — kills our own server',
        'log "kill-8086 END — teardown COMPLETE (servers down; reaper now closing window)"',
    ])

    # ── 6. REAPER — bounded background retry of the browser-close kill ────────────
    # The server kills already ran (above) and are NOT gated on this; the Nano is
    # already released. This loop ONLY closes the kiosk window, which can survive the
    # single SIGKILL in step 1 if chrome was GPU-wedged (D-state) at that instant —
    # in the field that miss left no pending signal, so the window stayed up on dead
    # servers. We retry the same kill ~1×/s for up to ~90s; the moment chrome leaves
    # D-state a retry lands and the window closes (no user re-click, no ALT+F4).
    #
    # Self-match safety (same discipline as the rest of the teardown): patterns stay
    # as the $VAR-protected $ROOT_PAT/$PKILL_PAT (set in browser_cmd — expanded values
    # never appear in this script's text), AND we resolve PIDs with pgrep, EXCLUDE any
    # in the teardown's own process group (PGID==$$, shared by its subshells), and
    # kill by explicit PID — so neither the match-check nor the kill can hit us. The
    # PGID exclusion also makes the "window gone" check correct: it can never see our
    # own subshells and loop to the timeout by mistake.
    reaper_cmd = "\n".join([
        'log "reaper START — retry browser-close until window gone (max 90s)"',
        'R_PGID="$(ps -o pgid= -p $$ | tr -d " ")"',
        'reaped=0',
        'for _i in $(seq 1 90); do',
        '    _rem=""',
        '    for _pid in $(pgrep -f -- "$ROOT_PAT" 2>/dev/null; pgrep -f -- "$PKILL_PAT" 2>/dev/null); do',
        '        _pg="$(ps -o pgid= -p "$_pid" 2>/dev/null | tr -d " ")"',
        '        [ "$_pg" = "$R_PGID" ] && continue   # our own teardown group — never count/kill',
        '        _rem="$_rem $_pid"',
        '    done',
        '    if [ -z "${_rem// /}" ]; then',
        '        log "reaper: window gone after ${_i}s"; reaped=1; break',
        '    fi',
        '    for _pid in $_rem; do kill -9 "$_pid" 2>/dev/null || true; done',
        '    sleep 1',
        'done',
        '[ "$reaped" = "1" ] || log "reaper: TIMEOUT after 90s — window still up [${_rem# }]"',
        'log "reaper END"',
    ])

    script = "\n".join([
        "#!/bin/bash",
        "# Detached Demo Center teardown — survives the death of :8086.",
        '# INSTRUMENTED — appends timestamped progress to ~/demos/logs/exit_kiosk.log.',
        'mkdir -p "$HOME/demos/logs"',
        'LOGF="$HOME/demos/logs/exit_kiosk.log"',
        r'log(){ printf "%s | %s\n" "$(date +"%Y-%m-%d %H:%M:%S.%N")" "$*" >> "$LOGF"; }',
        'log "detached teardown script STARTED (pid $$)"',
        "sleep 0.4   # let the 200 response flush to the caller first",
        "# 1. SIGKILL the kiosk browser FIRST (fire-and-forget) — Exit must dismiss the",
        "#    UI promptly; if the GPU is wedged the kill pends and lands on recovery.",
        browser_cmd,
        "# 2. Stop every demo: process_pattern kills FIRST (self-match-safe, by PID —",
        "#    catches a still-loading proc with no bound port yet), then docker downs",
        "#    and port kills as fallthrough.",
        'log "demo-stop START"',
        *proc_kill_cmds,
        *demo_cmds,
        'log "demo-stop END"',
        "# 3. Release all Ollama models (timeout-bounded)",
        'log "ollama-release START"',
        ollama_cmd,
        'log "ollama-release END"',
        "# 4/5. Kill :8085 then :8086 (LAST) — UNCONDITIONALLY (release the Nano)",
        server_kill_cmd,
        "# 6. Reaper — close the kiosk window even if it survived the step-1 SIGKILL",
        "#    (GPU-wedged). Runs AFTER the server kills; does not gate them.",
        reaper_cmd,
    ])

    # Launch fully detached (new session) so the `fuser -k 8086` above can't take
    # the rest of the script down with it. Popen returns immediately.
    subprocess.Popen(
        ["/bin/bash", "-c", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    return {"ok": True}


@app.post("/bench_stop")
async def bench_stop():
    """Stop the Performance Lab in the background (non-blocking)."""
    async def _do_stop():
        # Kill only the bench app Python processes by name
        # Do NOT use lsof -ti:8080 | kill — too aggressive,
        # kills parent shells and disconnects VS Code SSH
        proc1 = await asyncio.create_subprocess_shell(
            "pkill -f 'trt_bench_ui.py' 2>/dev/null; "
            "pkill -f 'trt_bench_ui_' 2>/dev/null; "
            "pkill -f 'bench_frontend' 2>/dev/null; "
            "sleep 1",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc1.communicate()
        # Stop Ollama models to release GPU memory
        proc2 = await asyncio.create_subprocess_shell(
            "ollama stop llama3.1:70b 2>/dev/null; "
            "ollama stop llama3.3:70b 2>/dev/null; "
            "ollama stop llama3.1:8b 2>/dev/null; "
            "ollama stop llama3.2:3b 2>/dev/null; "
            "sleep 2",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc2.communicate()
    # Fire and forget — do not await
    asyncio.create_task(_do_stop())
    return {"ok": True, "message": "Stopping in background — poll /bench_status"}

@app.get("/running")
async def get_running():
    """Return which specific demos are currently running."""
    running: list[str] = []

    container_names: str | None = None

    for demo_id, plan in INSTALL_PLANS.items():
        discovery = plan.get("discovery") or {}
        dtype = discovery.get("type")

        if dtype == "docker_container":
            if container_names is None:
                proc = await asyncio.create_subprocess_shell(
                    "docker ps --format '{{.Names}}' 2>/dev/null",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await proc.communicate()
                container_names = stdout.decode().strip().lower()
            if any(p in container_names for p in discovery.get("patterns", [])):
                if demo_id not in running:
                    running.append(demo_id)

        elif dtype == "process_pattern":
            for pat in discovery.get("patterns", []):
                proc = await asyncio.create_subprocess_shell(
                    f"pgrep -f '{pat}' 2>/dev/null",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                out, _ = await proc.communicate()
                if out.decode().strip():
                    if demo_id not in running:
                        running.append(demo_id)
                    break

        elif dtype == "port_probe":
            port = plan.get("port")
            if port and await check_port_alive(port, timeout=0.5):
                if demo_id not in running:
                    running.append(demo_id)

    return {"running": running}

async def _teardown_plan(plan: dict):
    """Stop a demo per its teardown_strategy, then port-kill fallthrough.

    Shared by /stop and the /start pre-launch safety net so both free a port
    the same way (docker compose down for containers, pkill for processes,
    then fuser/lsof on the declared port as a universal fallthrough).
    """
    strategy = plan.get("teardown_strategy") or {}
    stype = strategy.get("type")

    if stype == "docker_compose":
        demo_path = DEMOS_DIR / plan["dir"]
        proc = await asyncio.create_subprocess_shell(
            "docker compose down 2>/dev/null || docker-compose down 2>/dev/null",
            cwd=demo_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
    elif stype == "process_pattern":
        # Resolve PIDs with pgrep via EXEC (no `/bin/sh -c` wrapper, so nothing but
        # pgrep itself carries the pattern in its argv, and pgrep self-excludes its
        # own PID), then SIGKILL by PID. A shell-wrapped `pkill -f '<pat>'` would put
        # the pattern into the wrapper sh's cmdline and let pkill match/kill that
        # wrapper; the exec form sidesteps the self-match entirely. The install
        # server's own PID is skipped defensively (its cmdline can't contain a demo
        # pattern, but belt-and-suspenders).
        import os, signal
        for pat in strategy.get("patterns", []):
            proc = await asyncio.create_subprocess_exec(
                "pgrep", "-f", "--", pat,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await proc.communicate()
            for tok in out.decode().split():
                try:
                    pid = int(tok)
                except ValueError:
                    continue
                if pid == os.getpid():
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

    # Universal fallthrough: kill whatever is LISTENING on the demo's port.
    # The lsof branch is scoped to -sTCP:LISTEN so it matches fuser's semantics
    # (listener only). Without it, `lsof -ti:{port}` also returns processes that
    # merely hold an OUTBOUND connection to that port — e.g. THIS install server,
    # which probes demo ports via check_port_alive/_running/_port_check — and
    # `xargs kill -9` would then SIGKILL the install server itself (friendly fire).
    # Every port the demo occupies, not just the user-facing one. construction and
    # translation serve a static front end on 8091/8092 but run their real back end
    # on 8191/8192; killing only the front end left the back end alive holding its
    # model, invisible to /running (which probes the declared port) until the next
    # launch happened to reclaim it.
    ports = [plan.get("port"), *(plan.get("extra_ports") or [])]
    for port in [p for p in ports if p]:
        await asyncio.create_subprocess_shell(
            f"fuser -k {port}/tcp 2>/dev/null || lsof -ti:{port} -sTCP:LISTEN | xargs kill -9 2>/dev/null || true",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )


@app.post("/stop/{demo_id}")
async def stop_demo(demo_id: str):
    """Stop a running demo via its declared teardown_strategy, then port-kill fallthrough."""
    plan = INSTALL_PLANS.get(demo_id)
    if not plan:
        return {"ok": False, "error": "Unknown demo"}
    await _teardown_plan(plan)
    return {"ok": True}

@app.get("/start/{demo_id}")
async def start_demo_stream(demo_id: str):
    """Start a demo and stream output via SSE."""
    plan = INSTALL_PLANS.get(demo_id)
    if not plan:
        async def err():
            yield sse_event({"type": "error", "msg": f"Unknown: {demo_id}"})
        return StreamingResponse(err(), media_type="text/event-stream")

    demo_path = DEMOS_DIR / plan["dir"]
    cmd = plan["launch_cmd"].format(dir=str(demo_path))

    async def start_gen():
        yield "retry: -1\n\n"  # disable browser auto-reconnect
        # Safety net (#14): before launching, free this demo's port by stopping
        # any OTHER demo that shares it. Makes a port collision impossible
        # server-side, independent of (possibly stale) client state or which
        # client path (loadAndOpen/startDemo) initiated the launch.
        target_port = plan.get("port")
        if target_port:
            for other_id, other_plan in INSTALL_PLANS.items():
                if other_id == demo_id:
                    continue
                if other_plan.get("port") == target_port:
                    yield sse_event({"type": "output", "line": f"Freeing port {target_port} — stopping {other_plan['name']}..."})
                    await _teardown_plan(other_plan)
        yield sse_event({"type": "output", "line": f"Starting {plan['name']}..."})
        # new_session=True so a client disconnect can't SIGTERM the launched demo (#15).
        # Keep the last few lines: on failure they are the only explanation the
        # operator gets, and the client renders them verbatim rather than
        # translating them into a reassuring "still loading" message.
        tail: deque[str] = deque(maxlen=20)
        exit_code = 0
        async for line in stream_command(cmd, demo_path, new_session=True):
            if isinstance(line, CommandExit):
                exit_code = line.code
                continue
            tail.append(line)
            yield sse_event({"type": "output", "line": line})

        if exit_code != 0:
            yield sse_event({
                "type": "error",
                "msg": f"{plan['name']} failed to start (launch command exited {exit_code}).",
                "exit_code": exit_code,
                "tail": list(tail),
            })
            return

        yield sse_event({"type": "complete", "port": plan.get("port"), "path": plan.get("port_path", "/")})

    return StreamingResponse(start_gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/load_time/{demo_id}")
async def record_load_time(demo_id: str, seconds: float):
    """Record a measured load time (frontend-reported elapsed seconds, click → live)."""
    LAST_LOAD_TIMES[demo_id] = float(seconds)
    _persist_times()
    return {"ok": True, "demo_id": demo_id, "seconds": float(seconds)}

@app.get("/load_times")
async def get_load_times():
    """Return all most-recent measured load times."""
    return LAST_LOAD_TIMES

def _tail_file(path: Path, lines: int) -> list[str]:
    """Last `lines` lines of a file, reading only the tail of it."""
    TAIL_BYTES = 64 * 1024
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - TAIL_BYTES))
        chunk = f.read()
    text = chunk.decode(errors="replace")
    # A partial first line is likely if we seeked into the middle of one.
    if size > TAIL_BYTES:
        text = text.split("\n", 1)[-1]
    return text.splitlines()[-lines:]


@app.get("/demo_log/{demo_id}")
async def demo_log(demo_id: str, lines: int = 40):
    """Serve whatever record a demo left of its own startup.

    This exists because the single most useful line the Demo Center ever
    produced — `nohup: failed to run command '.../envs/zgx/bin/python3': No such
    file or directory` — was written into a log file that nothing ever read, while
    the UI showed a progress spinner for ten minutes. Native demos redirect their
    stdout/stderr into LOG_DIR (declared as `log_file` in the plan); containerised
    ones keep it inside the container, so fall back to `docker compose logs`.
    """
    plan = INSTALL_PLANS.get(demo_id)
    if not plan:
        return {"ok": False, "source": None, "lines": [f"Unknown demo: {demo_id}"]}

    lines = max(1, min(lines, 200))
    log_name = plan.get("log_file")

    if log_name:
        path = LOG_DIR / log_name
        if not path.exists():
            return {"ok": False, "source": str(path),
                    "lines": [f"No log file yet at {path} — the demo may never have been started."]}
        try:
            return {"ok": True, "source": str(path), "lines": _tail_file(path, lines)}
        except OSError as e:
            return {"ok": False, "source": str(path), "lines": [f"Could not read {path}: {e}"]}

    if plan.get("teardown_strategy", {}).get("type") == "docker_compose":
        demo_path = DEMOS_DIR / plan["dir"]
        cmd = f"docker compose logs --no-color --tail={lines}"
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd, cwd=demo_path,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        except asyncio.TimeoutError:
            return {"ok": False, "source": cmd, "lines": ["docker compose logs timed out after 15s"]}
        except OSError as e:
            return {"ok": False, "source": cmd, "lines": [f"Could not run docker compose logs: {e}"]}
        text = out.decode(errors="replace").splitlines()
        return {"ok": bool(text), "source": f"{cmd} (in {demo_path})",
                "lines": text[-lines:] or ["(no container output — the container may never have been created)"]}

    return {"ok": False, "source": None,
            "lines": [f"{plan['name']} keeps no separate log — its launch output is streamed to the console above."]}


@app.get("/health")
async def health():
    return {"ok": True, "demos_dir": str(DEMOS_DIR)}


# ── Hardware truth ─────────────────────────────────────────────────────────
# The launcher's system bar used to be static HTML: "NVIDIA CTK ✓ v1.18.2",
# "GPU / Compute  GB10 · sm_121", "Unified Memory 128GB" -- typed by hand, read
# by nothing, and never re-checked. It was already wrong on this box (the
# toolkit is 1.20.0), and on an AMD box every one of those green ticks would
# render identically and be false.
#
# Detection is cached: it shells out to nvidia-smi/rocminfo, the launcher polls,
# and the answer cannot change without a reboot. Pass ?refresh=1 to re-probe.
_HW_CACHE: dict | None = None


def _detect_versions() -> dict:
    """Docker/Compose versions, so the launcher stops asserting those too."""
    out = {}
    for key, cmd in (
        ("docker", ["docker", "--version"]),
        ("compose", ["docker", "compose", "version", "--short"]),
    ):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=6, check=False)
            out[key] = proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else "unknown"
        except (OSError, subprocess.SubprocessError):
            out[key] = "unknown"
    return out


# ── Live hardware telemetry ────────────────────────────────────────────────
# Distinct from /sysinfo, which is cached identity that cannot change without a
# reboot. This one moves, so it has its own poller.
#
# It is served from HERE, on the host, because the browser is what reads it.
# ballistics, pii, pharmacy and maritime's AMD stack are python:3.12-slim
# containers with no nvidia-smi, no rocm-smi and no /dev/kfd; maritime's NVIDIA
# stack is on bridge networking and cannot even reach this port. The browser can
# always reach both the demo and the launcher, so every awkward case disappears.
_TELEMETRY_POLLER = None


def _telemetry_poller():
    global _TELEMETRY_POLLER
    if _TELEMETRY_POLLER is None:
        _TELEMETRY_POLLER = zgx_gpu_telemetry.GpuPoller(interval_s=1.0)
        _TELEMETRY_POLLER.start()
    return _TELEMETRY_POLLER


@app.get("/telemetry")
async def telemetry():
    """Live GPU + CPU/RAM. Never 500s; an unreadable field is null, never 0."""
    sample = dict(_telemetry_poller().get())

    # Unified-memory parts (GB10, Strix Halo) report no separate VRAM figure, so
    # the GPU memory fields come back null. Fall back to system RAM and SAY that
    # is what happened -- the alternative is a dash on a machine whose memory is
    # perfectly well known.
    if not sample.get("mem_total_gb"):
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        sample["mem_total_gb"] = round(int(line.split()[1]) / (1024 ** 2), 1)
                        sample["mem_source"] = "system (unified)"
                        break
        except (OSError, ValueError, IndexError):
            pass
        if sample.get("ram_used_gb") is not None and sample.get("mem_used_gb") is None:
            sample["mem_used_gb"] = sample["ram_used_gb"]
    else:
        sample["mem_source"] = sample.get("source", "driver")
    return sample


@app.get("/sysinfo")
async def sysinfo(refresh: int = 0):
    """What this machine actually is. Never 500s -- an unknown field says so."""
    global _HW_CACHE
    if _HW_CACHE is None or refresh:
        hw = zgx_hardware.detect()
        hw.update(_detect_versions())
        try:
            usage = shutil.disk_usage(str(Path.home()))
            hw["disk_free_gb"] = round(usage.free / (1024 ** 3))
            hw["disk_total_gb"] = round(usage.total / (1024 ** 3))
        except OSError:
            hw["disk_free_gb"] = hw["disk_total_gb"] = None
        _HW_CACHE = hw
    return _HW_CACHE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("ZGX_BIND", "0.0.0.0"), port=8086, log_level="info")
