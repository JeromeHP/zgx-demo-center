"""
Pipeline Pulse v6 — FastAPI backend
HP ZGX Nano G1n · O&G AI Developer Demo
Port 8087

Endpoints:
  GET  /                    → serves pipeline_pulse_v6.html
  WS   /ws/telemetry        → streams real GPU stats via pynvml every 2s
  POST /api/inference       → queues Ollama request, streams SSE response
  GET  /api/status          → session stats, queue depth, model info
  GET  /api/reset           → reset session counters
"""

import asyncio
import json
import os
import time
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn
import websockets

# ── GPU telemetry + psutil (self-contained, no Performance Lab dep) ────────
# Pipeline Pulse runs standalone — no dependency on port 8080.
#
# Telemetry comes from the shared vendor-neutral shim (nvidia-smi or rocm-smi)
# rather than a hardcoded pynvml path, so this demo reports real numbers on an
# AMD box instead of silently showing NVIDIA-shaped defaults.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zgx_metrics import TokenLedger  # noqa: E402

LEDGER = TokenLedger()
try:
    from zgx_gpu_telemetry import read as read_gpu
    TELEMETRY_OK = True
except Exception:
    TELEMETRY_OK = False
    def read_gpu(handle=None):
        return {"available": False, "vendor": "unknown"}

try:
    import pynvml
    pynvml.nvmlInit()
    _handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    NVML_OK = True
except Exception:
    _handle = None
    NVML_OK = False

try:
    import psutil
    PSUTIL_OK = True
except Exception:
    PSUTIL_OK = False

# ── hw/sys caches (filled by background poller below) ───────────────────────
# Seeded as unknown, not zero: before the first poll completes the dashboard
# must not claim 0 MHz and "idle" -- those are readings, and it has none yet.
_hw_cache = {
    "clock": None, "gpu_util": None, "power": None, "temp": None,
    "throttling": False, "reasons": "None", "workload": "unknown",
    "vendor": "unknown", "telemetry_source": "none",
}
_sys_cache = {"cpu": None, "ram_gb": None, "ram_pct": None}

def _poll_throttle_nvidia():
    """Throttle reasons, NVIDIA only — there is no rocm-smi equivalent.

    Returned separately from the core metrics so the AMD path reports
    "unavailable" for this one field rather than losing the whole sample.
    """
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks_throttle_reasons.active",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4)
        if r.returncode != 0:
            return {}
        thr_raw = r.stdout.strip()
        try:
            thr_val = int(thr_raw, 16)
            throttling = thr_val not in (0x0, 0x2)
            reasons = hex(thr_val) if throttling else "None"
        except ValueError:
            throttling = thr_raw.lower() not in ("not applicable", "none", "0x0000000000000000")
            reasons = thr_raw if throttling else "None"
        return {"throttling": throttling, "reasons": reasons}
    except Exception:
        return {}


def _poll_hw_once():
    """Read GPU hw stats from whichever vendor tool this box has."""
    sample = read_gpu(_handle)
    if not sample.get("available"):
        return {}
    util = sample.get("util_pct")
    out = {
        "gpu_util": util,
        "power": sample.get("power_w"),
        "temp": sample.get("temp_c"),
        "clock": sample.get("clock_mhz"),
        "vendor": sample.get("vendor", "unknown"),
        "telemetry_source": sample.get("source", "none"),
        # An unknown utilisation must not read as "idle" — that is a claim.
        "workload": ("active" if util > 5 else "idle") if util is not None else "unknown",
    }
    if sample.get("vendor") == "NVIDIA":
        out.update(_poll_throttle_nvidia())
    return out

def _poll_sys_once():
    """Read CPU and RAM via psutil."""
    if not PSUTIL_OK:
        return {}
    try:
        cpu = psutil.cpu_percent(interval=None)
        vm  = psutil.virtual_memory()
        return {
            "cpu":     round(cpu, 1),
            "ram_gb":  round(vm.used / 1e9, 1),
            "ram_pct": round(vm.percent, 1),
        }
    except Exception:
        return {}

async def poll_perf_lab_ws():
    """Background poller: nvidia-smi + psutil every 2s (self-contained).
    Retained same function name so startup_tasks() needs no changes."""
    if PSUTIL_OK:
        try:
            psutil.cpu_percent(interval=None)  # prime — first call returns 0.0
        except Exception:
            pass
    while True:
        hw = _poll_hw_once()
        if hw:
            _hw_cache.update(hw)
        sys_stats = _poll_sys_once()
        if sys_stats:
            _sys_cache.update(sys_stats)
        await asyncio.sleep(2)


def get_gpu_memory() -> dict:
    """GPU memory, with an `ollama ps` proxy when the driver reports used=0.

    This used to be get_mem_from_pynvml(), and it INVENTED its answers. With no
    NVML it returned a flat {"mem_used_gb": 102.0, "mem_total_gb": 128.0}; the
    total defaulted to 128.0 regardless of the machine, and the caller's
    .get(..., 102.0) supplied 102 again if anything else went wrong. So on any
    box without NVML -- every AMD box -- the demo displayed "102 GB / 128 GB"
    as live telemetry, while the UI tooltip told the audience the figure was
    "sourced from the NVIDIA management library running on the Nano and updated
    once per second".

    Now: None means unknown, and the UI renders unknown as "--". The `ollama ps`
    proxy is kept -- it is legitimate, and on unified-memory parts where the
    driver reports 0 it is the only honest signal available -- but it is
    labelled as a proxy in the payload rather than passed off as a driver read.
    """
    mem_stats = {"mem_used_gb": None, "mem_total_gb": None, "mem_source": "unavailable"}

    sample = read_gpu(_handle)
    if sample.get("mem_total_gb"):
        mem_stats["mem_total_gb"] = sample["mem_total_gb"]
        mem_stats["mem_source"] = sample.get("source", "driver")
    if sample.get("mem_used_gb"):
        mem_stats["mem_used_gb"] = sample["mem_used_gb"]

    # Unified-memory parts (GB10, Strix Halo) commonly report total but not a
    # meaningful used figure. Fall back to the system total so the denominator
    # is real rather than a constant.
    if mem_stats["mem_total_gb"] is None:
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        mem_stats["mem_total_gb"] = round(int(line.split()[1]) / (1024 ** 2), 1)
                        mem_stats["mem_source"] = "system (unified)"
                        break
        except (OSError, ValueError, IndexError):
            pass

    mem_used = 0

    # Fallback: parse `ollama ps` for loaded model sizes.
    # Real output: "llama3.1:70b  711a9e8463af  102 GB  100% GPU  131072  Forever"
    # "102 GB" is two tokens, so scan for GB/MB unit tokens and read parts[i-1].
    if mem_used == 0:
        try:
            result = subprocess.run(['ollama', 'ps'], capture_output=True, text=True, timeout=3)
            lines = result.stdout.strip().split('\n')
            total_gb = 0.0
            for line in lines[1:]:  # skip header
                parts = line.split()
                for i, tok in enumerate(parts):
                    if tok in ('GB', 'GiB') and i > 0:
                        try:
                            total_gb += float(parts[i-1])
                        except ValueError:
                            pass
                    elif tok in ('MB', 'MiB') and i > 0:
                        try:
                            total_gb += float(parts[i-1]) / 1024.0
                        except ValueError:
                            pass
            if total_gb > 0:
                mem_used = total_gb * 1e9
                mem_stats["mem_source"] = "ollama ps (proxy)"
        except Exception:
            pass

    if mem_used:
        mem_stats["mem_used_gb"] = round(mem_used / 1e9, 1)
    return mem_stats

def get_gpu_stats() -> dict:
    """Unified telemetry for the dashboard.

    Every field that cannot be read is None, not a stand-in value. The previous
    version used `or 0` throughout and a literal "nvml_ok": True, so a box with
    no telemetry at all reported 0 MHz / 0 W / 0 C / idle and asserted that the
    NVIDIA management library was working. Zero is a reading; None is not.
    """
    mem = get_gpu_memory()
    return {
        "gpu_util":         _hw_cache.get("gpu_util"),
        "clock_mhz":        _hw_cache.get("clock"),
        "power_w":          _hw_cache.get("power"),
        "temp_c":           _hw_cache.get("temp"),
        "throttling":       bool(_hw_cache.get("throttling", False)),
        "throttle_reasons": _hw_cache.get("reasons") or ("n/a" if _hw_cache.get("vendor") == "AMD" else "None"),
        "workload":         _hw_cache.get("workload", "unknown"),
        "cpu_pct":          _sys_cache.get("cpu"),
        "ram_gb":           _sys_cache.get("ram_gb"),
        "ram_pct":          _sys_cache.get("ram_pct"),
        "mem_used_gb":      mem.get("mem_used_gb"),
        "mem_total_gb":     mem.get("mem_total_gb"),
        "mem_source":       mem.get("mem_source", "unavailable"),
        "gpu_vendor":       _hw_cache.get("vendor", "unknown"),
        "telemetry_source": _hw_cache.get("telemetry_source", "none"),
        "telemetry_ok":     bool(_hw_cache.get("telemetry_source", "none") != "none"),
        "nvml_ok":          bool(NVML_OK),
    }

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Pipeline Pulse v6")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:70b"
HTML_FILE    = Path(__file__).parent / "pipeline_pulse_v6.html"

# ── Session state ────────────────────────────────────────────────────────────
session = {
    "incident_count": 0,
    "cloud_calls":    0,          # always stays 0 — the point
    "last_latency_s": None,
    "queue_depth":    0,
}

# ── Startup: warm the model + kick off Performance Lab telemetry poller ────
@app.on_event("startup")
async def startup_tasks():
    # 1) Warm llama3.1:70b so the first real inference isn't cold
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            await client.post(
                "http://localhost:11434/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": "Ready.", "stream": False},
            )
    except Exception:
        pass
    # 2) Start background poller for ws://localhost:8080/ws (Performance Lab)
    asyncio.create_task(poll_perf_lab_ws())

# ── Inference queue (serialize all Ollama requests) ──────────────────────────
_infer_lock = asyncio.Lock()

async def run_inference(sensor: str, reading: str, context: str, prompt: str,
                         sev: str = "warning", detector: str = "anomaly detector"):
    """Queue-serialized call to Ollama. Yields text chunks."""
    async with _infer_lock:
        session["queue_depth"] += 1
        t0 = time.time()
        first_token = True
        try:
            threshold_word = "alert threshold" if sev == "critical" else "warning threshold"
            full_prompt = (
                "You are an oil & gas operations AI. A sensor has triggered:\n"
                f"sensor: {sensor}\n"
                f"value: {reading}\n"
                f"threshold crossed: {threshold_word}\n"
                f"model: {detector}\n\n"
                "Reply in exactly three lines, each labeled:\n"
                "**Severity:** <one word: Low, Medium, High, or Critical>\n"
                "**Anomaly:** <one sentence describing what the data suggests>\n"
                "**Recommendation:** <one sentence on immediate operator action>"
            )
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", OLLAMA_URL, json={
                    "model": OLLAMA_MODEL,
                    "prompt": full_prompt,
                    "stream": True,
                    "options": {
                        "num_predict": 110,
                        "temperature": 0.4,
                    },
                }) as resp:
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        token = chunk.get("response", "")
                        if token:
                            if first_token:
                                session["last_latency_s"] = round(time.time() - t0, 2)
                                first_token = False
                            yield token
                        if chunk.get("done"):
                            # The final chunk carries Ollama's real counts --
                            # prompt_eval_count, eval_count and eval_duration --
                            # and this loop used to `break` straight past them.
                            # The UI then displayed a client-side chars/4 guess
                            # as "tokens processed locally", which is a
                            # plausible invented number of exactly the kind this
                            # project keeps removing. Record the real one.
                            LEDGER.record(chunk, fallback_text="", wall_s=time.time() - t0)
                            break
            session["incident_count"] += 1
        except Exception as e:
            yield f"[Inference error: {e}]"
        finally:
            session["queue_depth"] = max(0, session["queue_depth"] - 1)

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_html():
    if HTML_FILE.exists():
        return HTMLResponse(content=HTML_FILE.read_text(), status_code=200)
    return HTMLResponse("<h1>pipeline_pulse_v6.html not found</h1>", status_code=404)

@app.get("/api/metrics")
async def api_metrics():
    """Session token totals for the shared chrome — real counts from Ollama."""
    return LEDGER.snapshot()


@app.get("/api/status")
async def api_status():
    gpu = get_gpu_stats()
    return JSONResponse({**session, **gpu, "model": OLLAMA_MODEL, "ollama_url": OLLAMA_URL})

@app.get("/api/reset")
async def api_reset():
    session["incident_count"] = 0
    session["last_latency_s"] = None
    return JSONResponse({"reset": True})

@app.post("/api/inference")
async def api_inference(request: Request):
    """SSE endpoint — streams Ollama response tokens."""
    body = await request.json()
    sensor   = body.get("sensor", "Unknown sensor")
    reading  = body.get("reading", "")
    context  = body.get("context", "")
    prompt   = body.get("prompt", f"Analyze this anomaly: {sensor} reading {reading}.")
    sev      = body.get("sev", "warning")
    detector = body.get("detector", "anomaly detector")

    async def event_stream():
        yield f"data: {json.dumps({'type': 'queued', 'queue_depth': session['queue_depth'] + 1})}\n\n"
        async for token in run_inference(sensor, reading, context, prompt, sev, detector):
            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'latency_s': session['last_latency_s'], 'count': session['incident_count']})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    """Streams real GPU telemetry every 2 seconds."""
    await ws.accept()
    try:
        while True:
            stats = get_gpu_stats()
            stats["ts"] = time.time()
            await ws.send_json(stats)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass

# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("pipeline_pulse_v6:app", host="0.0.0.0", port=int(os.environ.get("PP_PORT", 8087)), reload=False)
