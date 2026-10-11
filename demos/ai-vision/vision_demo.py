"""HP ZGX Nano — AI Vision Demo (port 8094).

FastAPI app that wires:
  - vision_engine (TRT YOLOv8s FP16 inference, 6 streams, 4 scenarios)
  - vision_hw (GPU telemetry via nvidia-smi)
  - Ollama llama3.1:70b for incident-brief generation
  - vision_frontend (single-string HTML/CSS/JS)

The TRT engine and Ollama both consume GPU memory — bidirectional conflict with
the Performance Lab is handled at the Demo Center level (not here).
"""
import asyncio
import json
import os
import time
from collections import deque
from datetime import datetime

import httpx
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from vision_engine import VisionEngine, SCENARIOS, COCO
from vision_hw import GpuPoller
from vision_frontend import HTML

from zgx_metrics import TokenLedger  # noqa: E402

LEDGER = TokenLedger()


PORT = int(os.environ.get("VISION_PORT", "8094"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:70b")

app = FastAPI()

# ── State ────────────────────────────────────────────────────────────────────
engine: VisionEngine = None  # set in startup
hw_poller: GpuPoller = None
alert_history = deque(maxlen=50)
alert_history_lock = asyncio.Lock()
pending_alerts: asyncio.Queue = None  # set in startup
ollama_lock: asyncio.Lock = None      # set in startup
scenario_switch_lock: asyncio.Lock = None  # set in startup; serializes /api/scenario switches
_pending_scenario = None  # latest target requested while a switch is in flight (coalesced)


# ── Alert wiring ─────────────────────────────────────────────────────────────
def _on_alert_sync(stream_id, stream_label, rule_name, message, class_counts):
    """Called from AlertEngine thread. Push onto the asyncio queue (thread-safe)."""
    payload = {
        "id": f"alert-{int(time.time()*1000)}",
        "stream_id": stream_id,
        "stream_label": stream_label,
        "rule": rule_name,
        "message": message,
        "class_counts": class_counts,
        "scenario": engine.current_scenario if engine else None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if pending_alerts is not None:
        try:
            pending_alerts.put_nowait(payload)
        except asyncio.QueueFull:
            pass


# ── Lifecycle ────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def _startup():
    global engine, hw_poller, pending_alerts, ollama_lock, scenario_switch_lock
    pending_alerts = asyncio.Queue(maxsize=64)
    ollama_lock = asyncio.Lock()
    scenario_switch_lock = asyncio.Lock()
    hw_poller = GpuPoller()
    hw_poller.start()
    engine = VisionEngine(on_alert=_on_alert_sync)
    engine.start_scenario("traffic")
    print(f"  [vision] startup complete — http://0.0.0.0:{PORT}", flush=True)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    # no-store: the kiosk runs a persistent chromium profile and the launcher opens
    # :8094/ with no cache-buster — without this, an OLD cached build of this page
    # (stale JS) gets served after a code update, silently masking frontend fixes.
    # The whole UI (HTML+CSS+JS) is inline in HTML, so one no-store covers everything.
    return HTMLResponse(HTML, headers={
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    })


@app.get("/api/scenarios")
async def api_scenarios():
    out = []
    for sid, cfg in SCENARIOS.items():
        out.append({
            "id": sid,
            "label": cfg["label"],
            "description": cfg["description"],
            "icon": cfg["icon"],
            "stream_count": cfg["stream_count"],
            "grid": cfg["grid"],
            "relevant_classes": cfg["relevant_classes"],
            "relevant_class_names": [COCO[c] for c in cfg["relevant_classes"]],
        })
    return JSONResponse({"scenarios": out, "active": engine.current_scenario if engine else None})


async def _switch_scenario_coalesced(scenario_id: str):
    """Serialize scenario switches behind scenario_switch_lock — one at a time.
    If more requests arrive WHILE a switch is running, only the LATEST target is
    remembered and applied when the in-flight switch finishes (coalesced), so a
    burst of clicks can never stack multiple back-to-back switches.

    start_scenario is sync and joins worker threads (settle + first-frame gate, up
    to a few seconds) — always run it off the event loop so /ws/frames keeps ticking
    and observes the engine's _switching pause."""
    global _pending_scenario
    if scenario_switch_lock.locked():
        _pending_scenario = scenario_id   # a switch is running — keep only the newest target
        return
    async with scenario_switch_lock:
        target = scenario_id
        while target is not None:
            _pending_scenario = None
            await asyncio.get_event_loop().run_in_executor(None, engine.start_scenario, target)
            target = _pending_scenario     # a newer target arrived mid-switch? apply it next

@app.post("/api/scenario/{scenario_id}")
async def api_scenario_switch(scenario_id: str):
    if scenario_id not in SCENARIOS:
        return JSONResponse({"ok": False, "error": "unknown scenario"}, status_code=404)
    await _switch_scenario_coalesced(scenario_id)
    return JSONResponse({"ok": True, "scenario": scenario_id})


@app.get("/api/metrics")
async def api_metrics():
    """Session token totals for the shared chrome."""
    return LEDGER.snapshot()


@app.get("/api/status")
async def api_status():
    metrics = engine.aggregate_metrics() if engine else {}
    hw = hw_poller.get() if hw_poller else {}
    return JSONResponse({
        "ok": True,
        "scenario": metrics.get("scenario"),
        "stream_count": metrics.get("stream_count", 0),
        "total_qps": metrics.get("total_qps", 0),
        "lat_p50_ms": metrics.get("lat_p50_ms", 0),
        "lat_p99_ms": metrics.get("lat_p99_ms", 0),
        "class_counts": metrics.get("class_counts", {}),
        "cloud_calls": 0,
        # Which detection backend is actually live. A run on the ONNX CPU
        # provider is several times slower than TensorRT FP16; showing the
        # header's old fixed "YOLOv8s TRT FP16" over it would misreport both
        # the stack and the performance.
        "backend_tier": getattr(engine, "backend_tier", "unknown") if engine else "unknown",
        "backend_provider": getattr(engine, "backend_provider", "unknown") if engine else "unknown",
        "backend_status": getattr(engine, "backend_status", "") if engine else "",
        "hw": hw,
        "model": OLLAMA_MODEL,
        "ollama_url": OLLAMA_URL,
    })


@app.websocket("/ws/frames")
async def ws_frames(ws: WebSocket):
    """Multiplex all active stream JPEGs over one WebSocket.

    Frames as binary messages: byte 0 = stream_id (0..N-1), bytes 1.. = JPEG.
    Per-stream _seq tracking skips re-sending identical frames. Survives
    scenario switches — the engine swaps StreamState objects in place; we
    just keep reading from the snapshot.
    """
    await ws.accept()
    last_sent = {}  # sid -> last seq sent
    try:
        while True:
            if engine is not None:
                if engine.is_switching():
                    # Old StreamState objects are being torn down; drop our seq cache so
                    # the first frame after the new readers come up always sends.
                    last_sent.clear()
                    await asyncio.sleep(0.05)
                    continue
                states = engine.get_states_snapshot()
                sends = []
                new_seqs = {}
                for state in states:
                    sid = state.sid
                    with state.lock:
                        seq = state._seq
                        jpeg = state.jpeg
                    if jpeg is not None and seq != last_sent.get(sid, -1):
                        sends.append(ws.send_bytes(bytes([sid]) + jpeg))
                        new_seqs[sid] = seq
                if sends:
                    await asyncio.gather(*sends)
                    for sid, seq in new_seqs.items():
                        last_sent[sid] = seq
            await asyncio.sleep(0.008)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"  [ws/frames] err: {e}", flush=True)


@app.websocket("/ws")
async def ws_telemetry(ws: WebSocket):
    """Telemetry WebSocket — bidirectional.

    Server pushes periodic telemetry. Client may send JSON commands; currently
    only {"switch": "<scenario_id>"} is supported. We send back
    {"switched": true, "scenario": "<id>"} on completion. The two directions
    run as concurrent tasks; a send-lock serializes outbound writes since
    Starlette WebSocket sends are not safe under concurrent access.
    """
    await ws.accept()
    send_lock = asyncio.Lock()

    async def send_telemetry_loop():
        while True:
            metrics = engine.aggregate_metrics() if engine else {}
            hw = hw_poller.get() if hw_poller else {}
            # Drain pending alerts onto the wire and into history
            new_alerts = []
            while not pending_alerts.empty():
                try:
                    a = pending_alerts.get_nowait()
                    new_alerts.append(a)
                    async with alert_history_lock:
                        alert_history.appendleft(a)
                except asyncio.QueueEmpty:
                    break
            async with send_lock:
                await ws.send_json({
                    "type": "telemetry",
                    "metrics": metrics,
                    "hw": hw,
                    "new_alerts": new_alerts,
                    # Live switch state so the frontend can drop a stuck loading overlay
                    # even if the first-frame signal never arrives (watchdog-aborted switch).
                    "switching": engine.is_switching() if engine else False,
                    "switch_stalled": bool(getattr(engine, "_switch_stalled", False)) if engine else False,
                })
            await asyncio.sleep(1.0)

    async def recv_command_loop():
        while True:
            text = await ws.receive_text()
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                continue
            switch_id = msg.get("switch")
            if not switch_id:
                continue
            if switch_id not in SCENARIOS:
                async with send_lock:
                    await ws.send_json({"switched": False, "error": "unknown scenario", "scenario": switch_id})
                continue
            # Serialize + coalesce-to-latest (same path as POST /api/scenario).
            await _switch_scenario_coalesced(switch_id)
            async with send_lock:
                await ws.send_json({"switched": True, "scenario": switch_id})

    try:
        await asyncio.gather(send_telemetry_loop(), recv_command_loop())
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"  [ws] err: {e}", flush=True)


# ── LLM alert briefing ───────────────────────────────────────────────────────
def _build_prompt(scenario_label, stream_id, alert_message, class_counts, timestamp):
    cc = ", ".join(f"{k}={v}" for k, v in sorted(class_counts.items())) or "(none)"
    return (
        "You are a security monitoring AI. Based on detection data from camera feeds, "
        "generate a brief (~80 word) incident report.\n\n"
        f"Scenario: {scenario_label}\n"
        f"Camera: Stream {stream_id}\n"
        f"Alert: {alert_message}\n"
        f"Detection counts: {cc}\n"
        f"Timestamp: {timestamp}\n\n"
        "Generate a professional, actionable incident brief."
    )


@app.post("/api/alert")
async def api_alert(req: Request):
    """Generate an LLM brief for a triggered alert. SSE streams tokens to the client."""
    body = await req.json()
    scenario = body.get("scenario", engine.current_scenario)
    scen_cfg = SCENARIOS.get(scenario, {})
    scenario_label = scen_cfg.get("label", scenario or "Unknown")
    stream_id = body.get("stream_id", -1)
    message = body.get("message", "Detection threshold exceeded")
    class_counts = body.get("class_counts", {})
    timestamp = body.get("timestamp", datetime.now().isoformat(timespec="seconds"))
    prompt = _build_prompt(scenario_label, stream_id, message, class_counts, timestamp)

    async def gen():
        # Serialize all LLM calls — only one stream at a time
        async with ollama_lock:
            yield f"event: start\ndata: {json.dumps({'stream_id': stream_id})}\n\n"
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True,
                "keep_alive": -1,
                "options": {"num_predict": 160, "temperature": 0.4},
            }
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, read=120.0)) as client:
                    async with client.stream("POST", f"{OLLAMA_URL}/api/generate", json=payload) as resp:
                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            try:
                                chunk = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            tok = chunk.get("response", "")
                            if tok:
                                yield f"event: token\ndata: {json.dumps({'text': tok})}\n\n"
                            if chunk.get("done"):
                                # Ollama's real counts ride on this chunk and
                                # were being returned past. This demo's LLM only
                                # fires on an alert, so the meter sits at "—"
                                # between incidents -- which is honest, and far
                                # better than a running estimate.
                                LEDGER.record(chunk, fallback_text="")
                                yield f"event: done\ndata: {json.dumps({'stream_id': stream_id})}\n\n"
                                return
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/alerts/history")
async def api_alerts_history():
    async with alert_history_lock:
        return JSONResponse({"alerts": list(alert_history)})


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
