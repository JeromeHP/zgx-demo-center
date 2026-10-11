"""
Port-of-Entry Vehicle Inspection Analyst
Powered by HP ZGX Nano AI Station

A Vision Language Model demo for on-prem damage triage of finished vehicles
offloaded from RoRo carriers at a port of destination. One image in, one
structured inspection report out, with a recommended terminal disposition.

Uses Mistral Small 3.2 (24B, vision) served by host Ollama over its OpenAI-compatible
/v1/chat/completions route. Single multimodal call -- the VLM emits labeled
plain-text lines that we parse into a structured report (this pattern is more
reliable than JSON schema for VLM output).

Demo Center port: 8097. Upstream ran vLLM inside the container and published
Grant Machine's port; both collided with this suite (and a vLLM per demo cannot
coexist with Maritime's). See README.md.

Compliance by Architecture: imagery (potentially commercially sensitive
pre-launch vehicles), inference, and inspector output all stay on the
device. Zero cloud dependency.
"""

import os
import io
import json
import asyncio
import random
import string
import time
import logging
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import httpx

# Session token accounting for the shared ZGX chrome (demos/ui/README.md).
try:  # noqa: E402
    from zgx_metrics import TokenLedger
except ImportError:  # pragma: no cover - depends on how the container launches
    # `uvicorn backend.main:app` puts /app on sys.path, not /app/backend, so the
    # plain import fails there. Same fallback as ballistics and maritime.
    from backend.zgx_metrics import TokenLedger

LEDGER = TokenLedger()

try:  # noqa: E402 -- same two launch styles as the zgx_metrics import above
    from inspection import (DISPOSITION_PLAYBOOK, INSPECTION_PROMPT, LANE_IMAGE_MAX, PORTS,
                            REPORT_MAX_TOKENS, SEVERITY_TIERS, TRIAGE_MAX_TOKENS, classify_severity,
                            format_prompt, parse_area_percent, parse_photo_findings, parse_vlm_output,
                            prepare_image_for_vlm, report_prompt, resolve_disposition, triage_prompt)
    from lane import LaneManager
except ImportError:  # pragma: no cover
    from backend.inspection import (DISPOSITION_PLAYBOOK, INSPECTION_PROMPT, LANE_IMAGE_MAX, PORTS,
                                    REPORT_MAX_TOKENS, SEVERITY_TIERS, TRIAGE_MAX_TOKENS,
                                    classify_severity, format_prompt, parse_area_percent,
                                    parse_photo_findings, parse_vlm_output, prepare_image_for_vlm,
                                    report_prompt, resolve_disposition, triage_prompt)
    from backend.lane import LaneManager


# -- Logging setup -----------------------------------------------------------
# Robust logging from the start: every stage logs INFO with timing,
# failures get full tracebacks. No silent failures.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vehicle-inspection")

# -- Configuration -----------------------------------------------------------
# 0.0.0.0, not the tailnet IP: the install API's readiness probe hits
# 127.0.0.1, and tools/lockdown-tailnet.sh filters at the network layer instead.
HOST = os.environ.get("HOST", "0.0.0.0")
# 8097 is this demo's slot in the Demo Center port map (RUNBOOK.md). The
# container runs with network_mode: host, so this is the HOST port -- upstream's
# container-internal 8000 would land on the shared maritime/pharmacy/pii port.
PORT = int(os.environ.get("PORT", 8097))
# Host Ollama, reached over loopback thanks to host networking.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral-small3.2")
INFERENCE_TIMEOUT = float(os.environ.get("INFERENCE_TIMEOUT", "300"))
# 0, not upstream's 0.3. An inspection verdict must not depend on the dice: at
# 0.3 the same showroom-clean photo came back NONE on one run and MAJOR on the
# next. Greedy decoding gives the presenter the same report every time.
INFERENCE_TEMPERATURE = float(os.environ.get("INFERENCE_TEMPERATURE", "0"))
# What the audience reads. OLLAMA_MODEL is `vehicle-inspector`, a context-capped
# registration of mistral-small3.2 (Modelfile.vehicle); the label keeps the
# real model name on screen.
MODEL_LABEL = os.environ.get("MODEL_LABEL", OLLAMA_MODEL)

# -- FastAPI app + lifespan --------------------------------------------------
# Use a lifespan handler rather than the deprecated @app.on_event hooks.

FRONTEND_DIR = os.environ.get("FRONTEND_DIR", "/app/frontend")
SAMPLES_DIR = os.environ.get("SAMPLES_DIR", "/app/sample-images")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    host_ip = os.environ.get("HOST_IP", "")
    banner = "\n" + "=" * 66 + "\n"
    banner += "  Port-of-Entry Vehicle Inspection Analyst\n"
    banner += f"  {MODEL_LABEL} | Ollama\n"
    banner += "  Compliance by Architecture: 100% on-prem inference\n"
    banner += "=" * 66 + "\n"
    if host_ip:
        banner += f"\n  \u27a1  http://{host_ip}:{PORT}\n"
    else:
        banner += f"\n  \u27a1  http://localhost:{PORT}\n"
    banner += "=" * 66 + "\n"
    print(banner)
    logger.info("Service started | model=%s | endpoint=%s", OLLAMA_MODEL, OLLAMA_BASE_URL)
    app.state.http_client = httpx.AsyncClient(timeout=INFERENCE_TIMEOUT)
    try:
        yield
    finally:
        # Shutdown
        await app.state.http_client.aclose()
        logger.info("Service shutting down")


app = FastAPI(
    title="Port-of-Entry Vehicle Inspection Analyst",
    description="VLM-powered damage triage for RoRo vehicle imports",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path = Path(FRONTEND_DIR)
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")
else:
    logger.warning("Frontend directory not found at %s", FRONTEND_DIR)

samples_path = Path(SAMPLES_DIR)
if samples_path.exists():
    app.mount("/samples", StaticFiles(directory=samples_path), name="samples")
else:
    logger.warning("Samples directory not found at %s", SAMPLES_DIR)

REPLAY_DIR = Path(os.environ.get("REPLAY_DIR", "/app/data/replays"))


# -- Inference ---------------------------------------------------------------

def _vlm_payload(images_b64: list[str], prompt: str, max_tokens: int, stream: bool) -> dict:
    """One user message: every image first (portal order), then the prompt,
    which refers to them as Photo 1..N in the same order."""
    content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
               for b64 in images_b64]
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": INFERENCE_TEMPERATURE,
        # No thinking control: the model must be a NON-thinking one. On Ollama
        # 0.34 no flag (reasoning_effort "none", think:false on /v1 or
        # /api/chat) stopped qwen3-vl:8b from reasoning, which ate the whole
        # token budget and left content empty. mistral-small3.2 does not think.
    }
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return payload


async def query_vlm(image_b64: str, prompt: str, max_tokens: int = REPORT_MAX_TOKENS) -> tuple[str, dict]:
    """Send a single image + prompt to the VLM. Returns (text, usage_dict)."""
    payload = _vlm_payload([image_b64], prompt, max_tokens, stream=False)

    t0 = time.time()
    try:
        logger.info("Submitting VLM call (max_tokens=%d)...", max_tokens)
        response = await app.state.http_client.post(
            f"{OLLAMA_BASE_URL}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        elapsed = time.time() - t0
        LEDGER.record(data, fallback_text=content, wall_s=elapsed)
        logger.info(
            "VLM call complete in %.2fs (prompt=%d, completion=%d, total=%d)",
            elapsed,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("total_tokens", 0),
        )
        return content, usage
    except httpx.HTTPStatusError as e:
        logger.error("Inference HTTP error %d: %s", e.response.status_code, e.response.text[:500])
        raise
    except Exception as e:
        logger.error("VLM call failed after %.2fs: %s", time.time() - t0, e)
        logger.debug(traceback.format_exc())
        raise


async def stream_vlm(images_b64: list[str], prompt: str, max_tokens: int, on_token) -> tuple[str, float, dict]:
    """Streamed variant for the lane: hands each text chunk to `on_token` as
    it arrives. Returns (text, wall seconds, usage). Cancelling the calling
    task closes the HTTP stream, which is what stops Ollama generating."""
    payload = _vlm_payload(images_b64, prompt, max_tokens, stream=True)
    t0 = time.time()
    first = None
    parts: list[str] = []
    usage: dict = {}
    async with app.state.http_client.stream("POST", f"{OLLAMA_BASE_URL}/chat/completions", json=payload) as resp:
        if resp.status_code >= 400:
            body = (await resp.aread()).decode(errors="replace")[:300]
            raise RuntimeError(f"inference server returned {resp.status_code}: {body}")
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    if first is None:
                        first = time.time()
                    parts.append(piece)
                    await on_token(piece)
    text = "".join(parts).strip()
    total = time.time() - t0
    # tok/s for the chrome from decode time only; the image prefill before the
    # first token is not generation and would understate the rate.
    LEDGER.record({"usage": usage} if usage else None, fallback_text=text,
                  wall_s=(time.time() - first) if first else total)
    logger.info("Streamed VLM call %.2fs (completion=%s)", total, usage.get("completion_tokens"))
    return text, total, usage


def _load_views(views: list[tuple[Path, str]]) -> list[str]:
    out = []
    for path, _angle in views:
        with Image.open(path) as im:
            out.append(prepare_image_for_vlm(im, max_dim=LANE_IMAGE_MAX))
    return out


async def lane_triage(views: list[tuple[Path, str]], port_key: str, on_token) -> tuple[str, float, dict]:
    """Every portal view of one vehicle in a single call."""
    images = await asyncio.to_thread(_load_views, views)
    prompt = format_prompt(triage_prompt([a for _, a in views]), port_key)
    return await stream_vlm(images, prompt, TRIAGE_MAX_TOKENS, on_token)


async def lane_report(views: list[tuple[Path, str]], port_key: str, on_token) -> tuple[str, float, dict]:
    images = await asyncio.to_thread(_load_views, views)
    prompt = format_prompt(report_prompt([a for _, a in views]), port_key)
    return await stream_vlm(images, prompt, REPORT_MAX_TOKENS, on_token)


LANE = LaneManager(samples_dir=samples_path, replay_dir=REPLAY_DIR, infer_triage=lane_triage,
                   infer_report=lane_report, model_label=MODEL_LABEL)


# -- Helpers -----------------------------------------------------------------

def generate_report_id() -> str:
    """Generate a Vehicle Port Inspection (VPI) report ID."""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    seq = "".join(random.choices(string.digits, k=4))
    return f"VPI-{date_str}-{seq}"


# -- Main pipeline -----------------------------------------------------------

async def run_inspection(
    image: Image.Image,
    port_key: str,
    vessel: str,
    vin: str,
) -> dict:
    """Full inspection pipeline.

    Steps:
      1. Preprocess image to base64.
      2. Build port context and assemble final prompt.
      3. Single VLM call.
      4. Parse labeled output into structured fields.
      5. Classify severity and resolve disposition.
      6. Assemble report.
    """
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Starting inspection | port=%s | vessel=%s | vin=%s", port_key, vessel or "n/a", vin or "n/a")

    # 1. Preprocess
    image_b64 = prepare_image_for_vlm(image)

    # 2. Build prompt
    port = PORTS.get(port_key, PORTS["baltimore"])
    final_prompt = format_prompt(INSPECTION_PROMPT, port_key)

    # 3. VLM call
    raw_output, usage = await query_vlm(image_b64, final_prompt)
    logger.debug("Raw VLM output:\n%s", raw_output)

    # 4. Parse
    parsed = parse_vlm_output(raw_output)

    # 5. Classify + disposition
    severity_key, severity_data = classify_severity(parsed)
    area_pct = parse_area_percent(parsed)
    disposition = resolve_disposition(parsed, severity_key)

    # 6. Assemble report
    report = {
        "report_id": generate_report_id(),
        "classification": "FOR DEMONSTRATION PURPOSES ONLY",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%d %b %Y %H%MZ").upper(),
        "port": {
            "key": port_key,
            "name": port["name"],
            "code": port["code"],
            "terminal": port["terminal"],
            "state": port["state"],
        },
        "vessel": vessel or "Not provided",
        "vin": vin or "Not provided",
        "vehicle": {
            "type": parsed["vehicle_type"],
            "view_angle": parsed["view_angle"],
        },
        "assessment": {
            "severity_key": severity_key,
            "severity_label": severity_data["label"],
            "severity_color": severity_data["color_hex"],
            "severity_description": severity_data["description"],
            "affected_panels": parsed["affected_panels"],
            "damage_observations": parsed["damage_observations"],
            "affected_area_percent": area_pct,
            "inspector_summary": parsed["inspector_summary"],
            # The grounding line the model wrote before grading (PHOTO_1).
            "photo_finding": parse_photo_findings(raw_output, ["photo"]).get("photo"),
        },
        "disposition": disposition,
        "raw_vlm_output": raw_output,
        "token_usage": usage,
        "pipeline_seconds": round(time.time() - t0, 2),
    }

    logger.info(
        "Inspection complete in %.2fs | severity=%s | disposition=%s | area=%s%%",
        report["pipeline_seconds"],
        severity_key,
        disposition["key"],
        area_pct if area_pct is not None else "n/a",
    )
    logger.info("=" * 60)
    return report


# -- API routes --------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = Path(FRONTEND_DIR) / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>Vehicle Inspection Analyst</h1><p>index.html not found.</p>")


@app.get("/api/metrics")
async def api_metrics():
    """Session token totals for the shared chrome."""
    return LEDGER.snapshot()


@app.get("/api/health")
async def health_check():
    """Health check. Probes the inference endpoint AND that our model is served.

    `{base}/models` is the OpenAI-compatible listing Ollama serves. "Endpoint
    up" alone is not "demo ready": a box where `ollama pull` never ran answers
    the listing fine and then fails every inspection, so the model must be in it.
    """
    endpoint_ok = False
    model_ok = False
    error = None
    try:
        resp = await app.state.http_client.get(f"{OLLAMA_BASE_URL.rstrip('/')}/models", timeout=5.0)
        endpoint_ok = resp.status_code == 200
        if endpoint_ok:
            served = {m.get("id") for m in (resp.json().get("data") or [])}
            # Ollama lists "name:tag"; an untagged name means ":latest".
            model_ok = OLLAMA_MODEL in served or f"{OLLAMA_MODEL}:latest" in served
            if not model_ok:
                error = f"{OLLAMA_MODEL} is not registered in Ollama (see install_plans.py: ollama create)"
    except Exception as e:
        error = str(e)

    return {
        # Marker the container healthcheck greps for, so a passing probe is
        # proof of THIS service and not whatever else answers on the port.
        "service": "vehicle-inspection",
        "status": "healthy" if model_ok else "degraded",
        "inference_server": "ready" if endpoint_ok else "not ready",
        "error": error,
        "model": MODEL_LABEL,
        "ollama_model": OLLAMA_MODEL,
        "inference_engine": "Ollama",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/inspect")
async def inspect_endpoint(
    image: UploadFile = File(..., description="Vehicle photograph"),
    port: str = Form("baltimore"),
    vessel: str = Form(""),
    vin: str = Form(""),
):
    """Run inspection on a single vehicle photograph."""
    if port not in PORTS:
        logger.warning("Unknown port '%s', defaulting to baltimore", port)
        port = "baltimore"

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image")

    try:
        data = await image.read()
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large (max 20MB)")
        pil_image = Image.open(io.BytesIO(data))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Image loading failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Could not load image: {e}")

    try:
        report = await run_inspection(
            image=pil_image,
            port_key=port,
            vessel=vessel.strip(),
            vin=vin.strip(),
        )
        return JSONResponse(content=report)
    except Exception as e:
        logger.error("Pipeline failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Inspection failed: {e}")


@app.post("/api/inspect_sample")
async def inspect_sample_endpoint(
    sample: str = Form(..., description="Filename of a sample image under SAMPLES_DIR"),
    port: str = Form("baltimore"),
    vessel: str = Form(""),
    vin: str = Form(""),
):
    """Run inspection against a bundled sample image (demo convenience)."""
    if port not in PORTS:
        port = "baltimore"

    sample_path = samples_path / sample
    # Guard against path traversal
    try:
        sample_path = sample_path.resolve()
        samples_path_resolved = samples_path.resolve()
        if not str(sample_path).startswith(str(samples_path_resolved)):
            raise HTTPException(status_code=400, detail="Invalid sample path")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid sample path")

    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample not found: {sample}")

    try:
        pil_image = Image.open(sample_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open sample: {e}")

    try:
        report = await run_inspection(
            image=pil_image,
            port_key=port,
            vessel=vessel.strip(),
            vin=vin.strip(),
        )
        return JSONResponse(content=report)
    except Exception as e:
        logger.error("Pipeline failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Inspection failed: {e}")


# -- Inspection lane -----------------------------------------------------------

@app.get("/api/lane/state")
async def lane_state():
    return LANE.snapshot()


@app.post("/api/lane/start")
async def lane_start(request: Request):
    body = await request.json()
    try:
        return await LANE.start(port=body.get("port", "baltimore"), vessel=body.get("vessel", ""),
                                mode=body.get("mode", "live"))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/api/lane/stop")
async def lane_stop():
    await LANE.stop()
    return LANE.snapshot()


@app.post("/api/lane/{vid}/decision")
async def lane_decision(vid: str, request: Request):
    body = await request.json()
    try:
        return await LANE.decide(vid, body.get("action", ""), body.get("note", ""))
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown vehicle")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/lane/{vid}/full_report")
async def lane_full_report(vid: str):
    try:
        return await LANE.request_full_report(vid)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown vehicle")


@app.get("/api/lane/report.csv")
async def lane_report_csv():
    run = LANE.run
    name = f"vessel-inspection-{run.run_id if run else 'empty'}.csv"
    return Response(content=LANE.csv(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/lane/events")
async def lane_events(request: Request):
    """Server-sent events. A client loads /api/lane/state first, then applies
    these; on reconnect it simply reloads the snapshot."""
    queue = LANE.subscribe()

    async def gen():
        try:
            yield "retry: 2000\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            LANE.unsubscribe(queue)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
