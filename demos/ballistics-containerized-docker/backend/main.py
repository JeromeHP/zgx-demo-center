"""
NAWSCL Range Surveillance — Aircraft Classification
Slim backend: FastAPI → Ollama (host) → qwen2.5vl or compatible VL model

Drops: vLLM server process, Qwen3-VL-FP8 weight mount, NVIDIA base image
Keeps: all API endpoints, token counter display, structured report format
"""

import base64
import io
import os
import random
import uuid
from datetime import datetime

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pillow_avif  # noqa: F401 — registers AVIF decode support with Pillow
from PIL import Image, UnidentifiedImageError

# Shared Ollama adapter — same file across all slim demos
from backend.ollama_client import generate, health as ollama_health

app = FastAPI(title="NAWSCL Aircraft Classification — Slim")

# ── Static frontend ──────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="frontend"), name="static")


def _normalize_image_to_jpeg_b64(image_bytes: bytes) -> str:
    """
    Uploads can be in any format the browser's file picker allows - including
    ones Ollama's own image decoder rejects outright (AVIF and HEIC from
    phone cameras being the most likely at a live demo; found this exact
    failure via a sample image in the sibling navy demo mislabeled .png that
    was actually AVIF). Normalizing to JPEG server-side means the model
    always gets something it can decode instead of surfacing a 502.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Unrecognized image format — try a JPEG or PNG")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


# ── Health ───────────────────────────────────────────────────────────────────
@app.get("/api/metrics")
async def api_metrics():
    """Session token totals for the shared chrome.

    The counts are recorded inside ollama_client, so this module only exposes
    them. Caught by scripts/audit_ui_standard.sh: the client was counting and
    nothing was serving the result, so the chrome's token panel stayed hidden.
    """
    try:
        from ollama_client import LEDGER
    except ImportError:
        from backend.ollama_client import LEDGER
    return LEDGER.snapshot()


@app.get("/api/health")
async def health():
    ollama_status = await ollama_health()
    return {
        "status":      "ok" if ollama_status["ollama_reachable"] else "degraded",
        "backend":     "ollama-slim",
        # frontend/index.html's status dot checks vllm_server === 'ready' —
        # a holdover field name from the original (pre-Ollama) backend.
        "vllm_server": "ready" if ollama_status.get("model_loaded") else "loading",
        **ollama_status,
    }


# ── Range areas metadata ─────────────────────────────────────────────────────
RANGES = {
    "baker_range":     {"designation": "R-2508N", "landmarks": "Echo Range, George Range, Superior Valley"},
    "charlie_range":   {"designation": "R-2508S", "landmarks": "Mojave B Range, Randsburg Wash, Searles Valley"},
    "coso_complex":    {"designation": "—",       "landmarks": "Coso Hot Springs, Darwin Plateau, Haiwee Reservoir"},
    "armitage":        {"designation": "NAF China Lake", "landmarks": "Main Runway 21/03, Ridgecrest"},
    "superior_valley": {"designation": "—",       "landmarks": "Panamint Valley, Trona Pinnacles, Searles Lake"},
    "north_range":     {"designation": "R-2502",  "landmarks": "Owens Valley, Darwin Hills, Centennial Flat"},
}


@app.get("/api/ranges")
async def get_ranges():
    return {"ranges": RANGES}


# ── Known platforms ──────────────────────────────────────────────────────────
KNOWN_PLATFORMS = {
    "manned": ["F/A-18E/F Super Hornet", "F-35C Lightning II", "EA-18G Growler", "AV-8B Harrier"],
    "unmanned": ["MQ-1 Predator", "MQ-8 Fire Scout", "MQ-9 Reaper", "MQ-25 Stingray", "BQM-74 Chukar"],
}

# Approximate real-world center points near NAWSCL China Lake, for flavor —
# synthetic demo data only, not derived from the image.
RANGE_CENTERS = {
    "baker_range":     (35.62, -117.68),
    "charlie_range":   (35.30, -117.55),
    "coso_complex":    (36.03, -117.75),
    "armitage":        (35.68, -117.69),
    "superior_valley": (35.55, -117.35),
    "north_range":     (36.20, -117.75),
}


def _dms(deg: float, is_lat: bool) -> str:
    hemi = ("N" if deg >= 0 else "S") if is_lat else ("E" if deg >= 0 else "W")
    deg = abs(deg)
    d = int(deg)
    m_full = (deg - d) * 60
    m = int(m_full)
    s = (m_full - m) * 60
    return f"{d}°{m}'{s:.1f}\"{hemi}"


def _synthetic_location(range_area: str) -> dict:
    """
    frontend/index.html's "Position Data [Synthetic]" section reads
    report.location.* — the field was missing entirely before, which threw
    inside displayReport() on every single request (caught by the same
    handler that shows "Failed to classify aircraft").
    """
    lat0, lon0 = RANGE_CENTERS.get(range_area, RANGE_CENTERS["baker_range"])
    lat = lat0 + random.uniform(-0.05, 0.05)
    lon = lon0 + random.uniform(-0.05, 0.05)
    return {
        "coordinates_dms":     f"{_dms(lat, True)} {_dms(lon, False)}",
        "coordinates_decimal": {"lat": round(lat, 4), "lon": round(lon, 4)},
        "grid_reference":      f"{random.randint(10, 59)}S{random.randint(10000, 99999)}{random.randint(10000, 99999)}",
        "relative_position":   f"{random.randint(2, 18)} NM {random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])} of range center",
        "range_area":          RANGES.get(range_area, RANGES["baker_range"])["designation"],
        "altitude_ft":         random.randint(500, 35000),
    }


# ── Primary analysis endpoint ─────────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    range_area: str = Form("baker_range"),
    custom_instructions: str = Form(""),
):
    # Read and encode image
    image_bytes = await image.read()
    image_b64 = _normalize_image_to_jpeg_b64(image_bytes)

    range_info = RANGES.get(range_area, RANGES["baker_range"])

    manned_list   = ", ".join(KNOWN_PLATFORMS["manned"])
    unmanned_list = ", ".join(KNOWN_PLATFORMS["unmanned"])

    prompt = f"""You are an AI system supporting range surveillance operations at Naval Air Weapons Station China Lake (NAWSCL).
Analyze the aircraft in this image and produce a structured report.

RANGE AREA: {range_area.replace('_', ' ').title()} ({range_info['designation']})
KEY LANDMARKS: {range_info['landmarks']}
{f'ADDITIONAL INSTRUCTIONS: {custom_instructions}' if custom_instructions else ''}

KNOWN MANNED PLATFORMS: {manned_list}
KNOWN UNMANNED PLATFORMS: {unmanned_list}

Respond ONLY with this exact structure:

CLASSIFICATION: [MANNED or UNMANNED]
PLATFORM: [specific aircraft designation, or UNKNOWN]
PLATFORM CATEGORY: [e.g., UNMANNED - ISR/STRIKE UAS]

PHYSICAL CHARACTERISTICS:
[2-3 sentences describing visible airframe features]

FLIGHT STATUS:
[Altitude estimate, flight profile, heading if determinable]

MISSION PROFILE ASSESSMENT:
[Likely mission based on platform type and observable configuration]

PAYLOAD IDENTIFICATION:
[Visible stores, sensors, or pods; note if not determinable]

CONFIDENCE LEVEL: [HIGH / MEDIUM / LOW]
CONFIDENCE RATIONALE: [One sentence explaining confidence rating]

RANGE SAFETY RECOMMENDATIONS:
[1-2 actionable recommendations for range safety personnel]

CLASSIFICATION BANNER: UNCLASSIFIED // FOR DEMONSTRATION PURPOSES ONLY // NAWSCL CHINA LAKE"""

    try:
        result = await generate(prompt, image_b64=image_b64)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama inference error: {str(e)}")

    response_text = result.get("response", "")

    # Parse MANNED/UNMANNED from response
    classification = "UNKNOWN"
    if "CLASSIFICATION: MANNED" in response_text.upper():
        classification = "MANNED"
    elif "CLASSIFICATION: UNMANNED" in response_text.upper():
        classification = "UNMANNED"

    # Extract platform line
    platform_match = "UNKNOWN"
    for line in response_text.splitlines():
        if line.strip().upper().startswith("PLATFORM:") and "CATEGORY" not in line.upper():
            platform_match = line.split(":", 1)[-1].strip()
            break

    platform_category = "UNKNOWN"
    for line in response_text.splitlines():
        if line.strip().upper().startswith("PLATFORM CATEGORY:"):
            platform_category = line.split(":", 1)[-1].strip()
            break

    classification_banner = "UNCLASSIFIED // FOR DEMONSTRATION PURPOSES ONLY // NAWSCL CHINA LAKE"
    for line in response_text.splitlines():
        if line.strip().upper().startswith("CLASSIFICATION BANNER:"):
            classification_banner = line.split(":", 1)[-1].strip()
            break

    # Token counts — Ollama returns these natively
    token_usage = {
        "prompt_tokens":     result.get("prompt_eval_count", 0),
        "completion_tokens": result.get("eval_count", 0),
        "total_tokens":      result.get("prompt_eval_count", 0) + result.get("eval_count", 0),
    }

    return JSONResponse({
        "report_id":          f"RANGE-{datetime.utcnow().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}",
        "capture_time":       datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "classification":     classification_banner,
        "manned_unmanned":    classification,
        "platform_match":     platform_match,
        "platform_category":  platform_category,
        "location":           _synthetic_location(range_area),
        "analysis":           response_text,
        "token_usage":        token_usage,
        "model":              result.get("model", os.getenv("OLLAMA_MODEL")),
        "range_area":         range_area,
    })
