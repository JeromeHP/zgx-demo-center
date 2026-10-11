"""
Maritime Surveillance Intelligence Generator
Powered by HP ZGX Nano AI Station

A Vision Language Model demo for US Navy reconnaissance imagery analysis
with synthetic geolocation and structured intelligence report generation.

Uses Qwen3-VL-8B-Instruct served via vLLM for image understanding
and intelligence report generation in a single model.
"""

import os
import time
import io
import base64
import random
import string
from datetime import datetime, timezone
from pathlib import Path
import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pillow_avif  # noqa: F401 — registers AVIF decode support with Pillow
import httpx

# Session token accounting for the shared ZGX chrome (demos/ui/README.md).
try:  # noqa: E402
    from zgx_metrics import TokenLedger
except ImportError:  # pragma: no cover - depends on how the container launches
    # Two launch styles in this suite put a different directory on sys.path:
    #   python3 backend/main.py      -> /app/backend  (plain import works)
    #   uvicorn backend.main:app     -> /app          (backend is a package)
    # pii and ballistics use the second and crash-looped on ModuleNotFoundError
    # until this fallback existed. Covering both is cheaper than making every
    # demo agree on a launch style.
    from backend.zgx_metrics import TokenLedger

LEDGER = TokenLedger()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8090/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "/models/Qwen3-VL-8B-Instruct")

# Operating regions with realistic coordinate bounds
OPERATING_REGIONS = {
    "western_pacific": {
        "name": "Western Pacific",
        "lat_range": (15.0, 35.0),
        "lon_range": (120.0, 145.0),
        "landmarks": ["Philippine Sea", "East China Sea", "Taiwan Strait"]
    },
    "south_china_sea": {
        "name": "South China Sea",
        "lat_range": (5.0, 22.0),
        "lon_range": (105.0, 120.0),
        "landmarks": ["Spratly Islands", "Paracel Islands", "Gulf of Tonkin"]
    },
    "arabian_gulf": {
        "name": "Arabian Gulf / Persian Gulf",
        "lat_range": (24.0, 30.0),
        "lon_range": (48.0, 56.0),
        "landmarks": ["Strait of Hormuz", "Gulf of Oman", "Bahrain"]
    },
    "gulf_of_aden": {
        "name": "Gulf of Aden",
        "lat_range": (11.0, 15.0),
        "lon_range": (43.0, 51.0),
        "landmarks": ["Bab el-Mandeb Strait", "Socotra Island", "Djibouti"]
    },
    "eastern_mediterranean": {
        "name": "Eastern Mediterranean",
        "lat_range": (32.0, 37.0),
        "lon_range": (28.0, 36.0),
        "landmarks": ["Cyprus", "Crete", "Suez Canal approaches"]
    },
    "north_atlantic": {
        "name": "North Atlantic",
        "lat_range": (35.0, 60.0),
        "lon_range": (-45.0, -10.0),
        "landmarks": ["GIUK Gap", "Azores", "Bay of Biscay"]
    },
    "california_coast": {
        "name": "California Coast (TRAINING)",
        "lat_range": (32.5, 38.0),
        "lon_range": (-124.0, -117.0),
        "landmarks": ["San Diego", "Point Loma", "Monterey Bay"]
    },
    "indo_pacific": {
        "name": "Indo-Pacific Region",
        "lat_range": (-10.0, 10.0),
        "lon_range": (95.0, 130.0),
        "landmarks": ["Malacca Strait", "Singapore Strait", "Java Sea"]
    }
}

app = FastAPI(
    title="Maritime Surveillance Intelligence Generator",
    description="VLM-powered reconnaissance imagery analysis for US Navy",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files from frontend directory
FRONTEND_DIR = os.environ.get("FRONTEND_DIR", "/app/frontend")
frontend_path = Path(FRONTEND_DIR)
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Async HTTP client for calling vLLM
http_client = httpx.AsyncClient(timeout=120.0)


# ── vLLM interaction ────────────────────────────────────────────────────────

async def query_vlm(image_b64: str, prompt: str, max_tokens: int = 512) -> tuple:
    """Send an image + prompt to Qwen3-VL via vLLM's OpenAI-compatible API.
    Returns (content_text, usage_dict)."""
    # Wall time for the call. The OpenAI-compatible shape carries no decode
    # duration (Ollama only reports eval_duration on its native /api/generate),
    # so without this the tok/s meter has nothing to divide by and reads "—"
    # forever. Wall time is the weaker number -- it includes prompt processing
    # -- but it is a real measurement rather than an absent one.
    _t0 = time.monotonic()
    try:
        response = await http_client.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json={
                "model": VLLM_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "chat_template_kwargs": {"enable_thinking": False}
            }
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        # Session accounting for the shared chrome. Recorded inside query_vlm so
        # BOTH callers are counted -- the analysis pass and the threat
        # justification -- rather than at each call site, where a future third
        # caller would silently go uncounted.
        LEDGER.record(data, fallback_text=content or "", wall_s=time.monotonic() - _t0)
        return content, usage
    except Exception as e:
        logger.error(f"vLLM query failed: {e}")
        raise


async def analyze_image_with_vlm(image_b64: str) -> tuple:
    """
    Analyze image using Qwen3-VL with a single comprehensive prompt.
    Returns (structured_analysis_dict, usage_dict).
    """
    logger.info("Analyzing image with Qwen3-VL...")

    prompt = """Analyze this aerial/satellite maritime reconnaissance image. Answer each question on its own line in exactly this format:

VESSEL_TYPE: [What kind of ship is this? e.g., cargo ship, container ship, tanker, fishing boat, military vessel, destroyer, frigate, aircraft carrier, cruise ship]
DESCRIPTION: [Brief description of what you see in this aerial image]
CARGO: [What is visible on deck? Describe any cargo, containers, equipment, weapons systems, or aircraft]
ACTIVITY: [Is the ship moving, anchored, or docked? Describe its current activity]
SIZE: [Is this a small boat, medium vessel, or large ship?]
HEADING: [What direction does the ship appear to be traveling?]

Be specific and concise. Base your answers only on what you can see in the image."""

    raw_response, usage = await query_vlm(image_b64, prompt, max_tokens=400)

    # Parse the structured response
    results = {
        "vessel_type": "UNIDENTIFIED VESSEL",
        "description": "No description available",
        "cargo": "Unknown",
        "activity": "Unknown",
        "size": "Unknown",
        "heading": "Unable to determine"
    }

    for line in raw_response.strip().split("\n"):
        line = line.strip()
        for key in results.keys():
            tag = key.upper() + ":"
            if line.upper().startswith(tag):
                value = line[len(tag):].strip()
                if value:
                    results[key] = value
                break

    logger.info(f"VLM analysis results: {results}")
    return results, usage


# ── Geolocation generation ──────────────────────────────────────────────────

def generate_synthetic_coordinates(region: str) -> dict:
    """Generate realistic synthetic coordinates within the specified region."""
    if region not in OPERATING_REGIONS:
        region = "western_pacific"

    region_data = OPERATING_REGIONS[region]

    lat = random.uniform(*region_data["lat_range"])
    lon = random.uniform(*region_data["lon_range"])

    # Convert to degrees, minutes, seconds
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"

    lat_abs = abs(lat)
    lon_abs = abs(lon)

    lat_deg = int(lat_abs)
    lat_min = int((lat_abs - lat_deg) * 60)
    lat_sec = int(((lat_abs - lat_deg) * 60 - lat_min) * 60)

    lon_deg = int(lon_abs)
    lon_min = int((lon_abs - lon_deg) * 60)
    lon_sec = int(((lon_abs - lon_deg) * 60 - lon_min) * 60)

    # Generate MGRS-style grid reference (simplified)
    grid_zone = f"{int((lon + 180) / 6) + 1:02d}"
    grid_band = chr(ord('C') + int((lat + 80) / 8))
    grid_square = ''.join(random.choices(string.ascii_uppercase[:8], k=2))
    grid_easting = f"{random.randint(1000, 9999)}"
    grid_northing = f"{random.randint(1000, 9999)}"

    # Select nearby landmark
    landmark = random.choice(region_data["landmarks"])
    distance_nm = random.randint(15, 200)
    bearing = random.randint(0, 359)

    return {
        "decimal": {"lat": round(lat, 6), "lon": round(lon, 6)},
        "dms": f"{lat_deg}\u00b0{lat_min:02d}'{lat_sec:02d}\"{lat_dir}, {lon_deg}\u00b0{lon_min:02d}'{lon_sec:02d}\"{lon_dir}",
        "mgrs": f"{grid_zone}{grid_band} {grid_square} {grid_easting} {grid_northing}",
        "relative": f"{distance_nm}nm {bearing}\u00b0 from {landmark}",
        "region_name": region_data["name"]
    }


def generate_report_id() -> str:
    """Generate a realistic report identifier."""
    prefix = random.choice(["RECON", "SURV", "ISR", "MARI"])
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    seq = f"{random.randint(1, 9999):04d}"
    return f"{prefix}-{date_str}-{seq}"


# ── Intelligence report generation ──────────────────────────────────────────

async def generate_threat_justification(image_b64: str, vessel_type: str, threat_level: str) -> tuple:
    """Use VLM to generate a contextual threat justification. Returns (text, usage_dict)."""
    prompt = f"""Based on this maritime reconnaissance image, provide a single sentence justification for a {threat_level} threat assessment of this {vessel_type}. Be specific to what you observe in the image. One sentence only."""

    try:
        response, usage = await query_vlm(image_b64, prompt, max_tokens=100)
        # Take first sentence only
        if ". " in response:
            response = response.split(". ")[0] + "."
        return response, usage
    except Exception:
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def classify_threat(image_analysis: dict) -> tuple:
    """Determine threat level and vessel category from analysis."""
    all_text = " ".join(image_analysis.values()).lower()

    if any(word in all_text for word in ["warship", "navy", "military", "destroyer", "frigate", "cruiser", "corvette", "battleship", "aircraft carrier"]):
        return "HIGH", "MILITARY - SURFACE WARFARE", "Military combatant detected - assess nationality, capabilities, and intent"
    elif any(word in all_text for word in ["submarine", "sub "]):
        return "CRITICAL", "MILITARY - SUBSURFACE", "Subsurface contact - immediate tracking and classification required"
    elif any(word in all_text for word in ["russian", "chinese", "iranian", "north korean"]):
        return "MEDIUM", "FOREIGN NATIONAL - MONITOR", "Foreign national vessel - increased monitoring recommended"
    elif any(word in all_text for word in ["patrol", "coast guard", "cutter"]):
        return "MEDIUM", "GOVERNMENT - LAW ENFORCEMENT", "Government vessel - determine nationality and jurisdiction"
    elif any(word in all_text for word in ["cargo", "container", "freight", "merchant", "commercial"]):
        return "NONE", "COMMERCIAL - MERCHANT MARINE", "Commercial cargo vessel engaged in routine maritime trade operations"
    elif any(word in all_text for word in ["tanker", "oil", "petroleum"]):
        return "LOW", "COMMERCIAL - BULK CARRIER", "Strategic cargo vessel - monitor for sanctions compliance"
    elif any(word in all_text for word in ["fishing", "trawler"]):
        return "LOW", "COMMERCIAL - FISHING", "Fishing vessel - potential for surveillance or militia activity"
    elif any(word in all_text for word in ["cruise", "passenger", "ferry"]):
        return "NONE", "COMMERCIAL - PASSENGER", "Passenger vessel engaged in routine commercial operations"
    else:
        return "LOW", "UNCLASSIFIED", "Vessel type undetermined - recommend visual confirmation"


def generate_recommendations(threat_level: str, cargo: str, custom_instructions: str = "") -> str:
    """Generate contextual recommendations based on threat level and cargo."""
    recommendations = []

    if threat_level == "CRITICAL":
        recommendations.append("IMMEDIATE: Alert fleet command and request additional ISR assets")
        recommendations.append("Initiate continuous tracking and maintain safe distance")
        recommendations.append("Prepare anti-submarine warfare assets if applicable")
    elif threat_level == "HIGH":
        recommendations.append("Alert regional command and increase surveillance priority")
        recommendations.append("Attempt visual identification of hull number and flag state")
        recommendations.append("Monitor for weapons systems activation or hostile maneuvering")
    elif threat_level == "MEDIUM":
        recommendations.append("Increase monitoring frequency and document all activities")
        recommendations.append("Coordinate with allied assets for corroborating intelligence")
    else:
        recommendations.append("Continue routine monitoring")

    # Cargo-specific recommendations
    cargo_lower = cargo.lower()
    if any(word in cargo_lower for word in ["warship", "military", "weapon"]):
        recommendations.append("PRIORITY: Assess military cargo and potential threat capability")
    elif any(word in cargo_lower for word in ["russian", "chinese", "iranian", "north korean"]):
        recommendations.append("Flag for foreign national vessel monitoring program")
    elif any(word in cargo_lower for word in ["container", "cargo", "freight"]):
        recommendations.append("Verify cargo manifest against sanctions lists if possible")
    elif any(word in cargo_lower for word in ["oil", "tanker", "petroleum", "fuel"]):
        recommendations.append("Monitor for potential sanctions evasion or illegal transfer")
    elif any(word in cargo_lower for word in ["fish", "fishing"]):
        recommendations.append("Check for illegal fishing or maritime militia indicators")

    recommendations.append("Cross-reference with AIS/maritime tracking data")
    recommendations.append("Verify vessel registry and flag state")

    if custom_instructions:
        recommendations.append(custom_instructions)

    return "\n   - ".join(recommendations)


async def build_intelligence_report(image_analysis: dict, image_b64: str, custom_instructions: str = "") -> tuple:
    """Build structured intelligence report from VLM analysis. Returns (report_text, usage_dict)."""
    vessel_type = image_analysis.get("vessel_type", "UNIDENTIFIED VESSEL")
    vessel_description = image_analysis.get("description", "No description available")
    cargo = image_analysis.get("cargo", "Unknown")
    activity = image_analysis.get("activity", "Unknown")
    size = image_analysis.get("size", "Unknown")
    heading = image_analysis.get("heading", "Unable to determine")

    # Classify threat
    threat_level, vessel_category, default_justification = classify_threat(image_analysis)

    # Get VLM-generated threat justification
    threat_justification, justification_usage = await generate_threat_justification(image_b64, vessel_type, threat_level)
    if not threat_justification:
        threat_justification = default_justification

    # Generate recommendations
    recommendations_text = generate_recommendations(threat_level, cargo, custom_instructions)

    assessment = f"""1. VESSEL CLASSIFICATION
    Category: {vessel_category}

2. PHYSICAL CHARACTERISTICS
   Estimated Size: {size.capitalize()}
   Cargo/Payload: {cargo.capitalize()}
   Visual Description: {vessel_description.capitalize()}

3. ACTIVITY ASSESSMENT
   Current Status: {activity.upper()}
   Heading: {heading.capitalize()}

4. THREAT ASSESSMENT
   \u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
   THREAT LEVEL: {threat_level}
   \u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
   Justification: {threat_justification}

5. CONFIDENCE LEVEL: HIGH
   Assessment based on Qwen3-VL vision-language model analysis

6. RECOMMENDATIONS
   - {recommendations_text}"""

    return assessment, justification_usage


# ── Image analysis pipeline ─────────────────────────────────────────────────

async def analyze_image(image: Image.Image, region: str, custom_instructions: str = "") -> dict:
    """Full analysis pipeline: VLM analysis → threat classification → report generation."""

    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize if too large
    max_size = 1024
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    # Convert to base64 for vLLM API
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Get structured analysis from VLM
    image_analysis, analysis_usage = await analyze_image_with_vlm(image_b64)

    # Build intelligence report
    analysis, justification_usage = await build_intelligence_report(image_analysis, image_b64, custom_instructions)

    # Accumulate token usage across both VLM calls
    token_usage = {
        "prompt_tokens": analysis_usage.get("prompt_tokens", 0) + justification_usage.get("prompt_tokens", 0),
        "completion_tokens": analysis_usage.get("completion_tokens", 0) + justification_usage.get("completion_tokens", 0),
        "total_tokens": analysis_usage.get("total_tokens", 0) + justification_usage.get("total_tokens", 0)
    }

    # Generate synthetic geolocation
    geo_data = generate_synthetic_coordinates(region)
    report_id = generate_report_id()
    capture_time = datetime.now(timezone.utc).strftime("%d %b %Y %H%MZ").upper()

    report = {
        "report_id": report_id,
        "classification": "UNCLASSIFIED // FOR DEMONSTRATION PURPOSES ONLY",
        "capture_time": capture_time,
        "location": {
            "coordinates_dms": geo_data["dms"],
            "coordinates_decimal": geo_data["decimal"],
            "grid_reference": geo_data["mgrs"],
            "relative_position": geo_data["relative"],
            "operating_region": geo_data["region_name"]
        },
        "analysis": analysis,
        "raw_analysis": image_analysis,
        "token_usage": token_usage,
        "generated_at": datetime.now(timezone.utc).isoformat()
    }

    return report


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application page."""
    index_path = Path(FRONTEND_DIR) / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>Maritime Surveillance Intelligence Generator</h1><p>index.html not found</p>")


@app.get("/api/metrics")
async def api_metrics():
    """Session token totals for the shared chrome."""
    return LEDGER.snapshot()


@app.get("/api/health")
async def health_check():
    """Health check — verifies the inference endpoint is actually serving.

    This probed `{base}/health`, which only vLLM serves. Pointed at host Ollama
    (the non-CUDA path) it got a 404 and reported "degraded / not ready" for a
    server that was up and answering, because it was asking the wrong question.

    `{base}/models` is the OpenAI-compatible listing, which vLLM AND Ollama both
    serve, so it works for either backend. The reported engine and model came
    from two hardcoded strings that named vLLM and the FP8 checkpoint whatever
    was really running; they are read from configuration now.
    """
    backend_healthy = False
    served = None
    try:
        resp = await http_client.get(f"{VLLM_BASE_URL.rstrip('/')}/models", timeout=5.0)
        backend_healthy = resp.status_code == 200
        if backend_healthy:
            data = resp.json().get("data") or []
            served = [m.get("id") for m in data if m.get("id")]
    except Exception:
        pass

    external = os.environ.get("INFERENCE_BACKEND", "vllm") == "external"
    return {
        # A marker the container healthcheck greps for. Under network_mode:
        # host this app shares port 8000 with the pharmacy and pii demos, so a
        # bare `curl -f /api/health` can succeed against THEIR health endpoint
        # and report this container healthy while its own bind failed.
        "service": "maritime-surveillance",
        "status": "healthy" if backend_healthy else "degraded",
        "vllm_server": "ready" if backend_healthy else "not ready",
        "model": VLLM_MODEL,
        "inference_engine": "external OpenAI-compatible endpoint" if external else "vLLM",
        "endpoint": VLLM_BASE_URL,
        "models_served": served,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/regions")
async def get_regions():
    """Get available operating regions."""
    return {
        "regions": [
            {"id": k, "name": v["name"], "landmarks": v["landmarks"]}
            for k, v in OPERATING_REGIONS.items()
        ]
    }


@app.post("/api/analyze")
async def analyze_endpoint(
    image: UploadFile = File(...),
    region: str = Form("western_pacific"),
    custom_instructions: str = Form("")
):
    """Analyze an uploaded image and generate an intelligence report."""

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_data = await image.read()

    if len(image_data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 20MB)")

    try:
        img = Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    try:
        report = await analyze_image(img, region, custom_instructions)
        return JSONResponse(content=report)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    host_ip = os.environ.get("HOST_IP", "")
    print("\n" + "=" * 60)
    print("  Maritime Surveillance Intelligence Generator")
    print("  Qwen3-VL-8B-Instruct | HP ZGX Nano | vLLM")
    print("=" * 60)
    if host_ip:
        print(f"\n  \u27a1  http://{host_ip}:{PORT}")
    else:
        print(f"\n  \u27a1  http://localhost:{PORT}")
    print("=" * 60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up HTTP client."""
    await http_client.aclose()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
