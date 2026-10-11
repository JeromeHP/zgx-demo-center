from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import httpx
import time
import os
import threading
from rag_ddi import retrieve_interaction, format_rag_context, warm_up

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEVERITY_MAP = {
    "none": "None", "no interaction": "None", "no known": "None",
    "low": "Low", "minor": "Low", "mild": "Low",
    "moderate": "Moderate",
    "high": "High", "major": "High", "significant": "High",
    "contraindicated": "Contraindicated", "avoid": "Contraindicated", "do not use": "Contraindicated",
}

def normalize_severity(text: str) -> str:
    return SEVERITY_MAP.get(text.lower().strip(), text)

app = FastAPI(title="Medical AI Drug Interaction Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = os.environ.get("FRONTEND_DIR", "/app/frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ── Pydantic models ─────────────────────────────────────────────────────────

class SimpleDrugQuery(BaseModel):
    drug1: str
    drug2: str

class ComplexDrugQuery(BaseModel):
    medications: List[str]
    age: int
    conditions: List[str]
    lab_values: Optional[dict] = {}
    additional_context: Optional[str] = ""


# ── Ollama config ────────────────────────────────────────────────────────────

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "medical-mixtral")

logger.info(f"Using Ollama at {OLLAMA_BASE}, model: {OLLAMA_MODEL}")

# Session token accounting for the shared ZGX chrome (see demos/ui/README.md).
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


def ollama_generate(prompt: str, max_tokens: int = 512) -> str:
    """Call Ollama API for inference."""
    response = httpx.post(
        f"{OLLAMA_BASE}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": max_tokens,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "stop": ["</s>", "[INST]"],
            },
            "keep_alive": -1,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()
    # Record the tokens Ollama reports. This used to be
    #     return response.json()["response"].strip()
    # which discarded prompt_eval_count, eval_count and eval_duration at the
    # return statement -- the whole dict was right there and thrown away, so
    # this demo could never say how many tokens it had processed locally.
    LEDGER.record(data, fallback_text=data.get("response", ""))
    return (data.get("response") or "").strip()


# ── RAG warm-up ──────────────────────────────────────────────────────────────
# The first query used to cold-fail: the embed (nomic-embed-text) timed out
# while the 28GB mixtral cold-loaded and saturated the GPU, so retrieval
# silently returned available:False and the LLM produced an un-grounded
# severity. We prime RAG on startup (in a background thread so the server still
# binds immediately) and gate /api/health on RAG_READY so the launcher's
# readiness poll doesn't go green until retrieval actually works.

RAG_READY = False


def _warm_up_rag():
    global RAG_READY
    for attempt in range(1, 11):
        try:
            if warm_up():
                RAG_READY = True
                logger.info(f"RAG warm-up complete on attempt {attempt} — retrieval primed")
                return
            logger.warning(f"RAG warm-up attempt {attempt}/10 did not succeed")
        except Exception as e:
            logger.warning(f"RAG warm-up attempt {attempt}/10 errored: {e}")
        time.sleep(5)
    logger.error("RAG warm-up failed after 10 attempts — /api/health stays unhealthy")


@app.on_event("startup")
async def _on_startup():
    threading.Thread(target=_warm_up_rag, name="rag-warmup", daemon=True).start()


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    try:
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Frontend not found. Check container build.</h1>",
            status_code=500
        )

@app.post("/api/simple_interaction")
async def simple_drug_interaction(query: SimpleDrugQuery):
    try:
        # Retrieve DDI reference data
        rag_results = retrieve_interaction(query.drug1, query.drug2)
        rag_context = format_rag_context(rag_results)

        # Extract severity from DB if available — scan all hits for exact drug name match
        db_severity = None
        if rag_results.get("available") and rag_results.get("hits"):
            queried = {query.drug1.lower().strip(), query.drug2.lower().strip()}
            for hit in rag_results["hits"]:
                if not hit.get("exact_match"):
                    continue
                matched = {hit["metadata"].get("drug_a", "").lower(), hit["metadata"].get("drug_b", "").lower()}
                if queried == matched:
                    db_severity = hit["metadata"].get("severity")
                    break

        prompt = f"""[INST] You are a clinical pharmacist providing drug interaction analysis.

{rag_context}

Analyze the interaction between {query.drug1} and {query.drug2}.

{"The reference database confirms severity: " + db_severity + ". Use this classification." if db_severity else ""}

You MUST begin your response with exactly this format on the first line:
SEVERITY: <one of: None | Low | Moderate | High | Contraindicated>

Then provide:
- Mechanism of interaction
- Clinical significance
- Monitoring recommendations
- Alternative medications if severity is High or Contraindicated

Be concise but thorough. [/INST]"""

        start_time = time.time()
        response = ollama_generate(prompt, max_tokens=512)
        inference_time = time.time() - start_time

        # Use DB severity if we have an exact match, otherwise parse from model output
        if db_severity:
            severity = db_severity
        else:
            # Parse severity from first line
            severity = "Unknown"
            lines = response.strip().splitlines()
            for line in lines:
                if line.strip().upper().startswith("SEVERITY:"):
                    raw = line.split(":", 1)[1].strip()
                    severity = normalize_severity(raw)
                    break

        severity = normalize_severity(severity)

        return JSONResponse(content={
            "success": True,
            "drug1": query.drug1,
            "drug2": query.drug2,
            "severity": severity,
            "analysis": response,
            "inference_time": round(inference_time, 2),
            "complexity_level": "simple",
            "rag_used": rag_results.get("available", False),
            "db_match": db_severity is not None,
        })

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/complex_interaction")
async def complex_drug_interaction(query: ComplexDrugQuery):
    try:
        medications_str = ", ".join(query.medications)
        conditions_str = ", ".join(query.conditions) if query.conditions else "None reported"

        lab_context = ""
        if query.lab_values:
            lab_items = [f"{k}: {v}" for k, v in query.lab_values.items()]
            lab_context = f"Lab values: {', '.join(lab_items)}"

        # Build all drug pairs from the medications list
        from itertools import combinations
        drug_pairs = list(combinations([m.lower().strip() for m in query.medications], 2))

        # Retrieve DDI data for each pair
        rag_contexts = []
        high_severity_pairs = []
        db_severity_map = {}
        for drug_a, drug_b in drug_pairs:
            result = retrieve_interaction(drug_a, drug_b)
            if result.get("available") and result.get("hits"):
                queried = {drug_a, drug_b}
                for hit in result["hits"]:
                    if not hit.get("exact_match"):
                        continue
                    matched_a = hit["metadata"].get("drug_a", "").lower()
                    matched_b = hit["metadata"].get("drug_b", "").lower()
                    matched = {matched_a, matched_b}
                    if queried != matched:
                        continue
                    severity = normalize_severity(hit["metadata"].get("severity", ""))
                    rag_contexts.append(f"- {drug_a.title()} + {drug_b.title()}: SEVERITY={severity.upper()} (database-confirmed, do not change) — {hit['metadata'].get('mechanism', '')}")
                    db_severity_map[f"{drug_a} + {drug_b}"] = severity
                    db_severity_map[f"{drug_b} + {drug_a}"] = severity
                    if severity in ["High", "Contraindicated"]:
                        high_severity_pairs.append(f"{drug_a} + {drug_b} ({severity})")
                    break  # found the exact match, stop searching hits

        rag_block = ""
        if rag_contexts:
            rag_block = "REFERENCE DATA (from clinical DDI database):\n" + "\n".join(rag_contexts)

        prompt = f"""[INST] You are a clinical decision support AI. Analyze this complex patient case.

Patient Profile:
- Age: {query.age} years old
- Current medications: {medications_str}
- Medical conditions: {conditions_str}
{lab_context}

{f'CLINICALLY VALIDATED INTERACTIONS (from DDI database — use these severities exactly, do not upgrade or downgrade):' + chr(10) + chr(10).join(rag_contexts) if rag_contexts else ''}

{f'The following pairs are confirmed HIGH severity by the database: {", ".join(high_severity_pairs)}' if high_severity_pairs else ''}

IMPORTANT: Only use the database severities for the pairs listed above. For all other drug pairs not listed, assess severity independently based on your clinical knowledge and clearly label them as "estimated" severity.

Provide:
1. Drug-drug interactions — list database-confirmed pairs first with their exact severity, then estimated pairs separately
2. Risk assessment considering age, conditions, and lab values
3. Monitoring recommendations
4. Priority actions

Be thorough but concise. [/INST]"""

        start_time = time.time()
        response = ollama_generate(prompt, max_tokens=400)
        inference_time = time.time() - start_time

        # Replace any incorrect severity mentions in the response for confirmed pairs
        import re
        for pair, correct_severity in db_severity_map.items():
            pattern = re.compile(re.escape(pair) + r'\s*\(([^)]+)\)', re.IGNORECASE)
            response = pattern.sub(f"{pair} ({correct_severity})", response)

        base_score = min(10, len(query.medications) * 1.5 + (query.age / 10) - 5)
        severity_bonus = len(high_severity_pairs) * 1.5
        risk_score = min(10, base_score + severity_bonus)
        risk_level = "Low" if risk_score < 4 else "Moderate" if risk_score < 7 else "High"

        return JSONResponse(content={
            "success": True,
            "patient_summary": {
                "medications": query.medications,
                "age": query.age,
                "conditions": query.conditions
            },
            "analysis": response,
            "risk_score": round(risk_score, 1),
            "risk_level": risk_level,
            "inference_time": round(inference_time, 2),
            "complexity_level": "complex",
            "rag_used": len(rag_contexts) > 0,
            "db_interactions": len(high_severity_pairs),
            "db_severity_map": {k: v for k, v in db_severity_map.items() if '+' in k and k.split(' + ')[0] <= k.split(' + ')[1]},
        })

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics")
async def api_metrics():
    """Session token totals for the shared chrome. Hidden if absent."""
    return LEDGER.snapshot()


@app.get("/api/health")
async def health_check():
    try:
        r = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5.0)
        models = [m["name"] for m in r.json().get("models", [])]
        model_available = any(OLLAMA_MODEL in m for m in models)
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "model_loaded": False, "rag_ready": RAG_READY, "error": "Cannot reach Ollama"}
        )
    # Healthy only when the LLM is available AND RAG retrieval is primed, so the
    # launcher's readiness poll doesn't go green before the first query works.
    healthy = model_available and RAG_READY
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "healthy" if healthy else "starting",
            "model_loaded": model_available,
            "rag_ready": RAG_READY,
            "model_type": "Fine-tuned Mixtral Medical (Quantized Q4)",
            "inference_engine": "ollama",
        },
    )

@app.get("/api/sample_queries")
async def get_sample_queries():
    return {
        "simple_examples": [
            {"drug1": "warfarin", "drug2": "aspirin"},
            {"drug1": "metformin", "drug2": "contrast dye"},
            {"drug1": "simvastatin", "drug2": "clarithromycin"}
        ],
        "complex_examples": [
            {
                "description": "Elderly patient with polypharmacy",
                "medications": ["warfarin", "aspirin", "omeprazole", "metoprolol", "amlodipine"],
                "age": 78,
                "conditions": ["atrial fibrillation", "hypertension", "GERD"],
                "lab_values": {"INR": "3.2", "CrCl": "45 mL/min"}
            },
            {
                "description": "Diabetic patient with cardiovascular disease",
                "medications": ["metformin", "glipizide", "atorvastatin", "lisinopril", "clopidogrel"],
                "age": 65,
                "conditions": ["type 2 diabetes", "CAD s/p stent", "hyperlipidemia"],
                "lab_values": {"HbA1c": "8.2%", "eGFR": "58"}
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn

    host_ip = os.environ.get("HOST_IP", "")
    port = 8000

    print("\n" + "=" * 60)
    print("  Medical AI Drug Interaction Demo")
    print("  Fine-tuned Mixtral-8x7B via Ollama | HP ZGX Nano")
    print("=" * 60)
    if host_ip:
        print(f"\n  ➜  http://{host_ip}:{port}")
    else:
        print(f"\n  ➜  http://localhost:{port}")
    print(f"  ➜  Health: http://{'localhost' if not host_ip else host_ip}:{port}/api/health")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port)
