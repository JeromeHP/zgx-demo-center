"""
ZGX AI API Server — Medical Triage Voice Agent
-------------------------------------------------
FastAPI server running the full voice pipeline:
  - whisper.cpp server (STT, GPU) via HTTP on port 8178
  - vLLM (LLM, GPU) via HTTP on port 8001
  - Piper TTS (subprocess)

Endpoints:
  GET  /health         - Component status
  POST /transcribe     - Audio → text
  POST /generate       - Text → LLM response
  POST /speak          - Text → audio
  POST /process        - Audio → text → LLM → audio (all-in-one)
  GET  /conversations  - Conversation history for dashboard
  POST /reset          - Clear conversation history
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
import tempfile
import subprocess
import os
import logging
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WHISPER_SERVER_URL = os.getenv("WHISPER_SERVER_URL", "http://localhost:8178")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8001/v1/chat/completions")
VLLM_MODEL = os.getenv("VLLM_MODEL", "/models/Llama-3.1-8B-Instruct-AWQ-INT4")
PIPER_MODEL = os.getenv("PIPER_MODEL", "/models/piper-tts/en_US-lessac-medium.onnx")

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """You are a medical triage assistant. You help patients describe their symptoms and provide initial triage guidance. 

Guidelines:
- Ask clarifying questions about symptoms (location, severity, duration, onset)
- Use a 1-10 pain scale when relevant
- Provide general guidance but always recommend seeing a healthcare provider for serious concerns
- Be empathetic and professional
- Keep responses concise (2-4 sentences) for natural conversation
- Never diagnose conditions — only provide triage-level guidance""")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zgx-ai-api")

# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------
conversations = []

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="ZGX AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _cleanup(path: str):
    """Silently remove a temp file."""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# STT via whisper.cpp server (GPU-accelerated)
# ---------------------------------------------------------------------------

def transcribe_audio(wav_bytes: bytes) -> str:
    """
    Send WAV audio to whisper.cpp server for GPU-accelerated transcription.
    Uses the OpenAI-compatible /inference endpoint.
    """
    try:
        response = requests.post(
            f"{WHISPER_SERVER_URL}/inference",
            files={"file": ("recording.wav", wav_bytes, "audio/wav")},
            data={
                "response_format": "json",
                "temperature": "0.0",
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("text", "").strip()
    except requests.ConnectionError:
        raise HTTPException(503, "whisper.cpp server is not reachable on port 8178.")
    except requests.Timeout:
        raise HTTPException(504, "STT request timed out.")
    except requests.HTTPError as e:
        raise HTTPException(502, f"whisper.cpp server returned an error: {e}")


# ---------------------------------------------------------------------------
# LLM via vLLM
# ---------------------------------------------------------------------------

def generate_response(transcript: str) -> tuple:
    """Send transcript to vLLM. Returns (response_text, usage_dict)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Add conversation history for context
    for conv in conversations[-6:]:  # Last 3 exchanges
        messages.append({"role": "user", "content": conv["user"]})
        messages.append({"role": "assistant", "content": conv["assistant"]})

    messages.append({"role": "user", "content": transcript})

    try:
        response = requests.post(
            VLLM_URL,
            json={
                "model": VLLM_MODEL,
                "messages": messages,
                "max_tokens": 256,
                "temperature": 0.7,
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.ConnectionError:
        raise HTTPException(503, "vLLM server is not reachable on port 8001.")
    except requests.Timeout:
        raise HTTPException(504, "LLM request timed out.")
    except requests.HTTPError as e:
        raise HTTPException(502, f"vLLM returned an error: {e}")

    data = response.json()

    try:
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return text.strip(), usage
    except (KeyError, IndexError):
        raise HTTPException(500, f"Unexpected vLLM response: {data}")


# ---------------------------------------------------------------------------
# TTS via Piper
# ---------------------------------------------------------------------------

def synthesize_speech(text: str) -> str:
    """Run Piper TTS and return path to output wav."""
    output_file = tempfile.mktemp(suffix=".wav", prefix="tts_")

    cmd = [
        "piper",
        "--model", PIPER_MODEL,
        "--output_file", output_file,
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate(input=text.encode("utf-8"))

    if process.returncode != 0:
        _cleanup(output_file)
        raise HTTPException(500, f"Piper TTS failed: {stderr.decode()}")

    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        _cleanup(output_file)
        raise HTTPException(500, "Piper produced no output.")

    return output_file


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check for all components."""
    stt_ok = False
    llm_ok = False

    # Check whisper.cpp server
    try:
        r = requests.get(f"{WHISPER_SERVER_URL}/health", timeout=5)
        stt_ok = r.status_code == 200
    except Exception:
        pass

    # Check vLLM
    try:
        r = requests.get("http://localhost:8001/health", timeout=5)
        llm_ok = r.status_code == 200
    except Exception:
        pass

    return {
        "status": "healthy" if (stt_ok and llm_ok) else "degraded",
        # Backward compatibility with start_services.sh and dashboard
        "whisper": stt_ok,
        "whisper_model": "whisper.cpp (GPU)",
        "llm": llm_ok,
        "tts": os.path.exists(PIPER_MODEL),
    }


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """STT endpoint."""
    try:
        content = await audio.read()
        text = transcribe_audio(content)
        return {"text": text}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/transcribe error: {e}")
        raise HTTPException(500, f"Transcription failed: {e}")


@app.post("/generate")
async def generate(prompt: dict):
    """LLM generation endpoint."""
    if "prompt" not in prompt:
        raise HTTPException(400, "Missing 'prompt' field.")

    text, usage = generate_response(prompt["prompt"])
    return {"text": text, "usage": usage}


@app.post("/speak")
async def speak(text: dict):
    """TTS endpoint."""
    if "text" not in text:
        raise HTTPException(400, "Missing 'text' field.")

    output_file = synthesize_speech(text["text"])
    return FileResponse(output_file, media_type="audio/wav", filename="response.wav")


@app.post("/process")
async def process(audio: UploadFile = File(...)):
    """All-in-one pipeline: audio → STT → LLM → TTS → audio."""
    tts_path = None

    try:
        content = await audio.read()
        logger.info(f"Received {len(content)} bytes of audio")

        # 1. Transcribe via whisper.cpp (GPU)
        t0 = time.time()
        transcript = transcribe_audio(content)
        stt_time = time.time() - t0
        logger.info(f"STT ({stt_time:.1f}s): {transcript}")

        if not transcript or not transcript.strip():
            raise HTTPException(400, "No speech detected.")

        # 2. Generate LLM response
        t0 = time.time()
        ai_response, usage = generate_response(transcript)
        llm_time = time.time() - t0
        logger.info(f"LLM ({llm_time:.1f}s): {ai_response[:100]}...")

        # 3. Store conversation
        conversations.append({
            "user": transcript, "patient": transcript,
            "assistant": ai_response,
            "timestamp": datetime.now().isoformat(),
            "stt_time": round(stt_time, 2),
            "llm_time": round(llm_time, 2),
            "usage": usage,
        })

        # 4. Synthesize speech
        t0 = time.time()
        tts_path = synthesize_speech(ai_response)
        tts_time = time.time() - t0
        logger.info(f"TTS ({tts_time:.1f}s): {os.path.getsize(tts_path)} bytes")

        total = stt_time + llm_time + tts_time
        logger.info(f"Total: {total:.1f}s (STT={stt_time:.1f} LLM={llm_time:.1f} TTS={tts_time:.1f})")

        from starlette.background import BackgroundTask

        return FileResponse(
            tts_path,
            media_type="audio/wav",
            filename="response.wav",
            headers={
                "X-Transcript": transcript[:500],
                "X-Response": ai_response[:500],
                "X-STT-Time": f"{stt_time:.2f}",
                "X-LLM-Time": f"{llm_time:.2f}",
                "X-TTS-Time": f"{tts_time:.2f}",
            },
            background=BackgroundTask(_cleanup, tts_path),
        )

    except HTTPException:
        _cleanup(tts_path)
        raise
    except Exception as e:
        _cleanup(tts_path)
        logger.error(f"/process error: {e}")
        raise HTTPException(500, f"Processing failed: {e}")


@app.get("/conversations")
async def get_conversations():
    """Return conversation history for the dashboard."""
    return {"conversations": conversations, "count": len(conversations)}


@app.post("/reset")
async def reset():
    """Clear conversation history."""
    conversations.clear()
    return {"status": "reset"}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("ZGX AI API — Medical Triage Voice Agent")
    logger.info("=" * 60)
    logger.info(f"  STT:  whisper.cpp @ {WHISPER_SERVER_URL}")
    logger.info(f"  LLM:  vLLM @ {VLLM_URL}")
    logger.info(f"  TTS:  Piper @ {PIPER_MODEL}")
    logger.info("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8090)
