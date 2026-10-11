"""
ollama_client.py — shared inference adapter for all ZGX slim demos

Replaces llama.cpp / vLLM direct calls with HTTP calls to the host Ollama service.
Import this in any demo's main.py instead of instantiating a local model.

Ollama API reference: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import os
import httpx

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://host-gateway:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")

# How long to wait for a response (seconds). VL inference on large images can be slow.
TIMEOUT = 120.0


async def generate(prompt: str, image_b64: str | None = None) -> dict:
    """
    Send a prompt (and optional base64-encoded image) to Ollama.
    Returns the full Ollama response dict so callers can access token counts.

    Response keys of interest:
        response          — the model's text output
        prompt_eval_count — tokens consumed by the prompt (mirrors cloud "input tokens")
        eval_count        — tokens generated in the response (mirrors cloud "output tokens")
    """
    payload: dict = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if image_b64:
        payload["images"] = [image_b64]

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
        resp = await resp
        resp.raise_for_status()
        return resp.json()


async def chat(messages: list[dict], image_b64: str | None = None) -> dict:
    """
    Multi-turn chat variant using Ollama's /api/chat endpoint.
    messages format: [{"role": "user", "content": "..."}, ...]

    Use this when demo UX is conversational rather than single-shot.
    """
    if image_b64 and messages:
        # Attach image to last user message
        last = messages[-1]
        if last.get("role") == "user":
            messages[-1] = {**last, "images": [image_b64]}

    payload = {
        "model":    OLLAMA_MODEL,
        "messages": messages,
        "stream":   False,
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()


async def health() -> dict:
    """Check that Ollama is reachable and the target model is loaded."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            model_loaded = any(OLLAMA_MODEL in m for m in models)
            return {
                "ollama_reachable": True,
                "model_loaded":     model_loaded,
                "model_name":       OLLAMA_MODEL,
                "available_models": models,
            }
    except Exception as e:
        return {
            "ollama_reachable": False,
            "model_loaded":     False,
            "model_name":       OLLAMA_MODEL,
            "error":            str(e),
        }
