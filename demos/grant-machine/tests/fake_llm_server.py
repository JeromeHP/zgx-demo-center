"""A scripted, OpenAI-compatible fake server for tests. No network, no GPU.

Usage in a test:
    app = create_app()
    app.state.queue.append(ScriptedResponse(content="hello"))
    client = LLMClient(settings, http_client=httpx.Client(
        transport=httpx.ASGITransport(app=app), base_url="http://fake/v1"
    ))
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


@dataclass
class ScriptedToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text as the model would emit it; may be malformed


@dataclass
class ScriptedResponse:
    content: str = ""
    reasoning_content: str | None = None
    # Servers disagree on the field name: llama-server says "reasoning_content",
    # Ollama's OpenAI-compat layer says "reasoning". Same payload either way.
    reasoning_field_name: str = "reasoning_content"
    inline_think: str | None = None  # if set, wraps as <think>{inline_think}</think> ahead of content
    tool_calls: list[ScriptedToolCall] = field(default_factory=list)
    usage: dict[str, int] | None = None  # None -> omit usage entirely
    finish_reason: str = "stop"
    stream_chunk_size: int = 6
    tool_arg_fragments: int = 2


def create_app(model_alias: str = "qwen3.6-35b-a3b") -> FastAPI:
    app = FastAPI()
    app.state.model_alias = model_alias
    app.state.queue = []
    app.state.received_requests = []

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": app.state.model_alias, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        app.state.received_requests.append(body)
        if not app.state.queue:
            return JSONResponse(
                status_code=500, content={"error": "fake_llm_server: no scripted response queued"}
            )
        scripted: ScriptedResponse = app.state.queue.pop(0)
        stream = bool(body.get("stream"))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage")) and scripted.usage is not None

        if stream:
            return StreamingResponse(_stream_response(scripted, include_usage), media_type="text/event-stream")
        return JSONResponse(_full_response(scripted))

    return app


def _message_content(scripted: ScriptedResponse) -> str:
    if scripted.inline_think is not None:
        return f"<think>{scripted.inline_think}</think>{scripted.content}"
    return scripted.content


def _usage_payload(usage: dict[str, int]) -> dict[str, int]:
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def _full_response(scripted: ScriptedResponse) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": _message_content(scripted)}
    if scripted.reasoning_content is not None:
        message[scripted.reasoning_field_name] = scripted.reasoning_content
    if scripted.tool_calls:
        message["tool_calls"] = [
            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
            for tc in scripted.tool_calls
        ]
    resp: dict[str, Any] = {
        "id": "fake-chatcmpl",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "fake",
        "choices": [{"index": 0, "message": message, "finish_reason": scripted.finish_reason}],
    }
    if scripted.usage is not None:
        resp["usage"] = _usage_payload(scripted.usage)
    return resp


def _chunk(text: str, size: int) -> list[str]:
    if not text:
        return []
    return [text[i : i + size] for i in range(0, len(text), size)]


def _split_fragments(text: str, n: int) -> list[str]:
    if n <= 1 or len(text) <= 1:
        return [text] if text else [""]
    size = max(1, -(-len(text) // n))  # ceil division so we don't drop the tail
    return [text[i : i + size] for i in range(0, len(text), size)]


async def _stream_response(scripted: ScriptedResponse, send_usage: bool):
    created = int(time.time())
    base = {"id": "fake-chatcmpl", "object": "chat.completion.chunk", "created": created, "model": "fake"}

    def sse(payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    yield sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})

    if scripted.reasoning_content is not None:
        for piece in _chunk(scripted.reasoning_content, scripted.stream_chunk_size):
            yield sse(
                {
                    **base,
                    "choices": [
                        {"index": 0, "delta": {scripted.reasoning_field_name: piece}, "finish_reason": None}
                    ],
                }
            )

    content = _message_content(scripted) if scripted.inline_think is not None else scripted.content
    for piece in _chunk(content, scripted.stream_chunk_size):
        yield sse({**base, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})

    for idx, tc in enumerate(scripted.tool_calls):
        fragments = _split_fragments(tc.arguments, scripted.tool_arg_fragments)
        first = {
            "index": idx,
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.name, "arguments": fragments[0]},
        }
        yield sse({**base, "choices": [{"index": 0, "delta": {"tool_calls": [first]}, "finish_reason": None}]})
        for frag in fragments[1:]:
            more = {"index": idx, "function": {"arguments": frag}}
            yield sse({**base, "choices": [{"index": 0, "delta": {"tool_calls": [more]}, "finish_reason": None}]})

    finish_reason = "tool_calls" if scripted.tool_calls else scripted.finish_reason
    final_payload: dict[str, Any] = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}
    if send_usage:
        final_payload["usage"] = _usage_payload(scripted.usage)  # type: ignore[arg-type]
    yield sse(final_payload)
    yield "data: [DONE]\n\n"
