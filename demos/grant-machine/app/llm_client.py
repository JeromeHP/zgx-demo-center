"""OpenAI-compatible LLM client with reasoning/tool-call normalization (Spec B).

Talks to any OpenAI-compatible /v1/chat/completions endpoint (llama-server,
Ollama, vLLM, ...). Normalizes three server styles into one ChatResult shape:
  - `reasoning_content` field (llama-server --reasoning-format deepseek)
  - `reasoning` field (Ollama's OpenAI-compat layer)
  - inline `<think>...</think>` tags left in `content`
so the rest of the app never has to care which style the current endpoint uses.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx
import openai

from .settings import Settings, SettingsError

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

_RETRY_DELAYS_S = (2.0, 5.0)

# Servers disagree on what to call the separated-reasoning field: llama-server
# emits `reasoning_content`, Ollama's OpenAI-compat layer emits `reasoning`.
# Same shape in both the non-streamed message and the streamed delta, so one
# helper covers both call sites.
_REASONING_FIELDS = ("reasoning_content", "reasoning")


def _reasoning_from(obj: Any) -> str | None:
    """Read the separated-reasoning field under whichever name this server uses."""
    for name in _REASONING_FIELDS:
        value = getattr(obj, name, None)
        if value:
            return value
    return None


class LLMTimeout(Exception):
    def __init__(self, message: str, elapsed_s: float):
        super().__init__(message)
        self.elapsed_s = elapsed_s


class LLMJSONError(Exception):
    def __init__(self, message: str, raw_text: str):
        super().__init__(message)
        self.raw_text = raw_text


@dataclass
class ToolCall:
    id: str
    name: str
    arguments_raw: str
    arguments: dict[str, Any] | None
    parse_error: str | None = None


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    estimated: bool = False


@dataclass
class Timings:
    ttft_s: float | None
    total_s: float
    decode_tps: float | None


@dataclass
class ChatResult:
    content: str
    reasoning: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=lambda: Usage(0, 0, estimated=True))
    timings: Timings = field(default_factory=lambda: Timings(None, 0.0, None))
    reasoning_source: str = "none"  # "field" | "inline_tags" | "none"


def _estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 3.5) if text else 0


def _try_parse_json(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    if not raw:
        return None, "empty arguments"
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def _split_inline_think(text: str) -> tuple[str, str, bool]:
    """Split one non-streamed content string into (content, reasoning, found)."""
    start = text.find(_THINK_OPEN)
    if start == -1:
        return text, "", False
    end = text.find(_THINK_CLOSE, start)
    if end == -1:
        # Unclosed tag: everything after the opener is reasoning.
        return text[:start], text[start + len(_THINK_OPEN) :], True
    reasoning = text[start + len(_THINK_OPEN) : end]
    remainder = text[:start] + text[end + len(_THINK_CLOSE) :]
    return remainder, reasoning, True


class _InlineThinkStreamState:
    """Splits streamed content deltas around <think>...</think>, tolerating a
    tag split across chunk boundaries by holding back a short tail."""

    def __init__(self) -> None:
        self._buffer = ""
        self.in_think = False
        self.found_think = False

    def feed(self, delta: str) -> tuple[str, str]:
        self._buffer += delta
        content_out = ""
        reasoning_out = ""
        while True:
            if not self.in_think:
                idx = self._buffer.find(_THINK_OPEN)
                if idx == -1:
                    safe_len = max(0, len(self._buffer) - (len(_THINK_OPEN) - 1))
                    content_out += self._buffer[:safe_len]
                    self._buffer = self._buffer[safe_len:]
                    break
                content_out += self._buffer[:idx]
                self._buffer = self._buffer[idx + len(_THINK_OPEN) :]
                self.in_think = True
                self.found_think = True
            else:
                idx = self._buffer.find(_THINK_CLOSE)
                if idx == -1:
                    safe_len = max(0, len(self._buffer) - (len(_THINK_CLOSE) - 1))
                    reasoning_out += self._buffer[:safe_len]
                    self._buffer = self._buffer[safe_len:]
                    break
                reasoning_out += self._buffer[:idx]
                self._buffer = self._buffer[idx + len(_THINK_CLOSE) :]
                self.in_think = False
        return content_out, reasoning_out

    def flush_remainder(self) -> tuple[str, str]:
        """Call once the stream ends. Remaining buffered text goes to
        reasoning if we're still inside an unclosed <think>, else content."""
        remainder = self._buffer
        self._buffer = ""
        if self.in_think:
            return "", remainder
        return remainder, ""


class LLMClient:
    def __init__(self, settings: Settings, http_client: httpx.Client | None = None):
        self._settings = settings
        self._llm = settings.llm
        self._client = openai.OpenAI(
            base_url=self._llm.base_url,
            api_key=self._llm.api_key,
            timeout=self._llm.request_timeout_s,
            http_client=http_client,
            max_retries=0,  # we own retries below
        )

    def list_models(self) -> list[str]:
        page = self._client.models.list()
        return [m.id for m in page.data]

    def chat(
        self,
        profile: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        json_mode: bool = False,
        on_reasoning: Callable[[str], None] | None = None,
        on_content: Callable[[str], None] | None = None,
    ) -> ChatResult:
        profile_cfg = self._llm.profiles.get(profile)
        if profile_cfg is None:
            raise SettingsError(f"unknown LLM profile: {profile!r}")

        result = self._chat_once(profile_cfg, messages, tools, stream, on_reasoning, on_content)

        if json_mode:
            try:
                json.loads(result.content)
            except json.JSONDecodeError as exc:
                repair_messages = list(messages) + [
                    {"role": "assistant", "content": result.content},
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response was not valid JSON ({exc}). "
                            "Return only valid JSON, with no other text."
                        ),
                    },
                ]
                result = self._chat_once(profile_cfg, repair_messages, tools, stream, on_reasoning, on_content)
                try:
                    json.loads(result.content)
                except json.JSONDecodeError as exc2:
                    raise LLMJSONError(
                        f"model did not return valid JSON after one repair attempt: {exc2}",
                        result.content,
                    ) from exc2

        return result

    # -- internals ---------------------------------------------------------

    def _build_request_kwargs(self, profile_cfg, messages, tools) -> dict[str, Any]:
        extra_body: dict[str, Any] = {
            "top_k": profile_cfg.top_k,
            "min_p": profile_cfg.min_p,
        }
        # Two mutually exclusive ways to control thinking mode, selected by
        # llm.thinking_control. llama-server reads chat_template_kwargs and has
        # no reasoning_effort; Ollama silently ignores chat_template_kwargs (and
        # `think`, on the /v1 route) and honours only reasoning_effort, where
        # "none" is the sole value that actually suppresses thinking. Sending the
        # wrong one is not an error anywhere -- it is simply ignored, which is why
        # this has to be explicit rather than "send both and hope".
        if self._llm.thinking_control == "reasoning_effort":
            extra_body["reasoning_effort"] = "medium" if profile_cfg.enable_thinking else "none"
        else:
            extra_body["chat_template_kwargs"] = {"enable_thinking": profile_cfg.enable_thinking}

        kwargs: dict[str, Any] = {
            "model": self._llm.model,
            "messages": messages,
            "temperature": profile_cfg.temperature,
            "top_p": profile_cfg.top_p,
            "presence_penalty": profile_cfg.presence_penalty,
            "max_tokens": profile_cfg.max_tokens,
            "extra_body": extra_body,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            kwargs["parallel_tool_calls"] = False
        return kwargs

    def _call_with_retries(self, fn: Callable[[], Any]) -> Any:
        start = time.monotonic()
        attempt = 0
        while True:
            try:
                return fn()
            except openai.APITimeoutError as exc:
                raise LLMTimeout(str(exc), time.monotonic() - start) from exc
            except openai.APIConnectionError:
                if attempt >= len(_RETRY_DELAYS_S):
                    raise
                time.sleep(_RETRY_DELAYS_S[attempt])
                attempt += 1
            except openai.APIStatusError as exc:
                status = exc.status_code
                if 500 <= status < 600 and attempt < len(_RETRY_DELAYS_S):
                    time.sleep(_RETRY_DELAYS_S[attempt])
                    attempt += 1
                    continue
                raise

    def _reasoning_source_for(self, forced_field: bool) -> str:
        return "field" if forced_field else "inline_tags"

    def _chat_once(self, profile_cfg, messages, tools, stream, on_reasoning, on_content) -> ChatResult:
        kwargs = self._build_request_kwargs(profile_cfg, messages, tools)
        if stream:
            return self._run_stream(kwargs, on_reasoning, on_content)
        return self._run_once(kwargs)

    def _run_once(self, kwargs: dict[str, Any]) -> ChatResult:
        start = time.monotonic()
        response = self._call_with_retries(lambda: self._client.chat.completions.create(**kwargs))
        total_s = time.monotonic() - start

        message = response.choices[0].message
        raw_content = message.content or ""
        reasoning_field = _reasoning_from(message)

        mode = self._llm.reasoning_mode
        if mode == "field":
            content, reasoning = raw_content, (reasoning_field or "")
            reasoning_source = "field" if reasoning else "none"
        elif mode == "inline_tags":
            content, reasoning, found = _split_inline_think(raw_content)
            reasoning_source = "inline_tags" if found else "none"
        else:  # auto
            if reasoning_field:
                content, reasoning, reasoning_source = raw_content, reasoning_field, "field"
            else:
                content, reasoning, found = _split_inline_think(raw_content)
                reasoning_source = "inline_tags" if found else "none"

        tool_calls: list[ToolCall] = []
        for tc in message.tool_calls or []:
            raw_args = tc.function.arguments or ""
            parsed, err = _try_parse_json(raw_args)
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments_raw=raw_args, arguments=parsed, parse_error=err)
            )

        usage = self._usage_from_response(response, content, reasoning, tool_calls)
        decode_tps = usage.completion_tokens / total_s if total_s > 0 and usage.completion_tokens else None
        return ChatResult(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            usage=usage,
            timings=Timings(ttft_s=None, total_s=total_s, decode_tps=decode_tps),
            reasoning_source=reasoning_source,
        )

    def _usage_from_response(self, response, content: str, reasoning: str, tool_calls: list[ToolCall]) -> Usage:
        usage = getattr(response, "usage", None)
        if usage is not None:
            return Usage(prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens, estimated=False)
        args_text = "".join(tc.arguments_raw for tc in tool_calls)
        return Usage(
            prompt_tokens=0,
            completion_tokens=_estimate_tokens(content + reasoning + args_text),
            estimated=True,
        )

    def _run_stream(self, kwargs: dict[str, Any], on_reasoning, on_content) -> ChatResult:
        kwargs = dict(kwargs)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        mode = self._llm.reasoning_mode
        start = time.monotonic()
        stream = self._call_with_retries(lambda: self._client.chat.completions.create(**kwargs))

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        reasoning_field_seen = False
        think_state = _InlineThinkStreamState()
        tool_frags: dict[int, dict[str, Any]] = {}
        usage_obj = None
        first_token_time: float | None = None

        def note_first_token() -> None:
            nonlocal first_token_time
            if first_token_time is None:
                first_token_time = time.monotonic()

        for chunk in stream:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage_obj = chunk_usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            reasoning_delta = _reasoning_from(delta) if mode != "inline_tags" else None
            if reasoning_delta:
                reasoning_field_seen = True
                note_first_token()
                reasoning_parts.append(reasoning_delta)
                if on_reasoning:
                    on_reasoning(reasoning_delta)

            content_delta = getattr(delta, "content", None)
            if content_delta:
                note_first_token()
                if mode == "field" or (mode == "auto" and reasoning_field_seen):
                    content_parts.append(content_delta)
                    if on_content:
                        on_content(content_delta)
                else:
                    # mode == "inline_tags", or "auto" with no reasoning_content seen yet
                    c_out, r_out = think_state.feed(content_delta)
                    if c_out:
                        content_parts.append(c_out)
                        if on_content:
                            on_content(c_out)
                    if r_out:
                        reasoning_parts.append(r_out)
                        if on_reasoning:
                            on_reasoning(r_out)

            tool_calls_delta = getattr(delta, "tool_calls", None)
            if tool_calls_delta:
                note_first_token()
                for tc in tool_calls_delta:
                    frag = tool_frags.setdefault(tc.index, {"id": None, "name": None, "arguments": ""})
                    if tc.id:
                        frag["id"] = tc.id
                    if tc.function is not None:
                        if tc.function.name:
                            frag["name"] = tc.function.name
                        if tc.function.arguments:
                            frag["arguments"] += tc.function.arguments

        c_tail, r_tail = think_state.flush_remainder()
        if c_tail:
            content_parts.append(c_tail)
            if on_content:
                on_content(c_tail)
        if r_tail:
            reasoning_parts.append(r_tail)
            if on_reasoning:
                on_reasoning(r_tail)

        total_s = time.monotonic() - start
        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)

        if mode == "field":
            reasoning_source = "field" if reasoning else "none"
        elif mode == "inline_tags":
            reasoning_source = "inline_tags" if think_state.found_think else "none"
        else:
            if reasoning_field_seen:
                reasoning_source = "field"
            elif think_state.found_think:
                reasoning_source = "inline_tags"
            else:
                reasoning_source = "none"

        tool_calls: list[ToolCall] = []
        for idx in sorted(tool_frags):
            frag = tool_frags[idx]
            raw_args = frag["arguments"]
            parsed, err = _try_parse_json(raw_args)
            tool_calls.append(
                ToolCall(
                    id=frag["id"] or f"call_{idx}",
                    name=frag["name"] or "",
                    arguments_raw=raw_args,
                    arguments=parsed,
                    parse_error=err,
                )
            )

        if usage_obj is not None:
            usage = Usage(
                prompt_tokens=usage_obj.prompt_tokens, completion_tokens=usage_obj.completion_tokens, estimated=False
            )
        else:
            args_text = "".join(tc.arguments_raw for tc in tool_calls)
            usage = Usage(prompt_tokens=0, completion_tokens=_estimate_tokens(content + reasoning + args_text), estimated=True)

        ttft_s = (first_token_time - start) if first_token_time is not None else None
        decode_tps = usage.completion_tokens / total_s if total_s > 0 and usage.completion_tokens else None

        return ChatResult(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            usage=usage,
            timings=Timings(ttft_s=ttft_s, total_s=total_s, decode_tps=decode_tps),
            reasoning_source=reasoning_source,
        )
