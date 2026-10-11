from __future__ import annotations

import pytest

from app.llm_client import LLMClient
from app.settings import SettingsError
from tests.fake_llm_server import ScriptedResponse, ScriptedToolCall


def _client_for(fake_llm_server, make_settings) -> LLMClient:
    settings = make_settings(overrides={"llm": {"base_url": f"{fake_llm_server.base_url}/v1"}})
    return LLMClient(settings)


# -- reasoning normalization -------------------------------------------------


def test_reasoning_content_field_non_stream(make_settings, fake_llm_app, fake_llm_server):
    fake_llm_app.state.queue.append(
        ScriptedResponse(content="the answer", reasoning_content="thinking it through")
    )
    client = _client_for(fake_llm_server, make_settings)

    result = client.chat("orchestrator", [{"role": "user", "content": "hi"}])

    assert result.content == "the answer"
    assert result.reasoning == "thinking it through"
    assert result.reasoning_source == "field"


def test_ollama_reasoning_field_non_stream(make_settings, fake_llm_app, fake_llm_server):
    """Ollama names the separated-reasoning field `reasoning`, not `reasoning_content`."""
    fake_llm_app.state.queue.append(
        ScriptedResponse(
            content="the answer",
            reasoning_content="thinking it through",
            reasoning_field_name="reasoning",
        )
    )
    client = _client_for(fake_llm_server, make_settings)

    result = client.chat("orchestrator", [{"role": "user", "content": "hi"}])

    assert result.content == "the answer"
    assert result.reasoning == "thinking it through"
    assert result.reasoning_source == "field"


def test_ollama_reasoning_field_streaming_reaches_callback(make_settings, fake_llm_app, fake_llm_server):
    fake_llm_app.state.queue.append(
        ScriptedResponse(
            content="streamed answer",
            reasoning_content="streamed reasoning",
            reasoning_field_name="reasoning",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
    )
    client = _client_for(fake_llm_server, make_settings)

    seen: list[str] = []
    result = client.chat(
        "orchestrator", [{"role": "user", "content": "hi"}], stream=True, on_reasoning=seen.append
    )

    assert result.reasoning == "streamed reasoning"
    assert result.reasoning_source == "field"
    assert "".join(seen) == "streamed reasoning"


def test_inline_think_non_stream(make_settings, fake_llm_app, fake_llm_server):
    fake_llm_app.state.queue.append(ScriptedResponse(content="the answer", inline_think="thinking it through"))
    client = _client_for(fake_llm_server, make_settings)

    result = client.chat("orchestrator", [{"role": "user", "content": "hi"}])

    assert result.content == "the answer"
    assert result.reasoning == "thinking it through"
    assert result.reasoning_source == "inline_tags"
    assert "<think>" not in result.content


def test_reasoning_content_field_streaming(make_settings, fake_llm_app, fake_llm_server):
    fake_llm_app.state.queue.append(
        ScriptedResponse(
            content="streamed answer",
            reasoning_content="streamed reasoning",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
    )
    client = _client_for(fake_llm_server, make_settings)

    result = client.chat("orchestrator", [{"role": "user", "content": "hi"}], stream=True)

    assert result.content == "streamed answer"
    assert result.reasoning == "streamed reasoning"
    assert result.reasoning_source == "field"


def test_inline_think_split_across_stream_chunks(make_settings, fake_llm_app, fake_llm_server):
    # stream_chunk_size=3 forces the <think> / </think> tags themselves to straddle chunk boundaries
    fake_llm_app.state.queue.append(
        ScriptedResponse(content="final answer here", inline_think="deep thought", stream_chunk_size=3)
    )
    client = _client_for(fake_llm_server, make_settings)

    result = client.chat("orchestrator", [{"role": "user", "content": "hi"}], stream=True)

    assert result.content == "final answer here"
    assert result.reasoning == "deep thought"
    assert result.reasoning_source == "inline_tags"
    assert "<think>" not in result.content
    assert "</think>" not in result.content


def test_reasoning_deltas_streamed_to_callback(make_settings, fake_llm_app, fake_llm_server):
    fake_llm_app.state.queue.append(ScriptedResponse(content="answer", reasoning_content="reasoning text"))
    client = _client_for(fake_llm_server, make_settings)

    reasoning_chunks: list[str] = []
    content_chunks: list[str] = []
    client.chat(
        "orchestrator",
        [{"role": "user", "content": "hi"}],
        stream=True,
        on_reasoning=reasoning_chunks.append,
        on_content=content_chunks.append,
    )

    assert "".join(reasoning_chunks) == "reasoning text"
    assert "".join(content_chunks) == "answer"


def test_content_deltas_streamed_to_callback_without_reasoning_field(make_settings, fake_llm_app, fake_llm_server):
    # Regression test: no reasoning_content field at all means every content
    # delta goes through the inline-<think> buffering state machine (mode
    # "auto" with reasoning_field_seen=False), which withholds a short tail
    # at each step. That tail must still reach on_content via flush_remainder,
    # not just end up in the final ChatResult.content.
    long_content = "This is the project_summary draft. The project produced 84 completers against a target of 120. [[prior-annual-report-2025]]"
    fake_llm_app.state.queue.append(ScriptedResponse(content=long_content, stream_chunk_size=6))
    client = _client_for(fake_llm_server, make_settings)

    content_chunks: list[str] = []
    result = client.chat(
        "drafter", [{"role": "user", "content": "hi"}], stream=True, on_content=content_chunks.append
    )

    assert "".join(content_chunks) == long_content
    assert result.content == long_content


# -- tool calls ---------------------------------------------------------------


def test_streaming_tool_call_assembly_across_fragments(make_settings, fake_llm_app, fake_llm_server):
    fake_llm_app.state.queue.append(
        ScriptedResponse(
            content="",
            tool_calls=[
                ScriptedToolCall(id="call_1", name="search_vault", arguments='{"query": "prior outcomes", "k": 5}')
            ],
            tool_arg_fragments=4,
        )
    )
    client = _client_for(fake_llm_server, make_settings)

    result = client.chat(
        "orchestrator", [{"role": "user", "content": "hi"}], tools=[{"type": "function"}], stream=True
    )

    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.name == "search_vault"
    assert call.parse_error is None
    assert call.arguments == {"query": "prior outcomes", "k": 5}


def test_malformed_tool_args_set_parse_error(make_settings, fake_llm_app, fake_llm_server):
    fake_llm_app.state.queue.append(
        ScriptedResponse(
            content="",
            tool_calls=[ScriptedToolCall(id="call_1", name="search_vault", arguments='{"query": "oops",')],
        )
    )
    client = _client_for(fake_llm_server, make_settings)

    result = client.chat("orchestrator", [{"role": "user", "content": "hi"}], tools=[{"type": "function"}])

    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.arguments is None
    assert call.parse_error is not None


def test_malformed_tool_args_streaming_set_parse_error(make_settings, fake_llm_app, fake_llm_server):
    fake_llm_app.state.queue.append(
        ScriptedResponse(
            content="",
            tool_calls=[ScriptedToolCall(id="call_1", name="search_vault", arguments='{"query": "oops",')],
            tool_arg_fragments=3,
        )
    )
    client = _client_for(fake_llm_server, make_settings)

    result = client.chat(
        "orchestrator", [{"role": "user", "content": "hi"}], tools=[{"type": "function"}], stream=True
    )

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].arguments is None
    assert result.tool_calls[0].parse_error is not None


# -- usage ----------------------------------------------------------------------


def test_missing_usage_sets_estimated_true(make_settings, fake_llm_app, fake_llm_server):
    fake_llm_app.state.queue.append(ScriptedResponse(content="a reasonably sized answer", usage=None))
    client = _client_for(fake_llm_server, make_settings)

    result = client.chat("drafter", [{"role": "user", "content": "hi"}])

    assert result.usage.estimated is True
    assert result.usage.completion_tokens > 0


def test_present_usage_is_not_estimated(make_settings, fake_llm_app, fake_llm_server):
    fake_llm_app.state.queue.append(
        ScriptedResponse(content="answer", usage={"prompt_tokens": 42, "completion_tokens": 7})
    )
    client = _client_for(fake_llm_server, make_settings)

    result = client.chat("drafter", [{"role": "user", "content": "hi"}])

    assert result.usage.estimated is False
    assert result.usage.prompt_tokens == 42
    assert result.usage.completion_tokens == 7


def test_missing_usage_streaming_sets_estimated_true(make_settings, fake_llm_app, fake_llm_server):
    fake_llm_app.state.queue.append(ScriptedResponse(content="a reasonably sized streamed answer", usage=None))
    client = _client_for(fake_llm_server, make_settings)

    result = client.chat("drafter", [{"role": "user", "content": "hi"}], stream=True)

    assert result.usage.estimated is True
    assert result.usage.completion_tokens > 0


# -- timings / decode tps (Spec I) ------------------------------------------------


def test_decode_tps_computed_from_usage_and_elapsed_time(make_settings, fake_llm_app, fake_llm_server):
    fake_llm_app.state.queue.append(
        ScriptedResponse(content="answer text here", usage={"prompt_tokens": 10, "completion_tokens": 50})
    )
    client = _client_for(fake_llm_server, make_settings)

    result = client.chat("drafter", [{"role": "user", "content": "hi"}], stream=True)

    assert result.timings.total_s > 0
    assert result.timings.decode_tps is not None
    assert result.timings.decode_tps == pytest.approx(result.usage.completion_tokens / result.timings.total_s)


def test_decode_tps_none_when_no_completion_tokens(make_settings, fake_llm_app, fake_llm_server):
    fake_llm_app.state.queue.append(ScriptedResponse(content="", usage={"prompt_tokens": 10, "completion_tokens": 0}))
    client = _client_for(fake_llm_server, make_settings)

    result = client.chat("drafter", [{"role": "user", "content": "hi"}])

    assert result.timings.decode_tps is None


def test_ttft_recorded_for_streaming_before_total_time(make_settings, fake_llm_app, fake_llm_server):
    fake_llm_app.state.queue.append(ScriptedResponse(content="some streamed content", usage={"prompt_tokens": 5, "completion_tokens": 5}))
    client = _client_for(fake_llm_server, make_settings)

    result = client.chat("drafter", [{"role": "user", "content": "hi"}], stream=True)

    assert result.timings.ttft_s is not None
    assert result.timings.ttft_s <= result.timings.total_s


# -- sampling params land in extra_body correctly --------------------------------


def test_sampling_params_reach_the_server(make_settings, fake_llm_app, fake_llm_server):
    fake_llm_app.state.queue.append(ScriptedResponse(content="ok"))
    settings = make_settings(overrides={"llm": {"base_url": f"{fake_llm_server.base_url}/v1"}})
    client = LLMClient(settings)

    client.chat("reviewer", [{"role": "user", "content": "hi"}])

    assert len(fake_llm_app.state.received_requests) == 1
    body = fake_llm_app.state.received_requests[0]
    profile = settings.llm.profiles["reviewer"]

    assert body["temperature"] == profile.temperature
    assert body["top_p"] == profile.top_p
    assert body["presence_penalty"] == profile.presence_penalty
    assert body["max_tokens"] == profile.max_tokens
    assert body["top_k"] == profile.top_k
    assert body["min_p"] == profile.min_p
    assert body["chat_template_kwargs"]["enable_thinking"] == profile.enable_thinking


@pytest.mark.parametrize(
    "profile,expected_effort",
    [("orchestrator", "medium"), ("reviewer", "none")],  # thinking on / off
)
def test_reasoning_effort_sent_instead_of_chat_template_kwargs(
    make_settings, fake_llm_app, fake_llm_server, profile, expected_effort
):
    """thinking_control=reasoning_effort is for Ollama, which ignores chat_template_kwargs."""
    fake_llm_app.state.queue.append(ScriptedResponse(content="ok"))
    settings = make_settings(
        overrides={
            "llm": {
                "base_url": f"{fake_llm_server.base_url}/v1",
                "thinking_control": "reasoning_effort",
            }
        }
    )

    LLMClient(settings).chat(profile, [{"role": "user", "content": "hi"}])

    body = fake_llm_app.state.received_requests[0]
    assert body["reasoning_effort"] == expected_effort
    assert "chat_template_kwargs" not in body


def test_chat_template_kwargs_is_the_default_thinking_control(make_settings, fake_llm_app, fake_llm_server):
    """The llama-server request shape must be unchanged when the setting is absent."""
    fake_llm_app.state.queue.append(ScriptedResponse(content="ok"))
    settings = make_settings(overrides={"llm": {"base_url": f"{fake_llm_server.base_url}/v1"}})

    assert settings.llm.thinking_control == "chat_template"
    LLMClient(settings).chat("orchestrator", [{"role": "user", "content": "hi"}])

    body = fake_llm_app.state.received_requests[0]
    assert body["chat_template_kwargs"]["enable_thinking"] is True
    assert "reasoning_effort" not in body


def test_unknown_profile_raises(make_settings, fake_llm_app, fake_llm_server):
    client = _client_for(fake_llm_server, make_settings)

    with pytest.raises(SettingsError):
        client.chat("not_a_real_profile", [{"role": "user", "content": "hi"}])
