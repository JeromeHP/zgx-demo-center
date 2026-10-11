"""Tool-calling orchestrator loop with harness guardrails (Phase 6.3)."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from app.agent.prompts import (
    DISCOVERY_TASK_BRIEF,
    FULL_SCOPE_GUIDANCE,
    GRADE_TASK_BRIEF,
    INTAKE_SCOPE_GUIDANCE,
    quick_scope_guidance,
    discovery_prompt,
    grade_prompt,
    orchestrator_prompt,
    task_brief,
)
from app.agent.state import RunState
from app.agent.tools import ToolContext, execute_tool, schemas_for
from app.llm_client import LLMClient
from app.metrics import cost_equivalent, human_baseline_minutes

_MAX_IDENTICAL_REPEATS = 3
_MAX_NUDGES = 3
_HISTORY_FRACTION_TRIGGER = 0.65
_KEEP_RECENT_TOOL_MESSAGES = 6


def _summarize(name: str, result: dict[str, Any]) -> str:
    if "error" in result:
        return f"{name}: error ({result['error']})"
    if name == "search_vault":
        hits = result.get("hits", [])
        top = hits[0]["doc_id"] if hits else "none"
        return f"search_vault: {len(hits)} hits, top doc_id={top}"
    if name == "read_document":
        return f"read_document: {result.get('doc_id')} ({len(result.get('text', ''))} chars)"
    if name == "check_budget":
        return f"check_budget: total=${result.get('total_usd', 0):,}, {len(result.get('issues', []))} issue(s)"
    if name == "draft_section":
        return f"draft_section: {result.get('section_id')} v{result.get('version')}, {result.get('word_count')} words"
    if name == "review_section":
        return f"review_section: mean_score={result.get('mean_score')}, passed={result.get('passed')}"
    if name == "verify_citations":
        return f"verify_citations: passed={result.get('passed')}, unsupported={len(result.get('unsupported_claims', []))}"
    if name == "update_compliance":
        if "updated" in result:
            return f"update_compliance: {len(result['updated'])} updated, {len(result.get('rejected') or [])} rejected"
        return f"update_compliance: {result.get('requirement_id')} -> {result.get('status')}"
    if name == "flag_issue":
        return f"flag_issue: {result.get('issue_id', 'rejected')}"
    if name == "get_requirements":
        return f"get_requirements: {len(result.get('requirements', []))} requirement(s)"
    if name == "check_claims":
        return (f"check_claims: {result.get('checked', 0)} claim(s), "
                f"{len(result.get('contradicted') or [])} contradicted")
    if name == "draft_profile":
        return f"draft_profile: v{result.get('version')}, {result.get('word_count')} words"
    if name == "search_grants":
        if result.get("error"):
            return f"search_grants: {result['error']}"
        return f"search_grants: {result.get('hit_count', 0)} hits for {result.get('terms_sent')}"
    if name == "get_opportunity":
        opp = result.get("opportunity") or {}
        return f"get_opportunity: {opp.get('number', result.get('error', '?'))}"
    if name == "fetch_url":
        return f"fetch_url: {result.get('doc_id', result.get('error', '?'))}"
    if name == "search_web":
        return f"search_web: {len(result.get('results', []))} result(s)" if not result.get("error") else f"search_web: {result['error']}"
    if name == "propose_opportunity":
        return f"propose_opportunity: {result.get('number', result.get('error', '?'))}"
    if name == "finish":
        return f"finish: accepted={result.get('accepted')}"
    return f"{name}: ok"


def _result_for_model(result: dict[str, Any], max_chars: int) -> str:
    text = json.dumps(result)
    if len(text) <= max_chars:
        return text
    return json.dumps({"truncated": True, "original_chars": len(text), "preview": text[: max(0, max_chars - 200)]})


def _estimate_tokens_for_messages(messages: list[dict[str, Any]]) -> float:
    chars = sum(len(json.dumps(m, default=str)) for m in messages)
    return chars / 3.5


def _compact_history_if_needed(
    messages: list[dict[str, Any]], context_tokens: int, tool_summaries: dict[str, str]
) -> list[dict[str, Any]]:
    if _estimate_tokens_for_messages(messages) <= _HISTORY_FRACTION_TRIGGER * context_tokens:
        return messages

    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    keep_recent = set(tool_indices[-_KEEP_RECENT_TOOL_MESSAGES:])

    compacted: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        if m.get("role") == "tool" and i not in keep_recent:
            call_id = m.get("tool_call_id", "")
            summary = tool_summaries.get(call_id, "previous result")
            m = {**m, "content": json.dumps({"truncated": True, "summary": summary})}
        compacted.append(m)
    return compacted


def run_orchestrator(
    ctx: ToolContext,
    llm: LLMClient,
    scope_name: str,
    sections: list[str],
    max_steps: int,
    solicitation_info: dict[str, Any],
    is_cancelled: Callable[[], bool] = lambda: False,
) -> RunState:
    state = ctx.state
    ctx.scope_sections = sections

    # The playbook decides the prompt, the brief, and which tools the model is even
    # offered. A discovery run has no solicitation and must not be handed
    # draft_section/check_budget: the surest way to get an agent to do the wrong job
    # is to leave the wrong tools within reach.
    if ctx.playbook == "discovery":
        system_prompt = discovery_prompt(
            institution_name=ctx.institution_name,
            today=solicitation_info.get("today", ""),
        )
        brief = DISCOVERY_TASK_BRIEF
    elif ctx.playbook == "grade":
        system_prompt = grade_prompt(
            institution_name=ctx.institution_name,
            today=solicitation_info.get("today", ""),
            solicitation_id=solicitation_info.get("solicitation_id", ""),
            program=solicitation_info.get("program", ""),
            sections=", ".join(sections) or "(none loaded)",
        )
        brief = GRADE_TASK_BRIEF
    else:
        system_prompt = orchestrator_prompt(
            institution_name=ctx.institution_name,
            today=solicitation_info.get("today", ""),
            deadline=solicitation_info.get("deadline", ""),
            solicitation_id=solicitation_info.get("solicitation_id", ""),
            program=solicitation_info.get("program", ""),
            track_name=solicitation_info.get("track_name", ""),
            scope_name=scope_name,
            sections=", ".join(sections),
            # Under an uploaded solicitation there are no known partner letters to
            # name, so the quick/full guidance would send the agent hunting for
            # documents that are not there. Otherwise the quick guidance names the
            # letters of THIS scenario's partners -- see quick_scope_guidance().
            letters_and_compliance_guidance=(
                INTAKE_SCOPE_GUIDANCE
                if ctx.uploaded_solicitation
                else (
                    quick_scope_guidance(ctx.partner_names)
                    if scope_name == "quick"
                    else FULL_SCOPE_GUIDANCE
                )
            ),
        )
        brief = task_brief(scope_name=scope_name, sections=", ".join(sections))

    tool_schemas = schemas_for(ctx.playbook)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": brief},
    ]

    nudges = 0
    last_signature: tuple[str, str] | None = None
    repeat_count = 0
    tool_summaries: dict[str, str] = {}
    run_start = time.monotonic()
    last_metrics_emit = 0.0

    def emit_metrics(force: bool = False) -> None:
        nonlocal last_metrics_emit
        now = time.monotonic()
        if not force and now - last_metrics_emit < 1.0:
            return
        last_metrics_emit = now
        ctx.event_bus.emit(
            "metrics",
            {
                "elapsed_s": now - run_start,
                "prompt_tokens": ctx.total_usage.prompt_tokens,
                "completion_tokens": ctx.total_usage.completion_tokens,
                "tokens_estimated": ctx.total_usage.estimated,
                "decode_tps_recent": ctx.last_decode_tps,
                "llm_calls": ctx.llm_call_count,
                "tool_calls": sum(state.tool_calls_made.values()),
                "cost_equivalent": cost_equivalent(ctx.total_usage.prompt_tokens, ctx.total_usage.completion_tokens, ctx.settings),
                "human_baseline": human_baseline_minutes(ctx.settings, ctx.scope_sections),
            },
        )

    for step in range(1, max_steps + 1):
        if is_cancelled():
            state.status = "cancelled"
            break
        state.steps = step
        ctx.event_bus.emit("step_started", {"step": step})

        def on_reasoning(delta: str) -> None:
            ctx.event_bus.emit("thinking_delta", {"text": delta})

        def on_content(delta: str) -> None:
            ctx.event_bus.emit("content_delta", {"text": delta})

        result = llm.chat(
            "orchestrator", messages, tools=tool_schemas, stream=True, on_reasoning=on_reasoning, on_content=on_content
        )
        ctx.add_usage(result.usage)
        ctx.note_llm_call(result.timings)
        emit_metrics()

        if result.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": result.content or None,
                    "tool_calls": [
                        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments_raw}}
                        for c in result.tool_calls
                    ],
                }
            )

            finished = False
            for call in result.tool_calls:
                signature = (call.name, call.arguments_raw)
                repeat_count = repeat_count + 1 if signature == last_signature else 1
                last_signature = signature

                ctx.event_bus.emit(
                    "tool_call", {"call_id": call.id, "name": call.name, "arguments": call.arguments, "display": f"Calling {call.name}"}
                )

                if repeat_count > _MAX_IDENTICAL_REPEATS:
                    tool_result: dict[str, Any] = {
                        "error": "repeated_call",
                        "detail": f"{call.name} called with identical arguments {repeat_count} times in a row; change your approach.",
                    }
                elif call.parse_error:
                    tool_result = {"error": "invalid_arguments", "detail": call.parse_error}
                else:
                    state.record_tool_call(call.name)
                    tool_result = execute_tool(ctx, call.name, call.arguments or {})

                summary = _summarize(call.name, tool_result)
                tool_summaries[call.id] = summary
                ctx.event_bus.emit(
                    "tool_result", {"call_id": call.id, "name": call.name, "ok": "error" not in tool_result, "summary": summary, "result": tool_result}
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": _result_for_model(tool_result, ctx.settings.agent.max_tool_result_chars),
                    }
                )

                state.save(ctx.runtime_dir)

                if call.name == "finish" and tool_result.get("accepted"):
                    state.status = "completed"
                    finished = True
                    break

            if finished:
                break
        else:
            nudges += 1
            if nudges > _MAX_NUDGES:
                state.status = "incomplete"
                break
            messages.append({"role": "assistant", "content": result.content or ""})
            messages.append(
                {"role": "user", "content": "Continue the task using the tools. Call finish when every step in scope is complete."}
            )

        messages = _compact_history_if_needed(messages, ctx.settings.llm.context_tokens, tool_summaries)
    else:
        state.status = "incomplete"

    if state.status == "running":
        state.status = "incomplete"

    emit_metrics(force=True)
    elapsed_s = time.monotonic() - run_start
    state.save(ctx.runtime_dir)
    ctx.event_bus.emit(
        "run_finished", {"status": state.status, "summary": state.summary, "elapsed_s": elapsed_s, "totals": state.tool_calls_made}
    )
    return state
