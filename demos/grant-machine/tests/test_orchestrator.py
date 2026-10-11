from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.agent.orchestrator import run_orchestrator
from app.agent.state import ComplianceEntry, RunState
from app.agent.tools import ToolContext
from app.events import EventBus
from app.ingest.loaders import load
from app.llm_client import LLMClient
from app.retrieval.bm25_index import BM25Index
from tests.fake_llm_server import ScriptedResponse, ScriptedToolCall

SOLICITATION_INFO = {
    "today": "2026-09-16",
    "deadline": "2026-10-01",
    "solicitation_id": "NSF 24-584",
    "program": "ATE",
    "track_name": "Track 2 Projects",
}


def _write_vault(vault_dir: Path) -> None:
    vault_dir.mkdir(parents=True, exist_ok=True)
    (vault_dir / "budget-satp-draft.csv").write_text(
        "# doc_id: budget-satp-draft\n# title: Test Budget\n# synthetic: true\n"
        "year,category,line_item,amount_usd,fringe_applied,indirect_applied,notes\n"
        "1,senior_personnel,PI effort,50000,0,0.36,\n"
        "1,indirect,Indirect,18000,0,0.36,\n",
        encoding="utf-8",
    )
    (vault_dir / "prior-report.md").write_text(
        "---\ntitle: Prior Outcomes Report\n---\n"
        "# Prior Outcomes Report\n\nThe project produced 84 completers against a target of 120.\n",
        encoding="utf-8",
    )


def _make_ctx(
    tmp_path: Path,
    make_settings,
    fake_llm_server,
    requirements: list[dict[str, Any]] | None = None,
    settings_overrides: dict[str, Any] | None = None,
) -> tuple[ToolContext, LLMClient]:
    vault_dir = tmp_path / "vault"
    _write_vault(vault_dir)

    overrides: dict[str, Any] = {"llm": {"base_url": f"{fake_llm_server.base_url}/v1"}}
    if settings_overrides:
        from tests.conftest import deep_merge_into

        deep_merge_into(overrides, settings_overrides)
    settings = make_settings(overrides=overrides)

    index = BM25Index()
    docs = [load(p) for p in vault_dir.glob("*") if p.suffix in (".md", ".csv")]
    index.build(docs)

    state = RunState(run_id="test-run", mode="story", scope="quick", requirements=requirements or [])
    event_bus = EventBus(run_id="test-run")  # no log_path: keep runs in memory only for tests
    ctx = ToolContext(
        settings=settings,
        llm=None,  # set by caller after construction (LLMClient needs settings)
        vault_dir=vault_dir,
        index=index,
        state=state,
        event_bus=event_bus,
        runtime_dir=tmp_path / "runtime",
        institution_name="Harrow Creek Community College",
    )
    llm = LLMClient(settings)
    ctx.llm = llm
    return ctx, llm


def _tc(name: str, args: dict[str, Any], call_id: str = "call_1") -> ScriptedToolCall:
    return ScriptedToolCall(id=call_id, name=name, arguments=json.dumps(args))


REQ_RESULTS = [{"id": "R-1", "category": "project_description", "text": "Report prior results honestly.", "section_link": "results_prior_support"}]


def test_happy_path_quick_scope_completes(tmp_path, make_settings, fake_llm_app, fake_llm_server):
    ctx, llm = _make_ctx(tmp_path, make_settings, fake_llm_server, requirements=REQ_RESULTS)

    q = fake_llm_app.state.queue
    q.append(ScriptedResponse(content="", tool_calls=[_tc("get_requirements", {})]))
    q.append(ScriptedResponse(content="", tool_calls=[_tc("check_budget", {"track": "track_2"})]))
    q.append(
        ScriptedResponse(
            content="",
            tool_calls=[
                _tc(
                    "draft_section",
                    {"section_id": "results_prior_support", "evidence": [{"doc_id": "prior-report"}], "guidance": ""},
                )
            ],
        )
    )
    q.append(ScriptedResponse(content="The prior project produced 84 completers against a target of 120. [[prior-report]]"))
    q.append(ScriptedResponse(content="", tool_calls=[_tc("review_section", {"section_id": "results_prior_support"})]))
    q.append(
        ScriptedResponse(
            content=json.dumps({"criteria": [{"id": "R-1", "score": 5, "rationale": "ok", "gaps": []}], "mean_score": 5.0, "must_fix": []})
        )
    )
    q.append(ScriptedResponse(content="", tool_calls=[_tc("verify_citations", {"section_id": "results_prior_support"})]))
    q.append(
        ScriptedResponse(
            content="",
            tool_calls=[
                _tc("update_compliance", {"item": {"requirement_id": "R-1", "status": "met", "evidence": ["prior-report"], "note": "ok"}})
            ],
        )
    )
    q.append(ScriptedResponse(content="", tool_calls=[_tc("finish", {"summary": "Done."})]))

    state = run_orchestrator(ctx, llm, "quick", ["results_prior_support"], max_steps=10, solicitation_info=SOLICITATION_INFO)

    assert state.status == "completed"
    assert state.compliance["R-1"].status == "met"
    assert state.latest_review("results_prior_support").passed
    assert state.latest_citation_check("results_prior_support").passed
    event_types = [e.type for e in ctx.event_bus.events]
    assert event_types[0] == "step_started"
    assert "draft_completed" in event_types
    assert "section_scored" in event_types
    assert "citations_checked" in event_types
    assert event_types[-1] == "run_finished"


def test_malformed_tool_args_then_model_corrects(tmp_path, make_settings, fake_llm_app, fake_llm_server):
    ctx, llm = _make_ctx(tmp_path, make_settings, fake_llm_server, requirements=[])
    q = fake_llm_app.state.queue
    # malformed JSON arguments (truncated)
    q.append(
        ScriptedResponse(
            content="",
            tool_calls=[ScriptedToolCall(id="call_1", name="search_vault", arguments='{"query": "oops much longer than')],
        )
    )
    q.append(ScriptedResponse(content="", tool_calls=[_tc("search_vault", {"query": "prior outcomes"}, call_id="call_2")]))

    state = run_orchestrator(ctx, llm, "quick", [], max_steps=2, solicitation_info=SOLICITATION_INFO)

    tool_results = [e for e in ctx.event_bus.events if e.type == "tool_result"]
    assert tool_results[0].data["ok"] is False
    assert tool_results[0].data["result"]["error"] == "invalid_arguments"
    assert tool_results[1].data["ok"] is True
    assert state.tool_calls_made.get("search_vault") == 1  # only the corrected call was actually executed


def test_finish_rejected_early_then_completes(tmp_path, make_settings, fake_llm_app, fake_llm_server):
    ctx, llm = _make_ctx(tmp_path, make_settings, fake_llm_server, requirements=[])
    q = fake_llm_app.state.queue
    q.append(ScriptedResponse(content="", tool_calls=[_tc("finish", {"summary": "too early"})]))
    q.append(ScriptedResponse(content="", tool_calls=[_tc("get_requirements", {})]))
    q.append(ScriptedResponse(content="", tool_calls=[_tc("check_budget", {"track": "track_2"})]))
    q.append(ScriptedResponse(content="", tool_calls=[_tc("finish", {"summary": "now complete"})]))

    state = run_orchestrator(ctx, llm, "quick", [], max_steps=10, solicitation_info=SOLICITATION_INFO)

    tool_results = [e for e in ctx.event_bus.events if e.type == "tool_result"]
    first_finish = tool_results[0]
    assert first_finish.data["result"]["accepted"] is False
    assert first_finish.data["result"]["missing"]
    assert state.status == "completed"
    assert state.summary == "now complete"


def test_revision_cap_reached_marks_needs_human(tmp_path, make_settings, fake_llm_app, fake_llm_server):
    ctx, llm = _make_ctx(
        tmp_path,
        make_settings,
        fake_llm_server,
        requirements=REQ_RESULTS,
        settings_overrides={"agent": {"max_revisions_per_section": 0}},
    )
    ctx.state.compliance["R-1"] = ComplianceEntry(requirement_id="R-1")

    q = fake_llm_app.state.queue
    q.append(
        ScriptedResponse(
            content="",
            tool_calls=[_tc("draft_section", {"section_id": "results_prior_support", "evidence": [{"doc_id": "prior-report"}]})],
        )
    )
    q.append(ScriptedResponse(content="The project produced 84 completers. [[prior-report]]"))
    q.append(ScriptedResponse(content="", tool_calls=[_tc("review_section", {"section_id": "results_prior_support"})]))
    q.append(
        ScriptedResponse(
            content=json.dumps(
                {"criteria": [{"id": "R-1", "score": 2, "rationale": "weak", "gaps": ["more detail"]}], "mean_score": 2.0, "must_fix": ["add detail"]}
            )
        )
    )

    run_orchestrator(ctx, llm, "quick", ["results_prior_support"], max_steps=2, solicitation_info=SOLICITATION_INFO)

    assert ctx.state.compliance["R-1"].status == "needs_human"
    assert not ctx.state.latest_review("results_prior_support").passed


def test_draft_section_refuses_once_revision_cap_reached(tmp_path, make_settings, fake_llm_app, fake_llm_server):
    # Regression test: after max_revisions_per_section is reached, draft_section
    # must refuse further calls for that section, not silently redraft it again
    # (an earlier bug let a live e2e run draft the same section 5-7 times).
    ctx, llm = _make_ctx(
        tmp_path,
        make_settings,
        fake_llm_server,
        requirements=REQ_RESULTS,
        settings_overrides={"agent": {"max_revisions_per_section": 0}},
    )
    ctx.state.compliance["R-1"] = ComplianceEntry(requirement_id="R-1")

    q = fake_llm_app.state.queue
    q.append(
        ScriptedResponse(
            content="",
            tool_calls=[_tc("draft_section", {"section_id": "results_prior_support", "evidence": [{"doc_id": "prior-report"}]}, call_id="c1")],
        )
    )
    q.append(ScriptedResponse(content="The project produced 84 completers. [[prior-report]]"))
    q.append(ScriptedResponse(content="", tool_calls=[_tc("review_section", {"section_id": "results_prior_support"}, call_id="c2")]))
    q.append(
        ScriptedResponse(
            content=json.dumps(
                {"criteria": [{"id": "R-1", "score": 2, "rationale": "weak", "gaps": ["more"]}], "mean_score": 2.0, "must_fix": ["add detail"]}
            )
        )
    )
    # cap is 0, so this second draft attempt must be refused, not executed
    q.append(
        ScriptedResponse(
            content="",
            tool_calls=[_tc("draft_section", {"section_id": "results_prior_support", "evidence": [{"doc_id": "prior-report"}]}, call_id="c3")],
        )
    )

    run_orchestrator(ctx, llm, "quick", ["results_prior_support"], max_steps=3, solicitation_info=SOLICITATION_INFO)

    tool_results = [e for e in ctx.event_bus.events if e.type == "tool_result"]
    second_draft_result = [r for r in tool_results if r.data["call_id"] == "c3"][0]
    assert second_draft_result.data["ok"] is False
    assert second_draft_result.data["result"]["error"] == "revision_cap_reached"
    assert len(ctx.state.drafts["results_prior_support"]) == 1  # no second version was ever created


def test_max_steps_reached_marks_incomplete(tmp_path, make_settings, fake_llm_app, fake_llm_server):
    ctx, llm = _make_ctx(tmp_path, make_settings, fake_llm_server, requirements=[])
    q = fake_llm_app.state.queue
    q.append(ScriptedResponse(content="", tool_calls=[_tc("search_vault", {"query": "prior outcomes"}, call_id="c1")]))
    q.append(ScriptedResponse(content="", tool_calls=[_tc("search_vault", {"query": "budget details"}, call_id="c2")]))

    state = run_orchestrator(ctx, llm, "quick", [], max_steps=2, solicitation_info=SOLICITATION_INFO)

    assert state.status == "incomplete"
    assert state.steps == 2


def test_cancellation_mid_run(tmp_path, make_settings, fake_llm_app, fake_llm_server):
    ctx, llm = _make_ctx(tmp_path, make_settings, fake_llm_server, requirements=[])
    q = fake_llm_app.state.queue
    q.append(ScriptedResponse(content="", tool_calls=[_tc("search_vault", {"query": "prior outcomes"})]))

    calls: list[int] = []

    def is_cancelled() -> bool:
        calls.append(1)
        return len(calls) > 1

    state = run_orchestrator(
        ctx, llm, "quick", [], max_steps=10, solicitation_info=SOLICITATION_INFO, is_cancelled=is_cancelled
    )

    assert state.status == "cancelled"
    assert state.steps == 1


def test_inline_think_reasoning_reaches_thinking_delta_events(tmp_path, make_settings, fake_llm_app, fake_llm_server):
    ctx, llm = _make_ctx(tmp_path, make_settings, fake_llm_server, requirements=[])
    q = fake_llm_app.state.queue
    q.append(
        ScriptedResponse(content="", inline_think="considering the best search query", tool_calls=[_tc("search_vault", {"query": "prior outcomes"})])
    )

    run_orchestrator(ctx, llm, "quick", [], max_steps=1, solicitation_info=SOLICITATION_INFO)

    reasoning_text = "".join(e.data["text"] for e in ctx.event_bus.events if e.type == "thinking_delta")
    assert reasoning_text == "considering the best search query"


def test_missing_usage_marks_estimated(tmp_path, make_settings, fake_llm_app, fake_llm_server):
    ctx, llm = _make_ctx(tmp_path, make_settings, fake_llm_server, requirements=[])
    q = fake_llm_app.state.queue
    q.append(ScriptedResponse(content="", tool_calls=[_tc("search_vault", {"query": "prior outcomes"})], usage=None))

    run_orchestrator(ctx, llm, "quick", [], max_steps=1, solicitation_info=SOLICITATION_INFO)

    assert ctx.total_usage.estimated is True


def test_metrics_event_emitted_with_real_data(tmp_path, make_settings, fake_llm_app, fake_llm_server):
    ctx, llm = _make_ctx(tmp_path, make_settings, fake_llm_server, requirements=[])
    q = fake_llm_app.state.queue
    q.append(
        ScriptedResponse(
            content="", tool_calls=[_tc("search_vault", {"query": "prior outcomes"})], usage={"prompt_tokens": 50, "completion_tokens": 10}
        )
    )

    run_orchestrator(ctx, llm, "quick", [], max_steps=1, solicitation_info=SOLICITATION_INFO)

    metrics_events = [e for e in ctx.event_bus.events if e.type == "metrics"]
    assert metrics_events
    last = metrics_events[-1].data
    assert last["prompt_tokens"] == 50
    assert last["completion_tokens"] == 10
    assert last["llm_calls"] == 1
    assert last["elapsed_s"] >= 0
    assert last["cost_equivalent"] == {"configured": False, "usd": None, "label": None}
