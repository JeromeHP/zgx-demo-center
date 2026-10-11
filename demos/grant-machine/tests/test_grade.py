"""Grading an already-written proposal (v2 Phase 3).

Two guardrails carry this mode: the agent cannot rewrite what it is assessing, and
it cannot leave a requirement unruled — a skipped requirement reads as compliant to
whoever receives the scorecard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.state import CitationCheckRecord, DraftVersion, ReviewResult, RunState
from app.agent.tools import GRADE_TOOLS, ToolContext, execute_tool, schemas_for


class _Events:
    def __init__(self):
        self.emitted = []

    def emit(self, name, payload):
        self.emitted.append((name, payload))


def _ctx(tmp_path: Path, *, requirements=None, sections=("specific_aims",)) -> ToolContext:
    state = RunState(run_id="g1", mode="grade", scope="quick")
    state.requirements = requirements if requirements is not None else [
        {"id": "R-001", "category": "eligibility", "text": "Applicant must be an accredited IHE."},
        {"id": "R-002", "category": "project_description", "text": "State measurable objectives."},
    ]
    for sid in sections:
        state.drafts[sid] = [DraftVersion(version=1, text="Submitted text.", word_count=2)]
    ctx = ToolContext(
        settings=None, llm=None, vault_dir=tmp_path, index=None, state=state,
        event_bus=_Events(), runtime_dir=tmp_path, playbook="grade",
    )
    ctx.scope_sections = list(sections)
    return ctx


# -- tool surface --------------------------------------------------------------


def test_the_grader_cannot_rewrite_what_it_is_assessing():
    """An agent asked to assess work must not be able to quietly rewrite it and then
    score its own version."""
    names = {t["function"]["name"] for t in schemas_for("grade")}
    assert "draft_section" not in names
    assert "review_section" in names
    # A submitted proposal has no [[doc_id]] markers, so citation verification is
    # meaningless here and would block every requirement.
    assert "verify_citations" not in names
    assert set(GRADE_TOOLS) == names


# -- the evidence gate ---------------------------------------------------------


def test_met_requires_pointing_at_the_section_that_satisfies_it(tmp_path):
    ctx = _ctx(tmp_path)
    out = execute_tool(ctx, "update_compliance", {
        "item": {"requirement_id": "R-001", "status": "met", "evidence": [], "note": "looks fine"}})
    assert out["error"] == "guardrail_rejected"
    assert "evidence" in out["detail"]
    assert "R-001" not in ctx.state.compliance


def test_met_is_rejected_when_the_evidence_names_no_real_section(tmp_path):
    ctx = _ctx(tmp_path)
    out = execute_tool(ctx, "update_compliance", {
        "item": {"requirement_id": "R-001", "status": "met", "evidence": ["chapter_nine"]}})
    assert out["error"] == "guardrail_rejected"
    assert "specific_aims" in out["detail"], "the error should name the sections that exist"


def test_met_is_accepted_with_real_evidence(tmp_path):
    ctx = _ctx(tmp_path)
    out = execute_tool(ctx, "update_compliance", {
        "item": {"requirement_id": "R-001", "status": "met", "evidence": ["specific_aims"]}})
    assert out["status"] == "met"
    assert ctx.state.compliance["R-001"].evidence == ["specific_aims"]


@pytest.mark.parametrize("status", ["partial", "missing", "needs_human", "not_applicable"])
def test_other_statuses_need_no_evidence(tmp_path, status):
    """Only a positive finding has to be backed up; saying something is absent does not."""
    ctx = _ctx(tmp_path)
    out = execute_tool(ctx, "update_compliance", {"item": {"requirement_id": "R-002", "status": status}})
    assert out["status"] == status


# -- the finish gate -----------------------------------------------------------


def _rule_everything(ctx, status="not_applicable"):
    for r in ctx.state.requirements:
        execute_tool(ctx, "update_compliance", {"item": {"requirement_id": r["id"], "status": status}})


def test_finish_is_refused_until_every_section_is_scored(tmp_path):
    ctx = _ctx(tmp_path, sections=("specific_aims", "approach"))
    ctx.state.tool_calls_made["get_requirements"] = 1
    _rule_everything(ctx)
    ctx.state.reviews["specific_aims"] = [ReviewResult(version=1, mean_score=3.0)]

    out = execute_tool(ctx, "finish", {"summary": "done"})
    assert out["accepted"] is False
    assert any("approach" in m for m in out["missing"])


def test_finish_is_refused_while_any_requirement_is_unruled(tmp_path):
    """A requirement nobody ruled on reads as compliant to whoever gets the scorecard."""
    ctx = _ctx(tmp_path)
    ctx.state.tool_calls_made["get_requirements"] = 1
    ctx.state.reviews["specific_aims"] = [ReviewResult(version=1, mean_score=3.0)]
    execute_tool(ctx, "update_compliance", {"item": {"requirement_id": "R-001", "status": "partial"}})

    out = execute_tool(ctx, "finish", {"summary": "done"})
    assert out["accepted"] is False
    assert any("no compliance ruling" in m for m in out["missing"])
    assert any("R-002" in str(m) for m in out["missing"])


def test_finish_is_refused_until_claims_have_been_checked(tmp_path):
    """Scoring prose without checking its figures is how a proposal claiming 120
    completers, where the records say 84, gets praised for exceeding targets."""
    ctx = _ctx(tmp_path)
    ctx.state.tool_calls_made["get_requirements"] = 1
    ctx.state.reviews["specific_aims"] = [ReviewResult(version=1, mean_score=3.0)]
    _rule_everything(ctx)

    out = execute_tool(ctx, "finish", {"summary": "done"})
    assert out["accepted"] is False
    assert any("check_claims" in m for m in out["missing"])


def test_finish_accepted_once_the_assessment_is_complete(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.state.tool_calls_made["get_requirements"] = 1
    ctx.state.reviews["specific_aims"] = [ReviewResult(version=1, mean_score=2.5)]
    ctx.state.citation_checks["specific_aims"] = [CitationCheckRecord(version=1, passed=True)]
    _rule_everything(ctx, "missing")

    out = execute_tool(ctx, "finish", {"summary": "Two requirements unmet; eligibility unconfirmed."})
    assert out["accepted"] is True and out["sections_scored"] == 1
    assert ctx.state.summary.startswith("Two requirements unmet")


def test_finish_is_refused_with_no_submission_loaded(tmp_path):
    ctx = _ctx(tmp_path, sections=())
    ctx.state.tool_calls_made["get_requirements"] = 1
    _rule_everything(ctx)
    out = execute_tool(ctx, "finish", {"summary": "done"})
    assert out["accepted"] is False
    assert any("no submission sections" in m for m in out["missing"])


# -- the proposal playbook is unaffected ---------------------------------------


def test_the_proposal_playbook_still_uses_the_citation_gate(tmp_path):
    """Grading relaxed the 'met' gate for its own mode only."""
    ctx = _ctx(tmp_path)
    ctx.playbook = "proposal"
    ctx.state.requirements = [{"id": "R-001", "category": "x", "text": "t", "section_link": "specific_aims"}]
    out = execute_tool(ctx, "update_compliance", {
        "item": {"requirement_id": "R-001", "status": "met", "evidence": ["specific_aims"]}})
    assert out["error"] == "guardrail_rejected"
    assert "verify_citations" in out["detail"]


# -- batching ------------------------------------------------------------------


def test_many_requirements_can_be_ruled_in_one_call(tmp_path):
    """Ruling on 54 requirements one call each spent 39 of a 60-step budget on
    bookkeeping and left the run no room to finish."""
    ctx = _ctx(tmp_path)
    out = execute_tool(ctx, "update_compliance", {"items": [
        {"requirement_id": "R-001", "status": "met", "evidence": ["specific_aims"]},
        {"requirement_id": "R-002", "status": "missing", "note": "no measurable objectives given"},
    ]})
    assert out["updated"] == ["R-001", "R-002"]
    assert not out["rejected"]
    assert ctx.state.compliance["R-002"].note.startswith("no measurable")


def test_a_batch_applies_the_guardrail_to_each_item(tmp_path):
    ctx = _ctx(tmp_path)
    out = execute_tool(ctx, "update_compliance", {"items": [
        {"requirement_id": "R-001", "status": "met", "evidence": ["specific_aims"]},
        {"requirement_id": "R-002", "status": "met"},  # no evidence — must be refused
    ]})
    assert out["updated"] == ["R-001"]
    assert [r["requirement_id"] for r in out["rejected"]] == ["R-002"]
    assert "R-002" not in ctx.state.compliance


def test_the_single_item_form_still_works(tmp_path):
    ctx = _ctx(tmp_path)
    out = execute_tool(ctx, "update_compliance", {"item": {"requirement_id": "R-001", "status": "partial"}})
    assert out["status"] == "partial"


def test_neither_form_supplied_is_an_error_not_a_crash(tmp_path):
    ctx = _ctx(tmp_path)
    assert execute_tool(ctx, "update_compliance", {})["error"] == "invalid_arguments"


def test_a_ruling_on_a_requirement_that_does_not_exist_is_refused(tmp_path):
    """A live run filed rulings under "BUDGET_OVER_CAP" and "FRINGE_RATE_MISMATCH" —
    issue codes, not requirement ids — and setdefault created them silently, giving a
    54-requirement solicitation 56 rows with no requirement text behind two of them."""
    ctx = _ctx(tmp_path)
    out = execute_tool(ctx, "update_compliance", {
        "item": {"requirement_id": "BUDGET_OVER_CAP", "status": "missing", "note": "over cap"}})

    assert out["error"] == "unknown_requirement"
    assert "flag_issue" in out["detail"], "should point at the right tool for this"
    assert "BUDGET_OVER_CAP" not in ctx.state.compliance
    assert set(ctx.state.compliance) <= {r["id"] for r in ctx.state.requirements}


def test_a_batch_rejects_only_the_unknown_ids(tmp_path):
    ctx = _ctx(tmp_path)
    out = execute_tool(ctx, "update_compliance", {"items": [
        {"requirement_id": "R-001", "status": "partial"},
        {"requirement_id": "FRINGE_RATE_MISMATCH", "status": "missing"},
    ]})
    assert out["updated"] == ["R-001"]
    assert [r["requirement_id"] for r in out["rejected"]] == ["FRINGE_RATE_MISMATCH"]
