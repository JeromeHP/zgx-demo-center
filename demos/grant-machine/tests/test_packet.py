"""The packet is the hand-off artifact, so it has to be a real document.

Regression guard for the bug these tests were written against: the packet was served
as text/markdown and rendered in the browser as unstyled plain text.
"""

from __future__ import annotations

from app.agent.state import ComplianceEntry, CitationCheckRecord, DraftVersion, Issue, ReviewResult, RunState
from app.packet import _escape, _inline, render_packet_pdf


def _populated_state() -> RunState:
    state = RunState(run_id="abc123def456", mode="story", scope="quick", status="completed")
    state.steps = 23
    state.summary = "Budget exceeds the Track 2 cap by $38,450. **Fringe rate error** must be fixed."
    state.requirements = [{"id": "R-009", "text": "Include current letters of collaboration."}]
    state.compliance["R-009"] = ComplianceEntry("R-009", "missing", ["letter-northgate-2021"], "2021 letter is stale")
    state.compliance["R-025"] = ComplianceEntry("R-025", "partial", [], "needs diversity data")
    state.issues = [
        Issue(id="i1", source="check", severity="high", title="Budget over cap",
              detail="Total $1,038,450 exceeds the $1,000,000 cap.", code="BUDGET_OVER_CAP",
              recommended_action="Reduce equipment."),
        Issue(id="i2", source="agent", severity="medium", title="Stale letter", detail="Northgate letter is from 2021."),
    ]
    state.drafts["results_prior_support"] = [
        DraftVersion(version=1, text="## Outcomes\n\n- 84 of 120 completers [[prior-outcomes-data]]\n\n*Documented* causes.",
                     citations=["prior-outcomes-data"], word_count=120)
    ]
    state.reviews["results_prior_support"] = [
        ReviewResult(version=1, mean_score=3.4, passed=False, must_fix=["Add diversity data"])
    ]
    state.citation_checks["results_prior_support"] = [CitationCheckRecord(version=1, passed=True)]
    return state


def test_packet_is_a_real_pdf():
    pdf = render_packet_pdf(_populated_state(), institution="Harrow Creek Community College",
                            solicitation="NSF 24-584", device_label="HP ZGX Nano", model="qwen3.6:35b")
    assert pdf.startswith(b"%PDF-"), "output must be a PDF, not markdown or HTML"
    assert pdf.rstrip().endswith(b"%%EOF")
    assert len(pdf) > 3000


def test_packet_renders_without_a_completed_run():
    """An empty or cancelled run must still produce a document, not a traceback."""
    pdf = render_packet_pdf(RunState(run_id="empty"))
    assert pdf.startswith(b"%PDF-")


def test_draft_provenance_is_carried_in_the_document_metadata():
    pdf = render_packet_pdf(_populated_state(), solicitation="NSF 24-584")
    assert b"DRAFT" in pdf


def test_model_text_is_escaped_before_markdown_is_applied():
    """ReportLab's Paragraph parses mini-HTML, so unescaped model output could break
    the document or inject markup. Escape first, transform second."""
    assert _escape("<b>x</b> & y") == "&lt;b&gt;x&lt;/b&gt; &amp; y"
    out = _inline("<script>alert(1)</script> **bold**")
    assert "&lt;script&gt;" in out
    assert "<b>bold</b>" in out


def test_a_draft_full_of_hostile_markup_still_renders():
    state = RunState(run_id="hostile")
    state.drafts["s"] = [DraftVersion(version=1, text="<para>&<unclosed **x** `c` [[doc#c001]]")]
    pdf = render_packet_pdf(state)
    assert pdf.startswith(b"%PDF-")


def test_citations_survive_into_the_document():
    out = _inline("A claim [[prior-outcomes-data#c003]].")
    assert "prior-outcomes-data#c003" in out


def test_a_run_persisted_to_disk_can_be_rehydrated_for_its_packet():
    """Runs live in memory, so without this a restart made every past run's packet a
    404 while its state.json sat on disk."""
    original = _populated_state()
    restored = RunState.from_dict(original.to_dict())

    assert restored.run_id == original.run_id
    assert restored.status == original.status
    assert restored.summary == original.summary
    assert set(restored.compliance) == set(original.compliance)
    assert restored.compliance["R-009"].status == "missing"
    assert restored.drafts["results_prior_support"][0].text == original.drafts["results_prior_support"][0].text
    assert restored.latest_review("results_prior_support").mean_score == 3.4
    assert restored.latest_citation_check("results_prior_support").passed is True
    assert [i.title for i in restored.issues] == [i.title for i in original.issues]

    assert render_packet_pdf(restored).startswith(b"%PDF-")


def test_from_dict_tolerates_an_older_state_file():
    """Unknown/missing keys must not make an old run unreadable."""
    state = RunState.from_dict({"run_id": "old", "mode": "story", "unexpected_key": 1})
    assert state.run_id == "old" and state.steps == 0
    assert render_packet_pdf(state).startswith(b"%PDF-")


# -- mode-specific content -----------------------------------------------------


def test_a_grading_packet_does_not_claim_to_have_drafted_the_submission():
    from app.agent.state import CitationCheckRecord

    state = _populated_state()
    state.mode = "grade"
    state.citation_checks["results_prior_support"] = [CitationCheckRecord(
        version=1, passed=False,
        unsupported_claims=[{"claim": "The project graduated 120 completers.", "verdict": "contradicted",
                             "detail": "the records show 84", "evidence": ["prior-outcomes-data"]}],
    )]
    pdf = render_packet_pdf(state)
    assert pdf.startswith(b"%PDF-")
    text = _pdf_text(pdf)
    assert "Submitted sections, as assessed" in text
    assert "Drafted sections" not in text
    assert "contradicted" in text.lower()


def test_a_discovery_packet_carries_the_shortlist_and_what_left_the_device():
    from app.agent.state import ProposedOpportunity

    state = RunState(run_id="d1", mode="discover", status="completed")
    state.summary = "Two opportunities worth pursuing."
    state.proposals = [ProposedOpportunity(
        opportunity_id="324369", number="PD-20-144Y", title="Foundational Research in Robotics",
        agency="NSF", close_date="", fit_rationale="Matches the robotics programme [[inst-profile]].",
        confidence="high", concerns=["Confirm two-year IHE eligibility"])]
    state.egress_log = [
        {"allowed": True, "url": "https://api.grants.gov/v1/api/search2", "reason": "ok",
         "terms": [{"term": "robotics", "allowed": True, "reason": "vocabulary_match"}]},
        {"allowed": False, "url": "https://api.grants.gov/v1/api/search2", "reason": "terms_rejected",
         "terms": [{"term": "Zephyr Cascade", "allowed": False, "reason": "verbatim_from_upload"}]},
    ]
    text = _pdf_text(render_packet_pdf(state))
    assert "Recommended opportunities" in text
    assert "PD-20-144Y" in text
    assert "A human must check" in text
    assert "What left this device" in text
    assert "robotics" in text
    # The refused term must NOT be reprinted in a document that may be shared.
    assert "Zephyr Cascade" not in text


def _pdf_text(pdf: bytes) -> str:
    from io import BytesIO

    from pypdf import PdfReader

    return " ".join(p.extract_text() or "" for p in PdfReader(BytesIO(pdf)).pages)


def test_a_discovery_packet_does_not_assert_a_solicitation_it_never_worked_to():
    """Discovery is looking FOR a solicitation. Printing the shipped default asserted a
    connection to NSF 24-584 the run had nothing to do with."""
    state = RunState(run_id="d2", mode="discover", status="completed")
    state.summary = "Three candidates."
    text = _pdf_text(render_packet_pdf(state, solicitation="NSF 24-584 — ATE", institution="Harrow Creek"))
    assert "Harrow Creek" in text
    assert "NSF 24-584" not in text
    # ...and no empty sections that mean nothing for this mode.
    assert "No compliance entries" not in text
    assert "No issues were flagged" not in text


def test_an_assessing_packet_still_reports_an_empty_finding_as_a_finding():
    """"No issues were flagged" IS informative for a run whose job was to look."""
    state = RunState(run_id="s2", mode="story", status="completed")
    state.requirements = [{"id": "R-001", "text": "t"}]
    text = _pdf_text(render_packet_pdf(state))
    assert "No issues were flagged" in text
    assert "Compliance matrix" in text
