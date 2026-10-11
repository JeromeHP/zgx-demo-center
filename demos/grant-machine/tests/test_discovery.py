"""Discovery tools and their harness guardrails (v2 Phase 1).

The guardrail that matters here is the discovery analogue of the citation gate: the
agent cannot recommend a funding opportunity it never actually retrieved.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from app.agent.state import CitationCheckRecord, DraftVersion, RunState
from app.agent.tools import (
    DISCOVERY_TOOLS,
    PROPOSAL_TOOLS,
    ToolContext,
    execute_tool,
    schemas_for,
)
from app.discovery import GrantsGovClient, WebResearch
from app.discovery.web import html_to_markdown
from app.egress import EgressGuard, SessionCorpus, Vocabulary

REPO_ROOT = Path(__file__).resolve().parents[1]
VOCAB_DIR = REPO_ROOT / "data" / "vocabulary"

SEARCH_PAYLOAD = {
    "errorcode": 0,
    "data": {
        "hitCount": 2,
        "oppHits": [
            {"id": "324369", "number": "PD-20-144Y", "title": "Foundational Research in Robotics",
             "agency": "National Science Foundation", "agencyCode": "NSF", "oppStatus": "posted",
             "closeDate": "", "openDate": "Feb 12, 2020", "docType": "synopsis", "cfdaList": ["47.070"]},
            {"id": "999111", "number": "24-584", "title": "Advanced Technological Education",
             "agency": "National Science Foundation", "agencyCode": "NSF", "oppStatus": "posted",
             "closeDate": "Oct 01, 2026", "openDate": "", "docType": "synopsis", "cfdaList": []},
        ],
    },
}


class _Events:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict]] = []

    def emit(self, name, payload):
        self.emitted.append((name, payload))

    def names(self):
        return [n for n, _ in self.emitted]


@pytest.fixture
def corpus() -> SessionCorpus:
    c = SessionCorpus()
    c.add_text("Our Zephyr Cascade Fabrication Protocol supports robotics and technician education. "
               "PI Marisol Thackeray-Vance.")
    return c


def _guard(corpus, handler) -> EgressGuard:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return EgressGuard(Vocabulary.load(VOCAB_DIR), corpus, client=client)


def _ctx(tmp_path: Path, corpus, handler, *, with_profile: bool = True) -> tuple[ToolContext, _Events]:
    guard = _guard(corpus, handler)
    state = RunState(run_id="disc1", mode="discover", scope="quick")
    if with_profile:
        state.drafts["institution_profile"] = [
            DraftVersion(version=1, text="We do robotics [[inst-profile]].", citations=["[[inst-profile]]"])
        ]
        state.citation_checks["institution_profile"] = [CitationCheckRecord(version=1, passed=True)]
    events = _Events()
    ctx = ToolContext(
        settings=None, llm=None, vault_dir=tmp_path, index=None, state=state,
        event_bus=events, runtime_dir=tmp_path, playbook="discovery",
        grants=GrantsGovClient(guard),
        web=WebResearch(guard, session_dir=tmp_path, searxng_url=""),
    )
    return ctx, events


# -- tool surface --------------------------------------------------------------


def test_each_playbook_only_offers_its_own_tools():
    discovery = {t["function"]["name"] for t in schemas_for("discovery")}
    proposal = {t["function"]["name"] for t in schemas_for("proposal")}

    assert "search_grants" in discovery and "search_grants" not in proposal
    # A discovery run has no solicitation, so proposal-writing tools must be out of reach.
    assert "draft_section" not in discovery
    assert "check_budget" not in discovery
    assert "get_requirements" not in discovery
    assert set(DISCOVERY_TOOLS) == discovery
    assert set(PROPOSAL_TOOLS) == proposal


# -- search --------------------------------------------------------------------


def test_search_returns_opportunities_and_remembers_them(tmp_path, corpus):
    ctx, events = _ctx(tmp_path, corpus, lambda r: httpx.Response(200, json=SEARCH_PAYLOAD))
    out = execute_tool(ctx, "search_grants", {"terms": ["robotics", "technician education"]})

    assert out["hit_count"] == 2
    assert {o["number"] for o in out["opportunities"]} == {"PD-20-144Y", "24-584"}
    assert set(ctx.state.opportunities_seen) == {"324369", "999111"}
    assert "grants_searched" in events.names()


def test_search_refuses_institution_language_and_explains_why(tmp_path, corpus):
    sent: list[str] = []

    def handler(request):
        sent.append(request.content.decode())
        return httpx.Response(200, json=SEARCH_PAYLOAD)

    ctx, _ = _ctx(tmp_path, corpus, handler)
    out = execute_tool(ctx, "search_grants", {"terms": ["Zephyr Cascade Fabrication Protocol"]})

    assert out["error"] == "no_permitted_terms"
    assert out["rejected_terms"][0]["reason"] == "verbatim_from_upload"
    assert not sent, "nothing should have been transmitted"


def test_a_mixed_term_list_sends_only_the_permitted_terms(tmp_path, corpus):
    sent: list[str] = []

    def handler(request):
        sent.append(request.content.decode())
        return httpx.Response(200, json=SEARCH_PAYLOAD)

    ctx, _ = _ctx(tmp_path, corpus, handler)
    out = execute_tool(ctx, "search_grants", {"terms": ["robotics", "Marisol Thackeray-Vance"]})

    assert out["terms_sent"] == ["robotics"]
    assert [r["term"] for r in out["rejected_terms"]] == ["Marisol Thackeray-Vance"]
    assert "Thackeray" not in " ".join(sent)


# -- the anti-fabrication gate -------------------------------------------------


def test_cannot_propose_an_opportunity_that_was_never_retrieved(tmp_path, corpus):
    ctx, _ = _ctx(tmp_path, corpus, lambda r: httpx.Response(200, json=SEARCH_PAYLOAD))
    out = execute_tool(ctx, "propose_opportunity", {
        "opportunity_id": "8675309",
        "fit_rationale": "Looks like a great fit [[inst-profile]].",
        "confidence": "high",
    })
    assert out["error"] == "unknown_opportunity"
    assert ctx.state.proposals == []


def test_can_propose_an_opportunity_that_was_retrieved(tmp_path, corpus):
    ctx, events = _ctx(tmp_path, corpus, lambda r: httpx.Response(200, json=SEARCH_PAYLOAD))
    execute_tool(ctx, "search_grants", {"terms": ["robotics"]})
    out = execute_tool(ctx, "propose_opportunity", {
        "opportunity_id": "324369",
        "fit_rationale": "Matches the robotics programme [[inst-profile]].",
        "confidence": "high",
        "concerns": ["Confirm two-year IHE eligibility"],
    })

    assert out["accepted"] is True and out["number"] == "PD-20-144Y"
    assert len(ctx.state.proposals) == 1
    assert ctx.state.proposals[0].concerns == ["Confirm two-year IHE eligibility"]
    assert "opportunity_proposed" in events.names()


def test_a_proposal_must_cite_its_evidence(tmp_path, corpus):
    ctx, _ = _ctx(tmp_path, corpus, lambda r: httpx.Response(200, json=SEARCH_PAYLOAD))
    execute_tool(ctx, "search_grants", {"terms": ["robotics"]})
    out = execute_tool(ctx, "propose_opportunity", {
        "opportunity_id": "324369", "fit_rationale": "Trust me, it fits.", "confidence": "high",
    })
    assert out["error"] == "uncited_rationale"


def test_cannot_propose_before_a_profile_exists(tmp_path, corpus):
    ctx, _ = _ctx(tmp_path, corpus, lambda r: httpx.Response(200, json=SEARCH_PAYLOAD), with_profile=False)
    execute_tool(ctx, "search_grants", {"terms": ["robotics"]})
    out = execute_tool(ctx, "propose_opportunity", {
        "opportunity_id": "324369", "fit_rationale": "Fits [[x]].", "confidence": "high",
    })
    assert out["error"] == "no_profile"


# -- the finish gate -----------------------------------------------------------


def test_finish_is_refused_until_the_discovery_work_is_actually_done(tmp_path, corpus):
    ctx, _ = _ctx(tmp_path, corpus, lambda r: httpx.Response(200, json=SEARCH_PAYLOAD), with_profile=False)

    out = execute_tool(ctx, "finish", {"summary": "done"})
    assert out["accepted"] is False
    assert any("draft_profile" in m for m in out["missing"])
    assert any("search_grants" in m for m in out["missing"])


def test_finish_is_allowed_after_an_honest_search_that_found_nothing(tmp_path, corpus):
    """Reporting "nothing suitable" is a real answer. Requiring a minimum number of
    proposals would pressure the agent to pad the shortlist with near-misses, which is
    exactly what the prompt tells it not to do."""
    empty = {"errorcode": 0, "data": {"hitCount": 0, "oppHits": []}}
    ctx, _ = _ctx(tmp_path, corpus, lambda r: httpx.Response(200, json=empty))
    execute_tool(ctx, "search_grants", {"terms": ["robotics"]})

    out = execute_tool(ctx, "finish", {"summary": "No current opportunities fit this institution."})
    assert out["accepted"] is True and out["proposals"] == 0


def test_finish_is_refused_while_the_profile_has_unsupported_claims(tmp_path, corpus):
    ctx, _ = _ctx(tmp_path, corpus, lambda r: httpx.Response(200, json=SEARCH_PAYLOAD))
    ctx.state.citation_checks["institution_profile"] = [
        CitationCheckRecord(version=1, passed=False, unsupported_claims=[{"claim": "we lead the world"}])
    ]
    execute_tool(ctx, "search_grants", {"terms": ["robotics"]})
    execute_tool(ctx, "propose_opportunity", {
        "opportunity_id": "324369", "fit_rationale": "Fits [[inst-profile]].", "confidence": "high"})

    out = execute_tool(ctx, "finish", {"summary": "done"})
    assert out["accepted"] is False
    assert any("unsupported claim" in m for m in out["missing"])


def test_finish_accepted_when_everything_is_in_place(tmp_path, corpus):
    ctx, _ = _ctx(tmp_path, corpus, lambda r: httpx.Response(200, json=SEARCH_PAYLOAD))
    execute_tool(ctx, "search_grants", {"terms": ["robotics"]})
    execute_tool(ctx, "propose_opportunity", {
        "opportunity_id": "324369", "fit_rationale": "Fits [[inst-profile]].", "confidence": "high"})

    out = execute_tool(ctx, "finish", {"summary": "Two opportunities worth pursuing."})
    assert out["accepted"] is True and out["proposals"] == 1
    assert ctx.state.summary == "Two opportunities worth pursuing."


# -- fetched pages become citable documents ------------------------------------


def test_a_fetched_page_becomes_a_citable_session_document(tmp_path, corpus):
    html = "<html><head><title>ATE Program</title></head><body><nav>skip</nav><h1>ATE</h1><p>Supports technician education.</p></body></html>"
    ctx, events = _ctx(tmp_path, corpus, lambda r: httpx.Response(200, html=html, headers={"content-type": "text/html"}))

    out = execute_tool(ctx, "fetch_url", {"url": "https://www.nsf.gov/ate"})

    assert not out.get("error")
    assert out["doc_id"].startswith("web-nsf-gov-")
    assert (tmp_path / f"{out['doc_id']}.md").is_file(), "must be written where the vault index can see it"
    saved = (tmp_path / f"{out['doc_id']}.md").read_text()
    assert "source_url: https://www.nsf.gov/ate" in saved
    assert "technician education" in saved
    assert "page_fetched" in events.names()


def test_fetched_text_cannot_then_be_echoed_back_out_as_a_search_term(tmp_path, corpus):
    """Once text is on the device it is the institution's working material."""
    html = "<html><body><p>Quadrupedal locomotion benchmark suite</p></body></html>"
    ctx, _ = _ctx(tmp_path, corpus, lambda r: httpx.Response(200, html=html, headers={"content-type": "text/html"}))
    execute_tool(ctx, "fetch_url", {"url": "https://example.org/page"})

    decision = ctx.grants._guard.validator.validate("Quadrupedal locomotion benchmark suite")
    assert not decision.allowed


def test_html_conversion_drops_chrome_and_keeps_content():
    title, md = html_to_markdown(
        "<html><head><title>T</title><style>x{}</style></head>"
        "<body><nav>menu</nav><h1>Heading</h1><p>Body text.</p><footer>foot</footer></body></html>"
    )
    assert title == "T"
    assert "Heading" in md and "Body text." in md
    assert "menu" not in md and "foot" not in md and "x{}" not in md


# -- graceful degradation ------------------------------------------------------


def test_web_search_reports_unavailable_without_a_provider(tmp_path, corpus):
    ctx, _ = _ctx(tmp_path, corpus, lambda r: httpx.Response(200, json={}))
    out = execute_tool(ctx, "search_web", {"terms": ["robotics"]})
    assert out["error"] == "web_search_unavailable"
    assert "search_grants" in out["detail"], "should point at the capability that still works"


def test_detail_lookup_rejects_a_non_numeric_id(tmp_path, corpus):
    ctx, _ = _ctx(tmp_path, corpus, lambda r: httpx.Response(200, json={}))
    out = execute_tool(ctx, "get_opportunity", {"opportunity_id": "../../etc/passwd"})
    assert out["error"] == "invalid_opportunity_id"


# -- the funding report's fields --------------------------------------------------

DETAIL_PAYLOAD = {
    "errorcode": 0,
    "data": {
        "opportunityNumber": "PD-20-144Y", "opportunityTitle": "Foundational Research in Robotics",
        "docType": "synopsis", "cfdas": [],
        "synopsis": {"agencyName": "National Science Foundation", "agencyCode": "NSF",
                     "responseDate": "", "synopsisDesc": "Robotics research.", "awardCeiling": "750000",
                     "costSharing": False, "applicantTypes": [],
                     "fundingDescLinkUrl": "http://www.nsf.gov/publications/pub_summ.jsp?ods_key=nsf20000"},
    },
}


def _catalogue(request):
    return httpx.Response(200, json=DETAIL_PAYLOAD if "fetchOpportunity" in str(request.url) else SEARCH_PAYLOAD)


def test_a_proposal_carries_what_the_report_needs(tmp_path, corpus):
    """Why it fits, what it would serve, and where a person goes to check it."""
    ctx, events = _ctx(tmp_path, corpus, _catalogue)
    execute_tool(ctx, "search_grants", {"terms": ["robotics"]})
    execute_tool(ctx, "get_opportunity", {"opportunity_id": "324369"})
    execute_tool(ctx, "propose_opportunity", {
        "opportunity_id": "324369",
        "fit_rationale": "Matches the robotics programme [[inst-profile]].",
        "confidence": "high",
        "aligned_with": ["robotics technician pathway", "  "],
    })
    p = ctx.state.proposals[0]
    assert p.aligned_with == ["robotics technician pathway"]
    assert p.url == "https://grants.gov/search-results-detail/324369"
    # The agency's full announcement, upgraded to https.
    assert p.announcement_url == "https://www.nsf.gov/publications/pub_summ.jsp?ods_key=nsf20000"
    assert p.award_ceiling == "750000" and p.cost_sharing is False
    emitted = dict(events.emitted)["opportunity_proposed"]
    assert emitted["announcement_url"] and emitted["aligned_with"]


def test_a_later_search_does_not_erase_the_detail(tmp_path, corpus):
    ctx, _ = _ctx(tmp_path, corpus, _catalogue)
    # The guard only lets an id out once the catalogue has returned it: search first.
    execute_tool(ctx, "search_grants", {"terms": ["robotics"]})
    execute_tool(ctx, "get_opportunity", {"opportunity_id": "324369"})
    execute_tool(ctx, "search_grants", {"terms": ["robotics"]})  # thin synopsis of the same id
    assert ctx.state.opportunities_seen["324369"]["announcement_url"].startswith("https://www.nsf.gov/")
