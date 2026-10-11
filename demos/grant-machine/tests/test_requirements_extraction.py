"""Runtime requirement extraction (v2 Phase 2: Intake).

The guard that makes this trustworthy is quote validation: a requirement the model
invented cannot supply a source_quote that is genuinely in the document.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.ingest.requirements import (
    CATEGORIES,
    ExtractionProgress,
    content_key,
    dedupe,
    extract_requirements,
    propose_sections,
    split_sections,
    validate,
)

SOLICITATION = """# Fictitious Foundation Call

## Table of Contents
Skip me.

## I. Eligibility
Applicants must be accredited public institutions of higher education located in the
United States. Individuals are not eligible to apply under this call for proposals.

## II. Proposal Preparation
The project narrative must not exceed fifteen pages, single spaced, in 11-point type.
Each proposal must include a data management plan of no more than two pages.

## VIII. Agency Contacts
Call us.
"""


class FakeClient:
    """Returns scripted JSON, and records what it was asked."""

    def __init__(self, by_heading: dict[str, list[dict]], sections=None):
        self.by_heading = by_heading
        self.sections = sections
        self.calls: list[str] = []

    def chat(self, profile, messages, **kwargs):
        prompt = messages[-1]["content"]
        self.calls.append(prompt)

        class R:
            pass

        r = R()
        if "planning the structure" in prompt:
            r.content = json.dumps({"sections": self.sections or []})
            return r
        for heading, items in self.by_heading.items():
            if f"Section heading: {heading}" in prompt:
                r.content = json.dumps({"requirements": items})
                return r
        r.content = json.dumps({"requirements": []})
        return r


# -- section splitting ---------------------------------------------------------


def test_split_sections_skips_boilerplate_headings():
    headings = [h for h, _ in split_sections(SOLICITATION)]
    assert "I. Eligibility" in headings
    assert "II. Proposal Preparation" in headings
    assert "Table of Contents" not in headings
    assert "VIII. Agency Contacts" not in headings


# -- the quote guard -----------------------------------------------------------


def test_a_requirement_whose_quote_is_not_in_the_document_is_discarded():
    body = "Applicants must be accredited public institutions."
    kept, discarded = validate(
        [
            {"category": "eligibility", "text": "Must be accredited",
             "source_quote": "Applicants must be accredited public institutions."},
            {"category": "eligibility", "text": "Must have a Nobel laureate on staff",
             "source_quote": "Applicants must employ at least one Nobel laureate."},
        ],
        body, "I. Eligibility",
    )
    assert [k["text"] for k in kept] == ["Must be accredited"]
    assert discarded == 1


def test_candidates_with_a_bad_category_or_no_text_are_discarded():
    body = "Some text here."
    kept, discarded = validate(
        [
            {"category": "invented_category", "text": "x", "source_quote": "Some text here."},
            {"category": "eligibility", "text": "", "source_quote": "Some text here."},
            {"category": "eligibility", "text": "ok", "source_quote": ""},
        ],
        body, "H",
    )
    assert kept == [] and discarded == 3


def test_near_duplicate_requirements_are_collapsed():
    out = dedupe([
        {"category": "formatting", "text": "The narrative must not exceed fifteen pages"},
        {"category": "formatting", "text": "The narrative must not exceed 15 pages."},
        {"category": "budget", "text": "The narrative must not exceed fifteen pages"},
    ])
    # Same category collapses; a different category is a different requirement.
    assert len(out) == 2


# -- end to end ----------------------------------------------------------------


def _client() -> FakeClient:
    return FakeClient({
        "I. Eligibility": [{
            "category": "eligibility",
            "text": "Applicants must be accredited public IHEs in the US",
            "source_quote": "Applicants must be accredited public institutions of higher education",
        }],
        "II. Proposal Preparation": [
            {"category": "formatting", "text": "Narrative limited to fifteen pages",
             "source_quote": "must not exceed fifteen pages"},
            {"category": "supplementary_documents", "text": "Include a data management plan",
             "source_quote": "data management plan of no more than two pages"},
            {"category": "formatting", "text": "Totally invented rule",
             "source_quote": "proposals must be submitted in triplicate on vellum"},
        ],
    })


def test_extraction_produces_the_shape_the_app_already_consumes(tmp_path):
    doc = extract_requirements(SOLICITATION, _client(), model="test",
                               solicitation_id="FF-01", program="Fictitious Programme",
                               cache_dir=tmp_path)

    assert doc["solicitation_id"] == "FF-01"
    assert doc["program"] == "Fictitious Programme"
    reqs = doc["requirements"]
    assert len(reqs) == 3, "the invented requirement must have been dropped"
    assert all(r["id"].startswith("R-") for r in reqs)
    assert all(r["category"] in CATEGORIES for r in reqs)
    # Runtime extraction must never claim to have been human reviewed.
    assert all(r["human_reviewed"] is False for r in reqs)
    assert doc["extraction"]["discarded_unverifiable_quote"] == 1


def test_progress_is_reported_per_section(tmp_path):
    seen: list[ExtractionProgress] = []
    extract_requirements(SOLICITATION, _client(), model="test",
                         on_progress=seen.append, cache_dir=tmp_path)
    assert len(seen) == 2  # two non-boilerplate sections
    assert {p.total for p in seen} == {2}
    assert [p.index for p in seen] == [1, 2]


def test_results_are_cached_by_content(tmp_path):
    client = _client()
    first = extract_requirements(SOLICITATION, client, model="test", cache_dir=tmp_path)
    calls_after_first = len(client.calls)

    second = extract_requirements(SOLICITATION, client, model="test", cache_dir=tmp_path)
    assert len(client.calls) == calls_after_first, "a cache hit must make no model calls"
    assert second["requirements"] == first["requirements"]

    # Different content, or a different model, is a different key.
    assert content_key(SOLICITATION, "a") != content_key(SOLICITATION, "b")
    assert content_key(SOLICITATION, "a") != content_key(SOLICITATION + " ", "a")


def test_a_corrupt_cache_entry_is_re_extracted_not_fatal(tmp_path):
    client = _client()
    key = content_key(SOLICITATION, "test")
    (tmp_path / f"{key}.json").write_text("{ not json", encoding="utf-8")

    doc = extract_requirements(SOLICITATION, client, model="test", cache_dir=tmp_path)
    assert len(doc["requirements"]) == 3


# -- section planning ----------------------------------------------------------


def test_sections_are_derived_from_the_solicitation():
    client = FakeClient({}, sections=[
        {"id": "Project Narrative", "title": "Project Narrative", "word_target": "15 pages"},
        {"id": "data_mgmt", "title": "Data Management Plan", "word_target": "2 pages"},
    ])
    out = propose_sections(client, {"requirements": [], "solicitation_id": "FF-01"})
    assert [s["id"] for s in out] == ["project_narrative", "data_mgmt"]
    assert out[0]["title"] == "Project Narrative"


def test_section_planning_falls_back_rather_than_leaving_a_run_with_nothing_to_do():
    out = propose_sections(FakeClient({}, sections=[]), {"requirements": []})
    assert len(out) == 1 and out[0]["id"] == "project_narrative"


# -- response shape tolerance --------------------------------------------------


class ShapeClient:
    """Returns whatever raw JSON string it was given."""

    def __init__(self, raw: str):
        self.raw = raw

    def chat(self, profile, messages, **kwargs):
        class R:
            pass

        r = R()
        r.content = self.raw
        return r


def test_a_bare_json_array_is_accepted():
    """Models return the array wrapped or bare. The wrapped form was all the NSF
    document ever produced, so `data.get(...)` on a bare list raised AttributeError --
    uncaught, and it killed an entire run on the first real NIH announcement."""
    from app.ingest.requirements import extract_from_section

    raw = json.dumps([{"category": "eligibility", "text": "t", "source_quote": "q"}])
    assert extract_from_section(ShapeClient(raw), "H", "body") == [
        {"category": "eligibility", "text": "t", "source_quote": "q"}
    ]


def test_unusable_response_shapes_yield_nothing_rather_than_raising():
    from app.ingest.requirements import extract_from_section

    for raw in ['"just a string"', "42", "null", "not json at all", json.dumps({"requirements": "nope"})]:
        assert extract_from_section(ShapeClient(raw), "H", "body") == []


def test_non_dict_entries_are_discarded_not_fatal():
    kept, discarded = validate(["a string", 42, None], "body", "H")
    assert kept == [] and discarded == 3
    assert dedupe(["a string", {"category": "budget", "text": "x"}]) == [{"category": "budget", "text": "x"}]


# -- a failure that used to be indistinguishable from success -------------------------
# propose_sections swallowed every exception into "one generic section", so a build run
# against an unreachable model server committed exactly what a successful one would.


class _DeadClient:
    def chat(self, *a, **kw):
        raise RuntimeError("Connection error.")


def test_section_proposal_falls_back_at_runtime():
    from app.ingest.requirements import propose_sections

    sections = propose_sections(_DeadClient(), {"requirements": []})
    assert [s["id"] for s in sections] == ["project_narrative"]


def test_section_proposal_is_strict_for_build_scripts():
    import pytest

    from app.ingest.requirements import propose_sections

    with pytest.raises(RuntimeError):
        propose_sections(_DeadClient(), {"requirements": []}, strict=True)
