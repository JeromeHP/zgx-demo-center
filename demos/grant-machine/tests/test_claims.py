"""Cross-checking a submission's figures against the institution's records."""

from __future__ import annotations

import json

from app.checks.claims import check_claims, extract_numeric_claims


class _Hit:
    def __init__(self, doc_id, snippet, text=None):
        self.doc_id, self.snippet = doc_id, snippet
        # The full chunk. check_claims judges against this, not the excerpt.
        self.text = snippet if text is None else text


class _Index:
    def __init__(self, snippet=""):
        self.snippet = snippet

    def search(self, query, k=3):
        return [_Hit("prior-outcomes-data", self.snippet)] if self.snippet else []


class _LLM:
    def __init__(self, verdict="contradicted", detail="records show 84, not 120"):
        self.payload = json.dumps({"verdict": verdict, "detail": detail})
        self.calls = 0

    def chat(self, profile, messages, **kwargs):
        self.calls += 1

        class R:
            pass

        r = R()
        r.content = self.payload
        return r


def test_only_sentences_carrying_numbers_are_checked():
    claims = extract_numeric_claims(
        "We graduated 120 completers. The programme is excellent and transformative. "
        "Placement reached 95% within six months."
    )
    assert [n for _, n in claims] == [["120"], ["95"]]
    assert all(any(ch.isdigit() for ch in c) for c, _ in claims)


def test_a_figure_contradicted_by_the_records_is_reported():
    """The case this exists for: the submission says 120, the records say 84."""
    index = _Index("Completers: 84 against a target of 120.")
    out = check_claims("The project graduated 120 completers against a target of 120.", index, _LLM())
    assert len(out) == 1
    assert out[0].verdict == "contradicted"
    assert out[0].evidence == ["prior-outcomes-data"]


def test_matching_figures_are_still_judged_against_the_evidence():
    """There is no "the numbers appear, so it is supported" shortcut, and there must
    not be: for "120 completers against a target of 120", a record reading
    "Completers: 84 against a target of 120" contains 120 — as the target."""
    llm = _LLM(verdict="supported", detail="enrollment matches")
    index = _Index("Enrollment is 4,180 students across all programmes.")
    out = check_claims("Total enrollment is 4,180 students.", index, llm)
    assert out[0].verdict == "supported"
    assert llm.calls == 1, "the evidence must actually be judged, not pattern-matched"


def test_a_claim_with_no_supporting_document_is_unverifiable_not_false():
    out = check_claims("We placed 95% of completers.", _Index(""), _LLM())
    assert out[0].verdict == "unverifiable"
    assert "no supporting document" in out[0].detail


def test_a_judging_failure_degrades_to_unverifiable():
    class Broken:
        def chat(self, *a, **k):
            raise RuntimeError("model down")

    out = check_claims("We served 400 students.", _Index("Served 250 students."), Broken())
    assert out[0].verdict == "unverifiable"


def test_an_unexpected_verdict_string_is_not_trusted():
    out = check_claims("We served 400 students.", _Index("Served 250 students."), _LLM(verdict="definitely true"))
    assert out[0].verdict == "unverifiable"


# -- the two bugs that let a real overstatement through ------------------------


def test_single_digit_figures_are_checked():
    """Citation verification ignores bare single digits to avoid matching stray
    numerals near a citation. Claim checking must not: "supported 6 graduate students"
    against a record of 4 is exactly the overstatement this exists to catch, and the
    shared pattern made it invisible."""
    claims = extract_numeric_claims(
        "The team produced 14 peer-reviewed publications and supported 6 graduate students."
    )
    assert claims[0][1] == ["14", "6"]


def test_a_markdown_heading_does_not_ride_into_the_claim():
    claims = extract_numeric_claims("## Results from Prior Support\n\nWe published 9 papers.")
    assert claims and "Results from Prior Support" not in claims[0][0]
    assert claims[0][1] == ["9"]


def test_the_judge_sees_the_whole_chunk_not_a_truncated_excerpt():
    """A 400-character excerpt centred on the first query token stopped immediately
    before the value being checked, so the judge answered "unverifiable" while looking
    at a table whose relevant cell had been trimmed away."""
    seen = {}

    class Recorder:
        def chat(self, profile, messages, **kwargs):
            seen["prompt"] = messages[-1]["content"]

            class R:
                content = json.dumps({"verdict": "contradicted", "detail": "record says 9"})

            return R()

    class Index:
        def search(self, query, k=3):
            return [_Hit("outcomes", snippet="| peer reviewed papers ", text="| peer reviewed papers | 9 |")]

    out = check_claims("We produced 14 peer-reviewed publications.", Index(), Recorder())
    assert "| peer reviewed papers | 9 |" in seen["prompt"], "the judge must see the value"
    assert out[0].verdict == "contradicted"


# -- status assertions ---------------------------------------------------------


def test_status_assertions_are_checked_even_with_no_numbers_in_them():
    """A clinical trial proposal asserting a current IRB approval — against a record
    showing it had expired in March — sailed through untouched, because the sentence
    contained no digits."""
    for text in [
        "IRB approval is current and in good standing.",
        "An independent data safety monitoring board is in place.",
        "This cost share commitment is confirmed.",
        "All five planned articulation agreements were executed with regional partners.",
    ]:
        assert extract_numeric_claims(text), f"not checked: {text!r}"


def test_ordinary_prose_and_intentions_are_not_treated_as_claims():
    """The status list is deliberately narrow. Scoring prose is what the section review
    is for; treating every sentence as a checkable claim would spend model calls on
    opinions."""
    for text in [
        "The mitochondrial hypothesis is interesting and worth pursuing.",
        "We will deploy sensors at twelve sites.",
        "This approach is novel and addresses an important gap.",
    ]:
        assert not extract_numeric_claims(text), f"should not be checked: {text!r}"


def test_a_status_claim_with_no_figures_still_reaches_the_judge():
    asked = {}

    class Recorder:
        def chat(self, profile, messages, **kwargs):
            asked["prompt"] = messages[-1]["content"]

            class R:
                content = json.dumps({"verdict": "contradicted", "detail": "expired 2026-03-31"})

            return R()

    class Index:
        def search(self, query, k=3):
            return [_Hit("regulatory", "", text="IRB approval EXPIRED on March 31, 2026.")]

    out = check_claims("IRB approval is current and in good standing.", Index(), Recorder())
    assert out[0].numbers == [], "a status claim carries no figures"
    assert out[0].verdict == "contradicted"
    assert "EXPIRED" in asked["prompt"]
