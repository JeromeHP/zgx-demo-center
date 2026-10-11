from __future__ import annotations

from pathlib import Path

from app.checks.citations import verify_citations


def _write_doc(vault_dir: Path, doc_id: str, text: str) -> None:
    (vault_dir / f"{doc_id}.md").write_text(text, encoding="utf-8")


def test_supported_citation_passes(tmp_path):
    _write_doc(
        tmp_path,
        "prior-annual-report-2025",
        "# Report\n\nThe project produced 84 completers against a target of 120.\n",
    )
    draft = "The prior project produced 84 completers against a target of 120. [[prior-annual-report-2025]]"
    result = verify_citations(draft, tmp_path)
    assert result.passed
    assert result.missing_docs == []
    assert result.unsupported_claims == []


def test_unsupported_number_flagged(tmp_path):
    _write_doc(tmp_path, "prior-annual-report-2025", "# Report\n\nThe project produced 84 completers.\n")
    draft = "The prior project produced 200 completers. [[prior-annual-report-2025]]"
    result = verify_citations(draft, tmp_path)
    assert not result.passed
    assert len(result.unsupported_claims) == 1
    assert result.unsupported_claims[0].number == "200"


def test_missing_doc_flagged(tmp_path):
    draft = "The project produced 84 completers. [[does-not-exist]]"
    result = verify_citations(draft, tmp_path)
    assert not result.passed
    assert "does-not-exist" in result.missing_docs


def test_uncited_number_is_a_warning_not_a_failure(tmp_path):
    # The draft carries a citation elsewhere, because a draft with no citations at all
    # is now a failure in its own right (see "the vacuous pass" below). What is being
    # tested here is unchanged: a *number* nobody cited is a warning, not a failure.
    _write_doc(tmp_path, "prior-annual-report-2025", "# Report\n\nThe programme ran for 3 years.\n")
    draft = (
        "The project produced 84 completers with no citation attached.\n\n"
        "The programme ran for 3 years. [[prior-annual-report-2025]]"
    )
    result = verify_citations(draft, tmp_path)
    assert result.passed
    assert len(result.uncited_numbers) == 1
    assert result.uncited_numbers[0].number == "84"


def test_currency_and_percentage_normalization(tmp_path):
    _write_doc(tmp_path, "budget-satp-draft", "# Budget\n\nTotal budget is $1,038,450 with fringe at 28.0%.\n")
    draft = "The draft budget totals $1,038,450 with a 28.0% fringe rate. [[budget-satp-draft]]"
    result = verify_citations(draft, tmp_path)
    assert result.passed


def test_chunk_id_must_exist(tmp_path):
    _write_doc(tmp_path, "doc-a", "# Heading\n\n" + " ".join(f"word{i}" for i in range(600)) + "\n")
    draft = "Some claim with 42 in it. [[doc-a#c099]]"
    result = verify_citations(draft, tmp_path)
    assert not result.passed
    assert "doc-a#c099" in result.missing_docs


def test_known_doc_ids_restricts_lookup_without_touching_disk(tmp_path):
    _write_doc(tmp_path, "doc-a", "# Heading\n\nSome text with 55 in it.\n")
    draft = "A claim with 55 in it. [[doc-a]]"
    result = verify_citations(draft, tmp_path, known_doc_ids=set())
    assert not result.passed
    assert "doc-a" in result.missing_docs


# -- the vacuous pass ---------------------------------------------------------------
# Found in a live Ashgrove run: the draft cited [[doc#doc#c000]] (a chunk_id already
# contains its doc_id, and the model was shown both), the two-segment pattern did not
# match it, so the run recorded `citations: []` -- and `verify_citations` PASSED,
# because with nothing parsed there was nothing left to object to. A draft whose every
# citation is malformed must fail, and so must one with no citations at all; otherwise
# the gate reports success about an empty set.


def test_doubled_chunk_id_form_parses(tmp_path):
    _write_doc(tmp_path, "aumc-profile", "# Profile\n\nThe centre enrolled 6,840 students.\n")
    draft = "The centre enrolled 6,840 students. [[aumc-profile#aumc-profile#c000]]"
    result = verify_citations(draft, tmp_path)
    assert result.citation_count == 1
    assert result.malformed_citations == []
    assert result.passed


def test_draft_with_no_citations_fails(tmp_path):
    _write_doc(tmp_path, "aumc-profile", "# Profile\n\nThe centre enrolled 6,840 students.\n")
    result = verify_citations("The centre is well placed to deliver this work.", tmp_path)
    assert not result.passed, "a draft with nothing to verify has not been verified"
    assert result.citation_count == 0


def test_malformed_marker_fails_rather_than_being_ignored(tmp_path):
    _write_doc(tmp_path, "aumc-profile", "# Profile\n\nThe centre enrolled 6,840 students.\n")
    draft = "The centre enrolled 6,840 students. [[aumc profile, page 2]]"
    result = verify_citations(draft, tmp_path)
    assert not result.passed
    assert result.malformed_citations == ["[[aumc profile, page 2]]"]


def test_one_bad_marker_fails_a_draft_with_good_ones(tmp_path):
    _write_doc(tmp_path, "aumc-profile", "# Profile\n\nThe centre enrolled 6,840 students.\n")
    draft = (
        "The centre enrolled 6,840 students. [[aumc-profile]]\n"
        "Facilities are extensive. [[see the facilities inventory]]"
    )
    result = verify_citations(draft, tmp_path)
    assert not result.passed
    assert len(result.malformed_citations) == 1


def test_chunk_level_citation_resolves_in_both_forms(tmp_path):
    # Chunk ids are `doc_id#cNNN`, and the raw second segment used to be compared
    # against that, so every chunk-level citation -- the exact syntax the prompt asks
    # for -- was reported as a missing document.
    _write_doc(tmp_path, "doc-a", "# Heading\n\nA claim with 55 in it.\n")
    for form in ("[[doc-a#c000]]", "[[doc-a#doc-a#c000]]"):
        result = verify_citations(f"A claim with 55 in it. {form}", tmp_path)
        assert result.passed, f"{form} should resolve: {result.missing_docs}"
    bad = verify_citations("A claim with 55 in it. [[doc-a#c099]]", tmp_path)
    assert not bad.passed
    assert bad.missing_docs == ["doc-a#c099"]
