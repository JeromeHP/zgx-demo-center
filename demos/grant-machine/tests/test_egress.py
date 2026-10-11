"""Adversarial tests for the egress boundary (Phase 0).

The containment claim -- "your documents never leave this device" -- is only worth
making if it is enforced. These tests try to break it. The red-team test at the
bottom is the one that matters: it plants a distinctive phrase in an upload and
proves it cannot reach the wire by any of the paths a caller might take.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.egress import (
    EgressBlocked,
    EgressGuard,
    SessionCorpus,
    TermValidator,
    Vocabulary,
    normalize,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VOCAB_DIR = REPO_ROOT / "data" / "vocabulary"

# A phrase that exists nowhere in the vocabulary or the public record. If this string
# ever appears in an outbound payload, containment has failed.
SECRET_PHRASE = "zephyr cascade fabrication protocol"
SECRET_PI = "Marisol Thackeray-Vance"
SECRET_AWARD = "award 8814402"


@pytest.fixture
def vocab() -> Vocabulary:
    return Vocabulary.load(VOCAB_DIR)


@pytest.fixture
def corpus() -> SessionCorpus:
    c = SessionCorpus()
    c.add_text(
        f"Internal strategy memo. Our {SECRET_PHRASE} underpins the robotics program. "
        f"Principal investigator {SECRET_PI} leads the effort under {SECRET_AWARD}. "
        "We also do machine learning and coastal resilience work."
    )
    return c


@pytest.fixture
def guard(vocab, corpus) -> EgressGuard:
    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(transport))
    return EgressGuard(vocab, corpus, client=client)


# -- the vocabulary is real ----------------------------------------------------


def test_vocabulary_loads_a_substantial_allowlist(vocab):
    assert len(vocab.entries) > 500
    assert "machine learning" in vocab.entries
    assert "coastal resilience" in vocab.entries


# -- layer 1: default-deny term validation -------------------------------------


def test_vocabulary_term_is_allowed(vocab, corpus):
    d = TermValidator(vocab, corpus).validate("quantum computing")
    assert d.allowed and d.reason == "vocabulary_match"


def test_term_not_in_vocabulary_is_refused(vocab, corpus):
    """Default-deny: absent from the vocabulary AND absent from the uploads."""
    d = TermValidator(vocab, corpus).validate("competitive underwater basket weaving")
    assert not d.allowed and d.reason == "not_in_vocabulary"


def test_phrase_lifted_verbatim_from_an_upload_is_refused(vocab, corpus):
    d = TermValidator(vocab, corpus).validate(SECRET_PHRASE)
    assert not d.allowed and d.reason == "verbatim_from_upload"


def test_a_vocabulary_term_is_still_allowed_when_it_also_appears_in_an_upload(vocab, corpus):
    """The uploads mention robotics; 'robotics' is a public field name, not private
    data, so it must still be usable or discovery finds nothing."""
    assert corpus.contains_verbatim("robotics")
    d = TermValidator(vocab, corpus).validate("robotics")
    assert d.allowed, "a published field name must not be blocked just because the user also used it"


def test_investigator_name_is_refused(vocab, corpus):
    d = TermValidator(vocab, corpus).validate(SECRET_PI)
    assert not d.allowed


def test_identifiers_and_figures_are_refused(vocab, corpus):
    v = TermValidator(vocab, corpus)
    assert not v.validate("award 8814402").allowed
    assert v.validate("NSF 24-584").reason in {"contains_digits", "not_in_vocabulary", "verbatim_from_upload"}
    assert not v.validate("$1,038,450").allowed


def test_overlong_terms_are_refused(vocab, corpus):
    d = TermValidator(vocab, corpus).validate("a b c d e f g h i j")
    assert not d.allowed and d.reason == "too_long"


def test_empty_terms_are_refused(vocab, corpus):
    v = TermValidator(vocab, corpus)
    assert not v.validate("").allowed
    assert not v.validate("   ").allowed
    assert not v.validate("!!!").allowed


# -- transport guard -----------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:8095/api/health",        # the app itself
        "https://localhost:11434/v1/models",        # the model endpoint
        "https://192.168.1.33:8086/status/grant",   # the Demo Center install API
        "https://10.0.0.5/internal",
        "https://100.80.240.38/tailnet",            # CGNAT / tailnet
    ],
)
def test_non_public_hosts_are_refused(guard, url):
    with pytest.raises(EgressBlocked) as exc:
        guard.request("GET", url, terms=["robotics"])
    assert exc.value.reason == "host_not_public"


def test_plaintext_http_is_refused(guard):
    with pytest.raises(EgressBlocked) as exc:
        guard.request("GET", "http://api.grants.gov/v1/api/search2", terms=["robotics"])
    assert exc.value.reason == "scheme_not_https"


def test_request_budget_is_enforced(vocab, corpus):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    g = EgressGuard(vocab, corpus, max_requests=2, client=client)
    for _ in range(2):
        g.request("POST", "https://api.grants.gov/v1/api/search2", terms=["robotics"])
    with pytest.raises(EgressBlocked) as exc:
        g.request("POST", "https://api.grants.gov/v1/api/search2", terms=["robotics"])
    assert exc.value.reason == "request_budget_exhausted"


def test_oversized_responses_are_refused(vocab, corpus):
    big = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"x" * 5000)))
    g = EgressGuard(vocab, corpus, max_response_bytes=1000, client=big)
    with pytest.raises(EgressBlocked) as exc:
        g.request("GET", "https://example.org/big", terms=["robotics"])
    assert exc.value.reason == "response_too_large"


# -- layers 2 and 3: the payload actually being sent ----------------------------


def test_upload_content_in_the_request_body_is_blocked(guard):
    """A caller that skips validate_terms must still not get upload text out."""
    with pytest.raises(EgressBlocked) as exc:
        guard.request(
            "POST", "https://api.grants.gov/v1/api/search2",
            terms=["robotics"],
            json_body={"keyword": SECRET_PHRASE, "rows": 25},
        )
    assert exc.value.reason in {"payload_matches_upload", "payload_token_not_in_vocabulary"}


def test_upload_content_in_a_url_query_string_is_blocked(guard):
    with pytest.raises(EgressBlocked) as exc:
        guard.request("GET", f"https://example.org/search?q={SECRET_PHRASE.replace(' ', '+')}", terms=["robotics"])
    assert exc.value.reason in {"payload_matches_upload", "payload_token_not_in_vocabulary"}


def test_upload_content_nested_deep_in_a_json_body_is_blocked(guard):
    with pytest.raises(EgressBlocked) as exc:
        guard.request(
            "POST", "https://api.grants.gov/v1/api/search2",
            terms=["robotics"],
            json_body={"filters": [{"any": {"deep": [SECRET_PHRASE]}}]},
        )
    assert exc.value.reason in {"payload_matches_upload", "payload_token_not_in_vocabulary"}


def test_a_clean_vocabulary_payload_is_allowed(guard):
    resp = guard.request(
        "POST", "https://api.grants.gov/v1/api/search2",
        terms=["robotics", "machine learning"],
        json_body={"keyword": "robotics machine learning", "oppStatuses": "posted"},
    )
    assert resp.status_code == 200


# -- the audit log -------------------------------------------------------------


def test_every_decision_is_recorded_including_refusals(guard):
    with pytest.raises(EgressBlocked):
        guard.request("POST", "https://api.grants.gov/v1/api/search2", terms=[SECRET_PHRASE])
    guard.request("POST", "https://api.grants.gov/v1/api/search2", terms=["photonics"],
                  json_body={"keyword": "photonics"})

    assert len(guard.records) == 2
    blocked, allowed = guard.records
    assert not blocked.allowed and blocked.reason == "terms_rejected"
    assert allowed.allowed and allowed.reason == "ok"
    assert allowed.bytes_in > 0
    assert guard.request_count == 1
    assert guard.terms_sent == ["photonics"]


def test_records_serialize_for_the_event_stream(guard):
    guard.request("POST", "https://api.grants.gov/v1/api/search2", terms=["photonics"],
                  json_body={"keyword": "photonics"})
    payload = json.dumps(guard.records[0].as_dict())
    assert "photonics" in payload


# -- red team ------------------------------------------------------------------


def test_red_team_no_upload_content_reaches_the_wire_by_any_path(vocab):
    """Plant a distinctive phrase in an upload, then try every route out.

    Whatever is attempted, the phrase must never appear in a transmitted request,
    and every attempt must be visible in the audit log.
    """
    corpus = SessionCorpus()
    corpus.add_text(
        f"CONFIDENTIAL. The {SECRET_PHRASE} is our core differentiator. "
        f"{SECRET_PI} is PI. Prior work funded under {SECRET_AWARD}."
    )

    transmitted: list[str] = []

    def transport(request: httpx.Request) -> httpx.Response:
        transmitted.append(str(request.url))
        transmitted.append(request.content.decode("utf-8", errors="ignore"))
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(transport))
    guard = EgressGuard(vocab, corpus, client=client)

    attempts = [
        {"terms": [SECRET_PHRASE]},
        {"terms": [SECRET_PI]},
        {"terms": ["robotics"], "json_body": {"keyword": SECRET_PHRASE}},
        {"terms": ["robotics"], "params": {"q": SECRET_PHRASE}},
        {"terms": ["robotics"], "json_body": {"a": {"b": [SECRET_PI]}}},
        {"terms": ["robotics"], "json_body": {"keyword": f"robotics {SECRET_PHRASE}"}},
    ]
    for attempt in attempts:
        with pytest.raises(EgressBlocked):
            guard.request("POST", "https://api.grants.gov/v1/api/search2", **attempt)

    # Nothing distinctive reached the transport...
    wire = " ".join(transmitted).lower()
    for secret in (SECRET_PHRASE, SECRET_PI, "8814402"):
        assert normalize(secret) not in normalize(wire), f"{secret!r} reached the wire"
    assert not transmitted, "no request should have been transmitted at all"

    # ...and every attempt is on the record.
    assert len(guard.records) == len(attempts)
    assert all(not r.allowed for r in guard.records)
    assert guard.request_count == 0


def test_session_corpus_reads_uploaded_files_from_disk(tmp_path: Path):
    (tmp_path / "strategy.md").write_text(f"# Plan\n\nThe {SECRET_PHRASE} matters.\n", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "cv.txt").write_text(f"{SECRET_PI}, professor.\n", encoding="utf-8")

    corpus = SessionCorpus.from_dir(tmp_path)
    assert corpus.doc_count == 2
    assert corpus.contains_verbatim(SECRET_PHRASE)
    assert corpus.contains_verbatim(SECRET_PI)


def test_unparseable_uploads_still_contribute_to_the_blocklist(tmp_path: Path):
    """A file the loaders cannot parse must not become a containment hole."""
    (tmp_path / "weird.bin").write_bytes(b"\xff\xfe" + SECRET_PHRASE.encode() + b"\x00\x01")
    corpus = SessionCorpus.from_dir(tmp_path)
    assert corpus.contains_verbatim(SECRET_PHRASE)


# -- echo-back of server-supplied values ---------------------------------------


def test_an_identifier_the_server_supplied_can_be_sent_back(vocab, corpus):
    """Refusing to echo a catalogue's own identifiers makes the catalogue unusable:
    the first live discovery run spent its entire step budget on detail lookups
    blocked for carrying the numeric id the preceding search had just returned."""
    client = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"oppHits": [{"id": "324369"}]})))
    guard = EgressGuard(vocab, corpus, client=client)

    guard.request("POST", "https://api.grants.gov/v1/api/search2", terms=["robotics"],
                  json_body={"keyword": "robotics"})
    resp = guard.request("POST", "https://api.grants.gov/v1/api/fetchOpportunity",
                         terms=[], json_body={"opportunityId": "324369"})
    assert resp.status_code == 200


def test_an_identifier_never_seen_from_the_server_is_still_refused(guard):
    with pytest.raises(EgressBlocked) as exc:
        guard.request("POST", "https://api.grants.gov/v1/api/fetchOpportunity",
                      terms=[], json_body={"opportunityId": "8814402"})
    assert exc.value.reason == "payload_token_not_in_vocabulary"


def test_echo_back_never_overrides_the_upload_corpus(vocab):
    """A value in BOTH a server response and the institution's documents is their
    data, whoever else publishes it."""
    corpus = SessionCorpus()
    corpus.add_text("Prior work funded under award 8814402.")
    client = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"note": "see award 8814402 for context"})))
    guard = EgressGuard(vocab, corpus, client=client)

    guard.request("POST", "https://api.grants.gov/v1/api/search2", terms=["robotics"],
                  json_body={"keyword": "robotics"})
    with pytest.raises(EgressBlocked):
        guard.request("POST", "https://api.grants.gov/v1/api/fetchOpportunity",
                      terms=[], json_body={"opportunityId": "8814402"})


def test_a_link_found_on_a_fetched_page_can_be_followed(vocab, corpus):
    """Scanning the query string of a link the site itself published rejected ordinary
    navigation: a live run was refused "?prs=ATE" on an NSF programme page it had just
    been handed."""
    page = '<a href="https://www.nsf.gov/funding/pgm_summ.jsp?prs=ATE">ATE</a>'
    client = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, text=page, headers={"content-type": "text/html"})))
    guard = EgressGuard(vocab, corpus, client=client)

    guard.request("GET", "https://www.nsf.gov/advancedtecheducation", terms=[])
    resp = guard.request("GET", "https://www.nsf.gov/funding/pgm_summ.jsp?prs=ATE", terms=[])
    assert resp.status_code == 200


def test_a_constructed_url_carrying_upload_text_is_still_refused(vocab):
    """The relaxation above must not become a hole: a URL we were never given is
    scanned in full, however plausible it looks."""
    corpus = SessionCorpus()
    corpus.add_text(f"Internal: {SECRET_PHRASE}.")
    client = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, text="<html>nothing useful</html>",
                                 headers={"content-type": "text/html"})))
    guard = EgressGuard(vocab, corpus, client=client)

    guard.request("GET", "https://www.nsf.gov/advancedtecheducation", terms=[])
    with pytest.raises(EgressBlocked):
        guard.request("GET", f"https://www.nsf.gov/search?q={SECRET_PHRASE.replace(' ', '+')}", terms=[])


# -- operator-configured infrastructure ----------------------------------------


def test_a_configured_local_search_endpoint_is_reachable(vocab, corpus):
    """The SSRF guard would otherwise block a self-hosted SearXNG the operator
    deliberately stood up — the point of that guard is model-chosen URLs."""
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": []})))
    guard = EgressGuard(vocab, corpus, client=client, trusted_endpoints=["http://127.0.0.1:8096"])
    resp = guard.request("GET", "http://127.0.0.1:8096/search", terms=["robotics"],
                         params={"q": "robotics", "format": "json"})
    assert resp.status_code == 200


def test_trusting_one_local_endpoint_does_not_trust_the_rest_of_the_host(vocab, corpus):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    guard = EgressGuard(vocab, corpus, client=client, trusted_endpoints=["http://127.0.0.1:8096"])
    for url in ("http://127.0.0.1:11434/v1/models", "http://127.0.0.1:8086/status/grant"):
        with pytest.raises(EgressBlocked) as exc:
            guard.request("GET", url, terms=["robotics"])
        assert exc.value.reason == "host_not_public"


def test_a_lookalike_host_does_not_inherit_trust(vocab, corpus):
    """Prefix matching must respect a path boundary, or 8096.evil.test is trusted."""
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    guard = EgressGuard(vocab, corpus, client=client, trusted_endpoints=["http://127.0.0.1:8096"])
    with pytest.raises(EgressBlocked):
        guard.request("GET", "http://127.0.0.1:8096.evil.test/search", terms=["robotics"])


def test_a_trusted_endpoint_still_validates_terms_and_is_audited(vocab, corpus):
    """Trust relaxes the transport check only. What may be SENT is unchanged."""
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    guard = EgressGuard(vocab, corpus, client=client, trusted_endpoints=["http://127.0.0.1:8096"])
    with pytest.raises(EgressBlocked) as exc:
        guard.request("GET", "http://127.0.0.1:8096/search", terms=[SECRET_PHRASE])
    assert exc.value.reason == "terms_rejected"
    assert guard.records[-1].url.startswith("http://127.0.0.1:8096")
