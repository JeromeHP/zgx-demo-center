"""The single outbound chokepoint (Phase 0).

Grant Machine v2 lets the agent research the open web, but the institution's own
material -- uploaded documents, research descriptions, proposal drafts -- must never
leave the device. Nothing else in `app/` may call the network: `app/llm_client.py`
talks to the local model endpoint, and every other byte that leaves goes through
`EgressGuard.request()`. `scripts/audit_portability.sh` enforces that.

The containment claim is enforced in code, not prompted, in the same way the finish
gate and the citation gate are (see docs/ARCHITECTURE.md "Harness guardrails"). Three
independent layers, because the first one is the one a future change is most likely
to weaken:

  1. DEFAULT-DENY TERM VALIDATION. A search term leaves only if it matches the
     published vocabulary in data/vocabulary/ (generic research-field names plus the
     grants.gov federal funding taxonomy). Not "is it sensitive?" -- which needs a
     judgment call -- but "is it on the public list?", which does not.

  2. PAYLOAD TOKEN SCAN. Every word in the outbound query values is checked against
     the vocabulary's token set. Since layer 1 means the payload should contain
     nothing but vocabulary language, anything else is a bug or an escape, and is
     blocked. This is what catches a caller that bypasses `validate_terms`.

  3. SESSION CORPUS SCAN. The outbound payload is checked against n-grams of the
     uploaded documents themselves. Layers 1 and 2 should make this unreachable; it
     is here so that the red-team test ("upload a distinctive phrase, run a search,
     grep the audit log for it") is testing a real backstop rather than a tautology.

Transport is guarded separately: loopback/private/CGNAT targets are refused (reusing
app/netstatus.py's classifier, so SSRF cannot reach the model endpoint or the Demo
Center's own services), responses are size-capped, and each run has a request budget.

Every decision -- allowed or blocked -- is recorded and emitted as an `egress` event.
Nothing leaves without a line in that log.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse, parse_qsl

import httpx

from .netstatus import classify_host

# Host classes that may never be contacted. "public" is the only permitted class:
# loopback/private/cgnat would let a crafted URL reach the local model server, the
# Demo Center's install API, or anything else on the operator's network.
_ALLOWED_HOST_CLASSES = frozenset({"public"})

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PUNCT_RE = re.compile(r"[^a-z0-9]+")

# Structural words a funding API's own query grammar needs, which carry no
# institutional meaning. Kept deliberately tiny and explicit.
_STRUCTURAL_TOKENS = frozenset({
    "and", "or", "not", "the", "of", "for", "in", "on", "to", "a", "an",
    "posted", "forecasted", "closed", "archived", "grant", "grants",
    "opportunity", "opportunities", "json", "true", "false",
})


class EgressBlocked(Exception):
    """Raised when a request is refused. Carries the reason for the audit log."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def normalize(text: str) -> str:
    """Lowercase, strip punctuation to spaces, collapse whitespace."""
    return _PUNCT_RE.sub(" ", text.lower()).strip()


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# -- vocabulary ----------------------------------------------------------------


@dataclass(frozen=True)
class Vocabulary:
    """The allowlist. Terms not in here do not leave the device."""

    entries: frozenset[str]
    token_set: frozenset[str]

    @classmethod
    def load(cls, vocab_dir: Path) -> "Vocabulary":
        entries: set[str] = set()

        field_terms = vocab_dir / "field_terms.txt"
        if field_terms.is_file():
            for raw in field_terms.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if line and not line.startswith("#"):
                    entries.add(normalize(line))

        taxonomy = vocab_dir / "taxonomy.json"
        if taxonomy.is_file():
            doc = json.loads(taxonomy.read_text(encoding="utf-8"))
            for key in ("field_terms", "research_terms"):
                for term in doc.get(key, []):
                    entries.add(normalize(term))
            for values in (doc.get("taxonomy") or {}).values():
                for value in values:
                    entries.add(normalize(value))

        entries.discard("")
        token_set = {tok for entry in entries for tok in tokens(entry)}
        return cls(entries=frozenset(entries), token_set=frozenset(token_set | _STRUCTURAL_TOKENS))

    def matches(self, normalized_term: str) -> bool:
        return normalized_term in self.entries

    def unknown_tokens(self, text: str) -> list[str]:
        """Tokens in `text` that are not vocabulary language. Empty means clean."""
        return [t for t in tokens(text) if t not in self.token_set]


# -- session corpus (the material that must not leave) -------------------------


@dataclass
class SessionCorpus:
    """Normalized text of everything the user uploaded this session.

    Used only to *block*: the guard never sends corpus content anywhere, it checks
    outbound payloads against it.
    """

    normalized_docs: list[str] = field(default_factory=list)
    ngram_len: int = 3
    _ngrams: set[str] = field(default_factory=set, repr=False)

    @classmethod
    def from_dir(cls, session_dir: Path | None, ngram_len: int = 3) -> "SessionCorpus":
        corpus = cls(ngram_len=ngram_len)
        if session_dir is None or not Path(session_dir).is_dir():
            return corpus
        # Read raw bytes and decode leniently rather than going through the loaders:
        # this is a security check, so it should see the file's literal content and
        # never fail closed-open because a loader could not parse something.
        for path in sorted(Path(session_dir).rglob("*")):
            if path.is_file():
                try:
                    corpus.add_text(path.read_bytes().decode("utf-8", errors="ignore"))
                except OSError:
                    continue
        return corpus

    def add_text(self, text: str) -> None:
        norm = normalize(text)
        if not norm:
            return
        self.normalized_docs.append(norm)
        words = norm.split()
        n = self.ngram_len
        for i in range(len(words) - n + 1):
            self._ngrams.add(" ".join(words[i : i + n]))

    def contains_ngram_of(self, text: str) -> str | None:
        """Return the first n-gram of `text` that also occurs in the corpus."""
        words = normalize(text).split()
        n = self.ngram_len
        for i in range(len(words) - n + 1):
            gram = " ".join(words[i : i + n])
            if gram in self._ngrams:
                return gram
        return None

    def contains_verbatim(self, text: str) -> bool:
        norm = normalize(text)
        if not norm:
            return False
        padded = f" {norm} "
        return any(padded in f" {doc} " for doc in self.normalized_docs)

    @property
    def doc_count(self) -> int:
        return len(self.normalized_docs)


# -- decisions and audit records -----------------------------------------------


@dataclass(frozen=True)
class TermDecision:
    term: str
    allowed: bool
    reason: str
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"term": self.term, "allowed": self.allowed, "reason": self.reason, "detail": self.detail}


@dataclass(frozen=True)
class EgressRecord:
    ts: float
    method: str
    url: str
    host_class: str
    terms: list[TermDecision]
    allowed: bool
    reason: str
    bytes_out: int = 0
    bytes_in: int = 0
    status: int | None = None
    elapsed_s: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "method": self.method,
            "url": self.url,
            "host_class": self.host_class,
            "terms": [t.as_dict() for t in self.terms],
            "allowed": self.allowed,
            "reason": self.reason,
            "bytes_out": self.bytes_out,
            "bytes_in": self.bytes_in,
            "status": self.status,
            "elapsed_s": self.elapsed_s,
        }


# -- term validation (layer 1) -------------------------------------------------


class TermValidator:
    """Default-deny. A term leaves only if it is on the published vocabulary."""

    def __init__(self, vocabulary: Vocabulary, corpus: SessionCorpus, max_words: int = 6) -> None:
        self._vocab = vocabulary
        self._corpus = corpus
        self._max_words = max_words

    def validate(self, term: str) -> TermDecision:
        raw = (term or "").strip()
        if not raw:
            return TermDecision(term, False, "empty")

        norm = normalize(raw)
        if not norm:
            return TermDecision(term, False, "empty")

        word_count = len(norm.split())
        if word_count > self._max_words:
            return TermDecision(term, False, "too_long", f"{word_count} words > {self._max_words}")

        # The vocabulary decides first, and a match wins outright -- including when
        # the same words also appear in an upload. A published field name is public
        # language, not institutional data: "robotics" does not become a secret
        # because the strategic plan says it. Checking verbatim-from-upload ahead of
        # this looked safer and was actually useless -- it banned every field the
        # institution actually works in, which is precisely what discovery must
        # search for. Containment comes from default-deny below, not from pretending
        # public words are private.
        if self._vocab.matches(norm):
            return TermDecision(term, True, "vocabulary_match", norm)

        # Everything else is refused. The checks below do not change that verdict;
        # they choose the most informative reason for it, because "this came out of
        # your own document" is what the audit panel needs to show, and it is a far
        # better answer on stage than a flat "not on the list".
        if self._corpus.contains_verbatim(norm):
            return TermDecision(term, False, "verbatim_from_upload")

        if any(ch.isdigit() for ch in norm):
            return TermDecision(term, False, "contains_digits", "identifiers and figures never leave")

        return TermDecision(term, False, "not_in_vocabulary", "no entry in data/vocabulary/")

    def validate_all(self, terms: Iterable[str]) -> list[TermDecision]:
        return [self.validate(t) for t in terms]


# -- the guard -----------------------------------------------------------------


class EgressGuard:
    """Every outbound byte from the app passes through here."""

    def __init__(
        self,
        vocabulary: Vocabulary,
        corpus: SessionCorpus,
        *,
        max_requests: int = 40,
        timeout_s: float = 30.0,
        max_response_bytes: int = 5_000_000,
        max_term_words: int = 6,
        trusted_endpoints: Iterable[str] = (),
        on_record: Callable[[EgressRecord], None] | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.vocabulary = vocabulary
        self.corpus = corpus
        self.validator = TermValidator(vocabulary, corpus, max_words=max_term_words)
        self._max_requests = max_requests
        self._timeout_s = timeout_s
        self._max_response_bytes = max_response_bytes
        self._on_record = on_record
        self._client = client
        self.records: list[EgressRecord] = []
        # Infrastructure the OPERATOR configured, which the model cannot choose --
        # today just a self-hosted SearXNG. The host-class check exists to stop a
        # crafted URL reaching the model endpoint or the Demo Center's own services,
        # and it would equally block a local search service the operator deliberately
        # stood up. Exact-prefix, config-only, and everything else about the request
        # is unchanged: the query terms still go through validation, and the request
        # still appears in the audit log. What the search engine then sends upstream
        # is the validated terms and nothing else.
        self._trusted = tuple(t.rstrip("/") for t in trusted_endpoints if t)
        # Tokens seen in responses WE received. An identifier the catalogue just
        # handed us -- an opportunity id, a URL query key -- is not the institution's
        # data, and refusing to echo it back makes the catalogue unusable: the first
        # live run burned its whole step budget on fetchOpportunity calls blocked for
        # carrying the numeric id the previous search had just returned. Echoing is
        # allowed only when the token is ALSO absent from the upload corpus, so a
        # value that appears in both (an award number the institution cites, say)
        # stays blocked.
        self._inbound_tokens: set[str] = set()
        self._max_inbound_tokens = 50_000
        # Normalized text of responses, used to tell a link we were GIVEN from a URL
        # the model CONSTRUCTED. Following a link found on a public page is ordinary
        # research; assembling a URL whose query string carries institutional text is
        # exfiltration, and only the second should be refused.
        self._inbound_text: list[str] = []
        self._max_inbound_text_chars = 2_000_000

    # -- audit ----------------------------------------------------------------

    @property
    def request_count(self) -> int:
        return sum(1 for r in self.records if r.allowed)

    @property
    def terms_sent(self) -> list[str]:
        return [t.term for r in self.records if r.allowed for t in r.terms if t.allowed]

    def _is_trusted(self, url: str) -> bool:
        """Exact-prefix match on a configured endpoint, at a path boundary.

        The boundary check matters: without it a trusted "http://127.0.0.1:8096"
        would also vouch for "http://127.0.0.1:8096.evil.test".
        """
        return any(url == t or url.startswith(t + "/") or url.startswith(t + "?") for t in self._trusted)

    def saw_inbound(self, needle: str) -> bool:
        """Did a response we received contain this string?"""
        norm = normalize(needle)
        return bool(norm) and any(norm in doc for doc in self._inbound_text)

    def note_inbound(self, text: str) -> None:
        """Record a response so its identifiers and links may be used."""
        if sum(len(d) for d in self._inbound_text) < self._max_inbound_text_chars:
            self._inbound_text.append(normalize(text))
        if len(self._inbound_tokens) >= self._max_inbound_tokens:
            return
        for tok in tokens(text):
            self._inbound_tokens.add(tok)
            if len(self._inbound_tokens) >= self._max_inbound_tokens:
                return

    def _record(self, record: EgressRecord) -> EgressRecord:
        self.records.append(record)
        if self._on_record:
            self._on_record(record)
        return record

    # -- checks ---------------------------------------------------------------

    def _scan_payload(self, values: Iterable[str]) -> tuple[str, str] | None:
        """Layers 2 and 3 over the free-text values actually being sent.

        Returns (reason, detail) on a violation, or None when clean.
        """
        for value in values:
            if not value:
                continue

            # Layer 2 first: every word actually being sent must be vocabulary
            # language. Layer 1 should already guarantee this, so anything caught
            # here is a caller that bypassed validate_terms.
            unknown = self.vocabulary.unknown_tokens(value)
            # A token the server itself gave us may go back out, unless it also
            # occurs in the institution's own documents -- in which case it is their
            # data regardless of who else happens to publish it, and it stays put.
            unknown = [
                t for t in unknown
                if not (t in self._inbound_tokens and not self.corpus.contains_verbatim(t))
            ]
            if unknown:
                return ("payload_token_not_in_vocabulary", ", ".join(sorted(set(unknown))[:5]))

            # Layer 3: an n-gram of the upload appearing in the payload. Only
            # n-grams carrying at least one non-vocabulary token count -- a run of
            # words that is entirely public field language is not a leak, and
            # flagging it would block legitimate multi-term queries that happen to
            # echo a phrase from the user's own documents. With layer 2 passing,
            # this is unreachable by construction; it stays as the assertion that
            # makes the red-team test meaningful rather than tautological.
            leaked = self.corpus.contains_ngram_of(value)
            if leaked and self.vocabulary.unknown_tokens(leaked):
                return ("payload_matches_upload", leaked)
        return None

    @staticmethod
    def _payload_values(
        url: str,
        params: dict[str, Any] | None,
        json_body: Any,
        *,
        include_url_query: bool = True,
    ) -> list[str]:
        """Every free-text value that will actually be transmitted."""
        values = [v for _, v in parse_qsl(urlparse(url).query)] if include_url_query else []
        if params:
            values.extend(str(v) for v in params.values())

        def walk(node: Any) -> None:
            if isinstance(node, str):
                values.append(node)
            elif isinstance(node, dict):
                for v in node.values():
                    walk(v)
            elif isinstance(node, (list, tuple)):
                for v in node:
                    walk(v)

        walk(json_body)
        return values

    # -- the one way out ------------------------------------------------------

    def request(
        self,
        method: str,
        url: str,
        *,
        terms: Iterable[str] = (),
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        term_list = list(terms)
        decisions = self.validator.validate_all(term_list)
        host = urlparse(url).hostname or ""
        host_class = classify_host(host)

        def refuse(reason: str, detail: str = "") -> EgressRecord:
            rec = self._record(
                EgressRecord(
                    ts=time.time(), method=method.upper(), url=url, host_class=host_class,
                    terms=decisions, allowed=False, reason=reason,
                )
            )
            raise EgressBlocked(reason, detail) from None

        if not self._is_trusted(url):
            if host_class not in _ALLOWED_HOST_CLASSES:
                refuse("host_not_public", f"{host or '<none>'} classified {host_class}")
            if urlparse(url).scheme != "https":
                refuse("scheme_not_https", urlparse(url).scheme or "<none>")
        if self.request_count >= self._max_requests:
            refuse("request_budget_exhausted", f"{self._max_requests} requests")

        rejected = [d for d in decisions if not d.allowed]
        if rejected:
            refuse("terms_rejected", "; ".join(f"{d.term!r} ({d.reason})" for d in rejected))

        # A URL that appeared verbatim in a page or API response we already received
        # is a link, not a message: its query string was written by the site, not by
        # the model. Scanning those values rejected ordinary navigation -- a live run
        # was refused "?prs=ATE" on an NSF programme page it had just been handed.
        # A URL we have not seen is still scanned in full.
        scan_url_query = not self.saw_inbound(url)
        violation = self._scan_payload(
            self._payload_values(url, params, json_body, include_url_query=scan_url_query)
        )
        if violation:
            refuse(*violation)

        body_bytes = json.dumps(json_body).encode() if json_body is not None else b""
        started = time.monotonic()
        client = self._client or httpx.Client(timeout=self._timeout_s, follow_redirects=False)
        try:
            response = client.request(
                method.upper(), url, params=params, json=json_body,
                headers=headers or {"Content-Type": "application/json"},
            )
        except httpx.HTTPError as exc:
            self._record(
                EgressRecord(
                    ts=time.time(), method=method.upper(), url=url, host_class=host_class,
                    terms=decisions, allowed=True, reason=f"transport_error: {exc}",
                    bytes_out=len(body_bytes), elapsed_s=time.monotonic() - started,
                )
            )
            raise
        finally:
            if self._client is None:
                client.close()

        content = response.content or b""
        if len(content) > self._max_response_bytes:
            self._record(
                EgressRecord(
                    ts=time.time(), method=method.upper(), url=url, host_class=host_class,
                    terms=decisions, allowed=True, reason="response_too_large",
                    bytes_out=len(body_bytes), bytes_in=len(content), status=response.status_code,
                    elapsed_s=time.monotonic() - started,
                )
            )
            raise EgressBlocked("response_too_large", f"{len(content)} bytes")

        self.note_inbound(response.text or "")
        self._record(
            EgressRecord(
                ts=time.time(), method=method.upper(), url=url, host_class=host_class,
                terms=decisions, allowed=True, reason="ok",
                bytes_out=len(body_bytes), bytes_in=len(content),
                status=response.status_code, elapsed_s=time.monotonic() - started,
            )
        )
        return response
