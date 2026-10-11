"""Citation existence + number verification (Phase 5.4).

Citation syntax in drafts: [[doc_id]] or [[doc_id#chunk_id]].
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.ingest.chunker import chunk_document
from app.ingest.loaders import Document, load

# Chunk ids are already `doc_id#cNNN`, so a model shown both a doc_id and a chunk_id
# writes [[doc#doc#c000]] often enough to matter. The two-segment pattern did not
# match that at all, which meant the citation was invisible -- and a draft whose
# every citation was malformed passed verification with nothing to verify.
# Any [[...]] marker at all, so one the citation pattern rejects can be reported
# rather than silently skipped.
_MALFORMED_RE = re.compile(r"\[\[[^\]]{0,200}\]\]")

_CITATION_RE = re.compile(r"\[\[([A-Za-z0-9_\-]+)(?:#([A-Za-z0-9_\-]+))?(?:#([A-Za-z0-9_\-]+))?\]\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?!\[\[)(?=[A-Z\"'(0-9])")

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"

_NUMBER_RE = re.compile(
    r"\$\d[\d,]*(?:\.\d+)?"
    rf"|\d{{4}}-\d{{2}}-\d{{2}}"
    rf"|(?:{_MONTHS})\s+\d{{4}}"
    r"|\d+(?:\.\d+)?\s?%"
    r"|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"  # thousands-separated integers of any size
    r"|\b\d{2,}(?:\.\d+)?\b"  # bare integers/decimals of 2+ digits (>= 10)
)


@dataclass
class Citation:
    doc_id: str
    chunk_id: str | None


@dataclass
class UnsupportedClaim:
    sentence: str
    number: str
    citations: list[str]


@dataclass
class UncitedNumber:
    sentence: str
    number: str


@dataclass
class CitationCheckResult:
    passed: bool
    missing_docs: list[str] = field(default_factory=list)
    unsupported_claims: list[UnsupportedClaim] = field(default_factory=list)
    uncited_numbers: list[UncitedNumber] = field(default_factory=list)
    # Bracketed markers the pattern could not parse. A draft whose every citation is
    # malformed used to verify clean, because unparsed markers were invisible and the
    # check had nothing left to object to.
    malformed_citations: list[str] = field(default_factory=list)
    citation_count: int = 0


def _split_sentences(text: str) -> list[str]:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if not collapsed:
        return []
    return [p.strip() for p in _SENTENCE_SPLIT_RE.split(collapsed) if p.strip()]


def _normalize_number(raw: str) -> str:
    s = raw.strip().replace("$", "").replace(",", "")
    if s.endswith("%"):
        s = s[:-1].strip()
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def _normalize_doc_text(text: str) -> str:
    return text.replace("$", "").replace(",", "")


def _number_in_text(number: str, normalized_text: str) -> bool:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", number) or re.fullmatch(rf"(?:{_MONTHS})\s+\d{{4}}", number):
        return number in normalized_text
    # Simplification (documented limitation): a decimal-point neighbor can still
    # cause a false match (e.g. "84" inside "0.84"); acceptable for demo scope.
    pattern = re.compile(r"(?<!\d)" + re.escape(number) + r"(?!\d)")
    return bool(pattern.search(normalized_text))


class _DocCache:
    """Loads and caches vault documents plus their valid chunk ids."""

    def __init__(self, vault_dir: Path, known_doc_ids: set[str] | None):
        self._vault_dir = Path(vault_dir)
        self._known_doc_ids = known_doc_ids
        self._docs: dict[str, Document | None] = {}
        self._chunk_ids: dict[str, set[str]] = {}

    def get(self, doc_id: str) -> Document | None:
        if doc_id in self._docs:
            return self._docs[doc_id]
        if self._known_doc_ids is not None and doc_id not in self._known_doc_ids:
            self._docs[doc_id] = None
            return None
        found = None
        for ext in (".md", ".html", ".htm", ".pdf", ".csv"):
            candidate = self._vault_dir / f"{doc_id}{ext}"
            if candidate.is_file():
                found = load(candidate, doc_id=doc_id)
                break
        self._docs[doc_id] = found
        if found is not None:
            self._chunk_ids[doc_id] = {c.chunk_id for c in chunk_document(found)}
        return found

    def has_chunk(self, doc_id: str, chunk_id: str) -> bool:
        self.get(doc_id)
        return chunk_id in self._chunk_ids.get(doc_id, set())


def _parse_citation(m: "re.Match[str]") -> Citation:
    """One match -> a doc_id and, if given, the *canonical* chunk id.

    The chunk ids the retriever hands the model are already `doc_id#cNNN`, so the two
    ways a model naturally writes a chunk citation are `[[doc#c000]]` (splicing the
    suffix on) and `[[doc#doc#c000]]` (pasting the whole chunk id after the doc id).
    Both mean the same chunk. Neither used to resolve: the chunk id was taken as the
    raw second segment and compared against `doc#cNNN`, so *every* chunk-level
    citation was reported as a missing document -- the syntax the prompt asks for
    could not pass. Rebuild the canonical id instead of trusting the segmentation.
    """
    doc_id = m.group(1)
    second, third = m.group(2), m.group(3)
    if third is not None:
        # [[doc#doc#c000]] -- the middle segment is the doc id repeated.
        suffix = third
    elif second is not None:
        suffix = second
    else:
        return Citation(doc_id=doc_id, chunk_id=None)
    return Citation(doc_id=doc_id, chunk_id=f"{doc_id}#{suffix}")


def verify_citations(
    draft_text: str,
    vault_dir: Path,
    known_doc_ids: set[str] | None = None,
) -> CitationCheckResult:
    cache = _DocCache(vault_dir, known_doc_ids)
    missing_docs: list[str] = []
    unsupported_claims: list[UnsupportedClaim] = []
    uncited_numbers: list[UncitedNumber] = []

    def _add_missing(ref: str) -> None:
        if ref not in missing_docs:
            missing_docs.append(ref)

    for sentence in _split_sentences(draft_text):
        citations = [_parse_citation(m) for m in _CITATION_RE.finditer(sentence)]

        valid_citations: list[Citation] = []
        for c in citations:
            doc = cache.get(c.doc_id)
            if doc is None:
                _add_missing(c.doc_id)
                continue
            if c.chunk_id is not None and not cache.has_chunk(c.doc_id, c.chunk_id):
                _add_missing(c.chunk_id)
                continue
            valid_citations.append(c)

        sentence_no_citations = _CITATION_RE.sub("", sentence)
        numbers = [m.group(0) for m in _NUMBER_RE.finditer(sentence_no_citations)]
        if not numbers:
            continue

        if not citations:
            uncited_numbers.extend(UncitedNumber(sentence=sentence, number=n) for n in numbers)
            continue

        cited_texts = [_normalize_doc_text(cache.get(c.doc_id).text) for c in valid_citations if cache.get(c.doc_id)]

        for n in numbers:
            normalized = _normalize_number(n)
            if not any(_number_in_text(normalized, t) for t in cited_texts):
                unsupported_claims.append(
                    UnsupportedClaim(sentence=sentence, number=n, citations=[c.doc_id for c in citations])
                )

    # Anything in double brackets that did not parse as a citation.
    malformed = [
        m.group(0) for m in _MALFORMED_RE.finditer(draft_text)
        if not _CITATION_RE.fullmatch(m.group(0))
    ]
    citation_count = len(_CITATION_RE.findall(draft_text))

    # A draft with no parseable citations at all has not been verified -- it has been
    # found to contain nothing verifiable, which is the opposite. This is the same
    # shape of defect as an endpoint check that passes a server returning no reasoning:
    # the check answered about an empty set and reported success.
    passed = (
        not missing_docs
        and not unsupported_claims
        and not malformed
        and citation_count > 0
    )
    return CitationCheckResult(
        passed=passed,
        malformed_citations=malformed,
        citation_count=citation_count,
        missing_docs=missing_docs,
        unsupported_claims=unsupported_claims,
        uncited_numbers=uncited_numbers,
    )
