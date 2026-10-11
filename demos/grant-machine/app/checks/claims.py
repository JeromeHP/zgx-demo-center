"""Cross-check a submitted proposal's factual claims against the vault (v2 Phase 3).

In the proposal playbook the agent writes from vault evidence and `verify_citations`
catches anything it asserted without support. A *submitted* proposal has no
`[[doc_id]]` markers -- it was written by a human for a funder -- so that check cannot
apply, which left grading with nothing examining the submission's facts at all.

That gap was not theoretical. A test proposal claimed "graduating 120 completers
against a target of 120" while the institution's own prior-outcomes data recorded 84.
The grader scored that section 3/4 and praised it for "exceeding graduation targets".
A panel would not have.

So: pull the numeric claims out of a section, retrieve the evidence that ought to
support each, and report supported / contradicted / unverifiable. The retrieval and
number matching are deterministic; only the final judgement is the model's, and it is
given the evidence rather than asked to recall it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .citations import _normalize_number, _split_sentences

# Citation verification deliberately ignores bare single digits, to avoid matching
# stray numerals near a citation. Claim checking must not: "supported 6 graduate
# students" against a record of 4, and "9 papers" against a claim of 14, are exactly
# the overstatements this exists to catch, and the shared pattern made them invisible.
_CLAIM_NUMBER_RE = re.compile(
    r"\$\d[\d,]*(?:\.\d+)?"
    r"|\d+(?:\.\d+)?\s?%"
    r"|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"
    r"|\b\d+(?:\.\d+)?\b"
)

# Markdown headings ride along into the first sentence of a section otherwise, which
# makes the claim noisy and the retrieval worse.
_HEADING_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)

# Assertions of status, approval or commitment. These carry no figures, so a
# numbers-only extractor never saw them -- and they are exactly the claims a research
# administrator needs checked: "IRB approval is current" against a record showing it
# expired in March, "a data safety monitoring board is in place" against a plan that
# records it as not yet convened, "cost share is confirmed" against a register showing
# half of it pending. Deliberately a short, explainable list rather than an attempt to
# recognise assertions in general.
_STATUS_RE = re.compile(
    r"\b(?:is|are|was|were|has been|have been|remains|remain)\s+"
    r"(?:currently\s+|fully\s+|formally\s+)?"
    r"(?:current|active|approved|in place|in good standing|established|constituted|"
    r"convened|confirmed|committed|secured|obtained|registered|certified|valid)\b"
    r"|\b(?:all|each|every)\s+\w+(?:\s+\w+){0,3}\s+(?:were|have been|has been)\s+"
    r"(?:completed|executed|activated|signed|met|achieved|provided)\b",
    re.IGNORECASE,
)

# Sentences carrying a number are where overstatement lives. Prose without one is a
# matter of judgement, which is what the section score is already for.
_MAX_CLAIMS = 12
_EVIDENCE_CHUNKS = 3


@dataclass
class ClaimCheck:
    claim: str
    numbers: list[str]
    verdict: str  # supported | contradicted | unverifiable
    detail: str = ""
    evidence: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"claim": self.claim, "numbers": self.numbers, "verdict": self.verdict,
                "detail": self.detail, "evidence": self.evidence}


def extract_numeric_claims(text: str, limit: int = _MAX_CLAIMS) -> list[tuple[str, list[str]]]:
    """Checkable claims: sentences carrying figures, and sentences asserting status.

    Kept under the original name because that is what callers use; the status half was
    added when a clinical trial proposal asserting a current IRB approval -- against a
    record showing it had expired -- sailed through untouched because the sentence
    contained no digits.
    """
    claims: list[tuple[str, list[str]]] = []
    for sentence in _split_sentences(_HEADING_RE.sub("", text or "")):
        cleaned = sentence.strip()
        if not cleaned:
            continue
        numbers = [_normalize_number(m.group(0)) for m in _CLAIM_NUMBER_RE.finditer(cleaned)]
        numbers = [n for n in numbers if n]
        if numbers or _STATUS_RE.search(cleaned):
            claims.append((cleaned, sorted(set(numbers))))
        if len(claims) >= limit:
            break
    return claims


_JUDGE_PROMPT = """A grant proposal makes this claim:

"{claim}"

Here is the institution's own documentation that the claim should rest on:

{evidence}

Decide whether the documentation SUPPORTS the claim, CONTRADICTS it, or is
insufficient to tell (UNVERIFIABLE).

Be strict about numbers: if the claim states a figure and the documentation states a
different figure for the same thing, that is CONTRADICTED, however small the gap.

Be equally strict about status. If the claim asserts that something is approved, current,
in place, constituted, confirmed or committed, and the documentation records it as
expired, pending, not yet convened, partial or unsigned, that is CONTRADICTED.

If the documentation simply does not cover the claim, that is UNVERIFIABLE -- do not
guess. Return JSON only:
{{"verdict": "supported" | "contradicted" | "unverifiable", "detail": "<one sentence>"}}
"""


def check_claims(
    text: str,
    index: Any,
    llm: Any,
    *,
    limit: int = _MAX_CLAIMS,
    evidence_chunks: int = _EVIDENCE_CHUNKS,
) -> list[ClaimCheck]:
    """Check each numeric claim in `text` against evidence retrieved from `index`."""
    results: list[ClaimCheck] = []

    for claim, numbers in extract_numeric_claims(text, limit=limit):
        hits = index.search(claim, k=evidence_chunks) if index is not None else []
        if not hits:
            results.append(ClaimCheck(claim, numbers, "unverifiable",
                                      "no supporting document was found in the vault"))
            continue

        # The chunk, not the excerpt: the excerpt is a window centred on a query token
        # and routinely stops just before the figure being checked.
        evidence_text = "\n\n".join(f"[{h.doc_id}] {(h.text or h.snippet or '')[:2500]}" for h in hits)
        doc_ids = [h.doc_id for h in hits]

        # There is deliberately NO "the numbers all appear, so it is supported"
        # shortcut. It was written, and its own test killed it: for the claim
        # "graduated 120 completers against a target of 120", the record
        # "Completers: 84 against a target of 120" does contain 120 -- as the target.
        # The shortcut declared it supported and never asked. Presence of a figure is
        # not support for the claim it was lifted into, and this is the one check
        # whose entire purpose is catching numbers that look right and are not.
        try:
            reply = llm.chat(
                "reviewer",
                [{"role": "user", "content": _JUDGE_PROMPT.format(claim=claim, evidence=evidence_text)}],
                json_mode=True,
            )
            parsed = json.loads(reply.content)
            verdict = str(parsed.get("verdict", "")).lower().strip()
            detail = str(parsed.get("detail", ""))[:300]
        except Exception:  # noqa: BLE001 -- a judging failure must not fail the grade
            verdict, detail = "", ""

        if verdict not in ("supported", "contradicted", "unverifiable"):
            verdict, detail = "unverifiable", detail or "the check could not be completed"
        results.append(ClaimCheck(claim, numbers, verdict, detail, doc_ids))

    return results
