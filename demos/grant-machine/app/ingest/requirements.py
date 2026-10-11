"""Extract proposal requirements from a solicitation (v2 Phase 2: Intake).

Until now the requirements were a build-time artifact: `scripts/extract_requirements.py`
produced `data/solicitation/requirements.json` for NSF 24-584 once, by hand-reviewed
draft, and the app loaded that one file forever. Explore mode's UI, the build plan and
the presenter script all describe uploading your own solicitation, but the upload
became the *vault* and the agent kept working NSF 24-584's 54 requirements. This
module is the missing half.

The extraction logic itself is the script's, moved here so there is one implementation
rather than two that drift: section splitting, per-section extraction, the exact
substring validation of every `source_quote`, and near-duplicate removal.

Three properties worth keeping in mind:

  - **Quotes are validated, not trusted.** A candidate survives only if its
    `source_quote` appears character-for-character in the section it came from. That is
    the anti-hallucination guard for the whole feature: a requirement the model invented
    cannot produce a quote that is really in the document.
  - **Thinking is off.** This is mechanical extract-and-quote work. The original build
    measured 2-6+ minutes per section with thinking on for marginal accuracy gain;
    measured here at ~18s/section with it off.
  - **Results are cached by content hash.** Re-running on the same document is instant,
    which matters because a demo may well extract the same solicitation repeatedly.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

CATEGORIES = [
    "project_description",
    "merit_review",
    "program_specific",
    "formatting",
    "budget",
    "supplementary_documents",
    "eligibility",
]

# Pure boilerplate/administrative headings with no proposer-facing requirements.
_SKIP_HEADINGS = {
    "document history",
    "create a pdf",
    "about the national science foundation",
    "privacy act and public burden statements",
    "viii. agency contacts",
    "agency contacts",
    "table of contents",
    "contents",
}

_HEADING_RE = re.compile(r"^(#{2,4})\s+(.*)$", re.MULTILINE)
_WS_RE = re.compile(r"\s+")

# Bumped when the prompt or validation changes, so cached results from an older
# extraction are not silently reused.
PROMPT_VERSION = "2"


@dataclass
class ExtractionProgress:
    index: int
    total: int
    heading: str
    candidates: int


def split_sections(md_text: str) -> list[tuple[str, str]]:
    matches = list(_HEADING_RE.finditer(md_text))
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        if heading.lower() in _SKIP_HEADINGS:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        body = md_text[start:end].strip()
        if len(body) > 80:
            sections.append((heading, body))
    return sections


def _extraction_prompt(heading: str, body: str) -> str:
    return (
        "You are extracting proposal requirements from a section of a grant solicitation.\n\n"
        f"Section heading: {heading}\n\n"
        f"Section text:\n{body}\n\n"
        "List every concrete requirement a proposer must satisfy from this section text (formatting rules, "
        "required elements, eligibility rules, budget rules, review criteria, program-specific questions). "
        "For each one, give:\n"
        f'- "category": one of {CATEGORIES}\n'
        '- "text": a short plain-language restatement of the requirement\n'
        '- "source_quote": an EXACT, verbatim substring copied character-for-character from the section text '
        "above (do not paraphrase or summarize this field -- it must be found exactly as-is in the text)\n\n"
        "If this section has no proposer-facing requirements (pure background, history, or contact info), "
        'return an empty list. Return JSON only, in the form {"requirements": [...]}.'
    )


def extract_from_section(client: Any, heading: str, body: str) -> list[dict[str, Any]]:
    """Candidates from one section. A failure here yields nothing; it never aborts the run."""
    try:
        result = client.chat("extractor", [{"role": "user", "content": _extraction_prompt(heading, body)}], json_mode=True)
    except Exception:  # noqa: BLE001 -- one bad section must not abort the whole extraction
        return []
    try:
        data = json.loads(result.content)
    except (json.JSONDecodeError, TypeError):
        return []

    # Models return the array either wrapped or bare. The wrapped form was all the
    # NSF document ever produced, so `data.get(...)` on a bare list raised
    # AttributeError -- which was not in the caught set and killed an entire run on
    # the first real NIH announcement. Accept both shapes.
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("requirements") or data.get("items") or []
    else:
        return []
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip().lower())


def dedupe(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        if not any(
            k.get("category") == c.get("category")
            and difflib.SequenceMatcher(None, _normalize(k.get("text")), _normalize(c.get("text"))).ratio() > 0.85
            for k in kept
        ):
            kept.append(c)
    return kept


def validate(candidates: list[dict[str, Any]], body: str, heading: str) -> tuple[list[dict[str, Any]], int]:
    """Keep only candidates whose source_quote really occurs in `body`.

    This is the guard that makes extraction trustworthy: a requirement the model
    invented cannot supply a quote that is genuinely in the document.
    """
    kept, discarded = [], 0
    for c in candidates:
        if not isinstance(c, dict):
            discarded += 1
            continue
        quote, text, category = c.get("source_quote"), c.get("text"), c.get("category")
        if not (quote and text and category in CATEGORIES) or quote not in body:
            discarded += 1
            continue
        c["source_heading"] = heading
        kept.append(c)
    return kept, discarded


def content_key(md_text: str, model: str) -> str:
    digest = hashlib.sha256(f"{PROMPT_VERSION}\0{model}\0{md_text}".encode()).hexdigest()
    return digest[:32]


def extract_requirements(
    md_text: str,
    client: Any,
    *,
    model: str = "",
    solicitation_id: str = "",
    program: str = "",
    source_url: str = "",
    source_sha256: str = "",
    tracks: dict[str, Any] | None = None,
    max_workers: int = 4,
    on_progress: Callable[[ExtractionProgress], None] | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Extract a requirements document from solicitation Markdown.

    Returns the same shape the app already consumes from
    `data/solicitation/requirements.json`, so everything downstream -- the compliance
    matrix, `get_requirements`, the reviewer's criteria -- works unchanged.
    """
    key = content_key(md_text, model)
    cache_path = (cache_dir / f"{key}.json") if cache_dir else None
    if cache_path and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if on_progress:
                total = len(cached.get("requirements", []))
                on_progress(ExtractionProgress(index=1, total=1, heading="(cached)", candidates=total))
            return cached
        except (json.JSONDecodeError, OSError):
            pass  # a corrupt cache entry is not a reason to fail; re-extract

    sections = split_sections(md_text)
    total = len(sections)
    results: list[list[dict[str, Any]]] = [[] for _ in sections]
    discarded = 0
    done = 0

    def work(i: int) -> tuple[int, list[dict[str, Any]], int]:
        heading, body = sections[i]
        kept, bad = validate(extract_from_section(client, heading, body), body, heading)
        return i, kept, bad

    # Modest concurrency: measured 1.29x on GB10, since this is GPU-bound rather than
    # request-bound. Worth having, not worth tuning.
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        for i, kept, bad in pool.map(work, range(total)):
            results[i] = kept
            discarded += bad
            done += 1
            if on_progress:
                on_progress(ExtractionProgress(index=done, total=total, heading=sections[i][0], candidates=len(kept)))

    candidates = [c for group in results for c in group]
    deduped = dedupe(candidates)
    for i, c in enumerate(deduped, start=1):
        c["id"] = f"R-{i:03d}"
        c.setdefault("applies_to_tracks", [])
        c.setdefault("section_link", None)
        c.setdefault("verification", "agent")
        # Never claim human review for something extracted at runtime.
        c["human_reviewed"] = False

    doc = {
        "solicitation_id": solicitation_id or "(uploaded solicitation)",
        "program": program,
        "source_url": source_url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_sha256 or hashlib.sha256(md_text.encode()).hexdigest(),
        "tracks": tracks or {},
        "deadline_rule": "",
        "demo_deadline": "",
        "requirements": deduped,
        "extraction": {
            "sections_scanned": total,
            "candidates": len(candidates),
            "discarded_unverifiable_quote": discarded,
            "after_dedupe": len(deduped),
            "model": model,
            "prompt_version": PROMPT_VERSION,
        },
    }

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


_SECTIONS_PROMPT = """You are planning the structure of a grant proposal for this solicitation.

Solicitation: {solicitation_id} {program}

Requirements extracted from it:
{requirements}

Propose the narrative sections a proposer must write. Use the solicitation's own names for them
where it names them. Return JSON only:
{{"sections": [{{"id": "snake_case_id", "title": "Section Title", "word_target": "e.g. 400-700 words"}}]}}

Rules:
- Between 2 and 6 sections. Only sections the proposer actually writes as narrative prose --
  not budgets, forms, letters or attachments.
- Order them the way the solicitation does.
"""


def propose_sections(
    client: Any, requirements_doc: dict[str, Any], *, max_sections: int = 6, strict: bool = False
) -> list[dict[str, str]]:
    """Ask the model what narrative sections this solicitation calls for.

    The shipped build hardcodes three NSF ATE section names. An arbitrary solicitation
    has its own, so they are derived from the extracted requirements rather than
    assumed. Falls back to a single generic section if the model returns nothing
    usable -- a run with no sections at all would simply have nothing to do.
    """
    reqs = requirements_doc.get("requirements", [])
    listing = "\n".join(f"- [{r.get('category')}] {r.get('text')}" for r in reqs[:60])
    prompt = _SECTIONS_PROMPT.format(
        solicitation_id=requirements_doc.get("solicitation_id", ""),
        program=requirements_doc.get("program", ""),
        requirements=listing,
    )
    try:
        result = client.chat("extractor", [{"role": "user", "content": prompt}], json_mode=True)
        data = json.loads(result.content)
        sections = data.get("sections") or []
    except Exception as exc:  # noqa: BLE001
        # A run must not die because section naming failed -- but this used to fall
        # back to one generic section without a word anywhere, and an unreachable
        # model server produced exactly the same committed output as a successful
        # call. `strict` is for build scripts, which have no reason to accept that.
        if strict:
            raise
        logger.warning("section proposal failed (%s); falling back to a single generic section", exc)
        sections = []

    cleaned: list[dict[str, str]] = []
    for s in sections:
        if not isinstance(s, dict):
            continue
        sid = re.sub(r"[^a-z0-9_]+", "_", str(s.get("id") or "").lower()).strip("_")
        title = str(s.get("title") or "").strip()
        if sid and title:
            cleaned.append({"id": sid, "title": title, "word_target": str(s.get("word_target") or "400-700 words")})
        if len(cleaned) >= max_sections:
            break

    return cleaned or [{"id": "project_narrative", "title": "Project Narrative", "word_target": "600-900 words"}]
