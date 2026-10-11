"""Assemble the egress allowlist vocabulary (Phase 0).

The egress guard default-denies: a search term that does not match this vocabulary
never leaves the device. So the vocabulary is a security artifact, and every entry
needs a public, citable origin -- it must be language that appears in a funding
announcement, never language specific to an institution, person or project.

Two sources:
  1. data/vocabulary/field_terms.txt  -- committed, hand-authored generic field names
  2. the grants.gov search2 facets    -- the authoritative federal funding taxonomy
     (agencies, funding categories, eligibilities, funding instruments), fetched live

Writes data/vocabulary/taxonomy.json and records provenance in DATA_SOURCES.md.

Usage:
    python scripts/build_vocabulary.py            # fetch + write
    python scripts/build_vocabulary.py --offline  # rebuild from the committed file only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
VOCAB_DIR = REPO_ROOT / "data" / "vocabulary"
FIELD_TERMS_PATH = VOCAB_DIR / "field_terms.txt"
TAXONOMY_PATH = VOCAB_DIR / "taxonomy.json"

SEARCH2_URL = "https://api.grants.gov/v1/api/search2"
FACET_KEYS = ("fundingCategories", "agencies", "eligibilities", "fundingInstruments")

REPORTER_URL = "https://api.reporter.nih.gov/v2/projects/search"
REPORTER_YEARS = [2022, 2023, 2024]
REPORTER_PAGE = 100
REPORTER_PAGES_PER_YEAR = 6

# A hand lexicon of 372 broad field names was too coarse for real research groups:
# a materials lab's "neutron characterization" was refused, which makes Discover
# useless to exactly the people it is for. NIH RePORTER publishes the MeSH-style
# indexing terms assigned to funded projects -- real, public, discipline-specific
# research vocabulary, which is precisely the language a search term should be
# allowed to use.
#
# Entries are filtered rather than taken wholesale. Very short tokens and bare
# common words would inflate the token set that layer 2 of the egress guard checks
# against, weakening a backstop for no retrieval benefit.
_MIN_TERM_CHARS = 5
_MAX_TERM_WORDS = 5
_TERM_RE = re.compile(r"<([^<>]{3,80})>")
_BAD_TERM_RE = re.compile(r"[^a-zA-Z0-9 \-'/]")


def read_field_terms(path: Path = FIELD_TERMS_PATH) -> list[str]:
    terms = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        terms.append(line.lower())
    return sorted(set(terms))


def _walk_facet(entries: list[dict[str, Any]], out: set[str]) -> None:
    """Collect labels and codes, descending into subAgencyOptions."""
    for entry in entries:
        label = (entry.get("label") or "").strip()
        value = (entry.get("value") or "").strip()
        if label:
            out.add(label.lower())
        if value:
            out.add(value.lower())
        subs = entry.get("subAgencyOptions")
        if subs:
            _walk_facet(subs, out)


def fetch_taxonomy() -> dict[str, list[str]]:
    """Pull the facet taxonomy off a minimal search2 call.

    Imported lazily and called only by this build script -- the running app never
    reaches the network except through app/egress.py.
    """
    import httpx

    body = {"rows": 1, "oppStatuses": "posted|forecasted|closed"}
    resp = httpx.post(SEARCH2_URL, json=body, timeout=60.0)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("errorcode") not in (0, None):
        raise RuntimeError(f"search2 returned errorcode={payload.get('errorcode')}: {payload.get('msg')}")
    data = payload.get("data") or {}

    taxonomy: dict[str, list[str]] = {}
    for key in FACET_KEYS:
        collected: set[str] = set()
        _walk_facet(data.get(key) or [], collected)
        taxonomy[key] = sorted(collected)
    return taxonomy


NSF_AWARDS_URL = "https://api.nsf.gov/services/v1/awards.json"
NSF_PAGES = 12


def fetch_nsf_programs() -> list[str]:
    """NSF programme names, which are the taxonomy for everything NIH is not.

    The RePORTER harvest is biomedical by construction, so on its own it leaves
    physics, materials, engineering and geoscience thin. Programme names are
    abbreviated and shouty in the API, so they are lowercased and the obvious
    truncations dropped rather than taken as-is.
    """
    import httpx

    collected: set[str] = set()
    with httpx.Client(timeout=90.0) as client:
        for page in range(1, NSF_PAGES + 1):
            params = {"printFields": "fundProgramName", "rpp": 25, "offset": (page - 1) * 25 + 1}
            try:
                resp = client.get(NSF_AWARDS_URL, params=params)
                resp.raise_for_status()
                awards = (resp.json().get("response") or {}).get("award") or []
            except Exception as exc:  # noqa: BLE001
                print(f"  ! NSF awards page {page}: {exc}", file=sys.stderr)
                break
            if not awards:
                break
            for award in awards:
                name = (award.get("fundProgramName") or "").strip().lower()
                # The API truncates long names mid-word; a fragment is not a term.
                if not name or len(name) < _MIN_TERM_CHARS or _BAD_TERM_RE.search(name):
                    continue
                if len(name.split()) > _MAX_TERM_WORDS or name.endswith(("&", "-", ",")):
                    continue
                collected.add(name)
    return sorted(collected)


def fetch_research_terms() -> list[str]:
    """Harvest research vocabulary from NIH RePORTER's public project index.

    Best-effort: a failure here leaves the committed field lexicon in place rather
    than failing the build, because this is a recall improvement and not a
    correctness requirement.
    """
    import httpx

    collected: set[str] = set()
    with httpx.Client(timeout=90.0) as client:
        for year in REPORTER_YEARS:
            for page in range(REPORTER_PAGES_PER_YEAR):
                body = {
                    "criteria": {"fiscal_years": [year]},
                    "include_fields": ["Terms"],
                    "offset": page * REPORTER_PAGE,
                    "limit": REPORTER_PAGE,
                }
                try:
                    resp = client.post(REPORTER_URL, json=body)
                    resp.raise_for_status()
                    results = resp.json().get("results") or []
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! RePORTER {year} page {page}: {exc}", file=sys.stderr)
                    break
                if not results:
                    break
                for item in results:
                    for raw in _TERM_RE.findall(item.get("terms") or ""):
                        term = raw.strip().lower()
                        if len(term) < _MIN_TERM_CHARS or len(term.split()) > _MAX_TERM_WORDS:
                            continue
                        if _BAD_TERM_RE.search(term):
                            continue
                        collected.add(term)
    return sorted(collected)


def build(offline: bool) -> dict[str, Any]:
    field_terms = read_field_terms()

    if offline:
        existing = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8")) if TAXONOMY_PATH.is_file() else {}
        taxonomy = existing.get("taxonomy", {})
        research_terms = existing.get("research_terms", [])
        fetched_at = existing.get("fetched_at")
        source = existing.get("source")
    else:
        taxonomy = fetch_taxonomy()
        research_terms = sorted(set(fetch_research_terms()) | set(fetch_nsf_programs()))
        fetched_at = datetime.now(timezone.utc).isoformat()
        source = f"{SEARCH2_URL} + {REPORTER_URL}"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "fetched_at": fetched_at,
        "field_terms_sha256": hashlib.sha256(FIELD_TERMS_PATH.read_bytes()).hexdigest(),
        "field_terms": field_terms,
        "research_terms": research_terms,
        "taxonomy": taxonomy,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="rebuild from committed files, no network")
    args = parser.parse_args()

    try:
        doc = build(args.offline)
    except Exception as exc:  # noqa: BLE001 -- a build script should report, not traceback
        print(f"ERROR: vocabulary build failed: {exc}", file=sys.stderr)
        return 1

    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    TAXONOMY_PATH.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    counts = {k: len(v) for k, v in doc["taxonomy"].items()}
    total = len(doc["field_terms"]) + len(doc["research_terms"]) + sum(counts.values())
    print(f"Wrote {TAXONOMY_PATH.relative_to(REPO_ROOT)}")
    print(f"  field terms   : {len(doc['field_terms'])}")
    print(f"  research terms: {len(doc['research_terms'])}")
    for key, n in counts.items():
        print(f"  {key:<20}: {n}")
    print(f"  total entries: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
