"""Fetch the real solicitation each generated scenario applies to.

The five scenarios after Harrow Creek used to apply to synthetic calls the local
model wrote, all capped at $1,000,000 with one structure -- a large part of why
they read as the same scenario with the names changed. Each now applies to a real,
currently posted federal solicitation, which brings its real caps, durations and
cost rules with it. The institutions and people stay fictional.

Same discipline as scripts/fetch_solicitation.py: fetch, convert, and refuse a page
that does not contain the rules the scenario's planted issues are built on -- a
restructured page returning 200 would otherwise be shipped silently.

Usage:
    python scripts/fetch_scenario_solicitations.py            # all five
    python scripts/fetch_scenario_solicitations.py --only vantcross
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_SOURCES_PATH = REPO_ROOT / "DATA_SOURCES.md"


@dataclass(frozen=True)
class Source:
    stem: str             # seed / scenario directory name
    solicitation_id: str
    url: str
    required: tuple[tuple[str, str], ...]  # (what, regex) -- every one must match


SOURCES = [
    Source(
        "ashgrove", "PA-25-305",
        "https://grants.nih.gov/grants/guide/pa-files/PA-25-305.html",
        (
            ("is the Parent R01 Clinical Trial Required", r"Parent R01 Clinical Trial Required"),
            ("maximum project period", r"maximum project period is 5 years"),
            ("no cost sharing required", r"does not require cost sharing"),
            ("ClinicalTrials.gov registration", r"ClinicalTrials\.gov"),
            ("data and safety monitoring", r"Data and Safety Monitoring"),
            ("NIH keeps 45 CFR 75 indirect-cost thresholds", r"45 CFR 75 continue to apply"),
        ),
    ),
    Source(
        "vantcross", "PA-25-301",
        "https://grants.nih.gov/grants/guide/pa-files/PA-25-301.html",
        (
            ("is the Parent R01 Clinical Trial Not Allowed", r"Parent R01 Clinical Trial Not Allowed"),
            ("maximum project period", r"maximum project period is 5 years"),
            ("resubmissions accepted", r"Resubmission"),
            ("NIH keeps 45 CFR 75 indirect-cost thresholds", r"45 CFR 75 continue to apply"),
        ),
    ),
    Source(
        "meridian", "NSF 26-516",
        "https://www.nsf.gov/funding/opportunities/geo-core-geosciences-core-research-atmospheric-geospace-earth-ocean/nsf26-516/solicitation",
        (
            ("is GEO Core", r"NSF 26-516"),
            ("includes the Ocean Sciences program", r"Ocean Sciences"),
            ("cost sharing prohibited", r"voluntary committed cost sharing is prohibited"),
            ("prepared per the PAPPG", r"PAPPG"),
        ),
    ),
    Source(
        "sablecreek", "NSF 25-523",
        "https://www.nsf.gov/funding/opportunities/e-core-epscor-research-infrastructure-improvement-program-epscor/nsf25-523/solicitation",
        (
            ("is E-CORE", r"NSF 25-523"),
            ("four-year duration", r"duration of four \(4\) years"),
            ("$10 million maximum", r"maximum budget of \$10 million"),
            ("cost sharing prohibited", r"voluntary committed cost sharing is prohibited"),
            ("letters of support not allowed", r"Letters of support are not allowed"),
            ("20-page project description", r"20 pages maximum"),
        ),
    ),
    Source(
        "tierra_sandoval", "NSF 25-514",
        "https://www.nsf.gov/funding/opportunities/s-stem-nsf-scholarships-science-technology-engineering-mathematics/nsf25-514/solicitation",
        (
            ("is S-STEM", r"NSF 25-514"),
            ("Track 2 cap", r"Track 2[^.]{0,80}\$2,000,000"),
            ("60% to scholarships", r"at least 60% of all funds"),
            ("$15,000 scholarship maximum", r"\$15,000 per year for undergraduate"),
            ("external evaluator", r"external evaluator"),
            ("cost sharing prohibited", r"voluntary committed cost sharing is prohibited"),
        ),
    ),
]


def fetch(source: Source, client) -> tuple[str, str]:
    from app.discovery.web import html_to_markdown

    resp = client.get(source.url)
    resp.raise_for_status()
    title, md = html_to_markdown(resp.text)
    if title.strip().lower().startswith("expired"):
        raise RuntimeError(f"{source.url} has expired ({title[:60]!r})")
    missing = [what for what, pattern in source.required if not re.search(pattern, md, re.IGNORECASE)]
    if missing:
        raise RuntimeError(f"{source.solicitation_id} is missing: {', '.join(missing)} -- the page may have changed")
    return title.strip(), md


def write(source: Source, title: str, md: str) -> Path:
    out_dir = REPO_ROOT / "data" / "scenarios" / source.stem / "solicitation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{source.stem}-solicitation.md"
    retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = f"# {title}\n\nSource: {source.url}\n\n{md.strip()}\n"
    front = (
        "---\n"
        f"doc_id: {source.stem}-solicitation\n"
        f"title: {title!r}\n"
        f"solicitation_id: '{source.solicitation_id}'\n"
        "doc_type: md\n"
        "synthetic: false\n"
        f"source_url: {source.url}\n"
        f"retrieved_at: '{retrieved}'\n"
        f"sha256: {hashlib.sha256(body.encode('utf-8')).hexdigest()}\n"
        "license: U.S. Government work — public domain\n"
        "---\n\n"
    )
    path.write_text(front + body, encoding="utf-8")
    _record(source, path, retrieved)
    return path


def _record(source: Source, path: Path, retrieved: str) -> None:
    rel = path.relative_to(REPO_ROOT)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    row = (f"| `{rel}` | {source.url} | U.S. Government work — public domain "
           f"| retrieved {retrieved} | sha256:{sha} |")
    lines = DATA_SOURCES_PATH.read_text(encoding="utf-8").splitlines()
    # One row per file: replace a previous fetch (or the synthetic call it supersedes).
    lines = [ln for ln in lines if f"`{rel}`" not in ln]
    DATA_SOURCES_PATH.write_text("\n".join(lines).rstrip("\n") + "\n" + row + "\n", encoding="utf-8")


def main() -> int:
    import httpx

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=[s.stem for s in SOURCES])
    args = parser.parse_args()

    failures = 0
    headers = {"User-Agent": "Mozilla/5.0 (grant-machine; public-domain solicitation fetch)"}
    with httpx.Client(timeout=120.0, follow_redirects=True, headers=headers) as client:
        for source in SOURCES:
            if args.only and source.stem != args.only:
                continue
            try:
                title, md = fetch(source, client)
            except Exception as exc:  # noqa: BLE001 -- report and continue
                print(f"  ! {source.stem}: {exc}", file=sys.stderr)
                failures += 1
                continue
            path = write(source, title, md)
            print(f"  {source.stem}: {source.solicitation_id} -> {path.relative_to(REPO_ROOT)} "
                  f"({len(md.split()):,} words, {len(source.required)} rules verified)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
