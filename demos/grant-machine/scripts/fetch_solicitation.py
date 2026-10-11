"""Phase 4.1: fetch the NSF ATE solicitation, convert to Markdown, verify key
facts by grep, and record provenance in DATA_SOURCES.md.

Usage: uv run python scripts/fetch_solicitation.py
"""

from __future__ import annotations

import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify

REPO_ROOT = Path(__file__).resolve().parents[1]
SOLICITATION_URL = (
    "https://www.nsf.gov/funding/opportunities/ate-advanced-technological-education/nsf24-584/solicitation"
)
SOLICITATION_ID = "NSF 24-584"
HTML_PATH = REPO_ROOT / "data" / "solicitation" / "nsf24-584.html"
MD_PATH = REPO_ROOT / "data" / "solicitation" / "nsf24-584.md"
DATA_SOURCES_PATH = REPO_ROOT / "DATA_SOURCES.md"

# (description, pattern) - all must be found (case-insensitive) in the markdown text
REQUIRED_FACTS: list[tuple[str, str]] = [
    ("Track 2 award cap $1,000,000", r"\$1,000,000"),
    ("Track 2 duration up to three years", r"\bthree years\b"),
    ("Project Description 15-page limit", r"\b15 pages\b"),
    ("Deadline: first Thursday in October", r"first Thursday in October"),
    ("Evaluation Plan subsection required", r"Evaluation Plan"),
    ("Results from Prior NSF Support required", r"Results from Prior NSF Support"),
]


def fetch_html() -> str:
    resp = httpx.get(SOLICITATION_URL, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag_name in ("nav", "header", "footer", "script", "style"):
        for tag in soup.find_all(tag_name):
            tag.decompose()
    main = soup.find("main") or soup.body or soup
    md = markdownify(str(main), heading_style="ATX")
    # Collapse runs of blank lines left behind by stripped nav/footer elements.
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return md


def verify_facts(markdown_text: str) -> list[str]:
    missing = []
    for description, pattern in REQUIRED_FACTS:
        if not re.search(pattern, markdown_text, flags=re.IGNORECASE):
            missing.append(f"{description} (pattern: {pattern!r})")
    return missing


def write_data_sources_entry(html_sha256: str, retrieved_at: str) -> None:
    entry = (
        f"| `data/solicitation/nsf24-584.html`, `.md` | {SOLICITATION_URL} "
        f"| U.S. Government work — public domain | retrieved {retrieved_at} "
        f"| sha256:{html_sha256} |\n"
    )
    header = (
        "# Data Sources\n\n"
        "Every file under `data/` and where it came from.\n\n"
        "| Path | Source | License/Status | Retrieved/Generated | SHA-256 |\n"
        "|---|---|---|---|---|\n"
    )
    if not DATA_SOURCES_PATH.exists():
        DATA_SOURCES_PATH.write_text(header + entry, encoding="utf-8")
        return
    text = DATA_SOURCES_PATH.read_text(encoding="utf-8")
    if SOLICITATION_ID in text and "nsf24-584" in text:
        # Already recorded; leave the file alone (script is idempotent).
        return
    DATA_SOURCES_PATH.write_text(text.rstrip("\n") + "\n" + entry, encoding="utf-8")


def main() -> int:
    HTML_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {SOLICITATION_URL} ...")
    html = fetch_html()
    HTML_PATH.write_text(html, encoding="utf-8")
    html_sha256 = hashlib.sha256(html.encode("utf-8")).hexdigest()
    print(f"Wrote {HTML_PATH} ({len(html)} bytes, sha256={html_sha256})")

    if SOLICITATION_ID not in html and "24-584" not in html:
        print(
            f"ERROR: fetched page does not mention {SOLICITATION_ID}. "
            "HUMAN STEP: the solicitation may have been superseded; verify manually before proceeding.",
            file=sys.stderr,
        )
        return 1

    markdown_text = html_to_markdown(html)
    MD_PATH.write_text(markdown_text, encoding="utf-8")
    print(f"Wrote {MD_PATH} ({len(markdown_text)} chars)")

    missing = verify_facts(markdown_text)
    if missing:
        print("ERROR: the following required facts were not found in the converted text:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        print(
            "HUMAN STEP: the solicitation text may have changed in a way that affects the demo "
            "(track caps, page limits). Stop and review before proceeding.",
            file=sys.stderr,
        )
        return 1

    print("All required facts verified present:")
    for description, _ in REQUIRED_FACTS:
        print(f"  - OK: {description}")

    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_data_sources_entry(html_sha256, retrieved_at)
    print(f"Recorded provenance in {DATA_SOURCES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
