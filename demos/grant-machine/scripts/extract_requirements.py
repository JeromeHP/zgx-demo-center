"""Phase 4.2: extract candidate requirements from the solicitation using the
local model (extractor profile), validate every source_quote as an exact
substring, deduplicate, and write data/solicitation/requirements.draft.json.

Claude Code then hand-reviews the draft against the source text and writes
the final data/solicitation/requirements.json (setting human_reviewed: true
only on entries actually verified).

Usage: uv run python scripts/extract_requirements.py
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `from app...` when run as a script

from app.llm_client import LLMClient  # noqa: E402
from app.settings import REPO_ROOT, load_settings  # noqa: E402

MD_PATH = REPO_ROOT / "data" / "solicitation" / "nsf24-584.md"
HTML_PATH = REPO_ROOT / "data" / "solicitation" / "nsf24-584.html"
DRAFT_PATH = REPO_ROOT / "data" / "solicitation" / "requirements.draft.json"

# The extraction logic now lives in app/ingest/requirements.py so the runtime path
# (Explore/Intake, which extracts an uploaded solicitation) and this build-time script
# share one implementation instead of two that drift. This script keeps the parts that
# are specific to producing the committed NSF 24-584 draft: the known track caps, the
# source URL, and the hand-review workflow around it.
from app.ingest.requirements import (  # noqa: E402
    CATEGORIES,
    dedupe,
    extract_from_section,
    split_sections,
    validate,
)


def main() -> int:
    md_text = MD_PATH.read_text(encoding="utf-8")
    sections = split_sections(md_text)
    print(f"{len(sections)} sections to scan")

    settings = load_settings()
    # This is a mechanical extract-and-quote task, not one that benefits much from
    # deep reasoning; thinking mode was taking 2-6+ minutes per section (some
    # sections generated 4000-5700+ reasoning tokens) for marginal accuracy gain,
    # especially since every candidate is validated deterministically afterward
    # and hand-reviewed by Claude Code before becoming the final requirements.json.
    settings.llm.profiles["extractor"].enable_thinking = False
    settings.llm.profiles["extractor"].max_tokens = 4096
    client = LLMClient(settings)

    candidates: list[dict[str, Any]] = []
    discarded_bad_quote = 0
    for heading, body in sections:
        kept, bad = validate(extract_from_section(client, heading, body), body, heading)
        candidates.extend(kept)
        discarded_bad_quote += bad
        print(f"  {heading!r}: {len(kept)} candidates")

    print(f"Total candidates: {len(candidates)}, discarded (bad source_quote/category/text): {discarded_bad_quote}")
    deduped = dedupe(candidates)
    print(f"After dedup: {len(deduped)}")

    for i, c in enumerate(deduped, start=1):
        c["id"] = f"R-{i:03d}"
        c.setdefault("applies_to_tracks", ["track_2"])
        c.setdefault("section_link", None)
        c.setdefault("verification", "agent")
        c["human_reviewed"] = False

    html_sha256 = hashlib.sha256(HTML_PATH.read_bytes()).hexdigest()
    draft = {
        "solicitation_id": "NSF 24-584",
        "program": "Advanced Technological Education (ATE)",
        "source_url": "https://www.nsf.gov/funding/opportunities/ate-advanced-technological-education/nsf24-584/solicitation",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": html_sha256,
        # Track caps are entered from the verified solicitation text (Phase 4.1), not model extraction.
        "tracks": {"track_2": {"name": "Projects", "max_total_usd": 1000000, "max_years": 3}},
        "deadline_rule": "First Thursday in October, annually",
        "demo_deadline": "2026-10-01",
        "requirements": deduped,
    }
    DRAFT_PATH.write_text(json.dumps(draft, indent=2), encoding="utf-8")
    print(f"Wrote {DRAFT_PATH} ({len(deduped)} requirements)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
