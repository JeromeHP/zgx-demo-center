"""Fetch the real, public-domain funder documents the samples use.

Every scenario this demo invents is fictional, but everything it reads from the
outside world should be genuinely real: solicitations, opportunity records and
funded-award abstracts are U.S. Government works in the public domain.

Follows the provenance discipline scripts/fetch_solicitation.py established -- fetch,
convert, verify by grep that the document is what we think it is, and record source
URL, retrieval date and sha256 so DATA_SOURCES.md can account for every byte under
data/.

Usage:
    python scripts/fetch_samples.py            # fetch all
    python scripts/fetch_samples.py --only nih-nofo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "samples" / "files"
PROVENANCE_PATH = REPO_ROOT / "data" / "samples" / "provenance.json"

GRANTS_DETAIL_URL = "https://api.grants.gov/v1/api/fetchOpportunity"
NSF_AWARDS_URL = "https://api.nsf.gov/services/v1/awards.json"


def _client():
    import httpx

    return httpx.Client(timeout=120.0, follow_redirects=True)


def fetch_nih_nofo(client) -> tuple[str, str, dict[str, Any]]:
    """A current NIH notice of funding opportunity, converted to Markdown."""
    from app.discovery.web import html_to_markdown

    url = "https://grants.nih.gov/grants/guide/pa-files/PA-25-302.html"
    resp = client.get(url)
    resp.raise_for_status()
    title, md = html_to_markdown(resp.text)

    # NIH serves expired announcements at the same URL with an "Expired" title prefix,
    # and the first document fetched here was one. Shipping a dead announcement as a
    # sample invites somebody to present it as live, so refuse it rather than label it.
    if title.strip().lower().startswith("expired"):
        raise RuntimeError(f"{url} has expired ({title[:60]!r}); pick a current announcement")

    body = f"# {title}\n\nSource: {url}\n\n{md}\n"
    # If NIH restructures, a silent 200 of the wrong page would be worse than a 404.
    for needed in ("Eligibility", "Application", "Review"):
        if needed.lower() not in body.lower():
            raise RuntimeError(f"NIH NOFO does not look like a NOFO: missing {needed!r}")
    return "nih-nofo.md", body, {"source_url": url, "kind": "solicitation"}


def fetch_grants_gov_opportunity(client) -> tuple[str, str, dict[str, Any]]:
    """A real posted opportunity record, as a readable document."""
    import html as html_mod

    from app.discovery.web import html_to_markdown

    opportunity_id = "363657"  # EPA water infrastructure workforce development
    resp = client.post(GRANTS_DETAIL_URL, json={"opportunityId": opportunity_id})
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("errorcode") not in (0, None):
        raise RuntimeError(f"grants.gov: {payload.get('msg')}")
    data = payload.get("data") or {}
    syn = data.get("synopsis") or {}
    _, desc = html_to_markdown(html_mod.unescape(syn.get("synopsisDesc") or ""))
    title = html_mod.unescape(data.get("opportunityTitle") or "")
    number = data.get("opportunityNumber") or ""
    url = f"https://grants.gov/search-results-detail/{opportunity_id}"

    body = (
        f"# {title} ({number})\n\n"
        f"Agency: {html_mod.unescape(syn.get('agencyName') or '')}\n\n"
        f"Closing date: {syn.get('responseDate') or syn.get('responseDateDesc') or 'see announcement'}\n\n"
        f"Source: {url}\n\n{desc}\n"
    )
    if len(desc) < 400:
        raise RuntimeError(f"opportunity {opportunity_id} synopsis is too thin to be a useful sample")
    return "grants-gov-opportunity.md", body, {"source_url": url, "kind": "solicitation"}


def fetch_nsf_award_abstract(client) -> tuple[str, str, dict[str, Any]]:
    """A real funded NSF award abstract, as something to grade.

    It was funded, so it should score well and has no planted flaws. That is the
    point of shipping it next to the deliberately broken one: it shows the grader is
    reading rather than reciting answers somebody planted for it.
    """
    params = {
        "keyword": "advanced+technological+education",
        "printFields": "id,title,abstractText,awardeeName,fundsObligatedAmt,startDate",
        "rpp": 25,
    }
    resp = client.get(NSF_AWARDS_URL, params=params)
    resp.raise_for_status()
    awards = (resp.json().get("response") or {}).get("award") or []
    award = next((a for a in awards if len(a.get("abstractText") or "") > 2500), None)
    if award is None:
        raise RuntimeError("no NSF award with a substantial abstract was returned")

    url = f"https://www.nsf.gov/awardsearch/showAward?AWD_ID={award['id']}"
    body = (
        f"# {award.get('title', '').strip()}\n\n"
        f"A real, funded National Science Foundation award. Public domain (U.S. Government work).\n\n"
        f"Award: {award['id']}  \nRecipient: {award.get('awardeeName', '')}  \n"
        f"Start: {award.get('startDate', '')}  \nObligated: ${award.get('fundsObligatedAmt', '')}  \n"
        f"Source: {url}\n\n## Project Abstract\n\n{award.get('abstractText', '').strip()}\n"
    )
    return "nsf-funded-abstract.md", body, {"source_url": url, "kind": "proposal"}


FETCHERS: dict[str, Callable] = {
    "nih-nofo": fetch_nih_nofo,
    "grants-gov-opportunity": fetch_grants_gov_opportunity,
    "nsf-award-abstract": fetch_nsf_award_abstract,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=sorted(FETCHERS), help="fetch a single sample")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, Any] = {}
    if PROVENANCE_PATH.is_file():
        try:
            provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            provenance = {}

    names = [args.only] if args.only else sorted(FETCHERS)
    failures = 0
    with _client() as client:
        for name in names:
            try:
                filename, body, meta = FETCHERS[name](client)
            except Exception as exc:  # noqa: BLE001 -- report and continue
                print(f"  ! {name}: {exc}", file=sys.stderr)
                failures += 1
                continue
            path = OUT_DIR / filename
            path.write_text(body, encoding="utf-8")
            provenance[filename] = {
                **meta,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "license": "U.S. Government work — public domain",
            }
            print(f"  {filename}: {path.stat().st_size:,} bytes from {meta['source_url']}")

    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {PROVENANCE_PATH.relative_to(REPO_ROOT)}")
    return 1 if failures and not args.only else 0


if __name__ == "__main__":
    raise SystemExit(main())
