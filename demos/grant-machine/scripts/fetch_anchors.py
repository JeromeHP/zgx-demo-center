"""Real funded awards that set the scale of each generated scenario.

A scenario's institution and people are fictional, but its money should look like
the real thing: every generated scenario used to budget exactly $1,038,450 over three
years whether it was an NIH clinical trial or an NSF scholarship programme. These are
comparable real awards -- same agency, mechanism and field -- pulled from the NIH
RePORTER and NSF Award Search APIs.

Kept: award ids, titles, amounts, dates and programme. Deliberately NOT kept:
recipient institutions and investigator names -- nothing here should be able to leak
a real name into a seed. Written to data/anchors/, outside every vault, so the agent
never indexes it.

Usage:
    python scripts/fetch_anchors.py
    python scripts/fetch_anchors.py --only vantcross
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "anchors"
NIH_URL = "https://api.reporter.nih.gov/v2/projects/search"
NSF_URL = "https://api.nsf.gov/services/v1/awards.json"


def _years(start: str, end: str) -> float:
    def parse(s: str) -> date:
        s = s[:10]
        return date.fromisoformat(s) if "-" in s else datetime.strptime(s, "%m/%d/%Y").date()
    return round((parse(end) - parse(start)).days / 365.25, 1)


def nih(client, text: str, activity: str = "R01") -> list[dict[str, Any]]:
    criteria = {
        "advanced_text_search": {"operator": "and", "search_field": "projecttitle,terms", "search_text": text},
        "activity_codes": [activity],
        "fiscal_years": [2022, 2023, 2024, 2025],
        "award_types": ["1", "2"],  # new and competing renewals: a full first-year budget
    }
    fields = ["ProjectNum", "ProjectTitle", "AwardAmount", "DirectCostAmt", "ProjectStartDate",
              "ProjectEndDate", "AgencyIcAdmin", "FiscalYear"]
    resp = client.post(NIH_URL, json={"criteria": criteria, "include_fields": fields, "limit": 10})
    resp.raise_for_status()
    out = []
    for p in resp.json().get("results", []):
        out.append({
            "award_id": p["project_num"],
            "title": p["project_title"],
            "agency": f"NIH {p['agency_ic_admin']['abbreviation']}",
            "mechanism": activity,
            "annual_total_usd": p["award_amount"],
            "annual_direct_usd": p.get("direct_cost_amt"),
            "start": p["project_start_date"][:10],
            "end": p["project_end_date"][:10],
            "years": _years(p["project_start_date"], p["project_end_date"]),
            "url": f"https://reporter.nih.gov/project-details/{p['project_num']}",
        })
    return out


def nsf(client, keyword: str, program_contains: str) -> list[dict[str, Any]]:
    params = {"keyword": keyword, "rpp": 25, "dateStart": "01/01/2022",
              "printFields": "id,title,estimatedTotalAmt,startDate,expDate,fundProgramName"}
    resp = client.get(NSF_URL, params=params)
    resp.raise_for_status()
    out = []
    for a in (resp.json().get("response") or {}).get("award") or []:
        if program_contains.lower() not in (a.get("fundProgramName") or "").lower():
            continue  # the keyword search is loose; the programme filter is what makes it comparable
        total = int(a.get("estimatedTotalAmt") or 0)
        years = _years(a["startDate"], a["expDate"])
        out.append({
            "award_id": a["id"],
            "title": a["title"],
            "agency": "NSF",
            "program": a.get("fundProgramName"),
            "total_usd": total,
            "annual_total_usd": round(total / years) if years else None,
            "start": a["startDate"],
            "end": a["expDate"],
            "years": years,
            "url": f"https://www.nsf.gov/awardsearch/showAward?AWD_ID={a['id']}",
        })
    return out


ANCHORS: dict[str, tuple[str, Callable[[Any], list[dict[str, Any]]]]] = {
    "ashgrove": ("NIH R01 clinical trials in ICU sedation and respiratory failure",
                 lambda c: nih(c, "sedation intensive care unit trial") + nih(c, "mechanical ventilation weaning trial")
                 + nih(c, "acute respiratory failure trial") + nih(c, "intensive care unit delirium trial")),
    "vantcross": ("NIH R01 on skeletal-muscle mitochondria and insulin resistance",
                  lambda c: nih(c, "mitochondrial skeletal muscle insulin resistance")),
    # Standard research grants only: RAPID and EAGER awards are a fraction of the size
    # and dragged the median to a quarter of a normal Physical Oceanography award.
    "meridian": ("NSF Physical Oceanography research awards",
                 lambda c: [a for a in nsf(c, "estuary", "PHYSICAL OCEANOGRAPHY") + nsf(c, "coastal", "PHYSICAL OCEANOGRAPHY")
                            + nsf(c, "storm surge", "PHYSICAL OCEANOGRAPHY")
                            if not a["title"].upper().startswith(("RAPID", "EAGER", "CONFERENCE", "WORKSHOP"))]),
    # Lead awards only: a collaborative E-CORE is split into per-institution awards,
    # and the small partner pieces are not the scale a system office applies for.
    "sablecreek": ("NSF EPSCoR E-CORE lead awards",
                   lambda c: [a for a in nsf(c, '"RII E-CORE"', "E-CORE") if a["total_usd"] >= 3_000_000]),
    "tierra_sandoval": ("NSF S-STEM awards", lambda c: nsf(c, "computing", "S-STEM") + nsf(c, "scholarships", "S-STEM")
                        + nsf(c, "transfer students", "S-STEM")),
}


def summarise(awards: list[dict[str, Any]]) -> dict[str, Any]:
    annual = [a["annual_total_usd"] for a in awards if a.get("annual_total_usd")]
    years = [a["years"] for a in awards if a.get("years")]
    return {
        "n": len(awards),
        "median_annual_total_usd": int(statistics.median(annual)) if annual else None,
        "median_years": statistics.median(years) if years else None,
    }


def main() -> int:
    import httpx

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=sorted(ANCHORS))
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    failures = 0
    with httpx.Client(timeout=120.0) as client:
        for stem, (what, query) in ANCHORS.items():
            if args.only and stem != args.only:
                continue
            try:
                awards = query(client)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {stem}: {exc}", file=sys.stderr)
                failures += 1
                continue
            # De-duplicate: two NIH queries can return the same project.
            awards = list({a["award_id"]: a for a in awards}.values())
            if not awards:
                print(f"  ! {stem}: no comparable awards found", file=sys.stderr)
                failures += 1
                continue
            doc = {
                "scenario": stem,
                "comparable": what,
                "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "license": "U.S. Government work — public domain (award metadata)",
                "note": "Scale reference for the scenario seed. Recipient and investigator names are not stored.",
                "summary": summarise(awards),
                "awards": awards,
            }
            (OUT_DIR / f"{stem}.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
            s = doc["summary"]
            print(f"  {stem}: {s['n']} awards, median ${s['median_annual_total_usd']:,}/yr over {s['median_years']} yrs")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
