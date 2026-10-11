"""Give each scenario its own solicitation and requirements.

Current path: `--real`, which extracts from the real call fetched by
scripts/fetch_scenario_solicitations.py. The synthetic path below is kept for scenarios
with no suitable posted call; none of the shipped ones use it any more.

Story mode was hardwired: `data/vault/` for the corpus and NSF 24-584's curated
requirements, so the other scenarios could not be selected at all. A scenario picker
needs BOTH halves — an institution has a vault, but a run also needs something to be
applying to, and drafting a clinical trial against a technician-education solicitation
is incoherent.

So each scenario gets a synthetic solicitation matched to it, written from its seed,
and requirements extracted from that at build time and committed. Runtime stays
instant; the real solicitations we ship stay where they earn their keep, as Explore and
Grade samples where "nobody configured this for NIH" is the point.

The solicitation is SYNTHETIC and says so in its own front matter. The honesty rule
this project holds to is that invented institutions are labelled as invented — the
same applies here.

Real calls (the current path): scripts/fetch_scenario_solicitations.py fetches a real,
posted solicitation per scenario; `--real` skips writing a synthetic one and extracts
requirements from that instead. The call's money rules -- cap, duration, salary cap,
subaward MTDC limit, cost-share rule -- are hand-curated with quotes in
`solicitation/budget_rules.yaml` and merged into requirements.json, where the agent's
budget check and scripts/validate_vault.py both read them.

Usage:
    python scripts/build_scenario_solicitation.py --seed data/seeds/vantcross.yaml --real
    python scripts/build_scenario_solicitation.py --all --real
    python scripts/build_scenario_solicitation.py --seed data/seeds/vantcross.yaml   # synthetic (legacy)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingest.requirements import extract_requirements, propose_sections  # noqa: E402
from app.llm_client import LLMClient  # noqa: E402
from app.settings import REPO_ROOT, load_settings  # noqa: E402

SOLICITATION_PROMPT = """Write a funding opportunity announcement — the call for proposals itself,
not an application to it.

Programme: {program}
Funder: {funder}
Award ceiling: ${cap:,} total over {years} years
Deadline: {deadline}

It should read like a real federal announcement and must contain these sections, each with
concrete, checkable requirements a proposer must satisfy:

1. Programme Description — what the programme funds and its goals
2. Award Information — ceiling, duration, number of awards expected
3. Eligibility Information — who may apply, institutional requirements, PI limits
4. Proposal Preparation Instructions — required sections, page limits, formatting, required
   supplementary documents
5. Budget Requirements — allowable and unallowable costs, indirect cost treatment, cost share
6. Merit Review Criteria — the criteria reviewers will score against
7. Reporting Requirements — what an awardee must report and when

Write requirements as explicit obligations ("Proposals must...", "The budget may not exceed...")
rather than description, because they will be extracted and checked one by one. Be specific
about numbers: page limits, award ceiling, duration, deadlines.

Return Markdown only, starting with a level-1 heading. Around {words} words."""


def _strip_front_matter(text: str) -> tuple[dict[str, Any], str]:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return yaml.safe_load(text[4:end]) or {}, text[end + 5:]
    return {}, text


def build_real(seed_path: Path, settings, client: LLMClient) -> dict[str, Any]:
    """Requirements from the real call fetch_scenario_solicitations.py saved."""
    seed = yaml.safe_load(seed_path.read_text(encoding="utf-8"))
    stem = seed_path.stem
    out_dir = REPO_ROOT / "data" / "scenarios" / stem / "solicitation"
    sol_path = out_dir / f"{stem}-solicitation.md"
    if not sol_path.is_file():
        raise FileNotFoundError(f"{sol_path} -- run scripts/fetch_scenario_solicitations.py first")
    front, body = _strip_front_matter(sol_path.read_text(encoding="utf-8"))
    if front.get("synthetic", True):
        raise ValueError(f"{sol_path} is not a fetched real solicitation")

    rules_path = out_dir / "budget_rules.yaml"
    rules = yaml.safe_load(rules_path.read_text(encoding="utf-8")) if rules_path.is_file() else {}
    if not rules:
        raise FileNotFoundError(f"{rules_path} -- curate the call's money rules (with quotes) before building")
    value = lambda k: (rules.get(k) or {}).get("value") if isinstance(rules.get(k), dict) else rules.get(k)  # noqa: E731
    cap, years = value("cap_usd"), value("max_years") or seed["new_project"]["years"]

    extractor = settings.llm.profiles.get("extractor")
    if extractor is not None:
        extractor.enable_thinking = False
    sol = seed.get("solicitation") or {}
    print(f"  extracting requirements from {sol_path.relative_to(REPO_ROOT)} ...")
    doc = extract_requirements(
        body, client,
        model=settings.llm.model,
        solicitation_id=front.get("solicitation_id", sol.get("id", "")),
        program=sol.get("program") or front.get("title", ""),
        source_url=front.get("source_url", ""),
        tracks={"track_2": {"name": sol.get("track", "Award"), "max_total_usd": cap, "max_years": years}},
        cache_dir=REPO_ROOT / "runtime" / "cache" / "requirements",
    )
    doc["demo_deadline"] = seed["demo_context"]["deadline"]
    doc["synthetic"] = False
    doc["budget_rules"] = rules
    # A call's section structure can be curated too. The model's proposal for the NIH
    # clinical-trial NOFO was "Table of Page Limits" and two form pages -- things nobody
    # drafts -- so a sections.yaml beside the rules takes precedence when present.
    curated = out_dir / "sections.yaml"
    sections = (yaml.safe_load(curated.read_text(encoding="utf-8")) if curated.is_file()
                else propose_sections(client, doc, strict=True))
    doc["sections"] = sections
    print("  sections: " + ", ".join(sec["id"] for sec in sections))
    req_path = out_dir / "requirements.json"
    req_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    x = doc["extraction"]
    print(f"  wrote {req_path.relative_to(REPO_ROOT)}: {len(doc['requirements'])} requirements "
          f"from {x['sections_scanned']} sections ({x['discarded_unverifiable_quote']} discarded)")
    return doc


def build(seed_path: Path, settings, client: LLMClient) -> dict[str, Any]:
    seed = yaml.safe_load(seed_path.read_text(encoding="utf-8"))
    stem = seed_path.stem
    out_dir = REPO_ROOT / "data" / "scenarios" / stem / "solicitation"
    out_dir.mkdir(parents=True, exist_ok=True)

    np = seed["new_project"]
    prior = seed.get("prior_award", {})
    program = seed.get("solicitation_program") or f"{np['title']} Program"
    funder = seed.get("solicitation_funder") or prior.get("funder", "Federal funding agency")
    sol_id = seed.get("solicitation_id") or f"{stem.upper().replace('_','-')}-26-001"

    prompt = SOLICITATION_PROMPT.format(
        program=program, funder=funder.replace(" (fictional prior award for demo purposes)", "")
                                     .replace(" (fictional prior award)", ""),
        cap=1_000_000, years=np["years"], deadline=seed["demo_context"]["deadline"], words=1800,
    )
    print(f"  writing solicitation for {stem} ...")
    result = client.chat("drafter", [{"role": "user", "content": prompt}])
    body = (result.content or "").strip()

    front = (
        "---\n"
        f"doc_id: {stem}-solicitation\n"
        f"title: {program}\n"
        f"solicitation_id: '{sol_id}'\n"
        "doc_type: md\n"
        "synthetic: true\n"
        "note: >-\n"
        "  Synthetic solicitation written for this scenario so Story mode has something\n"
        "  coherent to apply to. The real solicitations this demo ships are in\n"
        "  data/samples/ and are labelled as real.\n"
        "---\n\n"
    )
    sol_path = out_dir / f"{stem}-solicitation.md"
    sol_path.write_text(front + body + "\n", encoding="utf-8")
    print(f"  wrote {sol_path.relative_to(REPO_ROOT)} ({len(body.split())} words)")

    extractor = settings.llm.profiles.get("extractor")
    if extractor is not None:
        extractor.enable_thinking = False

    doc = extract_requirements(
        body, client,
        model=settings.llm.model,
        solicitation_id=sol_id,
        program=program,
        source_url="",
        tracks={"track_2": {"name": "Projects", "max_total_usd": 1_000_000, "max_years": np["years"]}},
        cache_dir=REPO_ROOT / "runtime" / "cache" / "requirements",
    )
    doc["demo_deadline"] = seed["demo_context"]["deadline"]
    doc["synthetic"] = True

    # Story mode used to draft the three NSF ATE section names out of
    # config/default.yaml whatever the scenario was, so an academic medical centre
    # applying to a critical-care programme wrote a "Results from Prior NSF Support"
    # section. Explore already derives sections from the uploaded solicitation; bake
    # the same derivation in at build time so Story gets it without a per-run LLM call.
    sections = propose_sections(client, doc, strict=True)
    doc["sections"] = sections
    print("  sections: " + ", ".join(sec["id"] for sec in sections))

    req_path = out_dir / "requirements.json"
    req_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    x = doc["extraction"]
    print(f"  wrote {req_path.relative_to(REPO_ROOT)}: {len(doc['requirements'])} requirements "
          f"from {x['sections_scanned']} sections ({x['discarded_unverifiable_quote']} discarded)")
    return doc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="a single scenario seed")
    parser.add_argument("--all", action="store_true", help="every seed except harrow_creek")
    parser.add_argument("--real", action="store_true",
                        help="extract from the fetched real solicitation instead of writing a synthetic one")
    args = parser.parse_args()

    seeds = []
    if args.all:
        # Harrow Creek keeps the real, hand-curated NSF 24-584 requirements it was built
        # around; generating a synthetic replacement would be a downgrade.
        seeds = [p for p in sorted((REPO_ROOT / "data" / "seeds").glob("*.yaml")) if p.stem != "harrow_creek"]
    elif args.seed:
        seeds = [Path(args.seed)]
    else:
        parser.error("pass --seed or --all")

    settings = load_settings()
    client = LLMClient(settings)
    for seed_path in seeds:
        print(f"### {seed_path.stem}")
        try:
            (build_real if args.real else build)(seed_path, settings, client)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {seed_path.stem}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
