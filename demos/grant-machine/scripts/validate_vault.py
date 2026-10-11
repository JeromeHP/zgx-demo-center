"""Phase 4.5: validate the synthetic vault. Exits non-zero on any failure.

Usage: uv run python scripts/validate_vault.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `from app...` when run as a script

from app.checks.budget import BudgetRules, check_budget  # noqa: E402
from app.ingest.loaders import LoaderError, load  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
VAULT_DIR = REPO_ROOT / "data" / "vault"
SEED_OVERRIDE: Path | None = None
MANIFEST_PATH = VAULT_DIR / "manifest.json"
PLANTED_ISSUES_PATH = VAULT_DIR / "_planted_issues.json"
SEED_PATH = REPO_ROOT / "data" / "seeds" / "harrow_creek.yaml"
DENYLIST_PATH = REPO_ROOT / "data" / "seeds" / "denylist.txt"

REQUIRED_FRONT_MATTER_KEYS = {"doc_id", "title", "doc_type", "date", "synthetic", "generator", "seed_sha256"}
_VAULT_EXTENSIONS = (".md", ".html", ".htm", ".pdf", ".csv")
_NON_INDEXED_FILES = {"manifest.json", "_planted_issues.json"}


class ValidationErrors(list[str]):
    def add(self, message: str) -> None:
        self.append(message)


def _vault_content_files() -> list[Path]:
    return sorted(p for p in VAULT_DIR.iterdir() if p.suffix.lower() in _VAULT_EXTENSIONS)


def check_manifest_consistency(errors: ValidationErrors) -> None:
    if not MANIFEST_PATH.is_file():
        errors.add(f"missing {MANIFEST_PATH}")
        return
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_ids = {d["doc_id"] for d in manifest.get("documents", [])}
    file_ids = {p.stem for p in _vault_content_files()}

    for doc_id in sorted(manifest_ids - file_ids):
        errors.add(f"manifest lists {doc_id!r} but no vault file exists for it")
    for doc_id in sorted(file_ids - manifest_ids):
        errors.add(f"vault file {doc_id!r} exists but is not listed in manifest.json")


def check_front_matter(errors: ValidationErrors) -> None:
    for path in _vault_content_files():
        try:
            doc = load(path, doc_id=path.stem)
        except LoaderError as exc:
            errors.add(f"{path.name}: failed to load: {exc}")
            continue
        missing = REQUIRED_FRONT_MATTER_KEYS - set(doc.front_matter)
        if missing:
            errors.add(f"{path.name}: front matter missing keys: {sorted(missing)}")
        elif doc.front_matter.get("doc_id") != path.stem:
            errors.add(f"{path.name}: front matter doc_id {doc.front_matter.get('doc_id')!r} != filename {path.stem!r}")


def _doc_text(doc_id: str) -> str:
    for ext in _VAULT_EXTENSIONS:
        candidate = VAULT_DIR / f"{doc_id}{ext}"
        if candidate.is_file():
            doc = load(candidate, doc_id=doc_id)
            return doc.text
    return ""


def _doc_front_matter(doc_id: str) -> dict[str, Any]:
    for ext in _VAULT_EXTENSIONS:
        candidate = VAULT_DIR / f"{doc_id}{ext}"
        if candidate.is_file():
            return load(candidate, doc_id=doc_id).front_matter
    return {}


def check_seed_declared_facts(errors: ValidationErrors, seed: dict[str, Any]) -> None:
    """Facts a seed says must appear in named documents.

    Every planted issue is a fact that has to stay consistent across several
    generated documents, and nothing but a check like this keeps that true. Scenarios
    declare their own, so adding one needs no code here.
    """
    for doc_id, needles in (seed.get("key_facts") or {}).items():
        text = _doc_text(doc_id)
        if not text:
            errors.add(f"key facts: {doc_id} not found or empty, cannot verify")
            continue
        for needle in needles:
            if str(needle) not in text:
                errors.add(f"key facts: {doc_id} does not contain expected fact {str(needle)!r}")

    # A letter's date is part of its evidence: generated scenarios once dated every
    # letter "today", and the agent rightly flagged a partner letter signed on the day
    # of submission -- an issue nobody planted.
    for collab in seed.get("collaborators") or []:
        fm = _doc_front_matter(collab["letter_doc"])
        if fm and str(fm.get("date")) != str(collab["letter_date"]):
            errors.add(f"{collab['letter_doc']}: front matter date {fm.get('date')!r} != seed letter_date {collab['letter_date']!r}")

    for doc_id, forbidden in (seed.get("must_not_contain") or {}).items():
        text = _doc_text(doc_id)
        for needle in forbidden:
            # Whole-word, not substring. A short project acronym banned as a substring
            # fires inside ordinary words -- "SSI" matched "commissioning" and
            # "transmission" -- and a scenario cannot be written around that.
            pattern = r"\b" + re.escape(str(needle)) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                errors.add(f"{doc_id}: must not mention {str(needle)!r} (planted issue would be defeated)")


def check_scenario_isolation(errors: ValidationErrors, seed: dict[str, Any]) -> None:
    """No scenario may contain another scenario's proper nouns.

    Generated documents pick up whatever the generator hardcodes. The research
    scenario's first budget carried Harrow Creek's evaluator and project name,
    because both were literals in generate_vault.py -- a scenario quietly
    contaminated by the one it was modelled on is worse than an obviously broken one,
    because it reads fine.
    """
    seeds_dir = REPO_ROOT / "data" / "seeds"
    mine = {str(seed.get("institution", {}).get("short_name", "")).lower(),
            str(seed.get("institution", {}).get("name", "")).lower()}
    foreign: set[str] = set()
    for other_path in sorted(seeds_dir.glob("*.yaml")):
        other = yaml.safe_load(other_path.read_text(encoding="utf-8")) or {}
        inst = other.get("institution") or {}
        names = {str(inst.get("short_name", "")).lower(), str(inst.get("name", "")).lower()}
        if names & mine:
            continue  # this is us
        foreign |= {n for n in names if len(n) > 3}
        for key in ("people", "new_project"):
            block = other.get(key) or {}
            if key == "new_project":
                sn = str(block.get("short_name", "")).lower()
                if len(sn) > 3:
                    foreign.add(sn)
            else:
                for person in block.values():
                    if isinstance(person, dict):
                        for field in ("name", "org"):
                            v = str(person.get(field, "")).lower()
                            if len(v) > 3:
                                foreign.add(v)

    for path in _vault_content_files():
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for term in sorted(foreign):
            if term in text:
                errors.add(f"{path.name}: contains {term!r}, which belongs to a different scenario")


def check_key_facts(errors: ValidationErrors, seed: dict[str, Any]) -> None:
    # The block below is specific to the workforce scenario's documents. A scenario
    # without employer letters declares its facts in the seed instead.
    if not seed.get("employers"):
        check_seed_declared_facts(errors, seed)
        return

    prior = seed["prior_award"]
    new_project = seed["new_project"]
    rates = seed["rates"]

    def must_contain(doc_id: str, needles: list[str], label: str) -> None:
        text = _doc_text(doc_id)
        if not text:
            errors.add(f"{label}: {doc_id} not found or empty, cannot verify")
            return
        for needle in needles:
            if needle not in text:
                errors.add(f"{label}: {doc_id} does not contain expected fact {needle!r}")

    must_contain(
        "prior-annual-report-2025",
        [str(prior["outcomes"]["completers_actual"]), str(prior["outcomes"]["completers_target"])],
        "completers actual/target",
    )
    must_contain(
        "prior-evaluation-report-2025",
        [str(prior["outcomes"]["completers_actual"]), str(prior["outcomes"]["completers_target"])],
        "completers actual/target (evaluation report)",
    )
    must_contain("rate-agreement-fy2026", [f"{rates['fringe_rate_agreement_pct']}"], "agreement fringe rate")
    must_contain("rate-agreement-fy2026", [f"{rates['indirect_rate_pct']}"], "agreement indirect rate")
    must_contain("budget-justification-draft", [f"{new_project['fringe_rate_budget_pct']}"], "budget justification fringe rate")

    fm = _doc_front_matter("letter-northgate-2021")
    northgate = next(e for e in seed["employers"] if e["letter_doc"] == "letter-northgate-2021")
    if fm.get("date") != northgate["letter_date"]:
        errors.add(f"letter-northgate-2021: front matter date {fm.get('date')!r} != seed {northgate['letter_date']!r}")
    text = _doc_text("letter-northgate-2021")
    if re.search(r"\bSATP\b|Smart Automation Technician Pathway", text, re.IGNORECASE):
        errors.add("letter-northgate-2021: must not reference the new SATP project (planted stale-letter issue)")

    for emp in seed["employers"]:
        if emp["letter_doc"] == "letter-northgate-2021":
            continue
        fm = _doc_front_matter(emp["letter_doc"])
        if fm.get("date") != emp["letter_date"]:
            errors.add(f"{emp['letter_doc']}: front matter date {fm.get('date')!r} != seed {emp['letter_date']!r}")


def _scenario_budget_rules() -> dict[str, Any]:
    """requirements.json "budget_rules" for the vault being validated, if curated.

    Harrow Creek's live at data/solicitation/ and declare none, so it keeps the
    historical $1,000,000 / three-year checks.
    """
    for path in (VAULT_DIR / "solicitation" / "requirements.json",):
        if path.is_file():
            return (json.loads(path.read_text(encoding="utf-8")) or {}).get("budget_rules") or {}
    return {}


def _run_check_budget(seed: dict[str, Any]):
    rules = _scenario_budget_rules()
    if not rules:
        return check_budget(VAULT_DIR, cap_usd=1_000_000, track_years=seed["new_project"]["years"])
    value = lambda k: rules[k].get("value") if isinstance(rules.get(k), dict) else rules.get(k)  # noqa: E731
    return check_budget(
        VAULT_DIR,
        cap_usd=value("cap_usd"),
        track_years=value("max_years") or seed["new_project"]["years"],
        rules=BudgetRules.from_dict(rules),
    )


def check_budget_arithmetic(errors: ValidationErrors, seed: dict[str, Any]) -> None:
    budget_id = (seed.get("csv_documents") or {}).get("budget", "budget-satp-draft")
    if not (VAULT_DIR / f"{budget_id}.csv").is_file():
        errors.add(f"{budget_id}.csv not found")
        return
    result = _run_check_budget(seed)
    if seed.get("budget"):
        # Declared budgets are computed, not solved to a target: the CSV must be exactly
        # what the seed's lines produce.
        from app.budget_builder import build as build_declared

        expected = build_declared(seed)[2]["grand_total_usd"]
        if result.total_usd != expected:
            errors.add(f"budget grand total {result.total_usd} != {expected} computed from the seed's budget lines")
    elif result.total_usd != seed["new_project"]["budget_total_usd"]:
        errors.add(f"budget grand total {result.total_usd} != seed budget_total_usd {seed['new_project']['budget_total_usd']}")
    if "YEAR_SUM_MISMATCH" in {i.code for i in result.issues}:
        errors.add("budget CSV year sums do not add up to the grand total")


def check_planted_issues(errors: ValidationErrors, seed: dict[str, Any]) -> None:
    if not PLANTED_ISSUES_PATH.is_file():
        errors.add(f"missing {PLANTED_ISSUES_PATH}")
        return
    planted = json.loads(PLANTED_ISSUES_PATH.read_text(encoding="utf-8"))
    for issue in planted:
        for doc_id in issue["evidence"]:
            if not any((VAULT_DIR / f"{doc_id}{ext}").is_file() for ext in _VAULT_EXTENSIONS):
                errors.add(f"planted issue {issue['id']}: evidence doc {doc_id!r} does not exist in the vault")

    deterministic_codes = {i["code"] for i in planted if i["detection"] == "deterministic"}
    actual_codes = {i.code for i in _run_check_budget(seed).issues}
    missing = deterministic_codes - actual_codes
    if missing:
        errors.add(f"deterministic planted issues not detected by check_budget: {sorted(missing)}")
    # And the reverse: a budget error nobody planted is a bug in the seed, and would
    # show up on stage as an issue the presenter cannot explain.
    unplanted = actual_codes - deterministic_codes
    if unplanted:
        errors.add(f"check_budget finds issues that were not planted: {sorted(unplanted)}")


def check_denylist(errors: ValidationErrors) -> None:
    if not DENYLIST_PATH.is_file():
        errors.add(f"missing {DENYLIST_PATH}")
        return
    terms = [
        line.strip()
        for line in DENYLIST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for path in _vault_content_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for term in terms:
            if re.search(r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE):
                errors.add(f"{path.name}: contains denylisted string {term!r}")


def main() -> int:
    global VAULT_DIR, MANIFEST_PATH, PLANTED_ISSUES_PATH, SEED_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="scenario seed (default: harrow_creek.yaml)")
    parser.add_argument("--vault", help="vault directory to validate (default: data/vault)")
    args = parser.parse_args()

    if args.seed or args.vault:
        seed_path = Path(args.seed) if args.seed else SEED_PATH
        SEED_PATH = seed_path if seed_path.is_absolute() else (REPO_ROOT / seed_path)
        vault = Path(args.vault) if args.vault else (REPO_ROOT / "data" / "scenarios" / SEED_PATH.stem)
        VAULT_DIR = vault if vault.is_absolute() else (REPO_ROOT / vault)
        MANIFEST_PATH = VAULT_DIR / "manifest.json"
        PLANTED_ISSUES_PATH = VAULT_DIR / "_planted_issues.json"
        print(f"Validating {VAULT_DIR.relative_to(REPO_ROOT)} against {SEED_PATH.name}")

    seed = yaml.safe_load(SEED_PATH.read_text(encoding="utf-8"))
    errors = ValidationErrors()

    check_manifest_consistency(errors)
    check_front_matter(errors)
    check_key_facts(errors, seed)
    check_budget_arithmetic(errors, seed)
    check_planted_issues(errors, seed)
    check_denylist(errors)
    check_scenario_isolation(errors, seed)

    if errors:
        print(f"VALIDATION FAILED ({len(errors)} issue(s)):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("Vault validation OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
