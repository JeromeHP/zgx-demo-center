"""Deterministic budget validation (Phase 5.3). No LLM involved."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.ingest.loaders import LoaderError, load

_MTDC_EXCLUDED = {"senior_personnel", "other_personnel", "fringe", "equipment", "participant_support", "indirect",
                  "subaward", "cost_share"}
_PERSONNEL_CATEGORIES = {"senior_personnel", "other_personnel"}
_INDIRECT_EXCLUDED_CATEGORIES = {"equipment", "participant_support"}

_FRINGE_RATE_RE = re.compile(r"fringe[^%\n]{0,60}?(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


@dataclass
class BudgetIssue:
    code: str
    severity: str  # "high" | "medium" | "low"
    detail: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class BudgetRules:
    """What a scenario's solicitation says about money, beyond its cap and duration.

    Every field defaults to "no such rule", so a scenario that declares none -- the
    original Harrow Creek vault -- is checked exactly as it always was. Each value is
    curated from a real solicitation or policy and quoted in that scenario's
    requirements.json, next to where it came from.
    """

    salary_cap_usd: int | None = None           # NIH: Executive Level II
    subaward_mtdc_limit_usd: int | None = None  # F&A applies to the first $N of each subaward
    equipment_threshold_usd: int | None = None  # a unit below this is supplies, not equipment
    cost_share: str | None = None               # "prohibited" | None
    min_participant_support_pct: float | None = None  # S-STEM: share that must be scholarships
    max_participant_support_per_person_usd: int | None = None  # S-STEM: per scholar per year
    prior_approval_direct_usd: int | None = None  # NIH: $500K+ direct in any year needs IC agreement

    @classmethod
    def from_dict(cls, raw: dict | None) -> "BudgetRules":
        raw = raw or {}
        def value(key):
            v = raw.get(key)
            return v.get("value") if isinstance(v, dict) else v  # {"value": ..., "quote": ..., "source": ...}
        return cls(**{f: value(f) for f in cls.__dataclass_fields__ if value(f) is not None})


@dataclass
class BudgetCheckResult:
    total_usd: int
    cap_usd: int | None
    over_by_usd: int
    fringe_rate_budget: float | None
    fringe_rate_agreement: float | None
    total_if_agreement_fringe_usd: int | None
    issues: list[BudgetIssue] = field(default_factory=list)


def _parse_fringe_rate_from_text(text: str) -> float | None:
    m = _FRINGE_RATE_RE.search(text)
    if not m:
        return None
    return float(m.group(1)) / 100.0


def _group_by_year(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    by_year: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_year.setdefault(row["year"], []).append(row)
    return by_year


def _recompute_total_with_fringe_rate(rows: list[dict[str, str]], new_fringe_rate: float) -> int:
    total = 0
    for year_rows in _group_by_year(rows).values():
        personnel = sum(int(r["amount_usd"]) for r in year_rows if r["category"] in _PERSONNEL_CATEGORIES)
        indirect_rate = next((float(r["indirect_applied"]) for r in year_rows if r["category"] == "indirect"), 0.0)
        mtdc_other = sum(int(r["amount_usd"]) for r in year_rows if r["category"] not in _MTDC_EXCLUDED)
        equipment_participant = sum(
            int(r["amount_usd"]) for r in year_rows if r["category"] in _INDIRECT_EXCLUDED_CATEGORIES
        )
        # Subawards stay in the total; they carry F&A only where the budget applied it.
        subawards = sum(int(r["amount_usd"]) for r in year_rows if r["category"] == "subaward")
        subawards_in_base = sum(int(r["amount_usd"]) for r in year_rows
                                if r["category"] == "subaward" and float(r.get("indirect_applied") or 0) > 0)
        new_fringe = _round_half_up(new_fringe_rate * personnel)
        mtdc_base = personnel + new_fringe + mtdc_other
        new_indirect = _round_half_up(indirect_rate * (mtdc_base + subawards_in_base))
        total += mtdc_base + equipment_participant + subawards + new_indirect
    return total


def _discover(vault_dir: Path, keyword: str, suffix: str) -> Path | None:
    """Find a document by name when the configured id is not present."""
    matches = sorted(p for p in vault_dir.glob(f"*{suffix}") if keyword in p.stem.lower())
    return matches[0] if matches else None


def check_budget(
    vault_dir: Path,
    budget_doc_id: str = "budget-satp-draft",
    rate_agreement_doc_id: str = "rate-agreement-fy2026",
    cap_usd: int | None = 1_000_000,
    track_years: int = 3,
    equipment_threshold_usd: float = 10_000,
    rules: BudgetRules | None = None,
) -> BudgetCheckResult:
    vault_dir = Path(vault_dir)
    budget_path = vault_dir / f"{budget_doc_id}.csv"
    if not budget_path.is_file():
        # The defaults are the first scenario's document names. Rather than require
        # every caller to know which scenario it is looking at, fall back to finding
        # the budget by shape -- otherwise the agent's own budget check fails outright
        # on any vault but the original one.
        found = _discover(vault_dir, "budget", ".csv")
        if found is None:
            raise LoaderError(f"budget document not found: {budget_path}")
        budget_path, budget_doc_id = found, found.stem

    doc = load(budget_path, doc_id=budget_doc_id)
    if doc.csv_rows is None:
        raise LoaderError(f"{budget_path}: not a CSV document")
    # Committed cost share is the institution's money, not a request: it is recorded
    # in the budget so the cost-share rule can see it, and kept out of every total.
    cost_share_rows = [r for r in doc.csv_rows if r["category"] == "cost_share"]
    rows = [r for r in doc.csv_rows if r["category"] != "cost_share"]

    issues: list[BudgetIssue] = []

    total_usd = sum(int(r["amount_usd"]) for r in rows)
    # None: the solicitation sets no ceiling (NIH parent R01s), so there is nothing to exceed.
    over_by_usd = total_usd - cap_usd if cap_usd is not None else 0
    if over_by_usd > 0:
        issues.append(
            BudgetIssue(
                code="BUDGET_OVER_CAP",
                severity="high",
                detail=f"Draft budget total ${total_usd:,} exceeds the track maximum of ${cap_usd:,} by ${over_by_usd:,}.",
                evidence=[budget_doc_id],
            )
        )

    fringe_rate_budget = next((float(r["fringe_applied"]) for r in rows if r["category"] == "fringe" and float(r["fringe_applied"]) > 0), None)

    fringe_rate_agreement: float | None = None
    total_if_agreement_fringe_usd: int | None = None
    rate_agreement_path = vault_dir / f"{rate_agreement_doc_id}.md"
    if not rate_agreement_path.is_file():
        found = _discover(vault_dir, "rate-agreement", ".md")
        if found is not None:
            rate_agreement_path, rate_agreement_doc_id = found, found.stem
    if rate_agreement_path.is_file():
        agreement_doc = load(rate_agreement_path, doc_id=rate_agreement_doc_id)
        fringe_rate_agreement = _parse_fringe_rate_from_text(agreement_doc.text)
        if fringe_rate_agreement is not None and fringe_rate_budget is not None:
            total_if_agreement_fringe_usd = _recompute_total_with_fringe_rate(rows, fringe_rate_agreement)
            if abs(fringe_rate_budget - fringe_rate_agreement) > 1e-9:
                issues.append(
                    BudgetIssue(
                        code="FRINGE_RATE_MISMATCH",
                        severity="high",
                        detail=(
                            f"Budget applies {fringe_rate_budget:.1%} fringe; the rate agreement specifies "
                            f"{fringe_rate_agreement:.1%}. Correcting it changes the total to "
                            f"${total_if_agreement_fringe_usd:,}."
                        ),
                        evidence=[budget_doc_id, rate_agreement_doc_id],
                    )
                )

    # One code per category: two scenarios charging F&A on different excluded lines are
    # different mistakes, and a single generic code made them read as the same one.
    excluded_codes = {"equipment": "INDIRECT_ON_EQUIPMENT", "participant_support": "INDIRECT_ON_PARTICIPANT_SUPPORT"}
    for r in rows:
        if r["category"] in _INDIRECT_EXCLUDED_CATEGORIES:
            indirect_applied = float(r.get("indirect_applied") or 0)
            if indirect_applied > 0:
                issues.append(
                    BudgetIssue(
                        code=excluded_codes[r["category"]],
                        severity="high",
                        detail=(f"Indirect costs applied to {'an' if r['category'][0] in 'aeiou' else 'a'} "
                                f"{r['category'].replace('_', ' ')} line ({r['line_item']}, year {r['year']}), "
                                "which must be excluded from the indirect base."),
                        evidence=[budget_doc_id],
                    )
                )

    if rules is not None:
        issues.extend(_rule_issues(rows, cost_share_rows, rules, total_usd, budget_doc_id))

    issues = _merge_by_line(issues)

    by_year = _group_by_year(rows)
    year_totals = {year: sum(int(r["amount_usd"]) for r in year_rows) for year, year_rows in by_year.items()}
    if sum(year_totals.values()) != total_usd:
        issues.append(
            BudgetIssue(
                code="YEAR_SUM_MISMATCH",
                severity="high",
                detail="Per-year subtotals do not add up to the grand total.",
                evidence=[budget_doc_id],
            )
        )

    distinct_years = {y for y in by_year if y.isdigit()}
    if len(distinct_years) > track_years:
        issues.append(
            BudgetIssue(
                code="DURATION_OVER_LIMIT",
                severity="high",
                detail=f"Budget spans {len(distinct_years)} years, exceeding the track's {track_years}-year limit.",
                evidence=[budget_doc_id],
            )
        )

    return BudgetCheckResult(
        total_usd=total_usd,
        cap_usd=cap_usd,
        over_by_usd=over_by_usd,
        fringe_rate_budget=fringe_rate_budget,
        fringe_rate_agreement=fringe_rate_agreement,
        total_if_agreement_fringe_usd=total_if_agreement_fringe_usd,
        issues=issues,
    )


def _num(row: dict[str, str], key: str) -> float | None:
    raw = (row.get(key) or "").strip()
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def _rule_issues(
    rows: list[dict[str, str]],
    cost_share_rows: list[dict[str, str]],
    rules: BudgetRules,
    total_usd: int,
    budget_doc_id: str,
) -> list[BudgetIssue]:
    """Checks that exist only because a solicitation or policy says so.

    Optional CSV columns feed them: `annual_salary_usd` and `effort_pct` on personnel
    lines, `units` (scholars, instruments) on participant-support and equipment lines.
    A rule whose column is absent is skipped rather than guessed at.
    """
    issues: list[BudgetIssue] = []
    ev = [budget_doc_id]

    if rules.salary_cap_usd:
        for r in rows:
            salary, effort = _num(r, "annual_salary_usd"), _num(r, "effort_pct")
            if r["category"] not in _PERSONNEL_CATEGORIES or not salary or not effort:
                continue
            allowed = _round_half_up(min(salary, rules.salary_cap_usd) * effort / 100)
            if salary > rules.salary_cap_usd and int(r["amount_usd"]) > allowed:
                issues.append(BudgetIssue(
                    code="SALARY_OVER_CAP", severity="high",
                    detail=(f"{r['line_item']} (year {r['year']}) requests ${int(r['amount_usd']):,}: "
                            f"{effort:g}% of a ${salary:,.0f} salary. Salary above the "
                            f"${rules.salary_cap_usd:,} cap cannot be charged; the most allowable is ${allowed:,}."),
                    evidence=ev))

    if rules.prior_approval_direct_usd:
        # Direct costs = everything but F&A. Consortium F&A is excluded by the policy, but
        # a subaward line here is the partner's total, so this errs towards flagging.
        for year, year_rows in sorted(_group_by_year(rows).items()):
            direct = sum(int(r["amount_usd"]) for r in year_rows if r["category"] != "indirect")
            if direct >= rules.prior_approval_direct_usd:
                issues.append(BudgetIssue(
                    code="LARGE_BUDGET_PRIOR_APPROVAL", severity="high",
                    detail=(f"Year {year} requests ${direct:,} in direct costs. At ${rules.prior_approval_direct_usd:,} "
                            "or more in any year, the funding institute must agree in writing to accept the "
                            "application, requested at least six weeks before submission. Confirm that agreement is on file."),
                    evidence=ev))
                break  # one finding: the first year over the line is what triggers the rule

    if rules.subaward_mtdc_limit_usd is not None:
        by_sub: dict[str, list[dict[str, str]]] = {}
        for r in rows:
            if r["category"] == "subaward":
                # app.budget_builder writes one subaward as "X (in MTDC base)" and
                # "X (beyond MTDC limit)" rows; they are the same subaward.
                by_sub.setdefault(r["line_item"].split(" (")[0], []).append(r)
        for name, sub_rows in by_sub.items():
            charged = sum(int(r["amount_usd"]) for r in sub_rows if float(r.get("indirect_applied") or 0) > 0)
            if charged > rules.subaward_mtdc_limit_usd:
                rate = max(float(r.get("indirect_applied") or 0) for r in sub_rows)
                excess = _round_half_up((charged - rules.subaward_mtdc_limit_usd) * rate)
                issues.append(BudgetIssue(
                    code="SUBAWARD_FA_OVER_LIMIT", severity="high",
                    detail=(f"Indirect costs are charged on ${charged:,} of the {name}"
                            f"{'' if name.lower().endswith('subaward') else ' subaward'}; only the first "
                            f"${rules.subaward_mtdc_limit_usd:,} of each subaward is in the MTDC base. "
                            f"That overcharges about ${excess:,} in F&A."),
                    evidence=ev))

    if rules.equipment_threshold_usd:
        for r in rows:
            if r["category"] != "equipment" or int(r["amount_usd"]) <= 0:
                continue
            units = _num(r, "units") or 1
            unit_cost = int(r["amount_usd"]) / units
            if unit_cost < rules.equipment_threshold_usd:
                issues.append(BudgetIssue(
                    code="EQUIPMENT_BELOW_THRESHOLD", severity="medium",
                    detail=(f"{r['line_item']} (year {r['year']}) is budgeted as equipment at ${unit_cost:,.0f} per unit, "
                            f"below the ${rules.equipment_threshold_usd:,} equipment threshold. It is supplies, "
                            "and belongs in the indirect-cost base."),
                    evidence=ev))

    if rules.cost_share == "prohibited":
        committed = sum(int(r["amount_usd"]) for r in cost_share_rows)
        if committed > 0:
            issues.append(BudgetIssue(
                code="COST_SHARE_PROHIBITED", severity="high",
                detail=(f"The budget commits ${committed:,} in cost share. This solicitation prohibits "
                        "voluntary committed cost sharing; it must be removed from the budget and justification."),
                evidence=ev))

    if rules.min_participant_support_pct and total_usd:
        share = sum(int(r["amount_usd"]) for r in rows if r["category"] == "participant_support") / total_usd
        if share + 1e-9 < rules.min_participant_support_pct / 100:
            issues.append(BudgetIssue(
                code="PARTICIPANT_SUPPORT_BELOW_MINIMUM", severity="high",
                detail=(f"Scholarships are {share:.1%} of the total request; the solicitation requires at least "
                        f"{rules.min_participant_support_pct:g}%."),
                evidence=ev))

    if rules.max_participant_support_per_person_usd:
        for r in rows:
            units = _num(r, "units")
            if r["category"] != "participant_support" or not units:
                continue
            per_person = int(r["amount_usd"]) / units
            if per_person > rules.max_participant_support_per_person_usd:
                issues.append(BudgetIssue(
                    code="SCHOLARSHIP_OVER_MAX", severity="high",
                    detail=(f"{r['line_item']} (year {r['year']}) averages ${per_person:,.0f} per scholar; the "
                            f"maximum is ${rules.max_participant_support_per_person_usd:,} per year."),
                    evidence=ev))

    return issues


_YEAR_IN_DETAIL = re.compile(r" \(year (\d+)\)| year (\d+)\)")


def _merge_by_line(issues: list[BudgetIssue]) -> list[BudgetIssue]:
    """One finding per budget line, not one per year.

    A salary over the cap in all five years is one mistake. Reported per year it put five
    identical cards on the Issues tab, which reads as five problems to an audience.
    """
    merged: dict[tuple[str, str], tuple[BudgetIssue, list[str]]] = {}
    out: list[BudgetIssue] = []
    for issue in issues:
        m = _YEAR_IN_DETAIL.search(issue.detail)
        if not m:
            out.append(issue)
            continue
        year = m.group(1) or m.group(2)
        key = (issue.code, _YEAR_IN_DETAIL.sub("", issue.detail, count=1))
        if key in merged:
            merged[key][1].append(year)
            continue
        merged[key] = (issue, [year])
        out.append(issue)
    for first, years in merged.values():
        if len(years) > 1:
            label = f"years {years[0]}–{years[-1]}" if len(years) > 2 else f"years {' and '.join(years)}"
            first.detail = _YEAR_IN_DETAIL.sub(lambda m: m.group(0).replace(f"year {m.group(1) or m.group(2)}", label), first.detail, count=1)
    return out
