"""Build a scenario's draft budget from the lines its seed declares.

Every generated scenario used to share one hard-coded budget (scripts/generate_vault.py
build_budget_rows): the same $1,038,450, the same "Automation lab equipment" and
"Employer workshop stipends" -- for a clinical trial, a coastal model and a scholarship
programme alike. A seed may now declare its own lines, scaled from real comparable
awards (data/anchors/), and this computes fringe and F&A over them the way a sponsored
programs office would.

Planted budget errors are declared, not computed by accident: `fa: full` on a subaward
charges F&A beyond the MTDC limit, `fa: true` on an excluded line charges F&A where it
must not be, and a `cost_share` line commits cost share. Everything else is correct, so
app.checks.budget finds exactly what the seed planted.

Seeds without a `budget:` block keep the original builder, byte for byte.
"""

from __future__ import annotations

import math
from typing import Any

PERSONNEL = {"senior_personnel", "other_personnel"}
# Categories that are never in the MTDC base unless a line deliberately says otherwise.
EXCLUDED = {"equipment", "participant_support", "tuition_remission", "patient_care", "cost_share"}
EXTRA_COLUMNS = ("annual_salary_usd", "effort_pct", "units")
HEADER = ["year", "category", "line_item", "amount_usd", "fringe_applied", "indirect_applied", "notes"]


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _per_year(value: Any, years: int, escalate_pct: float = 0.0) -> list[int]:
    if isinstance(value, list):
        if len(value) != years:
            raise ValueError(f"expected {years} yearly amounts, got {len(value)}")
        return [int(v) for v in value]
    # A single figure is year 1; later years escalate, as real salary lines do.
    return [_round_half_up(float(value) * (1 + escalate_pct / 100) ** y) for y in range(years)]


def build(seed: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """(header, rows, summary) for a seed that declares `budget:`."""
    budget = seed["budget"]
    years = int(seed["new_project"]["years"])
    fringe_rate = float(budget["fringe_rate_pct"]) / 100
    indirect_rate = float(seed["rates"]["indirect_rate_pct"]) / 100
    sub_limit = budget.get("subaward_mtdc_limit_usd")

    lines = budget["lines"]
    used_extra = [c for c in EXTRA_COLUMNS if any(c in ln for ln in lines)]
    header = HEADER + used_extra

    rows: list[dict[str, Any]] = []
    sub_in_base_so_far: dict[str, int] = {}
    for year in range(1, years + 1):
        year_rows: list[dict[str, Any]] = []
        base = 0
        personnel = 0
        for ln in lines:
            cat = ln["category"]
            escalate = float(ln.get("escalate_pct", budget.get("salary_escalation_pct", 0) if cat in PERSONNEL else 0))
            amount = _per_year(ln["amount_usd"], years, escalate)[year - 1]
            if amount == 0 and not ln.get("keep_zero"):
                continue
            extra = {c: ln[c] for c in used_extra if c in ln}
            if "annual_salary_usd" in extra:  # the salary escalates with the line it drives
                extra["annual_salary_usd"] = _per_year(ln["annual_salary_usd"], years, escalate)[year - 1]
            if cat in PERSONNEL:
                personnel += amount
            if cat == "subaward":
                # F&A on the first $limit of each subaward over the whole project. The
                # correct split is written as two rows so a reader (and the check) can see
                # which dollars are in the base; `fa: full` plants the error instead.
                name = ln["line_item"]
                if ln.get("fa") == "full" or sub_limit is None:
                    in_base, out_base = amount, 0
                else:
                    remaining = max(int(sub_limit) - sub_in_base_so_far.get(name, 0), 0)
                    in_base = min(amount, remaining)
                    out_base = amount - in_base
                    sub_in_base_so_far[name] = sub_in_base_so_far.get(name, 0) + in_base
                if in_base:
                    year_rows.append({"year": year, "category": cat, "line_item": f"{name} (in MTDC base)",
                                      "amount_usd": in_base, "fringe_applied": 0, "indirect_applied": indirect_rate,
                                      "notes": ln.get("notes", ""), **extra})
                    base += in_base
                if out_base:
                    year_rows.append({"year": year, "category": cat, "line_item": f"{name} (beyond MTDC limit)",
                                      "amount_usd": out_base, "fringe_applied": 0, "indirect_applied": 0,
                                      "notes": "excluded from indirect base", **extra})
                continue
            in_base = (cat not in EXCLUDED) or bool(ln.get("fa"))
            if cat != "cost_share" and in_base:
                base += amount
            year_rows.append({"year": year, "category": cat, "line_item": ln["line_item"], "amount_usd": amount,
                              "fringe_applied": 0, "indirect_applied": indirect_rate if in_base and cat != "cost_share" else 0,
                              "notes": ln.get("notes", "" if in_base else "excluded from indirect base"), **extra})
        fringe = _round_half_up(fringe_rate * personnel)
        if fringe:
            year_rows.append({"year": year, "category": "fringe", "line_item": "Fringe benefits", "amount_usd": fringe,
                              "fringe_applied": fringe_rate, "indirect_applied": indirect_rate,
                              "notes": f"{fringe_rate:.1%} of personnel"})
            base += fringe
        indirect = _round_half_up(indirect_rate * base)
        year_rows.append({"year": year, "category": "indirect", "line_item": "Indirect costs (F&A)", "amount_usd": indirect,
                          "fringe_applied": 0, "indirect_applied": indirect_rate,
                          "notes": f"{indirect_rate:.1%} of MTDC base"})
        rows.extend(year_rows)

    requested = [r for r in rows if r["category"] != "cost_share"]
    summary = {
        "grand_total_usd": sum(r["amount_usd"] for r in requested),
        "by_year_usd": {y: sum(r["amount_usd"] for r in requested if r["year"] == y) for y in range(1, years + 1)},
        "cost_share_usd": sum(r["amount_usd"] for r in rows if r["category"] == "cost_share"),
        "fringe_rate_used_pct": budget["fringe_rate_pct"],
        "indirect_rate_pct": seed["rates"]["indirect_rate_pct"],
    }
    return header, rows, summary
