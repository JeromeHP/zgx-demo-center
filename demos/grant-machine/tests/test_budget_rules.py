"""Solicitation-specific budget rules, and the seed-declared budget builder.

Each rule check must fire when its rule is broken and stay silent otherwise -- and a
scenario that declares no rules (Harrow Creek) must be checked exactly as before.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.budget_builder import build
from app.checks.budget import BudgetRules, check_budget

_HEADER = "year,category,line_item,amount_usd,fringe_applied,indirect_applied,notes,annual_salary_usd,effort_pct,units\n"


def _budget(tmp_path: Path, rows: list[str]) -> Path:
    (tmp_path / "budget-test.csv").write_text("# doc_id: budget-test\n" + _HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    return tmp_path


def _codes(tmp_path, rows, **rules):
    result = check_budget(_budget(tmp_path, rows), budget_doc_id="budget-test", cap_usd=None, track_years=6,
                          rules=BudgetRules(**rules))
    return {i.code for i in result.issues}


def test_salary_over_cap(tmp_path):
    over = ["1,senior_personnel,PI,62500,0,0.5,,250000,25,"]      # 25% of $250,000
    capped = ["1,senior_personnel,PI,57000,0,0.5,,250000,25,"]    # 25% of the $228,000 cap
    assert "SALARY_OVER_CAP" in _codes(tmp_path, over, salary_cap_usd=228_000)
    assert "SALARY_OVER_CAP" not in _codes(tmp_path, capped, salary_cap_usd=228_000)


def test_subaward_fa_over_limit(tmp_path):
    full = ["1,subaward,Drumlin subaward (in MTDC base),60000,0,0.58,,,,"]
    split = ["1,subaward,Drumlin subaward (in MTDC base),25000,0,0.58,,,,",
             "1,subaward,Drumlin subaward (beyond MTDC limit),35000,0,0,,,,"]
    assert "SUBAWARD_FA_OVER_LIMIT" in _codes(tmp_path, full, subaward_mtdc_limit_usd=25_000)
    assert "SUBAWARD_FA_OVER_LIMIT" not in _codes(tmp_path, split, subaward_mtdc_limit_usd=25_000)


def test_equipment_below_threshold(tmp_path):
    rows = ["1,equipment,Pressure loggers,24000,0,0,,,,6"]  # $4,000 each
    assert "EQUIPMENT_BELOW_THRESHOLD" in _codes(tmp_path, rows, equipment_threshold_usd=10_000)
    assert "EQUIPMENT_BELOW_THRESHOLD" not in _codes(tmp_path, ["1,equipment,ADCP,48000,0,0,,,,1"],
                                                     equipment_threshold_usd=10_000)


def test_cost_share_prohibited(tmp_path):
    rows = ["1,senior_personnel,PI,50000,0,0.4,,,,", "1,cost_share,Campus match,125000,0,0,,,,"]
    codes = _codes(tmp_path, rows, cost_share="prohibited")
    assert "COST_SHARE_PROHIBITED" in codes
    # Cost share is the institution's money, not the request.
    result = check_budget(tmp_path, budget_doc_id="budget-test", cap_usd=None, rules=BudgetRules())
    assert result.total_usd == 50000


def test_scholarship_share_and_maximum(tmp_path):
    rows = ["1,participant_support,S-STEM scholarships,160000,0,0,,,,10",   # $16,000 each
            "1,senior_personnel,PI,140000,0,0.4,,,,"]
    codes = _codes(tmp_path, rows, min_participant_support_pct=60, max_participant_support_per_person_usd=15_000)
    assert {"PARTICIPANT_SUPPORT_BELOW_MINIMUM", "SCHOLARSHIP_OVER_MAX"} <= codes


def test_indirect_on_excluded_lines_is_named_by_category(tmp_path):
    rows = ["1,participant_support,Stipends,10000,0,0.4,,,,", "1,equipment,Instrument,50000,0,0.4,,,,"]
    assert {"INDIRECT_ON_PARTICIPANT_SUPPORT", "INDIRECT_ON_EQUIPMENT"} <= _codes(tmp_path, rows)


def test_no_rules_means_no_rule_issues(tmp_path):
    rows = ["1,senior_personnel,PI,62500,0,0.5,,250000,25,", "1,cost_share,Match,1000,0,0,,,,"]
    result = check_budget(_budget(tmp_path, rows), budget_doc_id="budget-test", cap_usd=None)
    assert result.issues == []


def test_harrow_creek_is_checked_exactly_as_before():
    vault = Path(__file__).resolve().parents[1] / "data" / "vault"
    codes = {i.code for i in check_budget(vault, cap_usd=1_000_000, track_years=3).issues}
    assert codes == {"BUDGET_OVER_CAP", "FRINGE_RATE_MISMATCH"}


# -- the builder ---------------------------------------------------------------------


def _seed(lines, **budget):
    return {
        "new_project": {"years": 2},
        "rates": {"indirect_rate_pct": 50.0},
        "budget": {"fringe_rate_pct": 30.0, "lines": lines, **budget},
    }


def test_builder_computes_fringe_and_mtdc():
    header, rows, summary = build(_seed([
        {"category": "senior_personnel", "line_item": "PI", "amount_usd": 100000},
        {"category": "equipment", "line_item": "Instrument", "amount_usd": [40000, 0]},
        {"category": "participant_support", "line_item": "Stipends", "amount_usd": 10000},
    ]))
    y1 = {r["category"]: r["amount_usd"] for r in rows if r["year"] == 1}
    assert y1["fringe"] == 30000
    assert y1["indirect"] == 65000  # 50% of personnel + fringe; equipment and stipends excluded
    assert summary["grand_total_usd"] == (100000 + 30000 + 40000 + 10000 + 65000) + (100000 + 30000 + 10000 + 65000)


def test_builder_limits_subaward_fa_unless_the_error_is_planted():
    lines = [{"category": "subaward", "line_item": "Partner subaward", "amount_usd": [40000, 40000]}]
    _, rows, _ = build(_seed(lines, subaward_mtdc_limit_usd=25000))
    in_base = sum(r["amount_usd"] for r in rows if "in MTDC base" in r["line_item"])
    assert in_base == 25000  # first $25,000 over the whole project, not per year

    _, planted, _ = build(_seed([{**lines[0], "fa": "full"}], subaward_mtdc_limit_usd=25000))
    assert sum(r["amount_usd"] for r in planted if "in MTDC base" in r["line_item"]) == 80000


@pytest.mark.parametrize("fa", [False, True])
def test_builder_only_charges_fa_on_excluded_lines_when_planted(fa):
    line = {"category": "participant_support", "line_item": "Stipends", "amount_usd": 10000}
    if fa:
        line["fa"] = True
    _, rows, _ = build(_seed([line]))
    stipend = next(r for r in rows if r["category"] == "participant_support")
    assert (stipend["indirect_applied"] > 0) is fa
