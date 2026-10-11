from __future__ import annotations

from pathlib import Path

import pytest

from app.checks.budget import check_budget

_HEADER = "year,category,line_item,amount_usd,fringe_applied,indirect_applied,notes\n"


def _write_budget_csv(vault_dir: Path, rows: list[str], doc_id: str = "budget-satp-draft") -> None:
    (vault_dir / f"{doc_id}.csv").write_text(
        f"# doc_id: {doc_id}\n# title: Test Budget\n# synthetic: true\n" + _HEADER + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def _write_rate_agreement(vault_dir: Path, fringe_pct: float, doc_id: str = "rate-agreement-fy2026") -> None:
    (vault_dir / f"{doc_id}.md").write_text(
        "---\ntitle: Rate Agreement\n---\n# Rate Agreement\n\n"
        f"The negotiated fringe benefit rate is {fringe_pct}% and the indirect cost rate is 36.0% of MTDC.\n",
        encoding="utf-8",
    )


def _over_cap_rows() -> list[str]:
    # personnel 100000, fringe@28%=28000, equipment 900000 (excluded from indirect),
    # indirect@36% of MTDC(100000+28000)=46080. Grand total 1,074,080 > $1,000,000 cap.
    return [
        "1,senior_personnel,PI effort,100000,0,0.36,",
        "1,fringe,Fringe benefits,28000,0.28,0.36,28.0% of personnel",
        "1,equipment,Lab equipment,900000,0,0,excluded from indirect base",
        "1,indirect,Indirect costs,46080,0,0.36,36.0% of MTDC base",
    ]


def test_over_cap_detected(tmp_path):
    _write_budget_csv(tmp_path, _over_cap_rows())
    result = check_budget(tmp_path, cap_usd=1_000_000)
    assert "BUDGET_OVER_CAP" in [i.code for i in result.issues]
    assert result.total_usd == 100000 + 28000 + 900000 + 46080
    assert result.over_by_usd == result.total_usd - 1_000_000


def test_fringe_rate_mismatch_detected_against_rate_agreement(tmp_path):
    _write_budget_csv(tmp_path, _over_cap_rows())
    _write_rate_agreement(tmp_path, fringe_pct=31.5)
    result = check_budget(tmp_path, cap_usd=2_000_000)  # high enough to isolate the fringe issue
    codes = [i.code for i in result.issues]
    assert "FRINGE_RATE_MISMATCH" in codes
    assert "BUDGET_OVER_CAP" not in codes
    assert result.fringe_rate_budget == pytest.approx(0.28)
    assert result.fringe_rate_agreement == pytest.approx(0.315)
    assert result.total_if_agreement_fringe_usd is not None
    assert result.total_if_agreement_fringe_usd > result.total_usd


def test_indirect_on_excluded_category_detected(tmp_path):
    rows = [
        "1,equipment,Lab equipment,50000,0,0.36,should not have indirect",
        "1,indirect,Indirect costs,18000,0,0.36,",
    ]
    _write_budget_csv(tmp_path, rows)
    result = check_budget(tmp_path, cap_usd=10_000_000)
    assert "INDIRECT_ON_EQUIPMENT" in [i.code for i in result.issues]


def test_duration_over_limit_detected(tmp_path):
    rows = []
    for year in (1, 2, 3, 4):
        rows.append(f"{year},senior_personnel,PI effort,50000,0,0.36,")
    _write_budget_csv(tmp_path, rows)
    result = check_budget(tmp_path, cap_usd=10_000_000, track_years=3)
    assert "DURATION_OVER_LIMIT" in [i.code for i in result.issues]


def test_clean_budget_produces_zero_issues(tmp_path):
    rows = [
        "1,senior_personnel,PI effort,100000,0,0.36,",
        "1,fringe,Fringe benefits,31500,0.315,0.36,31.5% of personnel",
        "1,equipment,Lab equipment,20000,0,0,excluded from indirect base",
        "1,participant_support,Stipends,5000,0,0,excluded from indirect base",
        "1,materials_supplies,Supplies,3000,0,0.36,",
        "1,indirect,Indirect costs,48420,0,0.36,36.0% of MTDC base",
    ]
    _write_budget_csv(tmp_path, rows)
    _write_rate_agreement(tmp_path, fringe_pct=31.5)
    result = check_budget(tmp_path, cap_usd=1_000_000, track_years=3)
    assert result.issues == []


def test_real_generated_vault_has_exactly_the_planted_budget_issues():
    repo_root = Path(__file__).resolve().parents[1]
    vault_dir = repo_root / "data" / "vault"
    if not (vault_dir / "budget-satp-draft.csv").is_file():
        pytest.skip("real vault budget CSV not generated yet")
    result = check_budget(vault_dir, cap_usd=1_000_000, track_years=3)
    codes = {i.code for i in result.issues}
    assert "BUDGET_OVER_CAP" in codes
    assert not codes & {"INDIRECT_ON_EQUIPMENT", "INDIRECT_ON_PARTICIPANT_SUPPORT"}
    assert "YEAR_SUM_MISMATCH" not in codes
    assert "DURATION_OVER_LIMIT" not in codes
    if (vault_dir / "rate-agreement-fy2026.md").is_file():
        assert "FRINGE_RATE_MISMATCH" in codes
