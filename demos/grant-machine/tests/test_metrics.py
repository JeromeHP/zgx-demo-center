from __future__ import annotations

from app.metrics import cost_equivalent, human_baseline_minutes


def test_cost_equivalent_unconfigured_when_pricing_null(make_settings):
    settings = make_settings()
    result = cost_equivalent(1000, 500, settings)
    assert result == {"configured": False, "usd": None, "label": None}


def test_cost_equivalent_computes_when_pricing_set(make_settings):
    settings = make_settings(
        overrides={"demo": {"cloud_pricing": {"input_per_million_usd": 3.0, "output_per_million_usd": 15.0, "label": "test list price"}}}
    )
    result = cost_equivalent(1_000_000, 1_000_000, settings)
    assert result["configured"] is True
    assert result["usd"] == 3.0 + 15.0
    assert result["label"] == "test list price"


def test_human_baseline_unconfigured_when_any_relevant_value_null(make_settings):
    settings = make_settings(overrides={"demo": {"human_baseline_minutes": {"compliance_matrix": 30, "budget_review": None}}})
    result = human_baseline_minutes(settings, scope_sections=[])
    assert result == {"configured": False, "minutes": None}


def test_human_baseline_configured_without_sections(make_settings):
    settings = make_settings(
        overrides={"demo": {"human_baseline_minutes": {"compliance_matrix": 30, "budget_review": 20, "section_draft": None}}}
    )
    result = human_baseline_minutes(settings, scope_sections=[])
    assert result == {"configured": True, "minutes": 50}


def test_human_baseline_includes_section_draft_when_in_scope(make_settings):
    settings = make_settings(
        overrides={"demo": {"human_baseline_minutes": {"compliance_matrix": 30, "budget_review": 20, "section_draft": 45}}}
    )
    result = human_baseline_minutes(settings, scope_sections=["results_prior_support"])
    assert result == {"configured": True, "minutes": 95}


def test_human_baseline_hidden_when_section_draft_null_but_in_scope(make_settings):
    settings = make_settings(
        overrides={"demo": {"human_baseline_minutes": {"compliance_matrix": 30, "budget_review": 20, "section_draft": None}}}
    )
    result = human_baseline_minutes(settings, scope_sections=["results_prior_support"])
    assert result == {"configured": False, "minutes": None}
