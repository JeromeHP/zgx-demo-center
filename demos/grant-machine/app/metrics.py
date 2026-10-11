"""Token/cost/human-baseline accounting (Spec I). Elapsed time and decode
tokens/s are tracked directly on ToolContext (app/agent/tools.py) and the
orchestrator loop; this module holds the pure-function math so it's testable
without a running agent."""

from __future__ import annotations

from typing import Any

from app.settings import Settings


def cost_equivalent(prompt_tokens: int, completion_tokens: int, settings: Settings) -> dict[str, Any]:
    pricing = settings.demo.cloud_pricing
    if pricing.input_per_million_usd is None or pricing.output_per_million_usd is None:
        return {"configured": False, "usd": None, "label": None}
    usd = (prompt_tokens / 1_000_000) * pricing.input_per_million_usd + (completion_tokens / 1_000_000) * pricing.output_per_million_usd
    return {"configured": True, "usd": usd, "label": pricing.label}


def human_baseline_minutes(settings: Settings, scope_sections: list[str]) -> dict[str, Any]:
    """Sum of configured minutes for the steps in scope; hidden (configured:
    False) if any in-scope value is null. compliance_matrix and budget_review
    are always in scope (every run loads requirements and checks the budget);
    section_draft only counts if the scope drafts at least one section."""
    b = settings.demo.human_baseline_minutes
    relevant = [b.compliance_matrix, b.budget_review]
    if scope_sections:
        relevant.append(b.section_draft)
    if any(v is None for v in relevant):
        return {"configured": False, "minutes": None}
    return {"configured": True, "minutes": sum(relevant)}  # type: ignore[arg-type]
