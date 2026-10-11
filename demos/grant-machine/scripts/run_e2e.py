"""Phase 10: live end-to-end runs against a real running app instance
(pointed at the real model server, not the fake one). Consumes SSE to
completion, evaluates the final state against the planted issues, and writes
a report + prints a summary table.

Usage:
    uv run python scripts/run_e2e.py --base-url http://127.0.0.1:8095 --scope quick --reps 3 --label zbook
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PLANTED_ISSUES_PATH = REPO_ROOT / "data" / "vault" / "_planted_issues.json"
RUN_TIMEOUT_S = 30 * 60


def run_once(base_url: str, scope: str) -> dict[str, Any]:
    resp = httpx.post(f"{base_url}/api/runs", json={"mode": "story", "scope": scope, "show_thinking": True}, timeout=30)
    resp.raise_for_status()
    run_id = resp.json()["run_id"]

    last_metrics: dict[str, Any] | None = None
    start = time.monotonic()
    with httpx.stream("GET", f"{base_url}/api/runs/{run_id}/events", timeout=RUN_TIMEOUT_S) as r:
        event_type = None
        for line in r.iter_lines():
            if time.monotonic() - start > RUN_TIMEOUT_S:
                raise TimeoutError(f"run {run_id} did not finish within {RUN_TIMEOUT_S}s")
            if line.startswith("event:"):
                event_type = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_str = line.split(":", 1)[1].strip()
                if event_type == "metrics":
                    last_metrics = json.loads(data_str)
                elif event_type == "run_finished":
                    break
            elif line == "":
                if event_type == "run_finished":
                    break
                event_type = None

    state = httpx.get(f"{base_url}/api/runs/{run_id}/state", timeout=30).json()
    return {"run_id": run_id, "state": state, "last_metrics": last_metrics}


def evaluate(result: dict[str, Any]) -> dict[str, Any]:
    state = result["state"]
    metrics = result["last_metrics"] or {}
    issues = state.get("issues", [])
    issue_codes = {i.get("code") for i in issues if i.get("code")}
    agent_issue_evidence = {d for i in issues if i.get("source") == "agent" for d in i.get("evidence", [])}

    drafts = state.get("drafts", {})
    citation_checks = state.get("citation_checks", {})
    reviews = state.get("reviews", {})

    p2_pass = False
    rps_drafts = drafts.get("results_prior_support")
    rps_checks = citation_checks.get("results_prior_support")
    if rps_drafts and rps_checks:
        latest_text = rps_drafts[-1]["text"]
        p2_pass = "84" in latest_text and "120" in latest_text and rps_checks[-1]["passed"]

    revisions_per_section = {sid: len(versions) - 1 for sid, versions in drafts.items()}
    unsupported_claims_final = {
        sid: len(checks[-1]["unsupported_claims"]) for sid, checks in citation_checks.items() if checks
    }
    mean_scores_final = {sid: revs[-1]["mean_score"] for sid, revs in reviews.items() if revs}

    return {
        "completed": state.get("status") == "completed",
        "status": state.get("status"),
        "p1_over_cap_caught": "BUDGET_OVER_CAP" in issue_codes,
        "p4_fringe_mismatch_caught": "FRINGE_RATE_MISMATCH" in issue_codes,
        "p2_honest_prior_outcomes": p2_pass,
        "p3_stale_letter_caught": "letter-northgate-2021" in agent_issue_evidence,
        "steps": state.get("steps"),
        "tool_calls_total": sum(state.get("tool_calls_made", {}).values()),
        "tool_calls_made": state.get("tool_calls_made"),
        "elapsed_s": metrics.get("elapsed_s"),
        "prompt_tokens": metrics.get("prompt_tokens"),
        "completion_tokens": metrics.get("completion_tokens"),
        "tokens_estimated": metrics.get("tokens_estimated"),
        "revisions_per_section": revisions_per_section,
        "unsupported_claims_final": unsupported_claims_final,
        "mean_scores_final": mean_scores_final,
        "run_id": result["run_id"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--scope", required=True, choices=["quick", "full"])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    reps_results: list[dict[str, Any]] = []
    for i in range(args.reps):
        print(f"[{args.label}/{args.scope}] rep {i + 1}/{args.reps} starting...")
        rep_start = time.monotonic()
        try:
            result = run_once(args.base_url, args.scope)
            evaluation = evaluate(result)
        except Exception as exc:  # noqa: BLE001 -- one rep's crash must not abort the batch
            evaluation = {"completed": False, "status": "harness_error", "error": str(exc)}
        wall_s = time.monotonic() - rep_start
        evaluation["wall_s"] = wall_s
        reps_results.append(evaluation)
        print(
            f"[{args.label}/{args.scope}] rep {i + 1}/{args.reps}: "
            f"status={evaluation.get('status')} wall={wall_s:.1f}s "
            f"P1={evaluation.get('p1_over_cap_caught')} P4={evaluation.get('p4_fringe_mismatch_caught')} "
            f"P2={evaluation.get('p2_honest_prior_outcomes')} P3={evaluation.get('p3_stale_letter_caught')}"
        )

    n = len(reps_results)
    completed_n = sum(1 for r in reps_results if r.get("completed"))
    p1_n = sum(1 for r in reps_results if r.get("p1_over_cap_caught"))
    p4_n = sum(1 for r in reps_results if r.get("p4_fringe_mismatch_caught"))
    p2_n = sum(1 for r in reps_results if r.get("p2_honest_prior_outcomes"))
    p3_n = sum(1 for r in reps_results if r.get("p3_stale_letter_caught"))
    wall_times = sorted(r["wall_s"] for r in reps_results if r.get("wall_s") is not None)
    median_wall = wall_times[len(wall_times) // 2] if wall_times else None

    summary = {
        "label": args.label,
        "scope": args.scope,
        "reps": n,
        "completed": f"{completed_n}/{n}",
        "p1": f"{p1_n}/{n}",
        "p4": f"{p4_n}/{n}",
        "p2": f"{p2_n}/{n}",
        "p3": f"{p3_n}/{n}",
        "median_wall_s": median_wall,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reps_detail": reps_results,
    }

    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = reports_dir / f"e2e_{args.label}_{args.scope}_{ts}.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n=== Summary: {args.label}/{args.scope} ({n} reps) ===")
    print(f"Completed: {summary['completed']}")
    print(f"P1 (over cap): {summary['p1']}")
    print(f"P4 (fringe mismatch): {summary['p4']}")
    print(f"P2 (honest prior outcomes): {summary['p2']}")
    print(f"P3 (stale letter): {summary['p3']}")
    print(f"Median wall time: {median_wall:.1f}s" if median_wall is not None else "Median wall time: n/a")
    print(f"Report written to {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
