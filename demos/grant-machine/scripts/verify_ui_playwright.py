"""Phase 8 UI verification: drives a full-scope scripted run (against the fake
LLM server, for determinism/speed) through a real running app instance with
Playwright, screenshots at 1920x1080 and 1366x768, and checks every output
panel updated. Also re-loads the page with all non-localhost requests
blocked and confirms it renders identically with zero blocked requests.

Usage: uv run python scripts/verify_ui_playwright.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FAKE_LLM_PORT = 18081
APP_PORT = 18095
SCREENSHOT_DIR = REPO_ROOT / "reports" / "ui_screenshots"


def _tc(name, args, call_id):
    from tests.fake_llm_server import ScriptedToolCall

    return ScriptedToolCall(id=call_id, name=name, arguments=json.dumps(args))


def build_scripted_queue():
    from tests.fake_llm_server import ScriptedResponse

    queue = []
    queue.append(ScriptedResponse(content="", tool_calls=[_tc("get_requirements", {}, "c1")]))
    queue.append(ScriptedResponse(content="", tool_calls=[_tc("check_budget", {"track": "track_2"}, "c2")]))

    sections = ["project_summary", "results_prior_support", "evaluation_plan"]
    call_n = 3
    for section_id in sections:
        queue.append(
            ScriptedResponse(
                content="",
                tool_calls=[
                    _tc(
                        "draft_section",
                        {"section_id": section_id, "evidence": [{"doc_id": "prior-annual-report-2025"}], "guidance": ""},
                        f"c{call_n}",
                    )
                ],
            )
        )
        call_n += 1
        queue.append(
            ScriptedResponse(
                content=f"This is the {section_id} draft. The project produced 84 completers against a target of 120. [[prior-annual-report-2025]]"
            )
        )
        queue.append(ScriptedResponse(content="", tool_calls=[_tc("review_section", {"section_id": section_id}, f"c{call_n}")]))
        call_n += 1
        queue.append(
            ScriptedResponse(
                content=json.dumps(
                    {"criteria": [{"id": "R-1", "score": 4.5, "rationale": "solid", "gaps": []}], "mean_score": 4.5, "must_fix": []}
                )
            )
        )
        queue.append(ScriptedResponse(content="", tool_calls=[_tc("verify_citations", {"section_id": section_id}, f"c{call_n}")]))
        call_n += 1

    queue.append(
        ScriptedResponse(
            content="",
            tool_calls=[
                _tc(
                    "flag_issue",
                    {
                        "issue": {
                            "severity": "high",
                            "title": "Stale support letter",
                            "detail": "The Northgate letter is dated 2021 and references the prior project.",
                            "evidence": ["letter-northgate-2021"],
                            "recommended_action": "Request an updated letter referencing SATP.",
                        }
                    },
                    f"c{call_n}",
                )
            ],
        )
    )
    call_n += 1
    queue.append(
        ScriptedResponse(
            content="",
            tool_calls=[
                _tc(
                    "update_compliance",
                    {"item": {"requirement_id": "R-001", "status": "partial", "evidence": ["prior-annual-report-2025"], "note": "In progress"}},
                    f"c{call_n}",
                )
            ],
        )
    )
    call_n += 1
    queue.append(ScriptedResponse(content="", tool_calls=[_tc("finish", {"summary": "Full-scope demo run complete."}, f"c{call_n}")]))
    return queue


def start_fake_llm_server():
    import uvicorn
    from tests.fake_llm_server import create_app

    app = create_app()
    app.state.queue.extend(build_scripted_queue())
    config = uvicorn.Config(app, host="127.0.0.1", port=FAKE_LLM_PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.02)
    return server, app


def start_app_server():
    # default.yaml's "full" scope is already [project_summary, results_prior_support,
    # evaluation_plan], matching the scripted queue below -- only base_url/port need overriding.
    os.environ["GM_LLM__BASE_URL"] = f"http://127.0.0.1:{FAKE_LLM_PORT}/v1"
    os.environ["GM_SERVER__PORT"] = str(APP_PORT)

    import uvicorn

    from app.main import app

    config = uvicorn.Config(app, host="127.0.0.1", port=APP_PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.02)
    return server


def run_browser_checks(fake_app) -> list[str]:
    from playwright.sync_api import sync_playwright

    failures: list[str] = []
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{APP_PORT}/"

    with sync_playwright() as p:
        browser = p.chromium.launch()

        for width, height in [(1920, 1080), (1366, 768)]:
            page = browser.new_page(viewport={"width": width, "height": height})
            page.goto(base_url)
            page.select_option("#scope-select", "full")
            fake_app.state.queue.extend(build_scripted_queue())
            page.click("#btn-start")

            try:
                page.wait_for_selector("#run-summary-overlay:not([hidden])", timeout=120_000)
            except Exception as exc:
                failures.append(f"{width}x{height}: run did not complete within timeout: {exc}")
                page.screenshot(path=str(SCREENSHOT_DIR / f"FAILURE_{width}x{height}.png"), full_page=True)
                page.close()
                continue

            page.screenshot(path=str(SCREENSHOT_DIR / f"summary_{width}x{height}.png"), full_page=True)

            page.click("#btn-summary-reset")
            time.sleep(0.3)

            # Re-run without closing, this time inspect panels before the summary overlay appears again is tricky;
            # instead, drive a second run and inspect panel content mid/at-completion via the API-backed DOM state
            # that persists after run_finished (event handlers already populated the tables).
            page.select_option("#scope-select", "full")
            fake_app.state.queue.extend(build_scripted_queue())
            page.click("#btn-start")
            page.wait_for_selector("#run-summary-overlay:not([hidden])", timeout=120_000)
            page.click("#btn-summary-close")

            page.click("[data-tab='compliance']")
            compliance_rows = page.locator("#compliance-table tbody tr").count()
            if compliance_rows == 0:
                failures.append(f"{width}x{height}: compliance table has no rows")
            page.screenshot(path=str(SCREENSHOT_DIR / f"compliance_{width}x{height}.png"))

            page.click("[data-tab='issues']")
            issues_cards = page.locator(".issue-card").count()
            if issues_cards == 0:
                failures.append(f"{width}x{height}: issues panel has no cards")
            badge_text = page.locator("#issues-badge").text_content()
            if not badge_text or badge_text.strip() == "0":
                failures.append(f"{width}x{height}: issues badge did not increment")
            page.screenshot(path=str(SCREENSHOT_DIR / f"issues_{width}x{height}.png"))

            page.click("[data-tab='budget']")
            budget_text = page.locator("#budget-summary").text_content()
            if not budget_text or "Total" not in budget_text:
                failures.append(f"{width}x{height}: budget panel did not populate")
            page.screenshot(path=str(SCREENSHOT_DIR / f"budget_{width}x{height}.png"))

            page.click("[data-tab='drafts']")
            draft_blocks = page.locator(".draft-block").count()
            if draft_blocks < 3:
                failures.append(f"{width}x{height}: expected 3 draft blocks, found {draft_blocks}")
            citation_chips = page.locator(".citation").count()
            if citation_chips == 0:
                failures.append(f"{width}x{height}: no citation chips rendered in drafts")
            page.screenshot(path=str(SCREENSHOT_DIR / f"drafts_{width}x{height}.png"))

            page.click("[data-tab='scores']")
            score_rows = page.locator(".score-bar-row").count()
            if score_rows < 3:
                failures.append(f"{width}x{height}: expected >=3 score bars, found {score_rows}")
            page.screenshot(path=str(SCREENSHOT_DIR / f"scores_{width}x{height}.png"))

            # citation chip opens the source drawer
            page.click("[data-tab='drafts']")
            page.locator(".citation").first.click()
            page.wait_for_selector("#source-drawer:not([hidden])", timeout=5000)
            drawer_text = page.locator("#source-drawer-body").text_content()
            if not drawer_text or "84" not in drawer_text:
                failures.append(f"{width}x{height}: source drawer did not load the cited document's text")
            page.screenshot(path=str(SCREENSHOT_DIR / f"drawer_{width}x{height}.png"))
            page.click("#source-drawer-close")

            # keyboard shortcuts: 1-5 switch tabs, T toggles reasoning, ? opens help
            page.click("body")
            page.keyboard.press("2")
            if page.locator("#panel-issues:not([hidden])").count() == 0:
                failures.append(f"{width}x{height}: '2' did not switch to the issues tab")
            page.keyboard.press("4")
            if page.locator("#panel-drafts:not([hidden])").count() == 0:
                failures.append(f"{width}x{height}: '4' did not switch to the drafts tab")

            reasoning_before = page.is_checked("#show-reasoning")
            page.keyboard.press("t")
            if page.is_checked("#show-reasoning") == reasoning_before:
                failures.append(f"{width}x{height}: 't' did not toggle the reasoning checkbox")
            page.keyboard.press("t")

            page.keyboard.press("?")
            if page.locator("#shortcut-help:not([hidden])").count() == 0:
                failures.append(f"{width}x{height}: '?' did not open shortcut help")
            page.keyboard.press("?")

            page.close()

        browser.close()

    return failures


def run_network_disabled_check() -> list[str]:
    from playwright.sync_api import sync_playwright

    failures: list[str] = []
    blocked: list[str] = []
    base_url = f"http://127.0.0.1:{APP_PORT}/"

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1920, "height": 1080})

        def handle_route(route):
            url = route.request.url
            host = url.split("//", 1)[-1].split("/", 1)[0].split(":")[0]
            if host not in ("127.0.0.1", "localhost"):
                blocked.append(url)
                route.abort()
            else:
                route.continue_()

        page.route("**/*", handle_route)
        page.goto(base_url)
        page.wait_for_load_state("networkidle")

        header_visible = page.locator("#header").is_visible()
        if not header_visible:
            failures.append("page did not render with networking disabled for non-localhost hosts")
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(SCREENSHOT_DIR / "network_disabled.png"), full_page=True)

        browser.close()

    if blocked:
        failures.append(f"{len(blocked)} non-localhost request(s) were blocked (should be zero): {blocked}")

    return failures


def run_replay_check(fake_app) -> list[str]:
    """Phase 9 acceptance: record a replay from a fake-server run, then play
    it back and confirm the REPLAY banner stays visible throughout."""
    import httpx
    from playwright.sync_api import sync_playwright

    failures: list[str] = []
    base = f"http://127.0.0.1:{APP_PORT}"
    replay_name = "phase9-verification-check"

    fake_app.state.queue.extend(build_scripted_queue())
    resp = httpx.post(f"{base}/api/runs", json={"mode": "story", "scope": "full", "show_thinking": True}, timeout=10)
    run_id = resp.json()["run_id"]

    for _ in range(100):
        time.sleep(0.2)
        state = httpx.get(f"{base}/api/runs/{run_id}/state", timeout=10).json()
        if state["status"] != "running":
            break
    else:
        failures.append("replay source run did not finish in time")
        return failures

    httpx.post(f"{base}/api/runs/{run_id}/save-replay", json={"name": replay_name}, timeout=10)
    replay_path = REPO_ROOT / "data" / "replays" / f"{replay_name}.jsonl"
    if not replay_path.is_file():
        failures.append("save-replay did not write a replay file")
        return failures

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            page.goto(f"{base}/")
            page.select_option("#mode-select", "replay")
            # <option> elements never satisfy Playwright's default "visible" actionability
            # check (they're rendered by the OS's native dropdown, not the layout tree).
            page.wait_for_selector(f"#replay-select option[value='{replay_name}']", state="attached", timeout=10_000)
            page.select_option("#replay-select", replay_name)
            page.click("#btn-start")

            # playReplay() is async (POST /api/replays/.../play, then connect SSE);
            # give that a moment to land before asserting the banner stays up.
            try:
                page.wait_for_selector("#replay-banner:not([hidden])", timeout=10_000)
                saw_banner_early = True
            except Exception:
                saw_banner_early = False
                failures.append("REPLAY banner was not visible early in playback")

            saw_banner_late = saw_banner_early
            for _ in range(60):
                if page.locator("#run-summary-overlay:not([hidden])").count() > 0:
                    break
                visible = page.locator("#replay-banner:not([hidden])").count() > 0
                if not visible:
                    failures.append("REPLAY banner was hidden mid-playback")
                    saw_banner_late = False
                    break
                saw_banner_late = True
                time.sleep(0.2)

            if not saw_banner_late:
                failures.append("REPLAY banner was not visible at the end of playback")

            page.screenshot(path=str(SCREENSHOT_DIR / "replay_banner.png"), full_page=True)
            browser.close()
    finally:
        replay_path.unlink(missing_ok=True)  # verification artifact only; Phase 10 commits the real one

    return failures


def main() -> int:
    print("Starting fake LLM server...")
    _, fake_app = start_fake_llm_server()
    print("Starting app server...")
    start_app_server()
    time.sleep(0.5)

    print("Running browser checks (1920x1080, 1366x768)...")
    failures = run_browser_checks(fake_app)

    print("Running network-disabled check...")
    failures += run_network_disabled_check()

    print("Running replay check (record + play back, REPLAY banner)...")
    failures += run_replay_check(fake_app)

    if failures:
        print(f"\nUI VERIFICATION FAILED ({len(failures)} issue(s)):", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"\nUI verification OK. Screenshots in {SCREENSHOT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
