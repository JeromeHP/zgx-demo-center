"""The UI is plain JS against a static document, so a renamed element fails at
runtime in the browser and nowhere else.

This caught a real one: `pill-app-conn-text` outlived the element it referenced when
Discover replaced that pill, throwing a TypeError on every network_status event — a
timer that fires for the whole run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"
JS = (WEB / "app.js").read_text(encoding="utf-8")
HTML = (WEB / "index.html").read_text(encoding="utf-8")

_HTML_IDS = set(re.findall(r'id="([^"]+)"', HTML))


def test_every_element_the_script_reaches_for_exists():
    referenced = set(re.findall(r'\$\$?\("#([a-zA-Z0-9_-]+)"\)', JS))
    assert referenced, "sanity: the selector pattern should match something"
    assert sorted(r for r in referenced if r not in _HTML_IDS) == []


def test_every_inline_handler_names_a_real_function():
    handlers = set(re.findall(r'on\w+="(\w+)\(', HTML))
    defined = set(re.findall(r"function (\w+)\(", JS))
    assert sorted(h for h in handlers if h not in defined) == []


@pytest.mark.parametrize("mode", ["draft", "discover", "grade", "replay"])
def test_every_mode_in_the_selector_has_its_controls(mode):
    assert f'<option value="{mode}">' in HTML
    if mode in ("draft", "discover", "grade", "replay"):
        assert f'id="{mode}-group"' in HTML, f"{mode} has no control group to show"


def test_every_output_tab_has_a_panel():
    tabs = set(re.findall(r'data-tab="([a-z]+)"', HTML))
    panels = set(re.findall(r'id="panel-([a-z]+)"', HTML))
    assert tabs and tabs == panels, f"tabs {tabs ^ panels} have no matching panel"


def test_the_ui_makes_no_external_requests():
    """Restated here as a test, not only in the portability audit: the UI must render
    with the network unplugged."""
    urls = [u for u in re.findall(r'https?://[^\s"\'<>)]+', HTML + JS)
            if not any(local in u for local in ("localhost", "127.0.0.1"))]
    assert urls == [], f"external URL in the UI: {urls}"


# -- the page must not scroll; the columns must ---------------------------------------
# The three columns declared `overflow-y: auto` with no bounded height, so none of them
# ever scrolled -- the document grew instead. Reported as two separate bugs: the token
# counter could not be reached, and every new trace card dragged the viewport back down,
# because scrollIntoView() scrolls every scrollable ancestor including the window.

CSS = (WEB / "styles.css").read_text(encoding="utf-8")


def test_the_columns_are_bounded_so_they_can_scroll():
    assert "min-height: 0" in CSS, "grid/flex children need min-height:0 or they never shrink"
    assert re.search(r"body\s*\{[^}]*overflow:\s*hidden", CSS, re.S), \
        "the page itself must not scroll on the desktop layout"


def test_the_trace_scrolls_itself_not_the_window():
    assert ".scrollIntoView(" not in JS, (
        "scrollIntoView scrolls every scrollable ancestor, the window included -- "
        "use stickTraceToBottom(), which moves only the trace column"
    )
    assert "function stickTraceToBottom" in JS


def test_autoscroll_is_conditional_on_the_reader_being_at_the_bottom():
    # Every call site must be guarded, or scrolling away from the live tail is futile.
    for call in re.finditer(r"(?<!function )\bstickTraceToBottom\(\)", JS):
        line_start = JS.rfind("\n", 0, call.start())
        context = JS[max(0, line_start - 400):call.end()]
        assert "traceAutoScroll" in context or "jump-to-live" in context, \
            f"unguarded stickTraceToBottom near: {JS[line_start:call.end()][:120]!r}"


def test_the_retired_mode_names_are_gone_from_the_ui():
    """Draft replaced Story and Explore; Discover is shown as Find Funding. Comments
    may still say where a name went, but no option, label or visible string may."""
    visible = re.sub(r"<!--.*?-->", "", HTML, flags=re.S)
    visible = re.sub(r"<[^>]+>", " ", visible)
    for word in ("Story", "Explore", "Discover"):
        assert not re.search(rf"\b{word}\b", visible), f"{word!r} is still visible in index.html"
    strings = re.findall(r'"[^"\n]*"|`[^`]*`', re.sub(r"//[^\n]*", "", JS))
    leaked = [s for s in strings if re.search(r"\b(Story|Explore|Discover)\b", s)]
    assert leaked == [], leaked
