"""The launcher must not assert hardware facts it has not checked.

Every string tested here was, until 2026-09-21, hardcoded HTML in the system
bar and the per-demo requirement rows: "NVIDIA CTK ✓ v1.18.2", "GB10 · sm_121",
"128GB", "Docker v29.1.3", "3.3TB / 3.6TB". No JavaScript ever wrote to those
cells, and setReqLine() returned early on data-stable="true", so they were
never re-evaluated either. Three were already false on the machine they were
describing; on an AMD box every one of them would have been.

These tests are the reason that cannot come back quietly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "zgx-demo-launcher.html"


@pytest.fixture(scope="module")
def html() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def markup(html: str) -> str:
    """The HTML with <script> bodies and <!-- comments --> stripped.

    Both must go. Script bodies, because the JS legitimately mentions these
    strings while parsing them away. Comments, because this fix's own
    explanation quotes every banned value verbatim -- and a test that matches
    its own commentary passes whether or not the markup was ever fixed. That
    is precisely how an earlier UI regression test in this repo came to assert
    nothing, so the fixture is the thing that has to be careful here.

    Neither is rendered, so neither can mislead a presenter.
    """
    without_scripts = re.sub(r"<script\b.*?</script>", "", html, flags=re.S | re.I)
    return re.sub(r"<!--.*?-->", "", without_scripts, flags=re.S)


# Hardware values the page must never state on its own authority.
ASSERTED_FACTS = [
    "v1.18.2",          # NVIDIA Container Toolkit -- actually 1.20.0 on this box
    "v29.1.3",          # Docker -- actually 29.2.1
    "v5.0.1",           # Compose -- actually 5.0.2
    "sm_121",
    "128GB VRAM",
    "3.3TB / 3.6TB",
]


@pytest.mark.parametrize("claim", ASSERTED_FACTS)
def test_no_hardcoded_hardware_claim_in_markup(markup, claim):
    assert claim not in markup, (
        f"{claim!r} is asserted in the launcher markup. Hardware facts must come "
        f"from GET /sysinfo, or they will be wrong on the other vendor's box -- "
        f"and silently wrong on this one."
    )


def test_system_bar_cells_are_filled_from_sysinfo(markup, html):
    for cell in ("sys-docker", "sys-compose", "sys-runtime", "sys-gpu",
                 "sys-mem", "sys-disk", "sys-arch", "sys-driver"):
        assert f'id="{cell}"' in markup, f"system bar is missing #{cell}"
        assert f"'{cell}'" in html or f'"{cell}"' in html, \
            f"#{cell} exists in markup but no JS ever writes to it"


def test_sysinfo_is_actually_fetched_and_called_on_load(html):
    assert "/sysinfo" in html, "the launcher never asks the server what hardware it is on"
    assert re.search(r"DOMContentLoaded[\s\S]{0,400}loadSysinfo\(\)", html), \
        "loadSysinfo() is defined but never called at startup"


def test_hardware_requirement_rows_are_no_longer_stable(markup):
    """data-stable='true' is what made setReqLine() skip these rows forever."""
    for row in re.findall(r'<div class="rline[^>]*>.*?</div>', markup):
        if "hw-mem" in row or "hw-runtime" in row or "hw-gpu" in row:
            assert 'data-stable="true"' not in row, (
                "a row carrying detected hardware is still marked data-stable, "
                "so it will never be re-evaluated:\n  " + row[:160]
            )


def test_brand_and_title_are_not_hardcoded_to_one_machine(markup):
    assert "<title>ZGX Demo Center</title>" in markup, \
        "the page title still names a specific machine"
    assert 'id="brand-device"' in markup and 'id="pill-device"' in markup


def test_every_hw_placeholder_has_a_filler(html):
    """A span nothing writes to renders a literal em-dash at a show."""
    placeholders = set(re.findall(r'class="(hw-[a-z]+)"', html))
    assert placeholders, "no hardware placeholders found at all"
    for cls in placeholders:
        assert f"'.{cls}'" in html or f'".{cls}"' in html, \
            f".{cls} is used in markup but applyHardwareToCards() never fills it"
