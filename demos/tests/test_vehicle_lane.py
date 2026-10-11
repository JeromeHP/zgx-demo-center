"""Vehicle Inspection's lane: routing, manifest, simulated VINs, replay freshness,
and the runner itself driven end to end with a fake model.

The invariant everything here protects: a vehicle is released without a human
only when the model affirmatively said there is no damage.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "automotive-vehicle-inspection"
sys.path.insert(0, str(DEMO))

from backend import inspection, lane, vin  # noqa: E402

VEHICLES = json.loads((DEMO / "sample-images" / "vehicles.json").read_text(encoding="utf-8"))["vehicles"]
COMPOSE = yaml.safe_load((DEMO / "docker-compose.yml").read_text(encoding="utf-8"))
MODEL = dict(e.split("=", 1) for e in next(iter(COMPOSE["services"].values()))["environment"])["OLLAMA_MODEL"]


# -- routing -----------------------------------------------------------------

@pytest.mark.parametrize("sev,lane_", [("NONE", "cleared"), ("MINOR", "review"), ("MODERATE", "review"),
                                       ("MAJOR", "hold"), ("CRITICAL", "hold")])
def test_route_table(sev, lane_):
    assert inspection.route(sev, True)[0] == lane_


@pytest.mark.parametrize("text", ["", "I cannot tell.", "DAMAGE_SEVERITY: unclear\nDISPOSITION: RELEASE",
                                  "VEHICLE_TYPE: sedan"])
def test_unreadable_output_never_auto_clears(text):
    triage, lane_, reason = lane.triage_from_text(text, 1.0, None)
    assert lane_ == "review"
    assert not triage["severity_explicit"]
    assert "incomplete" in reason


def test_self_contradicting_answer_goes_to_a_human_and_says_why():
    """Real output from a clean Alfa Stelvio on the GB10."""
    text = ("DAMAGE_SEVERITY: MINOR\nAFFECTED_PANELS: NONE\n"
            "DAMAGE_OBSERVATIONS: No exterior damage observed\nDISPOSITION: RELEASE")
    _, lane_, reason = lane.triage_from_text(text, 1.0, None)
    assert lane_ == "review" and reason.startswith("Inconsistent answer")


def test_ambiguous_severity_biases_toward_damage():
    parsed = inspection.parse_vlm_output("DAMAGE_SEVERITY: NONE or MINOR")
    assert inspection.classify_severity_checked(parsed) == ("MINOR", True)


def test_markdown_wrapped_labels_still_parse():
    parsed = inspection.parse_vlm_output("**DAMAGE_SEVERITY:** NONE\n- DISPOSITION: RELEASE")
    assert parsed["damage_severity"] == "NONE" and parsed["disposition"] == "RELEASE"


# -- prompts -----------------------------------------------------------------

ANGLES = ["front_left", "rear_right", "left"]


def test_every_prompt_embeds_the_one_rubric():
    assert inspection.RUBRIC in inspection.INSPECTION_PROMPT
    assert inspection.RUBRIC in inspection.triage_prompt(ANGLES)
    assert inspection.RUBRIC in inspection.report_prompt(ANGLES)


def test_spot_check_is_grounded_like_the_lane():
    """Spot Check writes what it sees (PHOTO_1) before it grades, and is told
    undamaged is the normal case -- without that, glossy clean cars drew
    invented scuffs and cracked headlights. It stays a one-photo prompt."""
    p = inspection.INSPECTION_PROMPT
    assert "Most vehicles arrive with no damage" in p
    out = p.split("OUTPUT FORMAT", 1)[1]
    assert out.index("PHOTO_1:") < out.index("DAMAGE_SEVERITY:")
    assert "PHOTO_2:" not in out and "DAMAGE_VIEWS" not in out


def test_detailed_report_is_grounded_per_camera():
    p = inspection.report_prompt(ANGLES)
    assert "Most vehicles arrive with no damage" in p
    out = p.split("OUTPUT FORMAT", 1)[1]
    assert out.index("PHOTO_3:") < out.index("DAMAGE_SEVERITY:") and "PHOTO_4:" not in out
    assert "DAMAGE_VIEWS:" in out and "INSPECTOR_SUMMARY:" in out


def test_triage_names_every_camera_and_asks_one_line_per_photo():
    p = inspection.triage_prompt(ANGLES)
    assert "Photo 1: FRONT-LEFT camera" in p and "Photo 3: LEFT SIDE camera" in p
    out = p.split("OUTPUT FORMAT", 1)[1]
    for field in ("PHOTO_1:", "PHOTO_2:", "PHOTO_3:", "DAMAGE_SEVERITY:", "DAMAGE_VIEWS:", "DISPOSITION:"):
        assert field in out
    assert "PHOTO_4:" not in out
    assert "INSPECTOR_SUMMARY" not in out, "the summary is what made the full report 3-4x slower"


def test_triage_does_not_prime_the_model_to_find_damage():
    """The first draft said "damage in any one photo counts" and showed an
    example DAMAGE_VIEWS of "1, 3"; every clean probe car came back MAJOR."""
    p = inspection.triage_prompt(ANGLES)
    assert "Most vehicles arrive with no damage" in p
    assert '"1, 3"' not in p


def test_damage_views_map_to_cameras_and_ignore_nonsense():
    parse = inspection.parse_damage_views
    assert parse({"damage_views": "Photo 3, 1, 9"}, ANGLES) == ["left", "front_left"]
    assert parse({"damage_views": "NONE"}, ANGLES) == []
    assert parse({}, ANGLES) == []


def test_photo_findings_map_to_cameras():
    text = "PHOTO_1: No damage\n**PHOTO_2:** dented rear door\nPHOTO_7: ghost\nDAMAGE_SEVERITY: MODERATE"
    assert inspection.parse_photo_findings(text, ANGLES) == {"front_left": "No damage", "rear_right": "dented rear door"}


# -- VINs ----------------------------------------------------------------------

def test_vin_check_digit_matches_a_known_real_vin():
    assert vin.is_valid("1M8GDM9AXKP042788")


@pytest.mark.parametrize("brand", sorted(vin.WMI))
def test_simulated_vins_are_valid_and_branded(brand):
    rng = random.Random(1)
    for year in (1968, 1995, 2019, 2025):
        v = vin.simulated_vin(brand, year, rng)
        assert vin.is_valid(v) and v.startswith(vin.WMI[brand])
        assert not set("IOQ") & set(v)


def test_simulated_vins_are_reproducible_per_seed():
    a = [vin.simulated_vin("Jeep", 2025, random.Random(5)) for _ in range(2)]
    assert a[0] == a[1]


# -- manifest ------------------------------------------------------------------
# The user's rule for every image: openly licensed and publicly on the web.
OPEN_LICENCE = re.compile(r"^(CC0( 1\.0)?|Public domain|Public Domain Mark( 1\.0)?|CC BY(-SA)? \d\.\d)$")
VIEWS = [(v, w) for v in VEHICLES for w in v["views"]]


def test_fourteen_stellantis_brands_and_nothing_else():
    brands = {v["brand"] for v in VEHICLES}
    assert brands <= lane.STELLANTIS_BRANDS
    assert len(lane.STELLANTIS_BRANDS) == 14 and brands == lane.STELLANTIS_BRANDS


@pytest.mark.parametrize("v", VEHICLES, ids=lambda v: v["id"])
def test_every_vehicle_is_a_real_multi_angle_set(v):
    angles = [w["angle"] for w in v["views"]]
    assert 2 <= len(angles) <= 4, "the portal shows 2-4 cameras per vehicle"
    assert len(set(angles)) == len(angles) and set(angles) <= set(inspection.ANGLES)
    assert all(w["file"].startswith(v["id"] + "/") for w in v["views"]), "one folder per vehicle"
    assert v["group"] in ("clean", "damage") and v["consignment"] in ("new", "used", "classic")
    assert v.get("vehicle") and v.get("model_year")


@pytest.mark.parametrize("vw", VIEWS, ids=lambda vw: vw[1]["file"])
def test_every_view_exists_is_open_and_attributed(vw):
    _, w = vw
    assert (DEMO / "sample-images" / w["file"]).is_file()
    assert w.get("author"), "CC BY / BY-SA require attribution"
    assert OPEN_LICENCE.match(w["license"]), f"not an allowed open licence: {w['license']!r} (no NC, no ND)"
    assert re.match(r"^https?://", w.get("source") or ""), "every image must be publicly available on the web"


def test_vessel_is_mostly_clean_with_a_real_spread_of_damage():
    damaged = [v for v in VEHICLES if v["group"] == "damage"]
    assert len([v for v in VEHICLES if v["group"] == "clean"]) > len(damaged)
    assert len(damaged) >= 10 and len({v["brand"] for v in damaged}) >= 6


def test_chargers_and_challengers_new_and_classic():
    names = [(v["vehicle"], v["consignment"]) for v in VEHICLES]
    for model in ("Charger", "Challenger"):
        assert any(model in n and c == "new" for n, c in names), model
    assert any("Charger" in n and c == "classic" for n, c in names)


def test_live_draw_size_and_mix():
    for seed in range(500):
        picked = lane.draw_live(VEHICLES, random.Random(seed))
        assert len(picked) == lane.LIVE_SIZE and len({p["id"] for p in picked}) == lane.LIVE_SIZE
        lanes = Counter(p.get("observed_lane") for p in picked)
        for name, n in lane.LIVE_MIX.items():
            available = sum(1 for s in VEHICLES if s.get("observed_lane") == name)
            assert lanes[name] >= min(n, available), (seed, name, lanes)
        early = {p.get("observed_lane") for p in picked[:5]}
        for name in ("hold", "review"):
            if any(s.get("observed_lane") == name for s in VEHICLES):
                assert name in early, "the audience should see routing in the first five"


# -- replay freshness ------------------------------------------------------------

def _replay():
    path = lane.latest_replay(DEMO / "data" / "replays")
    assert path, "no recorded vessel: run scripts/record_vessel_replay.py on the device"
    return lane.read_replay(path)


def test_replay_matches_the_current_prompt_and_model():
    header, _ = _replay()
    assert header["model"] == MODEL
    assert header["prompt_hash"] == inspection.prompt_hash(MODEL), (
        "the triage prompt changed since the replay was recorded; re-run scripts/record_vessel_replay.py")


def test_replay_covers_every_vehicle_and_every_view():
    header, rows = _replay()
    assert {r["id"] for r in rows} == {v["id"] for v in VEHICLES}
    assert set(header["photos"]) == {w["file"] for _, w in VIEWS}
    for f, digest in header["photos"].items():
        assert hashlib.sha256((DEMO / "sample-images" / f).read_bytes()).hexdigest()[:16] == digest, f


def test_observed_lanes_come_from_the_current_recording():
    header, _ = _replay()
    stamp = f"{MODEL} · {header['recorded_at'][:10]}"
    for v in VEHICLES:
        assert v.get("observed_lane") in ("cleared", "review", "hold", "unstable"), v["id"]
        assert v.get("observed_with") == stamp, v["id"]


def test_no_damaged_vehicle_auto_clears_in_the_recording():
    """Calibration gate: the recording is what replay shows and what the live
    mix is drawn from, so a damaged car that cleared would be on stage."""
    for v in VEHICLES:
        if v["group"] == "damage":
            assert v.get("observed_lane") in ("hold", "review"), v["id"]


# -- the runner, with a fake model ---------------------------------------------------

def _fake_samples(tmp: Path) -> Path:
    d = tmp / "samples"
    d.mkdir()
    def veh(i, brand, lane_, group):
        return {"id": i, "brand": brand, "vehicle": f"{brand} {i}", "model_year": 2025, "group": group,
                "consignment": "new", "observed_lane": lane_,
                "views": [{"file": f"{i}/front_left.jpg", "angle": "front_left", "author": "x", "license": "CC BY 4.0", "source": "https://s"},
                          {"file": f"{i}/rear_right.jpg", "angle": "rear_right", "author": "x", "license": "CC BY 4.0", "source": "https://s"}]}
    rows = [veh("a", "Jeep", "cleared", "clean"), veh("b", "Fiat", "review", "clean"),
            veh("c", "Peugeot", "hold", "damage"), veh("d", "Ram", "cleared", "clean")]
    (d / "vehicles.json").write_text(json.dumps({"vehicles": rows}))
    return d


TEXTS = {
    "a": "PHOTO_1: No damage\nPHOTO_2: No damage\nDAMAGE_SEVERITY: NONE\nAFFECTED_PANELS: NONE\nDAMAGE_OBSERVATIONS: No exterior damage observed\nDAMAGE_VIEWS: NONE\nDISPOSITION: RELEASE",
    "b": "PHOTO_1: Light scuff on bumper\nPHOTO_2: No damage\nDAMAGE_SEVERITY: MINOR\nAFFECTED_PANELS: front bumper\nDAMAGE_OBSERVATIONS: Light scuff\nDAMAGE_VIEWS: 1\nDISPOSITION: TOUCH_UP",
    "c": "PHOTO_1: Crumpled hood\nPHOTO_2: Rear intact\nDAMAGE_SEVERITY: CRITICAL\nAFFECTED_PANELS: hood\nDAMAGE_OBSERVATIONS: Crumpled hood\nDAMAGE_VIEWS: 1\nDISPOSITION: REJECT",
    "d": "garbled",
}
CALLS = []


async def _fake_infer(views, port, on_token):
    CALLS.append([a for _, a in views])
    text = TEXTS[views[0][0].parent.name]
    for word in text.split(" "):
        await on_token(word + " ")
    return text, 0.01, {"completion_tokens": 9}


def test_live_run_routes_decides_and_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "PORTAL_GAP_S", 0)
    monkeypatch.setattr(lane, "LIVE_SIZE", 4)
    monkeypatch.setattr(lane, "LIVE_MIX", {"cleared": 2, "review": 1, "hold": 1})

    async def go():
        mgr = lane.LaneManager(_fake_samples(tmp_path), tmp_path / "replays", _fake_infer, _fake_infer)
        q = mgr.subscribe()
        await mgr.start("baltimore", "", "live")
        await asyncio.wait_for(mgr._task, 5)
        events = [q.get_nowait()["type"] for _ in range(q.qsize())]
        by_file = {v.sample["id"]: v for v in mgr.run.vehicles}
        assert [by_file[f].lane for f in "abcd"] == ["cleared", "review", "hold", "review"]
        assert all(c == ["front_left", "rear_right"] for c in CALLS), "every camera goes in the one call"
        assert by_file["c"].triage["damage_views"] == ["front_left"]
        assert by_file["c"].triage["photo_findings"]["front_left"] == "Crumpled hood"
        assert mgr.run.status == "finished" and "token" in events and "run_finished" in events
        st = mgr.stats()
        assert (st["cleared"], st["review"], st["hold"], st["awaiting_human"]) == (1, 2, 1, 3)
        await mgr.decide(by_file["b"].vid, "RELEASE", "reflection, not a scuff")
        await mgr.decide(by_file["c"].vid, "REJECT")
        st = mgr.stats()
        assert (st["decided"], st["agree"], st["override"], st["awaiting_human"]) == (2, 1, 1, 1)
        csv = mgr.csv()
        assert "SIMULATED" in csv and "reflection, not a scuff" in csv
        with pytest.raises(ValueError):
            await mgr.decide(by_file["a"].vid, "TELEPORT")
        await mgr.request_full_report(by_file["c"].vid)
        await asyncio.wait_for(mgr._report_tasks[by_file["c"].vid], 5)
        assert by_file["c"].full_report["status"] == "done"

    asyncio.run(go())


def test_stop_cancels_the_in_flight_vehicle(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "LIVE_SIZE", 4)

    async def slow(path, port, on_token):
        await asyncio.sleep(30)
        return "", 0, {}

    async def go():
        mgr = lane.LaneManager(_fake_samples(tmp_path), tmp_path / "r", slow, slow)
        await mgr.start("baltimore", "", "live")
        await asyncio.sleep(0.05)
        await mgr.stop()
        assert mgr.run.status == "stopped"
        assert not any(v.state == "scanning" for v in mgr.run.vehicles)

    asyncio.run(go())


def test_replay_routes_through_the_same_code_and_never_calls_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "PORTAL_GAP_S", 0)
    monkeypatch.setattr(lane, "REPLAY_SPEED", 1000)
    samples = _fake_samples(tmp_path)
    rdir = tmp_path / "replays"
    rdir.mkdir()
    lines = [json.dumps({"type": "header", "recorded_at": "2026-10-06T00:00:00+00:00", "device": "test",
                         "model": "m", "model_label": "M", "prompt_hash": "x", "port": "brunswick"})]
    lines += [json.dumps({"type": "vehicle", "id": f, "text": t, "seconds": 4.0, "usage": {}}) for f, t in TEXTS.items()]
    (rdir / "vessel_20261006.jsonl").write_text("\n".join(lines))

    async def boom(*a):
        raise AssertionError("replay must not call the model")

    async def go():
        mgr = lane.LaneManager(samples, rdir, boom, boom)
        await mgr.start("baltimore", "", "replay")
        await asyncio.wait_for(mgr._task, 5)
        assert mgr.run.port == "brunswick" and mgr.run.replay["device"] == "test"
        assert [v.lane for v in mgr.run.vehicles] == ["cleared", "review", "hold", "review"]
        assert mgr.stats()["avg_s"] == 4.0, "stats show recorded model time, not playback time"

    asyncio.run(go())
