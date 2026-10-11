"""
Inspection logic shared by the single-photo Spot Check and the vessel lane.

Pure functions and constants only -- no FastAPI, no HTTP -- so the tests import
this module directly and the lane runner (lane.py) can be exercised without a
model server.

Two prompts, one rubric:

  INSPECTION_PROMPT  Spot Check: one photo, a grounding PHOTO_1 line, then the
                     full eight-field report (~9-16 s on the GB10).
  triage_prompt()    the lane: every portal view of ONE vehicle in a single
                     call, five short lines back (severity, panels, one
                     finding, which views show damage, disposition). The
                     answer is what costs time on the GB10, not the images:
                     4 views measured 5.7 s at 768 px against ~4.5 s for 1.
  report_prompt()    the full report over all views, on demand, for an
                     inspector looking at a flagged vehicle.

All three embed RUBRIC verbatim, so the severity tiers and the "do not
downgrade" rules cannot drift apart between the fast pass and the detailed
one. A test pins that.
"""

import base64
import hashlib
import io
import logging
import re

from PIL import Image

logger = logging.getLogger("vehicle-inspection")

# -- Ports of entry ----------------------------------------------------------
# Major US RoRo (roll-on / roll-off) terminals for finished-vehicle imports.
# Lean structure -- just enough context to make the report look real.
PORTS = {
    "baltimore": {
        "name": "Port of Baltimore",
        "code": "USBAL",
        "terminal": "Atlantic Terminal (Tradepoint)",
        "state": "Maryland",
    },
    "brunswick": {
        "name": "Port of Brunswick",
        "code": "USBQK",
        "terminal": "Colonel's Island Terminal",
        "state": "Georgia",
    },
    "long_beach": {
        "name": "Port of Long Beach",
        "code": "USLGB",
        "terminal": "Pier T RoRo Terminal",
        "state": "California",
    },
    "jacksonville": {
        "name": "Port of Jacksonville (JAXPORT)",
        "code": "USJAX",
        "terminal": "Blount Island Marine Terminal",
        "state": "Florida",
    },
}

# -- Damage severity tiers ---------------------------------------------------
SEVERITY_TIERS = {
    "NONE": {
        "label": "NO DAMAGE OBSERVED",
        "color_hex": "#3DBD5D",
        "description": "Vehicle exterior is in acceptable condition; no inspector action required.",
    },
    "MINOR": {
        "label": "MINOR COSMETIC",
        "color_hex": "#95C73E",
        "description": "Surface-level scratches, light scuffs, or paint transfer. Touch-up bay can resolve.",
    },
    "MODERATE": {
        "label": "MODERATE DAMAGE",
        "color_hex": "#E8A317",
        "description": "Paint chips, small dents, or edge rash beyond touch-up scope. Body shop repair likely.",
    },
    "MAJOR": {
        "label": "MAJOR DAMAGE",
        "color_hex": "#E66B2A",
        "description": "Panel damage, broken lighting assemblies, large dents, or deep gouges. Significant repair required.",
    },
    "CRITICAL": {
        "label": "CRITICAL DAMAGE",
        "color_hex": "#D94040",
        "description": "Structural concerns, broken glass, cracked frame elements, or multi-panel damage. Vehicle should not be accepted.",
    },
}

# -- Disposition playbook ----------------------------------------------------
# Two concrete actions per disposition -- enough to read as a real
# terminal-ops playbook without overbuilding.
DISPOSITION_PLAYBOOK = {
    "RELEASE": {
        "label": "RELEASE TO OUTBOUND YARD",
        "actions": [
            "Release vehicle to outbound yard / dealer transport queue",
            "No exception report required; routine acceptance",
        ],
    },
    "TOUCH_UP": {
        "label": "ROUTE TO TOUCH-UP BAY",
        "actions": [
            "Route to on-site touch-up bay for cosmetic correction",
            "Log entry in daily exception report; release post-repair",
        ],
    },
    "BODY_SHOP_HOLD": {
        "label": "HOLD FOR BODY SHOP",
        "actions": [
            "Hold in damaged-vehicle holding area pending body shop estimate",
            "Notify receiving dealer of estimated delivery delay (5-10 business days)",
        ],
    },
    "CARRIER_CLAIM": {
        "label": "FILE CARRIER DAMAGE CLAIM",
        "actions": [
            "Document damage with photographs and panel diagram for carrier claim",
            "Hold pending claim settlement; coordinate with carrier loss adjuster",
        ],
    },
    "REJECT": {
        "label": "REJECT ACCEPTANCE",
        "actions": [
            "Reject acceptance; initiate return-to-origin (RTO) process",
            "Notify OEM logistics, quality assurance, and receiving dealer",
        ],
    },
}

# -- Prompts -----------------------------------------------------------------
# The persona is a port inspector; the tiers and disposition options are
# spelled out so the model maps observations onto canonical categories.
_PERSONA = (
    "You are a senior vehicle damage inspector at {port_name} ({port_code}), {terminal}. "
    "You assess finished vehicles offloaded from RoRo (roll-on/roll-off) carriers and triage "
    "exterior damage so terminal operations can release, repair, or reject each unit."
)

RUBRIC = """DAMAGE SEVERITY TIERS
- NONE: No visible exterior damage
- MINOR: Surface-level scratches, light scuffs, paint transfer; touch-up bay can resolve
- MODERATE: Paint chips, small dents, edge rash, scuffed bumpers; body shop repair likely
- MAJOR: Panel damage, ANY cracked or broken lighting assembly (headlight, taillight, fog light, turn signal lens), large dents, deep gouges, torn trim, damaged grille
- CRITICAL: Visibly crumpled, crushed, or buckled sheet metal (hood, fender, quarter panel) consistent with impact; deformed/misaligned bumper assembly from collision (not surface dent); frontend or rear-end collision damage; cracked windshield or window glass; cracked frame or structural members; missing safety equipment; three or more panels with damage from the same impact event; estimated affected area greater than 25 percent

SEVERITY ASSIGNMENT RULES
- If you observe damage that matches an example in MAJOR or CRITICAL, your DAMAGE_SEVERITY must be at least that tier. Do not downgrade.
- Cracked or fractured headlight/taillight lenses are MAJOR even if the rest of the vehicle is clean. They allow water ingress and require full assembly replacement.
- A crumpled, crushed, or deformed hood, fender, or bumper consistent with impact is CRITICAL, not MAJOR. "Significant denting" or "crumpling" of sheet metal is collision damage, not body shop damage.
- If three or more panels show damage from a single impact event, severity is CRITICAL.
- If estimated affected area is greater than 25 percent, severity is CRITICAL.
- A single severe finding outranks many minor findings. Use the highest tier any single observation reaches.

DISPOSITION OPTIONS
- RELEASE: No damage observed; release to outbound yard
- TOUCH_UP: Minor cosmetic only; route to on-site touch-up bay
- BODY_SHOP_HOLD: Moderate damage; hold for body shop estimate
- CARRIER_CLAIM: Damage attributable to ocean transport or carrier handling; file carrier damage claim
- REJECT: Collision damage, crumpled or crushed sheet metal, or damage where likely repair cost approaches or exceeds vehicle value; vehicle should not be accepted; initiate return-to-origin. CRITICAL severity should always map to REJECT."""

# The eight report fields, shared by Spot Check and the lane's detailed report.
_REPORT_FIELDS = """VEHICLE_TYPE: [Describe what you see as "<make> <model> <body style>, <color>" when identifiable, otherwise e.g. "Partial view of vehicle rear quarter, silver"]
VIEW_ANGLE: [One of: FRONT, REAR, DRIVER_SIDE, PASSENGER_SIDE, FRONT_QUARTER, REAR_QUARTER, CLOSE_UP_PANEL]
DAMAGE_SEVERITY: [One of: NONE, MINOR, MODERATE, MAJOR, CRITICAL]
AFFECTED_PANELS: [Comma-separated list using standard panel terms (hood, front bumper, headlight assembly, grille, driver door, rear bumper, tail light, trunk lid, etc.). If no damage, write NONE.]
DAMAGE_OBSERVATIONS: [Specific damage description per affected area: type (scratch, dent, paint chip, crack, missing trim, paint peeling, etc.), location, approximate size. If no damage, write "No exterior damage observed".]
ESTIMATED_AFFECTED_AREA_PERCENT: [Approximate percentage of visible vehicle surface showing damage. Single number followed by %. Use 0% if none.]
DISPOSITION: [One of: RELEASE, TOUCH_UP, BODY_SHOP_HOLD, CARRIER_CLAIM, REJECT]
INSPECTOR_SUMMARY: [Two to three sentence summary of vehicle condition, damage, and recommended action, written for a terminal operations supervisor. Describe only what you actually observed. Do not append boilerplate negative findings (e.g., "no other damage observed", "no broken glass observed") unless you have specifically inspected for them. If you noted a cracked lighting assembly above, your summary must reflect that as a significant finding, not minimize it.]"""

# Grounding, same idea as the lane's triage (see _VIEWS_PREAMBLE below): say
# that undamaged is the normal case, and make the model write down what it
# actually sees BEFORE it grades. Without it, glossy or dark paint drew
# invented "light scuffs" and "cracked headlight lens" findings on clean cars.
_GROUND_ONE = ("Look at it carefully. Most vehicles arrive with no damage at all: report damage only if you can "
               "actually see it in the photo, and do not assume damage you cannot see.")

INSPECTION_PROMPT = _PERSONA + """

IMAGE
You are looking at one photograph submitted by a yard inspector. It may show a full vehicle, a partial view, or a close-up of a panel or area. """ + _GROUND_ONE + """

""" + RUBRIC + """

OUTPUT FORMAT
Answer each prompt on its own line in exactly this format. First say what you see in the photo, then the report. Be specific to what you actually observe in the image -- do not generalize.

PHOTO_1: [No damage, or a few words on the damage you see in the photo]
""" + _REPORT_FIELDS

# -- Multi-view (portal) prompts ------------------------------------------------
ANGLES = ("front_left", "front_right", "front", "left", "right", "rear_left", "rear_right", "rear")
ANGLE_LABELS = {
    "front_left": "FRONT-LEFT", "front_right": "FRONT-RIGHT", "front": "FRONT",
    "left": "LEFT SIDE", "right": "RIGHT SIDE",
    "rear_left": "REAR-LEFT", "rear_right": "REAR-RIGHT", "rear": "REAR",
}

_VIEWS_PREAMBLE = """IMAGES
You are looking at {n} photographs of the SAME vehicle, captured by the drive-through inspection portal cameras as it came off the carrier:
{cameras}
Look at every photo. Most vehicles arrive with no damage at all: report damage only if you can actually see it in a specific photo, and do not assume damage you cannot see. If you do see damage, your DAMAGE_SEVERITY is the highest tier that any single photo shows. Based on your answer the vehicle is either released automatically or sent to a human inspector."""

# Why the wording above, and the per-photo lines below. The first multi-view
# draft said "damage visible in any one photo counts" and gave an example
# DAMAGE_VIEWS of "1, 3". On the GB10 that turned all 12 clean probe vehicles
# into MAJOR -- every one "Cracked headlight lens", photo 1: the rubric's most
# emphasised example, recited rather than seen. The model could see the photos
# fine (asked plainly, it described each one correctly). Neutral wording plus
# one line per photo BEFORE the verdict fixed it: 8/8 clean NONE, 6/6 damaged
# MAJOR+, same speed. The per-photo lines also give the UI a finding per camera.

_PHOTO_LINE = "PHOTO_{i}: [No damage, or a few words on the damage you see in photo {i}]"

_TRIAGE_VERDICT = """DAMAGE_SEVERITY: [One of: NONE, MINOR, MODERATE, MAJOR, CRITICAL]
AFFECTED_PANELS: [Comma-separated standard panel terms, or NONE]
DAMAGE_OBSERVATIONS: [One short sentence on the most significant finding, or "No exterior damage observed"]
DAMAGE_VIEWS: [Numbers of the photos that show damage, or NONE]
DISPOSITION: [One of: RELEASE, TOUCH_UP, BODY_SHOP_HOLD, CARRIER_CLAIM, REJECT]"""


def _triage_output(n: int) -> str:
    photos = "\n".join(_PHOTO_LINE.format(i=i) for i in range(1, n + 1))
    return ("OUTPUT FORMAT\nAnswer in exactly these lines and nothing else. First one line per photo, "
            "then the verdict. Be specific to what you actually observe.\n\n" + photos + "\n" + _TRIAGE_VERDICT)


_REPORT_FIELDS_MULTI = _REPORT_FIELDS.replace(
    "DAMAGE_OBSERVATIONS:",
    "DAMAGE_VIEWS: [Numbers of the photos that show damage, or NONE]\nDAMAGE_OBSERVATIONS:", 1)


def _cameras(angles: list[str]) -> str:
    return "\n".join(f"Photo {i}: {ANGLE_LABELS.get(a, a.upper())} camera" for i, a in enumerate(angles, 1))


def triage_prompt(angles: list[str]) -> str:
    """The lane's prompt for one vehicle seen from `angles` (portal order)."""
    return (_PERSONA + "\n\n" + _VIEWS_PREAMBLE.format(n=len(angles), cameras=_cameras(angles))
            + "\n\n" + RUBRIC + "\n\n" + _triage_output(len(angles)))


def report_prompt(angles: list[str]) -> str:
    """The full eight-field report over every view, for a flagged vehicle.
    Grounded like triage: one line per photo before the report."""
    photos = "\n".join(_PHOTO_LINE.format(i=i) for i in range(1, len(angles) + 1))
    return (_PERSONA + "\n\n" + _VIEWS_PREAMBLE.format(n=len(angles), cameras=_cameras(angles))
            + "\n\n" + RUBRIC + "\n\nOUTPUT FORMAT\nAnswer each prompt on its own line in exactly this format. "
            "First one line per photo, then the report. Be specific to what you actually observe -- do not generalize.\n\n"
            + photos + "\n" + _REPORT_FIELDS_MULTI)


TRIAGE_MAX_TOKENS = 200
REPORT_MAX_TOKENS = 1024


def prompt_hash(model: str) -> str:
    """Identity of what a recorded replay was produced by. If the triage
    prompt or the model changes, recorded results no longer describe what a
    live run would do, and tests/test_vehicle_lane.py fails until re-recorded."""
    template = (_PERSONA + _VIEWS_PREAMBLE + RUBRIC + _PHOTO_LINE + _TRIAGE_VERDICT + _triage_output(2)
                + repr(sorted(ANGLE_LABELS.items())))
    return hashlib.sha256(f"{model}\n{template}".encode()).hexdigest()[:16]


def format_prompt(template: str, port_key: str) -> str:
    port = PORTS.get(port_key, PORTS["baltimore"])
    return template.format(port_name=port["name"], port_code=port["code"], terminal=port["terminal"])


# -- Output parsing ----------------------------------------------------------

NOT_PROVIDED = "Not provided by model"

PARSED_FIELDS = [
    "vehicle_type",
    "view_angle",
    "damage_severity",
    "affected_panels",
    "damage_views",
    "damage_observations",
    "estimated_affected_area_percent",
    "disposition",
    "inspector_summary",
]


def parse_vlm_output(raw_response: str) -> dict:
    """Parse labeled-line VLM output into a dict.

    Tolerant of minor formatting drift -- matches by label prefix only,
    case-insensitive. Missing fields get explicit "Not provided" sentinels.
    """
    parsed = {field: NOT_PROVIDED for field in PARSED_FIELDS}

    for line in (raw_response or "").strip().split("\n"):
        line = line.strip().lstrip("*-# ").replace("**", "")
        if not line:
            continue
        for field in PARSED_FIELDS:
            tag = field.upper() + ":"
            if line.upper().startswith(tag):
                value = line[len(tag):].strip()
                if value:
                    parsed[field] = value
                break

    logger.info(
        "Parsed VLM output: %s",
        {k: (v[:60] + "..." if len(v) > 60 else v) for k, v in parsed.items() if v != NOT_PROVIDED},
    )
    return parsed


# -- Classification ----------------------------------------------------------

_TIER_ORDER = ["CRITICAL", "MAJOR", "MODERATE", "MINOR", "NONE"]


def classify_severity_checked(parsed: dict) -> tuple[str, bool]:
    """(tier, explicit). `explicit` is False when the model gave no readable
    tier and MINOR was assumed -- the lane must route that to a human with a
    reason, never treat it as a real finding, and never auto-clear it.

    Matches most-severe first, so an ambiguous answer such as "MINOR to
    MODERATE" biases toward more damage."""
    raw = (parsed.get("damage_severity") or "").upper()
    for tier in _TIER_ORDER:
        if tier in raw:
            return tier, True
    logger.warning("Could not classify severity from: '%s', defaulting to MINOR", raw)
    return "MINOR", False


def classify_severity(parsed: dict) -> tuple[str, dict]:
    """Map the VLM's DAMAGE_SEVERITY output to one of the canonical tiers."""
    tier, _ = classify_severity_checked(parsed)
    return tier, SEVERITY_TIERS[tier]


def resolve_disposition(parsed: dict, severity_key: str) -> dict:
    """Resolve disposition. Trust the model first; on miss, default by severity."""
    raw = (parsed.get("disposition") or "").upper()
    for key in DISPOSITION_PLAYBOOK:
        if key in raw:
            return {
                "key": key,
                "label": DISPOSITION_PLAYBOOK[key]["label"],
                "actions": DISPOSITION_PLAYBOOK[key]["actions"],
            }

    # Fallback: severity-driven default
    fallback_map = {
        "NONE": "RELEASE",
        "MINOR": "TOUCH_UP",
        "MODERATE": "BODY_SHOP_HOLD",
        "MAJOR": "BODY_SHOP_HOLD",
        "CRITICAL": "REJECT",
    }
    key = fallback_map.get(severity_key, "BODY_SHOP_HOLD")
    logger.info("Disposition defaulted to %s based on severity %s", key, severity_key)
    return {
        "key": key,
        "label": DISPOSITION_PLAYBOOK[key]["label"],
        "actions": DISPOSITION_PLAYBOOK[key]["actions"],
    }


def parse_area_percent(parsed: dict) -> float | None:
    """Extract a numeric percent from the affected-area field."""
    raw = parsed.get("estimated_affected_area_percent", "")
    if not raw or raw == NOT_PROVIDED:
        return None
    digits = "".join(c for c in raw if c.isdigit() or c == ".")
    if not digits:
        return None
    try:
        value = float(digits)
        if 0 <= value <= 100:
            return value
    except ValueError:
        pass
    return None


def parse_damage_views(parsed: dict, angles: list[str]) -> list[str]:
    """DAMAGE_VIEWS ("1, 3" / "Photo 2" / NONE) -> the angle names of those
    photos, in portal order. Numbers outside 1..len(angles) are ignored rather
    than trusted: the UI outlines these cameras, so a wrong one would point the
    inspector at a clean panel."""
    raw = parsed.get("damage_views") or ""
    if raw == NOT_PROVIDED or raw.strip().upper().startswith("NONE"):
        return []
    picked = []
    for n in re.findall(r"\d+", raw):
        i = int(n) - 1
        if 0 <= i < len(angles) and angles[i] not in picked:
            picked.append(angles[i])
    return picked


def parse_photo_findings(raw_response: str, angles: list[str]) -> dict[str, str]:
    """PHOTO_n lines -> {angle: finding}, for the per-camera captions."""
    out = {}
    for line in (raw_response or "").splitlines():
        m = re.match(r"\s*[*#-]*\s*PHOTO[_ ]?(\d+)\**\s*:\s*(.+)", line, re.I)
        if m:
            i = int(m.group(1)) - 1
            if 0 <= i < len(angles):
                out[angles[i]] = m.group(2).strip().strip("*").strip()
    return out


# -- Lane routing ------------------------------------------------------------
# The whole point of the lane: auto-clear only what the model affirmatively
# called clean, and put a human on everything else.

LANE_CLEARED = "cleared"
LANE_REVIEW = "review"
LANE_HOLD = "hold"

LANES = {
    LANE_CLEARED: "Cleared",
    LANE_REVIEW: "Needs inspection",
    LANE_HOLD: "Hold",
}


def route(severity: str, explicit: bool) -> tuple[str, str]:
    """(lane, reason). Invariant: nothing but an explicit NONE auto-clears."""
    if not explicit:
        return LANE_REVIEW, "Model output incomplete; needs a human look"
    if severity == "NONE":
        return LANE_CLEARED, "No exterior damage observed; released automatically"
    if severity in ("MINOR", "MODERATE"):
        return LANE_REVIEW, f"{severity.title()} finding; confirm before release"
    if severity in ("MAJOR", "CRITICAL"):
        return LANE_HOLD, f"{severity.title()} damage; hold for inspector"
    return LANE_REVIEW, f"Unrecognised severity '{severity}'; needs a human look"


# -- Image prep --------------------------------------------------------------

# Portal views go to the model at 768 px: four views at 768 measured 5.7 s per
# vehicle against 6.7 s at 1024, with no difference in what was found.
LANE_IMAGE_MAX = 768


def prepare_image_for_vlm(image: Image.Image, max_dim: int = 1024) -> str:
    """Convert PIL image to base64 JPEG, resizing if needed.

    1024 px on the long side keeps the vision tokens for one image well inside
    the model's context alongside the prompt and the report.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
        logger.info("Resized image to %s", new_size)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
