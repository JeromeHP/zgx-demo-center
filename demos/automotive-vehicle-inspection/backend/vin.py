"""
Simulated VINs for the vessel manifest.

Every VIN this module produces is SIMULATED and is labelled that way everywhere
it is shown (lane cards, review drawer, CSV). They are built to look and
validate like real ones so the manifest reads credibly:

  - the World Manufacturer Identifier (positions 1-3) is the real Stellantis
    WMI for the brand,
  - position 9 is a valid ISO 3779 / North American check digit,
  - position 10 is the model-year code for the photo's approximate year,
  - I, O and Q never appear (they are illegal in a VIN).

The remaining positions are drawn from a seeded RNG, so a given run's manifest
is reproducible, but they encode nothing.
"""

import random

# Real WMIs. Where a brand builds in several plants, one representative is used.
WMI = {
    "Jeep": "1C4",       # FCA US, MPV
    "Ram": "1C6",        # FCA US, truck
    "Dodge": "2C3",      # FCA Canada, passenger car
    "Chrysler": "2C4",   # FCA Canada, MPV (Pacifica, Windsor)
    "Fiat": "ZFA",
    "Abarth": "ZFA",     # built and titled under Fiat's WMI
    "Alfa Romeo": "ZAR",
    "Maserati": "ZAM",
    "Lancia": "ZLA",
    "Peugeot": "VF3",
    "Citroën": "VF7",
    "DS": "VR1",
    "Opel": "W0L",
    "Vauxhall": "W0V",
}

VIN_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"  # no I, O, Q

_TRANSLIT = {
    **{str(d): d for d in range(10)},
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8,
    "J": 1, "K": 2, "L": 3, "M": 4, "N": 5, "P": 7, "R": 9,
    "S": 2, "T": 3, "U": 4, "V": 5, "W": 6, "X": 7, "Y": 8, "Z": 9,
}
_WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]

# Model-year codes cycle every 30 years; 1980-2009 and 2010-2039 share letters.
_YEAR_CODES = "ABCDEFGHJKLMNPRSTVWXY123456789"


def year_code(year: int) -> str:
    """Position-10 code. VINs did not exist before 1981, so earlier years clamp."""
    year = max(1981, min(2039, int(year)))
    return _YEAR_CODES[(year - 1980) % 30]


def check_digit(vin: str) -> str:
    total = sum(_TRANSLIT[c] * w for c, w in zip(vin, _WEIGHTS))
    r = total % 11
    return "X" if r == 10 else str(r)


def is_valid(vin: str) -> bool:
    return (
        len(vin) == 17
        and all(c in VIN_CHARS for c in vin)
        and vin[8] == check_digit(vin)
    )


def simulated_vin(brand: str, model_year: int, rng: random.Random) -> str:
    wmi = WMI.get(brand)
    if wmi is None:
        raise ValueError(f"no WMI for brand {brand!r}")
    vds = "".join(rng.choice(VIN_CHARS) for _ in range(5))
    plant = rng.choice(VIN_CHARS)
    serial = "".join(rng.choice("0123456789") for _ in range(6))
    draft = f"{wmi}{vds}0{year_code(model_year)}{plant}{serial}"
    return draft[:8] + check_digit(draft) + draft[9:]
