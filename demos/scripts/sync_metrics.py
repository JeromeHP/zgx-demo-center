#!/usr/bin/env python3
"""Copy the canonical token ledger into each demo that needs its own copy.

Containerised demos only `COPY backend/ frontend/` into their image, with the
build context set to the demo directory -- so demos/zgx_metrics.py is outside
their build context and cannot be imported. Rather than let each demo grow its
own token accounting (which is exactly how ai-vision and ew-spectrum-duel ended
up with byte-identical GPU pollers that had to be fixed twice), the canonical
file is copied in and `--check` proves the copies still match.

    python3 scripts/sync_metrics.py           # write the copies
    python3 scripts/sync_metrics.py --check   # verify, exit 1 if drifted
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "zgx_metrics.py"

# Containerised demos need the copy. Native demos import it from demos/ via the
# parent-dir sys.path trick and are deliberately absent here.
TARGETS = [
    "ballistics-containerized-docker/backend/zgx_metrics.py",
    "pii-demo-containerized-docker/backend/zgx_metrics.py",
    "healthcare-pharmacy-containerized-docker/backend/zgx_metrics.py",
    "Fed-Navy-containerized-docker/backend/zgx_metrics.py",
    "automotive-vehicle-inspection/backend/zgx_metrics.py",
]

HEADER = (
    "# GENERATED COPY -- do not edit.\n"
    "# Source: demos/zgx_metrics.py, copied by scripts/sync_metrics.py because\n"
    "# this demo's Docker build context cannot reach outside its own directory.\n"
    "# Verify with: python3 scripts/sync_metrics.py --check\n\n"
)


def main() -> int:
    check = "--check" in sys.argv
    body = HEADER + SRC.read_text(encoding="utf-8")
    print("== zgx_metrics " + ("check" if check else "sync") + " ==")
    ok = True
    for rel in TARGETS:
        dst = ROOT / rel
        if not dst.parent.is_dir():
            print(f"  skip {rel} (no backend dir)")
            continue
        current = dst.read_text(encoding="utf-8") if dst.is_file() else None
        if current == body:
            print(f"  ok   {rel}")
            continue
        if check:
            print(f"  DRIFT {rel}")
            ok = False
            continue
        dst.write_text(body, encoding="utf-8")
        print(f"  sync {rel}")
    if not ok:
        print("\nRun: python3 scripts/sync_metrics.py")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
