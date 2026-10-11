#!/usr/bin/env bash
# Fail if a demo states a hardware fact on its own authority.
#
# Why this exists: every frontend in this repo hardcoded "Powered by HP ZGX
# Nano", the pharmacy one added "with NVIDIA Grace Blackwell GB10", and the
# launcher asserted eight system-bar values that no code ever checked -- three
# of which were already wrong on the machine they described. None of it could be
# fixed for an AMD-funded show except by editing HTML per event.
#
# Presenter-visible hardware claims must come from GET /sysinfo. This script is
# what stops the next demo from typing one in again.
#
# Exit 0 = clean. Exit 1 = a claim was found; the offending line is printed.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

FAILED=0
EXCLUDES=(--binary-files=without-match
          --exclude-dir=__pycache__ --exclude-dir=.git --exclude-dir=node_modules
          --exclude-dir=models --exclude-dir=venv --exclude-dir=document-translate-env
          --exclude=*.pyc --exclude=*.log --exclude=*.json --exclude=*.md)

# Vendor and product names that must not be asserted in presenter-visible markup.
# ROCm/Radeon/gfx are here too: this guard is symmetric, or it just swaps which
# vendor gets hardcoded.
CLAIMS='HP ZGX Nano|ZGX Nano|GB10|Grace Blackwell|sm_121|sm_120|NVIDIA CTK|Strix Halo|Radeon|gfx[0-9]{4}|Ryzen'

# Frontends are what an audience sees. The launcher is covered by
# tests/test_launcher_hardware_wiring.py, which understands its structure.
# grant-machine lives on its own branch and has its own audit_portability.sh.
# Robotic-medical-AI-assistant is out of scope for AMD by decision (nvcr vLLM +
# awq_marlin + a CUDA whisper.cpp build + a physical Reachy Mini), so it keeps
# its NVIDIA copy honestly rather than being made to lie in the other direction.
TARGETS=$(find . -path ./grant-machine -prune \
               -o -path ./Robotic-medical-AI-assistant -prune -o \
                 \( -name 'index.html' -o -name '*frontend*.py' -o -name '*_demo.html' \) -print 2>/dev/null)

echo "== presenter-visible hardware claims =="
for f in $TARGETS; do
  # Strip HTML comments and the inlined zgx-device-label block before matching:
  # both legitimately mention these names while explaining or parsing them away,
  # and a check that matches its own documentation passes while testing nothing.
  stripped=$(python3 - "$f" <<'PY'
import pathlib, re, sys
t = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
# Order matters: the zgx-device-label markers are themselves HTML comments, so
# stripping comments first would delete the markers and leave the snippet body,
# whose own docstring quotes every banned string. Marked block first, then comments.
t = re.sub(r"<!--\s*zgx-device-label:begin.*?zgx-device-label:end\s*-->", "", t, flags=re.S)
t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
print(t)
PY
)
  hits=$(printf '%s' "$stripped" | grep -nEi "$CLAIMS" | grep -v 'data-zgx-device' || true)
  if [ -n "$hits" ]; then
    echo "FAIL: $f asserts hardware it never checked:"
    printf '%s\n' "$hits" | head -5 | sed 's/^/    /'
    FAILED=1
  fi
done
[ "$FAILED" -eq 0 ] && echo "  ok — no demo frontend names a vendor it did not detect"

# A CUDA-only wheel index silently installs an unusable torch on AMD.
echo "== CUDA-pinned wheel indexes =="
if grep -rnE "${EXCLUDES[@]}" 'download\.pytorch\.org/whl/[^"$ ]*cu[0-9]{3}' --include=*.sh . \
     | grep -v 'scripts/torch_index.sh'; then
  echo "FAIL: a CUDA wheel index is hardcoded above — use scripts/torch_index.sh"
  FAILED=1
else
  echo "  ok — torch indexes resolve through scripts/torch_index.sh"
fi

echo
if [ "$FAILED" -ne 0 ]; then echo "audit_hardware_strings: FAILED"; exit 1; fi
echo "audit_hardware_strings: PASS"
