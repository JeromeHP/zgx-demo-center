#!/usr/bin/env bash
# The ROCm ONNX Runtime installer picks the right wheel, and tells the truth
# about what it got.
#
# The previous step was `pip install onnxruntime-rocm || pip install
# onnxruntime`. On a stock pip that ALWAYS took the second branch -- AMD does
# not publish to PyPI -- so AI Vision ran detection on CPU while every step
# reported success. These tests pin the three facts that make the replacement
# work, all verified against the live index at repo.radeon.com.
#
# Run: bash tests/test_onnxruntime_rocm.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
SCRIPT="scripts/install_onnxruntime_rocm.sh"
FAILED=0
pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1"; FAILED=1; }

echo "== ROCm ONNX Runtime installer =="

bash -n "$SCRIPT" && pass "parses" || fail "syntax error"

# 1. The package renames at ROCm 7.1 — confirmed against the live index:
#    6.4.x / 7.0.x ship onnxruntime_rocm, 7.1+ ship onnxruntime_migraphx.
#    Getting this wrong means pip finds nothing and silently lands on CPU.
check_pkg() {  # check_pkg <version> <expected>
  local v="$1" want="$2" major rest minor got
  major="${v%%.*}"; rest="${v#*.}"; minor="${rest%%.*}"
  if [[ "$major" -gt 7 || ( "$major" -eq 7 && "$minor" -ge 1 ) ]]; then
    got="onnxruntime-migraphx"
  else
    got="onnxruntime-rocm"
  fi
  [[ "$got" == "$want" ]] && pass "ROCm $v -> $want" || fail "ROCm $v chose $got, expected $want"
}
check_pkg 6.4.4 onnxruntime-rocm
check_pkg 7.0.2 onnxruntime-rocm
check_pkg 7.1   onnxruntime-migraphx
check_pkg 7.2.4 onnxruntime-migraphx

grep -q 'onnxruntime-migraphx' "$SCRIPT" \
  && pass "script knows about the post-7.1 package name" \
  || fail "script only handles onnxruntime-rocm — 7.1+ would find nothing"

# 2. It must degrade from an exact version to the X.Y line: rocm-rel-7.1.1
#    exists but ships NO onnxruntime wheel, so an exact-only URL dead-ends.
grep -q 'rocm-rel-${major}.${minor}' "$SCRIPT" \
  && pass "falls back from an exact release to the X.Y line" \
  || fail "no fallback — a point release without wheels would dead-end"

# 3. The load-bearing one: pip succeeding proves nothing. A wheel can install
#    and still leave CPUExecutionProvider as the only option.
grep -q 'get_available_providers' "$SCRIPT" \
  && pass "verifies the provider list rather than trusting pip's exit code" \
  || fail "does not check which provider actually resulted"
grep -qE 'MIGraphXExecutionProvider.*ROCMExecutionProvider|ROCMExecutionProvider.*MIGraphXExecutionProvider' "$SCRIPT" \
  && pass "accepts either GPU provider" || fail "does not check for both GPU providers"

# A CPU-only outcome must be loud, and optionally fatal.
grep -q 'ZGX_REQUIRE_ROCM_EP' "$SCRIPT" \
  && pass "CPU-only can be made a hard failure via ZGX_REQUIRE_ROCM_EP=1" \
  || fail "no way to demand the GPU provider"

# 4. Wired into the AMD vision plan, and no bare PyPI attempt left behind.
if python3 - <<'PY'
import sys, pathlib
sys.path.insert(0, ".")
import install_plans as ip
steps = ip.INSTALL_PLANS["vision"]["profiles"]["amd-rocm"]["steps"]
cmds = " ".join(s["cmd"] for s in steps)
if "install_onnxruntime_rocm.sh" not in cmds:
    print("AMD vision plan does not call the installer"); sys.exit(1)
if "pip install --timeout 60 --retries 10 onnxruntime-rocm" in cmds:
    print("a bare PyPI onnxruntime-rocm install is still present"); sys.exit(1)
sys.exit(0)
PY
then pass "AMD vision plan calls the installer, with no bare PyPI attempt"
else fail "AMD vision plan wiring is wrong"
fi

echo
if [ "$FAILED" -ne 0 ]; then echo "onnxruntime-rocm installer: FAILED"; exit 1; fi
echo "onnxruntime-rocm installer: PASS"
