#!/usr/bin/env bash
# Install the ROCm-accelerated ONNX Runtime for AI Vision, and SAY which
# execution provider actually resulted.
#
# Why a script and not one pip line:
#
#   1. AMD does not publish these on PyPI. They live in per-ROCm-release flat
#      directories under repo.radeon.com, consumed with `pip -f`, so the URL has
#      to match the ROCm actually installed.
#   2. The package RENAMES at ROCm 7.1. Verified against the live index:
#        ROCm <= 7.0.x   onnxruntime-rocm       (ROCMExecutionProvider)
#        ROCm >= 7.1     onnxruntime-migraphx   (MIGraphXExecutionProvider)
#      A plain `pip install onnxruntime-rocm` against a 7.2 index finds nothing.
#   3. Not every point release ships a wheel at all -- rocm-rel-7.1.1 has
#      neither package -- so an exact-version URL has to degrade to a nearby one.
#   4. pip "succeeding" proves nothing. A wheel can install and still leave you
#      on CPUExecutionProvider. The only question that matters is what
#      onnxruntime reports afterwards, so that is what this checks.
#
# Exit codes:
#   0  a GPU execution provider is available (or CPU, with a loud warning)
#   1  onnxruntime is not importable at all
# Set ZGX_REQUIRE_ROCM_EP=1 to make a CPU-only outcome a hard failure.
set -uo pipefail

PY="${ZGX_PYTHON:-$HOME/miniforge3/envs/zgx/bin/python3}"
BASE="https://repo.radeon.com/rocm/manylinux"
GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; BLUE=$'\033[34m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
log()  { printf '%sℹ %s%s\n' "$BLUE" "$*" "$RESET"; }
ok()   { printf '%s✓ %s%s\n' "$GREEN" "$*" "$RESET"; }
warn() { printf '%s⚠ %s%s\n' "$YELLOW" "$*" "$RESET"; }
err()  { printf '%s✗ %s%s\n' "$RED" "$*" "$RESET" >&2; }

[[ -x "$PY" ]] || { err "python not found at $PY"; exit 1; }

# ── Which ROCm is installed? ────────────────────────────────────────────────
ROCM_VER=""
if [[ -r /opt/rocm/.info/version ]]; then
    ROCM_VER="$(head -1 /opt/rocm/.info/version | tr -d '[:space:]')"
elif command -v hipconfig >/dev/null 2>&1; then
    ROCM_VER="$(hipconfig --version 2>/dev/null | head -1 | tr -d '[:space:]')"
fi
ROCM_VER="${ROCM_VER%%-*}"   # 6.4.4-123 -> 6.4.4

if [[ -z "$ROCM_VER" ]]; then
    warn "Could not determine the ROCm version (no /opt/rocm/.info/version, no hipconfig)."
    warn "Falling back to the CPU build of onnxruntime."
    "$PY" -m pip install --timeout 60 --retries 10 onnxruntime >/dev/null 2>&1
else
    log "ROCm detected: $ROCM_VER"

    major="${ROCM_VER%%.*}"; rest="${ROCM_VER#*.}"; minor="${rest%%.*}"
    # The rename boundary, confirmed against the live index.
    if [[ "$major" -gt 7 || ( "$major" -eq 7 && "$minor" -ge 1 ) ]]; then
        PKG="onnxruntime-migraphx"
    else
        PKG="onnxruntime-rocm"
    fi
    log "Package for this ROCm line: $PKG"

    # Candidate indexes, most specific first. Not every point release ships one.
    CANDIDATES=("rocm-rel-${ROCM_VER}" "rocm-rel-${major}.${minor}")
    installed="no"
    for rel in "${CANDIDATES[@]}"; do
        url="$BASE/$rel/"
        if ! curl -sfI --max-time 20 "$url" >/dev/null 2>&1; then
            log "  no index at $rel — trying next"
            continue
        fi
        log "  installing $PKG from $rel"
        if "$PY" -m pip install --timeout 120 --retries 5 -f "$url" "$PKG"; then
            installed="yes"; break
        fi
        warn "  $PKG not installable from $rel"
    done

    if [[ "$installed" != "yes" ]]; then
        warn "No ROCm ONNX Runtime wheel matched ROCm $ROCM_VER — using the CPU build."
        warn "Check what exists: curl -s $BASE/ | grep rocm-rel"
        "$PY" -m pip install --timeout 60 --retries 10 onnxruntime >/dev/null 2>&1
    fi
fi

# ── The only check that counts ──────────────────────────────────────────────
PROVIDERS="$("$PY" -c 'import onnxruntime as ort; print(",".join(ort.get_available_providers()))' 2>/dev/null)"
if [[ -z "$PROVIDERS" ]]; then
    err "onnxruntime is not importable — AI Vision has no detection backend."
    exit 1
fi

echo
log "onnxruntime providers: $PROVIDERS"
if [[ "$PROVIDERS" == *MIGraphXExecutionProvider* || "$PROVIDERS" == *ROCMExecutionProvider* ]]; then
    ok "GPU execution provider available — AI Vision will run detection on the AMD GPU."
    exit 0
fi

printf '\n%s%s─────────────────────────────────────────────────────────────%s\n' "$BOLD" "$YELLOW" "$RESET"
warn "ONNX Runtime installed, but only the CPU provider is available."
warn "AI Vision WILL run, and its header will say CPU in amber — but detection"
warn "will be several times slower than the GPU path (~128 ms/frame measured)."
warn ""
warn "Usually one of:"
warn "  • the ROCm userspace is not installed (check: rocminfo)"
warn "  • the installed ROCm has no matching wheel (check: curl -s $BASE/)"
warn "  • this env is not Python 3.10 or 3.12 (wheels are cp310/cp312 only)"
printf '%s%s─────────────────────────────────────────────────────────────%s\n\n' "$BOLD" "$YELLOW" "$RESET"

if [[ "${ZGX_REQUIRE_ROCM_EP:-0}" == "1" ]]; then
    err "ZGX_REQUIRE_ROCM_EP=1 — treating a CPU-only outcome as a failure."
    exit 1
fi
exit 0
