#!/usr/bin/env bash
# ensure_demos_link.sh — all five layouts, exercised for real in a temp HOME.
#
# Case D (a real directory at ~/demos) used to print one suggestion, framed as
# "if it is an old install you no longer need": mv it to .bak and symlink. Two
# problems with that. It asserted something it had not checked -- a real
# directory at ~/demos is the layout make_bundle.sh ships and Case A treats as
# canonical. And the suggestion is expensive: the per-demo models live inside
# that tree ({dir}/models -- maritime 11GB, pharmacy 28GB, pii 20GB), so moving
# it aside orphans up to ~63GB that Phase 6 then re-downloads.
#
# Run: bash tests/test_demos_link.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
LIB="$PWD/ensure_demos_link.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FAILED=0
pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1"; FAILED=1; }

run() {  # run <fake_home> <checkout_demos>
  (cd "$2" && HOME="$1" bash -c "source '$LIB'; ensure_demos_link \"\$PWD\" false") 2>&1
}

echo "== ensure_demos_link.sh layouts =="

# A: the repo already lives at ~/demos (bundle unzipped in $HOME).
mkdir -p "$TMP/a/demos"
run "$TMP/a" "$TMP/a/demos" | grep -q "no link needed" \
  && pass "A: repo at ~/demos is accepted as canonical" || fail "A: rejected the bundle layout"

# B: nothing at ~/demos -> create the link.
mkdir -p "$TMP/b/co/demos"
run "$TMP/b" "$TMP/b/co/demos" | grep -q "Linked ~/demos" \
  && pass "B: creates the link" || fail "B: did not create the link"
[[ -L "$TMP/b/demos" ]] && pass "B: it is a symlink" || fail "B: not a symlink"

# C: already correct -> no-op.
run "$TMP/b" "$TMP/b/co/demos" | grep -q "already correct" \
  && pass "C: correct link is a no-op" || fail "C: did not recognise its own link"

# C': dangling link -> self-heal.
mkdir -p "$TMP/c/co/demos"; ln -s "$TMP/c/gone" "$TMP/c/demos"
run "$TMP/c" "$TMP/c/co/demos" | grep -q "Repointed dangling" \
  && pass "C': dangling link self-heals" || fail "C': did not repoint a dangling link"

# D1: a git checkout at ~/demos -> point at it, do NOT suggest moving first.
mkdir -p "$TMP/d1/demos" "$TMP/d1/co/demos"
git -C "$TMP/d1/demos" init -q . && git -C "$TMP/d1/demos" commit -q --allow-empty -m x
out="$(run "$TMP/d1" "$TMP/d1/co/demos")"
grep -q "itself a git checkout" <<<"$out" \
  && pass "D1: recognises a git checkout at ~/demos" || fail "D1: did not identify a git checkout"
grep -q "git pull && ./install.sh" <<<"$out" \
  && pass "D1: offers using it instead of moving it" || fail "D1: no use-it-instead option"

# D2: models present -> the size must be named and preserving them offered.
mkdir -p "$TMP/d2/demos/pharmacy-x/models" "$TMP/d2/co/demos"
touch "$TMP/d2/demos/install.sh" "$TMP/d2/demos/install_plans.py"
dd if=/dev/zero of="$TMP/d2/demos/pharmacy-x/models/blob" bs=1M count=1100 status=none
out="$(run "$TMP/d2" "$TMP/d2/co/demos")"
grep -qE "holds roughly [0-9]+GB of downloaded models" <<<"$out" \
  && pass "D2: names the GB at risk" || fail "D2: did not warn about model data"
grep -q "adopt_models.sh" <<<"$out" \
  && pass "D2: points at adopt_models.sh, not a raw mv one-liner" \
  || fail "D2: only offered the destructive move"
# The dry run must be offered before --apply, or the first thing a reader
# copy-pastes is the irreversible one.
if [[ "$(grep -n 'adopt_models.sh' <<<"$out" | head -1 | grep -c -- '--apply')" == "0" ]]; then
  pass "D2: the dry-run invocation is listed first"
else
  fail "D2: --apply appears before the dry run"
fi
# The number must be believable: 1100MiB is ~1GB, not 2+.
if grep -qE "holds roughly 1GB" <<<"$out"; then
  pass "D2: size is accurate (not rounded up per-path)"
else
  fail "D2: size looks wrong -- $(grep -oE 'roughly [0-9]+GB' <<<"$out")"
fi

# D3: a bare directory -> say so rather than guess.
#
# Output is captured before grepping, NOT piped. Case D correctly returns 1 (it
# is a refusal), and under `set -o pipefail` a direct `run ... | grep -q` fails
# on that return code even when grep matched -- which reported this passing
# check as broken. Command substitution drops the status, which is what the
# earlier D-cases already did.
mkdir -p "$TMP/d3/demos/random" "$TMP/d3/co/demos"
out="$(run "$TMP/d3" "$TMP/d3/co/demos")"
grep -q "Could not tell what that directory is" <<<"$out" \
  && pass "D3: unknown directory is reported as unknown" || fail "D3: guessed at an unknown directory"

echo
if [ "$FAILED" -ne 0 ]; then echo "demos-link layouts: FAILED"; exit 1; fi
echo "demos-link layouts: PASS"
