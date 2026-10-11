#!/usr/bin/env bash
# setup-prereqs.sh must not assume NVIDIA.
#
# It ran `apt-get install -y nvidia-container-toolkit` and then
# `nvidia-ctk runtime configure --runtime=docker` with no guard at all. Under
# `set -e` that aborted the entire prereq run on an AMD box -- the first wall
# anyone hits, before install.py has a chance to be clever about profiles.
#
# On x86_64 it was worse than an abort: NVIDIA's apt repo has amd64 packages, so
# the toolkit installed successfully and the configure call then wrote an
# `nvidia` runtime into /etc/docker/daemon.json that cannot exist on that box --
# leaving Docker quietly misconfigured instead of obviously broken.
#
# Run: bash tests/test_setup_prereqs.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
SCRIPT="setup-prereqs.sh"
FAILED=0
pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1"; FAILED=1; }

echo "== setup-prereqs.sh vendor guards =="

bash -n "$SCRIPT" && pass "parses" || fail "syntax error"

grep -q 'ACCEL="none"' "$SCRIPT" \
  && pass "detects an accelerator before any vendor-specific step" \
  || fail "no accelerator detection"

# /dev/kfd, not /dev/dri: an NVIDIA box exposes /dev/dri/renderD128 too, so
# keying on /dev/dri would classify every NVIDIA machine as AMD.
grep -q '/dev/kfd' "$SCRIPT" \
  && pass "uses /dev/kfd as the AMD discriminator" \
  || fail "missing /dev/kfd check"
if grep -qE 'ACCEL="amd"' "$SCRIPT" && grep -B3 'ACCEL="amd"' "$SCRIPT" | grep -q '/dev/dri'; then
  fail "/dev/dri is being used to detect AMD — an NVIDIA box has it too"
else
  pass "/dev/dri is not used to detect AMD"
fi

# The load-bearing one: the configure call must be inside an nvidia guard.
if grep -B6 'nvidia-ctk runtime configure' "$SCRIPT" | grep -qE 'if \[\[ "\$ACCEL" == "nvidia" \]\]'; then
  pass "nvidia-ctk runtime configure is guarded on ACCEL == nvidia"
else
  fail "nvidia-ctk runtime configure is NOT guarded — this aborts the whole script on AMD"
fi

# The toolkit install must sit INSIDE the accelerator if/elif chain, so the AMD
# and none branches short-circuit past it. A line-distance grep is the wrong
# tool here -- the chain is ~50 lines long, so a fixed -B window either misses
# it or spuriously matches. Walk the block structure instead.
if python3 - "$SCRIPT" <<'PY'
import re, sys
lines = pathlib_read = open(sys.argv[1], encoding="utf-8").read().splitlines()
start = next(i for i, l in enumerate(lines) if re.match(r'\s*if \[\[ "\$ACCEL" == "amd" \]\]', l))
depth = 0
end = None
for i in range(start, len(lines)):
    stripped = lines[i].strip()
    if re.match(r'(if |if\[)', stripped) or stripped.startswith("if "):
        depth += 1
    elif stripped == "fi" or stripped.startswith("fi "):
        depth -= 1
        if depth == 0:
            end = i
            break
if end is None:
    print("could not find the end of the accelerator chain"); sys.exit(1)
block = "\n".join(lines[start:end + 1])
if "apt-get install -y nvidia-container-toolkit" not in block:
    print("the toolkit install is OUTSIDE the accelerator chain — it would run on AMD")
    sys.exit(1)
sys.exit(0)
PY
then
  pass "the toolkit install sits inside the accelerator if/elif chain"
else
  fail "nvidia-container-toolkit install is reachable on the AMD branch"
fi

grep -q 'usermod -aG "\$grp"' "$SCRIPT" \
  && pass "AMD path adds the user to render+video (device nodes are unopenable without it)" \
  || fail "AMD path does not add render/video group membership"

echo
if [ "$FAILED" -ne 0 ]; then echo "setup-prereqs guards: FAILED"; exit 1; fi
echo "setup-prereqs guards: PASS"
