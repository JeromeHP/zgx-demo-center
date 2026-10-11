#!/usr/bin/env bash
# The shared chrome is present, current, and singular.
#
# Three things this stops, each of which already happened once:
#   - a demo drifting off the standard because its frontend was edited by hand
#   - a second HP logo appearing (five demos shipped THREE different files under
#     that one name before demos/ui/brand became canonical)
#   - a demo growing its own token accounting instead of using the shared ledger
#
# Exit 0 clean, 1 with the offending file named.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
FAILED=0
pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1"; FAILED=1; }

echo "== zgx-ui standard =="

# 1. Every frontend carries the current block.
if python3 scripts/sync_ui.py --check >/tmp/zgx_ui_check.$$ 2>&1; then
  pass "all frontends carry the current chrome"
else
  sed 's/^/    /' /tmp/zgx_ui_check.$$
  fail "a frontend is missing or has a stale chrome (run scripts/sync_ui.py)"
fi
rm -f /tmp/zgx_ui_check.$$

# 2. The container copies of the token ledger match the canonical file.
if python3 scripts/sync_metrics.py --check >/tmp/zgx_m_check.$$ 2>&1; then
  pass "container copies of zgx_metrics.py match the canonical file"
else
  sed 's/^/    /' /tmp/zgx_m_check.$$
  fail "a zgx_metrics.py copy has drifted (run scripts/sync_metrics.py)"
fi
rm -f /tmp/zgx_m_check.$$

# 3. One canonical logo. Demos may still ship their own file (the chrome hides
#    it with CSS), but it must not be a DIFFERENT image claiming to be the mark.
CANON="$(md5sum ui/brand/hp_logo.png | awk '{print $1}')"
divergent=0
while IFS= read -r f; do
  [ "$f" = "./ui/brand/hp_logo.png" ] && continue
  if [ "$(md5sum "$f" | awk '{print $1}')" != "$CANON" ]; then
    echo "    differs from ui/brand/hp_logo.png: $f"
    divergent=1
  fi
done < <(find . -name "hp_logo.png" -not -path "./node_modules/*" 2>/dev/null)
if [ "$divergent" -eq 0 ]; then
  pass "every hp_logo.png matches the canonical asset"
else
  fail "a divergent HP logo is present"
fi

# 4. Every demo that calls an LLM serves /api/metrics, so the chrome's token
#    panel has something to read.
#
#    This checks the OUTCOME, not the mechanism. Two earlier versions checked
#    the mechanism and both were wrong: the first flagged any read of Ollama's
#    token fields, which fired on ballistics and pii building their own
#    per-request token_usage for their own report cards; the second required
#    every such file to import TokenLedger, which fired on grant-machine, whose
#    instrumentation predates the shared ledger and is the implementation the
#    ledger was modelled on, and on ballistics, which records in its client
#    rather than its route module. Both were correct code. What matters is
#    whether the numbers reach the chrome.
LLM_DEMOS="ballistics-containerized-docker pii-demo-containerized-docker \
healthcare-pharmacy-containerized-docker Fed-Navy-containerized-docker \
ai-vision ew-spectrum-duel pipeline-pulse grant-machine \
automotive-vehicle-inspection"
missing=""
for d in $LLM_DEMOS; do
  if ! grep -rqs '"/api/metrics"' "$d" --include=*.py; then
    missing="$missing$d\n"
  fi
done
if [ -n "$missing" ]; then
  printf "%b" "$missing" | sed 's/^/    /'
  fail "an LLM demo does not serve /api/metrics — its token panel stays hidden"
else
  pass "every LLM demo serves /api/metrics"
fi

# 5. A containerised demo's frontend must be bind-mounted, and its ledger
#    import must survive both launch styles.
#
#    Two ways the chrome silently failed to appear, both found in the field:
#    the frontend was baked into the image so it did not update until someone
#    happened to rebuild, and `from zgx_metrics import ...` crash-looped under
#    `uvicorn backend.main:app` (which puts /app on sys.path, not /app/backend)
#    while working fine under `python3 backend/main.py`. Neither said why.
CONTAINERISED="pii-demo-containerized-docker healthcare-pharmacy-containerized-docker \
ballistics-containerized-docker Fed-Navy-containerized-docker \
automotive-vehicle-inspection"
prob=""
for d in $CONTAINERISED; do
  grep -q "./frontend:/app/frontend" "$d/docker-compose.yml" 2>/dev/null \
    || prob="$prob$d: frontend not bind-mounted\n"
  if grep -rqs "from zgx_metrics import" "$d/backend" 2>/dev/null \
     && ! grep -rqs "from backend.zgx_metrics import" "$d/backend" 2>/dev/null; then
    prob="$prob$d: zgx_metrics import has no package-style fallback\n"
  fi
done
if [ -n "$prob" ]; then
  printf "%b" "$prob" | sed 's/^/    /'
  fail "a containerised demo can silently lose the chrome"
else
  pass "containerised demos mount their frontend and import the ledger safely"
fi

# 6. No compose file bind-mounts a git-TRACKED directory read-write.
#
#    healthcare-pharmacy mounted its committed ChromaDB index read-write.
#    Chroma writes to sqlite on open, so merely RUNNING the demo dirtied five
#    tracked files and the operator's next `git pull` aborted with "local
#    changes would be overwritten" -- changes they never knowingly made. A
#    tracked directory is shipped data; if a demo needs to write to it, it
#    needs a copy, not the original.
rw=""
for f in $(find . -name "docker-compose*.yml" -not -path "./node_modules/*" 2>/dev/null); do
  d="$(dirname "$f")"
  # Bind mounts of the form  - ./thing:/somewhere   with no :ro suffix.
  while IFS= read -r mount; do
    host="${mount%%:*}"; host="${host#*- }"
    case "$mount" in *":ro"*) continue ;; esac
    case "$host" in ./*) ;; *) continue ;; esac
    path="$d/${host#./}"
    [ -d "$path" ] || continue
    if git -C . ls-files --error-unmatch "$path" >/dev/null 2>&1 \
       || [ -n "$(git -C . ls-files "$path" 2>/dev/null)" ]; then
      rw="$rw$f: ${host} is git-tracked and mounted read-write\n"
    fi
  done <<EOT
$(grep -E "^\s+- \./[^:]+:" "$f" 2>/dev/null)
EOT
done
if [ -n "$rw" ]; then
  printf "%b" "$rw" | sed 's/^/    /'
  fail "a tracked directory is mounted read-write — running the demo will dirty the repo"
else
  pass "no tracked directory is mounted read-write"
fi

echo
if [ "$FAILED" -ne 0 ]; then echo "zgx-ui standard: FAILED"; exit 1; fi
echo "zgx-ui standard: PASS"
