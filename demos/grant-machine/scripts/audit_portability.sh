#!/usr/bin/env bash
# Phase 11: fail on hardcoded home paths, external URLs in app/web, or GPU/copyleft deps.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
FAILED=0
GREP_EXCLUDES=(--exclude-dir=__pycache__ --exclude='*.pyc')

if grep -rnE "${GREP_EXCLUDES[@]}" 'Path\.home\(\)|expanduser\(|\$HOME|/home/[a-z]' app web tests; then
  echo "FAIL: hardcoded home path found above"
  FAILED=1
fi

# web/ stays absolute: no CDN scripts, no web fonts, no analytics, no external
# anything. The UI must render with the network unplugged.
#
# The shared Demo Center chrome (demos/ui/, injected between zgx-ui markers by
# scripts/sync_ui.py) is stripped before this check. It contains the literal
# "http://" while building a same-host launcher URL at
# `"http://" + location.hostname + ":8086"`, which the pattern below cannot
# distinguish from a CDN. It is exempt on its merits, not for convenience: it
# talks only to this machine's own launcher, it is shared reviewed code rather
# than anything grant-machine authored, and when that launcher is unreachable
# every meter renders an em-dash -- so the UI still renders with the network
# unplugged, which is the invariant this rule exists to protect.
WEB_SANS_CHROME="$(mktemp -d)"
trap 'rm -rf "$WEB_SANS_CHROME"' EXIT
cp -r web/. "$WEB_SANS_CHROME/"
python3 - "$WEB_SANS_CHROME" <<'PY'
import pathlib, re, sys
for f in pathlib.Path(sys.argv[1]).rglob("*"):
    if f.is_file() and f.suffix in {".html", ".js", ".css"}:
        t = f.read_text(encoding="utf-8", errors="replace")
        stripped = re.sub(r"<!--\s*zgx-ui:begin.*?zgx-ui:end\s*-->", "", t, flags=re.S)
        if stripped != t:
            f.write_text(stripped, encoding="utf-8")
PY
if grep -rnE "${GREP_EXCLUDES[@]}" 'https?://' "$WEB_SANS_CHROME" | grep -vE 'localhost|127\.0\.0\.1|nsf\.gov'; then
  echo "FAIL: external URL in web/ (the UI must make no external calls at all)"
  FAILED=1
fi

# app/ was also absolute until v2's Discover, which deliberately reaches the public
# funding catalogue. The invariant is now narrower rather than gone: outbound URLs
# may appear ONLY in the guarded discovery clients, which cannot transmit anything
# without going through app/egress.py. A URL anywhere else in app/ is a call that
# would bypass the boundary.
URL_ALLOWED='app/discovery/.*|app/egress\.py'
if grep -rnE "${GREP_EXCLUDES[@]}" 'https?://' app \
   | grep -vE "^(${URL_ALLOWED}):" \
   | grep -vE 'localhost|127\.0\.0\.1|nsf\.gov'; then
  echo "FAIL: external URL in app/ outside the guarded discovery clients (above)"
  FAILED=1
fi

# Phase 15 (v2): app/egress.py is the single outbound chokepoint, and the
# containment claim ("the institution's documents never leave this device") rests
# on it being the ONLY way out. Three pre-existing files legitimately touch the
# network and are listed explicitly:
#   app/llm_client.py -- the local model endpoint
#   app/main.py       -- /api/health probing that same local endpoint
#   app/netstatus.py  -- a bare TCP connect to the configured probe hosts, no payload
# Any OTHER file in app/ reaching for the network is a new hole in the boundary and
# fails the audit. Add to the allowlist only with a reason, never to silence this.
EGRESS_ALLOWED='app/egress.py|app/llm_client.py|app/main.py|app/netstatus.py'
NET_CALLERS=$(grep -rlE "${GREP_EXCLUDES[@]}" '(^|[^.])\b(import httpx|from httpx|import socket|import urllib\.request|import requests)\b' app 2>/dev/null \
              | grep -vE "^(${EGRESS_ALLOWED})$" || true)
if [ -n "$NET_CALLERS" ]; then
  echo "$NET_CALLERS"
  echo "FAIL: file(s) above reach the network outside app/egress.py — every outbound"
  echo "      byte must go through the egress guard so it is validated and audited."
  FAILED=1
fi

if grep -rniE "${GREP_EXCLUDES[@]}" 'torch|transformers|sentence_transformers|onnxruntime|cupy|pycuda' pyproject.toml app; then
  echo "FAIL: GPU/ML library found above (must not be an app dependency)"
  FAILED=1
fi

if grep -rnE "${GREP_EXCLUDES[@]}" 'fitz|pymupdf|html2text' pyproject.toml app; then
  echo "FAIL: copyleft-licensed library found above (AGPL/GPL)"
  FAILED=1
fi

if [ "$FAILED" -eq 0 ]; then
  echo "Portability audit OK"
  exit 0
else
  exit 1
fi
