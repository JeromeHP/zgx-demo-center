#!/usr/bin/env bash
# Phase 0 preflight: read-only. Prints each section and tees the full output
# to reports/preflight_<timestamp>.txt. No other writes.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORTS_DIR="$REPO_ROOT/reports"
mkdir -p "$REPORTS_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$REPORTS_DIR/preflight_${TS}.txt"

section() { echo; echo "=== $1 ==="; }

{
  section "OS / WSL"
  cat /etc/os-release | grep -E '^(NAME|VERSION)='
  uname -r
  nproc
  free -g
  df -h "$REPO_ROOT"

  section "GPU passthrough + ROCm"
  ls -l /dev/dxg 2>&1
  if command -v rocminfo >/dev/null 2>&1; then
    rocminfo | grep -E 'Name:|Marketing Name' | head -20
  else
    echo "rocminfo not found"
  fi
  if command -v hipconfig >/dev/null 2>&1; then
    hipconfig --version
  else
    echo "hipconfig not found"
  fi

  section "Existing llama.cpp (do not modify)"
  command -v llama-server || echo "llama-server not on PATH"
  FOUND_SERVERS="$(find "$HOME" -maxdepth 4 -type f -name llama-server 2>/dev/null)"
  echo "$FOUND_SERVERS"
  if [ -n "$FOUND_SERVERS" ]; then
    while IFS= read -r bin; do
      echo "--- $bin --version ---"
      "$bin" --version 2>&1 || true
      echo "--- $bin --list-devices ---"
      "$bin" --list-devices 2>&1 || true
    done <<< "$FOUND_SERVERS"
  fi

  section "Python tooling"
  python3 --version
  command -v uv && uv --version
  command -v git && git --version

  section "Docker"
  command -v docker && docker version --format '{{.Server.Version}}' 2>&1
  docker info 2>/dev/null | grep -E 'Operating System|Architecture|Server Version'

  section "Ports"
  ss -ltnp 2>/dev/null | grep -E ':(8080|8095)\b' || echo "ports 8080/8095 free"

  section "Disk for the model (need >= 30 GB free)"
  df -h "$HOME"
} | tee "$OUT"

echo
echo "Full report written to $OUT"
