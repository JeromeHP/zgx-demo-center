#!/usr/bin/env bash
# Start a self-hosted SearXNG for Discover's general web search.
#
# OPTIONAL. Without it, `search_web` reports itself unavailable and the federal
# catalogue search — which needs no provider — carries the run. So this exits 0 even
# when it cannot start anything: a missing search engine must not fail a demo install.
#
# Self-hosted on purpose. A commercial search API would mean a credential sitting on a
# demo box and every query routed through a third party, which is a poor fit for a
# demo whose argument is that the institution's material stays put.
set -uo pipefail

NAME="${SEARXNG_NAME:-searxng}"
PORT="${SEARXNG_PORT:-8096}"
CONFIG_DIR="${SEARXNG_CONFIG_DIR:-$HOME/searxng-config}"
CONFIG="$CONFIG_DIR/settings.yml"
IMAGE="${SEARXNG_IMAGE:-searxng/searxng:latest}"

if ! command -v docker >/dev/null 2>&1; then
  echo "SearXNG skipped: docker not available. Discover will run without general web search."
  exit 0
fi

if docker ps --filter "name=^/${NAME}$" --format '{{.Names}}' | grep -q "^${NAME}$"; then
  echo "SearXNG already running on :${PORT}."
  exit 0
fi

if [ ! -f "$CONFIG" ]; then
  mkdir -p "$CONFIG_DIR"
  # SearXNG refuses to start on the shipped default secret, and `formats: [json]` is
  # NOT on by default — without it /search?format=json returns 403 and the tool looks
  # broken rather than unconfigured.
  if command -v openssl >/dev/null 2>&1; then
    SECRET="$(openssl rand -hex 32)"
  else
    SECRET="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  fi
  cat > "$CONFIG" <<EOF
use_default_settings: true

server:
  secret_key: "${SECRET}"
  bind_address: "0.0.0.0"
  port: 8080
  limiter: false
  public_instance: false
  image_proxy: false

search:
  formats:
    - html
    - json
  safe_search: 0
  autocomplete: ""

ui:
  static_use_hash: true
EOF
  echo "Wrote $CONFIG"
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
# Loopback only: this is infrastructure for the app on this box, not a service to
# publish on the tailnet.
if ! docker run -d --name "$NAME" --restart unless-stopped \
     -p "127.0.0.1:${PORT}:8080" \
     -v "$CONFIG:/etc/searxng/settings.yml:ro" \
     -e "SEARXNG_BASE_URL=http://127.0.0.1:${PORT}/" \
     "$IMAGE" >/dev/null 2>&1; then
  echo "SearXNG could not start. Discover will run without general web search."
  exit 0
fi

for _ in $(seq 1 20); do
  if curl -s -o /dev/null "http://127.0.0.1:${PORT}/search?q=test&format=json"; then
    echo "SearXNG ready on http://127.0.0.1:${PORT} (JSON API enabled)."
    exit 0
  fi
  sleep 2
done
echo "SearXNG started but its JSON API did not answer in time; check: docker logs ${NAME}"
exit 0
