#!/usr/bin/env bash
# HP ZGX Demo Center — restrict all access to the tailnet
#
# WHAT THIS DOES
#   Makes every Demo Center service reachable over Tailscale and over loopback,
#   and unreachable from the LAN. Nothing about how the demos run changes: they
#   keep binding 0.0.0.0, so the install API's readiness checks (which probe
#   127.0.0.1) keep working exactly as before. Access is filtered at the network
#   layer instead, which is the only place it can be done without breaking those
#   checks.
#
# WHY NOT JUST BIND TO THE TAILNET IP
#   install_plans.py health-checks every demo at http://localhost:PORT/... and
#   check_port_alive() connects to 127.0.0.1. A service bound only to
#   100.x.y.z would fail both, so every demo would read as "never started" —
#   the exact failure this kit was just repaired for.
#
# THE DOCKER EXCEPTION
#   ufw does NOT filter Docker's published ports: dockerd DNATs them in the nat
#   table, ahead of ufw's INPUT rules. Four demos use `network_mode: host` and
#   are covered by ufw normally, but Fed-Navy (maritime) publishes "8000:8000",
#   so it needs a rule in Docker's own DOCKER-USER chain. That chain is flushed
#   on every dockerd restart, hence the systemd unit that re-applies it.
#
# USAGE
#   sudo bash lockdown-tailnet.sh --dry-run     # show every change, apply none
#   sudo bash lockdown-tailnet.sh               # apply (keeps LAN SSH as a lifeline)
#   sudo bash lockdown-tailnet.sh --no-lan-ssh  # also close SSH on the LAN
#   sudo bash lockdown-tailnet.sh --pin-ollama  # also bind Ollama to localhost
#   sudo bash lockdown-tailnet.sh --undo        # remove everything this added
#
# Re-running is safe; every step is idempotent.

set -euo pipefail

DRY_RUN=false; LAN_SSH=true; PIN_OLLAMA=false; UNDO=false
for arg in "$@"; do
    case "$arg" in
        --dry-run)    DRY_RUN=true ;;
        --no-lan-ssh) LAN_SSH=false ;;
        --pin-ollama) PIN_OLLAMA=true ;;
        --undo)       UNDO=true ;;
        -h|--help)    sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

ok()   { printf '\033[32m✓ %s\033[0m\n' "$1"; }
info() { printf '\033[34mℹ %s\033[0m\n' "$1"; }
warn() { printf '\033[33m⚠ %s\033[0m\n' "$1"; }
err()  { printf '\033[31m✗ %s\033[0m\n' "$1" >&2; }
run()  { if $DRY_RUN; then printf '  would run: %s\n' "$*"; else "$@"; fi; }

[[ ${EUID:-$(id -u)} -eq 0 ]] || { err "Run with sudo."; exit 1; }

TS_UNIT=/etc/systemd/system/zgx-docker-lan-block.service
OLLAMA_DROPIN=/etc/systemd/system/ollama.service.d/10-localhost.conf

# ── Undo ────────────────────────────────────────────────────────────────────
if $UNDO; then
    info "Reverting tailnet lockdown"
    run ufw --force disable
    run systemctl disable --now zgx-docker-lan-block.service 2>/dev/null || true
    run rm -f "$TS_UNIT"
    run rm -f "$OLLAMA_DROPIN"
    run systemctl daemon-reload
    run systemctl restart ollama 2>/dev/null || true
    ok "Reverted. Services are reachable on the LAN again."
    exit 0
fi

# ── Guard: never lock a headless box out of its own network ─────────────────
# This box has no monitor. If Tailscale is down when the firewall goes up and
# LAN SSH is also closed, the only way back in is physical access.
if ! tailscale status >/dev/null 2>&1; then
    err "Tailscale is not up. Refusing to apply a tailnet-only firewall."
    err "Bring it up first:  sudo tailscale up"
    exit 1
fi
TS_IF=tailscale0
ip link show "$TS_IF" >/dev/null 2>&1 || { err "No $TS_IF interface."; exit 1; }
TS_IP="$(tailscale ip -4 2>/dev/null | head -1)"
ok "Tailscale up — $TS_IF ($TS_IP)"

# The physical interface(s) traffic would otherwise arrive on. Docker bridges are
# excluded: blocking those would cut container networking, not LAN access.
mapfile -t LAN_IFS < <(ip -o link show up \
    | awk -F': ' '{print $2}' | cut -d@ -f1 \
    | grep -vE '^(lo|tailscale0|docker0|br-[0-9a-f]+|veth)' || true)
[[ ${#LAN_IFS[@]} -gt 0 ]] || warn "No physical interface detected — Docker rule will be skipped."
info "LAN interfaces: ${LAN_IFS[*]:-none}"

# Is this very session riding the tailnet? If not, applying this drops it.
SSH_SRC="${SSH_CLIENT%% *}"
if [[ -n "$SSH_SRC" ]]; then
    if [[ "$SSH_SRC" =~ ^100\.(6[4-9]|[7-9][0-9]|1[0-1][0-9]|12[0-7])\. ]]; then
        ok "This SSH session is over Tailscale ($SSH_SRC) — it will survive."
    else
        warn "This SSH session is from $SSH_SRC, which is NOT a tailnet address."
        $LAN_SSH && warn "LAN SSH stays open (default), so you will not be cut off." \
                 || { err "--no-lan-ssh would disconnect you right now. Aborting."; exit 1; }
    fi
fi

# ── 1. ufw: deny inbound except loopback, the tailnet, and WireGuard ────────
info "Configuring ufw"
# Docker manages its own FORWARD rules; ufw's default DROP there breaks container
# networking. Ingress is controlled by INPUT + DOCKER-USER below instead.
if [[ -f /etc/default/ufw ]]; then
    run sed -i 's/^DEFAULT_FORWARD_POLICY=.*/DEFAULT_FORWARD_POLICY="ACCEPT"/' /etc/default/ufw
fi
run ufw --force default deny incoming
run ufw --force default allow outgoing
run ufw allow in on lo
run ufw allow in on "$TS_IF"
# WireGuard's own packets arrive on the PHYSICAL interface, so the tailscale0
# rule above cannot cover them. Without this, Tailscale falls back to a DERP
# relay: it still works, but every demo frame takes a detour through the
# internet instead of going peer-to-peer.
run ufw allow 41641/udp comment 'tailscale wireguard'
if $LAN_SSH; then
    run ufw allow 22/tcp comment 'lifeline SSH — headless box, do not remove lightly'
    info "LAN SSH kept open as a lifeline (--no-lan-ssh to close it)"
else
    run ufw delete allow 22/tcp 2>/dev/null || true
    warn "LAN SSH closed — Tailscale is now the ONLY way into this box."
fi
run ufw --force enable
ok "ufw active"

# ── 2. Docker's published ports (Fed-Navy :8000) ────────────────────────────
# ufw cannot see these. DOCKER-USER is the chain dockerd consults first for
# traffic destined to containers, and it is the supported place to filter it.
if [[ ${#LAN_IFS[@]} -gt 0 ]]; then
    info "Blocking LAN access to Docker-published ports"
    for IFACE in "${LAN_IFS[@]}"; do
        if ! iptables -C DOCKER-USER -i "$IFACE" -j DROP 2>/dev/null; then
            run iptables -I DOCKER-USER -i "$IFACE" -j DROP
        else
            info "  rule for $IFACE already present"
        fi
    done

    # DOCKER-USER is recreated empty every time dockerd restarts, so the rule
    # must be re-applied on boot or it silently disappears.
    if ! $DRY_RUN; then
        cat > "$TS_UNIT" <<UNIT
[Unit]
Description=Block LAN access to Docker-published demo ports (tailnet-only policy)
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c 'for i in ${LAN_IFS[*]}; do /usr/sbin/iptables -C DOCKER-USER -i \$i -j DROP 2>/dev/null || /usr/sbin/iptables -I DOCKER-USER -i \$i -j DROP; done'

[Install]
WantedBy=multi-user.target
UNIT
    else
        printf '  would write: %s\n' "$TS_UNIT"
    fi
    run systemctl daemon-reload
    run systemctl enable --now zgx-docker-lan-block.service
    ok "Docker LAN block applied and made persistent"
fi

# ── 3. Optional: Ollama on localhost only ───────────────────────────────────
# Every demo reaches Ollama over localhost — the three Ollama-backed containers
# use network_mode: host precisely so they can — so nothing needs it on the wire.
if $PIN_OLLAMA; then
    warn "Restarting Ollama will unload resident models; the next demo query pays a cold load (llama3.1:70b is ~42GB)."
    if ! $DRY_RUN; then
        mkdir -p "$(dirname "$OLLAMA_DROPIN")"
        cat > "$OLLAMA_DROPIN" <<'UNIT'
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
UNIT
    else
        printf '  would write: %s (OLLAMA_HOST=127.0.0.1:11434)\n' "$OLLAMA_DROPIN"
    fi
    run systemctl daemon-reload
    run systemctl restart ollama
    ok "Ollama bound to 127.0.0.1 only"
fi

# ── Verify ──────────────────────────────────────────────────────────────────
echo
if $DRY_RUN; then
    info "Dry run — nothing was changed."
    exit 0
fi
info "Result — anything still on 0.0.0.0 is reachable ONLY via $TS_IF now:"
ss -tlnH | awk '{print "    "$4}' | sort -u
echo
info "Verify from a LAN device (should all fail/time out):"
echo "    curl -m 5 http://192.168.1.33:8085/zgx-demo-launcher.html"
info "Verify from a tailnet device (should work):"
echo "    curl -m 5 http://${TS_IP}:8085/zgx-demo-launcher.html"
echo
info "Undo everything:  sudo bash $0 --undo"
