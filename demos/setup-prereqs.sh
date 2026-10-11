#!/usr/bin/env bash
# HP ZGX Demo Center — host prerequisites installer (run with sudo, ONCE).
#
# Installs the system-level prereqs install.sh does NOT: Docker Engine + Compose,
# GPU container access, Ollama, Chromium, and WM helpers (wmctrl, xdotool).
# Enables the Docker/Ollama daemons and adds the invoking user to the docker and
# ollama groups. Idempotent, and both arm64- and amd64-aware.
#
# GPU container access is vendor-specific and detected, not assumed:
#   NVIDIA (nvidia-smi present) — installs the NVIDIA Container Toolkit and
#                                 wires the nvidia runtime into Docker.
#   AMD    (/dev/kfd present)   — installs NO toolkit (ROCm has none); adds the
#                                 user to the render + video groups instead.
#   neither                     — skips GPU setup; Ollama-backed demos still work.
#
# HARD STOP: does NOT run install.sh, and a REBOOT is required afterward (group
# membership only takes effect on a fresh login). Reboot, then run:
#   bash ~/demos/install.sh
set -euo pipefail

GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; BLUE=$'\033[34m'
BOLD=$'\033[1m'; RESET=$'\033[0m'
log()  { printf '%sℹ %s%s\n' "$BLUE" "$*" "$RESET"; }
ok()   { printf '%s✓ %s%s\n' "$GREEN" "$*" "$RESET"; }
warn() { printf '%s⚠ %s%s\n' "$YELLOW" "$*" "$RESET"; }
die()  { printf '%s✗ %s%s\n' "$RED" "$*" "$RESET" >&2; exit 1; }

# ── Must be root ──────────────────────────────────────────────────────────────
if [[ "${EUID}" -ne 0 ]]; then
    die "Run as root:  sudo bash setup-prereqs.sh"
fi

# The human we add to the docker group is the user who invoked sudo, NOT root.
TARGET_USER="${SUDO_USER:-}"
if [[ -z "$TARGET_USER" || "$TARGET_USER" == "root" ]]; then
    warn "SUDO_USER not set — cannot determine the non-root user for the docker group."
    warn "Re-run via:  sudo bash setup-prereqs.sh   (not from a root login shell)."
    TARGET_USER=""
fi

ARCH="$(dpkg --print-architecture)"   # arm64 on the ZGX Nano, amd64 on Strix Halo
# shellcheck disable=SC1091
. /etc/os-release                      # provides $VERSION_CODENAME, $PRETTY_NAME
log "Architecture: ${ARCH} | OS: ${PRETTY_NAME:-unknown}"

# ── Accelerator detection ─────────────────────────────────────────────────────
# This script used to install the NVIDIA Container Toolkit unconditionally and
# then run `nvidia-ctk runtime configure` with no guard. Under `set -e` that
# aborted the whole run on an AMD box -- and on x86_64 it was worse than an
# abort: NVIDIA's apt repo HAS amd64 packages, so the toolkit installed fine and
# then wired a `nvidia` runtime into /etc/docker/daemon.json that does not
# exist, leaving Docker misconfigured rather than obviously broken.
#
# Deliberately detected in plain bash rather than via demos/zgx_hardware.py:
# this is the earliest bootstrap step, runs as root before any dependency is
# installed, and must not acquire a Python requirement. The two checks match
# that module's -- nvidia-smi for CUDA, /dev/kfd for ROCm. /dev/kfd and NOT
# /dev/dri, because an NVIDIA box exposes /dev/dri/renderD128 too.
ACCEL="none"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    ACCEL="nvidia"
elif [[ -e /dev/kfd ]]; then
    ACCEL="amd"
fi
if [[ -n "${ZGX_HW_PROFILE:-}" ]]; then
    case "$ZGX_HW_PROFILE" in
        nvidia-cuda) ACCEL="nvidia" ;;
        amd-rocm)    ACCEL="amd" ;;
        cpu-only)    ACCEL="none" ;;
    esac
    warn "Accelerator forced to '${ACCEL}' by ZGX_HW_PROFILE=${ZGX_HW_PROFILE}"
fi
log "Accelerator: ${ACCEL}"

printf '%s=== HP ZGX Demo Center — host prerequisites ===%s\n' "$BOLD" "$RESET"

# ── Status tracking for the end-of-run summary ────────────────────────────────
# Each component marks itself "already present" or "installed"; group changes
# flip REBOOT_REQUIRED. The summary doubles as a survey of what the factory image
# already shipped (which varies between images).
ALREADY_PRESENT=()
INSTALLED_NOW=()
REBOOT_REQUIRED="no"
mark_present()   { ALREADY_PRESENT+=("$1"); }
mark_installed() { INSTALLED_NOW+=("$1"); }

# ── 1. Docker Engine + Compose plugin (Docker official apt repo) ──────────────
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    ok "Docker Engine + Compose plugin already present ($(docker --version | awk '{print $3}' | tr -d ,))"
    mark_present "Docker Engine + Compose"
else
    log "Installing Docker Engine + Compose plugin from Docker's official apt repo"
    apt-get update -y
    apt-get install -y ca-certificates curl gnupg
    install -m 0755 -d /etc/apt/keyrings
    if [[ ! -f /etc/apt/keyrings/docker.gpg ]]; then
        curl -fsSL "https://download.docker.com/linux/ubuntu/gpg" \
            | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        chmod a+r /etc/apt/keyrings/docker.gpg
    fi
    echo "deb [arch=${ARCH} signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
        > /etc/apt/sources.list.d/docker.list
    apt-get update -y
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    ok "Docker Engine + Compose plugin installed"
    mark_installed "Docker Engine + Compose"
fi

# ── 2. GPU container access ───────────────────────────────────────────────────
# NVIDIA and AMD do this completely differently, which is why one unconditional
# branch could never have covered both:
#   NVIDIA — the NVIDIA Container Toolkit + `nvidia-ctk runtime configure`
#   AMD    — no toolkit at all; containers get `--device /dev/kfd --device
#            /dev/dri` and the invoking user must be in the render+video groups
if [[ "$ACCEL" == "amd" ]]; then
    log "AMD/ROCm detected — skipping the NVIDIA Container Toolkit (not applicable)"
    ok "ROCm needs no container toolkit; containers use /dev/kfd + /dev/dri directly"
    mark_present "GPU container access (ROCm device nodes)"

    # Without render+video the device nodes exist but cannot be opened, and the
    # failure surfaces much later as a bare permission error inside a container.
    if [[ -n "$TARGET_USER" ]]; then
        for grp in render video; do
            if ! getent group "$grp" >/dev/null 2>&1; then
                warn "group '$grp' does not exist on this system — skipping"
                continue
            fi
            if id -nG "$TARGET_USER" 2>/dev/null | tr ' ' '\n' | grep -qx "$grp"; then
                ok "User '$TARGET_USER' already in the '$grp' group"
            else
                log "Adding user '$TARGET_USER' to the '$grp' group (ROCm device access)"
                usermod -aG "$grp" "$TARGET_USER"
                ok "Added '$TARGET_USER' to '$grp' (takes effect after reboot/re-login)"
                REBOOT_REQUIRED="yes"
            fi
        done
    fi

    if command -v rocminfo >/dev/null 2>&1; then
        ok "ROCm userspace present ($(rocminfo 2>/dev/null | grep -m1 'Marketing Name' | cut -d: -f2- | xargs || echo 'device listed'))"
    else
        warn "rocminfo not found — install ROCm for this distro before running install.sh."
        warn "  Ubuntu: follow AMD's amdgpu-install instructions for your kernel."
        warn "  The Ollama-backed demos work without it; ai-vision's GPU provider does not."
    fi

elif [[ "$ACCEL" == "none" ]]; then
    warn "No accelerator detected — skipping GPU container setup entirely."
    warn "  The Ollama-backed demos still install and run (on CPU); the"
    warn "  accelerator-bound ones will be skipped by install.py with a reason."

elif command -v nvidia-ctk >/dev/null 2>&1; then
    ok "NVIDIA Container Toolkit already present ($(nvidia-ctk --version 2>/dev/null | head -1 | awk '{print $NF}'))"
    mark_present "NVIDIA Container Toolkit"
else
    log "Installing NVIDIA Container Toolkit from NVIDIA's apt repo"
    install -m 0755 -d /etc/apt/keyrings
    if [[ ! -f /etc/apt/keyrings/nvidia-container-toolkit.gpg ]]; then
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | gpg --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit.gpg
        chmod a+r /etc/apt/keyrings/nvidia-container-toolkit.gpg
    fi
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit.gpg] https://#g' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -y
    apt-get install -y nvidia-container-toolkit
    ok "NVIDIA Container Toolkit installed"
    mark_installed "NVIDIA Container Toolkit"
fi

# Always (re)configure + restart — idempotent. Ensures the nvidia runtime is
# wired into /etc/docker/daemon.json even when the toolkit was already present.
#
# Guarded on ACCEL: this call had no guard at all, so under `set -e` it aborted
# the entire script on any box without nvidia-ctk, and on an x86_64 AMD box
# (where the toolkit installs happily from NVIDIA's amd64 repo) it would have
# written a runtime into daemon.json that cannot exist there.
if [[ "$ACCEL" == "nvidia" ]]; then
    log "Configuring Docker to use the NVIDIA runtime"
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker 2>/dev/null || true
    ok "NVIDIA runtime configured in Docker"
fi

# ── 2b. Ollama (official installer) ───────────────────────────────────────────
# Provisioned here (not mid-install) so the local LLM runtime is ready before
# install.sh, keeping the one privileged touchpoint in this already-sudo script
# and install.sh hands-off. The installer sets up the systemd service and an
# 'ollama' user/group. Idempotent: skip if already present.
if command -v ollama >/dev/null 2>&1; then
    ok "Ollama already present ($(ollama --version 2>/dev/null | awk '{print $NF}' | head -1))"
    mark_present "Ollama"
else
    log "Installing Ollama via the official installer (curl https://ollama.com/install.sh | sh)"
    curl -fsSL https://ollama.com/install.sh | sh
    ok "Ollama installed"
    mark_installed "Ollama"
fi

# Enable + start the Ollama daemon (idempotent — safe even if already enabled).
log "Enabling + starting the Ollama daemon"
systemctl enable --now ollama
ok "Ollama daemon enabled and running"

# Add the invoking user to the 'ollama' group EXPLICITLY — under sudo the
# installer's "current user" is root, so the real human would never be added. We
# add TARGET_USER (SUDO_USER) and verify with `id`; the reboot gate handles it
# taking effect.
if [[ -n "$TARGET_USER" ]] && getent group ollama >/dev/null 2>&1; then
    if id -nG "$TARGET_USER" 2>/dev/null | tr ' ' '\n' | grep -qx ollama; then
        ok "User '$TARGET_USER' already in the ollama group"
    else
        log "Adding user '$TARGET_USER' to the ollama group"
        usermod -aG ollama "$TARGET_USER"
        if id -nG "$TARGET_USER" 2>/dev/null | tr ' ' '\n' | grep -qx ollama; then
            ok "Added '$TARGET_USER' to the ollama group (takes effect after reboot/re-login)"
            REBOOT_REQUIRED="yes"
        else
            die "usermod -aG ollama '$TARGET_USER' did not take — check 'getent group ollama'."
        fi
    fi
fi

# ── 3. Chromium (snap) ────────────────────────────────────────────────────────
if command -v chromium >/dev/null 2>&1 || command -v chromium-browser >/dev/null 2>&1 \
   || [[ -x /snap/bin/chromium ]]; then
    ok "Chromium already present"
    mark_present "Chromium"
else
    log "Installing Chromium (snap)"
    if command -v snap >/dev/null 2>&1; then
        snap install chromium
        ok "Chromium installed"
        mark_installed "Chromium"
    else
        warn "snap not found — install snapd then Chromium manually:"
        warn "  sudo apt-get install -y snapd && sudo snap install chromium"
    fi
fi

# ── 4. Window-manager helpers (apt) — wmctrl + xdotool for fullscreen ─────────
need_apt_wm=()
if command -v wmctrl  >/dev/null 2>&1; then ok "wmctrl already present";  mark_present "wmctrl";  else need_apt_wm+=(wmctrl);  fi
if command -v xdotool >/dev/null 2>&1; then ok "xdotool already present"; mark_present "xdotool"; else need_apt_wm+=(xdotool); fi
if [[ ${#need_apt_wm[@]} -gt 0 ]]; then
    log "Installing window-manager helpers: ${need_apt_wm[*]}"
    apt-get install -y "${need_apt_wm[@]}"
    ok "Installed: ${need_apt_wm[*]}"
    for _wm in "${need_apt_wm[@]}"; do mark_installed "$_wm"; done
fi

# ── 5. Enable + start the Docker daemon ───────────────────────────────────────
log "Enabling + starting the Docker daemon"
systemctl enable --now docker
ok "Docker daemon enabled and running"

# ── 6. Add the invoking user to the docker group ──────────────────────────────
if [[ -n "$TARGET_USER" ]]; then
    if id -nG "$TARGET_USER" 2>/dev/null | tr ' ' '\n' | grep -qx docker; then
        ok "User '$TARGET_USER' already in the docker group"
    else
        log "Adding user '$TARGET_USER' to the docker group"
        usermod -aG docker "$TARGET_USER"
        ok "Added '$TARGET_USER' to the docker group (takes effect after reboot/re-login)"
        REBOOT_REQUIRED="yes"
    fi
fi

# ── End-of-run summary — per-component report ─────────────────────────────────
printf '\n%s──────────────── Prerequisite summary ────────────────%s\n' "$BOLD" "$RESET"
if [[ ${#ALREADY_PRESENT[@]} -gt 0 ]]; then
    ok "Already present: $(IFS=', '; echo "${ALREADY_PRESENT[*]}")"
else
    log "Already present: (none — this looks like a bare image)"
fi
if [[ ${#INSTALLED_NOW[@]} -gt 0 ]]; then
    ok "Installed now:   $(IFS=', '; echo "${INSTALLED_NOW[*]}")"
else
    ok "Installed now:   (nothing — every component was already present)"
fi
if [[ "$REBOOT_REQUIRED" == "yes" ]]; then
    warn "Reboot required: YES (new docker/ollama group membership needs a fresh login)"
else
    log  "Reboot required: no group changes were made this run"
fi

# ── HARD STOP — reboot required, do NOT chain into install.sh ──────────────────
printf '\n%s===================================================%s\n' "$BOLD" "$RESET"
printf '%s%sPrerequisites installed. A REBOOT IS REQUIRED.%s\n' "$BOLD" "$YELLOW" "$RESET"
printf '%s===================================================%s\n' "$BOLD" "$RESET"
cat <<EOF

The docker/ollama group membership for '${TARGET_USER:-your user}' only takes
effect on a fresh login, so you MUST reboot before running the Demo Center
installer.

  1. Reboot:            sudo reboot
  2. After reboot, run: bash ~/demos/install.sh

This script intentionally does NOT run install.sh for you.
EOF
exit 0
