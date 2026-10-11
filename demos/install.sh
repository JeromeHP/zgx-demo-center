#!/usr/bin/env bash
# HP ZGX Demo Center — Installer (bash bootstrap)
#
# This script is idempotent: safe to re-run. Each check determines if the
# underlying component is already present and skips the install step if so.
# Use --force to reinstall everything, --dry-run to preview without changes.

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Colors and logging helpers
# ──────────────────────────────────────────────────────────────────────────────
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
RED=$'\033[31m'
BLUE=$'\033[34m'
DIM=$'\033[2m'
BOLD=$'\033[1m'
RESET=$'\033[0m'

log_section() {
    printf '\n%s%s───────────────────────────────────────────────────%s\n' "$BOLD" "$BLUE" "$RESET"
    printf '%s%s%s%s\n' "$BOLD" "$BLUE" "$1" "$RESET"
    printf '%s%s───────────────────────────────────────────────────%s\n' "$BOLD" "$BLUE" "$RESET"
}

log_check()   { printf '%s✓ %s%s\n' "$GREEN" "$1" "$RESET"; }
log_install() { printf '%s⏳ %s...%s\n' "$YELLOW" "$1" "$RESET"; }
log_done()    { printf '%s✓ %s%s\n' "$GREEN" "$1" "$RESET"; }
log_skip()    { printf '%s✓ %s (already present)%s\n' "$DIM" "$1" "$RESET"; }
log_error()   { printf '%s✗ %s%s\n' "$RED" "$1" "$RESET" >&2; }
log_info()    { printf '%sℹ %s%s\n' "$BLUE" "$1" "$RESET"; }

# ──────────────────────────────────────────────────────────────────────────────
# Flag parsing
# ──────────────────────────────────────────────────────────────────────────────
DRY_RUN=false
FORCE=false
SKIP_DEMOS=()
ORIG_ARGS=("$@")

usage() {
    cat <<EOF
HP ZGX Demo Center — Installer

Usage: ./install.sh [OPTIONS]

Options:
  --help              Show this help message and exit
  --dry-run           Show what would happen without making changes
  --force             Reinstall everything even if already present
  --skip-<demo>       Skip a specific demo by id (can be repeated)
                      e.g. --skip-pharmacy --skip-maritime

Example:
  ./install.sh --dry-run
  ./install.sh --skip-maritime --skip-pii
  ./install.sh --force
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)
            usage
            exit 0
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --skip-preflight)
            # Hidden advanced flag: install.sh takes no action, just forwards
            # it to install.py (via ORIG_ARGS). Intentionally not in --help.
            shift
            ;;
        --skip-*)
            SKIP_DEMOS+=("${1#--skip-}")
            shift
            ;;
        *)
            log_error "Unknown flag: $1"
            usage >&2
            exit 1
            ;;
    esac
done

# ──────────────────────────────────────────────────────────────────────────────
# Banner
# ──────────────────────────────────────────────────────────────────────────────
printf '%s===================================================%s\n' "$BOLD" "$RESET"
printf '%sHP ZGX Demo Center — Installer%s\n' "$BOLD" "$RESET"
printf '%s===================================================%s\n' "$BOLD" "$RESET"
log_info "Beginning installation. This may take 30-60 minutes on a fresh system."

if $DRY_RUN; then
    log_info "DRY-RUN mode: no changes will be made."
fi
if $FORCE; then
    log_info "FORCE mode: components will be reinstalled even if present."
fi
if [[ ${#SKIP_DEMOS[@]} -gt 0 ]]; then
    log_info "Skipping demos: ${SKIP_DEMOS[*]}"
fi

# Status tracking for final summary
PYTHON_STATUS="✗"
CONDA_STATUS="✗"
ZGX_ENV_STATUS="✗"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ──────────────────────────────────────────────────────────────────────────────
# Section: ~/demos path
#
# Runs FIRST — before any prereq or Python work. Demos hardcode $HOME/demos/...
# by convention (see ensure_demos_link.sh), so nothing downstream is meaningful
# until that path resolves here. A conflict is fatal, not a warning: continuing
# would install into one tree and launch from another.
# ──────────────────────────────────────────────────────────────────────────────
log_section "Demo Path"

# shellcheck source=ensure_demos_link.sh
source "$SCRIPT_DIR/ensure_demos_link.sh"
if ! ensure_demos_link "$SCRIPT_DIR" "$DRY_RUN"; then
    log_error "Cannot continue until ~/demos resolves to this checkout."
    exit 1
fi

# ──────────────────────────────────────────────────────────────────────────────
# Section: Prerequisites
# ──────────────────────────────────────────────────────────────────────────────
log_section "Prerequisites"

# (a) Python 3.12+
python_ok=false
if command -v python3.12 >/dev/null 2>&1; then
    pyver=$(python3.12 --version 2>&1 | awk '{print $2}')
    pymajor=$(echo "$pyver" | cut -d. -f1)
    pyminor=$(echo "$pyver" | cut -d. -f2)
    if [[ "$pymajor" -eq 3 && "$pyminor" -ge 12 ]]; then
        python_ok=true
    fi
fi

if $python_ok && ! $FORCE; then
    log_skip "Python 3.12 ($pyver)"
    PYTHON_STATUS="✓"
else
    if $DRY_RUN; then
        log_info "would install Python 3.12 via apt"
        PYTHON_STATUS="(would install)"
    else
        log_install "Installing Python 3.12"
        if sudo apt update && sudo apt install -y python3.12 python3.12-venv; then
            log_done "Python 3.12 installed"
            PYTHON_STATUS="✓"
        else
            log_error "Failed to install Python 3.12 via apt"
            exit 1
        fi
    fi
fi

# (b) conda (miniforge preferred)
conda_ok=false
conda_bin=""
if command -v conda >/dev/null 2>&1; then
    conda_ok=true
    conda_bin="$(command -v conda)"
else
    # Non-interactive bash may not have conda on PATH even when installed.
    # Probe known install locations.
    for candidate in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
        if [[ -x "$candidate" ]]; then
            conda_ok=true
            conda_bin="$candidate"
            export PATH="$(dirname "$candidate"):$PATH"
            break
        fi
    done
fi

if $conda_ok && ! $FORCE; then
    cver=$("$conda_bin" --version 2>&1 | awk '{print $2}')
    log_skip "conda ($cver)"
    CONDA_STATUS="✓"
else
    if $DRY_RUN; then
        log_info "would install miniforge to ~/miniforge3"
        CONDA_STATUS="(would install)"
    else
        log_install "Installing miniforge to ~/miniforge3"
        installer_url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
        tmp_installer="/tmp/miniforge_installer.sh"
        if curl -fsSL "$installer_url" -o "$tmp_installer" \
           && bash "$tmp_installer" -b -p "$HOME/miniforge3" \
           && "$HOME/miniforge3/bin/conda" init bash >/dev/null; then
            rm -f "$tmp_installer"
            log_done "miniforge installed at ~/miniforge3"
            # Make conda available in current shell
            export PATH="$HOME/miniforge3/bin:$PATH"
            CONDA_STATUS="✓"
        else
            log_error "Failed to install miniforge"
            exit 1
        fi
    fi
fi

# (c) zgx conda environment
# No conda.sh sourcing needed — the handoff below resolves the env python by
# absolute path (no `conda activate`). `env list` errors are deliberately not
# swallowed, so a broken conda surfaces instead of reading as "env missing".
zgx_env_ok=false
if [[ -n "$conda_bin" ]]; then
    if "$conda_bin" env list | grep -qE "^zgx[[:space:]]"; then
        zgx_env_ok=true
    fi
fi

if $zgx_env_ok && ! $FORCE; then
    log_skip "conda env 'zgx'"
    ZGX_ENV_STATUS="✓"
else
    if $DRY_RUN; then
        log_info "would create conda env 'zgx' with python=3.12"
        ZGX_ENV_STATUS="(would create)"
    else
        log_install "Creating conda env 'zgx' with python=3.12"
        if conda create -n zgx python=3.12 -y; then
            log_done "conda env 'zgx' created"
            ZGX_ENV_STATUS="✓"
        else
            log_error "Failed to create conda env 'zgx'"
            exit 1
        fi
    fi
fi

# ──────────────────────────────────────────────────────────────────────────────
# Section: Handoff to Python orchestrator
# ──────────────────────────────────────────────────────────────────────────────
log_section "Python Orchestrator"

PYTHON_ORCHESTRATOR="$SCRIPT_DIR/install.py"

if $DRY_RUN; then
    log_info "would resolve the zgx env python by absolute path and hand off to install.py"
else
    # Resolve the zgx env's python by ABSOLUTE PATH — do NOT `conda activate`.
    # `conda activate` is a shell function that exists only after conda.sh is
    # sourced; on a box where this script just installed miniforge that hook was
    # never loaded, so the handoff died on an undefined function (with conda's
    # real error hidden by 2>/dev/null). The env lives at a fixed path — resolve
    # it directly: no shell function, no CONDA_PREFIX, no swallowed errors.
    zgx_python="$HOME/miniforge3/envs/zgx/bin/python"
    if [[ ! -x "$zgx_python" ]]; then
        log_error "Could not locate zgx env python at $zgx_python"
        exit 1
    fi
    actual_pyver=$("$zgx_python" --version 2>&1 | awk '{print $2}')
    actual_major=$(echo "$actual_pyver" | cut -d. -f1)
    actual_minor=$(echo "$actual_pyver" | cut -d. -f2)
    if [[ "$actual_major" -eq 3 && "$actual_minor" -ge 12 ]]; then
        log_check "zgx env python: $actual_pyver ($zgx_python)"
    else
        log_error "zgx env python is $actual_pyver — expected 3.12.x"
        exit 1
    fi

    if [[ -f "$PYTHON_ORCHESTRATOR" ]]; then
        log_info "Handing off to install.py"
        "$zgx_python" "$PYTHON_ORCHESTRATOR" "${ORIG_ARGS[@]}"
    else
        log_info "Python orchestrator (install.py) not yet implemented — this is the expected state for the Phase 4 bootstrap session."
        log_info "Bootstrap finished without orchestration handoff."
    fi
fi

# ──────────────────────────────────────────────────────────────────────────────
# Final summary
# ──────────────────────────────────────────────────────────────────────────────
printf '\n%s===================================================%s\n' "$BOLD" "$RESET"
printf '%sBootstrap complete.%s\n' "$BOLD" "$RESET"
printf '%s===================================================%s\n' "$BOLD" "$RESET"
printf 'Python 3.12 %s  conda %s  zgx env %s\n' "$PYTHON_STATUS" "$CONDA_STATUS" "$ZGX_ENV_STATUS"
