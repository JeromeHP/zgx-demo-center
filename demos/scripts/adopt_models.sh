#!/usr/bin/env bash
# Move downloaded models from an old ~/demos tree into this checkout, safely.
#
# Why this exists: ensure_demos_link.sh refuses when ~/demos is a real directory
# that is not this checkout, and the per-demo models live INSIDE that tree
# ({dir}/models -- maritime 11GB, pharmacy 28GB, pii 20GB, medical 4GB). A
# first AMD install hit this with 73GB sitting there. Re-downloading that is
# hours; an unchecked `mv` one-liner in an error message is how you lose it.
#
# What it does, per demo directory:
#   - only touches paths that exist in BOTH trees (never invents a destination)
#   - refuses to overwrite an existing models/ dir in this checkout
#   - reports whether the move is a rename (instant, same filesystem) or a copy
#   - --apply is required; the default is a dry run
#
# Usage:
#   bash scripts/adopt_models.sh /home/you/demos            # dry run
#   bash scripts/adopt_models.sh /home/you/demos --apply
set -uo pipefail

GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; BLUE=$'\033[34m'; RESET=$'\033[0m'
log()  { printf '%sℹ %s%s\n' "$BLUE" "$*" "$RESET"; }
# Integer GB division reported 602MB as "~0GB". Show MB below a gigabyte.
human_mb() { if [[ "$1" -ge 1024 ]]; then printf '~%dGB' "$(( $1 / 1024 ))"; else printf '%dMB' "$1"; fi; }
ok()   { printf '%s✓ %s%s\n' "$GREEN" "$*" "$RESET"; }
warn() { printf '%s⚠ %s%s\n' "$YELLOW" "$*" "$RESET"; }
die()  { printf '%s✗ %s%s\n' "$RED" "$*" "$RESET" >&2; exit 1; }

SRC="${1:-}"
APPLY="no"
[[ "${2:-}" == "--apply" ]] && APPLY="yes"
[[ -n "$SRC" ]] || die "usage: bash scripts/adopt_models.sh <old-demos-dir> [--apply]"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$(cd "$SRC" 2>/dev/null && pwd)" || die "not a directory: ${1}"
[[ "$SRC" != "$HERE" ]] || die "source and destination are the same directory"

log "from: $SRC"
log "to:   $HERE"
[[ "$APPLY" == "yes" ]] || warn "DRY RUN — nothing will move. Re-run with --apply."
echo

# Same filesystem? Then mv is a rename: instant, regardless of size.
src_fs="$(df -P "$SRC" | awk 'NR==2{print $1}')"
dst_fs="$(df -P "$HERE" | awk 'NR==2{print $1}')"
if [[ "$src_fs" == "$dst_fs" ]]; then
    ok "Same filesystem ($src_fs) — moves are instant renames, no data copied."
else
    warn "DIFFERENT filesystems ($src_fs -> $dst_fs) — this will physically copy."
    warn "Check free space on $dst_fs before using --apply."
fi
echo

moved=0; skipped=0; total_mb=0
shopt -s nullglob
for src_models in "$SRC"/*/models; do
    demo="$(basename "$(dirname "$src_models")")"
    dst_demo="$HERE/$demo"
    dst_models="$dst_demo/models"
    size_mb="$(du -sm "$src_models" 2>/dev/null | awk '{print $1}')"
    size_mb="${size_mb:-0}"

    if [[ ! -d "$dst_demo" ]]; then
        warn "skip  $demo — no such demo in this checkout (nothing to attach it to)"
        skipped=$((skipped + 1)); continue
    fi
    if [[ -e "$dst_models" ]]; then
        warn "skip  $demo — $dst_models already exists; refusing to merge or nest"
        warn "        inspect both, then move by hand if you want the older copy"
        skipped=$((skipped + 1)); continue
    fi

    if [[ "$APPLY" == "yes" ]]; then
        if mv "$src_models" "$dst_models"; then
            ok "moved $demo  (${size_mb}MB)"
            moved=$((moved + 1)); total_mb=$((total_mb + size_mb))
        else
            die "failed to move $src_models — stopping before anything else changes"
        fi
    else
        log "would move $demo  (${size_mb}MB)  ->  $dst_models"
        moved=$((moved + 1)); total_mb=$((total_mb + size_mb))
    fi
done

echo
if [[ "$APPLY" == "yes" ]]; then
    ok "Moved $moved model dir(s), $(human_mb "$total_mb"). Skipped: $skipped."
    echo
    log "Next, point ~/demos at this checkout:"
    log "  mv '$SRC' '${SRC}.bak' && ln -s '$HERE' '$SRC'"
    log "Then re-run ./install.sh — it will skip what is already downloaded."
else
    log "Would move $moved model dir(s), $(human_mb "$total_mb"). Would skip: $skipped."
    log "Re-run with --apply to do it."
fi
