#!/usr/bin/env bash
# HP ZGX Demo Center — delivery bundle builder
#
# Builds the single authed-browser download the rep takes onto a ZGX Nano: the
# tracked repo (clean `git archive` of HEAD) plus the two native-asset tarballs
# (assets/*.tar.gz — gitignored, shipped only here). HP-internal hosts (SharePoint,
# GitHub Enterprise) 403/401 an anonymous scripted fetch, so the assets can't be
# pulled at install time; they travel inside this zip. The rep downloads it via an
# authenticated browser, unzips in $HOME (yielding ~/demos/), and runs
# ~/demos/install.sh — Phase 4 then stages the assets LOCAL-FIRST from
# ~/demos/assets/ with no network. Reproducible: repo from git HEAD, assets verbatim.

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Colors and logging helpers (mirror install.sh)
# ──────────────────────────────────────────────────────────────────────────────
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
RED=$'\033[31m'
BLUE=$'\033[34m'
BOLD=$'\033[1m'
RESET=$'\033[0m'

log()  { echo "${BLUE}ℹ ${RESET}$*"; }
ok()   { echo "${GREEN}✓ ${RESET}$*"; }
warn() { echo "${YELLOW}⚠ ${RESET}$*"; }
die()  { echo "${RED}✗ ${RESET}$*" >&2; exit 1; }

# Repo root = directory of this script (never hardcode $HOME).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

VIDEO_TARBALL="assets/zgx-videos.tar.gz"
MODEL_TARBALL="assets/yolov8s-onnx.tar.gz"
OUT_ZIP="$HOME/zgx-demo-center-bundle.zip"

echo
echo "${BOLD}${BLUE}─── Building HP ZGX Demo Center delivery bundle ───${RESET}"

# 1. Verify the two asset tarballs are present in the repo.
[ -f "$VIDEO_TARBALL" ] || die "Missing $VIDEO_TARBALL — copy it into assets/ before bundling."
[ -f "$MODEL_TARBALL" ] || die "Missing $MODEL_TARBALL — copy it into assets/ before bundling."
ok "Asset tarballs present: $VIDEO_TARBALL, $MODEL_TARBALL"

# 2. Clean export of TRACKED files (demos/ prefix) into a temp staging dir.
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT
log "Exporting tracked files from HEAD → $STAGING/demos/"
git archive --prefix=demos/ HEAD | tar -x -C "$STAGING"
ok "Exported clean repo snapshot (git HEAD)"

# 3. Copy the two tarballs into the staged assets/ (they are gitignored, so the
#    archive does not contain them).
mkdir -p "$STAGING/demos/assets"
cp "$VIDEO_TARBALL" "$MODEL_TARBALL" "$STAGING/demos/assets/"
ok "Copied native-asset tarballs into the staged demos/assets/"

# 4. Zip the staged demos/ dir → $HOME (overwrite if it exists).
rm -f "$OUT_ZIP"
log "Creating bundle zip → $OUT_ZIP"
( cd "$STAGING" && zip -qr "$OUT_ZIP" demos )
[ -f "$OUT_ZIP" ] || die "zip did not produce $OUT_ZIP"

# 5. Summary.
SIZE="$(du -h "$OUT_ZIP" | cut -f1)"
echo
ok "Bundle built: $OUT_ZIP ($SIZE)"
log "Unzipping this in \$HOME yields ~/demos/ (repo + assets/) — then run ~/demos/install.sh"
