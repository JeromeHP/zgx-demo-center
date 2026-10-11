#!/usr/bin/env bash
# HP ZGX Demo Center — ~/demos path bootstrap (sourced, not executed)
#
# WHY THIS EXISTS
#   `~/demos` is the kit's load-bearing path, and that is deliberate. Demos come
#   from several contributors, and new ones routinely hardcode
#   `$HOME/demos/<demo>/...` in their launch scripts, model downloaders and
#   Modelfiles. The convention lets someone drop in a demo without knowing where
#   the repo was cloned. This function makes the convention TRUE regardless of
#   checkout location, so those hardcoded paths keep working everywhere.
#
#   Do NOT "fix" this by making the orchestration layer script-relative. That
#   would leave every contributed demo still pointing at ~/demos and split the
#   convention in two — core code on one path, contributions on another.
#
# CONTRACT
#   After ensure_demos_link returns 0, `$HOME/demos` resolves to this repo's
#   demos/ directory — either because it already IS that directory (bundle
#   installs unzip straight to ~/demos) or because it is a symlink we created.
#
#   It NEVER deletes or overwrites a real directory, and never repoints a link
#   that resolves somewhere live. Conflicts are loud failures with instructions,
#   never silent repairs — a wrong ~/demos must not be "fixed" into pointing at
#   the wrong checkout.
#
# USAGE
#   source "$(dirname "${BASH_SOURCE[0]}")/ensure_demos_link.sh"
#   ensure_demos_link "$SCRIPT_DIR" [dry_run]     # dry_run: "true" to preview

# Self-contained printers — this file is sourced by scripts that may or may not
# define the log_* helpers, so it must not depend on them.
_dl_ok()    { printf '\033[32m✓ %s\033[0m\n' "$1"; }
_dl_info()  { printf '\033[34mℹ %s\033[0m\n' "$1"; }
_dl_warn()  { printf '\033[33m⚠ %s\033[0m\n' "$1"; }
_dl_err()   { printf '\033[31m✗ %s\033[0m\n' "$1" >&2; }

# Resolve the home directory the link belongs in. Under `sudo` (setup-prereqs.sh
# territory) $HOME is root's, and a root-owned /root/demos would be worse than
# useless — it would satisfy no demo and be invisible to the real user.
_dl_home() {
    if [[ ${EUID:-$(id -u)} -eq 0 && -n "${SUDO_USER:-}" ]]; then
        getent passwd "$SUDO_USER" | cut -d: -f6
    else
        printf '%s' "$HOME"
    fi
}

# ensure_demos_link <repo_demos_dir> [dry_run]
#   Returns 0 if ~/demos resolves to <repo_demos_dir> (already, or after we made
#   it so). Returns 1 on an unresolvable conflict — caller should abort.
ensure_demos_link() {
    local repo_demos="${1:?ensure_demos_link: repo demos dir required}"
    local dry_run="${2:-false}"
    local home_dir link target

    repo_demos="$(readlink -f "$repo_demos" 2>/dev/null)" || {
        _dl_err "Cannot resolve repo demos directory: $1"
        return 1
    }
    if [[ ! -d "$repo_demos" ]]; then
        _dl_err "Not a directory: $repo_demos"
        return 1
    fi

    home_dir="$(_dl_home)"
    if [[ -z "$home_dir" ]]; then
        _dl_err "Could not determine home directory for the ~/demos link."
        return 1
    fi
    if [[ ${EUID:-$(id -u)} -eq 0 && -n "${SUDO_USER:-}" ]]; then
        _dl_warn "Running under sudo — using ${SUDO_USER}'s home ($home_dir), not root's."
    fi
    link="$home_dir/demos"

    # Case A: the repo already lives at ~/demos (bundle install unzipped to
    # $HOME, or someone cloned straight there). Nothing to do — and nothing to
    # touch. This is the canonical layout the scripts were written against.
    if [[ "$repo_demos" == "$link" ]]; then
        _dl_ok "~/demos is this checkout (no link needed)"
        return 0
    fi

    # Case B: nothing at ~/demos — the normal relocated-checkout path.
    if [[ ! -e "$link" && ! -L "$link" ]]; then
        if [[ "$dry_run" == "true" ]]; then
            _dl_info "would link ~/demos → $repo_demos"
            return 0
        fi
        if ln -s "$repo_demos" "$link" 2>/dev/null; then
            _dl_ok "Linked ~/demos → $repo_demos"
            return 0
        fi
        _dl_err "Failed to create ~/demos → $repo_demos (is $home_dir writable?)"
        return 1
    fi

    # Case C: a symlink is already there.
    if [[ -L "$link" ]]; then
        target="$(readlink -f "$link" 2>/dev/null)"

        if [[ "$target" == "$repo_demos" ]]; then
            _dl_ok "~/demos → $repo_demos (already correct)"
            return 0
        fi

        # Dangling link: points at nothing, so repointing destroys nothing.
        # This is the moved-or-renamed-checkout case and is safe to self-heal.
        if [[ ! -e "$link" ]]; then
            if [[ "$dry_run" == "true" ]]; then
                _dl_info "would repoint dangling ~/demos → $repo_demos"
                return 0
            fi
            if ln -sfn "$repo_demos" "$link" 2>/dev/null; then
                _dl_ok "Repointed dangling ~/demos → $repo_demos"
                return 0
            fi
            _dl_err "Failed to repoint ~/demos → $repo_demos"
            return 1
        fi

        # Live link to a DIFFERENT checkout. Repointing here would silently
        # steal a working install's path, so refuse and let a human decide.
        _dl_err "~/demos already points at a different checkout:"
        _dl_err "    ~/demos  → $target"
        _dl_err "    this one → $repo_demos"
        _dl_err "Installing would mix the two. Pick one:"
        _dl_err "  • use the existing checkout:  cd $target && ./install.sh"
        _dl_err "  • switch to this one:         ln -sfn '$repo_demos' '$link'"
        return 1
    fi

    # Case D: a real directory (or file) that isn't ours. Never remove it —
    # it may be someone's partially-unzipped bundle or unrelated work.
    #
    # This used to print one suggestion -- mv it to .bak and symlink -- framed
    # as "if it is an old install you no longer need". That framing asserted
    # something it had not checked, and the suggestion can be expensive: the
    # per-demo models live INSIDE this tree ({dir}/models, see install_plans.py
    # -- maritime 11GB, pharmacy 28GB, pii 20GB, medical 4GB), so moving it
    # aside orphans up to ~63GB of downloads that Phase 6 then re-fetches.
    #
    # A real directory at ~/demos is also a SUPPORTED layout, not a mistake:
    # make_bundle.sh ships a zip that unpacks in $HOME to exactly this, and
    # Case A above treats it as canonical. So inspect it and say what it is.
    if [[ -d "$link" ]]; then
        _dl_err "~/demos is an existing directory, not a link to this checkout."
        _dl_err "    ~/demos  = $link"
        _dl_err "    this one = $repo_demos"

        # Is it a checkout of this same project?
        _dl_other_kind="unknown"
        if [[ -e "$link/.git" ]] || git -C "$link" rev-parse --git-dir >/dev/null 2>&1; then
            _dl_other_kind="git"
        elif [[ -f "$link/install.sh" && -f "$link/install_plans.py" ]]; then
            _dl_other_kind="bundle"
        fi

        # How much model data would a naive move orphan?
        _dl_model_gb=0
        if command -v du >/dev/null 2>&1; then
            # Summed in MiB then converted: `du -sBG` rounds EACH path up to a
            # whole GB, which over-reported 3GiB of data as 5GB. A warning about
            # losing data has to be a number you can trust.
            _dl_model_gb="$(du -sm 2>/dev/null $link/*/models 2>/dev/null \
                            | awk '{t+=$1} END {printf "%d", (t/1024)+0.5}')"
            [[ -z "$_dl_model_gb" ]] && _dl_model_gb=0
        fi

        case "$_dl_other_kind" in
          git)
            _dl_err ""
            _dl_err "That directory is itself a git checkout of this project"
            _dl_err "  branch: $(git -C "$link" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
            _dl_err "  commit: $(git -C "$link" log --oneline -1 2>/dev/null || echo '?')"
            _dl_err ""
            _dl_err "It is already the canonical layout (see Case A), so the simplest"
            _dl_err "fix is to use it instead of this copy — no moving, nothing re-downloaded:"
            _dl_err "  cd '$link' && git pull && ./install.sh"
            ;;
          bundle)
            _dl_err ""
            _dl_err "That looks like an unpacked delivery bundle (install.sh present, no git)."
            _dl_err "That IS a supported layout — install.sh can be run from it directly:"
            _dl_err "  cd '$link' && ./install.sh"
            ;;
          *)
            _dl_err ""
            _dl_err "Could not tell what that directory is (no .git, no install.sh)."
            _dl_err "Look before moving it."
            ;;
        esac

        if [[ "${_dl_model_gb:-0}" -gt 0 ]]; then
            _dl_err ""
            _dl_warn "It holds roughly ${_dl_model_gb}GB of downloaded models under */models."
            _dl_warn "Moving it aside orphans those; the installer would re-download them."
            _dl_warn ""
            _dl_warn "To keep them, adopt them into this checkout first (dry run by default;"
            _dl_warn "it refuses to overwrite or nest, and skips demos this tree does not have):"
            _dl_warn "  bash '$repo_demos/scripts/adopt_models.sh' '$link'"
            _dl_warn "  bash '$repo_demos/scripts/adopt_models.sh' '$link' --apply"
            _dl_warn "then:"
            _dl_warn "  mv '$link' '${link}.bak' && ln -s '$repo_demos' '$link'"
        else
            _dl_err ""
            _dl_err "If you do want to switch to this checkout (no models found to preserve):"
            _dl_err "  mv '$link' '${link}.bak' && ln -s '$repo_demos' '$link'"
        fi
    else
        _dl_err "~/demos exists and is a file, not a directory or link. Remove or"
        _dl_err "rename it, then run again."
    fi
    return 1
}
