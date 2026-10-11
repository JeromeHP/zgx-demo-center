#!/usr/bin/env python3
"""HP ZGX Demo Center — installer Python orchestrator.

install.sh runs first (Python 3.12, miniforge, zgx env), activates the zgx
conda env, then execs `$CONDA_PREFIX/bin/python install.py <forwarded flags>`.
This module owns the orchestration phases that run inside the zgx env.

Phases (executed in this order):
  1. Preflight     — stdlib-only environment checks         (IMPLEMENTED)
  2. zgx deps      — pip install the server stack            (IMPLEMENTED)
  3. Ollama+models — install Ollama + pull global models     (IMPLEMENTED)
  4. native assets — stage onnx + videos (LOCAL-FIRST)       (IMPLEMENTED)
  5. start servers — run_demos.sh (launcher :8085 + API :8086) (IMPLEMENTED)
  6. web installs  — trigger ALL demos via /install/{id}     (IMPLEMENTED)
  4b TRT engine    — build yolov8s.trt with the zgx env's    (IMPLEMENTED)
                     TensorRT — runs AFTER Phase 6 because the
                     vision per-demo install is what pip-installs
                     that TensorRT wheel into the env.
  7. desktop icon  — ensure wrapper +x + generate .desktop   (IMPLEMENTED)

FAILURE HANDLING: a failing per-demo install in Phase 6, or an engine-build
failure in 4b, no longer aborts the run. The desktop icon + launch wrapper
(Phase 7) is ALWAYS created so the box is usable, and the orchestrator prints a
consolidated failure summary and exits non-zero at the VERY END.

ENGINE BUILD: the TRT engine is built in-env (zgx TensorRT), NOT in the
tensorrt:25.05-py3 container, so the engine is serialized by the SAME TensorRT
the runtime deserializes with (build==runtime). See phase_build_engine.

Chromium is NOT installed by this orchestrator — it is a prerequisite, verified
up front by the Phase 1 preflight browser check (so the unattended run never
stalls on an interactive sudo at the very end). See README.md → Prerequisites.

Phase 1 runs BEFORE any deps are installed, so it must use the Python
standard library only (no requests/aiohttp/etc.).
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zgx_hardware  # noqa: E402  (stdlib-only, like this file)

# ──────────────────────────────────────────────────────────────────────────────
# Logging helpers — mirror install.sh's visual style so the combined output of
# the bash bootstrap + this orchestrator reads as one continuous installer.
# ──────────────────────────────────────────────────────────────────────────────
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def log_section(msg: str) -> None:
    line = "───────────────────────────────────────────────────"
    print(f"\n{BOLD}{BLUE}{line}{RESET}")
    print(f"{BOLD}{BLUE}{msg}{RESET}")
    print(f"{BOLD}{BLUE}{line}{RESET}")


def log_check(msg: str) -> None:
    print(f"{GREEN}✓ {msg}{RESET}")


def log_install(msg: str) -> None:
    print(f"{YELLOW}⏳ {msg}...{RESET}")


def log_done(msg: str) -> None:
    print(f"{GREEN}✓ {msg}{RESET}")


def log_skip(msg: str) -> None:
    print(f"{DIM}✓ {msg} (already present){RESET}")


def log_error(msg: str) -> None:
    print(f"{RED}✗ {msg}{RESET}", file=sys.stderr)


def log_info(msg: str) -> None:
    print(f"{BLUE}ℹ {msg}{RESET}")


def log_warn(msg: str) -> None:
    print(f"{YELLOW}⚠ {msg}{RESET}")


# ──────────────────────────────────────────────────────────────────────────────
# Flag parsing — mirror install.sh's flag set. install.sh forwards every flag
# verbatim via ORIG_ARGS, so we see the same options here.
# ──────────────────────────────────────────────────────────────────────────────
class Flags:
    def __init__(self) -> None:
        self.dry_run = False
        self.force = False
        self.skip_preflight = False
        self.skip_demos: list[str] = []


def parse_flags(argv: list[str]) -> Flags:
    f = Flags()
    for arg in argv:
        if arg == "--dry-run":
            f.dry_run = True
        elif arg == "--force":
            f.force = True
        elif arg == "--skip-preflight":
            f.skip_preflight = True
        elif arg.startswith("--skip-"):
            # Repeatable per-demo skip. Accepted now, acted on in a later slice.
            f.skip_demos.append(arg[len("--skip-"):])
        elif arg in ("--help", "-h"):
            # install.sh owns the user-facing usage; ignore here.
            pass
        else:
            log_warn(f"Ignoring unrecognized flag: {arg}")
    return f


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 — PREFLIGHT (stdlib only; runs before deps exist)
# ──────────────────────────────────────────────────────────────────────────────
INTERNET_HOSTS = [
    "huggingface.co",
    "registry.ollama.ai",
    "registry-1.docker.io",
]
# Vendor-specific registries: only probed when that vendor's profile is active.
# nvcr.io hosts the vLLM/TensorRT containers; an AMD box never pulls from it, so
# requiring it there would fail the install over an unreachable host it does not need.
PROFILE_INTERNET_HOSTS = {
    "nvidia-cuda": ["nvcr.io"],
    "amd-rocm": [],
    "cpu-only": [],
}
DISK_NEED_GB = 150

# Detected once at import: every phase, and /sysinfo, must agree about what this
# machine is. Detection never raises -- worst case the profile is "cpu-only".
HARDWARE = zgx_hardware.detect()

# Shown on every preflight failure: the host prerequisites (Docker, Compose,
# NVIDIA Container Toolkit, Chromium, wmctrl/xdotool, docker group) are installed
# by setup-prereqs.sh, which requires a reboot before install.sh can run.
SETUP_HINT = (
    "Fix: run  sudo bash setup-prereqs.sh , then REBOOT, then re-run install.sh."
)


def _tcp_reachable(host: str, port: int = 443, timeout: float = 4.0) -> bool:
    """Return True if a TCP connection to host:port succeeds within timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _cmd_ok(args: list[str], timeout: float = 30.0) -> bool:
    """Return True iff the command runs and exits 0. Missing binary → False."""
    try:
        r = subprocess.run(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        return r.returncode == 0
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return False


# Browser binaries we accept as "present" (the wrapper resolves the same set at
# runtime). PATH-resolvable names go through shutil.which; the snap lives at a
# fixed path that may not be on PATH in every shell, so check it directly.
BROWSER_WHICH_NAMES = ["chromium", "chromium-browser", "google-chrome", "google-chrome-stable"]
BROWSER_FIXED_PATHS = ["/snap/bin/chromium"]


def _browser_present() -> bool:
    """True if a Chromium/Chrome browser binary is available (stdlib only)."""
    if any(shutil.which(name) for name in BROWSER_WHICH_NAMES):
        return True
    return any(os.path.exists(p) for p in BROWSER_FIXED_PATHS)


def phase_preflight(flags: Flags) -> None:
    """Run all environment checks, collect results, report together.

    Exits non-zero if ANY check fails. --skip-preflight bypasses entirely.
    --dry-run still runs (the checks are read-only).
    """
    log_section("Phase 1: Preflight")

    if flags.skip_preflight:
        log_warn("Preflight skipped (--skip-preflight)")
        return

    failures: list[str] = []

    # 1. Internet — TCP/443 to each required registry/host.
    for host in INTERNET_HOSTS + PROFILE_INTERNET_HOSTS.get(HARDWARE["profile"], []):
        if _tcp_reachable(host):
            log_check(f"Internet: {host} reachable")
        else:
            msg = (
                f"No internet connection to {host}. Installation needs to "
                "download AI models and container images. Connect to a network "
                "with internet (or the HP network) and run again."
            )
            log_error(msg)
            failures.append(msg)

    # 2. Disk free — >= DISK_NEED_GB on the volume containing $HOME.
    free_gb = shutil.disk_usage(os.path.expanduser("~")).free / (1024 ** 3)
    if free_gb >= DISK_NEED_GB:
        log_check(f"Disk free: {free_gb:.0f} GB (need ~{DISK_NEED_GB} GB)")
    else:
        msg = (
            f"Not enough disk space: need ~{DISK_NEED_GB} GB free, you have "
            f"{free_gb:.0f} GB. Free up space and run again."
        )
        log_error(msg)
        failures.append(msg)

    # 3. Docker Engine.
    if _cmd_ok(["docker", "info"]):
        log_check("Docker Engine running")
    else:
        msg = (
            "Docker isn't installed or isn't running. " + SETUP_HINT
        )
        log_error(msg)
        failures.append(msg)

    # 4. Docker Compose.
    if _cmd_ok(["docker", "compose", "version"]):
        log_check("Docker Compose available")
    else:
        msg = (
            "Docker Compose not found. " + SETUP_HINT
        )
        log_error(msg)
        failures.append(msg)

    # 5 + 6. Accelerator and its container plumbing.
    #
    # These two checks used to be unconditional `nvidia-ctk` and `nvidia-smi`
    # probes, which meant install.py refused to run at all on an AMD box --
    # before touching a single demo, several of which need no GPU whatsoever.
    # Now the profile is detected first and each vendor is held to its OWN
    # requirement, because the two stacks are not analogous: NVIDIA needs a
    # container toolkit, AMD needs device nodes plus group membership.
    #
    # The NVIDIA path below is behaviourally identical to what it replaced.
    hw = HARDWARE
    log_check(
        f"Hardware: {hw['device_label']} — {hw['accelerator']}, {hw['arch']}, "
        f"{hw['memory_gb']} GB ({hw['memory_source']})"
    )
    for warning in hw["warnings"]:
        log_warn(f"Hardware: {warning}")

    if hw["profile"] == "nvidia-cuda":
        if _cmd_ok(["nvidia-ctk", "--version"]):
            log_check(f"NVIDIA Container Toolkit available ({hw['container_runtime']})")
        else:
            msg = (
                "NVIDIA Container Toolkit not found — the Docker demos need it to "
                "use the GPU. " + SETUP_HINT
            )
            log_error(msg)
            failures.append(msg)
        log_check(f"NVIDIA GPU detected ({hw['accelerator']}, driver {hw['driver']})")

    elif hw["profile"] == "amd-rocm":
        log_check(f"AMD GPU detected ({hw['accelerator']}, {hw['driver']})")
        # There is no container toolkit on ROCm. What a container actually needs
        # is /dev/kfd + /dev/dri passed through, and a user who can open them --
        # missing group membership fails later, inside the container, as a bare
        # permission error that looks nothing like its cause.
        blockers = [w for w in hw["warnings"] if "group" in w or "render node" in w]
        if blockers:
            msg = (
                "ROCm GPU access is not usable yet: " + "; ".join(blockers) + ". "
                "Fix with:  sudo usermod -aG render,video $USER  then log out and back in."
            )
            log_error(msg)
            failures.append(msg)
        else:
            log_check("ROCm device access OK (/dev/kfd, /dev/dri, render+video groups)")

    else:
        # Not a failure. The Ollama-backed demos (ballistics, pii, pharmacy,
        # grant) and the CPU-only ones (construction) still install and run;
        # only the accelerator-bound demos are unavailable. Saying so beats
        # refusing to install anything, which is what used to happen.
        log_warn(
            "No supported accelerator detected — continuing in cpu-only mode. "
            "Demos needing CUDA or ROCm will be skipped; the rest work normally."
        )

    # 7. Chromium/Chrome browser — Demo Center runs fullscreen in Chromium.
    # Checked here (not installed in Phase 7) so the requirement surfaces up
    # front, before the long unattended download phases — not as a sudo prompt
    # buried at the very end of the run.
    if _browser_present():
        log_check("Chromium/Chrome browser found")
    else:
        msg = (
            "No web browser found. Demo Center runs in Chromium. " + SETUP_HINT
        )
        log_error(msg)
        failures.append(msg)

    # 8. wmctrl — start_demo_center.sh forces the kiosk window fullscreen with
    # `wmctrl -b add,fullscreen`. Without it the launcher opens but never goes
    # fullscreen. Installed by setup-prereqs.sh alongside xdotool.
    # Check that the binary is *present* (shutil.which, like the browser check),
    # NOT `wmctrl -m` — that queries the running window manager over $DISPLAY and
    # fails "Cannot open display" under headless/SSH installs even though wmctrl
    # is installed. Whether a display is up is a launch-time concern, not install.
    if shutil.which("wmctrl"):
        log_check("wmctrl available")
    else:
        msg = (
            "wmctrl not found — Demo Center needs it to open fullscreen. " + SETUP_HINT
        )
        log_error(msg)
        failures.append(msg)

    # 9. Ollama — the demos' local LLM runtime. Installed by setup-prereqs.sh
    # (NOT by this orchestrator anymore), so it's a host prerequisite like Docker.
    # Phase 3 only pulls models through it, so the binary must already be here.
    if _cmd_ok(["ollama", "--version"]):
        log_check("Ollama available")
    else:
        msg = (
            "Ollama not found — install.sh pulls the AI models through it. " + SETUP_HINT
        )
        log_error(msg)
        failures.append(msg)

    # 10. ~/demos resolves to THIS checkout. install.sh normally guarantees this
    # before handing off, but install.py can be run directly, and the link can be
    # clobbered between runs. Demos hardcode $HOME/demos/... by convention, so a
    # wrong link means we install into one tree and launch from another — which
    # historically surfaced much later as a confusing "launcher not found" in
    # phase_start_servers. Name it here instead.
    demos_link = Path.home() / "demos"
    try:
        link_ok = demos_link.resolve() == REPO_ROOT
    except OSError:
        link_ok = False
    if link_ok:
        log_check(f"~/demos → {REPO_ROOT}")
    else:
        actual = "missing"
        if demos_link.is_symlink() or demos_link.exists():
            try:
                actual = str(demos_link.resolve())
            except OSError:
                actual = "broken link"
        msg = (
            f"~/demos does not point at this checkout (it is: {actual}). The "
            f"demos hardcode $HOME/demos/... paths, so they would install here "
            f"({REPO_ROOT}) but launch from there. Fix with:\n"
            f"      ln -s '{REPO_ROOT}' ~/demos\n"
            f"    or just run install.sh, which sets it up for you."
        )
        log_error(msg)
        failures.append(msg)

    if failures:
        print()
        log_error(f"Preflight failed — {len(failures)} issue(s) must be fixed:")
        for m in failures:
            log_error(f"  • {m}")
        print()
        log_error(SETUP_HINT)
        sys.exit(1)

    log_done("Preflight passed")


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 — zgx PYTHON DEPS
# ──────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = REPO_ROOT / "requirements.txt"

# Import names to verify after install (pip names may differ, e.g. uvicorn[standard]).
VERIFY_IMPORTS = ["fastapi", "uvicorn", "httpx", "huggingface_hub"]


def phase_deps(flags: Flags) -> None:
    """Install the zgx env's Python deps from requirements.txt (idempotent)."""
    log_section("Phase 2: zgx Python dependencies")

    if not REQUIREMENTS_FILE.exists():
        log_error(f"requirements.txt not found at {REQUIREMENTS_FILE}")
        sys.exit(1)

    deps = [
        line.strip()
        for line in REQUIREMENTS_FILE.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if flags.dry_run:
        log_info(f"would install into: {sys.executable}")
        log_info(f"would install: {', '.join(deps)}")
        return

    log_install(f"Installing deps into {sys.executable}")
    rc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
    ).returncode
    if rc != 0:
        log_error("pip install failed for the zgx dependencies")
        sys.exit(1)
    log_done("Dependencies installed (pip skipped any already satisfied)")

    # Verify the critical imports resolve in this env.
    verify = subprocess.run(
        [sys.executable, "-c", "import " + ", ".join(VERIFY_IMPORTS)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if verify.returncode != 0:
        log_error("Dependency verification failed — imports did not resolve:")
        log_error(verify.stderr.decode(errors="replace").strip())
        sys.exit(1)
    log_check(f"Verified imports: {', '.join(VERIFY_IMPORTS)}")


# ──────────────────────────────────────────────────────────────────────────────
# Shared exec helpers (Phases 3 & 4)
# ──────────────────────────────────────────────────────────────────────────────
def _stream_command(args: list[str]) -> int:
    """Run a command, streaming stdout/stderr LIVE to this terminal. Returns rc.

    Inherits the parent's stdout/stderr (no capture) so a slow `ollama pull`
    or `trtexec` build shows progress instead of a frozen screen.
    """
    try:
        return subprocess.run(args).returncode
    except (FileNotFoundError, OSError) as e:
        log_error(f"Could not start {args[0]}: {e}")
        return 1


def _stream_shell(cmd: str) -> int:
    """Run a shell pipeline (e.g. curl … | sh), streaming output live."""
    try:
        return subprocess.run(cmd, shell=True).returncode
    except OSError as e:
        log_error(f"Shell command failed to start: {e}")
        return 1


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Ollama install + global model pulls
# ──────────────────────────────────────────────────────────────────────────────
# Models shared across demos (pipeline + vision use llama3.1:70b; pharmacy RAG
# uses nomic-embed-text). NOT pulled by any demo's own web-install step, so the
# installer owns them. Per-demo Ollama models (medical-mixtral, pii-detector,
# tinyllama-pii) are registered in a separate later slice — not here.
#
# Ollama itself is NO LONGER installed here — it is a host prerequisite provisioned
# by setup-prereqs.sh (so the one interactive sudo touchpoint stays in that script
# and this orchestrator runs hands-off). Phase 1 preflight verifies the binary;
# Phase 3 only ensures the daemon and pulls these models.
GLOBAL_OLLAMA_MODELS = ["llama3.1:70b", "nomic-embed-text"]


def _ollama_reachable() -> bool:
    """True if the Ollama daemon answers `ollama list`."""
    return _cmd_ok(["ollama", "list"])


def _ollama_has_model(tag: str) -> bool:
    """True if `ollama list` shows the given tag (matches bare tag → :latest)."""
    try:
        r = subprocess.run(
            ["ollama", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return False
    if r.returncode != 0:
        return False
    names = [
        line.split()[0]
        for line in r.stdout.decode(errors="replace").splitlines()[1:]
        if line.split()
    ]
    return any(n == tag or n.startswith(tag + ":") for n in names)


def _ensure_ollama_daemon() -> None:
    """Make the Ollama daemon reachable, starting it if needed (no root path)."""
    if _ollama_reachable():
        log_check("Ollama daemon reachable")
        return
    log_install("Starting Ollama daemon")
    # Prefer the systemd service the install script sets up; ignore failure
    # (may lack privileges) and fall back to a backgrounded `ollama serve`.
    subprocess.run(
        ["systemctl", "start", "ollama"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not _ollama_reachable():
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except (FileNotFoundError, OSError) as e:
            log_error(f"Could not start `ollama serve`: {e}")
            sys.exit(1)
    for _ in range(30):
        if _ollama_reachable():
            log_done("Ollama daemon reachable")
            return
        time.sleep(1)
    log_error("Ollama daemon did not become reachable within 30s.")
    sys.exit(1)


def phase_ollama_and_models(flags: Flags) -> None:
    log_section("Phase 3: Ollama models")

    # 3a. Ollama binary — a host prerequisite (installed by setup-prereqs.sh).
    # We do NOT install it here. If it's missing, fail with the standard hint.
    if not _cmd_ok(["ollama", "--version"]):
        log_error("Ollama isn't installed (it's a host prerequisite). " + SETUP_HINT)
        sys.exit(1)
    log_skip("Ollama")

    # 3b. Daemon reachable (real runs only — dry-run makes no changes).
    if not flags.dry_run:
        _ensure_ollama_daemon()

    # 3c. Global model pulls (skip-if-present; stream the slow 42GB pull).
    for model in GLOBAL_OLLAMA_MODELS:
        if not flags.dry_run and _ollama_has_model(model):
            log_skip(f"Ollama model {model}")
        elif flags.dry_run:
            log_info(f"would pull Ollama model {model} (if not already present)")
        else:
            log_install(f"Pulling Ollama model {model} (streaming progress)")
            if _stream_command(["ollama", "pull", model]) != 0:
                log_error(f"Failed to pull {model}.")
                sys.exit(1)
            log_done(f"Pulled {model}")


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 4 — native asset staging (onnx + videos). The TRT engine is built later
# (phase_build_engine, after Phase 6) using the zgx env's TensorRT.
# ──────────────────────────────────────────────────────────────────────────────
# Asset delivery is LOCAL-FIRST. The two native-asset tarballs ship bundled with
# the repo (gitignored) under assets/, because every HP-internal host we tested
# (SharePoint, GitHub Enterprise) 403/401s an anonymous scripted fetch. The rep
# downloads ONE authed-browser bundle (repo + assets), unzips it to $HOME, and
# runs install.sh — so by the time Phase 4 runs, the tarballs are sitting in
# assets/ ready to extract with no network at all. See make_bundle.sh.
#
# Each tarball unpacks into $HOME:
#   assets/yolov8s-onnx.tar.gz → ~/yolov8s.onnx
#   assets/zgx-videos.tar.gz   → ~/zgx_videos/ (traffic files + warehouse/
#                                outdoor/ retail/ subdirs)
#
# MODEL_ASSET_URL / VIDEO_ASSET_URL are an OPTIONAL fallback for a future hosted
# setup (an HP-controlled host that serves raw bytes to an anonymous GET). They
# stay as placeholders until such hosting exists; the local bundle is the primary
# and currently only working path.
LOCAL_MODEL_TARBALL = REPO_ROOT / "assets" / "yolov8s-onnx.tar.gz"
LOCAL_VIDEO_TARBALL = REPO_ROOT / "assets" / "zgx-videos.tar.gz"
MODEL_ASSET_URL = "PLACEHOLDER://asset-host/yolov8s-onnx.tar.gz"
VIDEO_ASSET_URL = "PLACEHOLDER://asset-host/zgx-videos.tar.gz"

# In-env TRT engine builder (Phase 4b). Run via THIS process's interpreter
# (sys.executable == the zgx env python after install.sh's handoff), so the
# engine is serialized by the SAME TensorRT that ai-vision/vision_engine.py
# deserializes with at runtime. Building in the tensorrt:25.05-py3 container
# (TRT 10.10) instead produced an engine the env's wheel (tensorrt-cu13
# 10.15.1.29) refuses to deserialize — the v239/v240 clean-box failure.
ENGINE_BUILDER = REPO_ROOT / "ai-vision" / "build_engine.py"


def _is_placeholder(url: str) -> bool:
    return url.startswith("PLACEHOLDER")


def _extract_local(tarball: Path, dest_dir: Path, label: str) -> None:
    """Extract a repo-bundled tar archive into `dest_dir` — no network.

    Stdlib only (tarfile). The bundle ships with the repo via make_bundle.sh, so
    the archive is trusted; extraction uses the 'data' filter to reject unsafe
    member paths (same hardening as the URL fallback below).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    log_install(f"Extracting {label} from local bundle {tarball}")
    try:
        with tarfile.open(tarball) as tf:
            tf.extractall(dest_dir, filter="data")
    except (OSError, tarfile.TarError) as e:
        log_error(f"Failed to extract {label} from {tarball}: {e}")
        sys.exit(1)
    log_done(f"{label} staged into {dest_dir} (local bundle)")


def _fetch_and_unpack(url: str, dest_dir: Path, label: str) -> None:
    """Download a tar archive from `url` and extract it into `dest_dir`.

    Stdlib only (urllib + tarfile) — no new pip deps. The archive is assumed to
    come from a trusted HP-controlled host; extraction uses the 'data' filter to
    reject unsafe member paths.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.gettempdir()) / f"zgx_asset_{label}.tar"
    log_install(f"Fetching {label} from {url}")
    try:
        urllib.request.urlretrieve(url, tmp)
        with tarfile.open(tmp) as tf:
            tf.extractall(dest_dir, filter="data")
    except (OSError, urllib.error.URLError, tarfile.TarError) as e:
        log_error(f"Failed to fetch/unpack {label}: {e}")
        if tmp.exists():
            tmp.unlink()
        sys.exit(1)
    if tmp.exists():
        tmp.unlink()
    log_done(f"{label} staged into {dest_dir}")


def _stage_native_asset(
    *,
    label: str,
    target_desc: str,
    already_present: bool,
    local_tarball: Path,
    url: str,
    dest_dir: Path,
    dry_run: bool,
) -> None:
    """Stage one native asset, LOCAL-FIRST then URL-fallback.

    Decision order (matches both assets):
      1. target already present  → skip.
      2. local bundle tarball at assets/<name>.tar.gz exists → extract locally.
      3. *_ASSET_URL configured (not a placeholder) → download + extract.
      4. otherwise → fail: missing, no local bundle, no configured URL.
    --dry-run reports which source it WOULD use without touching anything.
    """
    if already_present:
        log_skip(f"{label} ({target_desc})")
        return
    if local_tarball.exists():
        if dry_run:
            log_info(f"would stage {label} from local bundle {local_tarball} "
                     f"→ {dest_dir}")
            return
        _extract_local(local_tarball, dest_dir, label)
        return
    if not _is_placeholder(url):
        if dry_run:
            log_info(f"would fetch {label} from {url} → {dest_dir}")
            return
        _fetch_and_unpack(url, dest_dir, label)
        return
    # 4. missing — no local bundle, no configured URL.
    if dry_run:
        log_info(f"{label} missing — no local bundle at {local_tarball} and no "
                 "configured asset URL; would FAIL here.")
        return
    log_error(f"{label} is missing: no local bundle tarball at {local_tarball} "
              "and no asset URL configured. Re-download the delivery bundle "
              "(repo + assets/) and unzip it before running the installer.")
    sys.exit(1)


def phase_native_assets(flags: Flags) -> None:
    """Stage the native assets only (onnx + videos). Engine build is Phase 4b."""
    log_section("Phase 4: Native asset staging")
    home = Path.home()
    onnx = home / "yolov8s.onnx"
    videos = home / "zgx_videos"

    # 4a. yolov8s.onnx (local-first, URL fallback).
    _stage_native_asset(
        label="yolov8s.onnx",
        target_desc=str(onnx),
        already_present=onnx.exists(),
        local_tarball=LOCAL_MODEL_TARBALL,
        url=MODEL_ASSET_URL,
        dest_dir=home,
        dry_run=flags.dry_run,
    )

    # 4b. zgx_videos/ (skip if present + non-empty; local-first, URL fallback).
    videos_ok = videos.is_dir() and any(videos.iterdir())
    _stage_native_asset(
        label="zgx_videos",
        target_desc=str(videos),
        already_present=videos_ok,
        local_tarball=LOCAL_VIDEO_TARBALL,
        url=VIDEO_ASSET_URL,
        dest_dir=home,
        dry_run=flags.dry_run,
    )


def phase_build_engine(flags: Flags) -> bool:
    """Build yolov8s.trt with the zgx env's TensorRT. Returns True on success/skip.

    Runs AFTER Phase 6 so the vision per-demo install has already pip-installed
    tensorrt-cu13 into this env. Building here (sys.executable == zgx python)
    guarantees the engine is serialized by the SAME TensorRT the runtime
    deserializes with — eliminating the build/runtime serialization mismatch
    (the v239/v240 clean-box failure) that the old container build risked.

    Does NOT sys.exit on failure — returns False so main() can still create the
    desktop icon (Phase 7) and report the failure at the very end.
    """
    log_section("Phase 4b: TRT engine build (in-env TensorRT)")
    home = Path.home()
    onnx = home / "yolov8s.onnx"
    trt = home / "yolov8s.trt"      # == ENGINE_PATHS[0] in ai-vision/vision_backends.py

    # TensorRT is NVIDIA-only. On any other profile ai-vision runs the same
    # weights through ONNX Runtime instead (vision_backends.py), from the very
    # ~/yolov8s.onnx that Phase 4 already staged -- so there is nothing to build
    # and nothing missing. Skipping is the correct outcome, not a degraded one.
    if HARDWARE["profile"] != "nvidia-cuda":
        if onnx.is_file():
            log_skip(f"TRT engine build — {HARDWARE['profile']} uses ONNX Runtime "
                     f"against {onnx} (same weights, no engine needed)")
            return True
        log_error(f"{onnx} is missing and this profile ({HARDWARE['profile']}) "
                  "cannot build a TensorRT engine. Re-run Phase 4 asset staging.")
        return False

    if trt.exists():
        log_skip(f"yolov8s.trt ({trt})")
        return True
    if flags.dry_run:
        log_info(f"would build {trt} from {onnx} via {sys.executable} "
                 f"{ENGINE_BUILDER} (in-env TensorRT, FP16, 1088x1088), then "
                 "deserialize-assert it with the same TensorRT")
        return True
    if not onnx.exists():
        log_error(f"Cannot build TRT engine — {onnx} not present.")
        return False
    if not ENGINE_BUILDER.exists():
        log_error(f"Cannot build TRT engine — builder missing at {ENGINE_BUILDER}.")
        return False

    log_install("Building TRT engine with the zgx env's TensorRT (FP16)")
    log_info("This takes ~15-20 minutes on this hardware (inherent to the FP16 "
             "build, NOT a hang). Builder progress streams below; please wait.")
    # sys.executable is the zgx env python (install.sh handed off via it), so the
    # builder imports the env's tensorrt-cu13 wheel — the runtime's exact TRT.
    rc = _stream_command([sys.executable, str(ENGINE_BUILDER)])
    if rc != 0 or not trt.exists():
        log_error("TRT engine build failed (see builder output above).")
        return False
    log_done(f"Built TRT engine: {trt}")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 5 — start the Demo Center servers (launcher :8085 + install API :8086)
# ──────────────────────────────────────────────────────────────────────────────
LAUNCHER_FILE = "zgx-demo-launcher.html"
LAUNCHER_PORT = 8085
INSTALL_API_PORT = 8086
LAUNCHER_URL = f"http://localhost:{LAUNCHER_PORT}/{LAUNCHER_FILE}"
INSTALL_API_HEALTH_URL = f"http://localhost:{INSTALL_API_PORT}/health"
RUN_DEMOS_SCRIPT = REPO_ROOT / "run_demos.sh"  # ~/demos/run_demos.sh
SERVER_READY_TIMEOUT_S = 60


def _http_ok(url: str, timeout: float = 3.0) -> bool:
    """True if a GET to url returns HTTP 200 (stdlib urllib)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _servers_up() -> bool:
    """Both the launcher (8085) and the install API (8086) answer 200."""
    return _http_ok(LAUNCHER_URL) and _http_ok(INSTALL_API_HEALTH_URL)


def phase_start_servers(flags: Flags) -> None:
    log_section("Phase 5: Start servers")

    if _servers_up():
        log_skip("Demo Center servers already running (8085 + 8086)")
        return

    if flags.dry_run:
        log_info(f"would start servers via {RUN_DEMOS_SCRIPT} and verify health "
                 f"on :{LAUNCHER_PORT} + :{INSTALL_API_PORT}")
        return

    script = RUN_DEMOS_SCRIPT
    if not script.exists():
        log_error(f"run_demos.sh not found at {script}")
        sys.exit(1)

    log_install("Starting Demo Center servers (run_demos.sh, detached)")
    # run_demos.sh `wait`s in the foreground and traps INT/TERM to kill its
    # children — so it MUST be detached (own session, stdin from /dev/null) or
    # the servers would die when install.py exits.
    log = REPO_ROOT / "logs" / "demo_center_run.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log, "ab") as lf, open(os.devnull, "rb") as devnull:
            subprocess.Popen(
                ["bash", str(script)],
                stdin=devnull,
                stdout=lf,
                stderr=lf,
                start_new_session=True,
                cwd=str(script.parent),
            )
    except OSError as e:
        log_error(f"Could not launch run_demos.sh: {e}")
        sys.exit(1)

    # Poll until BOTH endpoints are ready.
    for _ in range(SERVER_READY_TIMEOUT_S):
        if _servers_up():
            log_done(f"Servers ready — launcher :{LAUNCHER_PORT}, "
                     f"install API :{INSTALL_API_PORT}")
            return
        time.sleep(1)
    log_error(f"Servers did not become ready within {SERVER_READY_TIMEOUT_S}s "
              f"(see {log}).")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 6 — trigger the 11 web-UI demo installs (server-side, no browser)
# ──────────────────────────────────────────────────────────────────────────────
# All installable demos, in launcher order. vision + pipeline + grant are native
# (no Docker) but DO have install steps — they pip-install their runtime deps
# into the zgx env (see install_plans.py). vision MUST run before phase_build_engine
# (it installs the env's tensorrt-cu13, which the in-env engine build then uses).
# spectrum must follow vision for the same reason — its install step trains a
# classifier and builds a TRT engine using that same in-env tensorrt.
# grant has no such coupling (no TensorRT, no engine build — it talks to host
# Ollama over HTTP), so it goes last and its position is not load-bearing.
# Every step is install-if-missing, so on a provisioned box these stream skips
# and finish fast.
WEB_INSTALL_IDS = ["maritime", "pharmacy", "pii", "ballistics", "medical", "construction", "translation", "vision", "spectrum", "pipeline", "grant", "vehicle"]

# ORDER IS LOAD-BEARING and must survive any profile filtering: "vision"
# pip-installs tensorrt-cu13 into the shared zgx env, which "spectrum" and
# Phase 4b both depend on. Filtering preserves order because it is a list
# comprehension over this list, never a set.


def _installable_demos(flags) -> tuple[list, list]:
    """Split the demo list into what this box can install and what it cannot.

    A demo whose accelerator requirement is unmet is SKIPPED with a reason
    rather than attempted and failed: on an AMD box the maritime and medical
    demos pull nvcr.io NVIDIA containers that will not run, and reporting that
    as an install failure buries the ones that did work.
    """
    import install_plans
    runnable, unsupported = [], []
    profile = HARDWARE["profile"]
    for demo_id in WEB_INSTALL_IDS:
        if demo_id in flags.skip_demos:
            continue
        plan = install_plans.INSTALL_PLANS.get(demo_id, {})
        needs = (plan.get("requires") or {}).get("profiles")
        if needs and profile not in needs:
            unsupported.append((demo_id, plan.get("name", demo_id), needs))
        else:
            runnable.append(demo_id)
    return runnable, unsupported
INSTALL_STREAM_READ_TIMEOUT_S = 600  # per-read socket timeout — guards a stalled stream


def _run_web_install(demo_id: str) -> str:
    """Drive GET /install/{demo_id} (SSE), parse events, print concise progress.

    Returns one of: 'ok' (completed), 'failed' (error event, stream problem, or
    any required step exiting non-zero).
    """
    url = f"http://localhost:{INSTALL_API_PORT}/install/{demo_id}"
    saw_complete = False
    step_failed = False
    try:
        with urllib.request.urlopen(url, timeout=INSTALL_STREAM_READ_TIMEOUT_S) as resp:
            for raw in resp:  # iterate the SSE stream line by line
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    ev = json.loads(line[len("data:"):].strip())
                except json.JSONDecodeError:
                    continue
                etype = ev.get("type")
                if etype == "start":
                    log_info(f"  [{demo_id}] {ev.get('name', demo_id)} "
                             f"({ev.get('steps', '?')} steps)")
                elif etype == "step_start":
                    print(f"    → step {ev.get('step', 0) + 1}: {ev.get('label', '')}", flush=True)
                elif etype == "step_skip":
                    print(f"    ↷ {ev.get('label', '')}: {ev.get('msg', 'skipped')}", flush=True)
                elif etype == "step_done":
                    # Honour the exit code. This printed "✓ {label}" no matter what
                    # the step returned, so Grant Machine's "Verifying imports" step
                    # could die with ModuleNotFoundError and the installer still
                    # reported a tick -- the demo then failed at launch instead, far
                    # from the cause. The verify step exists precisely to catch that
                    # and its result was being thrown away.
                    rc = ev.get("exit_code", 0) or 0
                    if rc == 0:
                        print(f"    ✓ {ev.get('label', '')}", flush=True)
                    elif ev.get("optional"):
                        log_warn(f"  [{demo_id}] optional step failed (rc={rc}): "
                                 f"{ev.get('label', '')}")
                    else:
                        log_error(f"  [{demo_id}] step FAILED (rc={rc}): "
                                  f"{ev.get('label', '')}")
                        step_failed = True
                elif etype == "output":
                    # Fix 3: stream the underlying tool's stdout/stderr through to
                    # the installer's output — docker build steps, model-download
                    # progress, and the SSE server's ~30s silence heartbeats. Before
                    # this, a 28 GB download or a multi-minute docker build produced
                    # ZERO output for 10–20 min and the operator had to `ps` the
                    # installer to tell it was still alive.
                    line_txt = ev.get("line", "")
                    if line_txt:
                        print(f"      {line_txt}", flush=True)
                elif etype == "note":
                    print(f"    ℹ {ev.get('msg', '')}", flush=True)
                elif etype == "complete":
                    saw_complete = True
                    break
                elif etype == "error":
                    log_error(f"  [{demo_id}] {ev.get('msg', 'install error')}")
                    return "failed"
    except (urllib.error.URLError, OSError, ValueError) as e:
        log_error(f"  [{demo_id}] stream failed: {e}")
        return "failed"
    if step_failed:
        # The stream reached "complete", but a required step exited non-zero.
        # Reporting ok here is what let a demo install "successfully" and then
        # fail at launch, with the real cause scrolled far off-screen.
        return "failed"
    return "ok" if saw_complete else "failed"


def phase_web_installs(flags: Flags) -> list[str]:
    """Trigger every demo's install. RETURNS the list of failed demo ids.

    Does NOT sys.exit on failure — a failing demo neither aborts the other demos
    nor the run. main() collects the returned failures, still creates the desktop
    icon (Phase 7), and exits non-zero with a summary at the very end.
    """
    log_section("Phase 6: Per-demo installs")

    targets, unsupported = _installable_demos(flags)
    skipped_by_flag = [d for d in WEB_INSTALL_IDS if d in flags.skip_demos]
    for demo_id, name, needs in unsupported:
        log_skip(f"{name} — needs {'/'.join(needs)}, this box is {HARDWARE['profile']}")
    if skipped_by_flag:
        log_info(f"Skipping (--skip): {', '.join(skipped_by_flag)}")

    if flags.dry_run:
        log_info(f"would trigger install for: {', '.join(targets)}")
        return []

    if not _servers_up():
        # Infrastructure problem, not a per-demo failure — but DON'T exit: let
        # main() still create the icon and report. Every target counts as failed.
        log_error("Install API not reachable on :8086 — cannot trigger installs.")
        return list(targets)

    succeeded, failed = [], []
    # Continue-and-collect: a failing demo does not abort the rest.
    for demo_id in targets:
        log_install(f"Triggering install: {demo_id}")
        if _run_web_install(demo_id) == "ok":
            log_done(f"{demo_id} install complete")
            succeeded.append(demo_id)
        else:
            failed.append(demo_id)

    log_info(f"Per-demo install summary — succeeded: "
             f"{', '.join(succeeded) or 'none'}; "
             f"failed: {', '.join(failed) or 'none'}")
    if failed:
        log_error(f"{len(failed)} demo install(s) failed: {', '.join(failed)}")
    return failed


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 7 — Desktop icon + launch wrapper
# ──────────────────────────────────────────────────────────────────────────────
# Repo-tracked launch wrapper (resolves chromium at runtime, ensures the stack,
# opens the launcher fullscreen) and the branded icon the .desktop entries point
# at. Both are committed in the repo; the .desktop files are generated on-device
# with absolute paths because a .desktop's Exec/Icon must be absolute and
# machine-local. The Chromium browser itself is NOT installed here — it is a
# prerequisite, verified up front by Phase 1 preflight (the browser check), so
# the long unattended install never stalls on an interactive sudo password at
# the very end.
#
# TWO PATHS, DELIBERATELY. Our own file operations (does the wrapper exist,
# chmod +x it) use REPO_ROOT — the PHYSICAL location, which is what we must
# touch on disk. The .desktop Exec/Icon instead point through $HOME/demos, the
# symlink ensure_demos_link.sh guarantees. A .desktop's Exec must be absolute,
# and REPO_ROOT is absolute but NOT stable: it is wherever the checkout happened
# to sit during the install. Move or rename the checkout afterwards (unzip to
# ~/Downloads, install, then tidy it into place) and every REPO_ROOT-baked entry
# is orphaned — GNOME runs a path that no longer exists, and because
# Terminal=false there is no error anywhere, just an icon that does nothing.
# $HOME/demos is the one path the kit re-points on every install.sh/run_demos.sh
# run, so an icon built on it survives the checkout moving. This also matches
# demo_install_server.py, which already resolves demos via Path.home()/"demos".
WRAPPER_FILE = REPO_ROOT / "start_demo_center.sh"
ICON_FILE = REPO_ROOT / "assets" / "ZGX_Demo_Center_icon_512.png"
DEMOS_LINK = Path.home() / "demos"
WRAPPER_LINK_PATH = DEMOS_LINK / "start_demo_center.sh"
ICON_LINK_PATH = DEMOS_LINK / "assets" / "ZGX_Demo_Center_icon_512.png"
DESKTOP_ENTRY_NAME = "HP ZGX Demo Center"
DESKTOP_FILE_BASENAME = "hp-zgx-demo-center.desktop"


def _desktop_exec_quote(path: Path) -> str:
    """Quote a path for a .desktop Exec= line.

    Per the freedesktop Desktop Entry spec an Exec argument containing reserved
    characters (space among them) must be double-quoted, with backslash and the
    quote character escaped inside. Unquoted, a checkout under a path with a
    space — "~/My Demos/..." — makes GNOME parse the Exec as several arguments
    and silently refuse to launch: another dead icon with no error.
    """
    escaped = str(path).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _desktop_entry_text() -> str:
    """The .desktop file body (absolute, symlink-stable on-device paths)."""
    return (
        "[Desktop Entry]\n"
        "Version=1.0\n"
        "Type=Application\n"
        f"Name={DESKTOP_ENTRY_NAME}\n"
        "Comment=Launch the HP ZGX AI Demo Center fullscreen\n"
        f"Exec={_desktop_exec_quote(WRAPPER_LINK_PATH)}\n"
        # Icon= is a plain path, not a command line — it takes no quoting.
        f"Icon={ICON_LINK_PATH}\n"
        "Terminal=false\n"
        "Categories=Utility;\n"
        "StartupNotify=true\n"
    )


def _write_desktop_file(path: Path, mark_trusted: bool) -> None:
    """Write + chmod +x a .desktop file; optionally mark it GNOME-trusted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_desktop_entry_text())
    path.chmod(0o755)
    log_done(f"Wrote desktop entry: {path}")
    if mark_trusted:
        if shutil.which("gio"):
            subprocess.run(
                ["gio", "set", str(path), "metadata::trusted", "true"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            log_check(f"Marked GNOME-trusted: {path}")
        else:
            log_warn("gio not found — skipping GNOME-trusted flag on the "
                     "Desktop icon (it may show as untrusted until launched once).")


def phase_chromium_desktop(flags: Flags) -> None:
    # Browser is a prerequisite (Phase 1), not installed here — this phase only
    # wires up the desktop icon + launch wrapper.
    log_section("Phase 7: Desktop icon + launch wrapper")
    home = Path.home()
    apps_desktop = home / ".local" / "share" / "applications" / DESKTOP_FILE_BASENAME
    desk_desktop = home / "Desktop" / DESKTOP_FILE_BASENAME

    # 7.1 — launch wrapper executable.
    if flags.dry_run:
        log_info(f"would chmod +x the launch wrapper: {WRAPPER_FILE}")
    elif not WRAPPER_FILE.exists():
        log_error(f"Launch wrapper not found at {WRAPPER_FILE} — it should be "
                  "tracked in the repo.")
        sys.exit(1)
    else:
        WRAPPER_FILE.chmod(0o755)
        log_done(f"Launch wrapper executable: {WRAPPER_FILE}")

    # 7.1b — the Exec= path is the ~/demos link, not REPO_ROOT, so confirm the
    # link actually reaches the wrapper before writing an entry that depends on
    # it. Preflight already asserts the link, but this phase ALWAYS runs (even
    # after earlier phases fail), so it can be reached with the link clobbered.
    # Warn rather than exit: a wrong icon must not fail an otherwise-good install.
    if not flags.dry_run and not WRAPPER_LINK_PATH.exists():
        log_warn(f"{WRAPPER_LINK_PATH} does not resolve — the desktop icon will "
                 f"be dead until ~/demos points at this checkout. Fix with:\n"
                 f"      ln -sfn '{REPO_ROOT}' '{DEMOS_LINK}'")

    # 7.2 — icon presence (warn-but-proceed if missing).
    if not ICON_FILE.exists():
        log_warn(f"Branded icon not found at {ICON_FILE} — the .desktop entries "
                 "will still point at it; it will render as a generic icon until "
                 "the file is present.")
    else:
        log_check(f"Branded icon present: {ICON_FILE}")

    # 7.3 — .desktop entries (applications menu + Desktop, GNOME-trusted).
    if flags.dry_run:
        log_info(f"would create .desktop entry: {apps_desktop}")
        log_info(f"would create .desktop entry: {desk_desktop} (GNOME-trusted)")
        log_info(f"  Exec={_desktop_exec_quote(WRAPPER_LINK_PATH)}")
        log_info(f"  Icon={ICON_LINK_PATH}")
        log_info("(browser is a prerequisite — Phase 1 verifies it; not installed here)")
        return

    _write_desktop_file(apps_desktop, mark_trusted=False)
    _write_desktop_file(desk_desktop, mark_trusted=True)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    # Line-buffer our own stdout (Fix 4c). When the installer's output is piped
    # (e.g. `install.sh | tee`), Python block-buffers stdout by default while the
    # streamed subprocesses (build_engine.py, `ollama pull`) write to the same fd
    # UNBUFFERED — so their output jumped AHEAD of our banners and a log line could
    # appear glued to the next ("completeBuilding…", with the engine output landing
    # before the Phase 4b banner). Line-buffering flushes every print at its newline,
    # so our prints and the subprocess output now interleave in true chronological
    # order. (No-op on a TTY, which is already line-buffered.)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    flags = parse_flags(sys.argv[1:])

    log_section("Demo Center — Python orchestrator")
    if flags.dry_run:
        log_info("DRY-RUN mode: no changes will be made.")
    if flags.force:
        log_info("FORCE mode: components will be reinstalled even if present.")
    if flags.skip_demos:
        log_info(f"Will skip demos (later slice): {', '.join(flags.skip_demos)}")

    # Phases that abort the run on failure (their own sys.exit): preflight, deps,
    # ollama+models, asset staging, server start.
    phase_preflight(flags)
    phase_deps(flags)
    phase_ollama_and_models(flags)
    phase_native_assets(flags)        # Phase 4 — stage onnx + videos
    phase_start_servers(flags)

    # Phase 6 — per-demo installs. Collect failures; do NOT abort. This also
    # installs the vision/pipeline runtime deps (incl. the env's tensorrt wheel).
    web_failed = phase_web_installs(flags)

    # Phase 4b — build the TRT engine WITH the env's TensorRT (installed by the
    # vision install above). Collect the result; do NOT abort.
    engine_ok = phase_build_engine(flags)

    # Phase 7 — desktop icon + launch wrapper. ALWAYS runs, even if a demo or the
    # engine build failed, so the box is launchable.
    phase_chromium_desktop(flags)

    # ── Done / summary ────────────────────────────────────────────────────────
    log_section("Done")
    if flags.dry_run:
        log_info("DRY-RUN complete — no changes were made.")
        return 0

    # Consolidated failure summary — exit non-zero HERE (after the icon exists)
    # if anything failed, so the rep sees one clear list at the end.
    problems: list[str] = []
    if web_failed:
        problems.append(f"demo install(s) failed: {', '.join(web_failed)}")
    if not engine_ok:
        problems.append("TRT engine build failed (AI Vision will not run until "
                        "yolov8s.trt is built)")

    if problems:
        log_error("Demo Center installed WITH ERRORS — the desktop icon was "
                  "created, but fix these and re-run install.sh (it skips what "
                  "already succeeded):")
        for p in problems:
            log_error(f"  • {p}")
        return 1

    log_done("Demo Center is installed.")
    log_info("Double-click the 'HP ZGX Demo Center' icon on the desktop to "
             "launch it (brings the demos up and opens fullscreen).")
    log_info(f"Or open  {LAUNCHER_URL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
