#!/usr/bin/env python3
"""
ZGX Demo Center — accelerator detection, shared by the installer and the launcher.

One question, answered once: what silicon is this box, and which install/launch
variant does it need? Everything downstream (install.py preflight, /sysinfo, the
launcher system bar, each demo's "Powered by ..." string) reads the answer from
here instead of hardcoding a vendor.

STDLIB ONLY. install.py is stdlib-only by design (see its module docstring) and
imports this during preflight, before any pip install has run.

Two rules this module holds to:

  1. It never raises. A probe that errors, times out, or is missing leaves its
     field "unknown". Callers get a dict, always.
  2. It never fabricates. An unanswered field is "unknown", not a plausible
     default -- because a confident wrong number is worse than a visible gap.
     The launcher has been printing "NVIDIA CTK v1.18.2" as hardcoded HTML on a
     box that actually runs 1.20.0; that is the failure mode this replaces.

Override with ZGX_HW_PROFILE=nvidia-cuda|amd-rocm|cpu-only (forces the profile,
keeps real probe data) and ZGX_DEVICE_LABEL="..." (forces the display name).
The override is also how the AMD path gets exercised on an NVIDIA box.
"""

from __future__ import annotations

import glob
import json
import os
import platform
import re
import shutil
import subprocess

PROFILE_NVIDIA = "nvidia-cuda"
PROFILE_AMD = "amd-rocm"
PROFILE_CPU = "cpu-only"
VALID_PROFILES = (PROFILE_NVIDIA, PROFILE_AMD, PROFILE_CPU)

UNKNOWN = "unknown"

# Probes are cheap but not free, and a wedged GPU driver can hang a query.
# Every subprocess gets a hard timeout; a timeout is a failed probe, not a crash.
PROBE_TIMEOUT_S = 6

# Deliberately tiny and explicit. A chip we have not seen falls through to a
# generic vendor+accelerator label rather than a guessed product name -- we do
# not claim to know which chassis an unrecognised GPU is bolted into.
KNOWN_SYSTEMS = {
    "GB10": "HP ZGX Nano G1n",
    "8060S": "AMD Strix Halo (Ryzen AI Max)",
}


def _run(cmd: list[str]) -> str | None:
    """Run a probe. Return stdout on success, None on any failure at all."""
    if not shutil.which(cmd[0]):
        return None
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    out = (proc.stdout or "").strip()
    return out or None


def _system_memory_gb() -> float | None:
    """Total RAM from /proc/meminfo, in GiB.

    Not a fallback of last resort -- it is the PRIMARY source on unified-memory
    parts. Measured on the Nano, `nvidia-smi --query-gpu=memory.total` returns
    the literal string "[N/A]" for the GB10, because there is no separate VRAM
    to report. Strix Halo has the same shape for the same reason.
    """
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / (1024 * 1024), 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def _parse_gpu_memory(raw: str | None) -> float | None:
    """Parse an 'N MiB' figure. '[N/A]' and anything unparseable return None."""
    if not raw:
        return None
    m = re.search(r"(\d+)\s*MiB", raw)
    if not m:
        return None
    return round(int(m.group(1)) / 1024, 1)


def _probe_nvidia() -> dict | None:
    """NVIDIA via nvidia-smi. Returns None if this is not an NVIDIA box."""
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ])
    if not out:
        return None
    # "NVIDIA GB10, [N/A], 580.173.02"
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    name = parts[0] if parts and parts[0] else UNKNOWN
    mem = _parse_gpu_memory(parts[1] if len(parts) > 1 else None)
    driver = parts[2] if len(parts) > 2 and parts[2] else UNKNOWN
    return {
        "profile": PROFILE_NVIDIA,
        "vendor": "NVIDIA",
        "accelerator": name,
        "driver": f"NVIDIA {driver}" if driver != UNKNOWN else UNKNOWN,
        "gpu_memory_gb": mem,
        "detected_by": ["nvidia-smi"],
    }


def _probe_amd() -> dict | None:
    """AMD via /dev/kfd plus rocminfo/rocm-smi. Returns None if not an AMD box.

    /dev/kfd is the discriminator, NOT /dev/dri. Checked on the Nano: an NVIDIA
    box also exposes /dev/dri/renderD128, so keying on /dev/dri would classify
    every machine as AMD. /dev/kfd is the ROCm compute node and is absent there.
    """
    if not os.path.exists("/dev/kfd"):
        return None

    detected_by = ["/dev/kfd"]
    name, gfx = UNKNOWN, UNKNOWN

    info = _run(["rocminfo"])
    if info:
        detected_by.append("rocminfo")
        marketing = re.search(r"Marketing Name:\s*(.+)", info)
        if marketing:
            name = marketing.group(1).strip()
        target = re.search(r"\b(gfx[0-9a-f]+)\b", info)
        if target:
            gfx = target.group(1)

    if name == UNKNOWN:
        product = _run(["rocm-smi", "--showproductname", "--csv"])
        if product:
            detected_by.append("rocm-smi")
            for line in product.splitlines()[1:]:
                cells = [c.strip() for c in line.split(",")]
                named = [c for c in cells[1:] if c and c.upper() != "N/A"]
                if named:
                    name = named[0]
                    break

    version = _run(["cat", "/opt/rocm/.info/version"]) or _run(["hipconfig", "--version"])
    driver = f"ROCm {version.splitlines()[0].strip()}" if version else UNKNOWN

    accelerator = name if gfx == UNKNOWN or gfx in name else f"{name} ({gfx})"
    return {
        "profile": PROFILE_AMD,
        "vendor": "AMD",
        "accelerator": accelerator,
        "driver": driver,
        "gpu_memory_gb": _parse_gpu_memory(_run(["rocm-smi", "--showmeminfo", "vram"])),
        "gfx_target": gfx,
        "detected_by": detected_by,
    }


def _container_runtime(profile: str) -> tuple[str, list[str]]:
    """What Docker needs to reach the GPU, and any blocking problems.

    The two vendors do this completely differently, which is why the installer
    cannot just check for one of them:
      NVIDIA -- the NVIDIA Container Toolkit, `nvidia-ctk runtime configure`.
      AMD    -- no toolkit at all; the container gets `--device /dev/kfd
                --device /dev/dri` and the invoking user must be in render+video.
    """
    problems: list[str] = []
    if profile == PROFILE_NVIDIA:
        out = _run(["nvidia-ctk", "--version"])
        if not out:
            return UNKNOWN, ["NVIDIA Container Toolkit (nvidia-ctk) not found"]
        m = re.search(r"version\s+(\S+)", out)
        return (f"nvidia-container-toolkit {m.group(1)}" if m else "nvidia-container-toolkit"), problems

    if profile == PROFILE_AMD:
        if not glob.glob("/dev/dri/renderD*"):
            problems.append("no /dev/dri/renderD* render node")
        # Group membership is checked, not assumed: without render+video the
        # device nodes are present but unreadable, and the container fails at
        # runtime with a permission error that looks nothing like its cause.
        try:
            groups = {g for g in subprocess.run(
                ["id", "-nG"], capture_output=True, text=True,
                timeout=PROBE_TIMEOUT_S, check=False).stdout.split()}
            for needed in ("render", "video"):
                if needed not in groups:
                    problems.append(f"user is not in the '{needed}' group")
        except (OSError, subprocess.SubprocessError):
            problems.append("could not read group membership")
        return "rocm (/dev/kfd + /dev/dri)", problems

    return "none", problems


def _device_label(profile: str, vendor: str, accelerator: str) -> str:
    override = os.environ.get("ZGX_DEVICE_LABEL", "").strip()
    if override:
        return override
    for token, label in KNOWN_SYSTEMS.items():
        if token.lower() in accelerator.lower():
            return label
    if profile == PROFILE_CPU or accelerator == UNKNOWN:
        return f"{platform.machine()} (no accelerator detected)"
    return accelerator if accelerator.startswith(vendor) else f"{vendor} {accelerator}"


def detect() -> dict:
    """Identify this machine. Never raises; unanswered fields read 'unknown'."""
    forced = os.environ.get("ZGX_HW_PROFILE", "").strip()
    if forced and forced not in VALID_PROFILES:
        forced = ""

    found = _probe_nvidia() or _probe_amd()
    if found is None:
        found = {
            "profile": PROFILE_CPU, "vendor": UNKNOWN, "accelerator": UNKNOWN,
            "driver": UNKNOWN, "gpu_memory_gb": None, "detected_by": [],
        }

    profile = forced or found["profile"]
    warnings: list[str] = []
    if forced and forced != found["profile"]:
        warnings.append(
            f"profile forced to {forced} via ZGX_HW_PROFILE; probes detected {found['profile']}"
        )

    runtime, problems = _container_runtime(profile)
    warnings.extend(problems)

    gpu_mem = found.get("gpu_memory_gb")
    sys_mem = _system_memory_gb()
    if gpu_mem:
        memory_gb, memory_source = gpu_mem, "gpu"
    elif sys_mem:
        memory_gb, memory_source = sys_mem, "system (unified)"
    else:
        memory_gb, memory_source = None, UNKNOWN

    accelerator = found["accelerator"]
    vendor = found["vendor"]
    return {
        "profile": profile,
        "vendor": vendor,
        "accelerator": accelerator,
        "gfx_target": found.get("gfx_target", ""),
        "arch": platform.machine(),
        "cpu_cores": os.cpu_count(),
        "driver": found["driver"],
        "memory_gb": memory_gb,
        "memory_source": memory_source,
        "container_runtime": runtime,
        "device_label": _device_label(profile, vendor, accelerator),
        "detected_by": found["detected_by"],
        "warnings": warnings,
    }


if __name__ == "__main__":
    print(json.dumps(detect(), indent=2))
