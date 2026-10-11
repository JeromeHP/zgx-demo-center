"""Vendor-neutral GPU telemetry: clock, power, temperature, utilisation, memory.

Three demos each grew their own copy of the same nvidia-smi query --
pipeline-pulse, ai-vision and ew-spectrum-duel -- and all three hardcoded
NVIDIA. On an AMD box they do not degrade, they simply report nothing while the
UI keeps its labels, which reads to an audience as "the telemetry is broken"
rather than "this build has no AMD backend".

This is the one implementation, with an nvidia backend and an amd backend behind
the same dict. Same rules as zgx_hardware: it never raises, and a field it
cannot read is None rather than 0 -- because 0 W and 0 degrees are values an
audience will believe, and "--" is not.

STDLIB ONLY for the GPU path, so the demos can import it without adding a
dependency. pynvml (GPU memory) and psutil (CPU/RAM) are used opportunistically
when present and never required -- exactly as pipeline-pulse already guarded
them before this module absorbed its CPU/RAM block.

    from zgx_gpu_telemetry import GpuPoller
    poller = GpuPoller(); poller.start()
    poller.get()   # -> dict, safe to call from any thread
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time

EMPTY = {
    "clock_mhz": None, "power_w": None, "temp_c": None, "util_pct": None,
    "mem_used_gb": None, "mem_total_gb": None,
    "available": False, "vendor": "unknown", "source": "none",
}

_NVIDIA_QUERY = [
    "nvidia-smi",
    "--query-gpu=clocks.current.graphics,power.draw,temperature.gpu,utilization.gpu",
    "--format=csv,noheader,nounits",
]


def _num(raw, cast):
    """Parse a value nvidia-smi/rocm-smi might render as '[N/A]', 'N/A' or ''."""
    if raw is None:
        return None
    text = str(raw).strip().strip("%").replace("Mhz", "").replace("MHz", "")
    text = re.sub(r"[()]", "", text)
    if not text or text.upper().startswith("N/A") or text.startswith("["):
        return None
    try:
        return cast(float(text))
    except (TypeError, ValueError):
        return None


def _run(cmd, timeout=4):
    if not shutil.which(cmd[0]):
        return None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def _read_nvidia(nvml_handle=None):
    out = _run(_NVIDIA_QUERY)
    if not out:
        return None
    parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
    if len(parts) < 4:
        return None
    data = dict(EMPTY)
    data.update({
        "clock_mhz": _num(parts[0], int), "power_w": _num(parts[1], float),
        "temp_c": _num(parts[2], int), "util_pct": _num(parts[3], int),
        "available": True, "vendor": "NVIDIA", "source": "nvidia-smi",
    })
    if nvml_handle is not None:
        try:
            import pynvml
            mi = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            data["mem_used_gb"] = round(mi.used / 1024 ** 3, 1)
            data["mem_total_gb"] = round(mi.total / 1024 ** 3, 1)
        except Exception:
            pass
    return data


# rocm-smi's JSON keys have moved between ROCm releases (and differ by part), so
# this matches on substrings rather than exact names. Verify against real output
# from scripts/capture_hw_fixture.sh on first contact with an AMD box; the key
# list is the thing most likely to need a line added, not the structure.
_AMD_FIELDS = [
    ("temp_c",       ("temperature (sensor edge)", "temperature (sensor junction)", "temp"), int),
    ("power_w",      ("average graphics package power", "current socket graphics package power", "power"), float),
    ("util_pct",     ("gpu use (%)", "gpu use"), int),
    ("clock_mhz",    ("sclk clock speed", "sclk"), int),
]


def _read_amd():
    out = _run(["rocm-smi", "--showtemp", "--showpower", "--showuse",
                "--showclocks", "--showmeminfo", "vram", "--json"])
    if not out:
        return None
    try:
        payload = json.loads(out)
    except ValueError:
        return None
    card = next((v for v in payload.values() if isinstance(v, dict)), None)
    if not card:
        return None

    lowered = {str(k).lower(): v for k, v in card.items()}
    data = dict(EMPTY)
    data.update({"available": True, "vendor": "AMD", "source": "rocm-smi"})
    for field, needles, cast in _AMD_FIELDS:
        for key, value in lowered.items():
            if any(n in key for n in needles):
                parsed = _num(value, cast)
                if parsed is not None:
                    data[field] = parsed
                    break

    used = total = None
    for key, value in lowered.items():
        if "vram" in key and "total memory" in key:
            total = _num(value, float)
        elif "vram" in key and "used memory" in key:
            used = _num(value, float)
    if total:
        data["mem_total_gb"] = round(total / 1024 ** 3, 1)
    if used is not None:
        data["mem_used_gb"] = round(used / 1024 ** 3, 1)
    return data


def read_system() -> dict:
    """CPU and RAM. Optional: psutil is not a hard dependency of this module.

    Moved here from pipeline-pulse (_poll_sys_once), which was the only CPU/RAM
    implementation in the tree -- so every other demo's hardware panel could
    show a GPU and nothing else. Same guarded-import shape it always had.
    """
    out = {"cpu_pct": None, "ram_used_gb": None, "ram_pct": None}
    try:
        import psutil
    except Exception:
        return out
    try:
        out["cpu_pct"] = round(psutil.cpu_percent(interval=None), 1)
        vm = psutil.virtual_memory()
        out["ram_used_gb"] = round(vm.used / (1024 ** 3), 1)
        out["ram_pct"] = round(vm.percent, 1)
    except Exception:
        pass
    return out


def read(nvml_handle=None, include_system: bool = False) -> dict:
    """One telemetry sample. Always returns a dict; never raises."""
    try:
        sample = _read_nvidia(nvml_handle) or _read_amd() or dict(EMPTY)
    except Exception:
        sample = dict(EMPTY)
    if include_system:
        try:
            sample.update(read_system())
        except Exception:
            sample.update({"cpu_pct": None, "ram_used_gb": None, "ram_pct": None})
    return sample


class GpuPoller(threading.Thread):
    """Background sampler. `get()` is cheap and thread-safe.

    Replaces three near-identical pollers. Unlike two of them it does not give
    up permanently when the first probe fails: a driver that is briefly wedged,
    or a tool that appears after start-up, recovers on the next tick.
    """

    def __init__(self, interval_s: float = 1.0):
        super().__init__(daemon=True, name="gpu-telemetry")
        self._interval = interval_s
        self._lock = threading.Lock()
        self._data = dict(EMPTY)
        self._nvml_handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml_handle = None

    def run(self):
        # psutil.cpu_percent(interval=None) returns 0.0 on its very first call
        # (it has no previous sample to diff against), so prime it once rather
        # than publishing a fabricated 0% as the opening reading.
        try:
            import psutil
            psutil.cpu_percent(interval=None)
        except Exception:
            pass
        while True:
            # read() is defensive, but this loop must survive even a read() that
            # raises -- an unhandled exception here kills the thread and freezes
            # every meter in the Demo Center at its last value, with nothing
            # saying the feed died. Found when a test stub with the wrong
            # signature silently killed the poller instead of failing loudly.
            try:
                sample = read(self._nvml_handle, include_system=True)
            except Exception:
                sample = dict(EMPTY)
            with self._lock:
                self._data = sample
            time.sleep(self._interval)

    def get(self) -> dict:
        with self._lock:
            return dict(self._data)


if __name__ == "__main__":
    print(json.dumps(read(), indent=2))
