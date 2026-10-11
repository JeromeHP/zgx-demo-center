"""GPU telemetry — thin adapter over the shared vendor-neutral poller.

This file used to be a ~95-line NVIDIA-only poller, duplicated verbatim between
ai-vision and ew-spectrum-duel (its own docstring said so, and said to fix bugs
in both places). Both hardcoded nvidia-smi, so on an AMD box they reported
nothing while the UI kept its labels -- which reads to an audience as broken
telemetry rather than an unsupported vendor.

The implementation now lives in demos/zgx_gpu_telemetry.py, with nvidia-smi and
rocm-smi backends. The old docstring justified the duplication by saying
make_bundle.sh archives each demo independently; it does not -- it runs a single
`git archive` of the whole repo HEAD, so the shared module travels with them.

The public surface is unchanged: GpuPoller().start() / .get().
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from zgx_gpu_telemetry import EMPTY, GpuPoller, read  # noqa: F401
except Exception:  # pragma: no cover - the demo must still start
    import threading

    EMPTY = {
        "clock_mhz": None, "power_w": None, "temp_c": None, "util_pct": None,
        "mem_used_gb": None, "mem_total_gb": None,
        "available": False, "vendor": "unknown", "source": "none",
    }

    def read(nvml_handle=None):
        return dict(EMPTY)

    class GpuPoller(threading.Thread):
        def __init__(self, interval_s=1.0):
            super().__init__(daemon=True, name="gpu-telemetry-stub")

        def run(self):
            pass

        def get(self):
            return dict(EMPTY)
