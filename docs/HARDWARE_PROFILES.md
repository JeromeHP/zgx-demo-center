# Hardware profiles — running the Demo Center on NVIDIA and AMD

Shows are funded by AMD and NVIDIA in turn, so the suite has to run on both
without being rebuilt per event. This is how that works, what it cost, and
what is still unverified.

## The profile

`demos/zgx_hardware.py` answers one question — what silicon is this? — and
everything downstream reads the answer instead of hardcoding a vendor.

```
$ python3 demos/zgx_hardware.py
{ "profile": "nvidia-cuda", "vendor": "NVIDIA", "accelerator": "NVIDIA GB10",
  "arch": "aarch64", "memory_gb": 121.6, "device_label": "HP ZGX Nano G1n", ... }
```

Profiles: `nvidia-cuda`, `amd-rocm`, `cpu-only`.

Two rules it holds to, both learned from bugs in this repo:

- **It never raises.** A missing tool or a wedged driver leaves a field
  `"unknown"`; callers always get a dict.
- **It never fabricates.** `"unknown"` rather than a plausible default. Pipeline
  Pulse used to report a flat `102 GB / 128 GB` when NVML was absent, while its
  own tooltip told the audience the figure came live from NVML.

### Overrides

```bash
ZGX_HW_PROFILE=amd-rocm      # force a profile (probes still report real data)
ZGX_DEVICE_LABEL="…"         # force the display name
VISION_BACKEND=onnxruntime   # force ai-vision's detection tier
MARITIME_COMPOSE="-f docker-compose.yml -f docker-compose.amd.yml"
```

`ZGX_HW_PROFILE` is how every AMD code path in this repo was exercised before
any AMD hardware existed. A forced run is flagged in `warnings`, so it can never
be mistaken for a real detection.

## Two axes, not one

The Nano is **aarch64 + CUDA**; Strix Halo is **x86_64 + ROCm**. Base images
matter as much as GPU stacks: `python:3.12-slim` is multi-arch, which is why
most demos were nearly free, and `nvcr.io/nvidia/*` is neither multi-arch nor
multi-vendor, which is why two were not.

## Per demo

| Demo | AMD path | Notes |
|---|---|---|
| grant, pii, pharmacy, ballistics, vehicle | unchanged | host Ollama over HTTP; no GPU code at all |
| construction | CPU wheels | runs `device="cpu"` anyway — it was downloading a CUDA torch it never used |
| translation | `rocm6.4` torch index | `bitsandbytes` dropped: declared, imported nowhere |
| pipeline-pulse | rocm-smi telemetry | presenter narration rewritten at render time |
| spectrum | numpy tier | `ew_infer.py` already tiered TRT → numpy → heuristic and labels the live tier |
| vision | ONNX Runtime | same `~/yolov8s.onnx` the `.trt` engine is built from |
| maritime | host Ollama | `docker-compose.amd.yml`; no vLLM, no GPU in the container |
| **medical** | **none — out of scope** | nvcr vLLM + `awq_marlin` + CUDA whisper.cpp + a physical robot |

## Adding a demo

1. If it needs an accelerator only one vendor has, declare it:
   `"requires": {"profiles": ["nvidia-cuda"]}`. It is then **skipped with a
   reason** rather than attempted and failed — a mid-install failure buries the
   demos that did work.
2. If it only needs different install steps, use
   `"profiles": {"amd-rocm": {"steps": [...]}}`. Merged by `resolve_plan()` in
   `demo_install_server.py`; `install_plans.py` stays pure data.
3. Never type a hardware fact into markup. Read `/sysinfo`.
   `scripts/audit_hardware_strings.sh` enforces this and is symmetric — Radeon,
   gfx####, Ryzen and Strix Halo are banned alongside the NVIDIA names, or the
   guard just swaps which vendor gets hardcoded.

## Confirmed on AMD silicon

Run on a Strix Halo desktop, **ROCm 7.2.1**, 2026-09-22:

- `scripts/install_onnxruntime_rocm.sh` selected `onnxruntime-migraphx` (correct
  for the post-7.1 rename — plain `onnxruntime-rocm` would have found nothing),
  resolved `rocm-rel-7.2.1`, installed `onnxruntime_migraphx-1.27.1-cp312-
  manylinux_2_34_x86_64`, and reported
  `MIGraphXExecutionProvider,CPUExecutionProvider`.
- So AI Vision's GPU detection path is real on ROCm, not just plausible. The
  header should read MIGraphX rather than amber CPU once the demo is restarted.

## What is NOT yet verified on AMD silicon

Everything above was developed and tested on the GB10 with `ZGX_HW_PROFILE`
forcing. That exercises every branch, but it cannot prove:

- **rocm-smi's JSON key names.** `zgx_gpu_telemetry.py` matches on substrings
  because those keys have moved between ROCm releases. The AMD test fixtures
  were written from documented output, **not** from a real gfx1151 box. Run
  `bash demos/scripts/capture_hw_fixture.sh` on the AMD machine first and
  replace them — a fixture invented from a man page passes while testing
  nothing.
- **Whether Spectrum Duel's numpy tier is fast enough.** Its docstring claims
  ~1 ms against a 50 ms frame budget, but the tile is now two-channel, so the
  cost has roughly doubled. Measure before deciding it needs a real port.
- **Throughput generally.** `reports/baseline_zbook.json` in grant-machine is a
  Strix Halo baseline (gfx1151, Radeon 8060S) at **23.8 tok/s** against the
  Nano's **77.2**, and quick-scope E2E of **1313 s vs 189.9 s**. That came from
  a laptop APU under WSL2 with llama.cpp and a 64 GB cap, so a native-Linux
  desktop with current ROCm should do better — but **measure before promising a
  live run on a slide.**


## The ROCm ONNX Runtime (AI Vision's GPU detection path)

AMD does **not** publish these wheels to PyPI, so `pip install onnxruntime-rocm`
on a stock pip quietly installs nothing of the sort and leaves you on CPU. They
live in per-release flat directories under `repo.radeon.com`, consumed with
`pip -f`.

`scripts/install_onnxruntime_rocm.sh` handles this. Three facts it encodes, all
verified against the live index rather than documentation:

| ROCm | Package | Provider |
|---|---|---|
| ≤ 7.0.x | `onnxruntime-rocm` | `ROCMExecutionProvider` |
| ≥ 7.1 | `onnxruntime-migraphx` | `MIGraphXExecutionProvider` |

- **The package renames at 7.1.** A plain `onnxruntime-rocm` against a 7.2 index
  finds nothing.
- **Not every point release ships a wheel.** `rocm-rel-7.1.1` has neither
  package, so an exact-version URL has to degrade to the `X.Y` line.
- **Wheels are cp310/cp312, x86_64 only.** The `zgx` env is Python 3.12, which
  matches.

It then checks `onnxruntime.get_available_providers()`, because **pip succeeding
proves nothing** — a wheel can install cleanly and still leave `CPUExecutionProvider`
as the only option. A CPU-only outcome prints a loud warning and explains the
three usual causes; `ZGX_REQUIRE_ROCM_EP=1` makes it a hard failure instead.

Run it on its own at any time:

```bash
bash demos/scripts/install_onnxruntime_rocm.sh
```
