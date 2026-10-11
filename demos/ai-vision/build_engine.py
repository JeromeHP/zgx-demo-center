#!/usr/bin/env python3
"""Build the YOLOv8s FP16 TensorRT engine with the zgx env's TensorRT.

WHY THIS EXISTS (the v239/v240 clean-box failure): TensorRT engines are NOT
portable across TRT builds. The Demo Center runtime (ai-vision/vision_engine.py)
deserializes the engine with the zgx env's wheel (tensorrt-cu13==10.15.1.29).
Building the engine in the tensorrt:25.05-py3 container instead (TRT 10.10)
produced a serialized engine the runtime wheel REFUSED to deserialize
("serialization version mismatch") — the engine silently failed to load on a
clean box. The fix is to build with the SAME TensorRT the runtime uses, i.e.
run THIS script with the zgx env python so build TRT == runtime TRT.

  Input : ~/yolov8s.onnx   (1088x1088, FP16, opset 20 — staged by install Phase 4)
  Output: ~/yolov8s.trt    (== ENGINE_PATHS[0] in vision_engine.py)

Run via the orchestrator (install.py phase_build_engine) or by hand:
    ~/miniforge3/envs/zgx/bin/python3 ~/demos/ai-vision/build_engine.py

Exit codes: 0 = built (or already valid), 1 = build/parse error,
            2 = post-build deserialize mismatch (the exact bug this guards).
"""
import sys
import threading
import time
from pathlib import Path

import tensorrt as trt

ONNX = Path.home() / "yolov8s.onnx"
ENGINE = Path.home() / "yolov8s.trt"
INPUT_HW = 1088  # vision_engine.py: INPUT_W == INPUT_H == 1088
WORKSPACE_BYTES = 4 << 30  # 4 GiB build workspace (plenty on 128GB unified)
HEARTBEAT_SECONDS = 30  # Fix 3: progress ping during the long, output-less build


def _engine_deserializes(logger: "trt.Logger") -> bool:
    """True if the on-disk engine deserializes with THIS TensorRT (idempotency)."""
    try:
        runtime = trt.Runtime(logger)
        with open(ENGINE, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine is not None
    except Exception:
        return False


def build() -> int:
    logger = trt.Logger(trt.Logger.WARNING)

    # Fix 4b — IDEMPOTENT: if a usable engine is already on disk, skip the rebuild.
    # "Usable" = it deserializes with the CURRENT TensorRT (a mere existence check
    # would skip a stale engine built by a different TRT — the exact failure this
    # builder exists to prevent). Lets the builder be re-run by hand for free.
    if ENGINE.exists():
        if _engine_deserializes(logger):
            print(f"✓ existing engine verified (TRT {trt.__version__}) — skipping "
                  f"rebuild ({ENGINE})", flush=True)
            return 0
        print(f"⚠ existing {ENGINE} did NOT deserialize with TRT {trt.__version__} "
              "— rebuilding.", flush=True)

    if not ONNX.exists():
        print(f"✗ ONNX not found: {ONNX}", file=sys.stderr)
        return 1

    builder = trt.Builder(logger)
    # TRT 10 networks are always explicit-batch; create_network(0) is correct
    # (the old EXPLICIT_BATCH flag was removed in TRT 10).
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(ONNX, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  onnx parse error: {parser.get_error(i)}", file=sys.stderr)
            return 1

    config = builder.create_builder_config()
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)

    print(f"Building engine from {ONNX} (FP16, {INPUT_HW}x{INPUT_HW}) with "
          f"tensorrt {trt.__version__} — ~15-20 min on GB10; please wait...",
          flush=True)
    # Fix 3: build_serialized_network is a single blocking C call that emits no
    # output for ~15-20 min. A background heartbeat prints every 30s so the
    # operator sees it's alive instead of staring at a frozen screen.
    _stop = threading.Event()
    _t0 = time.monotonic()

    def _heartbeat() -> None:
        while not _stop.wait(HEARTBEAT_SECONDS):
            el = int(time.monotonic() - _t0)
            print(f"⏳ TRT engine build still running — {el // 60}m{el % 60:02d}s "
                  "elapsed (FP16 build is compute-bound; no per-layer output is "
                  "expected)", flush=True)

    _hb = threading.Thread(target=_heartbeat, daemon=True)
    _hb.start()
    try:
        serialized = builder.build_serialized_network(network, config)
    finally:
        _stop.set()
    if serialized is None:
        print("✗ build_serialized_network returned None — build failed.",
              file=sys.stderr)
        return 1
    ENGINE.write_bytes(serialized)
    print(f"✓ Wrote {ENGINE} ({ENGINE.stat().st_size / 1e6:.1f} MB)", flush=True)

    # POST-BUILD ASSERT — deserialize with THIS SAME tensorrt. This is the whole
    # point: it fails loudly if the engine and runtime TRT ever diverge.
    runtime = trt.Runtime(logger)
    with open(ENGINE, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        print(f"✗ POST-BUILD DESERIALIZE FAILED with tensorrt {trt.__version__} — "
              "the engine is unusable by the runtime (serialization mismatch). "
              "Aborting so this never ships a dead engine.", file=sys.stderr)
        return 2
    print(f"✓ Post-build deserialize OK with tensorrt {trt.__version__} "
          "(build == runtime TRT).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(build())
