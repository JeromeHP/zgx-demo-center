"""
Spectrum Duel — jam-technique classifier with three fallback tiers.

AI Vision raises FileNotFoundError when its engine is missing, which kills the
demo on stage. This one never hard-fails:

  1. tensorrt-fp16  engine deserializes + CUDA context OK   ~0.03 ms
  2. numpy-cpu      TRT unavailable, weights present         ~1 ms
                    (identical math to tier 1 modulo FP16 rounding — the numpy
                    forward pass here IS the one the trainer used)
  3. heuristic      weights missing too                      ~microseconds
                    (visibly worse, and labelled as such)

Set EW_FORCE_CPU=1 to force tier 2, or EW_ENGINE_PATH to point elsewhere. The
UI shows the active tier, so a degraded run is always honestly labelled.

CUDA affinity: the TensorRT context is created in whichever thread calls
start(), and every classify() must come from that same thread. The demo calls
both from the simulation thread, so inference is inline — at 0.03 ms against a
50 ms frame budget there is nothing to gain from a queue, and a queue would
only add staleness and a failure mode.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from ew_engine import CLASSES, NCLASS, TILE_BINS, TILE_CH, TILE_ROWS

HERE = Path(__file__).resolve().parent
DEFAULT_ENGINE = HERE / "models" / "jammer_cls.trt"
DEFAULT_WEIGHTS = HERE / "models" / "jammer_cls.npz"

TIER_TRT = "tensorrt-fp16"
TIER_NUMPY = "numpy-cpu"
TIER_HEURISTIC = "heuristic"

TIER_LABEL = {
    TIER_TRT: "TensorRT FP16 · GPU",
    TIER_NUMPY: "CPU fallback · numpy",
    TIER_HEURISTIC: "DEGRADED · heuristic",
}


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class Classifier:
    def __init__(self, engine_path=None, weights_path=None, force_cpu=None):
        self.engine_path = Path(os.environ.get("EW_ENGINE_PATH", engine_path or DEFAULT_ENGINE))
        self.weights_path = Path(weights_path or DEFAULT_WEIGHTS)
        self.force_cpu = (os.environ.get("EW_FORCE_CPU", "") == "1") if force_cpu is None else force_cpu

        self.backend = TIER_HEURISTIC
        self.train_accuracy = None
        self._w = None
        self._trt = None
        self._lat = deque(maxlen=200)
        self._lock = threading.Lock()
        self._notes = []

        if self.weights_path.exists():
            try:
                d = np.load(self.weights_path, allow_pickle=False)
                self._w = {k: d[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")}
                if "accuracy" in d:
                    self.train_accuracy = float(d["accuracy"])
                self.backend = TIER_NUMPY
            except Exception as exc:
                self._notes.append(f"weights unreadable ({exc}) — heuristic tier")
        else:
            self._notes.append(f"{self.weights_path.name} missing — heuristic tier")

    # ── lifecycle (call from the classifying thread) ───────────────────────
    def start(self):
        if self.force_cpu:
            self._notes.append("EW_FORCE_CPU=1 — TensorRT skipped")
            return
        if self._w is None:
            return
        if not self.engine_path.exists():
            self._notes.append(f"{self.engine_path.name} missing — CPU fallback")
            return
        try:
            import pycuda.driver as cuda
            import tensorrt as trt

            cuda.init()
            ctx = cuda.Device(0).make_context()
            # Record the context BEFORE anything else can throw — otherwise a
            # failure below leaves it pushed with no reference, and pycuda
            # aborts the process ("context stack was not empty") on exit. This
            # is the exact path a TRT-version-mismatched engine takes, which is
            # precisely when the fallback is supposed to save us.
            self._trt = {"cuda": cuda, "ctx": ctx}
            logger = trt.Logger(trt.Logger.ERROR)
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
            if engine is None:
                raise RuntimeError("deserialize returned None (TRT version mismatch?)")
            exec_ctx = engine.create_execution_context()
            nbytes_in = TILE_CH * TILE_ROWS * TILE_BINS * 4
            d_in = cuda.mem_alloc(nbytes_in)
            out = np.empty((1, NCLASS), np.float32)
            d_out = cuda.mem_alloc(out.nbytes)
            exec_ctx.set_tensor_address("spec", int(d_in))
            exec_ctx.set_tensor_address("probs", int(d_out))
            stream = cuda.Stream()
            self._trt.update(engine=engine, exec_ctx=exec_ctx, d_in=d_in,
                             d_out=d_out, out=out, stream=stream,
                             trt_version=trt.__version__)
            self.backend = TIER_TRT
            warm = np.zeros((TILE_CH, TILE_ROWS, TILE_BINS), np.float32)
            for _ in range(20):
                self._infer_trt(warm)
            self._lat.clear()
        except Exception as exc:
            self._notes.append(f"TensorRT init failed ({exc}) — CPU fallback")
            self._teardown_trt()

    def _teardown_trt(self):
        t, self._trt = self._trt, None
        if t:
            # Order matters: cuMemFree needs a current context, so release the
            # allocations while the context is still pushed, then pop, then
            # detach.
            for k in ("d_in", "d_out", "exec_ctx", "engine", "stream"):
                try:
                    obj = t.pop(k, None)
                    if obj is not None and hasattr(obj, "free"):
                        obj.free()
                except Exception:
                    pass
            ctx = t.get("ctx")
            if ctx is not None:
                try:
                    ctx.pop()
                except Exception:
                    pass
                try:
                    ctx.detach()
                except Exception:
                    pass
        if self._w is not None:
            self.backend = TIER_NUMPY

    def close(self):
        self._teardown_trt()

    # ── the three tiers ────────────────────────────────────────────────────
    def _infer_trt(self, tile):
        t = self._trt
        cuda = t["cuda"]
        buf = np.ascontiguousarray(
            tile.reshape(1, TILE_CH, TILE_ROWS, TILE_BINS), np.float32)
        t0 = time.perf_counter()
        cuda.memcpy_htod_async(t["d_in"], buf, t["stream"])
        t["exec_ctx"].execute_async_v3(t["stream"].handle)
        cuda.memcpy_dtoh_async(t["out"], t["d_out"], t["stream"])
        t["stream"].synchronize()
        self._lat.append((time.perf_counter() - t0) * 1000.0)
        return t["out"][0].copy()

    def _infer_numpy(self, tile):
        from ew_train import POOL1, POOL2, conv_fwd, pool_fwd

        w = self._w
        t0 = time.perf_counter()
        x = tile.reshape(1, TILE_CH, TILE_ROWS, TILE_BINS).astype(np.float32)
        z1, _, _, _ = conv_fwd(x, w["W1"], w["b1"])
        p1, _ = pool_fwd(np.maximum(z1, 0), POOL1)
        z2, _, _, _ = conv_fwd(p1, w["W2"], w["b2"])
        p2, _ = pool_fwd(np.maximum(z2, 0), POOL2)
        logits = p2.reshape(1, -1) @ w["W3"] + w["b3"]
        self._lat.append((time.perf_counter() - t0) * 1000.0)
        return _softmax(logits[0])

    def _infer_heuristic(self, tile):
        """Last-resort rules on the tile. Deliberately crude, honestly labelled."""
        t0 = time.perf_counter()
        tile = np.asarray(tile).reshape(TILE_CH, TILE_ROWS, TILE_BINS)[0]
        floor = np.median(tile)
        hot = tile > floor + 0.18
        occ = hot.mean()
        col_energy = tile.mean(axis=0)
        peaks = np.argsort(col_energy)[-3:]
        per_row_peak = tile.argmax(axis=1).astype(float)
        drift = float(np.abs(np.diff(per_row_peak)).mean())
        spread = float(np.std(per_row_peak))

        scores = np.zeros(NCLASS)
        scores[CLASSES.index("CLEAR")] = 1.0 - min(occ * 6.0, 1.0)
        scores[CLASSES.index("BARRAGE")] = min(occ * 3.5, 1.0)
        scores[CLASSES.index("SWEPT")] = min(drift / 6.0, 1.0)
        scores[CLASSES.index("SPOT")] = 1.0 if (occ < 0.25 and spread < 3.0) else 0.2
        scores[CLASSES.index("DRFM")] = 0.6 if (len(peaks) >= 2 and 1 < abs(int(peaks[-1]) - int(peaks[-2])) < 8) else 0.15
        self._lat.append((time.perf_counter() - t0) * 1000.0)
        return _softmax(scores * 3.0)

    # ── public API ─────────────────────────────────────────────────────────
    def classify(self, tile: np.ndarray) -> dict:
        try:
            if self.backend == TIER_TRT:
                probs = self._infer_trt(tile)
            elif self.backend == TIER_NUMPY:
                probs = self._infer_numpy(tile)
            else:
                probs = self._infer_heuristic(tile)
        except Exception as exc:
            # Never let a classifier fault take the demo down — drop a tier.
            with self._lock:
                self._notes.append(f"{self.backend} faulted ({exc}) — dropping a tier")
            if self.backend == TIER_TRT:
                self._teardown_trt()
            elif self.backend == TIER_NUMPY:
                self._w = None
            if self._w is None:
                self.backend = TIER_HEURISTIC
            try:
                probs = self._infer_heuristic(tile)
            except Exception:
                # Absolute last resort: an all-CLEAR vote. The demo keeps
                # running rather than taking the sim thread down with it.
                probs = np.zeros(NCLASS, np.float32)
                probs[CLASSES.index("CLEAR")] = 1.0
            self.backend = TIER_HEURISTIC if self._w is None else self.backend

        i = int(np.argmax(probs))
        return {
            "label": CLASSES[i],
            "confidence": float(probs[i]),
            "probs": [float(p) for p in probs],
            "backend": self.backend,
            "backend_label": TIER_LABEL[self.backend],
            "latency_ms": round(self.latency_p50(), 4),
        }

    def latency_p50(self) -> float:
        return float(np.percentile(self._lat, 50)) if self._lat else 0.0

    def info(self) -> dict:
        return {
            "backend": self.backend,
            "backend_label": TIER_LABEL[self.backend],
            "latency_ms_p50": round(self.latency_p50(), 4),
            "engine_path": str(self.engine_path),
            "engine_present": self.engine_path.exists(),
            "trt_version": (self._trt or {}).get("trt_version"),
            "train_accuracy": self.train_accuracy,
            "classes": CLASSES,
            "notes": list(self._notes),
        }
