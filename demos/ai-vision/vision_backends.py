"""Detection backends for AI Vision: TensorRT on NVIDIA, ONNX Runtime elsewhere.

Why this exists
---------------
The inference path was TensorRT and pycuda inline in vision_engine.py, and when
the engine was missing VisionEngine.__init__ raised FileNotFoundError -- which
kills the demo on stage. The Spectrum Duel demo's ew_infer.py says so in its own
docstring: "AI Vision raises FileNotFoundError when its engine is missing, which
kills the demo on stage. This one never hard-fails." This module gives AI Vision
the same treatment, and adds the AMD path at the same time.

Tiers, in order:

  1. tensorrt     a prebuilt .trt engine + a working CUDA context   (NVIDIA)
  2. onnxruntime  yolov8s.onnx through ORT, best provider available (AMD, or
                  NVIDIA without an engine; CPU as the last provider)
  3. none         neither is usable -- the demo still starts and says so

Tier 2 is not merely a fallback. On AMD it is the intended path: ~/yolov8s.onnx
is already a shipped asset (assets/yolov8s-onnx.tar.gz, staged by install.py
Phase 4), and it is the same file the TensorRT engine is built from, so both
tiers run identical weights.

The active tier and the real provider are exposed so the UI can label them. A
demo running 8x slower on a CPU provider while displaying "TRT FP16" would be a
lie of exactly the kind this project keeps removing.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

TIER_TRT = "tensorrt"
TIER_ORT = "onnxruntime"
TIER_NONE = "none"

ENGINE_PATHS = [Path.home() / "yolov8s.trt", Path.home() / "models" / "yolov8s.trt"]
ONNX_PATHS = [Path.home() / "yolov8s.onnx", Path.home() / "models" / "yolov8s.onnx"]

# Preference order. MIGraphX and ROCm are the AMD paths; CUDA/TensorRT here are
# only reached when ORT is used on an NVIDIA box with no prebuilt engine. CPU is
# last and is always available, which is what makes tier 2 non-optional.
ORT_PROVIDER_PREFERENCE = [
    "MIGraphXExecutionProvider",
    "ROCMExecutionProvider",
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]


def _first_existing(paths):
    for p in paths:
        if p.is_file():
            return p
    return None


class Backend:
    """Common surface. open()/close() run on the inference thread."""

    tier = TIER_NONE
    detail = "no detection backend available"
    provider = "none"
    in_shape = (1, 3, 640, 640)
    out_shape = (1, 84, 8400)
    in_dtype = np.float32

    def open(self, n_slots: int) -> None: ...
    def infer(self, slot: int, arr: np.ndarray): raise NotImplementedError
    def close(self) -> None: ...

    @property
    def label(self) -> str:
        return f"{self.tier} · {self.provider}"


class TensorRTBackend(Backend):
    """The original path, behaviour-for-behaviour: per-slot execution contexts,
    pinned host buffers, device buffers, async memcpy, execute_async_v3."""

    tier = TIER_TRT
    provider = "cuda"

    def __init__(self, engine_path: Path):
        self.engine_path = str(engine_path)
        self.detail = f"TensorRT engine {Path(engine_path).name}"
        self._cuda = None
        self._ctx = None

    def open(self, n_slots: int) -> None:
        import pycuda.driver as cuda
        import tensorrt as trt

        cuda.init()
        self._cuda = cuda
        self._ctx = cuda.Device(0).make_context()

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as fh, trt.Runtime(logger) as rt:
            engine = rt.deserialize_cuda_engine(fh.read())
        if engine is None:
            raise RuntimeError("deserialize_cuda_engine returned None (TRT version mismatch?)")

        self._engine = engine
        in_name = engine.get_tensor_name(0)
        out_name = engine.get_tensor_name(1)
        self.in_shape = tuple(engine.get_tensor_shape(in_name))
        self.out_shape = tuple(engine.get_tensor_shape(out_name))
        self.in_dtype = trt.nptype(engine.get_tensor_dtype(in_name))
        out_dtype = trt.nptype(engine.get_tensor_dtype(out_name))
        n_in = int(np.prod(self.in_shape))
        n_out = int(np.prod(self.out_shape))

        self._ctxs, self._streams = [], []
        self._h_in, self._h_out, self._d_in, self._d_out = [], [], [], []
        for _ in range(n_slots):
            tc = engine.create_execution_context()
            st = cuda.Stream()
            hi = cuda.pagelocked_empty(n_in, self.in_dtype)
            ho = cuda.pagelocked_empty(n_out, out_dtype)
            di = cuda.mem_alloc(n_in * np.dtype(self.in_dtype).itemsize)
            do = cuda.mem_alloc(n_out * np.dtype(out_dtype).itemsize)
            tc.set_tensor_address(in_name, int(di))
            tc.set_tensor_address(out_name, int(do))
            self._ctxs.append(tc); self._streams.append(st)
            self._h_in.append(hi); self._h_out.append(ho)
            self._d_in.append(di); self._d_out.append(do)
        self.provider = "cuda"
        self.detail = (f"TensorRT {Path(self.engine_path).name}, {n_slots} contexts, "
                       f"in {self.in_shape} {self.in_dtype.__name__}")

    def infer(self, slot: int, arr: np.ndarray):
        cuda = self._cuda
        np.copyto(self._h_in[slot], arr)
        cuda.memcpy_htod_async(self._d_in[slot], self._h_in[slot], self._streams[slot])
        t0 = time.perf_counter()
        self._ctxs[slot].execute_async_v3(stream_handle=self._streams[slot].handle)
        self._streams[slot].synchronize()
        gpu_ms = (time.perf_counter() - t0) * 1000
        cuda.memcpy_dtoh_async(self._h_out[slot], self._d_out[slot], self._streams[slot])
        self._streams[slot].synchronize()
        return self._h_out[slot].copy(), gpu_ms

    def close(self) -> None:
        if self._ctx is not None:
            try:
                self._ctx.pop()
            except Exception:
                pass


class OnnxRuntimeBackend(Backend):
    """yolov8s.onnx through ONNX Runtime, on the best provider this box offers.

    Stateless per call, so slots need no per-slot allocation -- ORT owns its own
    device memory. `provider` reports what was actually selected, not what was
    requested: ORT silently falls back to CPU when a requested EP is unavailable,
    and a CPU run mislabelled as GPU is the failure mode this reports around.
    """

    tier = TIER_ORT

    def __init__(self, onnx_path: Path):
        self.onnx_path = str(onnx_path)
        self.detail = f"ONNX Runtime {Path(onnx_path).name}"

    def open(self, n_slots: int) -> None:
        import onnxruntime as ort

        available = ort.get_available_providers()
        wanted = [p for p in ORT_PROVIDER_PREFERENCE if p in available]
        if not wanted:
            wanted = ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(self.onnx_path, sess_options=opts, providers=wanted)

        actual = self._sess.get_providers()
        self.provider = actual[0] if actual else "unknown"
        inp = self._sess.get_inputs()[0]
        out = self._sess.get_outputs()[0]
        self._in_name = inp.name
        # A dynamic axis comes back as a string; substitute 1 for batch.
        self.in_shape = tuple(d if isinstance(d, int) else 1 for d in inp.shape)
        self.out_shape = tuple(d if isinstance(d, int) else 1 for d in out.shape)
        self.in_dtype = np.float16 if "float16" in inp.type else np.float32
        self.detail = (f"ONNX Runtime {Path(self.onnx_path).name} on {self.provider}, "
                       f"in {self.in_shape} {np.dtype(self.in_dtype).name}")
        if self.provider == "CPUExecutionProvider":
            self.detail += " — CPU provider: expect far lower throughput"

    def infer(self, slot: int, arr: np.ndarray):
        batch = np.ascontiguousarray(arr.reshape(self.in_shape).astype(self.in_dtype))
        t0 = time.perf_counter()
        out = self._sess.run(None, {self._in_name: batch})[0]
        gpu_ms = (time.perf_counter() - t0) * 1000
        return np.asarray(out).ravel(), gpu_ms


class NullBackend(Backend):
    """Neither tier is usable. The demo still starts, and says why.

    Previously this state raised FileNotFoundError out of VisionEngine.__init__,
    so a missing engine took the whole demo down at launch rather than degrading.
    """

    def __init__(self, notes):
        self.notes = list(notes)
        self.detail = "no detection backend — " + ("; ".join(self.notes) or "unknown reason")

    def infer(self, slot: int, arr: np.ndarray):
        raise RuntimeError(self.detail)


def select_backend(prefer: str | None = None):
    """Pick the best usable backend. Returns (backend, notes).

    `prefer` ("tensorrt" / "onnxruntime") forces a tier when it is usable --
    VISION_BACKEND in the environment -- which is how the ORT path gets
    exercised on a machine that also has a working TensorRT engine.
    """
    notes: list[str] = []

    engine = _first_existing(ENGINE_PATHS)
    onnx = _first_existing(ONNX_PATHS)

    def try_trt():
        if engine is None:
            notes.append(f"no .trt engine in {[str(p) for p in ENGINE_PATHS]}")
            return None
        try:
            import pycuda.driver  # noqa: F401
            import tensorrt  # noqa: F401
        except Exception as exc:
            notes.append(f"TensorRT/pycuda unavailable ({exc.__class__.__name__}) — this is expected on AMD")
            return None
        return TensorRTBackend(engine)

    def try_ort():
        if onnx is None:
            notes.append(f"no .onnx model in {[str(p) for p in ONNX_PATHS]}")
            return None
        try:
            import onnxruntime  # noqa: F401
        except Exception as exc:
            notes.append(f"onnxruntime not installed ({exc.__class__.__name__})")
            return None
        return OnnxRuntimeBackend(onnx)

    order = [try_ort, try_trt] if prefer == TIER_ORT else [try_trt, try_ort]
    if prefer == TIER_TRT:
        order = [try_trt]

    for factory in order:
        backend = factory()
        if backend is not None:
            return backend, notes
    return NullBackend(notes), notes
