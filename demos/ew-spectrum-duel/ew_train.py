"""
Spectrum Duel — install-time model build.

Trains the 5-class jam-technique CNN and builds a TensorRT FP16 engine from it.

Why it is written this way: the zgx env has numpy + tensorrt + pycuda but NO
torch, NO tensorflow and NO onnx, and *.onnx/*.engine/*.trt are gitignored so
nothing can be shipped pre-built. So we train the net in pure numpy (im2col
convolutions and hand-written backprop) and then hand the weights straight to
the TensorRT network-definition API — no ONNX round-trip, no framework.

Cost on a GB10 Nano: ~25-30 s total. (The YOLOv8 engine, by comparison, takes
15-20 minutes.) Idempotent — if the engine already deserializes under the
current TensorRT, it exits immediately.

Run:  $HOME/miniforge3/envs/zgx/bin/python3 ew_train.py [--force]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from ew_engine import CLASSES, NCLASS, TILE_BINS, TILE_CH, TILE_ROWS, make_dataset

HERE = Path(__file__).resolve().parent
# Artifacts live in models/ because that directory is gitignored by BOTH
# demos/.gitignore ("models/") and the repo root ("**/models/"). Note *.npz is
# NOT ignored on its own — dropping the weights in the demo root would commit
# them. Verify with: git check-ignore -v demos/ew-spectrum-duel/models/*
MODEL_DIR = HERE / "models"
ENGINE_PATH = MODEL_DIR / "jammer_cls.trt"
WEIGHTS_PATH = MODEL_DIR / "jammer_cls.npz"

# Bump this whenever ew_engine.py's physics or make_tile()'s data distribution
# changes. Without it, "the engine still deserializes" was the only skip test,
# so a box that had already installed once kept serving a model trained on the
# OLD physics forever — silently, and exactly on the machines that had been
# demoed the most. The stamp is written into the weights and re-checked here.
DATA_VERSION = 5

# Architecture — deliberately tiny. It has to separate five visually distinct
# time-frequency signatures, not do ImageNet.
C1, C2, K, PAD = 8, 16, 3, 1
# Pooling is (time, frequency) and deliberately NOT square.
#
# With 4x4 twice, the 64-row time axis collapsed to 4 cells — one cell per 0.8 s
# — and max-pooling within a cell means one bright old row hides sixteen quiet
# recent ones. The network then physically could not represent "the tone stopped
# three rows ago", which is exactly the judgement a straddling tile demands. It
# scored 43% on those, and adding them to the training set only traded that for
# accuracy on the easy tiles because the capacity to tell them apart was not
# there.
#
# Pooling time by 2 then 4 (not 4 then 4) keeps 8 time cells — 0.4 s each — for
# the cost of doubling one small matmul. Frequency stays at 4x4: after the 4:1
# decimation a 3-bin spot tone is already sub-cell, and it is the conv features,
# not the pooling grid, that separate it from a DRFM replica.
POOL1 = (2, 4)      # (time, freq)
POOL2 = (4, 4)
FEAT = (C2
        * (TILE_ROWS // POOL1[0] // POOL2[0])
        * (TILE_BINS // POOL1[1] // POOL2[1]))          # 16 * 8 * 4 = 512

N_TRAIN, N_TEST, EPOCHS, BATCH, LR = 4500, 900, 26, 128, 3e-3


# ── numpy conv/pool primitives ─────────────────────────────────────────────
def _im2col(x, k):
    n, c, h, w = x.shape
    oh, ow = h - k + 1, w - k + 1
    s = x.strides
    v = np.lib.stride_tricks.as_strided(
        x, (n, c, oh, ow, k, k), (s[0], s[1], s[2], s[3], s[2], s[3])
    )
    return v.reshape(n, c, oh, ow, k * k), oh, ow


def conv_fwd(x, W, b):
    xp = np.pad(x, ((0, 0), (0, 0), (PAD, PAD), (PAD, PAD)))
    n, f = x.shape[0], W.shape[0]
    col, oh, ow = _im2col(xp, K)
    colf = col.transpose(0, 2, 3, 1, 4).reshape(n * oh * ow, -1)
    out = colf @ W.reshape(f, -1).T + b
    return out.reshape(n, oh, ow, f).transpose(0, 3, 1, 2), colf, oh, ow


def conv_bwd(dz, colf, W, xshape, oh, ow):
    n, cin = xshape[0], xshape[1]
    f = W.shape[0]
    dzc = dz.transpose(0, 2, 3, 1).reshape(-1, f)
    gW = (dzc.T @ colf).reshape(W.shape)
    gb = dzc.sum(0)
    dcol = (dzc @ W.reshape(f, -1)).reshape(n, oh, ow, cin, K * K).transpose(0, 3, 1, 2, 4)
    dxp = np.zeros((n, cin, xshape[2] + 2 * PAD, xshape[3] + 2 * PAD), dz.dtype)
    for a in range(K):
        for bq in range(K):
            dxp[:, :, a:a + oh, bq:bq + ow] += dcol[:, :, :, :, a * K + bq]
    return gW, gb, dxp[:, :, PAD:PAD + xshape[2], PAD:PAD + xshape[3]]


def pool_fwd(x, k):
    """Max-pool by k = (kh, kw). Non-square on purpose — see POOL1 above."""
    kh, kw = k
    n, c, h, w = x.shape
    xr = x.reshape(n, c, h // kh, kh, w // kw, kw).transpose(0, 1, 2, 4, 3, 5)
    xr = xr.reshape(n, c, h // kh, w // kw, kh * kw)
    return xr.max(-1), xr.argmax(-1)


def pool_bwd(d, idx, k, shape):
    kh, kw = k
    n, c, oh, ow = d.shape
    z = np.zeros((n, c, oh, ow, kh * kw), d.dtype)
    np.put_along_axis(z, idx[..., None], d[..., None], -1)
    return (z.reshape(n, c, oh, ow, kh, kw).transpose(0, 1, 2, 4, 3, 5)
             .reshape(shape))


def softmax_ce(logits, y):
    e = np.exp(logits - logits.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-9).mean()
    d = p.copy()
    d[np.arange(len(y)), y] -= 1.0
    return loss, d / len(y), (p.argmax(1) == y).mean()


def train():
    rng = np.random.default_rng(20260818)
    t0 = time.time()
    print("  generating synthetic RF training set …", flush=True)
    Xtr, ytr, _ = make_dataset(N_TRAIN, rng)
    Xte, yte, tte = make_dataset(N_TEST, rng)
    print(f"  {N_TRAIN} train / {N_TEST} test tiles in {time.time()-t0:.1f}s", flush=True)

    W1 = (rng.standard_normal((C1, TILE_CH, K, K))
          * np.sqrt(2 / (9 * TILE_CH))).astype(np.float32)
    b1 = np.zeros(C1, np.float32)
    W2 = (rng.standard_normal((C2, C1, K, K)) * np.sqrt(2 / (9 * C1))).astype(np.float32)
    b2 = np.zeros(C2, np.float32)
    W3 = (rng.standard_normal((FEAT, NCLASS)) * np.sqrt(2 / FEAT)).astype(np.float32)
    b3 = np.zeros(NCLASS, np.float32)
    params = [W1, b1, W2, b2, W3, b3]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]

    def forward(x):
        z1, c1, oh1, ow1 = conv_fwd(x, W1, b1)
        a1 = np.maximum(z1, 0)
        p1, i1 = pool_fwd(a1, POOL1)
        z2, c2, oh2, ow2 = conv_fwd(p1, W2, b2)
        a2 = np.maximum(z2, 0)
        p2, i2 = pool_fwd(a2, POOL2)
        fl = p2.reshape(x.shape[0], -1)
        return fl @ W3 + b3, (x, c1, oh1, ow1, z1, a1, p1, i1, c2, oh2, ow2, z2, a2, p2, i2, fl)

    step = 0
    t0 = time.time()
    for ep in range(EPOCHS):
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], ytr[idx]
            logits, cache = forward(xb)
            _, dlogits, _ = softmax_ce(logits, yb)
            (x, c1, oh1, ow1, z1, a1, p1, i1, c2, oh2, ow2, z2, a2, p2, i2, fl) = cache
            gW3 = fl.T @ dlogits
            gb3 = dlogits.sum(0)
            dfl = (dlogits @ W3.T).reshape(p2.shape)
            dz2 = pool_bwd(dfl, i2, POOL2, a2.shape) * (z2 > 0)
            gW2, gb2, dp1 = conv_bwd(dz2, c2, W2, p1.shape, oh2, ow2)
            dz1 = pool_bwd(dp1, i1, POOL1, a1.shape) * (z1 > 0)
            gW1, gb1, _ = conv_bwd(dz1, c1, W1, x.shape, oh1, ow1)
            step += 1
            for j, (p_, g) in enumerate(zip(params, [gW1, gb1, gW2, gb2, gW3, gb3])):
                m[j] = 0.9 * m[j] + 0.1 * g
                v[j] = 0.999 * v[j] + 0.001 * g * g
                p_ -= LR * (m[j] / (1 - 0.9 ** step)) / (np.sqrt(v[j] / (1 - 0.999 ** step)) + 1e-8)
        if (ep + 1) % 2 == 0 or ep == EPOCHS - 1:
            lg, _ = forward(Xte)
            _, _, acc = softmax_ce(lg, yte)
            print(f"    epoch {ep+1:2d}/{EPOCHS}  test acc {acc*100:.1f}%", flush=True)

    logits, _ = forward(Xte)
    pred = logits.argmax(1)
    acc = float((pred == yte).mean())
    print(f"  trained in {time.time()-t0:.1f}s — final test accuracy {acc*100:.1f}%", flush=True)
    # Report the two regimes separately. One blended number hides the only part
    # that was ever hard: the tile that straddles a technique change.
    for name, sel in (("settled tiles   ", ~tte), ("straddling tiles", tte)):
        if sel.any():
            print(f"    {name}: {100*(pred[sel] == yte[sel]).mean():5.1f}%  "
                  f"(n={int(sel.sum())})", flush=True)
    print("  confusion (rows = truth, cols = predicted):", flush=True)
    print("           " + "".join(f"{c[:7]:>8}" for c in CLASSES), flush=True)
    for c in range(NCLASS):
        row = [int(((yte == c) & (pred == k)).sum()) for k in range(NCLASS)]
        sel = yte == c
        rec = 100 * (pred[sel] == c).mean() if sel.any() else 0.0
        print(f"  {CLASSES[c]:>8} " + "".join(f"{v:8d}" for v in row)
              + f"   recall {rec:5.1f}%", flush=True)

    MODEL_DIR.mkdir(exist_ok=True)
    settled_acc = float((pred[~tte] == yte[~tte]).mean()) if (~tte).any() else acc
    np.savez(WEIGHTS_PATH, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3,
             classes=np.array(CLASSES), accuracy=np.float32(acc),
             settled_accuracy=np.float32(settled_acc),
             data_version=np.int32(DATA_VERSION))
    print(f"  weights → models/{WEIGHTS_PATH.name}", flush=True)
    return acc


def build_engine():
    """Hand the numpy weights straight to TensorRT. No ONNX, no framework."""
    import tensorrt as trt

    d = np.load(WEIGHTS_PATH, allow_pickle=False)
    W1, b1, W2, b2 = d["W1"], d["b1"], d["W2"], d["b2"]
    W3, b3 = d["W3"], d["b3"]

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    net = builder.create_network(0)
    cfg = builder.create_builder_config()
    # FP16 only. INT8 is broken on SM 12.1 — calibration scales are silently
    # discarded and the engine quietly falls back (see CLAUDE.md).
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)

    x = net.add_input("spec", trt.float32, (1, TILE_CH, TILE_ROWS, TILE_BINS))

    c1 = net.add_convolution_nd(x, C1, (K, K), trt.Weights(np.ascontiguousarray(W1)),
                                trt.Weights(np.ascontiguousarray(b1)))
    c1.padding_nd = (PAD, PAD)
    r1 = net.add_activation(c1.get_output(0), trt.ActivationType.RELU)
    p1 = net.add_pooling_nd(r1.get_output(0), trt.PoolingType.MAX, POOL1)
    p1.stride_nd = POOL1

    c2 = net.add_convolution_nd(p1.get_output(0), C2, (K, K),
                                trt.Weights(np.ascontiguousarray(W2)),
                                trt.Weights(np.ascontiguousarray(b2)))
    c2.padding_nd = (PAD, PAD)
    r2 = net.add_activation(c2.get_output(0), trt.ActivationType.RELU)
    p2 = net.add_pooling_nd(r2.get_output(0), trt.PoolingType.MAX, POOL2)
    p2.stride_nd = POOL2

    flat = net.add_shuffle(p2.get_output(0))
    flat.reshape_dims = (1, FEAT)

    # TensorRT 10 removed add_fully_connected — the dense layer is a matmul
    # plus a broadcast add.
    wc = net.add_constant((FEAT, NCLASS), trt.Weights(np.ascontiguousarray(W3)))
    mm = net.add_matrix_multiply(flat.get_output(0), trt.MatrixOperation.NONE,
                                 wc.get_output(0), trt.MatrixOperation.NONE)
    bc = net.add_constant((1, NCLASS), trt.Weights(np.ascontiguousarray(b3.reshape(1, NCLASS))))
    add = net.add_elementwise(mm.get_output(0), bc.get_output(0), trt.ElementWiseOperation.SUM)
    sm = net.add_softmax(add.get_output(0))
    sm.axes = 1 << 1
    sm.get_output(0).name = "probs"
    net.mark_output(sm.get_output(0))

    t0 = time.time()
    plan = builder.build_serialized_network(net, cfg)
    if plan is None:
        raise RuntimeError("TensorRT returned no engine")
    blob = bytes(plan)
    MODEL_DIR.mkdir(exist_ok=True)
    ENGINE_PATH.write_bytes(blob)
    print(f"  engine built in {time.time()-t0:.1f}s → models/{ENGINE_PATH.name} "
          f"({len(blob)/1024:.0f} KB, TRT {trt.__version__})", flush=True)

    rt = trt.Runtime(logger)
    if rt.deserialize_cuda_engine(blob) is None:
        raise RuntimeError("engine failed to deserialize after build")
    print("  deserialize check OK", flush=True)


def engine_is_current() -> bool:
    """True if an engine exists and loads under the CURRENT TensorRT.

    Engines are not portable across TRT builds, so 'the file exists' is not
    good enough — mirror install.py's phase_build_engine and actually load it.
    """
    if not ENGINE_PATH.exists() or not WEIGHTS_PATH.exists():
        return False
    try:
        d = np.load(WEIGHTS_PATH, allow_pickle=False)
        if int(d.get("data_version", np.int32(0))) != DATA_VERSION:
            print(f"  weights predate the current RF model "
                  f"(v{int(d.get('data_version', np.int32(0)))} < v{DATA_VERSION})"
                  f" — retraining.", flush=True)
            return False
    except Exception:
        return False
    try:
        import tensorrt as trt
        rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        return rt.deserialize_cuda_engine(ENGINE_PATH.read_bytes()) is not None
    except Exception:
        return False


def main():
    force = "--force" in sys.argv
    print("Spectrum Duel — jam-classifier build", flush=True)
    if not force and engine_is_current():
        print("  engine already present and loads under this TensorRT — skipping.", flush=True)
        return 0
    acc = train()
    try:
        build_engine()
    except Exception as e:
        # Not fatal: ew_infer.py falls back to the numpy forward pass using the
        # weights we just saved, so the demo still runs (slower, CPU).
        print(f"  WARNING: TensorRT build failed ({e}).", flush=True)
        print("  Demo will run on the numpy CPU fallback.", flush=True)
        return 0
    print(f"Done — jam classifier ready ({acc*100:.1f}% test accuracy).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
