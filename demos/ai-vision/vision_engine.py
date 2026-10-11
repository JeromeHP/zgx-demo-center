"""TRT inference engine for AI Vision Demo.

Ports the YOLOv8s FP16 pipeline from bench_inference.py.
Key differences vs the Performance Lab:
  - 6 active streams per scenario (not 12), but 12 TRT contexts pre-allocated
  - Dynamic scenario switching: stop, swap VideoReaders, restart
  - Alert rule engine: per-stream sliding-window class counts trigger callbacks
  - No benchmark/throttle/measurement code — this is a demo, not a benchmark
"""
import os
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty

import cv2
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
INPUT_W = INPUT_H = 1088
DISPLAY_W, DISPLAY_H = 640, 360
JPEG_QUALITY = 75
CONF_THRESH = 0.25
IOU_THRESH = 0.45
MAX_STREAMS = 12
ACTIVE_STREAMS = 6
PREPROC_QUEUE_DEPTH = 16

from vision_backends import ENGINE_PATHS, TIER_NONE, select_backend  # noqa: E402

COCO = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]

STREAM_COLORS = ["#3b82f6","#22d3ee","#34d399","#fbbf24","#f472b6","#a78bfa"]

# ── Scenarios ────────────────────────────────────────────────────────────────
SCENARIOS = {
    "traffic": {
        "label": "Traffic Monitoring",
        "description": "Urban intersection and highway surveillance",
        "icon": "🚗",
        "videos_dir": str(Path.home() / "zgx_videos"),
        "videos": ["video1.mp4","video2.mp4","video3.mp4","video4.mp4","video9.mp4","video6.mp4"],
        "stream_count": 6,
        "grid": "2x3",
        "relevant_classes": [0, 1, 2, 3, 5, 7],
        "alert_rules": {
            "high_vehicle_count": {"threshold": 10, "classes": [2,3,5,7], "message": "Heavy traffic detected"},
            "pedestrian_in_road":  {"threshold": 1,  "classes": [0],       "message": "Pedestrian detected in traffic zone"},
        },
    },
    "warehouse": {
        "label": "Warehouse Safety",
        "description": "Industrial floor monitoring and forklift safety",
        "icon": "🏭",
        "videos_dir": str(Path.home() / "zgx_videos" / "warehouse"),
        "videos": ["warehouse_1.mp4","warehouse_2.mp4","warehouse_3.mp4","warehouse_4.mp4","warehouse_5.mp4","warehouse_6.mp4"],
        "stream_count": 6,
        "grid": "2x3",
        "relevant_classes": [0, 7, 24],
        "alert_rules": {
            "forklift_pedestrian": {"threshold": 1, "classes": [0,7], "message": "Person near forklift zone", "require_all_classes": True},
            "worker_count":        {"threshold": 5, "classes": [0],   "message": "High worker density detected"},
        },
    },
    "outdoor": {
        "label": "Outdoor Security",
        "description": "Urban surveillance and pedestrian monitoring",
        "icon": "🏙️",
        "videos_dir": str(Path.home() / "zgx_videos" / "outdoor"),
        "videos": ["outdoor_1.mp4","outdoor_2.mp4","outdoor_3.mp4","outdoor_4.mp4","outdoor_5.mp4","outdoor_6.mp4"],
        "stream_count": 6,
        "grid": "2x3",
        "relevant_classes": [0, 1, 2, 24, 26],
        "alert_rules": {
            "crowd_density":              {"threshold": 8, "classes": [0], "message": "Crowd density alert"},
            "vehicle_in_pedestrian_zone": {"threshold": 1, "classes": [2], "message": "Vehicle in pedestrian zone"},
        },
    },
    "retail": {
        "label": "Retail Analytics",
        "description": "In-store foot traffic and customer analytics",
        "icon": "🛒",
        "videos_dir": str(Path.home() / "zgx_videos" / "retail"),
        "videos": ["retail_1.mp4","retail_2.mp4","retail_3.mp4","retail_4.mp4","retail_5.mp4","retail_6.mp4"],
        "stream_count": 6,
        "grid": "2x3",
        "relevant_classes": [0, 24, 26, 67],
        "alert_rules": {
            "foot_traffic":     {"threshold": 6,  "classes": [0], "message": "High foot traffic"},
            "crowd_gathering":  {"threshold": 10, "classes": [0], "message": "Crowd gathering detected"},
        },
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def hex_bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def postprocess(h_out, out_shape, orig_h, orig_w):
    out = h_out.reshape(out_shape[1:]).T
    scores = out[:, 4:]
    sc = scores.max(axis=1)
    ci = scores.argmax(axis=1).astype(np.int32)
    mask = sc > CONF_THRESH
    if not mask.any():
        return []
    bx = out[mask, :4]; sc = sc[mask]; ci = ci[mask]
    sx, sy = orig_w / INPUT_W, orig_h / INPUT_H
    x1 = (bx[:,0] - bx[:,2]/2) * sx; y1 = (bx[:,1] - bx[:,3]/2) * sy
    x2 = (bx[:,0] + bx[:,2]/2) * sx; y2 = (bx[:,1] + bx[:,3]/2) * sy
    rects = np.stack([x1, y1, x2-x1, y2-y1], axis=1).tolist()
    idxs = cv2.dnn.NMSBoxes(rects, sc.tolist(), CONF_THRESH, IOU_THRESH)
    if len(idxs) == 0:
        return []
    idxs = idxs.flatten()
    return [{"x1": int(x1[k]), "y1": int(y1[k]), "x2": int(x2[k]), "y2": int(y2[k]),
             "score": float(sc[k]), "cls": int(ci[k])} for k in idxs]


def draw_boxes(frame, dets, color_bgr):
    for d in dets:
        cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), color_bgr, 1)
        lbl = f"{COCO[d['cls']]} {d['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        yy = max(d["y1"], th + 4)
        cv2.rectangle(frame, (d["x1"], yy - th - 3), (d["x1"] + tw + 2, yy), color_bgr, -1)
        cv2.putText(frame, lbl, (d["x1"] + 1, yy - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)


def _preproc(raw, in_dtype):
    orig_h, orig_w = raw.shape[:2]
    img = cv2.resize(raw, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(in_dtype) / in_dtype(255)
    return img.transpose(2, 0, 1).ravel(), orig_h, orig_w


# ── StreamState ──────────────────────────────────────────────────────────────
class StreamState:
    def __init__(self, sid, label, color, video_name):
        self.sid = sid
        self.label = label
        self.color = color
        self.video_name = video_name
        self.lock = threading.Lock()
        self.jpeg = None
        self.dets = []
        self.frame_count = 0
        self.lat_hist = deque(maxlen=200)
        self._fps_times = deque(maxlen=30)
        self._seq = 0
        # rolling class counts for alert detection (last 30 frames per class)
        self.class_history = defaultdict(lambda: deque(maxlen=30))

    def update(self, jpeg, dets, lat_ms):
        now = time.time()
        with self.lock:
            self.jpeg = jpeg
            self.dets = dets
            self._seq += 1
            self._fps_times.append(now)
            self.frame_count += 1
            self.lat_hist.append(lat_ms)
            # update class counts
            counts = defaultdict(int)
            for d in dets:
                counts[d["cls"]] += 1
            # push counts into history (one entry per frame, defaults to 0 for absent classes)
            seen = set(counts.keys())
            for cls in list(self.class_history.keys()) + list(seen):
                self.class_history[cls].append(counts.get(cls, 0))

    def fps(self):
        with self.lock:
            ts = list(self._fps_times)
        if len(ts) < 2:
            return 0.0
        span = ts[-1] - ts[0]
        return round((len(ts) - 1) / span, 1) if span > 0 else 0.0

    def latency_ms(self):
        with self.lock:
            h = list(self.lat_hist)
        return round(float(np.mean(h)), 1) if h else 0.0

    def get_jpeg(self):
        with self.lock:
            return self.jpeg

    def class_counts_max(self, classes, window=15):
        """Return max count over the last `window` frames for any class in `classes` (sum across classes)."""
        with self.lock:
            counts_per_frame = []
            # collect last `window` frames per class
            per_cls = {}
            for cls in classes:
                if cls in self.class_history:
                    per_cls[cls] = list(self.class_history[cls])[-window:]
                else:
                    per_cls[cls] = []
            n = max((len(v) for v in per_cls.values()), default=0)
            for i in range(n):
                total = 0
                for cls in classes:
                    arr = per_cls[cls]
                    if i < len(arr):
                        total += arr[i]
                counts_per_frame.append(total)
        return max(counts_per_frame) if counts_per_frame else 0

    def class_counts_present(self, classes, window=15, min_frames=10):
        """Return True if EVERY class in `classes` was present (count>=1) in at least
        `min_frames` of the last `window` frames. Used for the forklift-near-pedestrian rule."""
        with self.lock:
            for cls in classes:
                hist = list(self.class_history.get(cls, []))[-window:]
                if sum(1 for c in hist if c > 0) < min_frames:
                    return False
        return True


# ── VideoReader ──────────────────────────────────────────────────────────────
class VideoReader(threading.Thread):
    def __init__(self, path, sid):
        super().__init__(daemon=True, name=f"vread-{sid}")
        self._path = Path(path)
        self._sid = sid
        self._slot = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cap = None  # active VideoCapture, exposed so teardown can force-release

    def stop(self):
        self._stop.set()

    def release_cap(self):
        """Force-release the active VideoCapture so a blocking read() returns.
        Safe to call from any thread; the run loop tolerates a released cap."""
        cap = self._cap
        self._cap = None
        if cap is not None:
            try: cap.release()
            except Exception: pass

    def get_with_id(self):
        with self._lock:
            if self._slot is None:
                return None, -1
            return self._slot.copy(), self._frame_id

    def _open(self):
        try:
            cap = cv2.VideoCapture(str(self._path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                print(f"  [vread-{self._sid}] {self._path.name} {int(cap.get(3))}x{int(cap.get(4))} {fps:.1f}fps", flush=True)
                return cap, 1.0 / fps
            cap.release()
        except Exception as e:
            print(f"  [vread-{self._sid}] open error: {e}", flush=True)
        return None, 0.033

    def run(self):
        while not self._stop.is_set():
            if not self._path.exists():
                self._render_synthetic()
                time.sleep(0.5)
                continue
            cap, frame_delay = self._open()
            if cap is None:
                self._render_synthetic()
                time.sleep(1.0)
                continue
            self._cap = cap
            while not self._stop.is_set():
                try:
                    ok, frame = cap.read()
                except Exception:
                    break
                if not ok:
                    break
                with self._lock:
                    self._slot = frame
                    self._frame_id += 1
                time.sleep(frame_delay)
            self._cap = None
            try: cap.release()
            except Exception: pass

    def _render_synthetic(self):
        f = np.zeros((480, 854, 3), np.uint8)
        f[:] = [20, 20, 35]
        cv2.putText(f, f"NO VIDEO  S{self._sid+1:02d}", (220, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 60, 100), 2)
        with self._lock:
            self._slot = f
            self._frame_id += 1


# ── PreProcWorker ────────────────────────────────────────────────────────────
class PreProcWorker(threading.Thread):
    def __init__(self, reader, sid, in_dtype, engine_ref):
        super().__init__(daemon=True, name=f"preproc-{sid}")
        self.reader = reader
        self.sid = sid
        self.in_dtype = in_dtype
        self._engine = engine_ref
        self.out_q = Queue(maxsize=PREPROC_QUEUE_DEPTH)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        last_fid = -1
        while not self._stop.is_set():
            if not self._engine.inference_active:
                time.sleep(0.05)
                continue
            raw, fid = self.reader.get_with_id()
            if raw is None or fid == last_fid:
                time.sleep(0.001)
                continue
            last_fid = fid
            try:
                arr, orig_h, orig_w = _preproc(raw, self.in_dtype)
                self.out_q.put((arr, orig_h, orig_w, raw), timeout=0.5)
            except Exception:
                continue


# ── InferenceWorker ──────────────────────────────────────────────────────────
class InferenceWorker(threading.Thread):
    """Runs detection through whichever backend this box supports.

    The TensorRT/pycuda calls used to be inline here. They now live behind
    vision_backends.Backend so the same loop drives TensorRT on NVIDIA and ONNX
    Runtime on AMD -- see vision_backends.py for why both tiers run identical
    weights (the .trt engine is built from the same ~/yolov8s.onnx).
    """

    def __init__(self, backend, engine_ref):
        super().__init__(daemon=True, name="gpu-infer")
        self.backend = backend
        self._engine = engine_ref

    def run(self):
        try:
            self.backend.open(MAX_STREAMS)
        except Exception as exc:
            # A backend that cannot open must not take the process down: the
            # demo keeps serving its UI and reports the degraded state.
            print(f"  [GPU] backend {self.backend.tier} failed to open: {exc}", flush=True)
            self._engine.backend_status = f"{self.backend.tier} failed: {exc}"
            return
        try:
            self._loop()
        finally:
            self.backend.close()

    def _loop(self):
        backend = self.backend
        out_shape = backend.out_shape

        # tell the engine the dtype so it can spin up preprocs whenever scenarios swap
        self._engine.in_dtype = backend.in_dtype
        self._engine.backend_status = backend.detail
        self._engine.backend_tier = backend.tier
        self._engine.backend_provider = backend.provider

        print(f"  [GPU] detection backend: {backend.detail}", flush=True)

        postproc_pool = ThreadPoolExecutor(max_workers=12, thread_name_prefix="postproc")

        def _post_task(state, raw, hoc, oh, ow, gms):
            try:
                dets = postprocess(hoc, out_shape, oh, ow)
                disp = np.ascontiguousarray(cv2.resize(raw, (DISPLAY_W, DISPLAY_H)))
                sx, sy = DISPLAY_W / ow, DISPLAY_H / oh
                dd = [{**d,
                       "x1": int(d["x1"] * sx), "y1": int(d["y1"] * sy),
                       "x2": int(d["x2"] * sx), "y2": int(d["y2"] * sy)} for d in dets]
                draw_boxes(disp, dd, hex_bgr(state.color))
                _, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                state.update(buf.tobytes(), dd, gms)
            except Exception as e:
                print(f"  [postproc] err: {e}", flush=True)

        while True:
            if not self._engine.inference_active:
                time.sleep(0.1)
                continue
            states = self._engine.get_states_snapshot()
            ppws = self._engine.get_preprocs_snapshot()
            if not states or not ppws or len(states) != len(ppws):
                time.sleep(0.05)
                continue
            for i, (state, ppw) in enumerate(zip(states, ppws)):
                if i >= MAX_STREAMS:
                    break
                try:
                    arr, orig_h, orig_w, raw = ppw.out_q.get(timeout=0.05)
                except Empty:
                    continue
                try:
                    h_out_copy, gpu_ms = backend.infer(i, arr)
                except Exception as e:
                    print(f"  [GPU] inference error on slot {i}: {e}", flush=True)
                    continue
                postproc_pool.submit(_post_task, state, raw, h_out_copy, orig_h, orig_w, gpu_ms)


# ── AlertEngine ──────────────────────────────────────────────────────────────
class AlertEngine(threading.Thread):
    """Polls each StreamState's class history and fires callbacks when alert
    rules are sustained over the alert window. Tracks per-(stream, rule) cooldown
    so the same alert doesn't repeat-fire each tick."""

    POLL_INTERVAL_S = 1.0
    COOLDOWN_S = 30.0   # don't refire same (stream, rule) within this window
    SUSTAIN_FRAMES = 12 # threshold must be sustained across this many recent frames

    def __init__(self, engine_ref, on_alert):
        super().__init__(daemon=True, name="alert-engine")
        self._engine = engine_ref
        self._on_alert = on_alert  # callable(stream_id, rule_name, message, class_counts)
        self._stop = threading.Event()
        self._last_fired = {}  # (sid, rule) -> ts
        self._lock = threading.Lock()

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            time.sleep(self.POLL_INTERVAL_S)
            if not self._engine.inference_active:
                continue
            scen = self._engine.current_scenario
            cfg = SCENARIOS.get(scen)
            if not cfg:
                continue
            states = self._engine.get_states_snapshot()
            now = time.time()
            for state in states:
                for rule_name, rule in cfg["alert_rules"].items():
                    key = (state.sid, scen, rule_name)
                    last = self._last_fired.get(key, 0)
                    if now - last < self.COOLDOWN_S:
                        continue
                    classes = rule["classes"]
                    threshold = rule["threshold"]
                    require_all = rule.get("require_all_classes", False)
                    fire = False
                    if require_all:
                        # all classes must be present in window AND aggregate >= threshold
                        if state.class_counts_present(classes, window=self.SUSTAIN_FRAMES, min_frames=int(self.SUSTAIN_FRAMES * 0.6)):
                            fire = True
                    else:
                        if state.class_counts_max(classes, window=self.SUSTAIN_FRAMES) >= threshold:
                            fire = True
                    if fire:
                        self._last_fired[key] = now
                        try:
                            counts = self._engine.snapshot_class_counts(state)
                            self._on_alert(state.sid, state.label, rule_name, rule["message"], counts)
                        except Exception as e:
                            print(f"  [alert] callback err: {e}", flush=True)


# ── Engine ───────────────────────────────────────────────────────────────────
class VisionEngine:
    def __init__(self, on_alert=None):
        # Pick a detection backend instead of demanding a TensorRT engine.
        #
        # This used to raise FileNotFoundError when yolov8s.trt was missing,
        # taking the whole demo down at launch -- on an AMD box, every time,
        # since TensorRT cannot be installed there at all. Spectrum Duel's
        # ew_infer.py docstring calls this out by name as the thing not to do.
        # Now the backend tiers (TensorRT -> ONNX Runtime -> none) and the UI
        # reports which one is live.
        self.backend, self.backend_notes = select_backend(os.environ.get("VISION_BACKEND") or None)
        self.engine_path = getattr(self.backend, "engine_path", None)
        self.backend_tier = self.backend.tier
        self.backend_provider = self.backend.provider
        self.backend_status = self.backend.detail
        for note in self.backend_notes:
            print(f"  [GPU] {note}", flush=True)
        if self.backend.tier == TIER_NONE:
            print("  [GPU] no detection backend — video will play, detection is off", flush=True)
        self.in_dtype = np.float16  # placeholder; updated by InferenceWorker after engine load
        self._lock = threading.Lock()
        self._switching = threading.Event()  # set during start_scenario; consumers pause frame delivery
        self._switch_started_at = 0.0        # monotonic-ish start of the in-flight switch (0 = none)
        self._switch_stalled = False         # watchdog tripped: a switch overran and we force-resumed
        self.current_scenario = None
        self.streams = []      # active StreamState list
        self.readers = []      # active VideoReader list
        self.preprocs = []     # active PreProcWorker list
        self.inference_active = False
        self.cloud_calls = 0   # ALWAYS 0 — proof point
        self._on_alert = on_alert
        # Start the GPU loop now; it idles until inference_active=True
        self._gpu = InferenceWorker(self.backend, self)
        self._gpu.start()
        # Alert engine
        self._alert_engine = AlertEngine(self, self._handle_alert)
        self._alert_engine.start()
        # Switch watchdog — guarantees frame delivery can never stay paused forever.
        # If a scenario switch overruns (e.g. a cv2 VideoCapture.release() deadlock in
        # teardown), _switching would otherwise stay SET and the browser would show a
        # permanently frozen feed. The watchdog force-resumes delivery and flags the
        # stall so the frontend can drop its loading overlay.
        self._watchdog = threading.Thread(target=self._switch_watchdog, daemon=True, name="switch-watchdog")
        self._watchdog.start()


    def _handle_alert(self, sid, label, rule_name, message, counts):
        if self._on_alert:
            self._on_alert(sid, label, rule_name, message, counts)

    def is_switching(self):
        return self._switching.is_set()

    # ── Snapshots used by GPU/alert threads ──────────────────────────────────
    def get_states_snapshot(self):
        with self._lock:
            return list(self.streams)

    def get_preprocs_snapshot(self):
        with self._lock:
            return list(self.preprocs)

    def snapshot_class_counts(self, state):
        with state.lock:
            cls_to_recent_max = {}
            for cls, hist in state.class_history.items():
                arr = list(hist)[-15:]
                if arr:
                    cls_to_recent_max[cls] = max(arr)
        return {COCO[c]: v for c, v in cls_to_recent_max.items() if v > 0}

    # ── Scenario lifecycle ───────────────────────────────────────────────────
    # A scenario switch is a clean, frame-paused, serialized transition:
    #   pause frames → spin DOWN old (confirm stopped) → settle → spin UP new →
    #   wait for first real frame → resume frames.
    # _switching stays SET for the whole window, so /ws/frames delivers nothing to
    # the browser during the contended moment — chrome does zero GPU render work
    # while the backend thrashes, which is the direct guard against the GPU wedge.
    SETTLE_SECONDS = 0.3        # drain GPU / flush queues after spin-down, before spin-up
    FIRST_FRAME_TIMEOUT = 3.0   # cap on holding frames paused waiting for the first new frame
    SWITCH_WATCHDOG_S = 6.0     # if a switch holds _switching longer than this, force-resume frames
    CAP_RELEASE_TIMEOUT = 2.0   # total budget for releasing all old VideoCaptures during teardown

    def _switch_watchdog(self):
        """Backstop thread: a scenario switch must never leave frame delivery paused
        forever. If _switching has been set longer than SWITCH_WATCHDOG_S, force-clear
        it (resume delivery — degraded if the new streams aren't up yet) and flag the
        stall so the telemetry loop can tell the frontend to drop its overlay. The
        hung switch's own `finally` will clear _switching again later (a harmless
        no-op). Paired with the bounded teardown below, which removes the usual cause."""
        while True:
            time.sleep(0.5)
            started = self._switch_started_at
            if self._switching.is_set() and started and not self._switch_stalled:
                if (time.time() - started) > self.SWITCH_WATCHDOG_S:
                    print(f"  [engine] WATCHDOG: switch stuck >{self.SWITCH_WATCHDOG_S}s "
                          f"— force-resuming frame delivery", flush=True)
                    self._switch_stalled = True
                    self._switching.clear()

    def start_scenario(self, scenario_id):
        if scenario_id not in SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario_id}")
        print(f"  [engine] scenario switch requested → {scenario_id}", flush=True)
        self._switch_stalled = False
        self._switch_started_at = time.time()   # arm the watchdog
        self._switching.set()   # PAUSE frame delivery for the entire transition
        try:
            # 1. Spin DOWN the old scenario fully and CONFIRM stopped (joins old
            #    readers/preprocs) BEFORE creating new ones — never run old+new at once.
            self._teardown_streams()
            # 2. Settle: brief pause so the GPU drains and the preproc/inference
            #    queues flush before we load the next scenario's streams.
            time.sleep(self.SETTLE_SECONDS)
            # 3. Spin UP the new scenario.
            cfg = SCENARIOS[scenario_id]
            videos_dir = Path(cfg["videos_dir"]).expanduser()
            with self._lock:
                self.current_scenario = scenario_id
                self.streams = []
                self.readers = []
                self.preprocs = []
                for i, vname in enumerate(cfg["videos"][:cfg["stream_count"]]):
                    vpath = videos_dir / vname
                    color = STREAM_COLORS[i % len(STREAM_COLORS)]
                    state = StreamState(i, f"S{i+1:02d}", color, vname)
                    reader = VideoReader(vpath, sid=i)
                    ppw = PreProcWorker(reader, i, self.in_dtype, self)
                    self.streams.append(state)
                    self.readers.append(reader)
                    self.preprocs.append(ppw)
                print(f"  [engine] created {len(self.readers)} new readers for {scenario_id}", flush=True)
                for r in self.readers:
                    r.start()
                for p in self.preprocs:
                    p.start()
                self.inference_active = True
            # 4. Hold the pause until the FIRST real frame of the new scenario is
            #    decoded+inferred, so delivery resumes only when there is genuine
            #    content to show — not during the still-hot spin-up window. Bounded,
            #    so a stuck decode can never wedge the switch open.
            self._wait_first_frame(self.FIRST_FRAME_TIMEOUT)
        finally:
            self._switch_started_at = 0.0     # disarm the watchdog
            self._switching.clear()           # RESUME frame delivery — first frame is ready
        print(f"  [engine] scenario switch complete: {scenario_id} ({cfg['stream_count']} streams)", flush=True)

    def _wait_first_frame(self, timeout):
        """Block (in the switch worker thread) until at least one new stream has
        produced a frame, or `timeout` elapses. The browser's overlay dismisses on
        this same first frame, so the wait makes the overlay honest: as fast as the
        switch truly is, never dismissed early onto a frozen/old frame."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            for s in self.get_states_snapshot():
                with s.lock:
                    if s.jpeg is not None:
                        return True
            time.sleep(0.02)
        print("  [engine] first-frame wait timed out — resuming delivery anyway", flush=True)
        return False

    def _teardown_streams(self):
        with self._lock:
            self.inference_active = False
            old_readers = self.readers
            old_preprocs = self.preprocs
            self.streams = []
            self.readers = []
            self.preprocs = []
        print(f"  [engine] tearing down {len(old_readers)} readers / {len(old_preprocs)} preprocs", flush=True)
        # 1. Signal stop on workers
        for r in old_readers:
            try: r.stop()
            except Exception: pass
        for p in old_preprocs:
            try: p.stop()
            except Exception: pass
        # 2. Force-release VideoCapture handles so any blocking cap.read() returns.
        # BOUNDED + PARALLEL: cv2 VideoCapture.release() can DEADLOCK when it races an
        # in-flight cap.read() on the reader thread — observed in the field, rapid
        # scenario switching hung the switch right here, and because _switching stays
        # set across the switch the browser feed froze permanently. Run each release in
        # its own daemon thread and abandon any that don't finish within
        # CAP_RELEASE_TIMEOUT (total) — a cap that releases late beats a wedged engine.
        # Reader threads are daemon and joined-with-timeout below regardless.
        rel_threads = []
        for r in old_readers:
            t = threading.Thread(target=r.release_cap, daemon=True, name=f"release-{r.name}")
            t.start()
            rel_threads.append((t, r))
        rel_deadline = time.time() + self.CAP_RELEASE_TIMEOUT
        for t, r in rel_threads:
            t.join(max(0.0, rel_deadline - time.time()))
            if t.is_alive():
                print(f"  [engine] {r.name} cap.release() stuck — abandoning (daemon)", flush=True)
        print(f"  [engine] stop signaled + cv2 captures released; joining threads", flush=True)
        # 3. Join with timeout — never block forever; abandoned threads stay daemon
        for r in old_readers:
            try:
                r.join(timeout=2.0)
                if r.is_alive():
                    print(f"  [engine] {r.name} did not join within 2s — abandoning (daemon)", flush=True)
            except Exception: pass
        for p in old_preprocs:
            try:
                p.join(timeout=2.0)
                if p.is_alive():
                    print(f"  [engine] {p.name} did not join within 2s — abandoning (daemon)", flush=True)
            except Exception: pass
        print(f"  [engine] teardown complete", flush=True)

    # ── Public read API ──────────────────────────────────────────────────────
    def get_jpeg(self, sid):
        with self._lock:
            if sid < 0 or sid >= len(self.streams):
                return None
            s = self.streams[sid]
        return s.get_jpeg()

    def aggregate_metrics(self):
        with self._lock:
            states = list(self.streams)
            scen = self.current_scenario
        per_stream = []
        all_lats = []
        agg_class_counts = defaultdict(int)
        for s in states:
            with s.lock:
                lats = list(s.lat_hist)
                # take last frame's counts per class
                cls_counts = {}
                for cls, hist in s.class_history.items():
                    if hist:
                        cls_counts[cls] = hist[-1]
            all_lats.extend(lats)
            per_stream.append({
                "id": s.sid,
                "label": s.label,
                "color": s.color,
                "video": s.video_name,
                "fps": s.fps(),
                "lat_ms": round(float(np.mean(lats)), 1) if lats else 0.0,
                "frames": s.frame_count,
            })
            for cls, n in cls_counts.items():
                agg_class_counts[cls] += n
        total_qps = round(sum(p["fps"] for p in per_stream), 1)
        return {
            "scenario": scen,
            "stream_count": len(states),
            "streams": per_stream,
            "total_qps": total_qps,
            "lat_p50_ms": round(float(np.percentile(all_lats, 50)), 1) if len(all_lats) >= 10 else 0,
            "lat_p99_ms": round(float(np.percentile(all_lats, 99)), 1) if len(all_lats) >= 10 else 0,
            "class_counts": {COCO[c]: n for c, n in agg_class_counts.items() if n > 0},
            "cloud_calls": self.cloud_calls,
        }
