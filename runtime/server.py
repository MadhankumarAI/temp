"""Dashboard backend. Streams telemetry and map frames over one WebSocket.

DESIGN CONSTRAINT: the dashboard must not cost the pipeline anything.

That is the whole reason for the shape of this file. Perception runs on its own
thread and writes its latest result into a single slot. The socket reads that
slot on its own clock. They never wait for each other, and the queue between
them has a depth of one, so a browser that stalls (a tab in the background, a
laptop that slept) drops frames instead of applying backpressure to a
perception loop that has a sensor waiting on it. Dropping a frame the user
cannot see is free; delaying a frame the vehicle needs is not.

Everything expensive is off the hot path: telemetry samples nvidia-smi on its
own thread, and the map is serialised only for frames that are actually sent.

The payload is deliberately small. A full 5 cm map is megabytes a frame and no
browser will draw it at 20 Hz, so what goes over the wire is the drivability
lattice at 40 cm plus the boxes, which is what a viewer can actually render and
what a person can actually read.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
# Module level, NOT inside build_app, and this is not a style choice.
# `from __future__ import annotations` makes every annotation a string, and
# FastAPI resolves those strings against the MODULE's globals when it works out
# what each handler parameter is. With these imported inside the factory,
# `sock: WebSocket` resolved to nothing, FastAPI decided it must be a query
# parameter, and every websocket handshake was rejected with 403 while the
# route sat correctly registered in app.routes. Nothing raised, on either side.
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline import Config, Perception          # noqa: E402
from telemetry import Telemetry                  # noqa: E402


class Engine:
    """Runs perception in a thread and keeps the latest result.

    One slot, not a queue. A queue would let a slow consumer build a backlog,
    and a backlog of perception frames is worse than useless: by the time you
    draw the tenth stale one, the vehicle is somewhere else. The freshest frame
    is the only one worth having.
    """

    # How far around the vehicle the dashboard is sent. Not a rendering
    # preference: it is what stops the payload growing without bound.
    VIEW_M = 80.0

    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        self.per = None
        self.tel = Telemetry().start()
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.source = None
        self.state = "idle"
        self.err = None
        self.hist = []                       # recent frame times, for the chart

    def ready(self):
        if self.per is None:
            self.state = "loading"
            self.per = Perception(self.cfg)
            self.state = "idle"
        return self.per

    # ------------------------------------------------------------- control
    def start(self, root: Path, loop: bool = True, stride: int = 1):
        self.stop()
        self.ready()
        self.per.reset_map()
        self.hist = []
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(Path(root), loop, stride), daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None
        self.state = "idle"

    # ------------------------------------------------------------- worker
    def _run(self, root, loop, stride):
        try:
            sys.path.insert(0, str(HERE.parent / "integration"))
            import _bootstrap                                # noqa: F401
            import carla as CA
            meta = CA.read_meta(root)
            ids = list(CA.frame_ids_of(root, meta))[::max(stride, 1)]
            if not ids:
                raise RuntimeError(f"no frames under {root}")
            self.source = {"root": str(root), "frames": len(ids)}
            self.state = "running"
            k = 0
            while not self._stop.is_set():
                i = ids[k % len(ids)]
                k += 1
                if k > len(ids) and not loop:
                    break
                t0 = time.perf_counter()
                f = CA.load(root, i, meta)
                t_read = (time.perf_counter() - t0) * 1000
                r = self.per.process(f.pts, f.T)
                # Read time is reported but kept OUT of the frame budget: a
                # sensor hands points over in memory, so time spent opening an
                # npz is an artefact of replaying a capture and counting it
                # would make the pipeline look slower than it will ship.
                pay = self._payload(r, f.T, t_read)
                with self._lock:
                    self._latest = pay
                self.hist.append(r.ms["total"])
                if len(self.hist) > 240:
                    self.hist = self.hist[-240:]
        except Exception as e:                    # noqa: BLE001
            self.err = f"{type(e).__name__}: {e}"
            self.state = "error"
        else:
            self.state = "idle"

    # ------------------------------------------------------------- payload
    def _payload(self, r, T, t_read):
        tc = self.per.tc
        c = r.cells
        res = float(self.per.map.res[self.cfg.drive_level])
        ix, iy = c["ix"], c["iy"]
        keep = r.drive_cls != tc.UNKNOWN

        # BOUND THE PAYLOAD. The map is world-anchored and grows for as long as
        # the vehicle drives, so sending all of it means a frame that starts at
        # 280 KB and has no ceiling: a demo that is fine for a minute and
        # unusable after ten. Only cells within VIEW_M of the vehicle go out,
        # which is a constant regardless of how long the run has been going,
        # and it is also all a viewer can resolve on screen.
        #
        # Indices go as int16 RELATIVE to the vehicle's cell rather than int32
        # absolute. Half the bytes, and int16 cannot overflow inside the window
        # because the window is 400 cells across and the type holds 65,536.
        cx, cy = int(round(T[0, 3] / res)), int(round(T[1, 3] / res))
        rad = int(self.VIEW_M / res)
        keep &= (np.abs(ix - cx) <= rad) & (np.abs(iy - cy) <= rad)

        b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
        n = int(keep.sum())
        z = np.asarray(r.drive_z, np.float64)[keep] if r.drive_z is not None             else np.zeros(n)
        z0 = float(np.nanmin(z)) if n and np.isfinite(z).any() else 0.0
        zq = np.clip(np.nan_to_num(z - z0, nan=0.0) * 1000.0,
                     -32000, 32000).astype(np.int16)
        counts = {k: int((r.drive_cls == v).sum()) for k, v in
                  (("drivable", tc.DRIVABLE), ("marginal", tc.MARGINAL),
                   ("non_drivable", tc.NON_DRIVABLE),
                   ("unknown", tc.UNKNOWN))}
        tot = max(len(r.drive_cls), 1)
        return {
            "frame": r.frame,
            "res": res,
            "n_cells": n,
            "ix": b64((ix[keep] - cx).astype(np.int16)),
            "iy": b64((iy[keep] - cy).astype(np.int16)),
            "origin": [cx, cy],
            "view_m": self.VIEW_M,
            "cls": b64(r.drive_cls[keep].astype(np.uint8)),
            # Ground height, millimetres, int16, relative to the frame's own
            # floor. Two bytes a cell buys the viewer an actual surface
            # instead of a flat classification raster, and int16 in mm spans
            # +/-32 m of relief, which is more than a ground map ever holds.
            "z": b64(zq), "z0": round(float(z0), 3),
            "boxes": [{"c": self.per.classes[b["c"]], "s": b["s"], "b": b["b"]}
                      for b in r.boxes],
            "counts": r.counts,
            "clusters": r.n_clusters,
            "pose": [float(T[0, 3]), float(T[1, 3]), float(T[2, 3])],
            "drive": {k: v / tot for k, v in counts.items()},
            # The tier layout, so the viewer can draw where resolution changes
            # rather than describing it in a caption. This is the whole claim
            # of an adaptive map and it was nowhere in the payload.
            "tiers": [{"res": float(self.per.map.res[L]),
                       "half_extent": float(self.per.map.n[L] *
                                            self.per.map.res[L] / 2)}
                      for L in range(len(self.per.map.res))],
            "drive_n": counts,
            "ms": r.ms,
            "read_ms": round(t_read, 2),
            "npts": int(len(r.labels)),
        }

    def snapshot(self):
        with self._lock:
            pay = self._latest
        t = self.tel.snapshot()
        h = self.hist[-120:]
        stats = {}
        if h:
            a = np.array(h)
            stats = {"median": round(float(np.median(a)), 2),
                     "p95": round(float(np.percentile(a, 95)), 2),
                     "p99": round(float(np.percentile(a, 99)), 2),
                     "fps": round(1000 / max(float(np.median(a)), 1e-6), 1),
                     "n": len(a)}
        return {"state": self.state, "err": self.err, "source": self.source,
                "telemetry": t, "frame": pay, "stats": stats,
                "hist": [round(x, 2) for x in h],
                "classes": self.per.classes if self.per else [],
                "budget_ms": 100.0}


ENGINE = Engine()


def build_app():
    app = FastAPI(title="perception runtime")

    @app.get("/api/state")
    def state():
        return ENGINE.snapshot()

    @app.post("/api/start")
    async def start(body: dict):
        root = body.get("root")
        if not root:
            return {"ok": False, "err": "root is required"}
        ENGINE.err = None
        ENGINE.start(Path(root), bool(body.get("loop", True)),
                     int(body.get("stride", 1)))
        return {"ok": True}

    @app.post("/api/stop")
    async def stop():
        ENGINE.stop()
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        last = -1
        try:
            while True:
                s = ENGINE.snapshot()
                f = s.get("frame")
                # Telemetry every tick; the map only when it changed. Resending
                # an unchanged map at 20 Hz is the easy way to make a dashboard
                # feel heavy for no information.
                if f and f["frame"] == last:
                    s["frame"] = None
                elif f:
                    last = f["frame"]
                await sock.send_text(json.dumps(s))
                await asyncio.sleep(0.1)
        except (WebSocketDisconnect, RuntimeError):
            pass

    app.mount("/", StaticFiles(directory=HERE / "web", html=True), name="web")
    return app


app = build_app()
