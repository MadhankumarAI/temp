"""The perception stack as one object, ready to be called from a ROS 2 node.

No file I/O, no argparse, no globals to configure. Construct it once, call
`process()` per sweep, read the dict. Everything expensive happens in
`__init__` so the first `process()` costs what the thousandth does.

WHY THAT MATTERS MORE THAN IT SOUNDS

The first call through this stack used to cost ten seconds: numba compiles its
kernels on first use, CUDA builds its context, the caching allocator reaches
for its first block, and the GPU is at its idle clock. A node that does that
inside its first subscriber callback has already missed a hundred sweeps, and
worse, it looks like a latency problem in the algorithm. `warm()` pays all of
it up front against synthetic points, so by the time a sensor is connected
there is nothing left to pay.

SHARING A 6 GB CARD WITH A RENDERER

The simulator needs most of the card. This stack does not: the network is
0.81 M parameters and its peak tensor allocation is about 71 MB. The risk is
not size, it is the caching allocator, which grows to the high-water mark and
does not hand memory back, so a transient spike permanently denies the
renderer that much. `vram_fraction` caps it, and the cap is enforced by torch
rather than by hope: an allocation past it raises instead of quietly evicting
the simulator's textures and turning the render into a slideshow nobody
connects to the perception node.

WHAT IS DELIBERATELY NOT HERE

No ROS. The pure function is `(points, pose) -> results`, which is the thing
worth testing and the thing worth optimising; a node wrapping it is twenty
lines and belongs with the rest of the ROS package. Keeping it out means this
can be developed and profiled with a two second edit loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    ckpt: Path = ROOT / "pointnet-det" / "runs" / "carla_scratch" / "carla_scratch" / "best.pt"
    device: str = "auto"
    # Fraction of the card this process may allocate. 0.15 of 6 GB is 900 MB
    # against a measured 71 MB peak, so it is roomy for the stack and small
    # enough that the renderer never notices.
    vram_fraction: float = 0.15
    # Height noise of the ranging sensor, removed from cell roughness so that
    # roughness measures terrain and not the instrument. Measured, not tuned:
    # integration/carla.py reads it off a surface known to be flat.
    sigma_sensor: float = 0.074
    drive_level: int = 3          # 40 cm cells for the drivability verdict
    min_cell_pts: int = 3
    # Drivability is graded only within this radius of the vehicle, and that is
    # a correctness fix as much as a speed one. Grading the WHOLE accumulated
    # map means the cost grows for as long as the vehicle drives: measured on
    # this capture it was 10 ms at frame 500 and 36 ms at frame 1000, on its
    # way to blowing the budget purely because the run had been going a while.
    # A planner does not need a verdict on road it passed a minute ago either,
    # so bounding this costs nothing anybody wanted.
    drive_radius_m: float = 60.0
    max_range: float = 100.0
    # Objects the map must not accumulate: a moving car smeared across the
    # world frame lays a wall of false terrain along its whole path.
    hold_out_moving: bool = True


@dataclass
class Result:
    """One sweep's output. Plain numpy and plain floats, nothing torch."""
    labels: np.ndarray = None            # (N,) grid25 class per point
    boxes: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    n_clusters: int = 0
    cells: dict = None                   # drivability cells, world frame
    drive_cls: np.ndarray = None         # DRIVABLE/MARGINAL/NON_DRIVABLE/UNKNOWN
    drive_score: np.ndarray = None
    ms: dict = field(default_factory=dict)
    frame: int = 0


class Perception:
    """Ground, detection, 2.5D accumulation and drivability, warm and reusable."""

    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        import sys
        for p in (ROOT / "integration", ROOT / "pointnet-det" / "src",
                  ROOT / "mapping" / "Lidar-2.5d-mapping"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))

        import torch
        import _bootstrap                                   # noqa: F401
        import grid25 as g
        import dense25 as D
        import terrain_cells as tc
        from pnd.config import Config as PndConfig
        from pnd.ground import remove_ground
        from pnd.model import build
        from pnd.simulate import process as pnd_process
        from pnd.kitti import CLASSES
        from pnd.dataset import set_anchors

        self.torch, self.g, self.D, self.tc = torch, g, D, tc
        self._remove_ground = remove_ground
        self._process = pnd_process

        dev = self.cfg.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = dev

        if dev == "cuda":
            # Cap BEFORE the first allocation, or the allocator has already
            # taken its first block and the cap applies to nothing.
            try:
                torch.cuda.set_per_process_memory_fraction(
                    self.cfg.vram_fraction, 0)
            except Exception:
                pass
            # The input shape is (clusters, 6, 256) and the cluster count moves
            # every frame, so cudnn autotuning would re-benchmark constantly
            # and never amortise. Off on purpose.
            torch.backends.cudnn.benchmark = False

        ck = torch.load(self.cfg.ckpt, map_location=dev, weights_only=False)
        pcfg = PndConfig()
        for k, v in (ck.get("cfg") or {}).items():
            if hasattr(pcfg, k) and k not in ("device", "data_root",
                                              "cache_dir", "run_dir"):
                setattr(pcfg, k, v)
        pcfg.device = dev
        # The anchor table belongs to the checkpoint. Decoding with any other
        # scales every box by the ratio between the two: right centre, right
        # heading, wrong size, and nothing raises.
        if "anchors" in ck:
            set_anchors(ck["anchors"])
        self.pnd_cfg = pcfg
        self.classes = list(ck.get("classes") or CLASSES)[:pcfg.num_classes]

        self.model = build(pcfg).to(dev)
        self.model.load_state_dict(ck["model"])
        self.model.eval()

        from predict import NAME2GRID
        self.name2grid = NAME2GRID
        self._grid_of = np.array(
            [NAME2GRID.get(n, g.other) for n in self.classes], np.int64)
        self.moving_grid = {g.car, g.ped}

        self.map = D.FastMap()
        self.frame = 0
        self.warm()

    # ------------------------------------------------------------ warm-up
    def warm(self, n: int = 3):
        """Pay for every JIT, context and first-touch before a sensor is on.

        Synthetic points rather than a recorded sweep, so this has no data
        dependency and can run in a constructor on a machine with no dataset.
        A ground plane with a few boxes above it is enough to drive every
        branch: ground segmentation finds a plane, clustering finds clusters,
        the network gets a batch, the map gets both ground and non-ground.
        """
        rng = np.random.default_rng(0)
        n_gnd = 22000
        xy = rng.uniform(-40, 40, (n_gnd, 2))
        z = np.full(n_gnd, -1.9) + rng.normal(0, 0.02, n_gnd)

        # Objects are built as DENSE blocks rather than by lifting whatever
        # ground points happen to fall inside a footprint. That earlier version
        # produced clusters too thin to clear min_cluster_pts, so no batch ever
        # reached the network and the whole detection path went uncompiled: the
        # stack reported itself warm and then spent 650 ms on its first real
        # sweep compiling the canonicalisation kernel. A warm-up that does not
        # reach the network is not a warm-up.
        obj = []
        for cx, cy in ((12, 3), (18, -4), (25, 6), (-14, 5), (8, -9)):
            n = 700
            obj.append(np.column_stack([
                cx + rng.uniform(-2.0, 2.0, n),
                cy + rng.uniform(-0.9, 0.9, n),
                -1.9 + rng.uniform(0.05, 1.7, n)]))
        o = np.vstack(obj)
        pts = np.column_stack([
            np.r_[xy[:, 0], o[:, 0]], np.r_[xy[:, 1], o[:, 1]],
            np.r_[z, o[:, 2]],
            rng.uniform(0, 1, n_gnd + len(o))])

        # The pose MOVES between warm sweeps, and that is not cosmetic. The map
        # is a circular buffer anchored to the world, so its origin only shifts
        # when the vehicle travels far enough to need it, and the kernels that
        # do the shift (`_move`, `clear_strip`) never run at a standing pose.
        # Warming at the origin left them uncompiled and the FIRST REAL SWEEP
        # paid 127 ms of accumulation against 5 ms afterwards. A warm-up that
        # never moves does not warm a map that follows the vehicle.
        for k in range(max(n, 1)):
            T = np.eye(4)
            T[0, 3] = k * 30.0            # far enough to force a ring move
            T[1, 3] = k * 12.0
            self.process(pts, T, _warm=True)
        # the warm-up's own map is not the mission's map
        self.map = self.D.FastMap()
        self.frame = 0
        if self.device == "cuda":
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()
        return self

    # ------------------------------------------------------------ per sweep
    def process(self, points: np.ndarray, pose: np.ndarray,
                _warm: bool = False) -> Result:
        """One sweep.

        points  (N, 4) float, x y z intensity, sensor at the origin
        pose    (4, 4) world <- sensor
        """
        t = _Clock()
        pts = np.ascontiguousarray(points, np.float64)
        r = Result(frame=self.frame)

        with t("ground"):
            gr = self._remove_ground(pts[:, :3].astype(np.float32),
                                     thresh=self.pnd_cfg.ground_thresh)
        is_ground = gr[0]

        with t("detect"):
            # inference_mode, not no_grad: it also disables version counters
            # and view tracking, so the autograd bookkeeping never happens
            # rather than happening and being discarded.
            with self.torch.inference_mode():
                cls, boxes, _, ncl, _, _ = self._process(
                    pts, self.pnd_cfg, self.model, self.device, None,
                    ground=gr, terrain=False)

        with t("label"):
            lab = self._to_grid(cls, is_ground)

        with t("accumulate"):
            moving = None
            if self.cfg.hold_out_moving:
                moving = np.isin(lab, list(self.moving_grid))
            self.map.ingest(pts, lab, np.asarray(pose, np.float64),
                            moving=moving,
                            groundcls=(self.g.gnd, self.g.road))

        with t("drivability"):
            res = float(self.map.res[self.cfg.drive_level])
            cells = self.map.cells(self.cfg.drive_level,
                                   min_pts=self.cfg.min_cell_pts)
            sx, sy = float(pose[0, 3]), float(pose[1, 3])
            cells = _near(cells, sx, sy, res, self.cfg.drive_radius_m)
            score, dcls, _ = self.tc.cell_drivability(
                cells, (sx, sy), res=res,
                sigma_sensor=self.cfg.sigma_sensor)

        r.labels, r.boxes, r.n_clusters = lab, boxes, int(ncl)
        r.cells, r.drive_cls, r.drive_score = cells, dcls, score
        r.counts = self._counts(boxes)
        r.ms = t.ms
        r.ms["total"] = round(sum(t.ms.values()), 2)
        if not _warm:
            self.frame += 1
        return r

    # ------------------------------------------------------------ helpers
    def _to_grid(self, cls, is_ground):
        """Per-point display classes to grid25's eight."""
        from pnd.simulate import (CLS_DRIVABLE, CLS_MARGINAL, CLS_NONDRIV,
                                  CLS_OFFSET)
        out = np.full(len(cls), self.g.other, np.int64)
        out[is_ground] = self.g.road
        for c in (CLS_DRIVABLE, CLS_MARGINAL, CLS_NONDRIV):
            out[cls == c] = self.g.road
        # vectorised over the class list rather than a python loop per class:
        # the table is built once in __init__ and indexed here
        code = cls.astype(np.int64) - CLS_OFFSET
        hit = (code >= 0) & (code < len(self._grid_of))
        out[hit] = self._grid_of[code[hit]]
        return out

    def _counts(self, boxes):
        c = {}
        for b in boxes:
            n = self.classes[b["c"]] if b["c"] < len(self.classes) else "?"
            c[n] = c.get(n, 0) + 1
        return c

    def memstats(self):
        s = self.g.memstats(self.map.cells(0, min_pts=1)) \
            if hasattr(self.g, "memstats") else {}
        return s

    def reset_map(self):
        """Drop the accumulated map, keeping the warm model and kernels."""
        self.map = self.D.FastMap()
        return self


def _near(cells, sx, sy, res, radius):
    """Cells within `radius` of (sx, sy), as a Chebyshev window.

    Square rather than circular on purpose: it is one comparison per axis
    instead of a hypot over every cell, the difference at the corners is map
    nobody was going to plan through anyway, and this runs every frame.
    """
    if radius <= 0:
        return cells
    r = radius / res
    cx, cy = sx / res, sy / res
    m = ((np.abs(cells["ix"] - cx) <= r) & (np.abs(cells["iy"] - cy) <= r))
    if m.all():
        return cells                       # no copy when nothing is dropped
    return {k: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == m.shape
                else v)
            for k, v in cells.items()}


class _Clock:
    """Stage timing without a `t0 = ...` line before every block."""

    def __init__(self):
        self.ms = {}
        self._k = None

    def __call__(self, key):
        self._k = key
        return self

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.ms[self._k] = round((time.perf_counter() - self._t) * 1000, 2)
        return False
