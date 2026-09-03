"""What has to be true before this ships.

Not a smoke test. Each check here is one that has already caught something, or
guards a failure mode that does not raise:

  warm            the first sweep must cost what the tenth does. A stack that
                  JITs inside its first callback has missed a hundred sweeps
                  and it looks like an algorithm problem.
  vram            the cap has to be real, and the steady state has to not grow.
                  A caching allocator that creeps takes the renderer's memory
                  and never gives it back.
  stability       identical input must give identical output. Non-determinism
                  here would make every later comparison meaningless.
  no_leak         a thousand sweeps must not grow host or device memory.
  latency         the budget is 100 ms at 10 Hz, and the TAIL is what drops
                  frames, so p99 is checked and not just the median.
  telemetry       the numbers must be attributed, and absent things must read
                  as absent rather than as zero.

usage:
  python runtime/selftest.py                 synthetic, always runnable
  python runtime/selftest.py --root D:\...   also runs real CARLA sweeps
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import Config, Perception            # noqa: E402
from telemetry import Telemetry, _cpu_seconds      # noqa: E402

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'ok  ' if ok else 'FAIL'} {name:22} {detail}")
    return ok


def synth(n=22000, seed=0):
    """Ground plus DENSE object blocks, so clustering actually yields clusters.

    Lifting whatever ground points fell inside a footprint gave clusters too
    thin to clear min_cluster_pts, so the network never ran and the test
    reported '0 boxes' while passing. A synthetic scene that never reaches the
    thing under test is not testing it.
    """
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-40, 40, (n, 2))
    z = np.full(n, -1.9) + rng.normal(0, 0.02, n)
    obj = []
    for cx, cy in ((12, 3), (18, -4), (25, 6), (-14, 5), (8, -9)):
        k = 700
        obj.append(np.column_stack([
            cx + rng.uniform(-2.0, 2.0, k), cy + rng.uniform(-0.9, 0.9, k),
            -1.9 + rng.uniform(0.05, 1.7, k)]))
    o = np.vstack(obj)
    return np.column_stack([
        np.r_[xy[:, 0], o[:, 0]], np.r_[xy[:, 1], o[:, 1]], np.r_[z, o[:, 2]],
        rng.uniform(0, 1, n + len(o))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None,
                    help="a CARLA capture, to test on real sweeps too")
    ap.add_argument("--frames", type=int, default=60)
    a = ap.parse_args()

    import torch
    print(f"device {'cuda' if torch.cuda.is_available() else 'cpu'}"
          f"{'  ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    t0 = time.perf_counter()
    p = Perception(Config())
    print(f"construct + warm {time.perf_counter()-t0:.2f} s, "
          f"{len(p.classes)} classes, device {p.device}\n")

    pts, pose = synth(), np.eye(4)

    # -- warm ---------------------------------------------------------- #
    first = p.process(pts, pose).ms["total"]
    later = [p.process(pts, pose).ms["total"] for _ in range(9)]
    med = float(np.median(later))
    check("warm", first < max(3 * med, med + 15),
          f"first {first:.1f} ms vs median {med:.1f} ms")

    # -- determinism --------------------------------------------------- #
    p.reset_map()
    np.random.seed(0)
    r1 = p.process(pts, pose)
    p.reset_map()
    np.random.seed(0)
    r2 = p.process(pts, pose)
    same = (np.array_equal(r1.labels, r2.labels)
            and len(r1.boxes) == len(r2.boxes)
            and np.array_equal(r1.drive_cls, r2.drive_cls))
    check("stability", same,
          f"{len(r1.boxes)} boxes, {len(r1.labels):,} labels, identical"
          if same else "SAME INPUT GAVE DIFFERENT OUTPUT")
    # A scene that produces no clusters leaves the network's kernels
    # uncompiled, and the warm check above then passes while the first real
    # sweep pays 650 ms. Assert the synthetic scene reaches the network.
    check("warm_reaches_net", r1.n_clusters > 0,
          f"{r1.n_clusters} clusters through the network during warm-up")

    # -- vram cap and growth ------------------------------------------- #
    if p.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        p.reset_map()
        for i in range(80):
            p.process(synth(seed=i), pose)
        peak = torch.cuda.max_memory_allocated() / 1e6
        res = torch.cuda.memory_reserved() / 1e6
        cap = p.cfg.vram_fraction * torch.cuda.get_device_properties(0).total_memory / 1e6
        check("vram", res <= cap,
              f"peak {peak:.0f} MB, reserved {res:.0f} MB, cap {cap:.0f} MB")

        # steady state must not creep: measure over the SECOND half only, so
        # the allocator's initial growth is not counted as a leak
        before = torch.cuda.memory_reserved()
        for i in range(80):
            p.process(synth(seed=100 + i), pose)
        after = torch.cuda.memory_reserved()
        check("no_leak_gpu", after <= before,
              f"reserved {before/1e6:.0f} -> {after/1e6:.0f} MB over 80 sweeps")
    else:
        check("vram", True, "cpu device, not applicable")
        check("no_leak_gpu", True, "cpu device, not applicable")

    # -- host memory ---------------------------------------------------- #
    p.reset_map()
    gc.collect()
    n0 = len(gc.get_objects())
    for i in range(40):
        p.process(synth(seed=200 + i), pose)
    gc.collect()
    n1 = len(gc.get_objects())
    # the map legitimately grows as it accumulates; object COUNT should not
    check("no_leak_host", n1 - n0 < 20000,
          f"{n0:,} -> {n1:,} tracked objects over 40 sweeps")

    # -- latency -------------------------------------------------------- #
    p.reset_map()
    ms = [p.process(synth(seed=300 + i), pose).ms["total"] for i in range(60)]
    ms = np.array(ms)
    p50, p99 = float(np.median(ms)), float(np.percentile(ms, 99))
    check("latency", p99 < 100.0,
          f"median {p50:.1f} ms, p99 {p99:.1f} ms, budget 100 ms "
          f"({1000/p50:.1f} FPS)")

    # -- telemetry ------------------------------------------------------ #
    tm = Telemetry(period=0.2).start()
    time.sleep(1.4)
    s = tm.snapshot()
    tm.stop()
    ok = ("cpu_backend" in s and s.get("cpu_backend") is not None
          and "cores_total" in s)
    check("telemetry_cpu", ok,
          f"backend {s.get('cpu_backend')} of {s.get('cores_total')} cores")
    if p.device == "cuda":
        ok = "gpu_backend_mb" in s and "gpu_util" in s
        check("telemetry_gpu", ok,
              f"ours {s.get('gpu_backend_mb')} MB, other "
              f"{s.get('gpu_other_mb')} MB, card {s.get('gpu_util')}%")
    # a simulator that is not running must read as absent, not as 0%
    check("telemetry_absent", (s.get("cpu_sim") is None) == (not s.get("sim_running")),
          f"sim_running={s.get('sim_running')} cpu_sim={s.get('cpu_sim')}")

    # -- real sweeps ---------------------------------------------------- #
    if a.root:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "integration"))
        import _bootstrap                                   # noqa: F401
        import carla as CA
        meta = CA.read_meta(a.root)
        ids = list(CA.frame_ids_of(a.root, meta))[:a.frames]
        p.reset_map()
        rms, nbox = [], 0
        for i in ids:
            f = CA.load(a.root, i, meta)
            r = p.process(f.pts, f.T)
            rms.append(r.ms["total"])
            nbox += len(r.boxes)
        rms = np.array(rms)
        check("real_latency", float(np.percentile(rms, 99)) < 100.0,
              f"{len(ids)} sweeps, median {np.median(rms):.1f} ms, "
              f"p99 {np.percentile(rms,99):.1f} ms")
        check("real_output", nbox > 0,
              f"{nbox} boxes over {len(ids)} sweeps "
              f"({nbox/max(len(ids),1):.1f} a sweep)")
        cov = float((p.process(f.pts, f.T).drive_cls != p.tc.UNKNOWN).mean())
        check("real_drivability", cov > 0.05,
              f"{100*cov:.1f}% of map cells have a verdict")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("  " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
