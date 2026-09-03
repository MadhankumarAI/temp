# runtime

The perception backend as a shippable unit, plus an operator dashboard.

No ROS code here, deliberately. What this gives you is a pure function,
`(points, pose) -> results`, that is warm, bounded and tested. Wrapping it in a
node is the easy part and belongs in your ROS package, where it can depend on
your message types and your QoS choices rather than guessing at them.

---

## Install

Torch first, and from the CUDA index, or you silently get the CPU wheel and the
detector goes from 22 ms to about 190 ms with nothing reporting a problem:

```powershell
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r runtime/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

That last line must print `2.5.1+cu121 True` before any timing here means
anything.

## Run it

Backend + dashboard:

```powershell
python -m uvicorn server:app --app-dir runtime --host 127.0.0.1 --port 8020
```

Then <http://127.0.0.1:8020>, put a capture path in the box, press RUN.

Tests, which must pass before you ship it anywhere:

```powershell
python runtime\selftest.py --root D:\simulation_dataset_town01_final
```

---

## Using it from a ROS 2 node

The whole integration is this:

```python
from pipeline import Config, Perception

class Node(rclpy.node.Node):
    def __init__(self):
        super().__init__("perception")
        # Construct ONCE, in __init__, never in the callback. The constructor
        # warms every numba kernel, the CUDA context and the allocator, which
        # is about 1.5 s. Doing it in the first callback means the node misses
        # every sweep until it finishes, and it looks like a latency bug.
        self.per = Perception(Config())
        ...

    def on_cloud(self, msg):
        pts = read_points_numpy(msg)          # (N,4) x y z intensity
        T = self.lookup_pose(msg.header.stamp)
        r = self.per.process(pts, T)
        # r.labels, r.boxes, r.cells, r.drive_cls, r.ms
```

Four things that will bite if they are skipped:

**Copy the input header stamp forward, never restamp.** End-to-end latency is
`now - header.stamp` at the consumer. A restamped message produces a
plausible-looking number that is wrong and that nothing can falsify.

**Best-effort, keep-last, depth 1 on the cloud topic.** Reliable with a depth
of 10 silently builds a backlog: every stage reports healthy per-frame times
while the map falls seconds behind the vehicle.

**`process()` is not thread-safe.** It mutates one accumulated map. One node,
one instance, one callback at a time. A callback group that allows concurrency
will corrupt the map rather than raise.

**Pose is world←sensor, 4x4.** Getting this from TF is the integration's real
work; a wrong pose does not error, it smears the map along the path.

---

## Sharing a 6 GB card with CARLA

`Config.vram_fraction` caps what this process may allocate, default 0.15, which
is 900 MB of a 6 GB card against a measured 71 MB peak. The cap is enforced by
torch before the first allocation, so an unexpected spike raises here instead
of quietly evicting the renderer's textures.

The dashboard attributes resources rather than aggregating them, and is honest
about which half it can measure:

| | how |
|---|---|
| backend CPU | exact, `GetProcessTimes` on our pid |
| simulator CPU | exact, same call on its pid, found by name |
| backend GPU memory | exact, `torch.cuda.memory_reserved` |
| other GPU memory | **inferred**, total minus ours |
| GPU utilisation | **whole card only**, not split |

The last two are inferred because this driver does not expose per-process GPU
memory: on a consumer Windows card in WDDM mode `nvidia-smi
--query-compute-apps` returns `[N/A]`, verified. Attributing the remainder to
the simulator is right when the simulator is the only other GPU client, and the
field is called `other` rather than `carla` because a hardware-accelerated
browser is also in there. Utilisation is not split at all, because a split
would be invented.

A simulator that is not running reads as **absent**, not as 0%.

---

## Where your CARLA render goes

`web/index.html` has an empty panel, `#sim-slot`. Put an `<img>`, `<video>` or
`<canvas>` in it and the CSS sizes it to fill; nothing else needs changing. Its
GPU cost lands in the `other` column, so filling that panel will not make the
backend's own numbers move.

---

## What is measured, on a GTX 1650

Real CARLA sweeps at ~25 k points, 1,951 frames:

```
ground        1.6 ms      backend GPU     69 MB reserved of a 900 MB cap
detect       22.0         backend CPU     ~10 of 12 cores
label         0.5         frame          ~40 ms median, 25 FPS
accumulate    3.0         budget          100 ms at 10 Hz
drivability  12.7
```

Two bounds worth knowing about, because both were bugs first:

**Drivability is graded within `drive_radius_m` (60 m) of the vehicle**, not
over the whole map. Grading everything made the cost grow with how long the run
had been going: 10 ms at frame 500, 36 ms at frame 1000, on its way through the
budget for no reason but elapsed time.

**The dashboard payload is windowed to 80 m and sent as int16 offsets** from the
vehicle's cell. Sending the whole map started at 280 KB a frame and had no
ceiling; it is now flat at ~150 KB however long the run goes.

---

## Files

| | |
|---|---|
| `pipeline.py` | `Perception`, the pure function. This is the thing you wrap. |
| `telemetry.py` | per-process CPU, GPU accounting, simulator detection |
| `server.py` | FastAPI + one WebSocket; the engine runs perception on its own thread |
| `web/` | the dashboard |
| `selftest.py` | warm, determinism, VRAM cap, leaks, latency, telemetry |

`selftest.py` is not a smoke test. Every check in it either caught something
real or guards a failure that does not raise.
