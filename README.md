# Adaptive 2.5D LiDAR Perception — runtime

A LiDAR perception backend packaged to be embedded: ten-class object detection
(static and dynamic), a variable-resolution 2.5D elevation map, terrain
drivability, and an operator dashboard. Trained on CARLA Town01.

**Start here: [`runtime/README.md`](runtime/README.md)** — how to run it, how to
call it from a ROS 2 node, and what is measured.

```powershell
python -m uvicorn server:app --app-dir runtime --host 127.0.0.1 --port 8020
python runtime/selftest.py --root <a CARLA capture>
```

## What it does

- **Adaptive 2.5D map.** Cell size grows with range, 5 cm near the vehicle to
  40 cm at 100 m, so detail is spent where it can be resolved. 16x less memory
  than a uniform 5 cm grid over the same ground.
- **Object detection**, ten classes: vehicles and vulnerable road users, plus
  the static furniture a planner must not clip — poles, traffic lights, signs
  and built structure.
- **Terrain drivability** from slope, step and roughness, with the ranging
  sensor's own noise removed so roughness measures ground and not instrument.
- **Attributed telemetry**, so a backend sharing a GPU with a simulator can be
  told apart from it.

## Measured

On a GTX 1650 over real CARLA sweeps: **~40 ms a frame, 25 FPS** against a
100 ms budget at 10 Hz, using **69 MB of VRAM against a 900 MB cap** so a
simulator can have the rest of the card.

```
ground 1.6   detect 22.0   label 0.5   accumulate 3.0   drivability 12.7
```

Held-out test split, contiguous stretches never used for training or for
choosing a checkpoint: **88.8% mean F1** over the classes with enough examples,
and recall flat across distance — 89% at 0-12 m, 93% at 12-25 m, 94% at
25-50 m.

## Requirements

`torch`, `numba`, `numpy`, `scipy`, `fastapi`, `uvicorn`, `websockets`.

Everything outside `runtime/` is the dependency closure, laid out exactly as it
sits in the project it came from so no import path had to change: the detector
package, the grid engine, the dense tiered map, the drivability estimator, the
CARLA reader, and the trained checkpoint.
