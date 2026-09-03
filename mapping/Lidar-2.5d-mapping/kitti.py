"""
load a semantickitti frame as (points Nx3, labels N) in grid25's class set.

velodyne .bin is float32 (n,4) = x y z intensity, in the sensor frame with
the sensor at the origin, roughly 1.73 m above the road. .label is uint32
per point: low 16 bits semantic, high 16 instance.
"""

import numpy as np
from grid25 import gnd, road, bldg, pole, veg, car, ped, other

# semantickitti id -> ours. sidewalk and terrain are ground, not road: they
# are part of the terrain surface, so the kerb between them stays a step
# rather than a cliff at the edge of the known world.
_static = {40: road, 44: road, 60: road,
           48: gnd, 49: gnd, 72: gnd,
           50: bldg, 51: bldg, 52: bldg,
           80: pole, 81: pole,
           70: veg, 71: veg,
           10: car, 11: car, 13: car, 15: car, 16: car, 18: car, 20: car,
           30: ped, 31: ped, 32: ped}

# the moving-* ids are the only ground truth anywhere for is_dynamic. fold
# them onto their static class and carry the flag out separately.
_moving = {252: car, 253: ped, 254: ped, 255: ped,
           256: car, 257: car, 258: car, 259: car}

LUT = np.full(65536, other, np.int64)
MOV = np.zeros(65536, bool)
for k, v in _static.items():
    LUT[k] = v
for k, v in _moving.items():
    LUT[k] = v
    MOV[k] = True

names = {40:'road', 44:'parking', 48:'sidewalk', 49:'other-ground', 50:'building',
         51:'fence', 70:'vegetation', 71:'trunk', 72:'terrain', 80:'pole',
         81:'traffic-sign', 10:'car', 18:'truck', 30:'person', 0:'unlabeled'}


def load(binpath, labpath=None):
    p = np.fromfile(binpath, np.float32).reshape(-1, 4)
    pts = p[:, :3].astype(np.float64)
    if labpath is None:
        return pts, np.full(len(pts), other, np.int64), np.zeros(len(pts), bool)
    raw = np.fromfile(labpath, np.uint32) & 0xffff
    return pts, LUT[raw], MOV[raw]
