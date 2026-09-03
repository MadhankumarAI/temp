"""Read a CARLA capture into the shapes this pipeline already speaks.

The pipeline was built against KITTI, and CARLA differs in four ways that all
produce plausible-looking wrong answers rather than errors:

  1. CARLA's world is LEFT-handed -- x forward, y RIGHT, z up. grid25 and the
     detector both assume the velodyne convention, x forward, y LEFT. Used
     directly every scene is mirrored: a car parked on the left kerb maps to
     the right one, and the yaw of every box is negated. So `y` is flipped on
     the points and the pose is conjugated, not merely loaded.
  2. The semantic sensor and the ranging sensor are DIFFERENT sensors with
     different point counts (~46k against ~25k here). They do not index-align,
     so ground truth reaches the ranging cloud by nearest neighbour, and the
     match distance is reported rather than assumed to be zero.
  3. `object_idx` and `object_tag` are uint32, and a capture may store them
     either as their bit pattern in a float32 array or cast to float. Read the
     wrong way round the first gives denormals near 1e-43 that compare and sort
     without ever raising. `decode_ids` tells them apart.
  4. The lidar sits ~2.4 m up where KITTI's sits at 1.73 m. Nothing downstream
     reads absolute height -- ground removal fits planes and every grid25
     statistic is relative to the fitted ground -- but the world map does, so
     the mount enters through the pose and not through a shift of the points.

Tag numbering is the CARLA 0.9.14+ scheme. Confirmed against the data rather
than assumed: tag 1 sits at ground level, tag 7 is elevated AND carries actor
ids, and tag 28 exists at all -- none of which is true under the older scheme,
where 7 is Road and 13 is Sky.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import _bootstrap        # noqa: F401  (puts grid25 on the path)
import grid25 as g

# ------------------------------------------------------------------ tags

TAGS = [
    "Unlabeled", "Roads", "SideWalks", "Building", "Wall", "Fence", "Pole",
    "TrafficLight", "TrafficSign", "Vegetation", "Terrain", "Sky",
    "Pedestrian", "Rider", "Car", "Truck", "Bus", "Train", "Motorcycle",
    "Bicycle", "Static", "Dynamic", "Other", "Water", "RoadLine", "Ground",
    "Bridge", "RailTrack", "GuardRail",
]

# CARLA tag -> the eight classes grid25 quantises into.
#
# Deliberately lossy in the same direction the detector is: it has no class for
# building, vegetation or pole either, so a richer mapping here would make the
# ground truth express things nothing the model produces ever could, and every
# comparison would read as a failure of the model rather than of the class set.
_G = {
    "Roads": g.road, "RoadLine": g.road, "Ground": g.gnd, "Terrain": g.gnd,
    "SideWalks": g.gnd, "Building": g.bldg, "Wall": g.bldg, "Fence": g.bldg,
    "GuardRail": g.bldg, "Bridge": g.bldg,
    "Pole": g.pole, "TrafficLight": g.pole, "TrafficSign": g.pole,
    "Vegetation": g.veg,
    "Car": g.car, "Truck": g.car, "Bus": g.car, "Train": g.car,
    "Pedestrian": g.ped, "Rider": g.ped, "Motorcycle": g.ped,
    "Bicycle": g.ped,
}
TAG2GRID = np.array([_G.get(n, g.other) for n in TAGS], np.int64)

# CARLA tag -> the detector's own class list, for scoring detections.
#
# Motorcycle and Bicycle join Rider as Cyclist because that is the KITTI class
# they were trained as: a KITTI Cyclist box contains the rider AND the machine,
# so a CARLA bike with nobody on it is still the thing the network learned.
TAG2DET = {"Car": "Car", "Truck": "Truck", "Bus": "Truck",
           "Pedestrian": "Pedestrian", "Rider": "Cyclist",
           "Motorcycle": "Cyclist", "Bicycle": "Cyclist"}
DET_TAGS = {TAGS.index(k): v for k, v in TAG2DET.items()}

# CARLA's own name for drivable road surface. Used as the truth for terrain
# scoring, where it is better ground truth than KITTI offers -- it is the
# simulator's own answer, not a human annotation.
ROAD_TAGS = np.array([TAGS.index("Roads"), TAGS.index("RoadLine")])


# ------------------------------------------------------------- geometry

MIRROR = np.diag([1.0, -1.0, 1.0])


def carla_to_velo(pts: np.ndarray) -> np.ndarray:
    """Flip y. Only the first three columns are geometry; intensity rides
    along untouched."""
    out = np.array(pts, np.float64, copy=True)
    out[:, 1] *= -1.0
    return out


def _carla_rotation(roll, pitch, yaw):
    """CARLA's own Transform.get_matrix(), in radians.

    Written out rather than approximated by Rz(yaw) because roll and pitch are
    exactly zero in the first capture and that is a property of THAT drive, not
    of the format. A slope or a kerb strike puts values in them, and a yaw-only
    reader would silently keep returning a level vehicle.
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp,      -cp * sr,               cp * cr],
    ])


def pose(row, mount_z: float) -> np.ndarray:
    """(4,4) taking a lidar point at this frame into a right-handed z-up world.

    Two steps, and skipping either one is invisible until the accumulated map
    is drawn. The CARLA rotation is conjugated through the mirror -- M R M, not
    M R -- because a rotation expressed in a left-handed frame is not the same
    rotation once the axis it turns about has been flipped. Then the mount
    offset is applied in the VEHICLE frame, which is what lifts the world map
    off the road surface by the right amount.
    """
    R = _carla_rotation(np.radians(float(row["ego_roll"])),
                        np.radians(float(row["ego_pitch"])),
                        np.radians(float(row["ego_yaw"])))
    R = MIRROR @ R @ MIRROR
    t = MIRROR @ np.array([float(row["ego_x"]), float(row["ego_y"]),
                           float(row["ego_z"])])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t + R @ np.array([0.0, 0.0, mount_z])
    return T


# --------------------------------------------------------------- frames

@dataclass
class Frame:
    id: int
    pts: np.ndarray          # (N,4) x y z intensity, velodyne convention
    sem: np.ndarray          # (M,3) xyz from the semantic sensor, same frame
    tag: np.ndarray          # (M,) CARLA semantic tag
    obj: np.ndarray          # (M,) CARLA object id, 0 for non-actors
    T: np.ndarray            # (4,4) world <- lidar
    row: dict                # the driving_data.csv row


def read_meta(root: Path) -> dict[int, dict]:
    """sample_id -> row.

    The three path columns are ignored. They were written by a WSL run and
    point at /mnt/d/final/pothole/..., which does not exist on the machine the
    data now sits on; resolving files from `root` is the only thing that keeps
    working when the dataset is moved or copied.
    """
    with open(root / "driving_data.csv", newline="") as f:
        return {int(r["sample_id"]): r for r in csv.DictReader(f)}


def frame_ids(root: Path) -> list[int]:
    """Frames that have BOTH a scan and a pose, in order.

    A scan with no csv row has no pose, so it cannot enter the world map and
    cannot be scored against anything. Dropping it here rather than at the
    point of use keeps the count honest: the caller is told how many frames it
    is running on, not how many files exist.
    """
    return list(frame_ids_of(root, read_meta(root)))


def decode_ids(sem: np.ndarray):
    """(object_idx, object_tag) from a semantic array written either way.

    CARLA hands these over as uint32. A writer that drops them into a float32
    array unconverted stores the BIT PATTERN, and they read back as denormals
    near 1e-43; a writer that casts them stores 14.0 and 21.0. Both exist in
    this project's captures -- the first dataset was the bit form, the
    re-record is the cast form -- and each is silently wrong when read the
    other way. The bit form read as floats gives every point a tag of
    ~0.00000000000000000000000000000000000000000001, which compares and sorts
    without complaint; the cast form read as bits gives tag 1091567616, which
    at least indexes out of bounds.

    They are told apart by magnitude, which is unambiguous: real tags are small
    integers, bit patterns of small integers are denormals many orders of
    magnitude below 1.
    """
    if float(sem[:, 5].max()) > 1e-6:
        return sem[:, 4].astype(np.int64), sem[:, 5].astype(np.int64)
    bits = sem.view(np.uint32)
    return bits[:, 4].astype(np.int64), bits[:, 5].astype(np.int64)


def load(root: Path, i: int, meta: dict, mount_z: float = 2.4) -> Frame:
    raw = np.load(root / "raw_lidar" / f"lidar_{i:06d}.npz")["points"]
    sem = np.load(root / "semantic_lidar" / f"semantic_{i:06d}.npz")["points"]
    obj, tag = decode_ids(sem)
    return Frame(
        id=i,
        pts=carla_to_velo(raw),
        sem=carla_to_velo(sem[:, :3]),
        tag=tag,
        obj=obj,
        T=pose(meta[i], mount_z),
        row=meta[i],
    )


def truth(f: Frame, max_dist: float = 0.5):
    """Ground truth for the RANGING cloud: grid25 class, road mask, match quality.

    All three come out of one query because building the tree and searching it
    is the expensive part and they all want the same answer -- the semantic
    return nearest each ranging return. Two separate entry points meant two
    trees over 46k points per frame, for one lookup.

    The two sensors are configured alike, so in practice the ranging returns
    are a subset of the semantic ones and the match distance is 0. That is a
    property of the capture, not a guarantee: a different rotation frequency or
    range on either sensor breaks it silently, which is why the distance is
    measured and returned rather than assumed. Points whose nearest neighbour
    is beyond `max_dist` come back as `other` and are counted.
    """
    from scipy.spatial import cKDTree
    d, j = cKDTree(f.sem).query(f.pts[:, :3], workers=-1)
    near = d <= max_dist
    lab = TAG2GRID[f.tag[j]]
    lab[~near] = g.other
    road = np.isin(f.tag[j], ROAD_TAGS) & near
    return lab, road, dict(median_match_m=float(np.median(d)),
                           p95_match_m=float(np.percentile(d, 95)),
                           unmatched=int((~near).sum()), n=int(len(d)))


# ------------------------------------------------------- truth instances

def instances(f: Frame, min_pts: int = 20, ego_radius: float = 4.0) -> list[dict]:
    """One box per CARLA actor visible in this sweep, excluding the ego.

    THE EGO IS NOT AN OBSTACLE. CARLA's semantic sensor returns hits on the
    actor it is mounted to, so every sweep in the first capture carried a `Car`
    of 37 points at 2.2 m and 176 degrees -- the roof and boot of the vehicle
    the laser is bolted to. Left in, it is one guaranteed miss per frame in
    exactly the class that matters most, and the sequence reported 101 cars and
    a recall of zero while the street was empty. Anything whose centre is
    inside `ego_radius` is the mount, not traffic: the sensor cannot be that
    close to another vehicle.

    Derived from the semantic returns grouped by object id, because the capture
    does not record actor bounding boxes. That makes every box a HULL OF WHAT
    WAS SEEN, not the actor's true extent: a car observed from one side has no
    returns on its far flank and comes out roughly half as wide. Recall against
    these boxes is meaningful; absolute IoU against them is pessimistic, and a
    future capture should log `actor.bounding_box` beside the scan so the two
    can be compared.
    """
    out = []
    keep = np.isin(f.tag, list(DET_TAGS)) & (f.obj != 0)
    if not keep.any():
        return out
    key = f.obj[keep] * 64 + f.tag[keep]
    xyz = f.sem[keep]
    order = np.argsort(key, kind="stable")
    k = key[order]
    st = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    for a, b in zip(st, np.r_[st[1:], len(k)]):
        p = xyz[order[a:b]]
        if len(p) < min_pts:
            continue
        c = p.mean(0)
        if np.hypot(c[0], c[1]) < ego_radius:
            continue                       # the vehicle the sensor rides on
        # yaw from a 2D pca of the footprint, the same canonicalisation the
        # detector applies to its clusters, so the two headings are comparable
        d = p[:, :2] - c[:2]
        _, v = np.linalg.eigh(d.T @ d)
        yaw = float(np.arctan2(v[1, -1], v[0, -1]))
        cs, sn = np.cos(-yaw), np.sin(-yaw)
        loc = np.stack([d[:, 0] * cs - d[:, 1] * sn,
                        d[:, 0] * sn + d[:, 1] * cs], 1)
        lo, hi = loc.min(0), loc.max(0)
        out.append(dict(
            cls=DET_TAGS[int(k[a] % 64)],
            obj=int(k[a] // 64),
            npts=int(b - a),
            box=[float(c[0]), float(c[1]),
                 float((p[:, 2].min() + p[:, 2].max()) / 2),
                 float(hi[0] - lo[0]), float(hi[1] - lo[1]),
                 float(p[:, 2].max() - p[:, 2].min()), yaw],
            rng=float(np.hypot(c[0], c[1])),
        ))
    return out


# ---------------------------------------------------------------- check

def check(root: Path) -> dict:
    """Re-run the four defects found in the first capture.

    Here rather than in a notebook because they are the acceptance test for a
    re-record: a capture that passes all of them is one this pipeline can be
    trusted on, and the point of writing them down is that the next dataset
    gets checked by running something instead of by remembering to look.
    """
    meta = read_meta(root)
    scans = sorted((root / "raw_lidar").glob("lidar_*.npz"))
    half, tags = 0, np.zeros(len(TAGS), np.int64)
    npts, inten = [], []
    for p in scans:
        i = int(p.stem.split("_")[1])
        pts = np.load(p)["points"]
        npts.append(len(pts))
        inten.append(pts[:, 3])
        frac = float((pts[:, 1] < -1e-3).mean())
        if frac < 0.01 or frac > 0.99:
            half += 1
        s = root / "semantic_lidar" / f"semantic_{i:06d}.npz"
        if s.exists():
            _, t = decode_ids(np.load(s)["points"])
            tags += np.bincount(t, minlength=len(TAGS))[:len(TAGS)]

    grids = sorted((root / "adaptive_2_5d").glob("grid_*.npz"))
    labelled = sum(int(np.load(p)["semantic_class"].max()) > 0 for p in grids)

    noise = range_noise(root, meta)
    inside = sum(int(float(r["inside_depression"]) > 0) for r in meta.values())
    depths = {float(r["depression_depth_m"]) for r in meta.values()} - {0.0}
    ped = int(tags[TAGS.index("Pedestrian")])

    return {
        "frames": len(scans),
        "sweeps": {
            "half_sweep_frames": half,
            "ok": half == 0,
            "why": "each frame must span both signs of y; a frame entirely on "
                   "one side is half a rotation, from rotation_frequency not "
                   "matching the sensor tick",
        },
        "pedestrians": {
            "points": ped, "ok": ped > 0,
            "why": "CARLA tag 12; zero means no walker was ever in the beam",
        },
        # -1 in distance_to_center_m is the writer's SENTINEL for "not near a
        # depression", not a negative distance. An earlier version of this
        # check failed on it and was wrong to: the first capture's tell was
        # that the value sat near -1008 with `inside` never set, which is a
        # broken calculation, where -1 with `inside` set on some frames is a
        # working one. What is actually required is that some frame is inside a
        # depression and that the depressions differ from each other -- a
        # single hardcoded depth is the failure mode that started this.
        "depressions": {
            "frames_inside": inside,
            "distinct_depths": len(depths),
            "sentinel": SENTINEL,
            "ok": bool(inside > 0 and len(depths) > 1),
            "why": f"inside_depression must be set on some frames, and the "
                   f"depths must vary -- one depth repeated is a constant, not "
                   f"a measurement. {SENTINEL} means 'no depression near' and "
                   f"is fine",
        },
        "grid_labels": {
            "labelled_grids": labelled, "total": len(grids),
            "ok": bool(grids) and labelled == len(grids),
            "why": "semantic_class all zero means the labels were never "
                   "joined into the 2.5D grid build",
        },
        "metadata": {
            "rows": len(meta), "scans": len(scans),
            "ok": len(meta) == len(scans),
            "why": "a scan with no csv row has no pose and cannot be mapped",
        },
        # The two below are not defects in the capture script. They are the
        # domain gap, and they are checked here because they are the difference
        # between a detector that fires and one that returns nothing, and
        # neither announces itself: the pipeline runs clean and finds nothing.
        "density": {
            "points_per_sweep": int(np.mean(npts)),
            "kitti": KITTI_POINTS,
            "ok": bool(np.mean(npts) >= 0.7 * KITTI_POINTS),
            "why": f"the detector was trained on {KITTI_POINTS:,}-point sweeps "
                   f"and samples 256 points per cluster; at a fraction of that "
                   f"density a car at 20 m is a handful of returns, padded out "
                   f"by resampling into something the network never saw. Raise "
                   f"the lidar's channels and points_per_second",
        },
        "intensity": {
            "mean": float(np.mean([i.mean() for i in inten])),
            "saturated_frac": float(np.mean(
                [(i > 0.9).mean() for i in inten])),
            "ok": bool(np.mean([(i > 0.9).mean() for i in inten]) < 0.5),
            "why": "intensity is one of the network's six input channels. "
                   "KITTI reflectance averages 0.29 across the full 0-1 range; "
                   "a CARLA lidar left on its defaults returns ~0.94 for "
                   "everything, so the channel carries no information and does "
                   "not resemble training. Set atmosphere_attenuation_rate and "
                   "the noise/dropoff parameters, or drop the channel",
        },
        "range_noise": noise,
        "tag_points": {TAGS[i]: int(c) for i, c in enumerate(tags) if c},
    }


# The drivability estimator's roughness limit, from terrain_cells. Range noise
# above it cannot be told apart from a road that is actually rough.
MAX_ROUGH_M = 0.04


def range_noise(root: Path, meta: dict, frames: int = 5, radius: float = 8.0):
    """How far the ranging lidar scatters off a surface the simulator made flat.

    The measurement is free and unambiguous because CARLA gives two sensors
    over one scene and applies its noise model to only one of them. The
    semantic sensor returns exact ray hits, so its residual against a fitted
    plane is 0 by construction; anything the ranging sensor adds on the same
    patch of road is the noise model, not the road.

    Restricted to road tags within `radius` so that road CAMBER cannot be
    mistaken for noise -- a junction crowns by tens of centimetres over its
    width, which is larger than the thing being measured. The fitted tilt is
    returned so a patch that was not flat after all is visible rather than
    silently inflating the answer.
    """
    def residual(q):
        if len(q) < 50:
            return None, None
        A = np.c_[np.ones(len(q)), q[:, 0], q[:, 1]]
        c, *_ = np.linalg.lstsq(A, q[:, 2], rcond=None)
        return (float((q[:, 2] - A @ c).std()),
                float(np.degrees(np.arctan(np.hypot(c[1], c[2])))))

    rng, sem, tilt = [], [], []
    for i in list(frame_ids_of(root, meta))[:frames]:
        f = load(root, i, meta)
        s = f.sem[np.isin(f.tag, ROAD_TAGS)]
        s = s[np.hypot(s[:, 0], s[:, 1]) < radius]
        _, road, _ = truth(f)
        r = f.pts[road][:, :3]
        r = r[np.hypot(r[:, 0], r[:, 1]) < radius]
        a, ta = residual(r)
        b, _ = residual(s)
        if a is not None:
            rng.append(a)
            tilt.append(ta)
        if b is not None:
            sem.append(b)
    if not rng:
        return {"ok": True, "note": "no road returns near the sensor to measure"}
    return {
        "ranging_residual_m": float(np.median(rng)),
        "semantic_residual_m": float(np.median(sem)) if sem else None,
        "patch_tilt_deg": float(np.median(tilt)),
        "max_rough_m": MAX_ROUGH_M,
        "ok": bool(np.median(rng) <= MAX_ROUGH_M),
        "why": f"the ranging lidar scatters this far off a plane the semantic "
               f"sensor sees as exactly flat, so it is the sensor's "
               f"noise_stddev and not the road. Above the {MAX_ROUGH_M} m "
               f"roughness limit the drivability estimator cannot tell the "
               f"noise from a rough surface, and a flat road grades marginal "
               f"everywhere. Lower noise_stddev on the lidar blueprint",
    }


def frame_ids_of(root: Path, meta: dict):
    for p in sorted((root / "raw_lidar").glob("lidar_*.npz")):
        i = int(p.stem.split("_")[1])
        if i in meta:
            yield i


# what driving_data.csv writes in distance_to_center_m when the ego is not
# near any depression
SENTINEL = -1.0

# points in a KITTI Velodyne HDL-64E sweep, which is what the detector's
# cluster sizes and its 256-point sampling were tuned against
KITTI_POINTS = 124_600

CHECKS = ("sweeps", "pedestrians", "depressions", "grid_labels", "metadata",
          "density", "intensity", "range_noise")


def main():
    import argparse
    import json
    ap = argparse.ArgumentParser(description="acceptance check for a capture")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    r = check(a.root)
    if a.json:
        print(json.dumps(r, indent=1))
        return 0
    print(f"{r['frames']} scans in {a.root}\n")
    bad = 0
    for k in CHECKS:
        v = r[k]
        bad += not v["ok"]
        detail = ", ".join(
            f"{n}={x:.3f}" if isinstance(x, float) else f"{n}={x}"
            for n, x in v.items() if n not in ("ok", "why"))
        print(f"  {'ok  ' if v['ok'] else 'FAIL'} {k:<12} {detail}")
        if not v["ok"]:
            print(f"       {v['why']}")
    print(f"\n{len(CHECKS) - bad}/{len(CHECKS)} checks pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
