"""
per-point class labels from the pointnet detector in MadhankumarAI/trail,
mapped into grid25's class space.

WHAT THIS MODEL ACTUALLY IS
---------------------------
It is a cluster-wise 3D *detector*, not a semantic segmenter. `ClusterNet`
takes 256 points from one cluster and emits a single label over the classes it
was trained on, currently

    [Background, Car, Pedestrian, Cyclist, Van, Truck]

plus a box. It never labels an individual point, and it has no class at all for
road, sidewalk, building, vegetation or pole. The class list is read from the
checkpoint rather than assumed here, so this file does not need editing when
the model gains a class.

The per-point labels grid25 needs come from the three stages around it:

    remove_ground()   ground / not-ground per point   (geometry, no network)
    cluster_points()  cluster id per non-ground point (geometry, no network)
    ClusterNet        one class per cluster           (the network)

So the network contributes exactly one thing: the car / pedestrian / cyclist
split among the clusters. Everything static and non-ground -- buildings,
vegetation, poles, signs -- has no class to go to and lands in `other`.

WHAT THAT COSTS
---------------
Terrain analysis is unaffected: it only ever needed the ground / not-ground
split, and `remove_ground` supplies that directly. Kerbs, potholes, roughness,
slope and clearance all still work.

The semantic layer gets much coarser than SemanticKITTI ground truth. Road and
sidewalk collapse into one `ground`, and building / vegetation / pole collapse
into `other`. That is a real, visible downgrade and it should be presented as
one: a detector with no class for a thing cannot map that thing, whatever its
class count.

Cyclist maps onto `ped` so that it inherits the pedestrian priority override in
grid25.classify(): a vulnerable road user must not be voted away by a
road-dominated cell.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import grid25 as g

# The detector lives in this same project, so it is imported from there
# rather than vendored under trail/.
#
# That copy was a snapshot of this repo taken at some earlier point, and it had
# drifted: four classes where the checkpoint has six, four anchor rows, and an
# old best.pt. A 6-class checkpoint simply could not load against it. Two
# copies of one package in one repository will always drift; one of them is
# never the one being edited.
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'pointnet-det' / 'src'
CKPT = ROOT / 'best_5classes.pt'

# Where each detector class lands in grid25's eight.
#
# Built from a NAME map rather than written out as a positional array. The
# array form was [Background, Car, Pedestrian, Cyclist] and a checkpoint
# trained with more classes emits index 4 or 5 straight off the end of it --
# an IndexError at best, a silently wrong label if the array had happened to
# be longer.
#
# Van and Truck are vehicles and go to `car`. Cyclist goes to `ped` so it
# inherits the pedestrian priority override in grid25.classify(): a vulnerable
# road user must not be voted away by a road-dominated cell.
# The last four are new, and they light up three grid25 classes that nothing
# could previously reach. `bldg`, `pole` and `veg` have existed in grid25 since
# the beginning and were dead: a KITTI-trained detector has no class for a pole
# or a building, so every one of them landed in `other`. That was the honest
# thing to do with a detector that could not see them, and it is no longer
# necessary.
#
# TrafficLight and TrafficSign both go to `pole`, not because they are poles
# but because that is what they are to a planner: a thin vertical obstacle
# with clearance underneath. Structure goes to `bldg` for the same reason --
# `bldg` and `veg` differ in nothing a vehicle cares about, and the class list
# does not distinguish them either.
NAME2GRID = {
    'Background': g.other,
    'Car': g.car,
    'Van': g.car,
    'Truck': g.car,
    'Pedestrian': g.ped,
    'Cyclist': g.ped,
    'Pole': g.pole,
    'TrafficLight': g.pole,
    'TrafficSign': g.pole,
    'Structure': g.bldg,
}

# What the DETECTOR itself did with each point, which grid25's eight classes
# cannot express. The split that matters is the last two: a point in a cluster
# the network examined and rejected is a very different thing from a point the
# network never saw, and both land in `other`.
P_GROUND, P_BG, P_NONE = 0, 1, 2
PROV = ['ground', 'background', 'unclustered']


VULNERABLE = ('Pedestrian', 'Cyclist')   # never voted away by a busy cell


def provenance_names(num_classes):
    """The provenance labels for a model with this many classes.

    Published because the layout is no longer fixed. It used to be six entries
    ending in `unclustered`, and consumers sized arrays and hardcoded indices
    against that; with five object classes it is eight entries and the indices
    moved. Ask for the list rather than assuming it.
    """
    from pnd.kitti import CLASSES
    _, _, names = class_maps(list(CLASSES)[:num_classes])
    return names


def class_maps(names):
    """(det -> grid25, det -> provenance, provenance names) for a class list.

    Derived from the checkpoint's own class names, so the maps grow with the
    model instead of having to be edited in step with it.
    """
    prov_names = list(PROV)
    det2prov = np.empty(len(names), np.int64)
    det2grid = np.empty(len(names), np.int64)
    for i, n in enumerate(names):
        if n not in NAME2GRID:
            raise KeyError(f"no grid25 class for detector class {n!r}; "
                           f"add it to NAME2GRID")
        det2grid[i] = NAME2GRID[n]
        if n == 'Background':
            det2prov[i] = P_BG
        else:
            prov_names.append(n.lower())
            det2prov[i] = len(prov_names) - 1
    return det2grid, det2prov, prov_names


def _import():
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from pnd.config import Config
    from pnd.ground import remove_ground
    from pnd.cluster import cluster_points
    from pnd.bench_canon import pca2_batch
    from pnd.model import build
    return Config, remove_ground, cluster_points, pca2_batch, build


def best_device():
    """'cuda' when a GPU is actually usable, else 'cpu'.

    The default stays 'cpu' so nothing changes for a caller that does not ask.
    But every caller here wants the GPU when there is one, and the network is
    the only part of this pipeline that benefits: on this machine the forward
    pass is 188 ms on the CPU against about 20 on the GPU, which is most of the
    per-frame cost.
    """
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


def load(ckpt=CKPT, device='cpu'):
    """Load a checkpoint and rebuild the network it was trained as."""
    import torch
    Config, _, _, _, build = _import()
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    saved = ck.get('cfg', {}) or {}
    cfg = Config()
    for k, v in saved.items():
        if hasattr(cfg, k) and k not in ('device', 'data_root', 'cache_dir', 'run_dir'):
            setattr(cfg, k, v)
    cfg.device = device
    model = build(cfg)
    model.load_state_dict(ck['model'])
    # Same reason as the class list: the anchor table belongs to the
    # checkpoint, not to whatever the module default happens to be. Decoding a
    # CARLA-trained model against KITTI averages scales every box by the ratio
    # between the two tables and reports it as a bad model.
    if 'anchors' in ck:
        sys.path.insert(0, str(SRC)) if str(SRC) not in sys.path else None
        from pnd.dataset import set_anchors
        set_anchors(ck['anchors'])
    # .to(device) was missing, so `device` was accepted and ignored: the model
    # stayed on the CPU whatever was asked for, and the forward pass cost
    # 188 ms instead of about 20. cpu and cuda timed identically, which is the
    # tell -- a device argument that changes nothing is not being applied.
    model.to(device)
    model.eval()
    return model, cfg


def _features(xyz, inten, agl, maxrange, pca2_batch):
    """
    exactly Collate.__call__ with train=False: centre each cluster, remove its
    yaw from a 2D pca of the footprint, scale to the unit sphere, then append
    the three rotation-invariant channels. any deviation here silently feeds
    the network something it was never trained on.
    """
    B, P = xyz.shape[:2]
    rng_raw = np.linalg.norm(xyz, axis=2)          # range BEFORE any rotation
    yaw = np.zeros(B)
    pca2_batch(np.ascontiguousarray(xyz.reshape(-1, 3)),
               (np.arange(B + 1) * P).astype(np.int64), yaw)
    c, s = np.cos(-yaw), np.sin(-yaw)
    Rc = np.zeros((B, 3, 3))
    Rc[:, 0, 0] = c; Rc[:, 0, 1] = -s
    Rc[:, 1, 0] = s; Rc[:, 1, 1] = c
    Rc[:, 2, 2] = 1.0
    tc = xyz.mean(axis=1)
    xc = np.einsum('bij,bpj->bpi', Rc, xyz - tc[:, None, :])
    scale = np.maximum(np.linalg.norm(xc, axis=2).max(axis=1), 1e-6)
    xc = xc / scale[:, None, None]
    return np.concatenate([xc,
                           (agl / 3.0)[:, :, None],
                           (rng_raw / maxrange)[:, :, None],
                           inten[:, :, None]], axis=2).transpose(0, 2, 1)


def predict(pts4, model, cfg, batch=256, seed=0, with_prov=False):
    """
    pts4: (N, 4) velodyne x y z intensity, sensor at the origin.
    returns (labels in grid25 classes, info dict), and with with_prov=True a
    third array saying what the detector actually did with each point (PROV).
    """
    import torch
    _, remove_ground, cluster_points, pca2_batch, _ = _import()
    sys.path.insert(0, str(SRC)) if str(SRC) not in sys.path else None
    from pnd.kitti import CLASSES
    names = list(CLASSES)[:cfg.num_classes]
    det2grid, det2prov, prov_names = class_maps(names)

    rng = np.random.default_rng(seed)
    N = len(pts4)
    lab = np.full(N, g.other, np.int64)

    prov = np.full(N, P_NONE, np.int64)
    is_ground, agl, _ = remove_ground(pts4[:, :3], thresh=cfg.ground_thresh)
    lab[is_ground] = g.gnd
    prov[is_ground] = P_GROUND

    fg = np.flatnonzero(~is_ground)
    if len(fg) < cfg.min_cluster_pts:
        empty = dict(ground=int(is_ground.sum()), clusters=0, counts={})
        return (lab, empty, prov) if with_prov else (lab, empty)

    obj = pts4[fg]
    cl = cluster_points(obj[:, :3], voxel=cfg.cluster_voxel,
                        min_points=cfg.min_cluster_pts,
                        max_points=cfg.max_cluster_pts)
    ncl = int(cl.max()) + 1
    if ncl <= 0:
        empty = dict(ground=int(is_ground.sum()), clusters=0, counts={})
        return (lab, empty, prov) if with_prov else (lab, empty)

    # gather cfg.n_points per cluster, padding by resampling as training did
    order = np.argsort(cl, kind='stable')
    cs = cl[order]
    first = np.searchsorted(cs, np.arange(ncl))
    last = np.searchsorted(cs, np.arange(ncl), side='right')
    P = cfg.n_points
    sel = np.empty((ncl, P), np.int64)
    for k in range(ncl):
        idx = order[first[k]:last[k]]
        sel[k] = (rng.choice(idx, P, replace=False) if len(idx) >= P
                  else rng.choice(idx, P, replace=True))

    xyz = obj[sel][:, :, :3].astype(np.float64)
    inten = obj[sel][:, :, 3].astype(np.float64)
    a = agl[fg][sel].astype(np.float64)

    pred = np.empty(ncl, np.int64)
    conf = np.empty(ncl)
    with torch.no_grad():
        for i in range(0, ncl, batch):
            f = _features(xyz[i:i+batch], inten[i:i+batch], a[i:i+batch],
                          cfg.max_range, pca2_batch)
            x = torch.from_numpy(f).float().to(cfg.device)
            p = model(x)['logits'].softmax(1).cpu().numpy()
            pred[i:i+batch] = p.argmax(1)
            conf[i:i+batch] = p.max(1)

    hit = cl >= 0
    lab[fg[hit]] = det2grid[pred[cl[hit]]]
    prov[fg[hit]] = det2prov[pred[cl[hit]]]

    out = dict(
        ground=int(is_ground.sum()),
        clusters=ncl,
        clustered_points=int(hit.sum()),
        unclustered=int((~hit).sum()),
        counts={n: int((pred == i).sum()) for i, n in enumerate(names)},
        provcount={n: int((prov == i).sum()) for i, n in enumerate(prov_names)},
        mean_conf=float(conf.mean()))
    return (lab, out, prov) if with_prov else (lab, out)


if __name__ == '__main__':
    import time, kitti
    src = sys.argv[1] if len(sys.argv) > 1 else 'kitti/000000.bin'
    pts4 = np.fromfile(src, np.float32).reshape(-1, 4)
    model, cfg = load()
    print(f'model  canon={cfg.canon}  classes={cfg.num_classes}  '
          f'in_ch={cfg.in_ch}  params={sum(p.numel() for p in model.parameters()):,}')
    t0 = time.perf_counter()
    lab, info = predict(pts4, model, cfg)
    ms = (time.perf_counter() - t0) * 1000
    print(f'{len(pts4):,} points in {ms:.0f} ms')
    for k, v in info.items():
        print(f'  {k}: {v}')
    names = ['ground', 'road', 'building', 'pole', 'vegetation', 'car', 'ped', 'other']
    print('  per-point labels:', {names[i]: int(c)
                                  for i, c in enumerate(np.bincount(lab, minlength=8)) if c})
