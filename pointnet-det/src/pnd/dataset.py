"""
Dataset over the cached proposals, with canonicalisation applied in the
collate function.

Canonicalisation happens per *batch*, not per sample, so it runs through the
numba kernels in bench_canon.py. Measured on this workload: batched analytic
PCA is 0.098 ms for 64 clusters, versus 1.50 ms for np.linalg.eigh in a Python
loop -- 15x, and the loop version would sit in the DataLoader worker and starve
the GPU.

Channels handed to the network (in_ch = 6):

    0-2  canonicalised xyz, unit sphere
    3    height above ground     <- invariant to yaw, and to whatever the
    4    range from sensor       <- canonicaliser gets wrong
    5    intensity

Channels 3-5 are the insurance policy. If the canonical frame is inconsistent
across viewpoints -- which canon_study.py shows it is -- these still carry
usable signal, because none of them changes when the object rotates about z.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from .bench_canon import pca2_batch, pca3_batch
from .boxes import encode_heading
from .config import Config

# KITTI mean dimensions (l, w, h). Size is regressed as a log-ratio against
# these so a 0.8 m pedestrian and a 3.9 m car produce comparably scaled targets.
# Measured over the training labels at truncation < 0.5. The first three
# reproduce the previous hand-entered values exactly, which is the check that
# the extraction reads the right columns (KITTI stores h w l, not l w h).
#
# The last four have no KITTI labels to average, so they start as estimates and
# are meant to be replaced by measurement -- see set_anchors below. An anchor
# that is wrong by a constant factor does not fail loudly: training regresses
# log(dims / anchor) and inference multiplies by the anchor, so a mismatch
# lands the box with the right centre and the right yaw and the wrong size,
# which scores 0.00 AP while looking correct on a plot. That exact bug cost a
# day here once already, on Van.
ANCHORS = np.array([
    [1.00, 1.00, 1.00],     # background - unused
    [3.89, 1.62, 1.53],     # Car
    [0.84, 0.66, 1.76],     # Pedestrian
    [1.76, 0.60, 1.74],     # Cyclist
    [5.07, 1.90, 2.21],     # Van
    [10.13, 2.59, 3.24],    # Truck
    [0.30, 0.30, 5.00],     # Pole          estimate, replaced by measurement
    [0.40, 0.40, 3.50],     # TrafficLight  estimate
    [0.60, 0.25, 2.50],     # TrafficSign   estimate
    [3.00, 3.00, 3.00],     # Structure     estimate; wall/building/vegetation
], dtype=np.float32)


def set_anchors(a):
    """Replace the anchor table in place, for everything that imported it.

    In place on purpose. `evaluate`, `simulate` and `Collate` all did
    `from .dataset import ANCHORS` and hold a reference to this exact array, so
    rebinding the module name would update the table for nobody. Writing
    through it updates the table for everybody, which is the only way a
    measured anchor set can reach both the training target and the inference
    decode -- and those two disagreeing is precisely the failure described
    above.
    """
    a = np.asarray(a, np.float32)
    if a.shape != ANCHORS.shape:
        raise ValueError(f"anchor table is {ANCHORS.shape}, got {a.shape}; "
                         f"a class list and its anchors must be the same "
                         f"length or the indices mean different things")
    ANCHORS[:] = a


class ProposalSet(Dataset):
    def __init__(self, cfg: Config, split: str = "train"):
        self.cfg = cfg
        shards = sorted(cfg.cache_dir.glob("shard_*.npz"))
        if not shards:
            raise FileNotFoundError(
                f"no shards in {cfg.cache_dir}. Run: python -m pnd.proposals")

        pts, meta, frame = [], [], []
        for s in shards:
            z = np.load(s)
            pts.append(z["points"])
            meta.append(z["meta"])
            frame.append(z["frame"])
        self.points = np.concatenate(pts)
        self.meta = np.concatenate(meta)
        self.frame = np.concatenate(frame)

        # A proposal writer that measured its own anchors ships them beside the
        # shards. Adopt them, so the size target is a ratio against this
        # dataset's real dimensions rather than KITTI's averages or a guess.
        anc = cfg.cache_dir / "anchors.npy"
        if anc.exists():
            set_anchors(np.load(anc))

        # Split by FRAME, never by proposal: two clusters from the same scan
        # share ground plane, weather and pose, so splitting by proposal leaks
        # the validation set into training and flatters the numbers.
        #
        # And by contiguous BLOCKS of frames, not by shuffled frames. That was
        # `rng.shuffle(uf)`, which is right for KITTI -- its frames are sampled
        # from separate drives -- and wrong for a continuous capture. CARLA
        # records at about 0.1 s intervals, so under a metre of ego motion
        # separates one frame from the next and they show the same cars from
        # almost the same viewpoint. Shuffling puts frame 100 in train and 101
        # in validation, and the score that comes back measures memorisation.
        # Whole blocks mean validation is road the model never drove.
        #
        # There is a third split now. With two, the set used to pick best.pt is
        # also the set being reported, and over 80 epochs that choice fits it;
        # `test` is looked at once, at the end.
        uf = np.unique(self.frame)
        nb = max(int(cfg.split_blocks), 3)
        blocks = np.array_split(uf, min(nb, len(uf)))
        rng = np.random.default_rng(cfg.seed)
        order = rng.permutation(len(blocks))

        n_val = max(int(round(len(blocks) * cfg.val_frac)), 1)
        n_test = max(int(round(len(blocks) * cfg.test_frac)), 1) \
            if cfg.test_frac > 0 else 0
        if n_val + n_test >= len(blocks):
            raise ValueError(
                f"{len(blocks)} blocks cannot give {n_val} val + {n_test} "
                f"test and leave any to train on; lower val_frac/test_frac "
                f"or raise split_blocks")
        want = {"val": order[:n_val],
                "test": order[n_val:n_val + n_test],
                "train": order[n_val + n_test:]}
        if split not in want:
            raise ValueError(f"split must be train/val/test, got {split!r}")
        pick = set()
        for b in want[split]:
            pick.update(blocks[b].tolist())
        keep = np.isin(self.frame, list(pick))
        if not keep.any():
            raise ValueError(f"the {split!r} split is empty")

        self.points = self.points[keep]
        self.meta = self.meta[keep]
        self.frame = self.frame[keep]
        self.split = split
        self.n_blocks = len(want[split])
        self.n_frames = len(pick)

    def describe(self) -> str:
        """One line, printed before training rather than discovered after it.

        A split that looks fine in the ratio can still be useless: a rare class
        can land entirely in training, and then its validation F1 is 0.000 for
        the whole run and nothing says why. Worth two seconds to see.
        """
        from .kitti import CLASSES
        c = self.class_counts()
        named = "  ".join(f"{CLASSES[i]} {int(n)}"
                          for i, n in enumerate(c) if n)
        return (f"  {self.split:5} {len(self.meta):7,} proposals  "
                f"{self.n_frames:5,} frames in {self.n_blocks:3} blocks\n"
                f"        {named}")

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, i: int):
        return self.points[i], self.meta[i]

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.meta[:, 0].astype(int),
                           minlength=self.cfg.num_classes)


# --------------------------------------------------------------------------- #
def _rot_z(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3), np.float64)
    R[:, 0, 0] = c; R[:, 0, 1] = -s
    R[:, 1, 0] = s; R[:, 1, 1] = c
    R[:, 2, 2] = 1.0
    return R


class Collate:
    """Batched canonicalisation + feature assembly."""

    def __init__(self, cfg: Config, train: bool):
        self.cfg = cfg
        self.train = train

    def __call__(self, batch):
        pts = np.stack([b[0] for b in batch]).astype(np.float64)   # (B, P, 5)
        meta = np.stack([b[1] for b in batch]).astype(np.float64)  # (B, 9)
        B, P, _ = pts.shape
        cfg = self.cfg

        xyz = pts[:, :, :3].copy()
        inten = pts[:, :, 3]
        agl = pts[:, :, 4]
        rng_raw = np.linalg.norm(xyz, axis=2)          # before any rotation

        ctr_gt = meta[:, 1:4].copy()
        dims_gt = meta[:, 4:7].copy()
        yaw_gt = meta[:, 7].copy()
        cls = meta[:, 0].astype(np.int64)

        # ---- augmentation ---------------------------------------------- #
        if self.train:
            # yaw: rotate about the sensor z axis. Free, exactly matches the
            # real nuisance, and unlike canonicalisation it cannot be
            # inconsistent across viewpoints.
            if cfg.yaw_aug:
                a = np.random.uniform(-np.pi, np.pi, B)
                R = _rot_z(a)
                xyz = np.einsum("bij,bpj->bpi", R, xyz)
                ctr_gt = np.einsum("bij,bj->bi", R, ctr_gt)
                yaw_gt = yaw_gt + a

            # mirror about x. A car seen from the left is a valid car seen from
            # the right; heading negates with it.
            if cfg.aug_flip:
                f = np.random.rand(B) < 0.5
                xyz[f, :, 1] *= -1.0
                ctr_gt[f, 1] *= -1.0
                yaw_gt[f] *= -1.0

            # point dropout: keep a random fraction, then refill to P by
            # repeating what survived. Simulates the same object arriving with
            # far fewer returns because it is further away.
            if cfg.aug_dropout > 0:
                for b in range(B):
                    keep = np.random.uniform(1.0 - cfg.aug_dropout, 1.0)
                    k = max(int(P * keep), 8)
                    idx = np.random.choice(P, k, replace=False)
                    fill = np.random.choice(idx, P, replace=True)
                    xyz[b] = xyz[b][fill]
                    inten[b] = inten[b][fill]
                    agl[b] = agl[b][fill]
                    rng_raw[b] = rng_raw[b][fill]

            # scale: object-size variation the anchors do not cover.
            # Scale about the cluster centroid, NOT the sensor origin. Scaling
            # raw coordinates moves an object at 20 m by 1.6 m at 8% -- a large
            # translation masquerading as a size change, which corrupts the
            # centre target and desynchronises the range/height channels.
            if cfg.aug_scale > 0:
                sc = np.random.uniform(1 - cfg.aug_scale, 1 + cfg.aug_scale, B)
                c0 = xyz.mean(axis=1, keepdims=True)          # (B, 1, 3)
                xyz = c0 + (xyz - c0) * sc[:, None, None]
                ctr_gt = c0[:, 0, :] + (ctr_gt - c0[:, 0, :]) * sc[:, None]
                dims_gt = dims_gt * sc[:, None]

            # jitter: sensor range noise, a couple of centimetres
            if cfg.aug_jitter > 0:
                xyz += np.random.normal(0.0, cfg.aug_jitter, xyz.shape)

        # ---- canonicalise --------------------------------------------- #
        flat = np.ascontiguousarray(xyz.reshape(-1, 3))
        offs = (np.arange(B + 1) * P).astype(np.int64)
        Rc = np.zeros((B, 3, 3)); tc = np.zeros((B, 3)); lam = np.zeros((B, 3))

        mode = cfg.canon
        if mode in ("none", "tnet3"):
            tc = xyz.mean(axis=1)
            Rc = np.repeat(np.eye(3)[None], B, 0)
        elif mode == "pca3_skew":
            pca3_batch(flat, offs, Rc, tc, lam)
        elif mode in ("pca2_yaw", "pca4_ensemble"):
            yaw_c = np.zeros(B)
            pca2_batch(flat, offs, yaw_c)
            tc = xyz.mean(axis=1)
            Rc = _rot_z(-yaw_c)
        else:
            raise ValueError(mode)

        cen = xyz - tc[:, None, :]
        xc = np.einsum("bij,bpj->bpi", Rc, cen)
        scale = np.maximum(np.linalg.norm(xc, axis=2).max(axis=1), 1e-6)
        xc = xc / scale[:, None, None]

        # ---- targets, in the same canonical frame --------------------- #
        dc = np.einsum("bij,bj->bi", Rc, ctr_gt - tc) / scale[:, None]
        # Clip to the anchor table, NOT to a literal. This read `clip(cls, 0, 3)`
        # from when there were exactly four classes, so adding Van (4) and
        # Truck (5) silently folded both onto ANCHORS[3], the Cyclist anchor:
        # training regressed size as log(dims / cyclist) while inference decodes
        # exp(size_log) * ANCHORS[van]. The predicted box then comes out too big
        # by exactly the ratio between the two anchors, which is why Van boxes
        # landed with the centre within 0.2 m and the yaw within a degree and
        # still scored 0.00 AP -- the height ratio of predicted to truth was
        # 1.27, and ANCHORS[Van].h / ANCHORS[Cyclist].h is 2.21/1.74 = 1.27.
        anch = ANCHORS[np.clip(cls, 0, len(ANCHORS) - 1)]
        size_log = np.log(np.maximum(dims_gt, 1e-3) / np.maximum(anch, 1e-3))
        yaw_c_off = np.arctan2(Rc[:, 1, 0], Rc[:, 0, 0])
        yaw_t = yaw_gt + yaw_c_off

        feats = np.concatenate([
            xc,
            (agl / 3.0)[:, :, None],
            (rng_raw / cfg.max_range)[:, :, None],
            inten[:, :, None],
        ], axis=2).transpose(0, 2, 1)                  # (B, 6, P)

        hb, hr = encode_heading(yaw_t)
        out = {
            "x": torch.from_numpy(feats).float(),
            "cls": torch.from_numpy(cls),
            "center": torch.from_numpy(dc).float(),
            "size_log": torch.from_numpy(size_log).float(),
            "head_bin": torch.from_numpy(hb),
            "head_res": torch.from_numpy(hr).float(),
            "dims": torch.from_numpy(dims_gt).float(),
            "anchor": torch.from_numpy(anch.astype(np.float64)).float(),
            "scale": torch.from_numpy(scale).float(),
        }

        if mode == "pca4_ensemble":
            # the four det=+1 sign combinations of the two footprint axes.
            # PCAlign's argument: do not choose a sign, enumerate and let the
            # network pick the frame it is most confident in.
            copies = []
            for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                f = out["x"].clone()
                f[:, 0] *= sx
                f[:, 1] *= sy
                copies.append(f)
            out["x"] = torch.stack(copies, dim=1)      # (B, 4, 6, P)
        return out


def loaders(cfg: Config):
    tr = ProposalSet(cfg, "train")
    va = ProposalSet(cfg, "val")
    from torch.utils.data import DataLoader
    mk = lambda ds, trn: DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=trn, drop_last=trn,
        num_workers=cfg.num_workers, collate_fn=Collate(cfg, trn),
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=cfg.num_workers > 0)

    print(tr.describe())
    print(va.describe())
    if cfg.test_frac > 0:
        # Built and described, then dropped on the floor. It exists so the
        # frames are reserved -- held out of training from the first epoch --
        # and so its class balance is visible now. Scoring it is a separate,
        # deliberate act at the end, not something a training loop does.
        print(ProposalSet(cfg, "test").describe())
        print("  (test is reserved and unused; score it once, at the end)")

    return mk(tr, True), mk(va, False), tr
