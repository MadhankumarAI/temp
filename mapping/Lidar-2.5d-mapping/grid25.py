"""
adaptive variable-resolution 2.5d grid builder.

pure numpy, no ros. input is (x,y,z) points plus a per-point class label,
output is a sparse set of cells whose size grows with range from the sensor.

design notes:
  - every point is quantised at the FINEST resolution first, with no range
    logic at all. tier selection happens afterwards, per cell, from the
    cell centre. this is what prevents cells straddling a tier boundary and
    double-counting points.
  - tiers are powers of two so a parent index is just a right shift, and
    the merge is exact.
  - all per-cell accumulators are mergeable under +, min or max. that is
    why we store zsum/zsq instead of mean/variance: sums merge, means do not.
"""

import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------- config

res0 = 0.05                      # finest cell size, metres
bounds = (10.0, 25.0, 50.0)      # tier boundaries -> levels 0,1,2,3
maxrange = 100.0

nclass = 8
gnd, road, bldg, pole, veg, car, ped, other = range(nclass)

groundcls = (gnd, road)          # classes that define the terrain surface
critcls = (ped,)                 # classes that override a majority vote
critmin = 3                      # this many points is enough to claim a cell

gres = 0.25                      # ground raster resolution, metres
gpct = 20                        # percentile of z used as ground height
gnear = 2.0                      # max distance to a real ground return, metres

naz = 2048                       # azimuth bins -- set to the sensor's own
                                 # azimuth resolution, no finer
drb = 0.5                        # range bin of the visibility raster, metres
rgap = 0.5                       # a return must be at least this far beyond a
                                 # cell to count as having passed over it
nrb = int(maxrange / drb) + 2    # last bin left empty so 'beyond' always exists


# ------------------------------------------------------------- internals

def _group(key):
    """sort by key, return the permutation, group starts and group sizes."""
    o = np.argsort(key, kind='stable')
    k = key[o]
    st = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    cnt = np.diff(np.r_[st, len(k)])
    return o, st, cnt


def _pack(ix, iy):
    return (ix.astype(np.int64) << 32) | (iy.astype(np.int64) & 0xffffffff)


# ------------------------------------------------------------ pass 1

def quantise(x, y, z, lab, res=res0):
    """bin every point into a fine cell. no range logic here on purpose."""
    ix = np.floor(x / res).astype(np.int64)
    iy = np.floor(y / res).astype(np.int64)
    o, st, cnt = _group(_pack(ix, iy))

    zs, ls = z[o], lab[o]
    isg = np.isin(ls, groundcls)
    zo = np.where(isg, np.inf, zs)          # non-ground only, for clearance
    zg = np.where(isg, zs, 0.0)             # ground only, for terrain stats
    gw = isg.astype(np.float64)
    cid = np.repeat(np.arange(len(st)), cnt)

    return dict(
        ix=ix[o][st],
        iy=iy[o][st],
        n=cnt.astype(np.int64),
        zmin=np.minimum.reduceat(zs, st),
        zmax=np.maximum.reduceat(zs, st),
        zomin=np.minimum.reduceat(zo, st),
        zsum=np.add.reduceat(zs, st),
        zsq=np.add.reduceat(zs * zs, st),
        ng=np.add.reduceat(gw, st),
        gmin=np.minimum.reduceat(np.where(isg, zs, np.inf), st),
        gsum=np.add.reduceat(zg, st),
        gsq=np.add.reduceat(zg * zg, st),
        hist=np.bincount(cid * nclass + ls,
                         minlength=len(st) * nclass).reshape(-1, nclass),
    )


# ------------------------------------------------------------ pass 2

def merge(c, key):
    """
    merge cells sharing a key. identical maths to pass 1 because every
    accumulator is associative -- this is the point of the design.
    """
    o, st, cnt = _group(key)
    return dict(
        n=np.add.reduceat(c['n'][o], st),
        zmin=np.minimum.reduceat(c['zmin'][o], st),
        zmax=np.maximum.reduceat(c['zmax'][o], st),
        zomin=np.minimum.reduceat(c['zomin'][o], st),
        zsum=np.add.reduceat(c['zsum'][o], st),
        zsq=np.add.reduceat(c['zsq'][o], st),
        ng=np.add.reduceat(c['ng'][o], st),
        gmin=np.minimum.reduceat(c['gmin'][o], st),
        gsum=np.add.reduceat(c['gsum'][o], st),
        gsq=np.add.reduceat(c['gsq'][o], st),
        hist=np.add.reduceat(c['hist'][o], st, axis=0),
    ), o, st


def blocklevel(ix, iy):
    """
    the tier a fine cell belongs to, decided from the BLOCK it would merge
    into rather than from the fine cell itself.

    a block is coarsened only if the whole of it lies beyond the boundary. if
    any part reaches into the finer ring the block stays fine -- resolution
    never degrades across a boundary, which is the safe direction.

    deciding per fine cell instead lets one block split across two tiers: the
    children inside the ring stay fine and the ones outside merge into a
    parent whose footprint CONTAINS them but which holds none of their points.
    that is 82 overlapping footprints on kitti seq00 frame 0. points are still
    partitioned correctly either way -- the overlap is between footprints, so
    a planner asking what is at (x, y) gets two cells and no rule to pick.
    """
    lvl = np.zeros(len(ix), np.int64)
    for L in (3, 2, 1):
        res = res0 * (1 << L)
        ax, bx = (ix >> L) * res, ((ix >> L) + 1) * res
        ay, by = (iy >> L) * res, ((iy >> L) + 1) * res
        # nearest point of the block to the sensor, 0 on an axis it spans
        nx = np.where(ax > 0, ax, np.where(bx < 0, bx, 0.0))
        ny = np.where(ay > 0, ay, np.where(by < 0, by, 0.0))
        lvl = np.where((lvl == 0) & (np.hypot(nx, ny) >= bounds[L - 1]), L, lvl)
    return lvl


def foveate(c):
    """give each fine cell a level from its block, then merge up."""
    lvl = blocklevel(c['ix'], c['iy'])

    # arithmetic shift on signed ints is floor-division by 2**lvl,
    # which is exactly the parent index and works for negatives too
    px = c['ix'] >> lvl
    py = c['iy'] >> lvl

    key = (lvl << 62) | ((px & 0x7fffffff) << 31) | (py & 0x7fffffff)
    m, o, st = merge(c, key)
    m['lvl'] = lvl[o][st]
    m['ix'] = px[o][st]
    m['iy'] = py[o][st]
    m['res'] = res0 * (2.0 ** m['lvl'])
    m['cx'] = (m['ix'] + 0.5) * m['res']
    m['cy'] = (m['iy'] + 0.5) * m['res']
    return m


# ------------------------------------------------------- ground surface

def groundmap(x, y, z, lab, res=gres, pct=gpct):
    """
    coarse raster of terrain height built from ground-labelled points only.
    a low percentile of z is used rather than the minimum so that a single
    stray return below the surface cannot drag a cell down.
    returns (raster, ox, oy) where raster[j,i] is height at cell (i,j).
    """
    m = np.isin(lab, groundcls)
    gx, gy, gz = x[m], y[m], z[m]

    ix = np.floor(gx / res).astype(np.int64)
    iy = np.floor(gy / res).astype(np.int64)
    ox, oy = ix.min(), iy.min()
    w, h = ix.max() - ox + 1, iy.max() - oy + 1

    key = _pack(ix, iy)
    o = np.lexsort((gz, key))                 # sort by cell, then by z
    k, zs = key[o], gz[o]
    st = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    cnt = np.diff(np.r_[st, len(k)])

    pick = st + (cnt * pct) // 100            # percentile within each cell
    val = zs[pick]

    r = np.full((h, w), np.nan)
    r[iy[o][st] - oy, ix[o][st] - ox] = val
    dist = ndimage.distance_transform_edt(np.isnan(r)) * res
    return _fill(r), dist, ox, oy


def _fill(r):
    """
    fill unobserved cells from the nearest observed one, in a single pass.
    the distance transform gives us, for every empty cell, the index of its
    closest filled neighbour -- far cheaper than iterating a flood fill.
    """
    bad = np.isnan(r)
    if bad.all():
        return np.zeros_like(r)
    if bad.any():
        _, idx = ndimage.distance_transform_edt(bad, return_indices=True)
        r = r[tuple(idx)]
    # median, not mean: a box blur smears a kerb across its whole width,
    # a median filter removes noise while keeping the step edge intact
    return ndimage.median_filter(r, 3, mode='nearest')


def surfaces(r, stepwin=1.25, trendwin=4.0, res=gres):
    """
    two derived surfaces, because a kerb and a pothole are different animals.

    step  : rise above the local neighbourhood, window ~1m. a kerb is a
            *discontinuity* in the surface, not a deviation from it -- the
            ground raster climbs the kerb with it, so measuring against a
            smoothed surface reads zero. measure against the local minimum.

    trend : the large-scale drivable surface, window ~4m, median so that
            small depressions do not pull it down. a pothole is a deviation
            from THIS, and only shows if the reference is coarse enough not
            to follow the hole.
    """
    sw = max(3, int(round(stepwin / res)) | 1)
    tw = max(3, int(round(trendwin / res)) | 1)
    step = r - ndimage.minimum_filter(r, sw, mode='nearest')
    trend = ndimage.uniform_filter(r, tw, mode='nearest')
    return step, trend


def sample(r, ox, oy, cx, cy, res=gres):
    """bilinear lookup of ground height at arbitrary cell centres."""
    h, w = r.shape
    fx = np.clip(cx / res - ox, 0, w - 1.001)
    fy = np.clip(cy / res - oy, 0, h - 1.001)
    x0, y0 = fx.astype(int), fy.astype(int)
    tx, ty = fx - x0, fy - y0
    return (r[y0, x0] * (1 - tx) * (1 - ty) + r[y0, x0 + 1] * tx * (1 - ty) +
            r[y0 + 1, x0] * (1 - tx) * ty + r[y0 + 1, x0 + 1] * tx * ty)


# ---------------------------------------------------------- free space

def visibility(x, y, z):
    """
    lowest height at which any ray passes over each column, without casting
    a single ray.

    a return at (range R, height Z) means the sensor saw clean through every
    column in front of it at that azimuth: at range r < R the ray was at
    height Z*r/R. so a whole ray is one number, its elevation tangent
    t = Z/R, and the lowest ray over a column is the smallest t among the
    returns BEYOND it, times that column's range.

    that turns ray casting into a suffix minimum over range within each
    azimuth bin -- one sort, no stepping, no ragged per-ray arrays.
    """
    r = np.hypot(x, y)
    k = r > 1e-3
    r, t = r[k], z[k] / r[k]
    ab = np.clip(((np.arctan2(y[k], x[k]) + np.pi) *
                  (naz / (2 * np.pi))).astype(np.int64), 0, naz - 1)
    rb = np.minimum((r / drb).astype(np.int64), nrb - 1)

    key = ab * nrb + rb
    o, st, cnt = _group(key)
    v = np.full(naz * nrb, np.inf)
    v[key[o][st]] = np.minimum.reduceat(t[o], st)
    v = v.reshape(naz, nrb)

    # deliberately no smoothing across azimuth. borrowing the neighbouring
    # ray is what lets a wall seen at grazing incidence look see-through:
    # that ray passes BESIDE the wall, not through it. an azimuth with no
    # return beyond simply stays unseen, which is the safe answer.
    return np.minimum.accumulate(v[:, ::-1], axis=1)[:, ::-1]


def raylow(vis, cx, cy):
    """
    height of the lowest ray that passed over each cell and carried on.

    the rgap standoff is what makes this hold at grazing incidence. a wall
    seen almost edge-on crosses a whole range of ranges inside one azimuth
    bin, so its own returns appear to lie 'beyond' the cell they are in.
    they are not evidence of anything: skip the first half metre.
    """
    r = np.hypot(cx, cy)
    ab = np.clip(((np.arctan2(cy, cx) + np.pi) *
                  (naz / (2 * np.pi))).astype(np.int64), 0, naz - 1)
    rb = np.minimum(((r + rgap) / drb).astype(np.int64) + 1, nrb - 1)
    t = vis[ab, rb]
    return np.where(np.isfinite(t), t * r, np.inf)


# ------------------------------------------------------- derived fields

def classify(hist):
    """
    majority label, but with a priority override. a handful of pedestrian
    returns must not be voted away by a road-dominated cell -- that is the
    one place in this pipeline where an averaging bug becomes a safety bug.
    """
    cls = hist.argmax(1).astype(np.int64)
    crit = hist[:, list(critcls)].sum(1)
    cls[crit >= critmin] = critcls[0]
    return cls


def derive(m, r, dist, ox, oy, vis):
    stepr, trendr = surfaces(r)
    m['zg'] = sample(r, ox, oy, m['cx'], m['cy'])
    m['gdist'] = sample(dist, ox, oy, m['cx'], m['cy'])   # m to nearest real return
    m['step'] = sample(stepr, ox, oy, m['cx'], m['cy'])       # kerbs, steps up
    m['trend'] = sample(trendr, ox, oy, m['cx'], m['cy'])     # drivable surface
    # roughness is a property of the TERRAIN, not of everything in the
    # column. computed over all points, a cell under an overhang mixes
    # ground at 0.3m with steel at 4.3m and reads as impassably rough.
    ng = np.maximum(m['ng'], 1.0)
    gm = m['gsum'] / ng
    m['rough'] = np.where(m['ng'] > 2,
                          np.sqrt(np.maximum(m['gsq'] / ng - gm * gm, 0.0)),
                          0.0)
    m['cls'] = classify(m['hist'])
    m['zray'] = raylow(vis, m['cx'], m['cy'])

    gmin = np.where(np.isfinite(m['gmin']), m['gmin'], m['zg'])
    m['dip'] = gmin - m['trend']           # negative = pothole / depression
    m['obsh'] = m['zmax'] - m['zg']        # obstacle height above ground
    m['clear'] = np.where(np.isfinite(m['zomin']),
                          m['zomin'] - m['zg'], np.inf)
    return m


def traversable(m, vh=2.2, maxstep=0.12, maxdip=0.10, maxrough=0.08):
    """
    geometric drivability, deliberately independent of the semantic label
    so that an unlabelled hazard the network never saw still gets caught.
    """
    # an overhang may only be declared where the ground surface is actually
    # known. per-cell ground returns are too strict -- at 30m a 20cm cell
    # often catches the obstacle and no ground beneath it purely from
    # sampling density -- so test against distance to the nearest real
    # ground return instead. unknown space is never traversable.
    known = m['gdist'] < gnear
    # an overhang claim also needs positive evidence that the space beneath
    # it was swept. absence of a return below vh is not clearance -- a cell
    # holding the upper half of a wall has no low returns either, and before
    # this test a quarter of the wall cells read as drivable.
    swept = (m['zray'] - m['zg']) < vh
    low = (m['clear'] > vh) & known & swept
    solid = (m['obsh'] > maxstep) & ~low
    return (known & ~solid & (m['step'] < maxstep) & (m['dip'] > -maxdip) &
            (m['rough'] < maxrough))


# ---------------------------------------------------------------- entry

def build(pts, lab):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    keep = np.hypot(x, y) < maxrange
    x, y, z, lab = x[keep], y[keep], z[keep], lab[keep].astype(np.int64)

    fine = quantise(x, y, z, lab)
    m = foveate(fine)
    r, dist, ox, oy = groundmap(x, y, z, lab)
    vis = visibility(x, y, z)
    m = derive(m, r, dist, ox, oy, vis)
    m['trav'] = traversable(m)
    m['nfine'] = len(fine['n'])
    return m


def memstats(m, area=None):
    """cells actually stored vs a uniform fine grid over the same footprint."""
    if area is None:
        area = np.pi * maxrange ** 2
    uni = area / (res0 ** 2)
    return dict(cells=len(m['n']), fine=m['nfine'],
                uniform=int(uni), ratio=uni / len(m['n']))
