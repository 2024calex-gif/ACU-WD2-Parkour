"""Root offset solver: a smooth per-frame translation of the body that meets contacts.

The game keeps its own root displacement (TrackID 0) and Reference position track. When WD2 rotations are
put on an ACU clip, the body shape changes, so hands may float above or sink into the obstacle and feet may
float or sink. This module solves a smooth offset Delta(t) that is added to the original Reference position
(see apply_offset) so that the contacts are met.

Conventions (same as the rest of the pipeline):
  * Entity / world space: +X lateral, +Y forward (movement direction), +Z up. Lengths in metres, times in frames.
  * P is [F, K, 3]: world positions of K tracked body points per frame, WITHOUT the offset.
  * Delta is [F, 3]. X is always 0. The body point k at frame f is at P[f, k] + Delta[f].
  * Constraint frame arguments are indices 0 .. F-1 into the first axis of P (not frame numbers).
    contact_constraints_from_events() converts frame numbers to indices.

Objective (weighted least squares, solved with numpy.linalg.lstsq):
  sum  weight^2 * residual^2 over Height and Pin rows
  + lam_smooth * ||second difference of Delta||^2          (both Y and Z)
  + lam_small_Y * ||Delta_Y||^2 + lam_small_Z * ||Delta_Z||^2

Above constraints are inequalities (point z >= surface height at the point's y, i.e. no penetration). They are
solved as an active set: solve, find (point, frame) pairs below the surface by more than `tol`, freeze each new
pair's target at surface + clearance, and re-solve. For each frame the translation must satisfy the most
demanding active pair, so one equality per frame is used (the max over that frame's active pairs).
"""
from dataclasses import dataclass

import numpy as np

LAM_SMOOTH = 50.0          # second-difference penalty (both axes)
LAM_SMALL = (5.0, 0.1)     # (Y, Z) penalty on the offset itself
WEIGHT = 1000.0            # default weight of Height and Pin rows (residuals in m)
INEQ_WEIGHT = 1000.0       # weight of the active-set equalities of Above constraints
CLEARANCE = 0.01           # m, target height above a surface for a point pushed out of it
TOL = 1e-3                 # m, penetration below this is not a violation
MAX_ITER = 20

# Height of the joint centre above the skin contact (joints sit above the surface they touch).
HAND_OFFSET = 0.03
FOOT_OFFSET = 0.08
TOE_OFFSET = 0.03        # LeftFootThumb1 / RightFootThumb1
HEIGHT_OFFSET = (('FootThumb1', TOE_OFFSET), ('Hand', HAND_OFFSET), ('Foot', FOOT_OFFSET))


@dataclass
class Height:
    """Point `point` should be at world height z on the frames (indices) `frames`."""
    point: int
    frames: object
    z: float
    weight: float = WEIGHT


@dataclass
class Above:
    """Points `point` (int or 'all') must stay on or above `surface` on the frames `frames`.

    surface: None for ground (z = 0 everywhere), a Box, or any callable y_array -> z_array.
    """
    point: object
    frames: object
    surface: object = None


@dataclass
class Pin:
    """Delta must equal `value` on the frames. value: a scalar (both Y and Z) or a (dy, dz) pair."""
    frames: object
    value: object = 0.0
    weight: float = WEIGHT


@dataclass(frozen=True)
class Box:
    """Obstacle top: z = height for near_y <= y <= far_y, ground (z = 0) elsewhere. Callable on y arrays."""
    height: float
    near_y: float
    far_y: float

    def __call__(self, y):
        y = np.asarray(y, float)
        return np.where((y >= self.near_y) & (y <= self.far_y), self.height, 0.0)


def _indices(frames, F):
    idx = np.asarray(frames, dtype=int).ravel()
    if idx.size and (idx.min() < 0 or idx.max() >= F):
        raise ValueError('frame index outside 0..%d' % (F - 1))
    return idx


def _surface_z(surface, y):
    y = np.asarray(y, float)
    if surface is None:
        return np.zeros_like(y)
    return np.asarray(surface(y), float)


def _split(constraints):
    heights, aboves, pins = [], [], []
    for c in constraints:
        if isinstance(c, Height):
            heights.append(c)
        elif isinstance(c, Above):
            aboves.append(c)
        elif isinstance(c, Pin):
            pins.append(c)
        else:
            raise TypeError('unknown constraint %r' % (c,))
    return heights, aboves, pins


def solve_offset(P, constraints, lam_smooth=LAM_SMOOTH, lam_small=LAM_SMALL, ineq_weight=INEQ_WEIGHT,
                 clearance=CLEARANCE, tol=TOL, max_iter=MAX_ITER):
    """Solve the offset Delta [F, 3] (X = 0) for the body points P [F, K, 3].

    Returns (delta, info). info keys:
      iterations        number of linear solves (<= max_iter)
      max_residual      {'Height': m, 'Pin': m, 'Above_eq': m}  largest residual per constraint type
                        (Above_eq: how far an active pair is still BELOW its frozen target; above is fine)
      max_penetration   m, largest amount any Above pair is below its surface after solving (0 when none)
      n_violations      number of Above pairs still below the surface by more than tol
      converged         True when n_violations == 0
      active            sorted list of (point, frame) pairs that became equality constraints
    """
    P = np.asarray(P, float)
    if P.ndim != 3 or P.shape[2] != 3:
        raise ValueError('P must have shape [F, K, 3]')
    F, K = P.shape[0], P.shape[1]
    if F < 3:
        raise ValueError('need at least 3 frames')
    heights, aboves, pins = _split(constraints)
    n = 2 * F                       # unknowns: Y at 0..F-1, Z at F..2F-1
    lam_y, lam_z = lam_small

    # Static rows: smoothness, small-offset penalty, Height, Pin.
    d2 = np.zeros((F - 2, F))
    for i in range(F - 2):
        d2[i, i:i + 3] = (1.0, -2.0, 1.0)
    sw = np.sqrt(lam_smooth)
    a_smooth = np.zeros((2 * (F - 2), n))
    a_smooth[:F - 2, :F] = sw * d2
    a_smooth[F - 2:, F:] = sw * d2
    a_small = np.diag(np.concatenate([np.full(F, np.sqrt(lam_y)), np.full(F, np.sqrt(lam_z))]))
    b_small = np.zeros(n)

    a_rows, b_rows = [a_smooth, a_small], [np.zeros(2 * (F - 2)), b_small]
    for c in heights:
        idx = _indices(c.frames, F)
        if idx.size == 0:
            continue
        a = np.zeros((idx.size, n))
        a[np.arange(idx.size), F + idx] = c.weight
        a_rows.append(a)
        b_rows.append(c.weight * (c.z - P[idx, c.point, 2]))
    for c in pins:
        idx = _indices(c.frames, F)
        if idx.size == 0:
            continue
        vy, vz = (c.value, c.value) if np.ndim(c.value) == 0 else (c.value[0], c.value[1])
        ay = np.zeros((idx.size, n))
        ay[np.arange(idx.size), idx] = c.weight
        az = np.zeros((idx.size, n))
        az[np.arange(idx.size), F + idx] = c.weight
        a_rows += [ay, az]
        b_rows += [np.full(idx.size, c.weight * float(vy)), np.full(idx.size, c.weight * float(vz))]
    a_static = np.vstack(a_rows)
    b_static = np.concatenate(b_rows)

    # Above pairs: (point, frame) -> frozen target height of that point.
    above_pairs = []
    for c in aboves:
        ks = range(K) if c.point == 'all' else [int(c.point)]
        for f in _indices(c.frames, F):
            for k in ks:
                above_pairs.append((k, int(f), c))
    active = {}

    def solve(active_targets):
        rows_a, rows_b = [a_static], [b_static]
        per_frame = {}
        for (k, f), t in active_targets.items():
            need = t - P[f, k, 2]
            per_frame[f] = max(per_frame.get(f, -np.inf), need)
        if per_frame:
            fr = np.array(sorted(per_frame), dtype=int)
            a = np.zeros((fr.size, n))
            a[np.arange(fr.size), F + fr] = ineq_weight
            rows_a.append(a)
            rows_b.append(ineq_weight * np.array([per_frame[f] for f in fr]))
        x = np.linalg.lstsq(np.vstack(rows_a), np.concatenate(rows_b), rcond=None)[0]
        delta = np.zeros((F, 3))
        delta[:, 1] = x[:F]
        delta[:, 2] = x[F:]
        return delta

    def penetrations(delta):
        out = []
        for k, f, c in above_pairs:
            yw = P[f, k, 1] + delta[f, 1]
            zw = P[f, k, 2] + delta[f, 2]
            s = float(_surface_z(c.surface, yw))
            out.append((s - zw, k, f, s))
        return out

    n_solves = 0
    delta = solve(active)
    n_solves = 1
    for it in range(max_iter):
        if it == max_iter - 1:
            break
        new = {}
        for pen, k, f, s in penetrations(delta):
            if pen > tol and (k, f) not in active and (k, f) not in new:
                new[(k, f)] = s + clearance
        if not new:
            break
        active.update(new)
        delta = solve(active)
        n_solves += 1

    pens = penetrations(delta)
    max_pen = max([0.0] + [p for p, _k, _f, _s in pens])
    n_viol = sum(1 for p, _k, _f, _s in pens if p > tol)

    max_h = 0.0
    for c in heights:
        idx = _indices(c.frames, F)
        if idx.size:
            max_h = max(max_h, float(np.max(np.abs(P[idx, c.point, 2] + delta[idx, 2] - c.z))))
    max_p = 0.0
    for c in pins:
        idx = _indices(c.frames, F)
        if idx.size:
            vy, vz = (c.value, c.value) if np.ndim(c.value) == 0 else (c.value[0], c.value[1])
            max_p = max(max_p, float(np.max(np.abs(delta[idx, 1] - vy))),
                        float(np.max(np.abs(delta[idx, 2] - vz))))
    # Active pairs are inequalities: only "below target" counts. A point above its target is fine.
    max_eq = 0.0
    for (k, f), t in active.items():
        max_eq = max(max_eq, float(t - (P[f, k, 2] + delta[f, 2])))

    info = dict(iterations=n_solves,
                max_residual={'Height': max_h, 'Pin': max_p, 'Above_eq': max_eq},
                max_penetration=max_pen,
                n_violations=int(n_viol),
                converged=bool(n_viol == 0),
                active=sorted(active.keys()))
    return delta, info


def apply_offset(ref_pos_track, delta):
    """New Reference position track: original [F, 3] plus the offset [F, 3]."""
    ref = np.asarray(ref_pos_track, float)
    d = np.asarray(delta, float)
    if ref.shape != d.shape:
        raise ValueError('shape mismatch: %s vs %s' % (ref.shape, d.shape))
    return ref + d


def _height_offset(name):
    for key, off in HEIGHT_OFFSET:
        if name.endswith(key):
            return off
    raise ValueError('no height offset for effector %r' % name)


def contact_constraints_from_events(obstacle, frames, contacts, point_index, pin_first=3, pin_last=6,
                                    weight=WEIGHT):
    """Constraints for a clip from its WD2 (or ACU) contact intervals, in world space.

    obstacle     dict with height, near_y, far_y (acuclip.obstacle(dump), world space)
    frames       frame numbers of the P rows (seq.frames)
    contacts     {effector name: [(f0, f1, 'ground' | 'top'), ...]} with frame numbers (contacts.events()['contacts'] works)
    point_index  {effector name: index of that point in P}

    Returns a list: Height per contact interval (z = surface + joint offset, ground surface 0, top surface
    obstacle height), one Above('all') over the obstacle box, and Pins (value 0) at the first `pin_first` and
    last `pin_last` frames.
    """
    frames = np.asarray(frames)
    F = len(frames)
    out = []
    box = Box(float(obstacle['height']), float(obstacle['near_y']), float(obstacle['far_y']))
    for name, intervals in contacts.items():
        if not intervals:
            continue
        if name not in point_index:
            raise KeyError('no point index for effector %r' % name)
        off = _height_offset(name)
        for f0, f1, kind in intervals:
            if kind == 'top':
                surf = box.height
            elif kind == 'ground':
                surf = 0.0
            else:
                raise ValueError('contact kind must be top or ground, got %r' % kind)
            idx = np.nonzero((frames >= f0) & (frames <= f1))[0]
            if idx.size == 0:
                continue
            out.append(Height(point=int(point_index[name]), frames=idx, z=surf + off, weight=weight))
    out.append(Above(point='all', frames=np.arange(F), surface=box))
    if pin_first > 0:
        out.append(Pin(frames=np.arange(min(pin_first, F)), value=0.0, weight=weight))
    if pin_last > 0:
        out.append(Pin(frames=np.arange(max(0, F - pin_last), F), value=0.0, weight=weight))
    return out
