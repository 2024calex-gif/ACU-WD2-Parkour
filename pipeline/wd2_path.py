"""Reconstruct the Watch Dogs 2 frontdive body (pelvis) path from its contacts (T14, dive3).

WD2 has rotation tracks only, so the pelvis world position is not decoded. It is rebuilt here as one linear least-squares
problem over the 60 Hz grid 0.2 .. 1.8 s, with the pelvis-relative WD2 FK (pelvis at the origin, decoded rotations, the
WD2 world axes X right, Y forward, Z up) as the known geometry:

  * stance (0.2 .. takeoff 0.45, last frame before it): the stance foot is planted. Its world xy is a free anchor, its world
    z is the contact height (heavy). The stance foot is the one with the lower mean height over the window.
  * flight (takeoff 0.45 .. 0.75): the pelvis second difference equals (0, 0, -g) dt^2 (medium weight). The parabola joins
    the stance pose at takeoff and the hand plant at 0.75 (position and velocity follow from the shared unknowns).
  * hand plant (0.75 .. 0.92): both hands are planted: each hand has a free world xy anchor and a contact height (heavy).
  * roll (0.92 .. 1.18): the lowest non-finger body point (argmin of the WD2 relative height, so no iteration is needed:
    the pelvis is common to all points) rests on z = 0 with zero world velocity while it stays the lowest point (rolling
    contact, approximated).
  * run-out (1.2 .. 1.8): the lowest foot rests on the ground (heavy), with zero world xy velocity while it stays the lowest.
  * a small second-difference penalty on the whole path (regularises the directions the contacts do not fix).

Output frame convention: the pelvis path P(t) is in the WD2 world axes, which equal the ACU entity axes (entity Y forward,
Z up): A_REF * M = identity on positions (retarget.py), so WD2 positions are used as entity positions with no rotation.

Nothing here writes files; dive3.py does the bake and the outputs.
"""
import json
import os

import numpy as np

import config
import fk as fkmod
import retarget as R

FPS = 60
T_START = 0.2                      # s, first sample of the reconstructed path
T_END = 1.8                        # s, last sample
G = 9.81                           # m/s^2, flight gravity
DT = 1.0 / FPS
EVENTS_JSON = os.path.join(config.V2, 'data', 'wd2_events.json')
DIVE_KEY = 'diveroll_frontdive'

# contact heights of the WD2 joints above their surface (metres), the rootsolve offsets (ACU joint centres)
FOOT_Z = 0.08                      # Foot joint above the ground
HAND_Z = 0.03                      # Hand joint above the surface it is planted on

W_CONTACT = 1000.0                 # heavy: planted points (positions in m)
W_BALLISTIC = 100.0                # medium: flight second difference (positions in m)
W_VELOCITY = 100.0                 # medium: zero velocity of a rolling / running-out contact point (m)
LAM_SMOOTH = 0.1                   # small: second difference of the whole path (m)

FINGER_WORDS = ('Index', 'Middle', 'Ring', 'Pinky', 'Thumb', 'Meta', 'Wrist', 'Cuff')


def load_events(path=EVENTS_JSON):
    with open(path, encoding='utf-8') as f:
        return json.load(f)[DIVE_KEY]


def time_grid(t0=T_START, t1=T_END):
    n = int(round((t1 - t0) * FPS)) + 1
    return t0 + np.arange(n) / FPS


def relative_fk(rig, m, t):
    """Pelvis-relative WD2 world positions [F, B, 3] (pelvis at the origin), decoded rotations, WD2 bind offsets.

    rig: retarget.Rig (rig.wd are the WD2 bones, rig.source_locals gives the decoded local rotations at a 30 Hz tick)."""
    bones = rig.wd
    lpos = np.array([b['lpos'] for b in bones], float)
    locs = np.array([np.asarray(rig.source_locals(m, float(tt) * 30.0), float) for tt in t])
    return fkmod.fk(bones, locs, np.broadcast_to(lpos, (len(t), len(bones), 3)))[1]


def joint_sets(rig):
    """Index sets of the WD2 bones: hands, feet, body (non-finger, all joints except the fingers), names."""
    names = [b['name'] for b in rig.wd]
    idx = {n: i for i, n in enumerate(names)}
    parents = [int(b.get('parent', -1)) for b in rig.wd]
    fingers = [i for i, n in enumerate(names) if any(w in n for w in FINGER_WORDS)]
    body = [i for i in range(len(names)) if i not in set(fingers)]
    return dict(names=names, idx=idx, fingers=fingers, body=body, parents=parents)


class _System:
    """Sparse-ish linear least squares built row by row (dense solve; the problem is a few hundred unknowns)."""

    def __init__(self, nvar):
        self.nvar = nvar
        self.A = []
        self.b = []
        self.tags = []
        self.w = []

    def add(self, pairs, rhs, weight, tag):
        row = np.zeros(self.nvar)
        for c, v in pairs:
            row[c] += v * weight
        self.A.append(row)
        self.b.append(rhs * weight)
        self.tags.append(tag)
        self.w.append(float(weight))

    def solve(self):
        A = np.array(self.A)
        b = np.array(self.b)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        res = (A @ x - b) / np.array(self.w)          # unweighted: metres (flight rows: position residual of the 2nd diff)
        return x, res, self.tags


def stance_foot(rel, t, idx):
    """'L Foot' or 'R Foot': the foot with the lower mean relative height over the pre-takeoff window."""
    sel = (t >= T_START - 1e-9) & (t < 0.45 - 1e-9)
    zl = rel[sel, idx['L Foot'], 2].mean()
    zr = rel[sel, idx['R Foot'], 2].mean()
    return 'L Foot' if zl <= zr else 'R Foot'


def solve_path(rig, m, ev=None, t=None, rel=None, lam_smooth=LAM_SMOOTH, takeoff=None):
    """Least-squares pelvis path P [F, 3] (WD2 world, metres) and its diagnostics.

    rel: optional precomputed relative_fk(...) [F, B, 3] on grid t (for tests with synthetic points).
    Returns dict: t, P, rel, names, phases (frame index lists), anchors, residuals (per tag, rms and max, metres or m/s2),
    stance (the stance foot name), takeoff, plant (anchors of the hands), lowest (roll contact per frame), feet (run-out).
    """
    ev = ev or load_events()
    t = time_grid() if t is None else np.asarray(t, float)
    jt = joint_sets(rig)
    idx, body, names, parents = jt['idx'], jt['body'], jt['names'], jt['parents']
    rel = relative_fk(rig, m, t) if rel is None else np.asarray(rel, float)
    F = len(t)
    t_take = float(ev['takeoff'] if takeoff is None else takeoff)
    t_plant0, t_plant1 = ev['hand_plant_start'], ev['hand_plant_end']
    t_roll0, t_roll1 = ev['roll'][0], ev['roll'][1]
    t_feet = ev['landing']

    stance = stance_foot(rel, t, idx)
    s_idx = idx[stance]
    hands = {'L Hand': idx['L Hand'], 'R Hand': idx['R Hand']}

    # unknowns: P (3 F) then anchors: stance xy (2), hand xy (4)
    n_p = 3 * F
    a_stance = n_p
    a_hand = {'L Hand': n_p + 2, 'R Hand': n_p + 4}
    nvar = n_p + 6
    sys_ = _System(nvar)

    def pidx(k, c):
        return 3 * k + c

    pre = np.nonzero(t < t_take - 1e-9)[0]
    flight = np.nonzero((t >= t_take - 1e-9) & (t <= t_plant0 + 1e-9))[0]
    plant = np.nonzero((t >= t_plant0 - 1e-9) & (t <= t_plant1 + 1e-9))[0]
    roll = np.nonzero((t >= t_roll0 - 1e-9) & (t <= t_roll1 + 1e-9))[0]
    runout = np.nonzero(t >= t_feet - 1e-9)[0]
    phases = dict(stance=pre, flight=flight, plant=plant, roll=roll, runout=runout)

    # stance foot pinned before takeoff: x, y = anchor, z = FOOT_Z (heavy)
    for k in pre:
        for c in (0, 1):
            sys_.add([(pidx(k, c), 1.0), (a_stance + c, -1.0)], -rel[k, s_idx, c], W_CONTACT, 'stance_xy')
        sys_.add([(pidx(k, 2), 1.0)], FOOT_Z - rel[k, s_idx, 2], W_CONTACT, 'stance_z')

    # flight: second difference of the pelvis is (0, 0, -g) dt^2 (medium) on the interior frames of the window
    ga = np.array([0.0, 0.0, -G]) * DT * DT
    for k in range(1, F - 1):
        if not (t[k] >= t_take - 1e-9 and t[k] <= t_plant0 + 1e-9):
            continue
        if k - 1 < 0 or k + 1 >= F:
            continue
        for c in range(3):
            sys_.add([(pidx(k - 1, c), 1.0), (pidx(k, c), -2.0), (pidx(k + 1, c), 1.0)], ga[c], W_BALLISTIC, 'flight')

    # hand plant: both hands planted, free anchor (x, y), z = HAND_Z (heavy)
    for k in plant:
        for h, hi in hands.items():
            ah = a_hand[h]
            for c in (0, 1):
                sys_.add([(pidx(k, c), 1.0), (ah + c, -1.0)], -rel[k, hi, c], W_CONTACT, 'plant_xy')
            sys_.add([(pidx(k, 2), 1.0)], HAND_Z - rel[k, hi, 2], W_CONTACT, 'plant_z')

    # roll: lowest non-finger body point on z = 0; zero velocity of that point while it stays the lowest
    low_idx = []
    for k in roll:
        b = body[int(np.argmin(rel[k, body, 2]))]
        low_idx.append(b)
        sys_.add([(pidx(k, 2), 1.0)], -rel[k, b, 2], W_CONTACT, 'roll_z')
        if k - 1 >= 0 and k - 1 in set(roll) and low_idx[-2] == b:
            for c in range(3):
                sys_.add([(pidx(k, c), 1.0), (pidx(k - 1, c), -1.0)],
                         -(rel[k, b, c] - rel[k - 1, b, c]), W_VELOCITY, 'roll_vel')

    # run-out: the lowest foot on the ground (heavy), zero velocity while it stays the lowest foot
    foot_low = []
    feet = [idx['L Foot'], idx['R Foot']]
    for k in runout:
        f = feet[int(np.argmin(rel[k, feet, 2]))]
        foot_low.append(f)
        sys_.add([(pidx(k, 2), 1.0)], FOOT_Z - rel[k, f, 2], W_CONTACT, 'feet_z')
        if k - 1 >= 0 and k - 1 in set(runout) and foot_low[-2] == f:
            for c in range(3):
                sys_.add([(pidx(k, c), 1.0), (pidx(k - 1, c), -1.0)],
                         -(rel[k, f, c] - rel[k - 1, f, c]), W_VELOCITY, 'feet_vel')

    # origin: the pelvis at takeoff is at x = y = 0 (the contact anchors are otherwise free, so a horizontal shift of the
    # whole path is a null space; only displacements from takeoff are used downstream)
    k_to = int(np.argmin(np.abs(t - t_take)))
    for c in (0, 1):
        sys_.add([(pidx(k_to, c), 1.0)], 0.0, W_CONTACT, 'origin')

    # small second-difference penalty over the whole path
    for k in range(1, F - 1):
        for c in range(3):
            sys_.add([(pidx(k - 1, c), 1.0), (pidx(k, c), -2.0), (pidx(k + 1, c), 1.0)], 0.0, lam_smooth, 'smooth')

    x, res, tags = sys_.solve()
    P = x[:n_p].reshape(F, 3)
    anchors = dict(stance_xy=x[a_stance:a_stance + 2].copy(),
                   hand_L_xy=x[a_hand['L Hand']:a_hand['L Hand'] + 2].copy(),
                   hand_R_xy=x[a_hand['R Hand']:a_hand['R Hand'] + 2].copy())
    tags = np.asarray(tags)
    resid = {}
    for tag in sorted(set(tags)):
        r = res[tags == tag]
        resid[tag] = dict(rms=float(np.sqrt(np.mean(r ** 2))), max=float(np.max(np.abs(r))), n=int(r.size))
    return dict(t=t, P=P, rel=rel, names=names, idx=idx, body=body, phases=phases, stance=stance, parents=parents,
                anchors=anchors, residuals=resid, roll_low=np.array(low_idx, int), runout_foot=np.array(foot_low, int),
                takeoff=t_take, nvar=nvar)


def world_joints(sol, joint_idx=None):
    """WD2 world positions of the joints [F, B, 3] (pelvis path + pelvis-relative FK)."""
    return sol['P'][:, None, :] + sol['rel']


def lowest_body_z(sol):
    """Lowest non-finger body point height per frame (m)."""
    W = world_joints(sol)
    return W[:, sol['body'], 2].min(axis=1)


def clearance_region(sol, height=1.0, t0=0.45, t1=0.92):
    """Forward range (pelvis-relative y from the takeoff pelvis) where the lowest body point is >= height, over [t0, t1]."""
    t = sol['t']
    sel = (t >= t0 - 1e-9) & (t <= t1 + 1e-9)
    z = lowest_body_z(sol)
    yy = sol['P'][:, 1] - sol['P'][np.argmin(np.abs(t - t0)), 1]
    ok = sel & (z >= height)
    if not np.any(ok):
        return None
    return dict(y_min=float(yy[ok].min()), y_max=float(yy[ok].max()), t_min=float(t[ok].min()), t_max=float(t[ok].max()),
                n=int(ok.sum()))


# ------------------------------------------------------------------------------------------------ placement in ACU
ACU_HAND_Z = 0.03                  # m, ACU Hand joint above the surface (rootsolve HAND_OFFSET)
BUMP_T0, BUMP_T1 = 0.45, 0.75      # s (WD2 time), the raised-cosine bump lives on the flight only
TAKEOFF = 0.45


def path_at(sol, w):
    """WD2 pelvis path P at WD2 times w (linear between the 60 Hz samples), [..., 3]."""
    w = np.asarray(w, float)
    return np.stack([np.interp(w, sol['t'], sol['P'][:, c]) for c in range(3)], axis=-1)


class Placement:
    """Maps WD2 world points into ACU entity world space.

    x = x_take + (x_wd2 - x_wd2(takeoff)),  y = y_take + s * (y_wd2 - y_wd2(takeoff)),  z = z_wd2 (+ bump).
    s scales the forward travel; y_take is the ACU forward position of the pelvis at takeoff (the forward shift).
    """

    def __init__(self, sol, s, y_take, x_take=0.0, bump_amp=0.0):
        self.sol = sol
        self.s = float(s)
        self.y_take = float(y_take)
        self.x_take = float(x_take)
        self.bump_amp = float(bump_amp)
        self.p0 = path_at(sol, [TAKEOFF])[0]

    def bump(self, w):
        """Raised-cosine vertical bump over the flight (0 at takeoff and at the plant start, peak at the middle)."""
        w = np.asarray(w, float)
        u = (w - BUMP_T0) / (BUMP_T1 - BUMP_T0)
        b = np.where((u > 0) & (u < 1), 0.5 * (1.0 - np.cos(2.0 * np.pi * np.clip(u, 0, 1))), 0.0)
        return self.bump_amp * b

    def map_points(self, p):
        """ACU world points for WD2 world points p [..., 3] (no bump)."""
        p = np.asarray(p, float)
        out = np.empty_like(p)
        out[..., 0] = self.x_take + (p[..., 0] - self.p0[0])
        out[..., 1] = self.y_take + self.s * (p[..., 1] - self.p0[1])
        out[..., 2] = p[..., 2]
        return out

    def pelvis(self, w):
        """ACU world pelvis (Hips joint) target at WD2 times w, with the bump: [..., 3]."""
        q = self.map_points(path_at(self.sol, w))
        q[..., 2] = q[..., 2] + self.bump(w)
        return q

    def hand_anchor(self, side):
        """ACU world plant target of a hand: its WD2 anchor (x, y) mapped, z = ACU_HAND_Z."""
        a = self.sol['anchors']['hand_%s_xy' % side]
        p = np.array([a[0], a[1], ACU_HAND_Z])
        q = self.map_points(p)
        q[2] = ACU_HAND_Z
        return q

    def wd2_hand_anchor_y(self):
        return {s: float(self.sol['anchors']['hand_%s_xy' % s][1]) for s in ('L', 'R')}


def body_points_world(sol, pl, w, joints=None):
    """Approximate ACU world body joint positions [F, B, 3] for WD2 times w: placed pelvis + WD2 relative FK."""
    w = np.asarray(w, float)
    rel = np.stack([np.interp(w, sol['t'], sol['rel'][:, b, c]) for b in range(sol['rel'].shape[1]) for c in range(3)],
                   axis=-1).reshape(len(w), sol['rel'].shape[1], 3)
    pel = pl.pelvis(w)
    return pel[:, None, :] + rel               # body shape is not scaled: only the pelvis travel is (Placement.s)


def box_depth(pts, ob):
    """Largest depth (m) of points inside the obstacle box (y in (near, far), z in (0, height)); 0 when none."""
    z = pts[..., 2]
    y = pts[..., 1]
    inside = (y > ob['near_y']) & (y < ob['far_y']) & (z > 0.0) & (z < ob['height'])
    depth = np.minimum(np.minimum(ob['height'] - z, y - ob['near_y']), ob['far_y'] - y)
    return float(np.where(inside, depth, 0.0).max()) if np.any(inside) else 0.0


def required_lift(sol, pl, ob, body_idx, clear=0.05, w=None):
    """Lift (m) each flight sample needs so that no non-finger point is inside the box (z < top + clear, y in the box).

    pl: Placement (its bump is ignored: this is the lift before the bump). Returns (w, L) with L >= 0 per sample."""
    w = np.round(np.arange(BUMP_T0, BUMP_T1 + 1e-9, 1.0 / FPS), 6) if w is None else np.asarray(w, float)
    pts = body_points_world(sol, Placement(sol, pl.s, pl.y_take, x_take=pl.x_take, bump_amp=0.0), w)[:, body_idx]
    y, z = pts[..., 1], pts[..., 2]
    in_box = (y > ob['near_y']) & (y < ob['far_y'])
    need = np.where(in_box, float(ob['height']) + clear - z, 0.0)
    return w, np.maximum(need.max(axis=1), 0.0)


def raised_cosine(w, t0=BUMP_T0, t1=BUMP_T1):
    """Unit raised cosine on [t0, t1] (0 at both ends, 1 at the middle), as used by Placement.bump."""
    u = (np.asarray(w, float) - t0) / (t1 - t0)
    return np.where((u > 0) & (u < 1), 0.5 * (1.0 - np.cos(2.0 * np.pi * np.clip(u, 0, 1))), 0.0)


def min_amplitude_for_lift(w, lift):
    """Smallest raised-cosine amplitude A with A * shape(w) >= lift(w) everywhere; None when some needed lift has zero shape."""
    shape = raised_cosine(w)
    need = np.asarray(lift, float)
    bad = (shape <= 1e-12) & (need > 1e-9)
    if np.any(bad):
        return None
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(shape > 1e-12, need / np.where(shape > 1e-12, shape, 1.0), 0.0)
    return float(ratio.max())


def plot_side(path, sol, obstacle_top=1.0, moments=(0.45, 0.60, 0.75, 0.92, 1.18, 1.40)):
    """Side view (WD2 forward Y against Z) of the reconstructed path: pelvis, lowest body point, hands and feet, and
    skeleton snapshots at the given WD2 times. The 1.0 m obstacle top is drawn as a reference height."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = sol['t']
    P = sol['P']
    W = world_joints(sol)
    idx = sol['idx']
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=100, gridspec_kw=dict(width_ratios=[2.2, 1.0]))
    y0 = float(P[np.argmin(np.abs(t - 0.45)), 1])
    ax.plot(P[:, 1] - y0, P[:, 2], 'k-', lw=2.0, label='pelvis path')
    ax.plot(W[:, idx['L Hand'], 1] - y0, W[:, idx['L Hand'], 2], color='tab:red', lw=1.0, label='L hand')
    ax.plot(W[:, idx['R Hand'], 1] - y0, W[:, idx['R Hand'], 2], color='tab:orange', lw=1.0, label='R hand')
    ax.plot(W[:, idx['L Foot'], 1] - y0, W[:, idx['L Foot'], 2], color='tab:blue', lw=1.0, label='L foot')
    ax.plot(W[:, idx['R Foot'], 1] - y0, W[:, idx['R Foot'], 2], color='tab:cyan', lw=1.0, label='R foot')
    lowest = lowest_body_z(sol)
    ax.plot(P[:, 1] - y0, lowest, color='tab:green', lw=1.2, ls='--', label='lowest non-finger point')
    ax.axhline(0.0, color='#777', lw=0.8)
    ax.axhline(obstacle_top, color='#8b6f47', lw=1.0, ls=':', label='1.0 m (WD2 obstacle top, reference)')
    edges = [(i, p) for i, p in enumerate(sol_parents(sol)) if p >= 0]
    for tm in moments:
        k = int(np.argmin(np.abs(t - tm)))
        for a, p in edges:
            ax.plot([W[k, a, 1] - y0, W[k, p, 1] - y0], [W[k, a, 2], W[k, p, 2]], color='#999999', lw=0.8, alpha=0.7)
        ax.text(P[k, 1] - y0, P[k, 2] + 0.05, '%.2f s' % tm, fontsize=7)
    ax.set_xlabel('WD2 forward Y, relative to the takeoff pelvis (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('WD2 frontdive, reconstructed pelvis path (side view)', fontsize=10)
    ax.set_aspect('equal')
    ax.set_ylim(-0.1, 2.0)
    ax.legend(fontsize=7, loc='upper left')
    ax2.plot(t, P[:, 2], 'k-', label='pelvis z')
    ax2.plot(t, lowest, 'g--', label='lowest non-finger point')
    ax2.axvspan(0.45, 0.75, color='tab:blue', alpha=0.08, label='flight')
    ax2.axvspan(0.75, 0.92, color='tab:red', alpha=0.08, label='hand plant')
    ax2.axvspan(0.92, 1.18, color='tab:orange', alpha=0.08, label='roll')
    ax2.set_xlabel('WD2 time (s)')
    ax2.set_ylabel('height (m)')
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def sol_parents(sol):
    """Parent index of every WD2 joint (from the rig's bone records), kept on the solution by solve_path."""
    return sol['parents']
