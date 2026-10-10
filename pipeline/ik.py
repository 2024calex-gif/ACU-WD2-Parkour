"""Two-bone inverse kinematics on the ACU skeleton, and IK over a whole pose sequence.

Conventions are those of fk.py: quaternions (x, y, z, w), world = parent_world * local, and
root_rot / root_pos is an extra transform applied in front of the whole skeleton (to the root bone,
index 0 'Reference', exactly like fk.fk). With root_rot = identity and root_pos = the TrackID-0
displacement, the world transform is entity + displacement (acuclip.world_positions).

Only the ancestors of the end bone are evaluated: nothing else can move the end, so one solve is a
handful of quaternion products. Bones outside the chain keep their local rotations, and the local
positions are never changed.
"""
import dataclasses

import numpy as np

import dumps
import fk

# effector name -> (upper, lower, end) bone names of the ACU skeleton (parent chain checked in tests)
CHAINS = {
    'LeftHand': ('LeftArm', 'LeftForeArm', 'LeftHand'),
    'RightHand': ('RightArm', 'RightForeArm', 'RightHand'),
    'LeftFoot': ('LeftUpLeg', 'LeftLeg', 'LeftFoot'),
    'RightFoot': ('RightUpLeg', 'RightLeg', 'RightFoot'),
}

IDENTITY = (0.0, 0.0, 0.0, 1.0)
_AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def bone_index(bones, name):
    """Index of the bone called `name`. Names come from dumps.bone_names() (hash -> name), matched on the hash."""
    hashes = {int(h) for h, n in dumps.bone_names().items() if n == name}
    found = [i for i, b in enumerate(bones) if int(b['hash']) in hashes]
    if len(found) != 1:
        raise KeyError('bone %r matches %d bones in the skeleton' % (name, len(found)))
    return found[0]


def chain_indices(bones, effector):
    """(upper, lower, end) bone indices of an effector in CHAINS."""
    if effector not in CHAINS:
        raise KeyError('unknown effector %r (known: %s)' % (effector, ', '.join(sorted(CHAINS))))
    return tuple(bone_index(bones, n) for n in CHAINS[effector])


def _ancestors(bones, i):
    """Bone indices from the root down to bone i (both included)."""
    path = []
    j = int(i)
    while j != -1:
        path.append(j)
        j = int(bones[j]['parent'])
    return path[::-1]


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _perp(v, u):
    """Component of v perpendicular to the unit vector u."""
    return v - np.dot(v, u) * u


def _arc(a, b):
    """Shortest-arc quaternion (x, y, z, w) that turns unit vector a onto unit vector b."""
    d = float(np.dot(a, b))
    if 1.0 + d < 1e-12:
        # opposite directions: half a turn about an axis perpendicular to a
        axis = np.cross(a, _AXES[0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, _AXES[1])
        axis = _unit(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    c = np.cross(a, b)
    q = np.array([c[0], c[1], c[2], 1.0 + d])
    return q / np.linalg.norm(q)


def two_bone_ik(bones, local_rot, local_pos, chain, target, pole=None,
                root_rot=IDENTITY, root_pos=(0.0, 0.0, 0.0), max_reach=0.999,
                keep_end_world_rot=True):
    """Move the end bone of a chain onto a world target by changing the upper and lower bone rotations.

    bones: ACU skeleton (skel.load_acu()). local_rot [B, 4], local_pos [B, 3]: one pose.
    chain: (upper, lower, end) bone indices; lower must be a descendant of upper and end a descendant
        of lower (bones in between are allowed; they are carried along rigidly).
    target: world point the end joint should reach.
    pole: world point the middle joint (elbow / knee) bends towards. None keeps the current bend plane.
    max_reach: when the target is farther than max_reach * (upper + lower length) the chain is extended
        along the target direction to that distance. A target closer than |upper - lower| is clamped to it.
    keep_end_world_rot: keep the end bone's world rotation by changing its local rotation (3 bones change).
        Otherwise only upper and lower change, and the end rotates with the chain.

    Returns a NEW local_rot array [B, 4]. Only the chain's rotations differ from the input.
    Solution: upper rotates so the middle joint sits at the law-of-cosines elbow position on the
    pole side; lower then swings about the elbow so the end lands on the (clamped) target.
    """
    upper_i, lower_i, end_i = (int(x) for x in chain)
    lr = np.array(local_rot, float)
    lp = np.asarray(local_pos, float)
    if lr.shape != (len(bones), 4) or lp.shape != (len(bones), 3):
        raise ValueError('local arrays must be [%d, 4] and [%d, 3], got %s and %s'
                         % (len(bones), len(bones), lr.shape, lp.shape))
    rr = np.asarray(root_rot, float)
    rp = np.asarray(root_pos, float)

    path = _ancestors(bones, end_i)
    try:
        ku = path.index(upper_i)
        kl = path.index(lower_i)
    except ValueError:
        raise ValueError('chain %r is not an ancestry upper -> lower -> end' % (chain,))
    ke = len(path) - 1
    if not (ku < kl < ke):
        raise ValueError('chain %r is not an ancestry upper -> lower -> end' % (chain,))

    # current world transforms of the ancestors of the end, root first (same rule as fk.fk)
    WR = np.empty((len(path), 4))
    WP = np.empty((len(path), 3))
    for k, i in enumerate(path):
        if k == 0:
            WR[k] = fk.qmul(rr, lr[i])
            WP[k] = rp + fk.qrot(rr, lp[i])
        else:
            WR[k] = fk.qmul(WR[k - 1], lr[i])
            WP[k] = WP[k - 1] + fk.qrot(WR[k - 1], lp[i])

    A = WP[ku]          # upper joint: does not move
    J = WP[kl]          # middle joint
    C = WP[ke]          # end joint
    L1 = float(np.linalg.norm(J - A))
    L2 = float(np.linalg.norm(C - J))
    if L1 < 1e-9 or L2 < 1e-9:
        raise ValueError('degenerate chain: zero-length bone in %r' % (chain,))

    to = np.asarray(target, float) - A
    dist = float(np.linalg.norm(to))
    if dist < 1e-9:
        raise ValueError('target coincides with the upper joint of the chain')
    u = to / dist
    reach = min(dist, max_reach * (L1 + L2))
    reach = max(reach, abs(L1 - L2))
    cos_a = np.clip((L1 * L1 + reach * reach - L2 * L2) / (2.0 * L1 * reach), -1.0, 1.0)
    sin_a = np.sqrt(1.0 - cos_a * cos_a)

    # bend direction: towards the pole, else the current bend plane, else any perpendicular
    v = None
    if pole is not None:
        v = _perp(np.asarray(pole, float) - A, u)
        if np.linalg.norm(v) < 1e-9:
            v = None
    if v is None:
        v = _perp(J - A, u)
    if np.linalg.norm(v) < 1e-9:
        v = np.cross(u, _AXES[0])
        if np.linalg.norm(v) < 1e-9:
            v = np.cross(u, _AXES[1])
    v = _unit(v)

    E = A + L1 * (cos_a * u + sin_a * v)     # solved middle joint
    Tc = A + reach * u                        # clamped target

    # step 1: upper rotation takes the middle joint onto E (rigid turn of the whole upper subtree about A)
    q_up = _arc(_unit(J - A), _unit(E - A))
    parent_u = rr if ku == 0 else WR[ku - 1]
    lr[upper_i] = fk.qmul(fk.qconj(parent_u), fk.qmul(q_up, WR[ku]))
    WRn = WR.copy()
    WPn = WP.copy()
    for k in range(ku, len(path)):
        WRn[k] = fk.qmul(q_up, WR[k])
        WPn[k] = A + fk.qrot(q_up, WP[k] - A)

    # step 2: lower rotation about E takes the end onto the clamped target (length E-end == length E-target)
    q_low = _arc(_unit(WPn[ke] - E), _unit(Tc - E))
    parent_l = WRn[kl - 1]
    lr[lower_i] = fk.qmul(fk.qconj(parent_l), fk.qmul(q_low, WRn[kl]))

    # step 3: optionally restore the end's world rotation through its own local rotation
    if keep_end_world_rot:
        parent_e_new = fk.qmul(q_low, WRn[ke - 1])
        lr[end_i] = fk.qmul(fk.qconj(parent_e_new), WR[ke])
    return lr


def solve_sequence(seq, root_rot, root_pos, targets, poles=None, max_reach=0.999,
                   keep_end_world_rot=True):
    """Apply two-bone IK to a PoseSeq for several effectors over time.

    seq: acuclip.PoseSeq. root_rot [F, 4] and root_pos [F, 3]: per-frame transform applied in front of
        the skeleton (see module docstring). Targets are world points in the same frame.
    targets: dict effector name ('LeftHand', 'RightHand', 'LeftFoot', 'RightFoot') -> list of
        (frame_index, world_target [3], weight in [0, 1]); frame_index indexes seq.frames (0 .. F-1).
    poles: optional dict effector -> world pole point (constant over frames).

    For each (frame, effector) the IK solution is slerped in from the current local rotations of the
    chain bones by the weight: weight 0 leaves the frame unchanged, weight 1 applies the full IK.
    Effectors on one frame are applied in order (their chains are disjoint).
    Returns a new PoseSeq; frames, positions and all non-chain bones are untouched.
    """
    F = len(seq.frames)
    bones = seq.bones
    rr = np.asarray(root_rot, float).reshape(F, 4)
    rp = np.asarray(root_pos, float).reshape(F, 3)
    poles = poles or {}

    plan = {}
    for name, entries in targets.items():
        if name not in CHAINS:
            raise KeyError('unknown effector %r (known: %s)' % (name, ', '.join(sorted(CHAINS))))
        for fi, tgt, w in entries:
            fi = int(fi)
            w = float(w)
            if not 0 <= fi < F:
                raise IndexError('frame_index %d outside 0..%d' % (fi, F - 1))
            if not 0.0 <= w <= 1.0:
                raise ValueError('weight %r outside [0, 1]' % w)
            plan.setdefault(fi, []).append((name, np.asarray(tgt, float), w))

    chains = {name: chain_indices(bones, name) for name in {e for v in plan.values() for e, _, _ in v}}
    lr = np.array(seq.local_rot, float)
    for fi in sorted(plan):
        cur = lr[fi].copy()
        for name, tgt, w in plan[fi]:
            if w == 0.0:
                continue
            ch = chains[name]
            ik = two_bone_ik(bones, cur, seq.local_pos[fi], ch, tgt, pole=poles.get(name),
                             root_rot=rr[fi], root_pos=rp[fi], max_reach=max_reach,
                             keep_end_world_rot=keep_end_world_rot)
            idx = list(ch)
            cur[idx] = fk.slerp(cur[idx], ik[idx], w)
        lr[fi] = cur
    return dataclasses.replace(seq, local_rot=lr)
