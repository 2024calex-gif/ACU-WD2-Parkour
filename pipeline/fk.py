"""Vectorised forward kinematics and quaternion helpers (shared module).

Conventions (same as retarget.py / skel.py):
  * quaternions are (x, y, z, w), Hamilton product, world = parent_world * local
  * a local transform maps child-space to parent-space: p_world = p_parent + qrot(q_parent, p_local)
  * arrays may carry leading batch dimensions: local_rot [..., B, 4], local_pos [..., B, 3]
    (e.g. [F, B, 4] for F frames); root_rot / root_pos have shape batch + (4,) / batch + (3,)
    or are a single (4,) / (3,) vector shared by all batch elements.
"""
import numpy as np


def qmul(a, b):
    """Hamilton product a * b, broadcasting over leading dimensions."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def qconj(q):
    q = np.asarray(q, float)
    return np.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], axis=-1)


def qnormalize(q):
    q = np.asarray(q, float)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def qrot(q, v):
    """Rotate vector(s) v by unit quaternion(s) q (last axis = components)."""
    q = np.asarray(q, float)
    v = np.asarray(v, float)
    u = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def slerp(a, b, t):
    """Spherical interpolation with shortest path; t broadcasts against the leading dimensions."""
    a = qnormalize(a)
    b = qnormalize(b)
    d = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(d < 0, -b, b)
    d = np.abs(d)
    theta = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(theta)
    small = s < 1e-6
    safe_s = np.where(small, 1.0, s)
    t = np.asarray(t, float)
    if t.ndim > 0:
        t = t[..., None]
    wa = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / safe_s)
    wb = np.where(small, t, np.sin(t * theta) / safe_s)
    return qnormalize(wa * a + wb * b)


def check_parent_order(parents):
    """Every bone must come after its parent (parent index < own index); parent -1 means root."""
    for i, p in enumerate(parents):
        if p is None or not (p == -1 or 0 <= p < i):
            raise ValueError('bone %d has parent %r: parents must come before children' % (i, p))


def fk(bones, local_rot, local_pos, root_rot=(0, 0, 0, 1), root_pos=(0, 0, 0)):
    """Local transforms -> world transforms.

    bones: list of dicts with 'parent' (index or -1), ordered parent-before-child.
    local_rot [..., B, 4], local_pos [..., B, 3].
    root_rot / root_pos: transform applied to every root bone (parent -1), composed with that
    bone's own local transform: world_root = root * local_root.
    Returns (world_rot [..., B, 4], world_pos [..., B, 3]).
    """
    parents = [int(b['parent']) for b in bones]
    check_parent_order(parents)
    lr = np.asarray(local_rot, float)
    lp = np.asarray(local_pos, float)
    B = len(bones)
    if lr.shape[-2:] != (B, 4) or lp.shape[-2:] != (B, 3):
        raise ValueError('local arrays must be [..., %d, 4] and [..., %d, 3], got %s and %s'
                         % (B, B, lr.shape, lp.shape))
    batch = lr.shape[:-2]
    rr = np.broadcast_to(np.asarray(root_rot, float), batch + (4,))
    rp = np.broadcast_to(np.asarray(root_pos, float), batch + (3,))

    WR = np.empty(batch + (B, 4))
    WP = np.empty(batch + (B, 3))
    for i, p in enumerate(parents):
        if p < 0:
            WR[..., i, :] = qmul(rr, lr[..., i, :])
            WP[..., i, :] = rp + qrot(rr, lp[..., i, :])
        else:
            WR[..., i, :] = qmul(WR[..., p, :], lr[..., i, :])
            WP[..., i, :] = WP[..., p, :] + qrot(WR[..., p, :], lp[..., i, :])
    return WR, WP
