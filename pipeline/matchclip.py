"""Match the in-game pose of a captured vault to the original ACU clip: time map, yaw, and pelvis offset.

Per capture ("measure, then match", PLAN_v2 section 1):
  match_window(cap, clip_index)       time span in which the trigger clip is the main clip
  fit_time_map(cap, dump, window)     clip_time = o + r * (t - t_start) and a yaw about Z (game = Rz(yaw) @ clip)
                                      that best match the in-game bones relative to Hips with the clip's own bones
                                      relative to Hips (so vertical pelvis corrections do not enter the fit)
  pelvis_correction(cap, dump, fit)   in-game Hips minus clip-predicted Hips per frame, against the entity's
                                      world z rise since the clip start, with a linear fit dz = a * rise + b

Conventions: entity space as in acuclip.entity_positions (Z up, character forward +Y). Errors are in cm:
sqrt(mean over frames and bones of |d|^2) where d is the 3D offset of a bone from Hips (in-game minus clip).
Frames outside the clip (clip time < 0 or > clip length) are left out; a fit needs at least half the frames.
"""
import math
from dataclasses import dataclass

import numpy as np

import acuclip
import capture
import dumps
import ik

FPS = acuclip.FPS                                   # 60 Hz clip frames
FIT_BONES = ('LeftHand', 'RightHand', 'LeftFoot', 'RightFoot', 'Head', 'Spine2')
ROOT_BONE = 'Hips'
# A clip with one of these prefixes starting after the trigger ends the match window.
NEXT_CLIP_PREFIXES = ('xx_Parkour_Muraille_Exit', 'xx_Parkour_Landing', 'xx_Parkour_Leaping', 'xx_Nav', 'acu_nav')

O_GRID = (-0.3, 1.5, 0.01)          # clip time at t_start, s
R_GRID = (0.6, 1.6, 0.02)           # playback rate
YAW_GRID = (-180.0, 180.0, 5.0)     # deg
REFINE_START = (0.01, 0.02, 5.0)    # first half widths of the local refinement (o s, r, yaw deg)
REFINE_STEPS = 11                   # points per axis in each refinement pass
REFINE_SHRINK = 0.6
REFINE_MAX_PASSES = 80
MIN_FRAMES = 3
MIN_VALID_FRAC = 0.5
POOR_RMS_CM = 5.0                   # fits above this are reported as poor


# ---------------------------------------------------------------------------------------------- window

def match_window(cap, clip_index):
    """Time window in which the clip at cap.clips[clip_index] is the main clip.

    Starts at the clip's start event. Ends at the start of the first later clip whose name begins with one of
    NEXT_CLIP_PREFIXES, capped at the clip length (record 'length') and at the end of the capture.
    Returns dict: clip_index, clip_name, t_start, t_end, length_s, next_name, next_t, end_reason
    (end_reason is 'next clip', 'clip length' or 'capture end').
    """
    clip = cap.clips[clip_index]
    t_start = float(clip['t'])
    cap_end = float(cap.t[-1])
    length = clip.get('length')
    length = float(length) if length is not None else None
    clip_end = t_start + length if length is not None else cap_end
    limit = min(cap_end, clip_end)

    nxt = None
    for c in cap.clips:
        if c['t'] > t_start and c.get('name') and c['name'].startswith(NEXT_CLIP_PREFIXES):
            nxt = c
            break
    if nxt is not None and nxt['t'] < limit:
        t_end, reason = float(nxt['t']), 'next clip'
        next_name, next_t = nxt['name'], float(nxt['t'])
    else:
        t_end = limit
        reason = 'clip length' if clip_end <= cap_end else 'capture end'
        next_name = nxt['name'] if nxt is not None else None
        next_t = float(nxt['t']) if nxt is not None else None
    return dict(clip_index=int(clip_index), clip_name=clip.get('name'), t_start=t_start, t_end=t_end,
                length_s=t_end - t_start, next_name=next_name, next_t=next_t, end_reason=reason)


def dump_for_clip(clip):
    """Original dump of one capture clip entry (by handle, else by name), or None."""
    dump = None
    if clip.get('handle'):
        dump = dumps.load_by_handle(clip['handle'])
    if dump is None and clip.get('name'):
        dump = dumps.load_by_name(clip['name'])
    return dump


def trigger_dump(cap):
    """(clip_index, dump) of the trigger clip. dump is None when no original dump matches; (None, None) when
    the capture has no trigger clip."""
    clip = capture.trigger_clip(cap)
    if clip is None:
        return None, None
    idx = next(i for i, c in enumerate(cap.clips) if c is clip)
    return idx, dump_for_clip(clip)


# ---------------------------------------------------------------------------------------------- sampling

@dataclass
class _Ctx:
    t: np.ndarray          # [n] capture times of the frames used
    t_rel: np.ndarray      # [n] seconds since the clip start
    game_rel: np.ndarray   # [n, 6, 3] in-game FIT_BONES minus Hips
    game_hips: np.ndarray  # [n, 3] in-game Hips (entity space)
    clip_rel: np.ndarray   # [F, 6, 3] clip FIT_BONES minus Hips, 60 Hz frames
    clip_hips: np.ndarray  # [F, 3] clip Hips, 60 Hz frames
    bones: tuple = FIT_BONES   # the fit bones of this context (FIT_BONES, or those recorded in a partial pose)


def fit_bones_available(cap):
    """FIT_BONES that the capture recorded in at least one frame (a partial pose drops e.g. RightHand)."""
    return tuple(b for b in FIT_BONES if cap.bones.get(b) is not None)


def _prepare(cap, dump, window, partial=False):
    """Frames of the window with every fit bone and Hips finite, as a _Ctx.

    partial False: the capture must have a complete pose (pose_ok), else ValueError (the matchclip_report rule).
    partial True: a pose with missing bones is accepted; the fit uses the FIT_BONES the capture recorded
    (fit_bones_available), with the same cost over those bones. With every bone present it is the same fit.
    """
    if partial:
        if cap.bones_space != 'entity':
            raise ValueError('capture has no entity-space pose (bones_space=%r)' % (cap.bones_space,))
        bones = fit_bones_available(cap)
        if not bones:
            raise ValueError('capture has no fit bone')
    else:
        if not cap.pose_ok or cap.bones_space != 'entity':
            raise ValueError('capture has no entity-space pose (pose_ok=%s, bones_space=%r)'
                             % (cap.pose_ok, cap.bones_space))
        bones = FIT_BONES
    names = bones + (ROOT_BONE,)
    arrs = {}
    for nm in names:
        if cap.bones.get(nm) is None:
            raise ValueError('capture has no bone %r' % nm)
        arrs[nm] = cap.bones[nm][:, 0:3]
    m = (cap.t >= window['t_start']) & (cap.t <= window['t_end'])
    for nm in names:
        m &= np.all(np.isfinite(arrs[nm]), axis=1)
    if int(m.sum()) < MIN_FRAMES:
        raise ValueError('fewer than %d pose frames in the window' % MIN_FRAMES)

    hips = arrs[ROOT_BONE][m]
    game_rel = np.stack([arrs[b][m] - hips for b in bones], axis=1)

    seq = acuclip.pose_sequence(dump)
    P = acuclip.entity_positions(seq)
    bidx = [ik.bone_index(seq.bones, b) for b in bones]
    hidx = ik.bone_index(seq.bones, ROOT_BONE)
    clip_hips = P[:, hidx].copy()
    clip_rel = P[:, bidx] - clip_hips[:, None, :]
    return _Ctx(t=cap.t[m].copy(), t_rel=cap.t[m] - window['t_start'], game_rel=game_rel,
                game_hips=hips, clip_rel=clip_rel, clip_hips=clip_hips, bones=bones)


def _interp_frames(ctx, o, r):
    """Clip frames at clip time o + r * t_rel: (rel [S+(n,6,3)], hips [S+(n,3)], valid [S+(n)]).

    o and r are broadcastable arrays; the output shape is their broadcast shape plus the frame axis.
    Clip times outside the clip are invalid and their values are clamped (not used).
    """
    o = np.asarray(o, float)
    r = np.asarray(r, float)
    ct = o[..., None] + r[..., None] * ctx.t_rel
    idx = ct * FPS
    F = ctx.clip_rel.shape[0]
    valid = (idx >= 0.0) & (idx <= F - 1)
    x = np.clip(idx, 0.0, F - 1)
    i0 = np.minimum(np.floor(x).astype(int), F - 2)
    w = x - i0
    rel = ctx.clip_rel[i0] * (1.0 - w)[..., None, None] + ctx.clip_rel[i0 + 1] * w[..., None, None]
    hips = ctx.clip_hips[i0] * (1.0 - w)[..., None] + ctx.clip_hips[i0 + 1] * w[..., None]
    return rel, hips, valid


def _cost(ctx, o, r, yaw):
    """RMS error in cm for arrays of (o, r, yaw) (broadcast together). inf where too few frames are valid."""
    o, r, yaw = np.broadcast_arrays(np.asarray(o, float), np.asarray(r, float), np.asarray(yaw, float))
    rel, _, valid = _interp_frames(ctx, o, r)
    a = np.radians(yaw)
    c = np.cos(a)[..., None, None]
    s = np.sin(a)[..., None, None]
    x = rel[..., 0]
    y = rel[..., 1]
    z = rel[..., 2]
    rot = np.stack([c * x - s * y, s * x + c * y, z], axis=-1)
    d = rot - ctx.game_rel
    sq = np.sum(d * d, axis=-1).mean(axis=-1)          # S+(n,)
    cnt = valid.sum(axis=-1)
    n = ctx.t_rel.size
    ok = cnt >= max(MIN_FRAMES, MIN_VALID_FRAC * n)
    mse = np.where(valid, sq, 0.0).sum(axis=-1) / np.maximum(cnt, 1)
    rms = np.sqrt(mse) * 100.0
    return np.where(ok, rms, np.inf)


def _grid(spec):
    lo, hi, step = spec
    n = int(round((hi - lo) / step)) + 1
    return np.round(lo + step * np.arange(n), 6)


def _wrap_deg(a):
    return float((a + 180.0) % 360.0 - 180.0)


def _refine(ctx, p0, free):
    """Local shrinking-grid search around p0 = (o, r, yaw). Axes not in `free` stay fixed."""
    p = np.array(p0, float)
    h = np.array(REFINE_START, float)
    best = float(_cost(ctx, p[0], p[1], p[2]))
    for _ in range(REFINE_MAX_PASSES):
        axes = []
        for k in range(3):
            if free[k]:
                axes.append(p[k] + h[k] * np.linspace(-1.0, 1.0, REFINE_STEPS))
            else:
                axes.append(np.array([p[k]]))
        O, R, Y = np.meshgrid(*axes, indexing='ij')
        C = _cost(ctx, O, R, Y)
        idx = np.unravel_index(int(np.argmin(C)), C.shape)
        if C[idx] <= best:
            best = float(C[idx])
            p = np.array([O[idx], R[idx], Y[idx]], float)
        # shrink only when the best point is interior on every free axis; a best point on the edge means the
        # optimum is further out, so keep the window size and move on
        on_edge = any(free[k] and idx[k] in (0, REFINE_STEPS - 1) for k in range(3))
        if not on_edge:
            h = h * REFINE_SHRINK
        if np.all(h[[k for k in range(3) if free[k]]] < 1e-7):
            break
    return p, best


def _linfit(x, y):
    """(a, b, corr, R2) of y = a * x + b; NaN when x has no spread or fewer than 3 points."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 3 or np.std(x) < 1e-9 or np.std(y) < 1e-12:
        return float('nan'), float('nan'), float('nan'), float('nan')
    a, b = np.polyfit(x, y, 1)
    corr = float(np.corrcoef(x, y)[0, 1])
    return float(a), float(b), corr, corr * corr


# ---------------------------------------------------------------------------------------------- public API

def fit_time_map(cap, dump, window, partial=False):
    """Best clip time map and yaw for the window (see module docstring).

    Coarse grid over O_GRID x R_GRID x YAW_GRID, then local refinement. Also returns the r = 1 fit (o and yaw
    free) and the unfitted baseline (o = 0, r = 1, yaw = 0). partial=True accepts a pose with missing bones
    (see _prepare); the result then lists the bones used.

    Result dict: o (s), r, yaw_deg, rms_cm, poor (rms_cm > POOR_RMS_CM), at_edge (dict o, r: best coarse point on
    the grid border), n_frames, rms_r1 (dict o, yaw_deg, rms_cm), rms_identity_cm (o=0, r=1, yaw=0),
    rms_naive_cm (o=0, r=1, best yaw), window, bones (tuple of the fit bones).
    """
    ctx = _prepare(cap, dump, window, partial=partial)
    og, rg, yg = _grid(O_GRID), _grid(R_GRID), _grid(YAW_GRID)

    grid = np.empty((og.size, rg.size, yg.size))
    Rg = rg[:, None]
    Yg = yg[None, :]
    for i, o in enumerate(og):
        grid[i] = _cost(ctx, o, Rg, Yg)
    if not np.isfinite(grid).any():
        raise ValueError('no clip time gives enough valid frames in the grid')
    i, j, k = np.unravel_index(int(np.argmin(grid)), grid.shape)
    edge = dict(o=bool(i in (0, og.size - 1)), r=bool(j in (0, rg.size - 1)))
    p, rms = _refine(ctx, (og[i], rg[j], yg[k]), (True, True, True))

    grid1 = np.empty((og.size, yg.size))
    for ii, o in enumerate(og):
        grid1[ii] = _cost(ctx, o, 1.0, yg)
    ii, kk = np.unravel_index(int(np.argmin(grid1)), grid1.shape)
    p1, rms1 = _refine(ctx, (og[ii], 1.0, yg[kk]), (True, False, True))

    o, r, yaw = float(p[0]), float(p[1]), _wrap_deg(p[2])
    rms = float(_cost(ctx, o, r, yaw))
    return dict(
        o=o, r=r, yaw_deg=yaw, rms_cm=rms, poor=bool(rms > POOR_RMS_CM), at_edge=edge,
        n_frames=int(ctx.t_rel.size),
        rms_r1=dict(o=float(p1[0]), yaw_deg=_wrap_deg(p1[2]), rms_cm=float(rms1)),
        rms_identity_cm=float(_cost(ctx, 0.0, 1.0, 0.0)),
        rms_naive_cm=float(_cost(ctx, 0.0, 1.0, yaw)),
        window=dict(window), bones=tuple(ctx.bones),
    )


def fit_landscape(cap, dump, window, yaw_deg):
    """RMS (cm) over the (o, r) grids at a fixed yaw: (o_grid [No], r_grid [Nr], rms [No, Nr])."""
    ctx = _prepare(cap, dump, window)
    og, rg = _grid(O_GRID), _grid(R_GRID)
    out = np.empty((og.size, rg.size))
    for i, o in enumerate(og):
        out[i] = _cost(ctx, o, rg, float(yaw_deg))
    return og, rg, out


def pelvis_correction(cap, dump, fit):
    """Hips offset of the game against the clip, per frame of the window, with the entity's z rise.

    dhips = in-game Hips - Rz(yaw) @ clip Hips (entity space, clip time from the fit). rise = entity world z minus
    its z at the clip start. Linear fit dz_hips = a * rise + b over the valid frames.
    The z at the clip start is linearly interpolated from the capture samples, so when the entity starts rising
    exactly at the clip start, rise (and b) carry up to one sample's worth of rise as an offset.

    Result dict: t, rise (m), dhips [n,3] (m), fit_z dict(a, b, corr, r2), rise_max (m over the window),
    dxy_mean_cm (mean horizontal Hips offset).
    """
    window = fit['window']
    ctx = _prepare(cap, dump, window)
    _, hips_pred, valid = _interp_frames(ctx, fit['o'], fit['r'])
    a = math.radians(fit['yaw_deg'])
    c, s = math.cos(a), math.sin(a)
    hp = np.stack([c * hips_pred[:, 0] - s * hips_pred[:, 1],
                   s * hips_pred[:, 0] + c * hips_pred[:, 1],
                   hips_pred[:, 2]], axis=-1)
    dhips = ctx.game_hips - hp

    z = np.asarray(cap.entity_pos[:, 2], float)
    z0 = float(np.interp(window['t_start'], cap.t, z))
    rise = np.interp(ctx.t, cap.t, z) - z0

    v = valid
    ok_rise = rise[v] if v.any() else np.zeros(0)
    a_, b_, corr, r2 = _linfit(ok_rise, dhips[v, 2])
    dxy = np.linalg.norm(dhips[v, 0:2], axis=1) if v.any() else np.zeros(0)
    return dict(
        t=ctx.t[v], rise=ok_rise, dhips=dhips[v],
        fit_z=dict(a=a_, b=b_, corr=corr, r2=r2),
        rise_max=float(np.max(rise)) if rise.size else float('nan'),
        dxy_mean_cm=float(np.mean(dxy) * 100.0) if dxy.size else float('nan'),
    )
