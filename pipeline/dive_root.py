"""Replacement root displacement (TrackID 0) for the Sprint SlideOver Fd100cm-Up100cm: a dive arc for the WD2 frontdive.

The original TrackID 0 (LocalPosition, codec 27, infotype 3) is the entity's root displacement in the dump frame:
world = entity + displacement, and the obstacle markers are in that frame (near edge y = 4.298, far edge y = 5.298, top
z = 1.000 for Fd100cm-Up100cm). The original root runs at about 7 m/s, rises onto the top by clip 0.67 s, slides at about
3 m/s until 1.0 s, drops at about 1.2 s and runs on.

The replacement keeps the original exactly outside [T_TAKEOFF, T_JOIN] = [0.45, 1.20] clip s, and inside it:

  y(t) = y_orig(t) + D(t)      D(0.45) = D(1.2) = 0 and D'(0.45) = D'(1.2) = 0 (C1 joins), D(t_far) = far_y - y_orig(t_far)
                                so the root passes the far top edge at t_far (about 0.75 to 0.80 s). D' = g is piecewise
                                linear on 0.05 s knots, solved as a smallest-roughness g with the two equalities above.
  z(t)    = Z_PEAK * (1 - smoothstep(t_fall)) * smoothstep(t_rise), so z starts at 0 at the takeoff (C1), clears the top
                                by about Z_PEAK - 1.0 m over the obstacle, and is back to 0 at T_JOIN (C1), where the body
                                is about 0.6 m beyond the far edge.

D is the root's forward change against the original clip; because D(1.2) = 0 and y(1.2) is the original value, the root
at the played window end (W) equals the original, so the Exit (which plays from its own root delta, see dive_chain2.py)
joins at the same place.

Outputs of this module are pure functions; the plot is written by plot_root(). Run `python dive_root.py` to write the
plot and the summary into out\\dive2 (no other file is touched).
"""
import json
import os

import numpy as np

import acuclip
import bake2
import dumps

FPS = acuclip.FPS
SO_NAME = 'xx_Parkour_Sprint_SlideOver_Fd100cm-Up100cm'
T_TAKEOFF = 0.45            # clip s: identical to the original before this (ACU takeoff, contacts.events)
T_JOIN = 1.20               # clip s: identical from here on (the played window end W is at clip 1.2009)
T_FAR_CANDIDATES = (0.75, 0.76, 0.77, 0.78, 0.79, 0.80)
KNOT_STEP = 0.05
Z_PEAK = 1.15               # m: origin height over the obstacle (top is 1.00 m): 15 cm clearance for the origin
T_RISE_END = 0.62           # clip s: z reaches Z_PEAK (before the near top edge, about 0.60 s)
T_FALL_START = 0.80         # clip s: z starts to fall from Z_PEAK (the far top edge is crossed here)
T_FALL_END = 1.17           # clip s: z is back to 0 (about 0.6 m beyond the far edge), held to T_JOIN
MIN_SPEED = 0.25            # m/s: the root must stay strictly forward (monotone) with this margin
MAX_SPEED = 8.5             # m/s: flight speed ceiling ("near the run-in speed", about 7 m/s)
FINE_STEP = 0.001           # s: integration grid for D


def _smooth(x, width_s):
    """Centred moving average over width_s seconds (edge-padded), for the speed-based selection."""
    n = max(1, int(round(width_s / FINE_STEP)))
    k = np.ones(2 * n + 1) / (2 * n + 1)
    xp = np.concatenate([np.full(n, x[0]), x, np.full(n, x[-1])])
    return np.convolve(xp, k, mode='valid')


def _smoothstep(u):
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def original_root(dump):
    """(F, disp): frames 0..round(Length * 60) and the original root displacement [F, 3] (x, y, z)."""
    nf = int(round(float(dump['Length']) * FPS))
    F = np.arange(nf + 1)
    return F, acuclip.markers(dump, F)['disp']


def rows_at(arr, t):
    """Linear interpolation of a 60 Hz array [F, 3] at clip times t (seconds, scalar or array) -> [..., 3]."""
    x = np.asarray(t, float) * FPS
    idx = np.arange(arr.shape[0], dtype=float)
    cols = [np.interp(x, idx, arr[:, k]) for k in range(arr.shape[1])]
    return np.stack(cols, axis=-1)


def obstacle_geometry(dump):
    """Absolute obstacle geometry in the dump frame: near_y, far_y, height, depth (medians of the edge markers)."""
    ob = acuclip.obstacle(dump)
    return dict(near_y=ob['near_y'], far_y=ob['far_y'], height=ob['height'], depth=ob['depth'], spread=ob['spread'])


def _knots():
    n = int(round((T_JOIN - T_TAKEOFF) / KNOT_STEP))
    return np.linspace(T_TAKEOFF, T_JOIN, n + 1)


def _hat_matrix(knots, grid):
    """Phi[g_index, k]: integral from T_TAKEOFF to grid point of the k-th hat basis (piecewise linear g on the knots)."""
    K = len(knots)
    basis = np.zeros((len(grid), K))
    for k in range(K):
        e = np.zeros(K)
        e[k] = 1.0
        basis[:, k] = np.interp(grid, knots, e)
    dt = grid[1] - grid[0]
    cum = np.concatenate([np.zeros((1, K)), np.cumsum(0.5 * (basis[1:] + basis[:-1]) * dt, axis=0)], axis=0)
    return cum


def solve_g(t_far, far_y, y_orig_fn, knots=None):
    """Piecewise linear g = D' on the knots: minimal roughness (first differences) with g(0.45) = g(1.2) = 0,
    D(1.2) = 0 and D(t_far) = far_y - y_orig(t_far). Returns (knots, g, D_fn(grid), grid)."""
    knots = _knots() if knots is None else knots
    grid = np.round(np.arange(T_TAKEOFF, T_JOIN + 1e-9, FINE_STEP), 6)
    Phi = _hat_matrix(knots, grid)                          # D(grid) = Phi @ g_full
    K = len(knots)
    free = np.arange(1, K - 1)                              # g_0 and g_last are fixed at 0
    Pf = Phi[:, free]
    i_far = int(np.argmin(np.abs(grid - t_far)))
    A = np.stack([Pf[-1], Pf[i_far]])
    b = np.array([0.0, far_y - float(y_orig_fn(grid[i_far]))])
    # roughness: first differences of g (full vector), restricted to the free unknowns
    Dd = np.zeros((K - 1, K))
    for i in range(K - 1):
        Dd[i, i] = -1.0
        Dd[i, i + 1] = 1.0
    Lm = Dd[:, free]
    Q = 2.0 * Lm.T @ Lm + 1e-9 * np.eye(len(free))
    n = len(free)
    KKT = np.zeros((n + 2, n + 2))
    KKT[:n, :n] = Q
    KKT[:n, n:] = A.T
    KKT[n:, :n] = A
    rhs = np.concatenate([np.zeros(n), b])
    sol = np.linalg.solve(KKT, rhs)
    g = np.zeros(K)
    g[free] = sol[:n]
    D = Phi @ g
    return knots, g, D, grid


def build_root(dump, t_far=None, z_peak=Z_PEAK, verbose=False):
    """The replacement root for the SlideOver dump.

    t_far: target far-edge crossing time (clip s). None: among the candidates in T_FAR_CANDIDATES whose speed profile
    stays within [MIN_SPEED, MAX_SPEED], the one with the smallest peak deceleration (the search is printed when verbose).
    Returns dict with F, y, z (the new root [F, 3]), orig [F, 3], knots, g, geometry, metrics and the keys of TrackID 0.
    """
    F, orig = original_root(dump)
    geo = obstacle_geometry(dump)
    far_y = geo['far_y']

    def y_orig_fn(t):
        return float(rows_at(orig, np.array([t]))[0, 1])

    cands = (t_far,) if t_far is not None else T_FAR_CANDIDATES
    tried = []
    chosen = None
    for tf in cands:
        knots, g, D, grid = solve_g(tf, far_y, y_orig_fn)
        yo_grid = rows_at(orig, grid)[:, 1]
        vo = np.gradient(yo_grid, FINE_STEP)
        v_new = vo + np.interp(grid, knots, g)
        ok = bool(np.min(v_new[5:-5]) >= MIN_SPEED and np.max(v_new) <= MAX_SPEED)
        acc = np.gradient(_smooth(v_new, 0.05), FINE_STEP)
        tried.append(dict(t_far=float(tf), min_speed=float(np.min(v_new[5:-5])), max_speed=float(np.max(v_new)),
                          max_decel=float(-np.min(acc)), ok=ok))
        if verbose:
            print('  t_far %.2f: min speed %.2f max %.2f max decel %.1f m/s2 -> %s'
                  % (tf, tried[-1]['min_speed'], tried[-1]['max_speed'], tried[-1]['max_decel'],
                     'ok' if ok else 'rejected'), flush=True)
        if t_far is not None:
            chosen = (tf, knots, g, D, grid, v_new)
            break
        if ok and (chosen is None or tried[-1]['max_decel'] < chosen[6]):
            chosen = (tf, knots, g, D, grid, v_new, tried[-1]['max_decel'])
    if chosen is None:
        raise RuntimeError('no far-edge time in %s gives a monotone root within the speed limits' % (cands,))
    tf, knots, g, D, grid, v_new = chosen[:6]

    # the new root on the 60 Hz frames: original outside [T_TAKEOFF, T_JOIN]
    t = F / FPS
    y = orig[:, 1].copy()
    z = orig[:, 2].copy()
    inside = (t > T_TAKEOFF + 1e-9) & (t < T_JOIN - 1e-9)
    D_f = np.interp(t[inside], grid, D)
    y[inside] = orig[inside, 1] + D_f
    # z arc: smoothstep rise from T_TAKEOFF to T_RISE_END, plateau at z_peak, smoothstep fall to 0 at T_JOIN
    rise = _smoothstep((t - T_TAKEOFF) / (T_RISE_END - T_TAKEOFF))
    fall = _smoothstep((t - T_FALL_START) / (T_FALL_END - T_FALL_START))
    z_arc = z_peak * rise * (1.0 - fall)
    z[inside] = z_arc[inside]
    new = np.stack([orig[:, 0].copy(), y, z], axis=1)          # x: the original's own (identical outside the window)

    metrics = root_metrics(F, orig, new, geo, tf)
    keys = [{'Frame': int(f), 'LocalPosition': [round(float(new[f, 0]), 9), round(float(new[f, 1]), 9),
                                                round(float(new[f, 2]), 9)]} for f in F]
    return dict(F=F, orig=orig, new=new, knots=knots, g=g, grid=grid, D_grid=D, v_new_grid=v_new, geometry=geo,
                t_far=float(tf), metrics=metrics, keys=keys, tried=tried)


def root_metrics(F, orig, new, geo, t_far):
    """Checks used by the report: far-edge crossing, peak z, errors at the join, identity outside, speed, C1 at joins."""
    t = F / FPS
    far_y, near_y, top = geo['far_y'], geo['near_y'], geo['height']
    y, z = new[:, 1], new[:, 2]
    yo, zo = orig[:, 1], orig[:, 2]
    cross = np.nonzero((y[:-1] < far_y) & (y[1:] >= far_y))[0]
    t_cross = float(t[cross[0]] + (far_y - y[cross[0]]) / (y[cross[0] + 1] - y[cross[0]]) / FPS) if cross.size else None
    i_join = int(round(T_JOIN * FPS))
    i_take = int(round(T_TAKEOFF * FPS))
    over = (y >= near_y) & (y <= far_y)
    v = np.diff(y) * FPS
    dz_ident = float(np.max(np.abs(new[np.r_[0:i_take + 1, i_join:len(F)]] - orig[np.r_[0:i_take + 1, i_join:len(F)]])))
    h = 1.0 / FPS
    vo_join = float(np.interp(T_JOIN, t[1:], np.diff(yo) * FPS))
    vn_join = float(np.interp(T_JOIN, t[1:], np.diff(y) * FPS))
    vo_take = float(np.interp(T_TAKEOFF, t[1:], np.diff(yo) * FPS))
    vn_take = float(np.interp(T_TAKEOFF, t[1:], np.diff(y) * FPS))
    z_fall_end = None
    beyond = np.nonzero((z <= 1e-6) & (t > T_FALL_START))[0]
    if beyond.size:
        z_fall_end = float(t[beyond[0]])
    return dict(
        t_far=float(t_far), t_cross=t_cross,
        far_y=float(far_y), near_y=float(near_y), top=float(top),
        y_err_join_cm=float((y[i_join] - yo[i_join]) * 100.0),
        z_join=float(z[i_join]), z_peak=float(np.max(z)), t_z_peak=float(t[int(np.argmax(z))]),
        z_min_over_box=float(np.min(z[over])) if np.any(over) else None,
        clearance_over_box_cm=float((np.min(z[over]) - top) * 100.0) if np.any(over) else None,
        y_beyond_far_at_z0=float(y[int(beyond[0])] - far_y) if beyond.size else None,
        t_z0_after_fall=z_fall_end,
        identity_max_dev_m=dz_ident,
        v_min=float(np.min(v[1:-1])), v_max=float(np.max(v)), v_min_t=float(t[1 + int(np.argmin(v[1:-1]))]),
        v_orig_takeoff=float(vo_take), v_new_takeoff=float(vn_take), v_orig_join=float(vo_join), v_new_join=float(vn_join),
        c1_note='velocity equal to the original at both joins by construction (g = 0 there); 60 Hz differences are for information',
        vel_diff_takeoff_cm_s=float((vn_take - vo_take) * 100.0), vel_diff_join_cm_s=float((vn_join - vo_join) * 100.0),
        monotone=bool(np.all(v >= 0.0)),
        speed_at=dict((str(tt), round(float(np.interp(tt, t[1:], v)), 3)) for tt in (0.45, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)),
        step_frame_cm=float(np.max(np.linalg.norm(np.diff(new[:, 1:], axis=0), axis=1)) * 100.0),
    )


def root_track(keys, length):
    """A TrackID 0 track in the plugin format: codec 27, infotype 3, LocalPosition keys (60 Hz frames)."""
    return {'TrackID': '0', 'codec': 27, 'infotype': 3, 'Keyframes': keys}


def plot_root(path, dump, res, title=None):
    """Original and new y(t) and z(t) with the markers (far and near edge, landing point) and the played window."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    F, orig, new, geo, m = res['F'], res['orig'], res['new'], res['geometry'], res['metrics']
    t = F / FPS
    markers = acuclip.markers(dump, F)
    near_w = markers['near_edge'] + orig
    far_w = markers['far_edge'] + orig
    landing = markers['landing'] + orig if markers['landing'] is not None else None
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 9.5), dpi=100, sharex=True,
                                        gridspec_kw=dict(height_ratios=[1.1, 1.0, 1.0]))
    win0, win1 = 0.423, 0.423 + 1.046 * 0.744
    for ax in (ax1, ax2, ax3):
        ax.axvspan(win0, win1, color='#dfe9f7', zorder=0, label='played SlideOver window (o to W)')
        ax.axvline(T_TAKEOFF, color='#999', ls=':', lw=0.9)
        ax.axvline(T_JOIN, color='#999', ls=':', lw=0.9)
        ax.axvline(m['t_far'], color='tab:green', ls='--', lw=0.9)
    ax1.plot(t, orig[:, 1], color='#7a7a7a', lw=2.0, label='original root y')
    ax1.plot(t, new[:, 1], color='tab:red', lw=1.6, label='new root y')
    ax1.plot(t, far_w[:, 1] * 0 + geo['far_y'], color='k', lw=0.8, ls='-.', label='far top edge y = %.3f' % geo['far_y'])
    ax1.plot(t, near_w[:, 1] * 0 + geo['near_y'], color='#8b6f47', lw=0.8, ls='-.', label='near top edge y = %.3f' % geo['near_y'])
    if landing is not None:
        ax1.plot(t, landing[:, 1], color='tab:purple', lw=0.7, ls=':', label='landing marker (world)')
    if m['t_cross'] is not None:
        ax1.plot([m['t_cross']], [geo['far_y']], 'o', color='tab:green', ms=5, label='far edge crossed at %.3f s' % m['t_cross'])
    ax1.set_ylabel('root y (m, dump frame)')
    ax1.legend(fontsize=7, loc='upper left')
    ax2.plot(t, orig[:, 2], color='#7a7a7a', lw=2.0, label='original root z')
    ax2.plot(t, new[:, 2], color='tab:red', lw=1.6, label='new root z')
    ax2.axhline(geo['height'], color='#8b6f47', lw=0.9, ls='-.', label='obstacle top z = %.3f' % geo['height'])
    ax2.set_ylabel('root z (m)')
    ax2.legend(fontsize=7, loc='upper right')
    vo = np.gradient(orig[:, 1]) * FPS
    vn = np.gradient(new[:, 1]) * FPS
    ax3.plot(t, vo, color='#7a7a7a', lw=1.6, label='original root speed (m/s)')
    ax3.plot(t, vn, color='tab:red', lw=1.4, label='new root speed (m/s)')
    ax3.axhline(0.0, color='k', lw=0.6)
    ax3.set_ylabel('forward speed (m/s)')
    ax3.set_xlabel('clip time (s)')
    ax3.legend(fontsize=7, loc='upper right')
    ax3.set_xlim(0.0, 1.45)
    ttl = title or ('replacement TrackID 0, %s' % SO_NAME)
    ax1.set_title('%s\nfar edge at %.2f s; z peak %.2f m at %.2f s; y error at 1.20 s %.1f cm'
                  % (ttl, m['t_cross'] or float('nan'), m['z_peak'], m['t_z_peak'], m['y_err_join_cm']), fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_outputs(dump, res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plot_root(os.path.join(out_dir, 'root_curve.png'), dump, res)
    with open(os.path.join(out_dir, 'root_curve.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(t_far_target=res['t_far'], tried=res['tried'], metrics=res['metrics'],
                       knots=[float(k) for k in res['knots']], g=[float(x) for x in res['g']]), f, indent=1)


if __name__ == '__main__':
    import config
    out = os.path.join(config.OUT, 'dive2')
    dump = dumps.load_by_name(SO_NAME)
    res = build_root(dump, verbose=True)
    write_outputs(dump, res, out)
    print(json.dumps(res['metrics'], indent=1))
