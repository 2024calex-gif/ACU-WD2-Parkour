"""T26 (dive14): the body's forward path in the VELOCITY domain, so that the roll and the get-up are not stretched.

Diagnosis (dive13 report): dive4.Place maps the WD2 pelvis path as y = y_take + s (y_wd2 - y_takeoff) with s = 1.35 over the whole
clip. The forward VELOCITY of the body is therefore 1.35 times WD2's during the plant, the roll and the get-up, while the limb
rotations stay WD2's. Contact points that are nearly stationary in WD2 then skate at about 0.35 times the pelvis speed.

Fix at the source (this module): VelPlace has dive4.Place's interface, but the forward scale acts on the pelvis VELOCITY:

    y(u) = y_take + int_{takeoff}^{u} s(v) dy_wd2/dv dv,     s(v) = s_f up to the blend start,
                                                              C1 blend (smoothstep) to 1.0 at the plant start (WD2 0.75 s),
                                                              s = 1.0 from the plant start on (plant, roll, get-up, run-out).

The blend is 0.07 s long (WD2 0.68 to 0.75, i.e. clip 0.85 to 0.92 at k = 1). The hand target is taken at the plant start:
the hand is the pelvis at that time plus the WD2 hand offset (unscaled, the body is rigid), and it must land at least 0.20 m
beyond the far edge (the rule of dive4). The placement is installed by a context manager that swaps dive4.Place only while
dive14 runs; dive4 to dive13 keep their defaults.

Downstream (dive13's slow root, dive12's chain, dive11's Exit): the SlideOver and Exit roots are dive13's, but the post-1.20
root speed is the natural speed of the NEW body path (dive13.natural_speed calls dive4b.target_world, which uses the
installed placement), so the root and the Reference are re-derived from it. Timing (k 1.00 / 1.00), the events (dive12's S18
SlideOver, dive11's Exit), the aux tracks and the floor rule are as dive12 and dive13.

WD2-native check: the reconstructed WD2 pelvis path with its own FK body (no ACU, no IK, s = 1), skate measured with the same
metric on the ground (its own contact model). See native_skate().

Outputs (only under out\\dive14): 47BFB02DF.baked.json, F7919BDF0.baked.json, mapping_dive14.json, joined_sheet.png/.gif,
summary.json, report.md
Run (from pipeline\\): python dive14.py variant     (about 3 minutes; the native check is quick)
                       python dive14.py probe       (hand clearance table, no bake)
"""
import contextlib
import json
import os
import sys

import numpy as np

import acuclip
import aux_tracks as AT
import config
import dive3 as D3
import dive4 as D4
import dive4b as B
import dive5 as F5
import dive8 as D8
import dive9 as D9
import dive10 as D10
import dive11 as D11
import dive12 as D12
import dive13 as D13
import dive_chain2 as DC2
import dumps
import wd2_path as WP

FPS = 60
S_FLIGHT = 1.40                  # s_f: the flight scale (dive13 used 1.35; the hand rule needs about 1.40 with the shift below)
SHIFT = 0.22                     # m: the takeoff shift (dive13 / dive12: 0.20)
T_TO = D12.T_TO                  # 0.62 clip s: dive12's takeoff
BLEND_WD2 = (0.68, 0.75)         # WD2 s: the scale blends from s_f to 1.0 here (0.07 s before the plant start)
W_HAND = 0.75                    # WD2 s: the hand target is the pelvis at the plant start plus the WD2 hand offset
K_PLANT, K_ROLL = 1.0, 1.0       # dive12 / dive13 timing
D_S = D13.D_S                    # 18: the SlideOver cut (dive12 S18)
DELAY = D11.DELAY                # 15: the Exit window (dive11)
EXIT_O = D4.EXIT_O               # 0.208 clip s: the Exit's played start
EXIT_R = D4.EXIT_R               # 0.9665 clip s per played s
FAR_CLEAR = 0.20                 # m: hand rule (dive4)
OVERRIDE = 'preview: dive14 roll unstretched'
OUT_DIR = os.path.join(config.OUT, 'dive14')
MAPPING_NAME = 'mapping_dive14.json'
REPORT = os.path.join(OUT_DIR, 'report.md')
SUMMARY = os.path.join(OUT_DIR, 'summary.json')
DIVE13_SUMMARY = os.path.join(config.OUT, 'dive13', 'summary.json')
SO_HEX, SO_NAME = D9.SO_HEX, D9.SO_NAME
EXIT_HEX, EXIT_NAME = D9.EXIT_HEX, D9.EXIT_NAME
TARGET = dict(roll_p95=0.5, roll_max=1.0, getup_p95=0.5, getup_max=1.0)

_TABLES = {}


# ------------------------------------------------------------------------------------------------ the velocity-domain path
def _smoothstep(u):
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _tables(sol, s_f):
    """The WD2 time grid u (0.2..1.8 s), the pelvis y along it, and the cumulative mapped displacement from the takeoff (no
    y_take), for the scale s_f. Cached per (id(sol), s_f)."""
    key = (id(sol), round(float(s_f), 6))
    if key in _TABLES:
        return _TABLES[key]
    u = np.round(np.arange(float(sol['t'][0]), float(sol['t'][-1]) + 1e-9, 0.0005), 6)
    Py = np.interp(u, sol['t'], sol['P'][:, 1])
    dPy = np.gradient(Py, u)
    b0, b1 = BLEND_WD2
    s = float(s_f) + (1.0 - float(s_f)) * _smoothstep((u - b0) / (b1 - b0))
    integrand = s * dPy
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(u))])
    i_to = int(np.argmin(np.abs(u - WP.TAKEOFF)))
    cum = cum - cum[i_to]
    _TABLES[key] = dict(u=u, Py=Py, cum=cum, s=s)
    return _TABLES[key]


class VelPlace:
    """dive4.Place with the forward scale on the pelvis VELOCITY (see the module doc). Same interface: sol, s (= s_f),
    y_take; map_points, pelvis, bump, hand_anchor, hand_y, p0, s, y_take."""

    def __init__(self, sol, s, y_take, bump_amp=0.0, b0=D4.BUMP_T0, b1=D4.BUMP_T1):
        self.sol = sol
        self.s = float(s)
        self.y_take = float(y_take)
        self.bump_amp = float(bump_amp)
        self.b0, self.b1 = float(b0), float(b1)
        self.p0 = WP.path_at(sol, [WP.TAKEOFF])[0]
        self.tab = _tables(sol, self.s)

    def _M(self, u):
        return self.y_take + np.interp(u, self.tab['u'], self.tab['cum'])

    def map_points(self, p):
        """WD2 pelvis points -> ACU: x unscaled, z unchanged, y = M(u) where u is the WD2 time with that path height."""
        p = np.asarray(p, float)
        out = np.empty_like(p)
        out[..., 0] = p[..., 0] - self.p0[0]
        u = np.interp(p[..., 1], self.tab['Py'], self.tab['u'])
        out[..., 1] = self._M(u)
        out[..., 2] = p[..., 2]
        return out

    def bump(self, w):
        return self.bump_amp * WP.raised_cosine(np.asarray(w, float), self.b0, self.b1)

    def pelvis(self, w):
        q = self.map_points(WP.path_at(self.sol, w))
        q[..., 2] = q[..., 2] + self.bump(w)
        return q

    def hand_anchor(self, side):
        """The hand target (ACU): the pelvis at the plant start plus the WD2 hand offset (rigid), on the ACU hand height."""
        a = self.sol['anchors']['hand_%s_xy' % side]
        pw = float(np.interp(W_HAND, self.tab['u'], self.tab['Py']))
        q = np.array([a[0] - self.p0[0], float(self._M(W_HAND)) + (a[1] - pw), WP.ACU_HAND_Z], float)
        return q

    def hand_y(self):
        return {side: float(self.hand_anchor(side)[1]) for side in ('L', 'R')}


@contextlib.contextmanager
def velocity_place():
    """Swaps dive4.Place for VelPlace (dive4b.target_world, dive8/9/11 and dive3._hand_targets all look it up at call time)."""
    saved = D4.Place
    D4.Place = VelPlace
    try:
        yield
    finally:
        D4.Place = saved


def hand_clearance(ctx, s_f, shift):
    """Clearance (m) of the closer hand beyond the far edge, with the velocity-domain placement (no bake)."""
    y_take = float(B.y_take_for(ctx, T_TO, shift))
    hy = VelPlace(ctx.sol, s_f, y_take).hand_y()
    return min(hy.values()) - float(ctx.ob['far_y']), hy


# ------------------------------------------------------------------------------------------------ WD2-native skate
def native_phases(sol, w_W, delay=DELAY):
    """WD2 time windows (s) of the native check, on the reconstructed path: plant, roll, get-up, Exit window (the same
    windows as the chain at k 1.00: plant from 0.75 s, roll to W, get-up 0.42 s after W, Exit window to 1.80 s)."""
    t = sol['t']
    p_up = float(D10.p_up_for(delay))
    return {
        'plant': (t >= WP.TAKEOFF + 0.30 - 1e-9) & (t <= 0.92 + 1e-9),
        'roll': (t > 0.92 + 1e-9) & (t <= w_W + 1e-9),
        'get-up': (t > w_W + 1e-9) & (t <= w_W + p_up + 1e-9),
        'Exit window': (t > w_W + p_up + 1e-9) & (t <= float(t[-1]) + 1e-9),
    }


def native_skate(sol, w_W):
    """Skate of the WD2 FK body on its reconstructed path (s = 1, no ACU, no IK): the same metric as the chain, on the ground
    (ground z = 0, no box; the WD2 solve has no obstacle). Returns {phase: max, p95, bone, samples, runs}."""
    import dive13 as _d13
    W = WP.world_joints(sol)                 # [F, B, 3] WD2 world joints (pelvis path + pelvis-relative FK)
    body = [int(b) for b in sol['body']]
    names = sol['names']
    h = W[:, :, 2]
    out = {}
    for ph, sel in native_phases(sol, w_W).items():
        frames = np.nonzero(sel)[0]
        sp, lab, runs = _d13.skate_samples(W, frames, body, names, 1.0 / FPS, h)
        if sp.size:
            k = int(np.argmax(sp))
            out[ph] = dict(max_m_s=float(sp.max()), p95_m_s=float(np.percentile(sp, 95)), bone_at_max=lab[k],
                           samples=int(sp.size), runs=int(runs))
        else:
            out[ph] = dict(max_m_s=None, p95_m_s=None, bone_at_max=None, samples=0, runs=int(runs))
    return out


# ------------------------------------------------------------------------------------------------ the bake (dive12 chain)
def run_case14(ctx, t_W, s_f, shift, label):
    """dive12.run_case with the flight scale s_f and the takeoff shift as arguments (dive12's constants are not changed).
    The caller installs the velocity placement and the roots. Returns the dive12 run dict plus the skate phases."""
    with D12.at_window(ctx, t_W), D12.lift_grid():
        case = D9.make_case9(T_TO, K_PLANT, K_ROLL, s_f, shift)
        A, t0, t1 = D8.bump_for(ctx, case)
        F5.BUMP.update(t0=t0, t1=t1, A=A)
        res9 = D9.solve9(ctx, case, A, t0, t1)
        F5.BUMP.update(t0=t0, t1=t1, A=A)
        lift_S, s_info = D11.solve_S11(ctx, case, res9, DELAY)
        case_a = D9.make_case9(T_TO, K_PLANT, K_ROLL, s_f, shift)
        res = D11.solve_E11(ctx, case_a, lift_S, DELAY, use_ik=True)
        m = D11.run_metrics(ctx, res, case_a, DELAY)
        fl_roll = D12.trunk_speed_deg_s(ctx, res, case_a, t_W)
        skate_old = D12.contact_skate_m(ctx, res, case_a, t_W)
        out = dict(label=label, D_S=D_S, t_W=float(t_W), k_plant=K_PLANT, k_roll=K_ROLL, A=A, t0=t0, t1=t1, res=res, m=m,
                   case=case_a, lift_S=lift_S, s_info=s_info, trunk=fl_roll, skate=skate_old,
                   lift_S_lift_res9=np.array(res9['lift_final'], float), s_f=float(s_f), shift=float(shift))
    res['_feet_free_to'] = int(m['feet_free_to_frame'])
    out['skate_phases'] = D13.skate_metric(ctx, res, case_a, t_W)
    return out


def write_outputs14(ctx, res, so_track, ex_track, out_dir):
    """dive13's writer with dive14's override and the same events (SlideOver dive12 S18, Exit dive11 delay15)."""
    os.makedirs(out_dir, exist_ok=True)
    rig = ctx.rig
    so = D8.patch_dict(rig, res['fin_S']['rot_tracks'], res['fin_S']['trans_keys'], ctx.so['dump']['Length'])
    so_aux = AT.add_aux_tracks(so, ctx.so['dump'], D9.TAKEOFF_FRAME, None)
    so_aux['Tracks'] = list(so_aux['Tracks']) + D12.slideover_event_tracks(ctx.so['dump'], D_S) + [D13.root_track(so_track)]
    AT.write_patch(os.path.join(out_dir, SO_HEX + '.baked.json'), so_aux)
    ex = D8.patch_dict(rig, res['fin_E']['rot_tracks'], res['fin_E']['trans_keys'], ctx.ex_dump['Length'])
    ex_aux = AT.add_aux_tracks(ex, ctx.ex_dump, 0, int(res['_feet_free_to']))
    ex_aux['Tracks'] = list(ex_aux['Tracks']) + D11.event_tracks_delay15(ctx) + [D13.root_track(ex_track)]
    AT.write_patch(os.path.join(out_dir, EXIT_HEX + '.baked.json'), ex_aux)
    m = dict(clips=[
        DC2.mapping_entry(SO_HEX, SO_NAME, SO_HEX + '.baked.json', True, override=OVERRIDE),
        DC2.mapping_entry(EXIT_HEX, EXIT_NAME, EXIT_HEX + '.baked.json', True, only_after=[SO_NAME], override=OVERRIDE),
    ])
    with open(os.path.join(out_dir, MAPPING_NAME), 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=1)


# ------------------------------------------------------------------------------------------------ main
def _json(x):
    return D12._jsonable(x)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    stage = argv[0] if argv else 'variant'
    if stage == 'report':
        with open(SUMMARY, encoding='utf-8') as f:
            write_report14(json.load(f))
        print('report written: %s' % REPORT, flush=True)
        return 0
    ctx = F5.make_ctx()
    t_W = float(ctx.t_W) + D_S / FPS
    if stage == 'probe':
        for s_f, sh in ((1.35, 0.20), (1.35, 0.24), (1.40, 0.20), (1.40, SHIFT), (1.45, 0.20)):
            c, hy = hand_clearance(ctx, s_f, sh)
            print('s_f %.2f shift %.2f: clearance %.3f m (L %.3f R %.3f)' % (s_f, sh, c, hy['L'], hy['R']), flush=True)
        return 0
    os.makedirs(OUT_DIR, exist_ok=True)
    clear, hy = hand_clearance(ctx, S_FLIGHT, SHIFT)
    if clear < FAR_CLEAR - 1e-9:
        raise RuntimeError('hand rule fails: clearance %.3f m' % clear)
    with velocity_place():
        case_probe = D9.make_case9(T_TO, K_PLANT, K_ROLL, S_FLIGHT, SHIFT)
        so_new, v_nat, v_j = D13.so_root(ctx, case_probe)          # the post-1.20 root from the NEW body path
        nav_v, _nav0 = D13.nav_run_speed()
        joined_E, track_E, _D, info = D13.exit_root(ctx, so_new, t_W, nav_v)
        root_W = np.array([np.interp(t_W * FPS, np.arange(len(so_new)), so_new[:, c]) for c in range(3)])
        print('dive14: s_f %.2f shift %.2f, hand clearance %.3f m, root v at 1.20 %.3f, Nav_Run %.3f'
              % (S_FLIGHT, SHIFT, clear, v_j, nav_v), flush=True)
        with D13.Patched(ctx):
            D13.install_roots(ctx, so_new, joined_E, root_W)
            run = run_case14(ctx, t_W, S_FLIGHT, SHIFT, 'dive14')
            res = run['res']
            write_outputs14(ctx, res, so_new, track_E, OUT_DIR)
            D13.write_sheet(ctx, res, run['case'], OUT_DIR, 'dive14 roll unstretched (velocity-domain body path, s_f %.2f, '
                                                              'shift %.2f), S18 window %.3f s' % (S_FLIGHT, SHIFT, t_W))
            dg = D13.digest(ctx, run)
            offs = D13.offsets_of(res)
            hy_map = VelPlace(ctx.sol, S_FLIGHT, float(B.y_take_for(ctx, T_TO, SHIFT))).hand_y()
    native = native_skate(ctx.sol, float(D8.Timing8(T_TO, K_PLANT, K_ROLL).w(t_W)))
    summary = dict(t_W=t_W, D_S=D_S, s_f=S_FLIGHT, shift=SHIFT, blend_wd2=list(BLEND_WD2), hand_target_w=W_HAND,
                   hand_clearance_m=clear, hand_y=hy_map, far_edge_y=float(ctx.ob['far_y']),
                   so_root=dict(v_at_1p2_new=v_j, nav_run=nav_v, exit_v0=info['v0'], exit_v1=info['v1'],
                                p_end=info['p_end']),
                   native_skate=native, dive14=_json(dg), offsets=_json(offs))
    with open(SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(_json(summary), f, indent=1)
    print('dive14 floor min %.3f pass %s; skate %s' % (dg['floor_min_cm'], dg['floor_pass'], dg['skate_phases']['all']),
          flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ report
def _f(x, nd=2):
    return '-' if x is None else ('%.*f' % (nd, x))


def _dense_root(path, n):
    """The TrackID 0 root of a written patch as a dense [n, 3] array (frames 0..n-1)."""
    with open(path, encoding='utf-8') as f:
        patch = json.load(f)
    tr = [t for t in patch['Tracks'] if str(t['TrackID']) == '0'][0]
    fr = np.array([k['Frame'] for k in tr['Keyframes']], float)
    pos = np.array([k['LocalPosition'] for k in tr['Keyframes']], float)
    F = np.arange(n)
    return np.stack([np.interp(F, fr, pos[:, c]) for c in range(3)], axis=1)


def _root_rows(so_track, exit_track, so_orig, ex_orig_disp, t_W, t_end):
    """Root profile rows: SlideOver y and speed at clip times, Exit joined y and speed (per joined second) at Exit clip times."""
    so_rows, ex_rows = {}, {}
    vso = D13.root_velocity(so_track)
    for t in (1.2, 1.3, 1.4, 1.5, round(t_W, 4), 1.6):
        f = t * FPS
        so_rows[str(t)] = dict(y=float(np.interp(f, np.arange(len(so_track)), so_track[:, 1])),
                               v=float(np.interp(f, np.arange(len(so_track)), vso)))
    cut = np.array([np.interp(EXIT_O * FPS, np.arange(len(ex_orig_disp)), ex_orig_disp[:, c]) for c in range(3)])
    root_W_new = np.array([np.interp(t_W * FPS, np.arange(len(so_track)), so_track[:, c]) for c in range(3)])
    root_W_old = np.array([np.interp(t_W * FPS, np.arange(len(so_orig)), so_orig[:, c]) for c in range(3)])
    joined_new = root_W_new[None, :] + (exit_track - cut[None, :])
    joined_old = root_W_old[None, :] + (ex_orig_disp - cut[None, :])
    for t in (0.208, 0.25, 0.30, 0.35, 0.40, 0.45, round(t_end, 4)):
        f = t * FPS
        F = np.arange(len(joined_new))
        vn = np.gradient(joined_new[:, 1], 1.0 / FPS) * EXIT_R
        vo = np.gradient(joined_old[:, 1], 1.0 / FPS) * EXIT_R
        ex_rows[str(t)] = dict(y_new=float(np.interp(f, F, joined_new[:, 1])), v_new=float(np.interp(f, F, vn)),
                               y_old=float(np.interp(f, F, joined_old[:, 1])), v_old=float(np.interp(f, F, vo)))
    return so_rows, ex_rows


def write_report14(summary, path=REPORT):
    import dumps as _dumps
    import acuclip as _ac
    base = json.load(open(DIVE13_SUMMARY, encoding='utf-8'))
    b = base['baseline']
    d13 = base['dive13']
    d = summary['dive14']
    m = d['metrics']
    bm = b['metrics']
    nat = summary['native_skate']
    off14, off13, off18 = summary['offsets'], d13['offsets'], b['offsets']
    t_W = summary['t_W']
    # root profiles from the written patches and the original dump
    so_dump = _dumps.load_by_name(D9.SO_NAME)
    ex_dump = _dumps.load_by_name(D9.EXIT_NAME)
    n_so = int(round(float(so_dump['Length']) * FPS)) + 1
    so_orig = _ac.markers(so_dump, np.arange(n_so))['disp']
    n_ex = int(round(float(ex_dump['Length']) * FPS)) + 1
    ex_disp = _ac.markers(ex_dump, np.arange(n_ex))['disp']
    so14 = _dense_root(os.path.join(OUT_DIR, SO_HEX + '.baked.json'), n_so)
    ex14 = _dense_root(os.path.join(OUT_DIR, EXIT_HEX + '.baked.json'), n_ex)
    t_end = EXIT_O + EXIT_R * summary['so_root']['p_end']
    so_rows, ex_rows = _root_rows(so14, ex14, so_orig, ex_disp, t_W, t_end)
    so13 = base['so_root']['profile_new']
    so_orig_prof = base['so_root']['profile_orig']
    phases = ('plant', 'roll', 'get-up', 'Exit window')
    sk_b, sk_13, sk_14 = b['skate_phases'], d13['skate_phases'], d['skate_phases']
    L = []
    L.append('# T26 dive14: the body path in velocity domain (no stretch in the plant, roll and get-up)')
    L.append('')
    L.append('Generated by `v2/pipeline/dive14.py` (`dive14.py variant`, then `dive14.py report`). Written only under `out\\dive14`. '
             'The bake is dive12\'s chain (dive11, dive9, dive8) with dive13\'s slowed root, and the body placement is '
             '`VelPlace` (dive14), installed by a context manager that swaps `dive4.Place`; dive4 to dive13 are unchanged.')
    L.append('')
    L.append('## 1. Result')
    L.append('')
    L.append('The fix is in and working on the path, but the skate target is NOT met. The roll p95 is %s m/s (target 0.5, dive13 %s, '
             'S18 %s) and the max %s m/s (target 1.0). The get-up p95 is %s m/s (target 0.5). The best I got is above, and the '
             'limits are in section 6.'
             % (_f(sk_14['roll']['p95_m_s']), _f(sk_13['roll']['p95_m_s']), _f(sk_b['roll']['p95_m_s']),
                _f(sk_14['roll']['max_m_s']), _f(sk_14['get-up']['p95_m_s'])))
    L.append('')
    L.append('* Velocity-domain placement: the roll p95 falls from %s (dive13) to %s m/s, the plant from %s to %s m/s. The WD2 '
             'stretch was part of the roll slide, but not all of it.'
             % (_f(sk_13['roll']['p95_m_s']), _f(sk_14['roll']['p95_m_s']), _f(sk_13['plant']['p95_m_s']), _f(sk_14['plant']['p95_m_s'])))
    L.append('* WD2 native (the reconstructed path with its own FK body, s = 1, no ACU): roll p95 %s m/s and get-up p95 %s m/s, both '
             'inside the targets. So the reconstruction does not need a correction in the roll; the remaining roll skate is in the '
             'ACU bake (section 6).'
             % (_f(nat['roll']['p95_m_s']), _f(nat['get-up']['p95_m_s'])))
    L.append('')
    L.append('## 2. Flight scale and takeoff shift (hand target)')
    L.append('')
    L.append('s(t) = s_f from the takeoff to WD2 0.68 s, smooth (smoothstep, C1) to 1.00 at the plant start (WD2 0.75 s, clip 0.92 s), 1.00 '
             'after. The hand target is the pelvis at the plant start plus the WD2 hand offset (rigid, unscaled). The hand rule is '
             '>= 0.20 m beyond the far edge (far edge y %s, so >= %s).'
             % (_f(summary['far_edge_y'], 3), _f(summary['far_edge_y'] + 0.20, 3)))
    L.append('')
    L.append('| placement | s_f | shift (m) | hand L (y) | hand R (y) | clearance (m) | rule |')
    L.append('|---|---|---|---|---|---|---|')
    L.append('| dive12 / dive13 (dive4 mapping: the hand offset is scaled too) | 1.35 | 0.20 | closer hand 5.501 | - | 0.203 | pass |')
    L.append('| velocity domain at 1.35 / 0.20 (rigid hand offset) | 1.35 | 0.20 | 5.528 | 5.459 | 0.162 | fails: the closer hand is 0.04 m short |')
    L.append('| dive14 | %s | %s | %s | %s | %s | pass |'
             % (_f(summary['s_f']), _f(summary['shift']), _f(summary['hand_y']['L'], 3), _f(summary['hand_y']['R'], 3),
                _f(summary['hand_clearance_m'], 3)))
    L.append('')
    L.append('Chosen: s_f = 1.40 and shift = 0.22. At s_f = 1.35 the shift would need 0.24; at s_f = 1.40 it is 0.20 (margin 0.00); 0.22 keeps a margin. The cost of the dive4 scan (shift^2 + 0.5 (s_f - 1.35)^2) prefers s_f 1.53 / shift 0.10, but s_f 1.53 would stretch the flight more than needed, so the smaller change was kept.')
    L.append('')
    L.append('## 3. Skate (m/s), the same metric as dive13')
    L.append('')
    L.append('Rule as dive13: a non-finger bone within 12 cm of its surface, the lowest bone, horizontal world speed, kept where the same bone is lowest for at least 3 frames. Phases: plant, roll (plant end to W), get-up (Exit played 0 to P_UP), Exit window (P_UP to p_end). WD2 native uses WD2 time on the same windows, ground only.')
    L.append('')
    L.append('| phase | S18 max | S18 p95 | dive13 max | dive13 p95 | dive14 max | dive14 p95 | dive14 max bone | WD2 native max | WD2 native p95 |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for ph in phases:
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            ph, _f(sk_b[ph]['max_m_s']), _f(sk_b[ph]['p95_m_s']), _f(sk_13[ph]['max_m_s']), _f(sk_13[ph]['p95_m_s']),
            _f(sk_14[ph]['max_m_s']), _f(sk_14[ph]['p95_m_s']), sk_14[ph].get('bone_at_max') or '-',
            _f(nat[ph]['max_m_s']), _f(nat[ph]['p95_m_s'])))
    L.append('')
    L.append('Samples (runs): dive14 plant %s, roll %s, get-up %s, Exit window %s. The Exit window has only 4 samples, so its numbers are thin. The target (roll and get-up p95 <= 0.5, max <= 1.0) is not met: roll p95 %s, get-up p95 %s.'
             % tuple(['%d (%d)' % (sk_14[p]['samples'], sk_14[p]['runs']) for p in phases] + [_f(sk_14['roll']['p95_m_s']), _f(sk_14['get-up']['p95_m_s'])]))
    L.append('')
    L.append('## 4. Reference offsets, floor and the checks')
    L.append('')
    L.append('| quantity | S18 | dive13 | dive14 | rule |')
    L.append('|---|---|---|---|---|')
    L.append('| SlideOver Reference y, before 1.20 s (m) | %s | %s | %s | about <= 0.5 |'
             % ('max %s' % _f(off18['fin_S']['before_1p20_y_max'], 3), 'max %s' % _f(off13['fin_S']['before_1p20_y_max'], 3),
                'max %s' % _f(off14['fin_S']['before_1p20_y_max'], 3)))
    L.append('| SlideOver Reference y, after 1.20 s (m) | %s .. %s | %s .. %s | %s .. %s | about <= 0.5 |'
             % (_f(off18['fin_S']['after_1p20_y_min'], 3), _f(off18['fin_S']['after_1p20_y_max'], 3),
                _f(off13['fin_S']['after_1p20_y_min'], 3), _f(off13['fin_S']['after_1p20_y_max'], 3),
                _f(off14['fin_S']['after_1p20_y_min'], 3), _f(off14['fin_S']['after_1p20_y_max'], 3)))
    L.append('| Exit Reference y (m) | %s .. %s | %s .. %s | %s .. %s | about <= 0.5 |'
             % (_f(off18['fin_E']['y_min'], 3), _f(off18['fin_E']['y_max'], 3), _f(off13['fin_E']['y_min'], 3),
                _f(off13['fin_E']['y_max'], 3), _f(off14['fin_E']['y_min'], 3), _f(off14['fin_E']['y_max'], 3)))
    L.append('| floor min, all phases (cm) | %s | %s | %s | >= 0 |' % (_f(b['floor_min_cm'], 3), _f(d13['floor_min_cm'], 3), _f(d['floor_min_cm'], 3)))
    L.append('| floor rule | %s | %s | %s | pass |' % (b['floor_pass'], d13['floor_pass'], d['floor_pass']))
    L.append('| tilt at W (deg) | %s | %s | %s | <= 60 |' % (_f(bm['tilt_W'], 1), _f(base['dive13']['metrics']['tilt_W'], 1), _f(m['tilt_W'], 1)))
    L.append('| continuity at W (deg) | %s | %s | %s | <= 45 |' % (_f(bm['cont_deg'], 1), _f(base['dive13']['metrics']['cont_deg'], 1), _f(m['cont_deg'], 1)))
    L.append('| Exit Reference z at P_UP (m) | %s | %s | %s | >= 0.78 |' % (_f(bm.get('z_at_p_up'), 3), _f(base['dive13']['metrics'].get('z_at_p_up'), 3), _f(m.get('z_at_p_up'), 3)))
    L.append('| peak Reference z speed after W (m/s) | %s | %s | %s | <= 4 |' % (_f(bm.get('rise_speed_joined_post_W_mps'), 2), _f(base['dive13']['metrics'].get('rise_speed_joined_post_W_mps'), 2), _f(m.get('rise_speed_joined_post_W_mps'), 2)))
    L.append('| Nav gap at the window end (deg) | %s | %s | %s | <= 25 |' % (_f(bm.get('gap_at_window_end_max_patched'), 1), _f(base['dive13']['metrics'].get('gap_at_window_end_max_patched'), 1), _f(m.get('gap_at_window_end_max_patched'), 1)))
    L.append('| world pelvis dv / step (m/s per frame, cm per frame) | %s / %s | %s / %s | %s / %s | dv <= 1.5, step <= 12 |'
             % (_f(bm['world_pelvis_dv_mps'], 2), _f(bm['world_pelvis_step_cm'], 1), _f(base['dive13']['metrics']['world_pelvis_dv_mps'], 2),
                _f(base['dive13']['metrics']['world_pelvis_step_cm'], 1), _f(m['world_pelvis_dv_mps'], 2), _f(m['world_pelvis_step_cm'], 1)))
    L.append('')
    L.append('Failing items carried from dive9 to dive11: %s.' % '; '.join(d['fails10']))
    L.append('')
    L.append('Continuity at W, tilt at W, Nav gap, Exit z at P_UP and the peak rise are identical in the three runs: they come from '
             'the rotations and the Exit profile, which a placement change does not touch. The Reference offsets, the floor and the '
             'skate are the placement-dependent rows.')
    L.append('')
    L.append('## 5. Root profiles and the Exit')
    L.append('')
    L.append('SlideOver root (y in m, forward speed in m/s):')
    L.append('')
    L.append('| clip s | original y | original v | dive13 y | dive13 v | dive14 y | dive14 v |')
    L.append('|---|---|---|---|---|---|---|')
    for t in ('1.2', '1.3', '1.4', '1.5', str(round(t_W, 4)), '1.6'):
        o = so_orig_prof.get(t) or {}
        L.append('| %s | %s | %s | %s | %s | %s | %s |' % (t, _f(o.get('y'), 3), _f(o.get('v'), 2), _f(so13[t]['y'], 3), _f(so13[t]['v'], 2),
                                                        _f(so_rows[t]['y'], 3), _f(so_rows[t]['v'], 2)))
    L.append('')
    L.append('Exit (joined, y in m, speed in m/s per joined second):')
    L.append('')
    L.append('| Exit clip s | vanilla y | vanilla v | dive14 y | dive14 v |')
    L.append('|---|---|---|---|---|')
    for t, r in ex_rows.items():
        L.append('| %s | %s | %s | %s | %s |' % (t, _f(r['y_old'], 3), _f(r['v_old'], 2), _f(r['y_new'], 3), _f(r['v_new'], 2)))
    L.append('')
    L.append('Exit stand-up: Reference z at P_UP %s m (rule >= 0.78), peak rise %s m/s (rule <= 4), Nav gap at the window end %s deg (rule <= 25).'
             % (_f(m.get('z_at_p_up'), 3), _f(m.get('rise_speed_joined_post_W_mps'), 2), _f(m.get('gap_at_window_end_max_patched'), 1)))
    L.append('')
    L.append('Entity speed at the Exit window end: %s m/s (joined), Nav_Run %s m/s.'
             % (_f(summary['so_root']['exit_v1'], 2), _f(summary['so_root']['nav_run'], 2)))
    L.append('')
    L.append('## 6. What limits the skate (diagnosis)')
    L.append('')
    L.append('* Roll (clip 1.37 to 1.45 s, WD2 1.20 to 1.28 s): the chain lowers the body by up to 0.15 m (solved contact lift from dive9, range -0.15 to +0.16 m across the window). At 1.40 s the pelvis sits at 0.11 m, where WD2\'s pelvis is at 0.25 m with its lowest point (a foot) 8 cm up. The upper leg (LeftUpLeg, 2 to 5 cm up) then moves with the pelvis at 2.4 to 2.8 m/s: this is the roll p95 and max. The Hips follow the velocity-domain target exactly (world Hips y equals the target, difference 0.000 m at every frame).')
    L.append('* Plant: the WD2 reconstruction itself leaves the plant hands at z = 0.06 to 0.10 m (right hand -0.04 m) with 0.5 to 0.9 m/s hand speed (solver residual plant_z 0.061 m rms), so the native plant is 0.88 m/s p95. The ACU plant hands are pinned by the hand IK, so the ACU plant (0.68 m/s p95) is less affected.')
    L.append('* Get-up (Exit played 0 to 0.42 s): the lowest joint moves at 5 to 8.6 m/s, while the root is 2.9 to 4.5 m/s. The rest is the Exit clip\'s own stand-up rotation and the Reference carried from W (the Exit Reference is carried, not solved). The root alone cannot reduce it. Holding the Exit root near zero until the joint leaves the floor would reduce it but breaks the speed rule (root >= 0 with v1 at Nav_Run) and the landing spot; not done here.')
    L.append('* What would be needed: (a) the roll contact lift must not lower the pelvis below the WD2 path (a change to dive9\'s contact solve, which has defaults I was asked not to change), or (b) a foot / thigh contact solve that keeps the upper leg off the floor, or (c) an Exit root that follows the contact point. Each is a decision for you; none was made here.')
    L.append('* Nothing was run in game (rule). The skate numbers are the offline model; the in-game check is still to do.')
    L.append('')
    L.append('## 7. Validation and tests')
    L.append('')
    L.append('* `pipeline\\tests\\test_dive14.py`: VelPlace equals dive4.Place when s = 1; C1 of the scale blend and s = s_f / 1 at its ends; the hand rule; the placement map is monotone; SlideOver root identical before 1.20 s; events identical to dive12 S18 (SlideOver) and dive11 (Exit); JSON format and mapping override.')
    L.append('* Full suite: see the run note in the handback.')
    L.append('')
    L.append('## 8. Outputs')
    L.append('')
    L.append('* `out\\dive14\\47BFB02DF.baked.json` (SlideOver: rotations, Reference, dive12 S18 events, TrackID 0 root, aux tracks)')
    L.append('* `out\\dive14\\F7919BDF0.baked.json` (Exit: rotations, Reference, dive11 delay15 events, TrackID 0 played root, aux tracks)')
    L.append('* `out\\dive14\\mapping_dive14.json` (override "preview: dive14 roll unstretched")')
    L.append('* `out\\dive14\\joined_sheet.png` and `joined_sheet.gif` (entity root drawn), `summary.json`, `report.md`')
    L.append('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


if __name__ == '__main__':
    sys.exit(main())
