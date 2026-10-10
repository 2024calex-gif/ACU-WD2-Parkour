"""T24 (dive12): the dive11 dive with the SlideOver's cut moved later by D_S frames (S12: 12, S18: 18).

The game cuts from the SlideOver to the Exit at the integer event 0x400001 (infotype 5). In the SlideOver dump it sits at clip
frame 72 (slots 72 and 78). Moving every infotype-5 key at Frame >= F0 later by D_S frames moves that cut, so the window end
W moves from clip 1.2017 s to 1.2017 + D_S / 60 (S12 about 1.40 s, S18 about 1.50 s). The dive is then re-timed for that window:
the takeoff stays (T_to 0.62, s 1.35, shift +0.20) and k_plant, k_roll are chosen as close to 1.0 as the roll allows.

Reuses dive11 (the bake, the SlideOver lift passes, the Exit lift, the foot IK, the floor report, the metrics and the Exit's
event tracks), dive10 (the event shift and event tables), dive9 (solve9, the upright test, make_case9), dive8 (Timing8, bump_for,
bake8, patch_dict) and dive5/dive4 (the context). None of those modules is changed: they are imported and their functions called.
Only the context's SlideOver window attributes are moved (set_window), and they are restored after each run.

Outputs (only under out\\dive12):
  S12\\47BFB02DF.baked.json, S12\\F7919BDF0.baked.json, S12\\mapping_dive12.json, S12\\joined_sheet.png/.gif
  S18\\ (the same), summary.json, report.md
The SlideOver events are shifted in the patch (infotype 5, Slot, TrackID, codec as in dive10). The Exit events are dive11's
(D11.event_tracks_delay15, the 23 delay15 tracks).

Run: python dive12.py   (several minutes: the window screen, the dive11 baseline, then the two variants)
"""
import contextlib
import json
import os
import sys
import zlib

import numpy as np

import acuclip
import aux_tracks as AT
import bake2
import bake_chain as BCh
import config
import dive3 as D3
import dive4 as D4
import dive4b as B
import dive5 as F5
import dive8 as D8
import dive9 as D9
import dive10 as D10
import dive11 as D11
import dive_chain2 as DC2
import dumps
import wd2_path as WP

FPS = 60
D11_DELAY = D11.DELAY                  # 15: the Exit's window and events, as dive11
F0 = 50                                # clip frame: SlideOver keys at or after this frame move
VARIANTS = (('S12', 12), ('S18', 18))
T_TO, S_SCALE, SHIFT = 0.62, 1.35, 0.20          # dive9's takeoff, scale and shift (kept)
K_GRID = tuple(float(x) for x in np.round(np.arange(1.0, 2.0001, 0.1), 2))
PLANT_LO, PLANT_HI = D8.PLANT_W
W_MARGIN = 0.05                        # s: the roll ends this long before the new W (feet down)
TILT_MAX = D9.TILT_MAX
OUT_DIR = os.path.join(config.OUT, 'dive12')
REPORT = os.path.join(OUT_DIR, 'report.md')
SUMMARY = os.path.join(OUT_DIR, 'summary.json')
SO_HEX, SO_NAME = D9.SO_HEX, D9.SO_NAME
EXIT_HEX, EXIT_NAME = D9.EXIT_HEX, D9.EXIT_NAME
MAPPING_NAME = 'mapping_dive12.json'
W_ATTRS = ('W_played', 'span_so', 't_W', 'w_W', 'orig_W', 'root_W', 'disp_ex', 'P_W_rel_y')


def variant_dir(name):
    return os.path.join(OUT_DIR, name)


def override_for(D_S):
    return 'preview: dive12 slideover events +%d' % int(D_S)


# ------------------------------------------------------------------------------------------------ SlideOver events
def shift_slideover_keys(keys, D_S, last, from_frame=F0):
    """The SlideOver's event keys (dive10's rule): Frame >= from_frame moves D_S frames later, clamped to `last` (98), the last
    value kept per frame, strictly increasing. Keys before from_frame stay."""
    return D10.shift_event_keys(keys, int(D_S), int(last), from_frame=int(from_frame))


def slideover_event_tracks(dump, D_S, from_frame=F0):
    """The SlideOver's shifted infotype-5 tracks as patch tracks: TrackID and codec of the original, infotype 5 and Slot."""
    last = D10.last_frame(dump['Length'])
    out = []
    for slot, t in D10.event_tracks_of(dump):
        out.append({'TrackID': str(int(t['TrackID'])), 'codec': int(t['codec']), 'infotype': 5, 'Slot': int(slot),
                    'Keyframes': shift_slideover_keys(t['Keyframes'], D_S, last, from_frame)})
    return out


def slideover_event_table(dump, D_S, from_frame=F0):
    """Before and after per SlideOver event track: slot, TrackID, codec and (frame, value) keys; `moved` = any key moved."""
    last = D10.last_frame(dump['Length'])
    rows = []
    for slot, t in D10.event_tracks_of(dump):
        before = [(int(k['Frame']), int(k['Int'])) for k in t['Keyframes']]
        after = [(k['Frame'], k['Int']) for k in shift_slideover_keys(t['Keyframes'], D_S, last, from_frame)]
        rows.append(dict(slot=slot, TrackID=str(int(t['TrackID'])), codec=int(t['codec']), before=before, after=after,
                         moved=before != after))
    return rows


# ------------------------------------------------------------------------------------------------ the window
def set_window(ctx, t_W):
    """Moves the SlideOver window end to t_W (clip s) and every context attribute that depends on it (as dive4.Ctx.__init__
    computes them). Returns the saved values for restore_window."""
    saved = {k: getattr(ctx, k) for k in W_ATTRS}
    fr = np.arange(len(ctx.hips_orig_so))
    rel = np.asarray(ctx.disp_ex, float) - np.asarray(ctx.root_W, float)[None, :]    # the Exit's own displacement term
    ctx.t_W = float(t_W)
    ctx.span_so = ctx.t_W - ctx.o
    ctx.W_played = ctx.span_so / ctx.r
    ctx.w_W = float(ctx.w(ctx.t_W))
    ctx.orig_W = np.array([np.interp(ctx.t_W * FPS, fr, ctx.hips_orig_so[:, c]) for c in range(3)])
    ctx.root_W = np.array([np.interp(ctx.t_W * FPS, fr, ctx.disp_so[:, c]) for c in range(3)])
    ctx.disp_ex = ctx.root_W[None, :] + rel
    ctx.disp_ex[:, 0] = 0.0
    ctx.P_W_rel_y = float(WP.path_at(ctx.sol, [ctx.w_W])[0][1] - ctx.p0[1])
    return saved


def restore_window(ctx, saved):
    for k, v in saved.items():
        setattr(ctx, k, v)


@contextlib.contextmanager
def at_window(ctx, t_W):
    saved = set_window(ctx, t_W)
    try:
        yield ctx
    finally:
        restore_window(ctx, saved)


LIFT_K_RUN = 120                       # joined lift grid for dive12 runs (dive4b's LIFT_K = 100 is 1.67 s: too short for the
                                       # S18 joined pelvis, which needs 104 frames; the lift past the grid was zero)


@contextlib.contextmanager
def lift_grid(n=LIFT_K_RUN):
    """Sets dive4b.LIFT_K for the bakes of this process (restored after); dive4b's own default is not changed."""
    saved = B.LIFT_K
    B.LIFT_K = int(n)
    try:
        yield
    finally:
        B.LIFT_K = saved


def set_zero_bump():
    F5.BUMP.update(t0=None, t1=None, A=0.0)


# ------------------------------------------------------------------------------------------------ timing screen
def screen_rows(ctx, kps=K_GRID, krs=K_GRID):
    """Zero-lift SlideOver bake for every (k_plant, k_roll) at the context's window (the caller sets it). For each row: the
    feet-down time minus W (the roll margin, s), the trunk tilt at W, inversion and the tilt over the last 0.10 s before W."""
    set_zero_bump()
    chain = D9.chain_idx(ctx)
    rows = []
    for kp in kps:
        for kr in krs:
            case = D9.make_case9(T_TO, kp, kr, S_SCALE, SHIFT)
            res = D8.bake8(ctx, case, np.zeros(B.LIFT_K))
            P = D3._world(res['seg_S']['seq'], res['seg_S']['dump'], res['fin_S']['final'], res['fin_S']['ref_out'],
                          res['fin_S']['disp'])
            tilt, inv, up = D9.upright(P, chain)
            W = int(res['win_S'][-1])
            last = np.arange(int(np.ceil((ctx.t_W - 0.10) * FPS - 1e-9)), W + 1)
            rows.append(dict(k_plant=kp, k_roll=kr, feet_down_s=float(case.tm.t_feet),
                             margin_s=float(ctx.t_W - case.tm.t_feet), tilt_W=float(tilt[W]), inv_W=bool(inv[W]),
                             up_W=bool(up[W]), tilt_last_max=float(tilt[last].max()), inv_last=bool(inv[last].any()),
                             up_last=bool(up[last].all()), plant_s=float(case.tm.t_pe - case.tm.t_pl0)))
    return rows


def upright_ok(row):
    return (row['margin_s'] >= W_MARGIN - 1e-9 and not row['inv_W'] and row['up_W'] and row['tilt_W'] <= TILT_MAX
            and row['tilt_last_max'] <= TILT_MAX and not row['inv_last'] and row['up_last'])


def margin_of(W, kp, kr):
    """The roll margin (s): W minus the feet-down time of the time map (D8.Timing8, the same map the bakes use)."""
    return float(W) - D8.Timing8(T_TO, kp, kr).t_feet


def check_upright(ctx, kp, kr):
    """One zero-lift SlideOver bake at the context's window: the roll margin and the trunk over the last 0.10 s before W."""
    return screen_rows(ctx, kps=(kp,), krs=(kr,))[0]


def upright_candidates(ctx, t_W, n_keep=4, step=0.01, kmax=1.5, max_try=16):
    """The (k_plant, k_roll) closest to 1.0 on a `step` grid (distance (kp-1)^2 + (kr-1)^2, then the smaller k_roll) whose
    roll ends at least W_MARGIN before the window end t_W, and whose zero-lift trunk is upright at W (tilt <= 60 deg, head
    above hips, nothing inverted, over the last 0.10 s). The margin is analytic; the upright test is a bake, tried in order
    (at most max_try). The caller sets the window. Returns (the first n_keep upright rows in that order, all rows tried)."""
    grid = np.round(np.arange(1.0, kmax + 1e-9, step), 4)
    cands = []
    for kp in grid:
        for kr in grid:
            if margin_of(t_W, kp, kr) >= W_MARGIN - 1e-9:
                cands.append((float((kp - 1.0) ** 2 + (kr - 1.0) ** 2), float(kr), float(kp), float(kr)))
    cands.sort()
    tried, ok = [], []
    for _d, _kr, kp, kr in cands[:max_try]:
        row = check_upright(ctx, kp, kr)
        tried.append(row)
        if upright_ok(row):
            ok.append(row)
            if len(ok) >= n_keep:
                break
    return ok, tried


def write_run(ctx, run, D_S, out_dir):
    """The outputs of one accepted run (patches, mapping, sheet) in its variant folder, with the window of the run set."""
    with at_window(ctx, run['t_W']):
        write_patches(ctx, run['res'], D_S, out_dir, int(run['m']['feet_free_to_frame']))
        write_sheet(ctx, run['res'], run['case'], D_S, out_dir, run['label'])


# ------------------------------------------------------------------------------------------------ jarringness and skate
def world_slideover(res):
    fin, seg = res['fin_S'], res['seg_S']
    return D3._world(seg['seq'], seg['dump'], fin['final'], fin['ref_out'], fin['disp'])


def trunk_speed_deg_s(ctx, res, case, t_W):
    """Max angular speed (deg/s) of the trunk axis (Hips -> Head) over the roll (from the plant end to W), and its mean. The
    angle between consecutive 60 Hz frames, times 60."""
    P = world_slideover(res)
    chain = D9.chain_idx(ctx)
    hips, head = chain[0], chain[-1]
    win = np.asarray(res['win_S'], int)
    tS = win / FPS
    t_pe = case.tm.t_pe
    sel = (tS >= t_pe - 1e-9) & (tS <= t_W + 1e-9)
    frames = win[sel]
    axis = P[:, head] - P[:, hips]
    u = axis / np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)
    if len(frames) < 2:
        return dict(max_deg_s=0.0, mean_deg_s=0.0, frames=0)
    f = frames
    dots = np.clip(np.sum(u[f[1:]] * u[f[:-1]], axis=1), -1.0, 1.0)
    rate = np.degrees(np.arccos(dots)) * FPS
    return dict(max_deg_s=float(rate.max()), mean_deg_s=float(rate.mean()), frames=int(len(frames)),
                t_max_s=float(tS[sel][1:][int(np.argmax(rate))]) if len(rate) else None)


def contact_skate_m(ctx, res, case, t_W):
    """Forward slide of the contact points during the roll (plant end to W): for each non-finger body bone and each contiguous run
    of ground contact (height <= 3 cm, outside the box band), the horizontal distance from the run's first contact point.
    Returns the largest over the bones and runs (m), and the (min, max) of the Reference xy offset from the original pose."""
    P = world_slideover(res)
    body = [int(b) for b in ctx.body_nf]
    ob = ctx.ob
    near, far = float(ob['near_y']), float(ob['far_y'])
    win = np.asarray(res['win_S'], int)
    tS = win / FPS
    sel = (tS >= case.tm.t_pe - 1e-9) & (tS <= t_W + 1e-9)
    frames = win[sel]
    worst = 0.0
    for b in body:
        run0 = None
        for f in frames:
            z = P[f, b, 2]
            y = P[f, b, 1]
            on_ground = z <= 0.03 and not (near < y < far)
            if on_ground:
                if run0 is None:
                    run0 = P[f, b, :2].copy()
                worst = max(worst, float(np.linalg.norm(P[f, b, :2] - run0)))
            else:
                run0 = None
    seg = res['seg_S']
    off = None
    if 'ref0_orig' in seg:
        off = np.asarray(res['fin_S']['ref_out'], float)[frames, :2] - np.asarray(seg['ref0_orig'], float)[frames, :2]
    off_range = (None if off is None else
                 dict(min_m=[float(x) for x in off.min(axis=0)], max_m=[float(x) for x in off.max(axis=0)],
                      max_step_cm=float(np.max(np.linalg.norm(np.diff(off, axis=0), axis=1)) * 100.0)))
    return dict(skate_max_m=worst, ref_offset_xy=off_range)


# ------------------------------------------------------------------------------------------------ one case
def run_case(ctx, t_W, kp, kr, D_S, label, write_dir=None):
    """The dive11 chain at window t_W with (kp, kr): dive9's SlideOver lift, the SlideOver lift passes, the Exit lift with the
    foot IK, the metrics. Returns a dict. If write_dir is given, the patches, mapping and sheet are written there."""
    with at_window(ctx, t_W), lift_grid():
        case = D9.make_case9(T_TO, kp, kr, S_SCALE, SHIFT)
        A, t0, t1 = D8.bump_for(ctx, case)
        F5.BUMP.update(t0=t0, t1=t1, A=A)
        res9 = D9.solve9(ctx, case, A, t0, t1)
        F5.BUMP.update(t0=t0, t1=t1, A=A)
        lift_S, s_info = D11.solve_S11(ctx, case, res9, D11_DELAY)
        case_a = D9.make_case9(T_TO, kp, kr, S_SCALE, SHIFT)
        res = D11.solve_E11(ctx, case_a, lift_S, D11_DELAY, use_ik=True)
        m = D11.run_metrics(ctx, res, case_a, D11_DELAY)
        fl_roll = trunk_speed_deg_s(ctx, res, case_a, t_W)
        skate = contact_skate_m(ctx, res, case_a, t_W)
        out = dict(label=label, D_S=D_S, t_W=float(t_W), k_plant=kp, k_roll=kr, A=A, t0=t0, t1=t1, res=res, m=m,
                   case=case_a, lift_S=lift_S, s_info=s_info, trunk=fl_roll, skate=skate,
                   lift_S_lift_res9=np.array(res9['lift_final'], float))
        if write_dir is not None:
            write_patches(ctx, res, D_S, write_dir, int(m['feet_free_to_frame']))
            write_sheet(ctx, res, case_a, D_S, write_dir, label)
    return out


def write_patches(ctx, res, D_S, out_dir, feet_to):
    """The two patches and the mapping for one variant. The SlideOver carries its shifted events (D_S); the Exit carries
    dive11's delay15 events (D11.event_tracks_delay15), unchanged. feet_to: the Exit's aux feet-free frame (as dive11)."""
    os.makedirs(out_dir, exist_ok=True)
    rig = ctx.rig
    so = D8.patch_dict(rig, res['fin_S']['rot_tracks'], res['fin_S']['trans_keys'], ctx.so['dump']['Length'])
    so_aux = AT.add_aux_tracks(so, ctx.so['dump'], D9.TAKEOFF_FRAME, None)
    so_aux['Tracks'] = list(so_aux['Tracks']) + slideover_event_tracks(ctx.so['dump'], D_S)
    AT.write_patch(os.path.join(out_dir, SO_HEX + '.baked.json'), so_aux)
    ex = D8.patch_dict(rig, res['fin_E']['rot_tracks'], res['fin_E']['trans_keys'], ctx.ex_dump['Length'])
    ex_aux = AT.add_aux_tracks(ex, ctx.ex_dump, 0, feet_to)
    ex_aux['Tracks'] = list(ex_aux['Tracks']) + D11.event_tracks_delay15(ctx)
    AT.write_patch(os.path.join(out_dir, EXIT_HEX + '.baked.json'), ex_aux)
    m = dict(clips=[
        DC2.mapping_entry(SO_HEX, SO_NAME, SO_HEX + '.baked.json', True, override=override_for(D_S)),
        DC2.mapping_entry(EXIT_HEX, EXIT_NAME, EXIT_HEX + '.baked.json', True, only_after=[SO_NAME],
                          override=override_for(D_S)),
    ])
    with open(os.path.join(out_dir, MAPPING_NAME), 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=1)
    return ex_aux


def write_sheet(ctx, res, case, D_S, out_dir, label):
    """dive11's joined sheet and GIF, written to this variant's folder (dive11's write_sheet11 writes to dive11's paths)."""
    PfS = res['world']['_PS']
    PfE = D3._world(res['seg_E']['seq'], res['seg_E']['dump'], res['fin_E']['final'], res['fin_E']['ref_out'],
                    res['fin_E']['disp'])
    PoS = acuclip.world_positions(ctx.so['seq'], ctx.so['dump'])
    PoE = acuclip.world_positions(ctx.ex_seq, ctx.ex_dump)
    arrS = dict(Pf=PfS, Po=PoS)
    arrE = dict(Pf=PfE, Po=PoE)
    tl = B.Tl(ctx, case, res['w_exit'])
    S = dict(F=res['seg_S']['F'], disp=ctx.disp_so, name=SO_NAME)
    E = dict(F=res['seg_E']['F'], disp=ctx.disp_ex, name=EXIT_NAME)
    title = ('dive12 %s (SlideOver events +%d): k_plant %.1f, k_roll %.1f, W %.3f s, floor min %.1f cm'
             % (label, D_S, case.tm.k_plant, case.tm.k, ctx.t_W, D11.floor_min_cm(D11.floor_phases(ctx, res, case))))
    BCh.plot_joined(os.path.join(out_dir, 'joined_sheet.png'), ctx.rig, S, arrS, E, arrE, ctx.ob, tl, title, show_root=True)
    BCh.write_gif(os.path.join(out_dir, 'joined_sheet.gif'), ctx.rig, S, arrS, E, arrE, ctx.ob, tl)


# ------------------------------------------------------------------------------------------------ main
def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    return x


def digest_run(ctx, run):
    """The numbers of one run for the summary and the report (floor per phase, metrics, trunk, skate, IK, contact)."""
    res, m, case = run['res'], run['m'], run['case']
    fl = m['floor']
    d = dict(label=run['label'], D_S=run['D_S'], t_W=run['t_W'], k_plant=run['k_plant'], k_roll=run['k_roll'],
             margin_s=margin_of(run['t_W'], run['k_plant'], run['k_roll']), feet_down_s=float(case.tm.t_feet),
             roll_s=float(case.tm.t_feet - case.tm.t_pe), plant_s=float(case.tm.t_pe - case.tm.t_pl0),
             floor=fl, floor_min_cm=m['floor_min_cm'], floor_pass=m['floor_pass'], trunk=run['trunk'],
             skate=run['skate'], ik=res['ik'], chosen_it=res['chosen_it'], contact_hist=res['contact_hist'],
             lift_S_passes=run['s_info']['iters'], lift_S_extra_cm=float(np.max(run['lift_S'] - run['lift_S_lift_res9'])) * 100.0,
             lift_E_max_cm=float(res['lift_E_max_cm']), lift_E_min_cm=float(res['lift_E_min_cm']),
             fails10=m.get('fails10', []), metrics={k: v for k, v in m.items() if k not in ('floor', 'fails10')})
    return _jsonable(d)


def main(argv=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    ctx = F5.make_ctx()
    t_W0 = float(ctx.t_W)
    print('W0 (clip s) = %.4f' % t_W0, flush=True)
    summary = dict(t_W0=t_W0, F0=F0, takeoff=dict(T_to=T_TO, s=S_SCALE, shift=SHIFT), variants={}, screen={})
    # the dive11 chain at the old window (k 2.0 / 3.0, no event shift): the baseline for the jarringness and floor numbers
    base = run_case(ctx, t_W0, 2.0, 3.0, 0, 'dive11 baseline')
    summary['baseline'] = digest_run(ctx, base)
    print('baseline floor min %.2f cm, trunk max %.1f deg/s' % (base['m']['floor_min_cm'], base['trunk']['max_deg_s']),
          flush=True)
    summary['screen']['W0'] = _jsonable(check_upright(ctx, 2.0, 3.0))
    for name, D_S in VARIANTS:
        t_W = t_W0 + D_S / FPS
        with at_window(ctx, t_W):
            ok_rows, tried = upright_candidates(ctx, t_W)
        if not ok_rows:
            raise RuntimeError('%s: no upright timing with a %.2f s margin' % (name, W_MARGIN))
        attempts, accepted = [], None
        for pick in ok_rows:
            kp, kr = pick['k_plant'], pick['k_roll']
            print('%s: try k_plant %.2f, k_roll %.2f (margin %.3f s)' % (name, kp, kr, pick['margin_s']), flush=True)
            run = run_case(ctx, t_W, kp, kr, D_S, name)
            dg = digest_run(ctx, run)
            attempts.append(dict(k_plant=kp, k_roll=kr, margin_s=pick['margin_s'], floor_min_cm=dg['floor_min_cm'],
                                 floor_pass=dg['floor_pass'], floor=dg['floor']))
            print('%s: floor min %.2f cm, pass %s, per phase %s' % (
                name, dg['floor_min_cm'], dg['floor_pass'],
                {k: (None if v['min_cm'] is None else round(v['min_cm'], 2)) for k, v in dg['floor'].items()}), flush=True)
            if dg['floor_pass']:
                accepted = (run, dg)
                break
        if accepted is None:
            raise RuntimeError('%s: no upright timing passes the floor rule' % name)
        run, dg = accepted
        write_run(ctx, run, D_S, variant_dir(name))
        dg['search'] = _jsonable(tried)
        dg['attempts'] = _jsonable(attempts)
        dg['slideover_events'] = _jsonable(slideover_event_table(ctx.so['dump'], D_S))
        summary['variants'][name] = dg
        print('%s: accepted k_plant %.2f, k_roll %.2f; trunk max %.1f deg/s; tilt W %.1f' % (
            name, dg['k_plant'], dg['k_roll'], dg['trunk']['max_deg_s'], dg['metrics']['tilt_W']), flush=True)
    summary['exit_events_delay15'] = _jsonable(D10.event_table(ctx.ex_dump, D11_DELAY))
    with open(SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(_jsonable(summary), f, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
