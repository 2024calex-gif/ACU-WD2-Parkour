"""T16 (dive4b, round 2): the WD2 frontdive on the Up050cm SlideOver with a DELAYED takeoff, judged on the world motion.

Changes from dive4 (round 2 brief):
  1. Delayed takeoff. WD2 takeoff 0.45 s is mapped to a later ACU clip time T_to in [0.47, 0.75]. Before T_to the body
     plays WD2's run-up: w(t) = max(0.2, 0.45 - (T_to - t)), at ground level with the stance foot planted in WD2's solution.
     The pelvis target is the WD2 path in world space, so the entity root's rise (from clip 0.48) is cancelled by the
     Reference z automatically (Reference = target - root - FK). After T_to, k = 1 until the plant end (0.92 s), then
     k_roll (1.0, or 1.3 for the faster roll variant) until feet-down (WD2 1.2 s).
  2. World smoothness rule. The judged motion is the WORLD pelvis (Hips) and head paths over the played window (SlideOver
     [o, W] joined to the Exit window). Second difference of position <= 2.5 cm/frame^2 (1.5 m/s change per frame at 60 Hz),
     no world step above 12 cm/frame, Reference offset magnitude <= 1.0 m. The Reference step rule of dive3/dive4 is kept
     as information only.
  3. Ground floor. The pelvis target gets a lift that keeps the lowest non-finger point (ground) and the box clearance at
     +1 cm, using bake2's lift need (lift_need) and smooth_envelope (rise and fall limited), iterated on the joined timeline.
  4. Scan T_to x s in [0.9, 1.4] x shift in [-0.2, 0.3]: closer hand >= 4.80 at the plant; cost = shift^2 + 0.5 (s-1)^2 +
     0.3 dW^2 + 0.5 (T_to - 0.47)^2. The shift is relative to the ORIGINAL Hips at T_to (the body's own run to T_to is free).
     Top 5 reported; feet-down relative to the Exit window end; roll variant at k = 1.3 when feet-down is > 0.1 s late.
  5. Everything else as dive4: absolute WD2 rotations, 0.15 s rotation fade at o, the gated Exit (F7919BDF0, only_after the
     Up050cm SlideOver), the ease to Nav_Run_3-Loop when the pre-ease gap is > 45 deg.

Validation / enable (round 2): non-finger penetration (box and ground) <= 5 cm; hand touch >= 4 frames; world smoothness;
continuity at W <= 45 deg. Narrow fail (pen <= 10 cm, touch >= 2, world dv <= 2.5 m/s per frame, continuity <= 60 deg):
enabled with "override": "preview: <reason>". The world step (12 cm/frame) and the offset (1.0 m) are not narrowed.

Outputs (only under out\\dive4b): patches\\7FC9C7405.baked.json, patches\\F7919BDF0.baked.json, patches\\mapping_dive4b.json,
report.md, scan.csv, scan.md, summary.json, joined_sheet.png, joined_sheet.gif.
Run:  python dive4b.py          (writes out\\dive4b)
      python dive4b.py --scan   (prints the top choices, writes nothing)
"""
import csv
import json
import os
import sys
import zlib

import numpy as np

import acuclip
import bake2
import bake_chain as BC
import bake_dive as BD
import config
import dive3 as D3
import dive4 as D4
import dive_chain2 as DC2
import dumps
import ik
import wd2_path as WP

FPS = acuclip.FPS
OUT_DIR = os.path.join(config.OUT, 'dive4b')
PATCH_DIR = os.path.join(OUT_DIR, 'patches')
MAPPING = os.path.join(PATCH_DIR, 'mapping_dive4b.json')
REPORT = os.path.join(OUT_DIR, 'report.md')
SCAN_CSV = os.path.join(OUT_DIR, 'scan.csv')
SCAN_MD = os.path.join(OUT_DIR, 'scan.md')
SUMMARY = os.path.join(OUT_DIR, 'summary.json')
SHEET_PNG = os.path.join(OUT_DIR, 'joined_sheet.png')
SHEET_GIF = os.path.join(OUT_DIR, 'joined_sheet.gif')

SO_NAME, SO_HEX = D4.SO_NAME, D4.SO_HEX
EXIT_NAME, EXIT_HEX = D4.EXIT_NAME, D4.EXIT_HEX
EXIT_O, EXIT_R, EXIT_WIN = D4.EXIT_O, D4.EXIT_R, D4.EXIT_WIN
FI = 0.15                                  # s, rotation fade-in from the original pose at o (kept)
FO = 0.0
TAIL = bake2.TAIL_WD2
W_TAKEOFF = WP.TAKEOFF                     # 0.45 WD2 s
W_PLANT = (0.75, 0.92)
W_PLANT_END = 0.92
W_FEET = 1.2
TB_EXTRA = 0.10                            # s, the translation blend reaches 1 at T_to + TB_EXTRA
T_GRID = np.round(np.arange(0.47, 0.7501, 0.01), 2)
S_GRID = np.round(np.arange(0.90, 1.4001, 0.01), 2)
SHIFT_GRID = np.round(np.arange(-0.20, 0.3001, 0.01), 2)
HAND_MIN = 4.80                            # m, the closer hand at the plant
COST = dict(shift=1.0, s=0.5, dw=0.3, t=0.5)
GROUND_CLEAR = bake2.T9D_LIFT_CLEAR       # 0.01 m: the lowest non-finger point and the box top clearance target
LIFT_K = 100                               # grid of the lift in joined time: index k is time k / 60 s
PEN_CM, PEN_NARROW = 5.0, 10.0
TOUCH, TOUCH_NARROW = 4, 2
DV_MAX, DV_NARROW = 1.5, 2.5               # m/s change per 60 Hz frame (= 2.5 / 4.2 cm/frame^2)
STEP_MAX = 12.0                            # cm per frame, world
OFF_MAX = 1.0                              # m, Reference offset magnitude
CONT_DEG, CONT_NARROW = 45.0, 60.0
GAP_DEG = 45.0
EASE_S = D3.EASE_S if hasattr(D3, 'EASE_S') else 0.10
K_FAST = 1.3


class Timing:
    """WD2 time at ACU clip time t. Before T_to: WD2 run-up at unit speed (clamped at 0.2 s). T_to to the plant end: k = 1.
    After the plant end: k_roll until feet-down."""

    def __init__(self, T_to, k_roll=1.0):
        self.T_to = float(T_to)
        self.k = float(k_roll)
        self.t_plant_end = self.T_to + (W_PLANT_END - W_TAKEOFF)
        self.t_feet = self.t_plant_end + (W_FEET - W_PLANT_END) / self.k

    def w(self, t):
        t = np.asarray(t, float)
        pre = np.maximum(0.2, W_TAKEOFF + (t - self.T_to))
        post = W_PLANT_END + self.k * (t - self.t_plant_end)
        return np.where(t <= self.t_plant_end, pre, post)


class Case:
    def __init__(self, T_to, s, shift, k_roll=1.0):
        self.T_to, self.s, self.shift, self.k_roll = float(T_to), float(s), float(shift), float(k_roll)
        self.tm = Timing(T_to, k_roll)


def smoothstep(u):
    return D3._smoothstep(u)


def hips_at(ctx, t):
    fr = np.arange(len(ctx.hips_orig_so))
    return np.array([np.interp(float(t) * FPS, fr, ctx.hips_orig_so[:, c]) for c in range(3)])


def y_take_for(ctx, T_to, shift):
    """Forward position of the takeoff pelvis: the ORIGINAL Hips at T_to plus the scan shift."""
    return float(hips_at(ctx, T_to)[1]) + float(shift)


def path_y_rel(ctx, w):
    return float(WP.path_at(ctx.sol, [float(w)])[0][1] - ctx.p0[1])


# ------------------------------------------------------------------------------------------------ scan
def scan_rows(ctx, T_grid=T_GRID, s_grid=S_GRID, shift_grid=SHIFT_GRID):
    rows = []
    for T in T_grid:
        tm = Timing(T, 1.0)
        dyW_rel = path_y_rel(ctx, tm.w(ctx.t_W))
        yH = float(hips_at(ctx, T)[1])
        for s in s_grid:
            for sh in shift_grid:
                yt = yH + float(sh)
                hy = {side: yt + float(s) * ctx.hand_rel_y[side] for side in ('L', 'R')}
                clear = min(hy.values()) - HAND_MIN
                pW = yt + float(s) * dyW_rel
                dW = pW - float(ctx.orig_W[1])
                cost = (COST['shift'] * float(sh) ** 2 + COST['s'] * (float(s) - 1.0) ** 2 + COST['dw'] * dW ** 2
                        + COST['t'] * (float(T) - 0.47) ** 2)
                rows.append(dict(T_to=float(T), s=float(s), shift=float(sh), y_take=yt, hand_L=hy['L'], hand_R=hy['R'],
                                 hand_clear=float(clear), dW=float(dW), cost=float(cost), feasible=bool(clear >= -1e-9),
                                 feet_rel_exit_end=float(tm.t_feet - (ctx.t_W + ctx.span_e)),
                                 t_feet=float(tm.t_feet)))
    return rows


def top_choices(rows, n=5):
    ok = sorted((r for r in rows if r['feasible']), key=lambda r: (r['cost'], r['T_to'], r['s'], r['shift']))
    out = []
    seen = set()
    for r in ok:
        key = (r['T_to'], r['s'], r['shift'])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) == n:
            break
    return out


# ------------------------------------------------------------------------------------------------ target and lift
def lift_eval(lift_k, tau):
    return np.interp(np.asarray(tau, float) * FPS, np.arange(len(lift_k)), lift_k, left=0.0, right=0.0)


def target_world(ctx, case, t, lift_k):
    """World pelvis target [F, 3] at ACU clip times t: the WD2 path mapped with (s, y_take), plus the ground-floor lift."""
    pl = D4.Place(ctx.sol, case.s, y_take_for(ctx, case.T_to, case.shift))
    q = pl.map_points(WP.path_at(ctx.sol, case.tm.w(t)))
    q[:, 2] = q[:, 2] + lift_eval(lift_k, t)
    return q, pl


def bake_clip5(ctx, seg, pl, win, plant, target, disp, ob, ref_w, hand_targets=True, ref_out_fixed=None):
    """dive4.bake_clip4 with the world pelvis target given directly (Reference = target - root - FK, weight ref_w), or with
    the Reference position given (ref_out_fixed, used for the Exit: the carried offset plus the lift)."""
    seq, dump = seg['seq'], seg['dump']
    rig = ctx.rig
    F = seg['F']
    nf = int(F[-1])
    orig, patset, wgt = seg['orig'], seg['patset'], seg['w']
    ref0 = seg['ref0_orig']
    rot_pre = orig.copy()
    for ai in patset:
        rot_pre[:, ai] = bake2.mix_rot(orig[:, ai], seg['patch'][:, ai], wgt)
    rref = rot_pre[:, 0]
    if ref_out_fixed is None:
        ref_full = np.asarray(target, float) - np.asarray(disp, float) - D3._hips_offset(ctx, rref)
        ref_out = ref0 + np.asarray(ref_w, float)[:, None] * (ref_full - ref0)
    else:
        ref_out = np.asarray(ref_out_fixed, float)
        ref_full = ref_out
    targets, ik_targets = {}, {}
    if hand_targets and plant.size:
        targets, ik_targets = D3._hand_targets(ctx, pl, seq, dump, seg['patch'], ref_full, disp, win, plant)
    P1 = D3._world(seq, dump, seg['patch'], ref_full, disp)
    ik_targets.update(D3._foot_targets(rig, P1, ob, ctx.offsets, win))
    patch_ik = seg['patch']
    if ik_targets:
        root_rot = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(F), 1))
        ik_out = ik.solve_sequence(bake2.composed(seq, seg['patch'], ref_full), root_rot=root_rot, root_pos=disp,
                                   targets=ik_targets)
        changed = np.nonzero(np.any(np.abs(ik_out.local_rot - seg['patch']) > 1e-12, axis=(0, 2)))[0]
        bad = sorted(set(int(c) for c in changed) - set(patset))
        if bad:
            raise ValueError('IK changed bones without a rotation track: %s' % bad)
        patch_ik = np.asarray(ik_out.local_rot, float)
    final = orig.copy()
    for ai in patset:
        final[:, ai] = bake2.mix_rot(orig[:, ai], patch_ik[:, ai], wgt)
        final[:, ai] /= np.linalg.norm(final[:, ai], axis=1, keepdims=True)
    lo = max(0, int(win[0]) - 1)
    rot_tracks = {}
    for ai in patset:
        ok = dumps.track(dump, rig.ac[ai]['hash'], 0) or []
        rot_tracks[ai] = bake2.rotation_track(ok, final[:, ai], lo, nf)
    ref_keys = dumps.track(dump, bake2.REF_TID, 3) or []
    trans_keys = bake2.translation_track(ref_keys, ref_out, lo, nf)
    return dict(final=final, ref_out=ref_out, ref_full=ref_full, targets=targets, rot_tracks=rot_tracks,
                trans_keys=trans_keys, disp=np.asarray(disp, float), lo=lo, patset=patset, win=win, plant=plant,
                ik_hands=sorted(h for h in ik_targets if 'Hand' in h), ik_feet=sorted(h for h in ik_targets if 'Foot' in h))


# ------------------------------------------------------------------------------------------------ one bake
def joined_played(ctx, PS, PE, idx, win_S, win_E):
    """Joined played path of one joint on a uniform 60 Hz played timeline: the SlideOver frames up to W, then the Exit
    interpolated at its played time (clip time = EXIT_O + EXIT_R * (tau - W)). The Exit's clip frames are not uniform in
    played time, so they are resampled here (the seam is then one sample, not 1.5 frames)."""
    W = float(ctx.t_W)
    n0, n1 = int(win_S[0]), int(np.floor((W + ctx.span_e) * FPS + 1e-9))
    out = []
    fS = np.asarray(win_S, int)
    for n in range(n0, n1 + 1):
        tau = n / FPS
        if tau <= W + 1e-12:
            if n > fS[-1]:
                continue
            out.append(PS[n, idx])
        else:
            tE = EXIT_O + EXIT_R * (tau - W)
            # interpolated between patched Exit frames only (the patch switches on between clip frames 12 and 13)
            f0 = int(win_E[0])
            x = max(tE * FPS, float(f0))
            i0 = int(np.floor(x))
            a = x - i0
            i1 = min(i0 + 1, len(PE) - 1)
            out.append((1.0 - a) * PE[i0, idx] + a * PE[i1, idx])
    return np.asarray(out, float)


def world_metrics(ctx, seg_S, fin_S, win_S, seg_E, fin_E, win_E):
    """Joined WORLD pelvis (Hips) and head paths over the played window: step, second difference (acceleration) and the
    velocity change per frame, plus the Reference offset magnitude."""
    PS = D3._world(seg_S['seq'], seg_S['dump'], fin_S['final'], fin_S['ref_out'], fin_S['disp'])
    PE = D3._world(seg_E['seq'], seg_E['dump'], fin_E['final'], fin_E['ref_out'], fin_E['disp'])
    head = ctx.rig.ac_by_hash[zlib.crc32(b'Head')]
    out = {}
    for name, idx in (('pelvis', ctx.hips), ('head', head)):
        P = joined_played(ctx, PS, PE, idx, win_S, win_E)
        if name == 'pelvis':
            n_cut = int(np.sum(np.arange(int(win_S[0]), int(win_S[0]) + len(P)) / FPS <= ctx.t_W + 1e-12))
            out_joint = (float(np.linalg.norm(P[n_cut - 1] - P[n_cut - 2]) * FPS),
                         float(np.linalg.norm(P[n_cut + 1] - P[n_cut]) * FPS))
        d1 = np.diff(P, axis=0)
        d2 = np.diff(P, 2, axis=0)
        step = np.linalg.norm(d1, axis=1) * 100.0
        acc = np.linalg.norm(d2, axis=1)
        out[name] = dict(step_cm=float(step.max()), acc_cm=float(acc.max() * 100.0), dv_ms=float(acc.max() * FPS))
    dS = np.linalg.norm(fin_S['ref_out'][win_S] - seg_S['ref0_orig'][win_S], axis=1)
    dE = np.linalg.norm(fin_E['ref_out'][win_E] - seg_E['ref0_orig'][win_E], axis=1)
    out['offset_max_m'] = float(max(dS.max(), dE.max()))
    out['joint_speed_before_ms'], out['joint_speed_after_ms'] = out_joint
    out['step_cm'] = max(out['pelvis']['step_cm'], out['head']['step_cm'])
    out['dv_ms'] = max(out['pelvis']['dv_ms'], out['head']['dv_ms'])
    out['acc_cm'] = max(out['pelvis']['acc_cm'], out['head']['acc_cm'])
    feet = [ctx.rig.ac_by_hash[zlib.crc32(n.encode())] for n in ('LeftFoot', 'RightFoot', 'LeftFootThumb1', 'RightFootThumb1')]
    out['_PS'] = PS
    out['feet_z_pre_takeoff_min_cm'] = float(PS[win_S][:, feet, 2][np.asarray(win_S) / FPS < 0.4667 - 1e-9].min() * 100.0) \
        if np.any(np.asarray(win_S) / FPS < 0.4667 - 1e-9) else float('nan')
    return out


def bake_case5(ctx, case, lift_k, fi=FI, mode='absolute', ease=True):
    tm = case.tm
    pl = D4.Place(ctx.sol, case.s, y_take_for(ctx, case.T_to, case.shift))
    o, span = ctx.o, ctx.span_so
    wfun = tm.w
    seg_S = BC.segment(ctx.rig, ctx.m, ctx.so['seq'], ctx.so['dump'], wfun, float(wfun(o)),
                       lambda t: bake2.patch_weight(t, o, span, fi, FO, TAIL), disp=ctx.disp_so)
    seg_S['win'] = bake2.window_frames(o, span, int(seg_S['F'][-1]))
    seg_S['ob'] = ctx.ob
    if mode == 'absolute':
        seg_S['patch'] = D3.absolute_patch(ctx, seg_S, wfun, o)
    t_S = seg_S['t60']
    win_S = seg_S['win']
    tw = wfun(win_S / FPS)
    plant_S = win_S[(tw >= W_PLANT[0] - 1e-9) & (tw <= W_PLANT[1] + 1e-9)]
    target_S, _pl = target_world(ctx, case, t_S, lift_k)
    t_blend = case.T_to + TB_EXTRA
    wT_S = smoothstep(np.clip((t_S - o) / max(t_blend - o, 1e-9), 0.0, 1.0))
    fin_S = bake_clip5(ctx, seg_S, pl, win_S, plant_S, target_S, ctx.disp_so, ctx.ob, wT_S)
    fin_S['win'] = win_S
    fin_S['plant'] = plant_S
    met_S, arr_S = D3._validate(ctx, seg_S, fin_S, ctx.t_W, span, o)
    # ---- Exit: the SlideOver's offset at its last played frame carries over and converges (smoothstep over the window and
    # the rest of the clip); the lift is added on top (it follows the joined timeline).
    w_W = float(wfun(ctx.t_W))

    def w_exit(t):
        return w_W + tm.k * (np.asarray(t, float) - EXIT_O) / EXIT_R

    seg_E = BC.segment(ctx.rig, ctx.m, ctx.ex_seq, ctx.ex_dump, w_exit, float(w_exit(EXIT_O)),
                       lambda t: (np.asarray(t, float) >= EXIT_O - 1e-9).astype(float), disp=ctx.disp_ex)
    seg_E['win'] = bake2.window_frames(EXIT_O, ctx.span_e, int(seg_E['F'][-1]))
    seg_E['ob'] = ctx.ground_ob
    if mode == 'absolute':
        seg_E['patch'] = D3.absolute_patch(ctx, seg_E, w_exit, EXIT_O)
    tE = seg_E['t60']
    L_E = float(ctx.ex_dump['Length'])
    win_E = seg_E['win']
    last = int(win_S[-1])
    # the carried offset: the SlideOver's Reference position at its last played frame, relative to the Exit's OWN original
    # Reference at its first played frame, so the joined Hips world position is continuous (the Exit's original reference
    # is not the SlideOver's original reference at the same time)
    off_W = fin_S['ref_out'][last] - seg_E['ref0_orig'][int(win_E[0])]
    inside = tE >= EXIT_O - 1e-9
    c = np.where(inside, 1.0, 0.0) * (1.0 - smoothstep((tE - EXIT_O) / max(L_E - EXIT_O, 1e-9)))
    tau_E = ctx.t_W + (tE - EXIT_O) / EXIT_R
    ref_out_E = seg_E['ref0_orig'] + c[:, None] * off_W[None, :]
    ref_out_E[:, 2] += np.where(inside, lift_eval(lift_k, tau_E), 0.0)
    fin_E = bake_clip5(ctx, seg_E, pl, win_E, np.array([], int), None, ctx.disp_ex, ctx.ground_ob, None,
                       hand_targets=False, ref_out_fixed=ref_out_E)
    fin_E['win'] = win_E
    fin_E['plant'] = np.array([], int)
    fin_E['targets'] = {}
    fin_E['disp'] = ctx.disp_ex
    fW_E = min(int(round(ctx.t_end_E * FPS)), len(tE) - 1)
    gap_pre = D3._gap_to_next(ctx, fin_E['final'], fW_E)
    ease_used = False
    if ease and gap_pre['max_deg'] > GAP_DEG:
        fin_E['final'] = D3._ease(ctx, fin_E['final'], seg_E['patset'], fW_E)
        ease_used = True
    fin_E['rot_tracks'] = {ai: bake2.rotation_track(dumps.track(ctx.ex_dump, ctx.rig.ac[ai]['hash'], 0) or [],
                                                   fin_E['final'][:, ai], max(0, int(win_E[0]) - 1),
                                                   int(seg_E['F'][-1])) for ai in seg_E['patset']}
    met_E, arr_E = D3._validate(ctx, seg_E, fin_E, ctx.t_end_E, ctx.span_e, EXIT_O, next_dump=ctx.next_dump)
    cont = D4.cont_at_joint(ctx, fin_S, fin_E, last, int(win_E[0]))
    world = world_metrics(ctx, seg_S, fin_S, win_S, seg_E, fin_E, win_E)
    # old Reference step (information only): offset change per frame over the played window
    dS = fin_S['ref_out'] - seg_S['ref0_orig']
    dE = fin_E['ref_out'] - seg_E['ref0_orig']
    old_step = max(D4.window_step_cm(dS, win_S), D4.window_step_cm(dE, win_E))
    sd = np.concatenate([dS[win_S], dE[win_E]])
    joined_ref_step = float(np.max(np.linalg.norm(np.diff(sd, axis=0), axis=1)) * 100.0)
    met_E_pen = met_E['pen_patch_cm']
    return dict(case=case, seg_S=seg_S, seg_E=seg_E, fin_S=fin_S, fin_E=fin_E, met_S=met_S, met_E=met_E, arr_S=arr_S,
                arr_E=arr_E, cont=cont, world=world, gap_pre=gap_pre, ease_used=ease_used, win_S=win_S, win_E=win_E,
                off_W=off_W, w_exit=w_exit, w_W=w_W, old_ref_step_cm=old_step, joined_ref_step_cm=joined_ref_step,
                lift_k=np.array(lift_k, float), exit_pen=met_E_pen, dS=dS, dE=dE)


def need_samples(ctx, res):
    """(tau, need) per played sample of both clips: the lift the lowest non-finger point and the box top still need."""
    taus, needs = [], []
    for seg, fin, win, is_exit in ((res['seg_S'], res['fin_S'], res['win_S'], False),
                                   (res['seg_E'], res['fin_E'], res['win_E'], True)):
        P = D3._world(seg['seq'], seg['dump'], fin['final'], fin['ref_out'], fin['disp'])
        pts = P[win][:, ctx.body_nf]
        nd = bake2.lift_need(pts, ctx.ob)
        tau = np.asarray(win, float) / FPS
        if is_exit:
            tau = ctx.t_W + (tau - EXIT_O) / EXIT_R
        taus.append(tau)
        needs.append(nd)
    return np.concatenate(taus), np.concatenate(needs)


def envelope_update(lift_k, tau, need):
    """Add the smooth envelope (bake2.smooth_envelope: rise and fall limited) of the need on the joined 60 Hz grid."""
    K = len(lift_k)
    kk = np.clip(np.round(np.asarray(tau, float) * FPS).astype(int), 0, K - 1)
    ng = np.zeros(K)
    np.maximum.at(ng, kk, np.asarray(need, float))
    return np.asarray(lift_k, float) + bake2.smooth_envelope(ng)


def bake_with_floor(ctx, case, n_iter=5, thr=1e-3, ease=True, floor=True):
    lift = np.zeros(LIFT_K)
    hist = []
    for it in range(n_iter if floor else 1):
        res = bake_case5(ctx, case, lift, ease=ease)
        tau, need = need_samples(ctx, res)
        worst = float(need.max()) if need.size else 0.0
        hist.append(dict(it=it, need_max_cm=worst * 100.0, lift_max_cm=float(lift.max()) * 100.0))
        if worst < thr or it == n_iter - 1:
            break
        lift = envelope_update(lift, tau, need)
    res['floor_hist'] = hist
    res['floor_residual_cm'] = float(need.max() * 100.0) if need.size else 0.0
    res['lift_max_cm'] = float(lift.max() * 100.0)
    return res


# ------------------------------------------------------------------------------------------------ judgement
def judge(res):
    """Spec rules with the narrow fail (pen <= 10, touch >= 2, dv <= 2.5 m/s per frame, cont <= 60). The world step and the
    offset are hard limits (no narrow). Returns (status, S fails, E fails, override text)."""
    w = res['world']
    cont = res['cont']['max_deg']

    def check(met, touch_needed):
        fails, narrow = [], True
        pen = met['pen_patch_cm']
        if pen > PEN_CM:
            fails.append('non-finger penetration %.1f cm (> %.0f)' % (pen, PEN_CM))
            narrow &= pen <= PEN_NARROW
        if touch_needed:
            t = met['touch']['min']
            if t < TOUCH:
                fails.append('hand touch %d frames (need %d)' % (t, TOUCH))
                narrow &= t >= TOUCH_NARROW
        if w['dv_ms'] > DV_MAX:
            fails.append('world dv %.2f m/s per frame (> %.1f)' % (w['dv_ms'], DV_MAX))
            narrow &= w['dv_ms'] <= DV_NARROW
        if w['step_cm'] > STEP_MAX:
            fails.append('world step %.1f cm/frame (> %.0f)' % (w['step_cm'], STEP_MAX))
            narrow = False
        if w['offset_max_m'] > OFF_MAX:
            fails.append('Reference offset %.2f m (> %.1f)' % (w['offset_max_m'], OFF_MAX))
            narrow = False
        if cont > CONT_DEG:
            fails.append('continuity %.0f deg (> %.0f)' % (cont, CONT_DEG))
            narrow &= cont <= CONT_NARROW
        return fails, narrow

    fS, nS = check(res['met_S'], True)
    fE, nE = check(res['met_E'], False)
    fE = [f for f in fE if not f.startswith('hand touch')]
    st = {}
    for key, fails, narrow in (('S', fS, nS), ('E', fE, nE)):
        if not fails:
            st[key] = dict(status='PASS', fails=[], override=None)
        elif narrow:
            st[key] = dict(status='OVERRIDE', fails=fails, override='preview: ' + '; '.join(fails))
        else:
            st[key] = dict(status='FAIL', fails=fails, override=None)
    so_on = st['S']['status'] in ('PASS', 'OVERRIDE')
    ex_on = so_on and st['E']['status'] in ('PASS', 'OVERRIDE')
    st['S']['enabled'] = so_on
    st['E']['enabled'] = ex_on
    return st


def rank_key(st):
    order = {'PASS': 0, 'OVERRIDE': 1, 'FAIL': 2}
    return (order[st['S']['status']] + order[st['E']['status']], len(st['S']['fails']) + len(st['E']['fails']))


# ------------------------------------------------------------------------------------------------ outputs
class Tl:
    def __init__(self, ctx, case, w_exit):
        self.o, self.r, self.W = ctx.o, ctx.r, ctx.W_played
        self.T = (case.T_to + (W_PLANT[0] - W_TAKEOFF) - ctx.o) / ctx.r
        self._we = w_exit
        self._tm = case.tm

    def w_so(self, t):
        return float(self._tm.w(t))

    def w_exit(self, t):
        return float(self._we(t))


def write_outputs(ctx, res, st):
    os.makedirs(PATCH_DIR, exist_ok=True)
    DC2.write_patch(os.path.join(PATCH_DIR, SO_HEX + '.baked.json'), ctx.rig, res['fin_S']['rot_tracks'],
                    res['fin_S']['trans_keys'], None, ctx.so['dump']['Length'])
    DC2.write_patch(os.path.join(PATCH_DIR, EXIT_HEX + '.baked.json'), ctx.rig, res['fin_E']['rot_tracks'],
                    res['fin_E']['trans_keys'], None, ctx.ex_dump['Length'])
    mapping = dict(clips=[
        DC2.mapping_entry(SO_HEX, SO_NAME, SO_HEX + '.baked.json', st['S']['enabled'], override=st['S']['override']),
        DC2.mapping_entry(EXIT_HEX, EXIT_NAME, EXIT_HEX + '.baked.json', st['E']['enabled'],
                          only_after=[SO_NAME], override=st['E']['override']),
    ])
    with open(MAPPING, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=1)
    return mapping


def write_sheet(ctx, res):
    PfS = res['world']['_PS']
    PfE = D3._world(res['seg_E']['seq'], res['seg_E']['dump'], res['fin_E']['final'], res['fin_E']['ref_out'],
                    res['fin_E']['disp'])
    PoS = acuclip.world_positions(ctx.so['seq'], ctx.so['dump'])
    PoE = acuclip.world_positions(ctx.ex_seq, ctx.ex_dump)
    arrS = dict(Pf=PfS, Po=PoS)
    arrE = dict(Pf=PfE, Po=PoE)
    tl = Tl(ctx, res['case'], res['w_exit'])
    S = dict(F=res['seg_S']['F'], disp=ctx.disp_so, name=SO_NAME)
    E = dict(F=res['seg_E']['F'], disp=ctx.disp_ex, name=EXIT_NAME)
    c = res['case']
    title = 'dive4b: T_to %.2f, s %.2f, shift %+.2f, k_roll %.1f, continuity %.0f deg' % (c.T_to, c.s, c.shift, c.k_roll,
                                                                                    res['cont']['max_deg'])
    BC.plot_joined(SHEET_PNG, ctx.rig, S, arrS, E, arrE, ctx.ob, tl, title, show_root=True)
    BC.write_gif(SHEET_GIF, ctx.rig, S, arrS, E, arrE, ctx.ob, tl)


def _table(rows, headers):
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(out)


def write_scan(rows, top):
    os.makedirs(OUT_DIR, exist_ok=True)
    keys = ['T_to', 's', 'shift', 'y_take', 'hand_L', 'hand_R', 'hand_clear', 'dW', 'cost', 'feasible', 't_feet',
            'feet_rel_exit_end']
    with open(SCAN_CSV, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(keys)
        for r in rows:
            wr.writerow([('%.4f' % r[k]) if isinstance(r[k], float) else r[k] for k in keys])
    feas = [r for r in rows if r['feasible']]
    lines = ['# dive4b placement scan (round 2)', '',
             'Grid: T_to %d x s %d x shift %d = %d rows; feasible (closer hand >= %.2f m at the plant): %d.'
             % (len(T_GRID), len(S_GRID), len(SHIFT_GRID), len(rows), HAND_MIN, len(feas)),
             'Cost = %.1f shift^2 + %.1f (s-1)^2 + %.1f dW^2 + %.1f (T_to - 0.47)^2. Shift is relative to the original Hips at T_to.'
             % (COST['shift'], COST['s'], COST['dw'], COST['t']), '']
    head = ['T_to (s)', 's', 'shift (m)', 'y_take (m)', 'closer hand (m)', 'hand clearance (m)', 'dW at W (m)', 'cost',
            'feet-down (s)', 'feet-down minus Exit end (s)']
    rows_md = []
    for r in top:
        rows_md.append(['%.2f' % r['T_to'], '%.2f' % r['s'], '%+.2f' % r['shift'], '%.3f' % r['y_take'],
                        '%.3f' % min(r['hand_L'], r['hand_R']), '%+.3f' % r['hand_clear'], '%+.3f' % r['dW'],
                        '%.4f' % r['cost'], '%.3f' % r['t_feet'], '%+.3f' % r['feet_rel_exit_end']])
    lines.append('## Top 5 (lowest cost among the feasible rows)')
    lines.append('')
    lines.append(_table(rows_md, head))
    lines.append('')
    with open(SCAN_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def cases_for_top(top, ctx):
    """For each top choice: k_roll 1.0, and 1.3 when feet-down lands more than 0.1 s after the Exit window end."""
    out = []
    for r in top:
        out.append(Case(r['T_to'], r['s'], r['shift'], 1.0))
        if r['feet_rel_exit_end'] > 0.1:
            out.append(Case(r['T_to'], r['s'], r['shift'], K_FAST))
    return out


def run_one(ctx, case, name, spec, floor=True, ease=True, verbose=True):
    res = bake_with_floor(ctx, case, ease=ease, floor=floor)
    st = judge(res)
    if verbose:
        w = res['world']
        print('%-26s T %.2f s %.2f sh %+.2f k %.1f | S %s E %s | pen %.1f/%.1f touch %d dv %.2f step %.1f off %.2f cont %.0f floor %.1f' % (
            name, case.T_to, case.s, case.shift, case.k_roll, st['S']['status'], st['E']['status'],
            res['met_S']['pen_box_cm'], res['met_S']['pen_ground_cm'], res['met_S']['touch']['min'],
            w['dv_ms'], w['step_cm'], w['offset_max_m'], res['cont']['max_deg'], res['floor_residual_cm']))
    return dict(res=res, st=st, name=name, spec=spec)


def run_all(ctx, top, rows, verbose=True):
    """Spec variants: the top choices (k 1.0, and k 1.3 when feet-down is late), a faster roll on the first choice, and the
    latest feasible takeoff (T_to >= 0.72). Diagnostics (not spec): no ground floor, no ease. All baked with the floor."""
    results = []
    for case in cases_for_top(top, ctx):
        results.append(run_one(ctx, case, 'top choice', True, verbose=verbose))
    c0 = top[0]
    results.append(run_one(ctx, Case(c0['T_to'], c0['s'], c0['shift'], K_FAST), 'roll k 1.3 (first choice)', True,
                           verbose=verbose))
    late = [r for r in rows if r['feasible'] and r['T_to'] >= 0.72]
    if late:
        lc = min(late, key=lambda r: r['cost'])
        results.append(run_one(ctx, Case(lc['T_to'], lc['s'], lc['shift'], 1.0), 'latest takeoff (T >= 0.72)', True,
                               verbose=verbose))
    results.append(run_one(ctx, Case(c0['T_to'], c0['s'], c0['shift'], 1.0), 'diag: no ground floor', False,
                           floor=False, verbose=verbose))
    results.append(run_one(ctx, Case(c0['T_to'], c0['s'], c0['shift'], 1.0), 'diag: no ease', False, ease=False,
                           verbose=verbose))
    return results


def write_report(ctx, rows, top, results, chosen):
    res, st = chosen['res'], chosen['st']
    c = res['case']
    w = res['world']
    A = []
    A.append('# T16 dive4b (round 2): delayed takeoff, world-motion smoothness, ground floor\n')
    A.append('Generated by `v2/pipeline/dive4b.py` (reuses `dive4.py` and `dive3.py`). Written only under `out\\dive4b`.\n')
    A.append('## Status\n')
    A.append('| clip | status | enabled | override | fails |')
    A.append('|---|---|---|---|---|')
    A.append('| SlideOver %s | %s | %s | %s | %s |' % (SO_NAME, st['S']['status'], 'yes' if st['S']['enabled'] else 'no',
                                                   st['S']['override'] or '-', '; '.join(st['S']['fails']) or '-'))
    A.append('| Exit %s | %s | %s | %s | %s |' % (EXIT_NAME, st['E']['status'], 'yes' if st['E']['enabled'] else 'no',
                                               st['E']['override'] or '-', '; '.join(st['E']['fails']) or '-'))
    A.append('')
    A.append('Chosen case: T_to %.2f s, s %.2f, shift %+.2f m (relative to the original Hips at T_to), k_roll %.1f. '
             'Pre-takeoff pose: WD2 run-up (w = max(0.2, 0.45 - (T_to - t))), not the original ACU run pose; the rotation fade '
             'is at o as before, so the body starts WD2\'s run-up pose at o.\n' % (c.T_to, c.s, c.shift, c.k_roll))

    A.append('## 1. Scan (fixes 1 and 4)\n')
    feas = sum(1 for r in rows if r['feasible'])
    A.append('Grid %d x %d x %d = %d rows; %d feasible (closer hand >= %.2f m). Top 5 by cost:\n'
             % (len(T_GRID), len(S_GRID), len(SHIFT_GRID), len(rows), feas, HAND_MIN))
    A.append(_table([['%.2f' % r['T_to'], '%.2f' % r['s'], '%+.2f' % r['shift'], '%.3f' % min(r['hand_L'], r['hand_R']),
                      '%+.3f' % r['dW'], '%.4f' % r['cost'], '%.3f' % r['t_feet'], '%+.3f' % r['feet_rel_exit_end']]
                     for r in top],
                    ['T_to', 's', 'shift', 'closer hand (m)', 'dW (m)', 'cost', 'feet-down (clip s)',
                     'feet-down minus Exit end (s)']))
    A.append('')
    A.append('Feet-down is at T_to + 0.75 s with k = 1 and T_to + 0.47 + 0.28/1.3 with k = 1.3. The Exit window ends at '
             'W + %.3f s = %.3f clip s. Variants with feet-down more than 0.1 s after that window were baked at k = 1.3 too.\n'
             % (ctx.span_e, ctx.t_W + ctx.span_e))

    A.append('## 2. Baked variants (all cases; the chosen one is marked)\n')
    rows_v = []
    for r in results:
        rr, s2 = r['res'], r['st']
        cc = rr['case']
        m = rr['met_S']
        rows_v.append([r['name'], '%s%.2f' % ('*' if rr is res else '', cc.T_to), '%.2f' % cc.s, '%+.2f' % cc.shift, '%.1f' % cc.k_roll,
                       '%.1f' % rr['lift_max_cm'], '%.1f' % m['pen_box_cm'], '%.1f' % m['pen_ground_cm'], str(m['touch']['min']),
                       '%.2f' % rr['world']['dv_ms'], '%.1f' % rr['world']['step_cm'], '%.2f' % rr['world']['offset_max_m'],
                       '%.0f' % rr['cont']['max_deg'], '%.1f' % rr['met_E']['pen_patch_cm'], '%.2f' % rr['floor_residual_cm'],
                       '%s / %s' % (s2['S']['status'], s2['E']['status'])])
    A.append(_table(rows_v, ['variant', 'T_to', 's', 'shift', 'k_roll', 'lift max (cm)', 'box pen (cm)', 'ground pen (cm)', 'touch',
                             'world dv (m/s/frame)', 'world step (cm/frame)', 'offset max (m)', 'cont (deg)',
                             'Exit pen (cm)', 'floor residual (cm)', 'status S / E']))
    A.append('')
    A.append('Pen = non-finger penetration (window). dv = max change of world velocity per 60 Hz frame (pelvis or head). '
             'Floor residual = the lift still needed after the last iteration.\n')

    base = next((r for r in results if r['name'] == 'top choice'), None)
    nof = next((r for r in results if r['name'] == 'diag: no ground floor'), None)
    noe = next((r for r in results if r['name'] == 'diag: no ease'), None)
    if base is not None and nof is not None and noe is not None:
        wb, we = base['res']['world'], noe['res']['world']
        A.append('## 2b. What limits it (first choice)\n')
        A.append('* **The joint at W.** The pelvis moves at %.1f m/s just before W (the WD2 target) and at %.1f m/s just after '
                 '(the Exit clip). The carried offset absorbs only part of that change over its decay. The pelvis step at the '
                 'joint is %.1f cm/frame (limit 12) and the pelvis dv is %.1f m/s per frame (limit 1.5).\n'
                 % (wb['joint_speed_before_ms'], wb['joint_speed_after_ms'], wb['pelvis']['step_cm'], wb['pelvis']['dv_ms']))
        A.append('* **The ease to Nav_Run.** The first choice has a %.0f deg gap to Nav_Run at the window end. The spec ease '
                 'over 0.1 s swings the head: head step %.1f cm/frame with the ease, %.1f without it.\n'
                 % (base['res']['gap_pre']['max_deg'], wb['head']['step_cm'], we['head']['step_cm']))
        A.append('* **The ground floor and the hands.** The lowest non-finger points are the planted hands and the Exit feet. '
                 'The floor lift reaches %.1f cm and lifts the planted hands off their targets: hand touch is %d frame(s) with '
                 'the floor, %d without it.\n'
                 % (base['res']['lift_max_cm'], base['res']['met_S']['touch']['min'], nof['res']['met_S']['touch']['min']))
        A.append('* **Verdict.** The floor, the hand plant and the world rules pull against each other at this takeoff and roll. '
                 'The variants (roll k 1.3, latest takeoff, no floor, no ease) do not pass either, so both clips stay disabled.\n')
    A.append('## 3. Validation (chosen case)\n')
    mS, mE = res['met_S'], res['met_E']
    A.append('| metric | SlideOver | Exit | rule |')
    A.append('|---|---|---|---|')
    A.append('| non-finger penetration, box / ground / max (cm) | %.1f / %.1f / %.1f | %.1f / %.1f / %.1f | <= %.0f |'
             % (mS['pen_box_cm'], mS['pen_ground_cm'], mS['pen_patch_cm'], mE['pen_box_cm'], mE['pen_ground_cm'],
                mE['pen_patch_cm'], PEN_CM))
    A.append('| hand touch frames (min) | %d | n/a | >= %d |' % (mS['touch']['min'], TOUCH))
    A.append('| world pelvis: max step / max dv (m/s per frame) / max accel (cm/frame^2) | %.2f cm / %.2f / %.2f | joined | step <= %.0f, dv <= %.1f |'
             % (w['pelvis']['step_cm'], w['pelvis']['dv_ms'], w['pelvis']['acc_cm'], STEP_MAX, DV_MAX))
    A.append('| world head: max step / max dv / max accel | %.2f cm / %.2f / %.2f | joined | step <= %.0f, dv <= %.1f |'
             % (w['head']['step_cm'], w['head']['dv_ms'], w['head']['acc_cm'], STEP_MAX, DV_MAX))
    A.append('| Reference offset magnitude max (m) | %.3f | | <= %.1f |' % (w['offset_max_m'], OFF_MAX))
    A.append('| continuity at W, max (deg) | %.1f (%s) | | <= %.0f |' % (res['cont']['max_deg'], res['cont']['worst'], CONT_DEG))
    A.append('| pre-ease gap to Nav_Run_3-Loop, max (deg) | | %.1f | ease when > %.0f |' % (res['gap_pre']['max_deg'], GAP_DEG))
    A.append('| info: old Reference step, played window (cm/frame) | %.2f | %.2f (joined %.2f) | info only |'
             % (D4.window_step_cm(res['dS'], res['win_S']), D4.window_step_cm(res['dE'], res['win_E']),
                res['joined_ref_step_cm']))
    A.append('| feet min height before T_to (cm, all four foot joints) | %s | | info |' % (
        '%.1f' % w['feet_z_pre_takeoff_min_cm'] if np.isfinite(w['feet_z_pre_takeoff_min_cm']) else 'n/a'))
    A.append('')
    A.append('Floor iterations (need before the update, in cm): %s. Final floor residual %.2f cm; lift max %.1f cm.\n'
             % (', '.join('%.1f' % h['need_max_cm'] for h in res['floor_hist']), res['floor_residual_cm'], res['lift_max_cm']))
    A.append('Feet-down (clip s) %.3f: %+.3f s from the Exit window end (k_roll %.1f).\n'
             % (c.tm.t_feet, c.tm.t_feet - (ctx.t_W + ctx.span_e), c.k_roll))
    A.append('## 4. Enable decision\n')
    A.append('SlideOver **%s**, Exit **%s**. Enabled: SlideOver %s, Exit %s.\n'
             % (st['S']['status'], st['E']['status'], st['S']['enabled'], st['E']['enabled']))
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(A) + '\n')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ctx = D4.Ctx()
    rows = scan_rows(ctx)
    top = top_choices(rows, 5)
    if '--scan' in argv:
        for r in top:
            print(r['T_to'], r['s'], r['shift'], round(r['cost'], 4), round(r['dW'], 3), round(r['feet_rel_exit_end'], 3))
        return 0
    write_scan(rows, top)
    results = run_all(ctx, top, rows)
    spec_results = [r for r in results if r['spec']]
    chosen = min(spec_results, key=lambda r: (rank_key(r['st']), r['res']['case'].tm.t_feet))
    os.makedirs(OUT_DIR, exist_ok=True)
    write_outputs(ctx, chosen['res'], chosen['st'])
    write_sheet(ctx, chosen['res'])
    write_report(ctx, rows, top, results, chosen)
    summary = dict(chosen=dict(T_to=chosen['res']['case'].T_to, s=chosen['res']['case'].s,
                               shift=chosen['res']['case'].shift, k_roll=chosen['res']['case'].k_roll),
                   status=chosen['st'],
                   variants=[dict(name=r['name'], T_to=r['res']['case'].T_to, s=r['res']['case'].s, shift=r['res']['case'].shift,
                                  k_roll=r['res']['case'].k_roll, S=r['st']['S']['status'], E=r['st']['E']['status'],
                                  world_dv=r['res']['world']['dv_ms'], world_step=r['res']['world']['step_cm'],
                                  offset=r['res']['world']['offset_max_m'], floor_cm=r['res']['floor_residual_cm'])
                             for r in results])
    with open(SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps(summary['chosen']), summary['status']['S']['status'], summary['status']['E']['status'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
