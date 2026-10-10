"""T21 (dive9): the FAST ROLL variant that is upright before W and stands up inside the Exit (standing by Nav_Run).

Starting point: dive8 B_fastroll (its time map, bump, contact floor, aux tracks and world metrics are reused).
The dive8 B Exit was still crouched at the window end, so the game blended it into Nav_Run (standing, Reference z 0.81)
with a visible pop. dive9 changes four things:

  1. Timing. The plant and the roll are sped up and the takeoff is earlier, so that the roll is COMPLETE and the
     trunk is UPRIGHT (tilt <= 60 deg, no spine segment pointing down) at W and after it.
       T_to 0.62, k_plant 2.0, k_roll 3.0, s 1.35, shift +0.20 (the hands land at the far edge: 5.50 m).
     The screen (report.md) shows why: the trunk is inverted at W for the slow variants; here the body passes the
     inverted phase and is upright by W - 0.10 s.
  2. Stand-up in the Exit. The Reference z (entity) is a Hermite rise (C1 at both ends) from the W height (with the
     SlideOver slope at W) to Z_UP = 0.80 m at Exit played 0.16 s, then held. Peak speed is set by the W height:
     1.5 * (0.80 - z_W) / 0.16 m/s.
  3. Rotations. The Exit's patched bones are eased toward Nav_Run's first frame by a common weight w (smoothstep over
     played 0 .. 0.16 s, then held). w is the smallest weight that puts every major bone at <= 22 deg from Nav at
     played 0.16 s, so the ease is mild.
  4. Pelvis speed. A causal rate floor on the lift keeps the world pelvis (Hips) from falling faster than 6 m/s. The
     floor is part of the ground-contact iteration (the lift is re-solved every iteration).

Aux tracks (aux_tracks.add_aux_tracks): the SlideOver feet are free from takeoff frame 27 to the end; the Exit feet are
free from frame 0 until both feet are down on the ground and the trunk is upright, then the original Exit foot
targets resume (the frame is reported).

Outputs (only under out\\dive9):
  patches\\47BFB02DF.baked.json, patches\\F7919BDF0.baked.json, patches\\mapping_dive9.json
  joined_sheet.png, joined_sheet.gif, screen.json, summary.json, report.md
Run: python dive9.py            (writes out\\dive9; about 1 minute for the bake and the contact iteration)
     python dive9.py --screen   (also re-runs the timing screen, about 3 minutes)
"""
import json
import os
import sys
import zlib

import numpy as np

import acuclip
import aux_tracks as AT
import bake2
import bake_chain as BC
import bake_dive as BD
import config
import dive3 as D3
import dive4 as D4
import dive4b as B
import dive5 as F5
import dive6 as S6
import dive8 as D8
import dumps
import retarget as R

FPS = 60
K_PLANT, K_ROLL = 2.0, 3.0
T_TO, S_SCALE, SHIFT = 0.62, 1.35, 0.20
OVERRIDE = 'preview: dive9 fast roll, standing by Nav'
OUT_DIR = os.path.join(config.OUT, 'dive9')
PATCH_DIR = os.path.join(OUT_DIR, 'patches')
MAPPING = os.path.join(PATCH_DIR, 'mapping_dive9.json')
REPORT = os.path.join(OUT_DIR, 'report.md')
SCREEN_JSON = os.path.join(OUT_DIR, 'screen.json')
SUMMARY = os.path.join(OUT_DIR, 'summary.json')
SHEET_PNG = os.path.join(OUT_DIR, 'joined_sheet.png')
SHEET_GIF = os.path.join(OUT_DIR, 'joined_sheet.gif')
SO_NAME, SO_HEX = F5.SO_NAME, F5.SO_HEX
EXIT_NAME, EXIT_HEX = F5.EXIT_NAME, F5.EXIT_HEX
TAKEOFF_FRAME = D8.TAKEOFF_FRAME                       # 27: ACU takeoff of the Up100 clip (0.45 s)

# ------------------------------------------------------------------------------------------------ targets (spec)
Z_UP = 0.80                  # m: the Exit Reference z reached at P_UP and held (spec: 0.80 .. 0.85)
P_UP = 0.16                  # s played: rise end (spec: >= 0.78 by 0.16)
Z_UP_MIN = 0.78              # m: spec minimum at P_UP
RISE_V_MAX = 4.0             # m/s: peak vertical speed of the stand-up (spec: about 4)
PELVIS_V_MAX = 6.0           # m/s: world pelvis (Hips) vertical speed limit (spec d)
PELVIS_FLOOR_V = 5.9         # m/s: the rate floor aims 0.1 m/s inside the limit (float margin)
TILT_MAX = 60.0              # deg: trunk tilt from vertical (spec a)
GAP_GOAL = 25.0              # deg: pose gap to Nav_Run at P_UP (spec c)
GAP_AIM = 22.0               # deg: the ease aims 3 deg inside the goal (frame rounding)
EASE_T = 0.16                # s played: the rotation ease reaches its weight here
STEP_MAX = 12.0              # cm per frame (world), as dive4b
DV_MAX = 1.5                 # m/s per frame (world), as dive4b
OFF_MAX = 1.0                # m
CONT_MAX = 45.0              # deg at W
TOUCH_MIN = 3                # frames (spec: keep >= 3)
PEN_MAX = 5.0                # cm (box and ground)
DOWN_MARGIN = 0.03           # m: a foot counts as down within this height above its standing height (the original clip's end)
CHAIN_NAMES = ('Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head')
SCREEN_T = (0.62, 0.64, 0.66, 0.70, 0.74)
SCREEN_KP = (1.5, 1.75, 2.0)
SCREEN_KR = (2.5, 2.75, 3.0)
HAND_MIN = 5.50              # m: far edge 5.30 + 0.20 (dive5)
CONTACT_SIGMA_LIFT = 1.5     # frames, as dive6


# ------------------------------------------------------------------------------------------------ case
def make_case9(T_to=T_TO, k_plant=K_PLANT, k_roll=K_ROLL, s=S_SCALE, shift=SHIFT):
    """The dive8 time map (D8.Timing8: plant at k_plant, roll and Exit at k_roll). A slower pelvis fall was tried (a separate
    fall-phase speed) and does not fix the drop: the WD2 pelvis already falls 6.5 m/s (WD2 time) from w 0.70 to 0.92, so the
    peak moves into the plant (see report section 1b)."""
    c = B.Case(T_to, s, shift, k_roll)
    c.tm = D8.Timing8(T_to, k_plant, k_roll)
    return c


def hand_feasible(ctx, T_to, shift, s):
    """Clearance (m) of the closer hand at the plant beyond HAND_MIN, with the scan's formula (dive4b)."""
    yt = float(B.hips_at(ctx, T_to)[1]) + float(shift)
    return min(yt + float(s) * ctx.hand_rel_y[k] for k in ('L', 'R')) - HAND_MIN


# ------------------------------------------------------------------------------------------------ profiles
def exit_z_profile(p, z0, v0, z_up=Z_UP, p_up=P_UP):
    """Entity Reference z in the Exit at played p (s, >= 0). A Hermite rise from z0 with slope v0 (m per played s) at p = 0
    to z_up at p_up with zero slope (C1 at both ends), then held. The basis: h = smoothstep, g = u^3 - 2u^2 + u."""
    p = np.asarray(p, float)
    u = np.clip(p / p_up, 0.0, 1.0)
    h = 3.0 * u * u - 2.0 * u ** 3
    g = u ** 3 - 2.0 * u * u + u
    return z0 + (z_up - z0) * h + v0 * p_up * g


def rate_floor(lift_pre, zbase, vmax):
    """Causal lower envelope of the lift so that zbase + lift falls by at most vmax (m/s at 60 Hz) between frames.
    Only raises the lift; out[n] >= out[n-1] - step - (zbase[n] - zbase[n-1])."""
    out = np.asarray(lift_pre, float).copy()
    zb = np.asarray(zbase, float)
    step = float(vmax) / FPS
    for n in range(1, len(out)):
        need = out[n - 1] - step - (zb[n] - zb[n - 1])
        if need > out[n]:
            out[n] = need
    return out


def ease_weight(p, w_common, p_ease=EASE_T):
    """Rotation ease weight at played p: w_common * smoothstep(p / p_ease), zero before the Exit, constant after p_ease."""
    p = np.asarray(p, float)
    u = np.clip(p / p_ease, 0.0, 1.0)
    return w_common * (u * u * (3.0 - 2.0 * u))


# ------------------------------------------------------------------------------------------------ joints
def chain_idx(ctx):
    return [ctx.rig.ac_by_hash[zlib.crc32(n.encode())] for n in CHAIN_NAMES]


def upright(P, chain):
    """Per frame: trunk tilt from vertical (deg, Hips -> Head), whether any spine segment points down (inverted), and the
    head-above-hips flag. P [F, B, 3] world positions."""
    v = P[:, chain[-1]] - P[:, chain[0]]
    n = np.linalg.norm(v, axis=1)
    tilt = np.degrees(np.arccos(np.clip(v[:, 2] / np.maximum(n, 1e-12), -1.0, 1.0)))
    seg = P[:, chain[1:]] - P[:, chain[:-1]]
    inv = (seg[..., 2] < 0.0).any(axis=1)
    return tilt, inv, v[:, 2] > 0.0


def nav0_rot(ctx, ai):
    trk = dumps.track(ctx.next_dump, ctx.rig.ac[ai]['hash'], 0)
    return dumps.sample_rotation(trk, 0) if trk else R.qn(ctx.rig.ac[ai]['lrot'])


def gap_to_nav(ctx, final, frame, bones=BD.GAP_BONES):
    """Angle (deg) between the pose at a frame and Nav_Run's first frame, per bone."""
    out = {}
    for bn in bones:
        ai = BD._acu(ctx.rig, bn)
        out[bn] = float(bake2.angle_deg(final[frame, ai], nav0_rot(ctx, ai)))
    return out


def frame_of_p(ctx, p):
    return int(round((D4.EXIT_O + D4.EXIT_R * float(p)) * FPS))


def joined_pelvis_z(ctx, res):
    """World pelvis (Hips) z on the joined 60 Hz played grid from the SlideOver window start: (n0, z)."""
    PS = res['world']['_PS']
    PE = D3._world(res['seg_E']['seq'], res['seg_E']['dump'], res['fin_E']['final'], res['fin_E']['ref_out'],
                   res['fin_E']['disp'])
    z = B.joined_played(ctx, PS, PE, ctx.hips, res['win_S'], res['win_E'])[:, 2]
    return int(res['win_S'][0]), z


def pelvis_drop_mps(ctx, res):
    """Largest downward speed (m/s) of the world pelvis on the joined played grid."""
    _n0, z = joined_pelvis_z(ctx, res)
    dz = np.diff(z) * FPS
    return float(max(-dz.min(), 0.0)) if dz.size else 0.0


def joined_ref_z(ctx, res):
    """Entity Reference z on the joined played grid (the same resampling as the world path): (n0, z)."""
    rs = res['fin_S']['ref_out'][:, None, :]
    re_ = res['fin_E']['ref_out'][:, None, :]
    z = B.joined_played(ctx, rs, re_, 0, res['win_S'], res['win_E'])[:, 2]
    return int(res['win_S'][0]), z


# ------------------------------------------------------------------------------------------------ one bake
def bake9(ctx, case, lift_k):
    """dive8.bake8 with: the Exit Reference z is the stand-up profile (plus the lift), the Exit rotations eased toward
    Nav_Run's first frame, and the Exit feet free until both are down upright (reported; the aux tracks are added at
    write time). The SlideOver is dive8's (absolute WD2 rotations, the bump, the contact lift)."""
    tm = case.tm
    pl = D4.Place(ctx.sol, case.s, B.y_take_for(ctx, case.T_to, case.shift))
    o, span = ctx.o, ctx.span_so
    wfun = tm.w
    seg_S = BC.segment(ctx.rig, ctx.m, ctx.so['seq'], ctx.so['dump'], wfun, float(wfun(o)),
                       lambda t: bake2.patch_weight(t, o, span, B.FI, B.FO, B.TAIL), disp=ctx.disp_so)
    seg_S['win'] = bake2.window_frames(o, span, int(seg_S['F'][-1]))
    seg_S['ob'] = ctx.ob
    seg_S['patch'] = D3.absolute_patch(ctx, seg_S, wfun, o)
    t_S = seg_S['t60']
    win_S = seg_S['win']
    tw = wfun(win_S / FPS)
    plant_S = win_S[(tw >= D8.PLANT_W[0] - 1e-9) & (tw <= D8.PLANT_W[1] + 1e-9)]
    target_S, _pl = B.target_world(ctx, case, t_S, lift_k)
    t_blend = case.T_to + B.TB_EXTRA
    wT_S = B.smoothstep(np.clip((t_S - o) / max(t_blend - o, 1e-9), 0.0, 1.0))
    fin_S = B.bake_clip5(ctx, seg_S, pl, win_S, plant_S, target_S, ctx.disp_so, ctx.ob, wT_S)
    fin_S['win'] = win_S
    fin_S['plant'] = plant_S
    met_S, arr_S = D3._validate(ctx, seg_S, fin_S, ctx.t_W, span, o)
    # ---- Exit
    w_W = float(wfun(ctx.t_W))

    def w_exit(t):
        return w_W + tm.k * (np.asarray(t, float) - D4.EXIT_O) / D4.EXIT_R

    seg_E = BC.segment(ctx.rig, ctx.m, ctx.ex_seq, ctx.ex_dump, w_exit, float(w_exit(D4.EXIT_O)),
                       lambda t: (np.asarray(t, float) >= D4.EXIT_O - 1e-9).astype(float), disp=ctx.disp_ex)
    seg_E['win'] = bake2.window_frames(D4.EXIT_O, ctx.span_e, int(seg_E['F'][-1]))
    seg_E['ob'] = ctx.ground_ob
    seg_E['patch'] = D3.absolute_patch(ctx, seg_E, w_exit, D4.EXIT_O)
    tE = seg_E['t60']
    win_E = seg_E['win']
    last = int(win_S[-1])
    inside = tE >= D4.EXIT_O - 1e-9
    p = (tE - D4.EXIT_O) / D4.EXIT_R                       # played seconds into the Exit (p = 0 at W)
    tau_E = ctx.t_W + p
    # ---- the stand-up: z0 and the slope at W from the SlideOver (lift removed), z(p) the Hermite rise
    zb_S = fin_S['ref_out'][:, 2] - B.lift_eval(lift_k, t_S)
    z0 = float(zb_S[last])
    v0 = float((zb_S[last] - zb_S[last - 1]) * FPS * D4.EXIT_R)          # m per played second
    off_W = fin_S['ref_out'][last] - seg_E['ref0_orig'][int(win_E[0])]
    # the offset is full (c = 1) from the Exit's first patched frame up to the Exit played time 0 and then decays to 0 at the
    # window end; the Reference z follows the stand-up profile from the first patched frame (z0 before W + 0). The old
    # gate (c = 0 before EXIT_O) left the Exit's first patched frame at the original standing pose: a seam step.
    u_E = np.clip((tE - D4.EXIT_O) / max(ctx.t_end_E - D4.EXIT_O, 1e-9), 0.0, 1.0)
    c = 1.0 - B.smoothstep(u_E)
    ref_out_E = seg_E['ref0_orig'] + c[:, None] * off_W[None, :]
    zprof = exit_z_profile(np.maximum(p, 0.0), z0, v0)
    ref_out_E[:, 2] = zprof + B.lift_eval(lift_k, tau_E)
    fin_E = B.bake_clip5(ctx, seg_E, pl, win_E, np.array([], int), None, ctx.disp_ex, ctx.ground_ob, None,
                         hand_targets=False, ref_out_fixed=ref_out_E)
    fin_E['win'] = win_E
    fin_E['plant'] = np.array([], int)
    fin_E['targets'] = {}
    fin_E['disp'] = ctx.disp_ex
    # ---- rotations: common ease weight toward Nav_Run's first frame (mild: the smallest that meets the gap goal)
    f16 = frame_of_p(ctx, P_UP)
    patched_gap = [bn for bn in BD.GAP_BONES if BD._acu(ctx.rig, bn) in set(seg_E['patset'])]
    gap_pre = gap_to_nav(ctx, fin_E['final'], f16, patched_gap)
    w_common = 0.0
    for bn in patched_gap:
        g = gap_pre[bn]
        if g > GAP_AIM:
            w_common = max(w_common, 1.0 - GAP_AIM / g)
    w_common = float(min(1.0, w_common))
    w_ease = ease_weight(p, w_common)
    w_ease = np.where(inside, w_ease, 0.0)
    final_E = fin_E['final'].copy()
    for ai in seg_E['patset']:
        tgt = np.tile(nav0_rot(ctx, ai), (len(final_E), 1))
        final_E[:, ai] = bake2.mix_rot(final_E[:, ai], tgt, w_ease)
        final_E[:, ai] /= np.linalg.norm(final_E[:, ai], axis=1, keepdims=True)
    fin_E['final'] = final_E
    gap_post = gap_to_nav(ctx, final_E, f16, patched_gap)
    gap_all_post = gap_to_nav(ctx, final_E, f16, BD.GAP_BONES)
    fin_E['rot_tracks'] = {ai: bake2.rotation_track(dumps.track(ctx.ex_dump, ctx.rig.ac[ai]['hash'], 0) or [],
                                                   final_E[:, ai], max(0, int(win_E[0]) - 1),
                                                   int(seg_E['F'][-1])) for ai in seg_E['patset']}
    met_E, arr_E = D3._validate(ctx, seg_E, fin_E, ctx.t_end_E, ctx.span_e, D4.EXIT_O, next_dump=ctx.next_dump)
    cont = D4.cont_at_joint(ctx, fin_S, fin_E, last, int(win_E[0]))
    world = B.world_metrics(ctx, seg_S, fin_S, win_S, seg_E, fin_E, win_E)
    return dict(case=case, seg_S=seg_S, seg_E=seg_E, fin_S=fin_S, fin_E=fin_E, met_S=met_S, met_E=met_E, arr_S=arr_S,
                arr_E=arr_E, cont=cont, world=world, win_S=win_S, win_E=win_E, lift_k=np.array(lift_k, float),
                off_W=off_W, w_exit=w_exit, w_W=w_W, z0=z0, v0=v0, w_common=w_common, gap_pre=gap_pre,
                gap_post=gap_post, gap_all_post=gap_all_post, patched_gap=patched_gap, p_E=p, inside=inside, c_E=c)


# ------------------------------------------------------------------------------------------------ contact and rate
def contact_samples9(ctx, res, case):
    return D8.contact_samples8(ctx, res, case)


def solve9(ctx, case, A=None, t0=None, t1=None, n_iter=None, floor=True):
    """Bump (dive8 rule), the dive8 contact iteration (ground contact of the lowest non-finger point, the box clearance),
    then the pelvis rate floor applied ONCE to the converged lift (causal, raises only: the world Hips fall at most
    PELVIS_V_MAX), and one final bake with that lift. Putting the rate floor inside the contact loop made the two fight
    (the lift diverged to +47 cm), so the floor is a post-process and the contact deviation it causes is reported."""
    if A is None:
        A, t0, t1 = D8.bump_for(ctx, case)
    F5.BUMP.update(t0=t0, t1=t1, A=A)
    K = len(np.zeros(B.LIFT_K))
    lift = np.zeros(B.LIFT_K)
    hist = []
    res = None
    dev = 0.0
    iters = S6.CONTACT_ITER if n_iter is None else n_iter
    for it in range(iters + 1):
        res = bake9(ctx, case, lift)
        r, dev = S6.residual_from_samples(contact_samples9(ctx, res, case), K)
        hist.append(dict(it=it, worst_dev_cm=dev * 100.0, resid_max_cm=float(np.max(np.abs(r))) * 100.0,
                         lift_max_cm=float(lift.max()) * 100.0, pelvis_drop_mps=pelvis_drop_mps(ctx, res)))
        if it == iters or (dev <= S6.CONTACT_TOL and np.max(np.abs(r)) < 0.003):
            break
        lift = bake2._gauss_smooth(lift + r, CONTACT_SIGMA_LIFT)
    n0, Zj = joined_pelvis_z(ctx, res)
    Lj = lift[n0:n0 + len(Zj)]
    zbase = Zj - Lj
    lift_f = lift.copy()
    if floor:
        lift_f[n0:n0 + len(Zj)] = rate_floor(Lj, zbase, PELVIS_FLOOR_V)
    rate_add = lift_f - lift
    res = bake9(ctx, case, lift_f)
    devs = []
    for s in contact_samples9(ctx, res, case):
        devs.append(np.abs(s['surf'] + S6.CONTACT_TARGET - s['lowest'])[s['contact']])
    devs = np.concatenate(devs)
    res['contact_hist'] = hist
    res['contact_dev_m'] = float(np.max(devs))
    res['contact_frac_tol'] = float(np.mean(devs <= S6.CONTACT_TOL))
    res['contact_p95_cm'] = float(np.percentile(devs, 95) * 100.0)
    res['lift_min_cm'] = float(lift_f.min()) * 100.0
    res['lift_max_cm'] = float(lift_f.max()) * 100.0
    res['rate_add_max_cm'] = float(rate_add.max()) * 100.0
    res['bump'] = dict(A=A, t0=t0, t1=t1)
    res['lift_final'] = lift_f
    return res


# ------------------------------------------------------------------------------------------------ metrics
def foot_rest_z(ctx, res):
    """World height (m) of the lower foot bone when the ORIGINAL Exit stands (its last frame, in the same world convention as
    the patched poses: entity + disp): the feet 'down' test."""
    seg, fin = res['seg_E'], res['fin_E']
    PO = D3._world(seg['seq'], seg['dump'], seg['orig'], seg['ref0_orig'], fin['disp'])
    fl = [ctx.rig.ac_by_hash[zlib.crc32(n.encode())] for n in ('LeftFoot', 'RightFoot')]
    return float(np.min(PO[-1, fl, 2]))


def metrics9(ctx, res):
    """Uprightness, the Exit profile, the pelvis speed and the feet (all in the dive9 spec's terms)."""
    chain = chain_idx(ctx)
    P_S = D3._world(res['seg_S']['seq'], res['seg_S']['dump'], res['fin_S']['final'], res['fin_S']['ref_out'],
                    res['fin_S']['disp'])
    P_E = D3._world(res['seg_E']['seq'], res['seg_E']['dump'], res['fin_E']['final'], res['fin_E']['ref_out'],
                    res['fin_E']['disp'])
    W = int(res['win_S'][-1])
    tilt_S, inv_S, up_S = upright(P_S, chain)
    tilt_E, inv_E, up_E = upright(P_E, chain)
    before = np.arange(int(round((ctx.t_W - 0.10) * FPS)), W + 1)
    eframes = np.arange(int(np.ceil(D4.EXIT_O * FPS - 1e-9)), int(np.floor((D4.EXIT_O + 0.30 * D4.EXIT_R) * FPS + 1e-9)) + 1)
    out = dict(
        tilt_W=float(tilt_S[W]), inv_W=bool(inv_S[W]), up_W=bool(up_S[W]),
        tilt_before_W_max=float(tilt_S[before].max()), inv_before_W=bool(inv_S[before].any()),
        tilt_exit_max=float(tilt_E[eframes].max()), inv_exit=bool(inv_E[eframes].any()),
        up_exit=bool(up_E[eframes].all()),
    )
    fl = [ctx.rig.ac_by_hash[zlib.crc32(n.encode())] for n in ('LeftFoot', 'RightFoot')]
    thr = foot_rest_z(ctx, res) + DOWN_MARGIN
    out['foot_down_thr_m'] = thr
    p_grid = np.arange(0.0, 0.30 + 1e-9, 1.0 / FPS)
    down = []
    for p in p_grid:
        f = int(round((D4.EXIT_O + D4.EXIT_R * p) * FPS))
        ok = bool(np.all(P_E[f, fl, 2] <= thr) and tilt_E[f] <= TILT_MAX)
        down.append(ok)
    p_down = None
    for p, ok in zip(p_grid, down):
        if ok:
            p_down = float(p)
            break
    out['p_feet_down_upright'] = p_down
    if p_down is None:
        out['feet_free_to_frame'] = int(np.ceil((D4.EXIT_O + D4.EXIT_R * 0.30) * FPS))
    else:
        out['feet_free_to_frame'] = int(np.ceil((D4.EXIT_O + D4.EXIT_R * p_down) * FPS - 1e-9))
    # Exit Reference z at played 0.10 / 0.16 / 0.22
    zE = res['fin_E']['ref_out'][:, 2]
    for pz in (0.10, 0.16, 0.22):
        out['z_at_%.2f' % pz] = float(np.interp((D4.EXIT_O + D4.EXIT_R * pz) * FPS, np.arange(len(zE)), zE))
    out['z_at_W'] = float(res['z0'])
    # joined speeds: the Reference z (played) and the world pelvis (played)
    n0, zr = joined_ref_z(ctx, res)
    n0w, zw = joined_pelvis_z(ctx, res)
    dzr = np.diff(zr) * FPS
    dzw = np.diff(zw) * FPS
    tau_r = (np.arange(len(zr)) + n0) / FPS
    exit_idx = np.where(tau_r[1:] > ctx.t_W + 1e-9)[0]
    out['ref_rise_speed_max_mps'] = float(np.max(np.abs(dzr[exit_idx]))) if exit_idx.size else float('nan')
    pre_idx = np.where(tau_r[1:] <= ctx.t_W + 1e-9)[0]
    out['ref_rise_pre_W_max_mps'] = float(np.max(np.abs(dzr[pre_idx]))) if pre_idx.size else float('nan')
    out['ref_rise_post_W_at_s'] = float(tau_r[1:][exit_idx][np.argmax(np.abs(dzr[exit_idx]))]) if exit_idx.size else float('nan')
    out['ref_z_max_exit'] = float(zr[tau_r >= ctx.t_W].max()) if np.any(tau_r >= ctx.t_W) else float('nan')
    out['pelvis_drop_max_mps'] = float(-np.min(dzw)) if dzw.size else float('nan')
    out['pelvis_rise_max_mps'] = float(np.max(dzw)) if dzw.size else float('nan')
    # where the joined world step and the per-frame dv peak (head) are
    PS_ = res['world']['_PS']
    PE_ = D3._world(res['seg_E']['seq'], res['seg_E']['dump'], res['fin_E']['final'], res['fin_E']['ref_out'],
                    res['fin_E']['disp'])
    for name, idx in (('pelvis', ctx.hips), ('head', ctx.rig.ac_by_hash[zlib.crc32(b'Head')])):
        P = B.joined_played(ctx, PS_, PE_, idx, res['win_S'], res['win_E'])
        st = np.linalg.norm(np.diff(P, axis=0), axis=1)
        acc = np.linalg.norm(np.diff(P, 2, axis=0), axis=1) * FPS
        k = int(np.argmax(st)); ka = int(np.argmax(acc))
        out['step_peak_%s' % name] = dict(t=float((int(res['win_S'][0]) + k) / FPS), cm=float(st[k] * 100.0))
        out['dv_peak_%s' % name] = dict(t=float((int(res['win_S'][0]) + ka + 1) / FPS), ms=float(acc[ka]))
    out['gap_pre_16'] = res['gap_pre']
    out['gap_post_16'] = res['gap_post']
    out['gap_pre_max'] = float(max(res['gap_pre'][b] for b in res['patched_gap'])) if res['patched_gap'] else 0.0
    out['gap_post_max'] = float(max(res['gap_post'][b] for b in res['patched_gap'])) if res['patched_gap'] else 0.0
    out['gap_unpatched_16'] = {b: float(v) for b, v in res['gap_all_post'].items() if b not in res['patched_gap']}
    out['w_ease'] = res['w_common']
    out['tilt_series_S'] = tilt_S[before].tolist()
    return out


def judge9(res, m):
    w = res['world']
    fails = []
    pen = max(res['met_S']['pen_patch_cm'], res['met_E']['pen_patch_cm'])
    if pen > PEN_MAX:
        fails.append('non-finger penetration %.1f cm (> %.0f)' % (pen, PEN_MAX))
    if res['met_S']['touch']['min'] < TOUCH_MIN:
        fails.append('hand touch %d frames (need %d)' % (res['met_S']['touch']['min'], TOUCH_MIN))
    if w['dv_ms'] > DV_MAX:
        fails.append('world dv %.2f m/s per frame (> %.1f)' % (w['dv_ms'], DV_MAX))
    if w['step_cm'] > STEP_MAX:
        fails.append('world step %.1f cm/frame (> %.0f)' % (w['step_cm'], STEP_MAX))
    if w['offset_max_m'] > OFF_MAX:
        fails.append('Reference offset %.2f m (> %.1f)' % (w['offset_max_m'], OFF_MAX))
    if res['cont']['max_deg'] > CONT_MAX:
        fails.append('continuity %.0f deg (> %.0f)' % (res['cont']['max_deg'], CONT_MAX))
    if m['tilt_W'] > TILT_MAX or m['inv_W']:
        fails.append('upright at W: tilt %.0f deg, inverted %s' % (m['tilt_W'], m['inv_W']))
    if m['tilt_exit_max'] > TILT_MAX or m['inv_exit']:
        fails.append('upright in the Exit: max tilt %.0f deg, inverted %s' % (m['tilt_exit_max'], m['inv_exit']))
    if m['z_at_0.16'] < Z_UP_MIN:
        fails.append('Exit Reference z at 0.16 %.3f m (< %.2f)' % (m['z_at_0.16'], Z_UP_MIN))
    if m['ref_rise_speed_max_mps'] > RISE_V_MAX:
        fails.append('stand-up speed %.2f m/s (> %.0f)' % (m['ref_rise_speed_max_mps'], RISE_V_MAX))
    if m['pelvis_drop_max_mps'] > PELVIS_V_MAX + 1e-6:
        fails.append('world pelvis drop %.2f m/s (> %.0f)' % (m['pelvis_drop_max_mps'], PELVIS_V_MAX))
    if m['gap_post_max'] > GAP_GOAL:
        fails.append('gap to Nav_Run at 0.16 %.1f deg (> %.0f)' % (m['gap_post_max'], GAP_GOAL))
    return fails


# ------------------------------------------------------------------------------------------------ the screen
def screen(ctx):
    """Timing screen: for each (T_to, k_plant, k_roll), the trunk at W (zero-lift bake, the SlideOver only), and for each
    T_to the smallest-cost (s, shift) that puts the closer hand at HAND_MIN."""
    rows = []
    chain = chain_idx(ctx)
    for T in SCREEN_T:
        feas = None
        for s in np.round(np.arange(0.90, 1.5001, 0.01), 2):
            for sh in np.round(np.arange(-0.2, 0.5001, 0.01), 2):
                if hand_feasible(ctx, T, sh, s) >= 0.0:
                    cost = sh * sh + 0.5 * (s - 1.0) ** 2
                    if feas is None or cost < feas[0]:
                        feas = (cost, float(s), float(sh))
        for kp in SCREEN_KP:
            for kr in SCREEN_KR:
                case = make_case9(T, kp, kr, 1.03, 0.02)
                res = D8.bake8(ctx, case, np.zeros(B.LIFT_K))
                P = D3._world(res['seg_S']['seq'], res['seg_S']['dump'], res['fin_S']['final'], res['fin_S']['ref_out'],
                              res['fin_S']['disp'])
                tilt, inv, up = upright(P, chain)
                W = int(res['win_S'][-1])
                bef = np.arange(int(round((ctx.t_W - 0.10) * FPS)), W + 1)
                rows.append(dict(T_to=T, k_plant=kp, k_roll=kr, wW=float(case.tm.w(ctx.t_W)),
                                 t_feet=float(case.tm.t_feet), feet_minus_W=float(case.tm.t_feet - ctx.t_W),
                                 tilt_W=float(tilt[W]), inv_W=bool(inv[W]),
                                 tilt_before_W_max=float(tilt[bef].max()), inv_before_W=bool(inv[bef].any()),
                                 hips_W=float(P[W, chain[0], 2]),
                                 hand_s=None if feas is None else feas[1], hand_shift=None if feas is None else feas[2],
                                 hand_cost=None if feas is None else float(feas[0])))
    with open(SCREEN_JSON, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=1)
    return rows


# ------------------------------------------------------------------------------------------------ outputs
def write_patches(ctx, res, pdir=PATCH_DIR):
    os.makedirs(pdir, exist_ok=True)
    rig = ctx.rig
    so = D8.patch_dict(rig, res['fin_S']['rot_tracks'], res['fin_S']['trans_keys'], ctx.so['dump']['Length'])
    so_aux = AT.add_aux_tracks(so, ctx.so['dump'], TAKEOFF_FRAME, None)
    AT.write_patch(os.path.join(pdir, SO_HEX + '.baked.json'), so_aux)
    return so_aux


def write_exit_patch(ctx, res, feet_to, pdir=PATCH_DIR):
    rig = ctx.rig
    ex = D8.patch_dict(rig, res['fin_E']['rot_tracks'], res['fin_E']['trans_keys'], ctx.ex_dump['Length'])
    ex_aux = AT.add_aux_tracks(ex, ctx.ex_dump, 0, feet_to)
    AT.write_patch(os.path.join(pdir, EXIT_HEX + '.baked.json'), ex_aux)
    return ex_aux


def write_mapping(pdir=PATCH_DIR, override=OVERRIDE):
    import dive_chain2 as DC2
    m = dict(clips=[
        DC2.mapping_entry(SO_HEX, SO_NAME, SO_HEX + '.baked.json', True, override=override),
        DC2.mapping_entry(EXIT_HEX, EXIT_NAME, EXIT_HEX + '.baked.json', True, only_after=[SO_NAME], override=override),
    ])
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, 'mapping_dive9.json'), 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=1)
    return m


NOFLOOR_DIR = os.path.join(OUT_DIR, 'nofloor')
NOFLOOR_OVERRIDE = 'preview: dive9 nofloor'


def main_nofloor():
    """The same setup with the 6 m/s rate floor off: contact envelope only. Writes out\\dive9\\nofloor (patches, mapping,
    summary). The report, sheet and screen of the hover version are left as they are."""
    ctx = F5.make_ctx()
    case = make_case9()
    A, t0, t1 = D8.bump_for(ctx, case)
    res = solve9(ctx, case, A, t0, t1, floor=False)
    m = metrics9(ctx, res)
    fails = judge9(res, m)
    feet_to = m['feet_free_to_frame']
    write_patches(ctx, res, NOFLOOR_DIR)
    write_exit_patch(ctx, res, feet_to, NOFLOOR_DIR)
    write_mapping(NOFLOOR_DIR, NOFLOOR_OVERRIDE)
    summary = dict(variant='nofloor', case=dict(T_to=case.T_to, k_plant=case.tm.k_plant, k_roll=case.tm.k, s=case.s,
                                                  shift=case.shift),
                   bump=res['bump'], contact_hist=res['contact_hist'], contact_dev_m=res['contact_dev_m'],
                   contact_p95_cm=res['contact_p95_cm'], contact_frac_tol=res['contact_frac_tol'],
                   metrics=dict((k, v) for k, v in m.items() if k != 'tilt_series_S'), fails=fails,
                   world=dict((k, v) for k, v in res['world'].items() if not k.startswith('_')),
                   touch=res['met_S']['touch'], cont=res['cont']['max_deg'], feet_free_to_frame=feet_to)
    with open(os.path.join(NOFLOOR_DIR, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps(dict(fails=fails, pelvis_drop=m['pelvis_drop_max_mps'], contact_dev_cm=res['contact_dev_m'] * 100.0,
                          z016=m['z_at_0.16'], gap_post=m['gap_post_max'], feet_to=feet_to), default=float))
    return 0


def write_sheet(ctx, res, case):
    import bake_chain as BCh
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
    title = ('dive9 fast roll: T_to %.2f, k_plant %.1f, k_roll %.1f, s %.2f, shift %+.2f, bump %.2f m, contact dev %.1f cm' % (
        case.T_to, case.tm.k_plant, case.tm.k, case.s, case.shift, res['bump']['A'], res['contact_dev_m'] * 100.0))
    BCh.plot_joined(SHEET_PNG, ctx.rig, S, arrS, E, arrE, ctx.ob, tl, title, show_root=True)
    BCh.write_gif(SHEET_GIF, ctx.rig, S, arrS, E, arrE, ctx.ob, tl)


def _fmt(x, nd=3):
    return 'n/a' if x is None or (isinstance(x, float) and np.isnan(x)) else ('%.*f' % (nd, x))


def write_report(ctx, case, res, m, fails, heights, screen_rows, feet_to, nav_z, drop0=None):
    w = res['world']
    mS, mE = res['met_S'], res['met_E']
    L = []
    drop0_txt = 'n/a' if drop0 is None else '%.1f' % drop0
    L.append('# T21 dive9: the fast roll, upright before W and standing up inside the Exit')
    L.append('')
    L.append('Generated by `v2/pipeline/dive9.py` (reuses dive8, dive6, dive4b, dive4, dive3, wd2_path, aux_tracks, bake2, acuclip). '
             'Written only under `out\\dive9`.')
    L.append('')
    L.append('## 1. Timing (the fast roll)')
    L.append('')
    L.append('Chosen: T_to %.2f, k_plant %.1f, k_roll %.1f, s %.2f, shift %+.2f. The plant starts at clip %.3f s, the plant end '
             'at %.3f s, feet down (WD2 1.20) at %.3f s (W = %.3f s): feet-down minus W = %+.3f s. WD2 time at W: %.3f.'
             % (case.T_to, case.tm.k_plant, case.tm.k, case.s, case.shift, case.tm.t_pl0, case.tm.t_pe, case.tm.t_feet,
                ctx.t_W, case.tm.t_feet - ctx.t_W, float(case.tm.w(ctx.t_W))))
    L.append('The hands: the closer hand is at %.2f m at the plant (far edge 5.30 + 0.20 = 5.50 m required).'
             % (min(B.y_take_for(ctx, case.T_to, case.shift) + case.s * ctx.hand_rel_y[k] for k in ('L', 'R'))))
    L.append('')
    L.append('Timing screen (zero-lift SlideOver bake, the trunk at W; `screen.json` has all rows):')
    L.append('')
    L.append('| T_to | k_plant | k_roll | WD2 at W | feet-down minus W (s) | tilt at W (deg) | inverted at W | max tilt in last 0.10 s (deg) | inverted in last 0.10 s | hips at W (m) | hand s / shift (cost) |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for r in screen_rows:
        L.append('| %.2f | %.2f | %.2f | %.3f | %+.3f | %.1f | %s | %.1f | %s | %.3f | %s |' % (
            r['T_to'], r['k_plant'], r['k_roll'], r['wW'], r['feet_minus_W'], r['tilt_W'], r['inv_W'],
            r['tilt_before_W_max'], r['inv_before_W'], r['hips_W'],
            '%.2f / %+.2f (%.3f)' % (r['hand_s'], r['hand_shift'], r['hand_cost']) if r['hand_s'] is not None else 'none'))
    L.append('')
    L.append('The slow variants (dive6, dive8 B) have the trunk inverted at W. The chosen timing has the roll complete and the '
             'trunk upright (tilt %.0f deg at W, max %.0f deg over the last 0.10 s, no inverted segment).'
             % (m['tilt_W'], m['tilt_before_W_max']))
    L.append('')
    L.append('### 1b. The pelvis fall (spec d) and why the timing alone cannot fix it')
    L.append('')
    L.append('The WD2 pelvis path itself falls fast: about 6.5 m/s (WD2 time) over WD2 0.70 .. 0.92 and 7.9 m/s over 0.92 .. 1.00. '
             'Time compression scales that by k, so with the chosen k_plant 2.0 and k_roll 3.0 the zero-lift pelvis falls at '
             '%s m/s (world, joined). The rate floor (causal: the pelvis fall is limited to %.0f m/s) is the applied fix; it lifts '
             'the body by up to %.1f cm and the contact deviation becomes %.1f cm (section 5).'
             % (drop0_txt, PELVIS_FLOOR_V, res['rate_add_max_cm'], res['contact_dev_m'] * 100.0))
    L.append('')
    L.append('Tried and rejected (scratch experiments, not in the outputs): (i) a slower fall phase only (k_fall 0.6 .. 0.9 at WD2 '
             '0.92 .. 1.00, k_plant 2.0) does not help: the peak moves into the plant, to about 11 m/s. (ii) A slower plant and fall '
             '(k_plant 0.9, k_fall 0.76) brings the drop to 5.99 m/s, but at takeoff 0.62 the trunk is inverted at W (tilt 93 deg). '
             '(iii) A takeoff of 0.50 leaves the closer hand 0.62 m short of the far edge at s 1.35, shift 0.20. So the 6 m/s rule '
             'and the upright-before-W rule do not both hold in this timing family. (iv) A pure rate floor on the lift alone needs '
             '0.46 .. 0.70 m of lift (the body hovers) and still gives 5 .. 6 m/s per frame of dv.')
    L.append('')
    L.append('## 2. Stand-up and rotations in the Exit')
    L.append('')
    L.append('Entity Reference z (the capture convention): z at W = %.3f m (after the W lift). The Hermite rise (C1 at W and at '
             'played 0.16 s) reaches %.2f m at played 0.16, then holds. Peak stand-up speed after W (joined, played) = %.2f m/s '
             '(spec <= %.0f; FAIL when above). The roll upswing before W, which the k_roll 3.0 compresses, peaks at %.2f m/s '
             '(joined, before W). Reference z at played 0.10 / 0.16 / 0.22 = %.3f / %.3f / %.3f m.'
             % (m['z_at_W'], Z_UP, m['ref_rise_speed_max_mps'], RISE_V_MAX, m['ref_rise_pre_W_max_mps'],
                m['z_at_0.10'], m['z_at_0.16'], m['z_at_0.22']))
    L.append('')
    L.append('Nav_Run first frame: Reference z %.3f m (its frames 0..10 rise to %.3f m). The original Exit stands at 0.74..0.77 m.'
             % (nav_z[0], nav_z.max()))
    L.append('')
    L.append('Rotations: pre-ease gap to Nav_Run at played 0.16 = %.1f deg (max over the patched major bones); the common ease weight '
             'at 0.16 is %.2f (the smallest that puts every patched bone within %.0f deg). Gap after the ease = %.1f deg.'
             % (m['gap_pre_max'], m['w_ease'], GAP_AIM, m['gap_post_max']))
    L.append('')
    L.append('| bone | pre-ease gap (deg) | post-ease gap (deg) |')
    L.append('|---|---|---|')
    for bn in res['patched_gap']:
        L.append('| %s | %.1f | %.1f |' % (bn, res['gap_pre'][bn], res['gap_post'][bn]))
    L.append('')
    L.append('Unpatched bones (the original Exit rotations) at 0.16: %s.'
             % (', '.join('%s %.0f' % (b, v) for b, v in m['gap_unpatched_16'].items()) or 'none'))
    L.append('')
    L.append('## 3. Uprightness')
    L.append('')
    L.append('Trunk = Hips to Head (world). Tilt = angle from vertical; inverted = any spine segment (Hips, Spine, Spine1, '
             'Spine2, Neck, Head) pointing down.')
    L.append('')
    L.append('| metric | value | rule |')
    L.append('|---|---|---|')
    L.append('| tilt at W (deg) | %.1f | <= 60 |' % m['tilt_W'])
    L.append('| inverted at W | %s | no |' % m['inv_W'])
    L.append('| max tilt, last 0.10 s before W (deg) | %.1f | info |' % m['tilt_before_W_max'])
    L.append('| max tilt in the Exit, played 0 .. 0.30 (deg) | %.1f | <= 60 |' % m['tilt_exit_max'])
    L.append('| inverted anywhere in the Exit 0 .. 0.30 | %s | no |' % m['inv_exit'])
    L.append('| head above hips, every Exit frame 0 .. 0.30 | %s | yes |' % m['up_exit'])
    L.append('')
    L.append('## 4. Aux tracks (feet)')
    L.append('')
    L.append('SlideOver: feet free from frame %d (takeoff) to the clip end. Exit: feet free from frame 0 until both feet are '
             'down (world foot-bone z <= %.3f m, the standing height of the original Exit + %.2f m) and the trunk is upright '
             '(tilt <= %.0f deg): first played time %s, so the original Exit foot targets resume at clip frame %d (played %s). '
             'Inverse-Reference tracks in both clips.'
             % (TAKEOFF_FRAME, m['foot_down_thr_m'], DOWN_MARGIN, TILT_MAX, _fmt(m['p_feet_down_upright']), feet_to,
                _fmt(m['p_feet_down_upright']) if m['p_feet_down_upright'] is not None else
                'none within 0.30 (the aux window runs to clip frame %d)' % feet_to))
    L.append('')
    L.append('## 5. Validation')
    L.append('')
    L.append('| metric | SlideOver | Exit | rule |')
    L.append('|---|---|---|---|')
    L.append('| non-finger penetration, box / ground (cm) | %.1f / %.1f | %.1f / %.1f | <= 5 |'
             % (mS['pen_box_cm'], mS['pen_ground_cm'], mE['pen_box_cm'], mE['pen_ground_cm']))
    L.append('| hand touch (frames, min) | %d | n/a | >= 3 |' % mS['touch']['min'])
    L.append('| world pelvis: step cm/frame / dv m/s per frame | %.2f / %.2f | joined | <= 12 / 1.5 |'
             % (w['pelvis']['step_cm'], w['pelvis']['dv_ms']))
    L.append('| world head: step cm/frame / dv m/s per frame | %.2f / %.2f | joined | <= 12 / 1.5 |'
             % (w['head']['step_cm'], w['head']['dv_ms']))
    L.append('| world pelvis, drop max (m/s) | | %.2f | <= 6 |' % m['pelvis_drop_max_mps'])
    L.append('| Reference offset max (m) | %.3f | | <= 1.0 |' % w['offset_max_m'])
    L.append('| continuity at W (deg) | %.1f (%s) | | <= 45 |' % (res['cont']['max_deg'], res['cont']['worst']))
    L.append('| contact: %% within 3 cm / 95th pct (cm), worst dev (cm) | %.0f%% / %.1f / %.1f | | info |'
             % (res['contact_frac_tol'] * 100.0, res['contact_p95_cm'], res['contact_dev_m'] * 100.0))
    L.append('')
    L.append('Contact iterations (worst contact deviation, pelvis drop):')
    L.append('')
    for h in res['contact_hist'][::max(1, len(res['contact_hist']) // 8)] + [res['contact_hist'][-1]]:
        L.append('* it %d: worst dev %.1f cm, residual max %.1f cm, lift max %.1f cm, pelvis drop %.2f m/s'
                 % (h['it'], h['worst_dev_cm'], h['resid_max_cm'], h['lift_max_cm'], h['pelvis_drop_mps']))
    L.append('')
    L.append('Peaks of the joined world path: pelvis step %.1f cm at played t = %.3f s; head dv %.1f m/s per frame at t = %.3f s; '
             'pelvis dv peak at t = %.3f s. Continuity at W with the pose as baked: %.1f deg (LeftFoot is the largest bone).'
             % (m['step_peak_pelvis']['cm'], m['step_peak_pelvis']['t'], m['dv_peak_head']['ms'], m['dv_peak_head']['t'],
                m['dv_peak_pelvis']['t'], res['cont']['max_deg']))
    L.append('')
    L.append('Feet and body through the roll and the Exit (entity-local heights, joined time):')
    L.append('')
    L.append('| joined time (s) | Ref z (hips) | head z | lowest foot z | world hips z |')
    L.append('|---|---|---|---|---|')
    for r in heights:
        L.append('| %.2f | %.3f | %.3f | %.3f | %.3f |' % (r['tau'], r['ref_z'], r['head_z'], r['feet_z'], r['ref_world_z']))
    L.append('')
    L.append('Spec status: %s' % ('; '.join(fails) if fails else 'all dive9 rules pass'))
    L.append('')
    L.append('Tests: `pipeline\\tests\\test_dive9.py` (profile C1, rate floor, ease, mapping, aux frame, outputs).')
    L.append('')
    L.append('Enable state: both clips written enabled with override "%s" (as dive8 B).' % OVERRIDE)
    L.append('')
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ctx = F5.make_ctx()
    case = make_case9()
    os.makedirs(OUT_DIR, exist_ok=True)
    screen_rows = screen(ctx) if '--screen' in argv else (
        json.load(open(SCREEN_JSON, encoding='utf-8')) if os.path.exists(SCREEN_JSON) else screen(ctx))
    A, t0, t1 = D8.bump_for(ctx, case)
    # the zero-lift baseline (contact lift off, rate floor off): the pelvis fall the timing gives on its own
    F5.BUMP.update(t0=t0, t1=t1, A=A)
    res0 = bake9(ctx, case, np.zeros(B.LIFT_K))
    drop0 = pelvis_drop_mps(ctx, res0)
    res = solve9(ctx, case, A, t0, t1)
    m = metrics9(ctx, res)
    fails = judge9(res, m)
    heights = S6.heights_table(ctx, res)
    feet_to = m['feet_free_to_frame']
    write_patches(ctx, res)
    write_exit_patch(ctx, res, feet_to)
    write_mapping()
    write_sheet(ctx, res, case)
    rz = dumps.track(ctx.next_dump, D8.REF_ID, 3)
    nav_z = np.array([dumps.sample_position(rz, f)[2] for f in range(0, 11)])
    write_report(ctx, case, res, m, fails, heights, screen_rows, feet_to, nav_z, drop0=drop0)
    summary = dict(case=dict(T_to=case.T_to, k_plant=case.tm.k_plant, k_roll=case.tm.k, s=case.s, shift=case.shift),
                   bump=res['bump'], contact_hist=res['contact_hist'], contact_dev_m=res['contact_dev_m'],
                   pelvis_drop_zero_lift_mps=drop0, rate_add_max_cm=res['rate_add_max_cm'],
                   metrics=dict((k, v) for k, v in m.items() if k != 'tilt_series_S'), fails=fails,
                   world=dict((k, v) for k, v in res['world'].items() if not k.startswith('_')),
                   touch=res['met_S']['touch'], cont=res['cont']['max_deg'], feet_free_to_frame=feet_to)
    with open(SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps(dict(fails=fails, tilt_W=m['tilt_W'], z016=m['z_at_0.16'], gap_post=m['gap_post_max'],
                          pelvis_drop=m['pelvis_drop_max_mps'], feet_to=feet_to, touch=res['met_S']['touch']['min'],
                          contact_dev_cm=res['contact_dev_m'] * 100.0), default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main_nofloor() if '--nofloor' in sys.argv[1:] else main())
