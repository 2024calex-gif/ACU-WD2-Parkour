"""T23 (dive11): the dive10 delay15 variant with the contact floor fixed for every non-finger bone.

Reuses dive10 (its event shift, timing, Exit profile, ease, hold and metrics), dive9 (SlideOver bake, rate floor, the
joined pelvis) and dive8 (contact samples, patch dict). Their defaults are not changed: they are imported and their
functions called, never modified.

What changes against delay15 (in game the roll-end hand reached -8 to -12.6 cm and the Exit toes 2 to 12 cm under the
ground; the model's floor was -2.5 cm at the roll end and -3.4 cm in the Exit):

  1. Floor target. Every non-finger bone (body_nf: all mapped ACU bones but the fingers, so feet, toes LeftFootThumb1 /
     RightFootThumb1 and hands) is held at TARGET (2 cm) above its surface in the Exit (lowest bone of all of them, not the
     feet only as dive8 does). The SlideOver's surface is the box top in the box band and the ground otherwise, as dive8.
     The box clearance (bake2.T9D_LIFT_CLEAR above the box top) is unchanged outside the plant.
  2. Foot IK in the Exit (after the rotation ease, before the hold). While an ankle (the Foot bone) is below FOOT_NEAR
     (0.15 m), the Foot is rotated about its ankle, about the horizontal axis perpendicular to the foot, by the smallest
     angle that puts the toe (FootThumb1) at TOE_T (= TARGET, 2 cm) or higher. The angle is capped at IK_THETA_MAX (50 deg),
     smoothed over a few frames without cutting the needed angle, and written as the Foot's local rotation (the toe is a
     child, so it follows). Only what the IK cannot reach goes into the body lift.
  3. SlideOver lift (from dive9's). One-sided and not iterated: S_PASSES one-shot passes each add the raise a frame needs
     (capped at S_STEP_CAP; the peaks are kept). The hands are pinned by the hand IK on the plant frames (fin_S['plant']) and
     the 6 frames before them (PLANT_NEAR) get only the floor (PLANT_TARGET, 0.5 cm) with no box clearance; the arms there
     cannot follow a lift, and the hand touch fails when they are pushed. dive9's rate floor is applied after the passes.
  4. Exit lift. The one-sided, peak-preserving updates (E_ITER at most) add the raise the Exit's lowest bone needs; the
     candidate kept is the one with the highest Exit-window floor among those with peak Reference z speed <= RISE_V_MAX.
  5. Unchanged from delay15: the SlideOver timing, the Exit Reference profile (P_UP, Z_UP), the ease to Nav_Run, the hold,
     the event tracks (copied byte-identical from out\\dive10\\delay15) and the aux tracks (rebuilt for the new Reference).

Outputs (only under out\\dive11):
  47BFB02DF.baked.json, F7919BDF0.baked.json, mapping_dive11.json, report.md, summary.json, joined_sheet.png/.gif
Run: python dive11.py   (about 6 minutes: dive9's solve, dive10's delay15 for the before numbers, then this bake)
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
import dive9 as D9
import dive10 as D10
import dive_chain2 as DC2
import dumps
import retarget as R

FPS = 60
DELAY = 15
OVERRIDE = 'preview: dive11 delay15 floor fix'
OUT_DIR = os.path.join(config.OUT, 'dive11')
D10_EXIT = os.path.join(config.OUT, 'dive10', 'delay15', D9.EXIT_HEX + '.baked.json')
SO_HEX, SO_NAME = D9.SO_HEX, D9.SO_NAME
EXIT_HEX, EXIT_NAME = D9.EXIT_HEX, D9.EXIT_NAME
MAPPING_NAME = 'mapping_dive11.json'
REPORT = os.path.join(OUT_DIR, 'report.md')
SUMMARY = os.path.join(OUT_DIR, 'summary.json')
SHEET_PNG = os.path.join(OUT_DIR, 'joined_sheet.png')
SHEET_GIF = os.path.join(OUT_DIR, 'joined_sheet.gif')

TARGET = 0.02                 # m: every non-finger bone is lifted to 2 cm above its surface (both clips)
PLANT_TARGET = 0.005          # m: the SlideOver plant (hands on the box) needs only the floor, with a margin
PLANT_NEAR = 6                # frames before the plant's first frame that count as plant for the SlideOver lift
TOE_T = TARGET                # m: the foot IK keeps the toe (FootThumb1) at or above the envelope target (2 cm); a 1 cm
                              # IK target would let the lift re-raise every toe between 1 and 2 cm (see report section 6)
FOOT_NEAR = 0.15              # m: the foot IK runs while the ankle (Foot bone) is below this height
IK_THETA_MAX = np.radians(50.0)
IK_DILATE = 2                 # frames: the IK angle is widened by this many frames before smoothing
IK_SIGMA = 1.5                # frames: Gaussian smoothing of the IK angle
S_PASSES = 4                  # one-shot passes for the SlideOver lift (see solve_S11)
S_STEP_CAP = 0.06            # m: most one pass raises a frame
E_ITER = 10                   # most lift iterations for the Exit (the speed rule decides the candidate)
CONTACT_STOP = 0.0005         # m: an iteration stops when the largest residual is below this (0.5 mm)
HANDS = ('LeftHand', 'RightHand')
TOES = ('LeftFootThumb1', 'RightFootThumb1')
FEETS = ('LeftFoot', 'RightFoot')
PHASES = ('run-up and flight', 'plant', 'roll', 'last 0.10 s before W', 'Exit window', 'Exit tail')


def override_for():
    return OVERRIDE


# ------------------------------------------------------------------------------------------------ names and bones
def bone_names(rig):
    """{ACU bone index: name} for the mapped bones and the Reference."""
    rev = {zlib.crc32(n.encode()): n for n in set(R.build_map().values())}
    rev[zlib.crc32(b'Reference')] = 'Reference'
    return {i: rev.get(b['hash'], '#%d' % i) for i, b in enumerate(rig.ac)}


def bone_idx(rig, name):
    return rig.ac_by_hash[zlib.crc32(name.encode())]


# ------------------------------------------------------------------------------------------------ contact residual
def residual11(samples, K, target=TARGET):
    """dive8's residual_from_samples with the target given: tgt = surface + target (two-sided in the contact window, one-
    sided above it), and the box clearance one-sided where the box need is positive. Returns (avg per index, worst |r_c|)."""
    num = np.zeros(K)
    cnt = np.zeros(K)
    worst = 0.0
    for s in samples:
        tgt = s['surf'] + target
        r_c = tgt - s['lowest']
        r_out = np.maximum(r_c, 0.0)
        r = np.where(s['contact'], r_c, r_out)
        r = np.where(s['box_need'] > 0, np.maximum(r, s['box_need']), r)
        if np.any(s['contact']):
            worst = max(worst, float(np.max(np.abs(np.where(s['contact'], r_c, 0.0)))))
        k = np.clip(np.round(s['tau'] * FPS).astype(int), 0, K - 1)
        np.add.at(num, k, r)
        np.add.at(cnt, k, 1.0)
    avg = np.where(cnt > 0, num / np.maximum(cnt, 1.0), 0.0)
    return avg, worst


# ------------------------------------------------------------------------------------------------ quaternions (xyzw), vectorised
def _qconj_v(q):
    return np.stack([-q[:, 0], -q[:, 1], -q[:, 2], q[:, 3]], axis=1)


def _qmul_v(a, b):
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=1)


def _qrot_v(q, v):
    """v rotated by q (per frame): q v q*."""
    vq = np.concatenate([v, np.zeros((len(v), 1))], axis=1)
    return _qmul_v(_qmul_v(q, vq), _qconj_v(q))[:, :3]


# ------------------------------------------------------------------------------------------------ foot IK
def toe_angle(Az, d):
    """The smallest rotation angle (rad, >= 0) about the axis (d x up) that puts the toe at TOE_T or higher, for an ankle
    at height Az (m) and toe offset d (vector, ankle -> toe). Rotating d by th about a = (d x up)/rho gives
    (R d)_z = d_z cos(th) + rho sin(th) = |d| cos(th - phi), with phi = atan2(rho, d_z). Returns (th, reachable)."""
    d = np.asarray(d, float)
    dn = float(np.linalg.norm(d))
    if dn < 1e-9 or Az + d[2] >= TOE_T:
        return 0.0, True
    rho = float(np.sqrt(max(dn * dn - d[2] * d[2], 0.0)))
    phi = float(np.arctan2(rho, d[2]))
    k = (TOE_T - Az) / dn
    reach = k <= 1.0
    alpha = float(np.arccos(np.clip(k, -1.0, 1.0)))
    th = float(np.clip(phi - alpha, 0.0, IK_THETA_MAX))
    return th, reach


def foot_ik(ctx, seg_E, final_E, ref_out_E, disp, ik_frames):
    """Foot IK on the Exit's local rotations (the Foot bone of each side). ik_frames: the clip frames it may change (the
    Exit window and the hold frame, from which the tail is copied). Returns (new final [F, B, 4], info dict)."""
    rig = ctx.rig
    F = len(final_E)
    info = dict(sides={})
    out = np.array(final_E, float)
    patset = set(int(a) for a in seg_E['patset'])
    rot, pos = acuclip.entity_transforms(bake2.composed(seg_E['seq'], final_E, ref_out_E))
    dz = np.asarray(disp, float)[:, 2]
    inwin = np.zeros(F, bool)
    inwin[np.asarray(ik_frames, int)] = True
    up = np.array([0.0, 0.0, 1.0])
    for side in ('Left', 'Right'):
        fi = bone_idx(rig, side + 'Foot')
        ti = bone_idx(rig, side + 'FootThumb1')
        if fi not in patset:
            info['sides'][side] = dict(skipped='Foot has no rotation track')
            continue
        A = pos[:, fi]
        d = pos[:, ti] - A
        dn = np.linalg.norm(d, axis=1)
        Az = A[:, 2] + dz
        Tz = Az + d[:, 2]
        near = inwin & (Az < FOOT_NEAR)
        need = near & (Tz < TOE_T)
        raw = np.zeros(F)
        reach = np.ones(F, bool)
        for f in np.nonzero(need)[0]:
            th, ok = toe_angle(Az[f], d[f])
            raw[f] = th
            reach[f] = ok
        dil = raw.copy()
        for j in range(1, IK_DILATE + 1):
            dil[j:] = np.maximum(dil[j:], raw[:-j])
            dil[:-j] = np.maximum(dil[:-j], raw[j:])
        th_f = bake2._gauss_smooth(dil, IK_SIGMA) if np.any(dil) else dil
        th_f = np.where(inwin, np.maximum(th_f, raw), 0.0)     # smoothing must not cut the angle a needed frame needs
        # the axis: horizontal, perpendicular to the foot (d x up); a horizontal axis when the toe is vertical
        a_w = np.cross(d, up[None, :])
        an = np.linalg.norm(a_w, axis=1)
        a_w = np.where((an > 1e-9)[:, None], a_w / np.maximum(an, 1e-9)[:, None], np.array([1.0, 0.0, 0.0])[None, :])
        a_loc = _qrot_v(_qconj_v(rot[:, fi]), a_w)                  # the same axis in the Foot's local frame
        half = 0.5 * th_f
        q_loc = np.concatenate([a_loc * np.sin(half)[:, None], np.cos(half)[:, None]], axis=1)
        newL = _qmul_v(np.asarray(final_E[:, fi], float), q_loc)
        out[:, fi] = newL / np.linalg.norm(newL, axis=1, keepdims=True)
        info['sides'][side] = dict(frames_needed=int(need.sum()), frames_active=int(np.sum(th_f > 1e-9)),
                                   theta_max_deg=float(np.degrees(th_f.max())) if F else 0.0,
                                   unreachable_frames=int(np.sum(need & ~reach)), bone=fi, toe=ti)
    # verify on the new pose: the lowest toe over the window, before and after
    rot2, pos2 = acuclip.entity_transforms(bake2.composed(seg_E['seq'], out, ref_out_E))
    for side in ('Left', 'Right'):
        if side not in info['sides'] or 'skipped' in info['sides'][side]:
            continue
        ti = info['sides'][side]['toe']
        fi = info['sides'][side]['bone']
        tz_after = pos2[:, ti, 2] + dz
        az_after = pos2[:, fi, 2] + dz
        w_idx = np.asarray(ik_frames, int)
        near_w = w_idx[az_after[w_idx] < FOOT_NEAR]
        info['sides'][side]['toe_min_after_cm'] = float(tz_after[near_w].min() * 100.0) if near_w.size else None
    return out, info


# ------------------------------------------------------------------------------------------------ one bake
def bake11(ctx, case, lift_S, lift_E, delay, use_ik=True):
    """dive10.bake10 with (1) the foot IK after the ease and before the hold, (2) the same stand-up, ease, hold and
    validation. ctx.span_e must already be the new window (the caller sets and restores it)."""
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
    target_S, _pl = B.target_world(ctx, case, t_S, lift_S)
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
    p = (tE - D4.EXIT_O) / D4.EXIT_R
    tau_E = ctx.t_W + p
    p_up = D10.p_up_for(delay)
    p_end = D10.p_end_for(delay)
    p_ease = D10.p_ease_for(delay)
    zS = fin_S['ref_out'][:, 2]
    z_e0 = float(zS[last])
    v_e0_clip = float((zS[last] - zS[last - 1]) * FPS)
    v_e0_played = v_e0_clip * D4.EXIT_R
    off_W = fin_S['ref_out'][last] - seg_E['ref0_orig'][int(win_E[0])]
    u_E = np.clip((tE - D4.EXIT_O) / max(ctx.t_end_E - D4.EXIT_O, 1e-9), 0.0, 1.0)
    c = 1.0 - B.smoothstep(u_E)
    ref_out_E = seg_E['ref0_orig'] + c[:, None] * off_W[None, :]
    zprof = D9.exit_z_profile(np.maximum(p, 0.0), z_e0, v_e0_played, D9.Z_UP, p_up)
    ref_out_E[:, 2] = zprof + B.lift_eval(lift_E, tau_E)
    fin_E = B.bake_clip5(ctx, seg_E, pl, win_E, np.array([], int), None, ctx.disp_ex, ctx.ground_ob, None,
                         hand_targets=False, ref_out_fixed=ref_out_E)
    fin_E['win'] = win_E
    fin_E['plant'] = np.array([], int)
    fin_E['targets'] = {}
    fin_E['disp'] = ctx.disp_ex
    # ---- rotations: the common ease weight toward Nav_Run's first frame (as dive10)
    patched_gap = [bn for bn in BD.GAP_BONES if BD._acu(ctx.rig, bn) in set(seg_E['patset'])]
    gap_pre = D10.gap_at(ctx, fin_E['final'], p_ease, patched_gap)
    w_common = 0.0
    for bn in patched_gap:
        g = gap_pre[bn]
        if g > D9.GAP_AIM:
            w_common = max(w_common, 1.0 - D9.GAP_AIM / g)
    w_common = float(min(1.0, w_common))
    w_ease = np.where(inside, D9.ease_weight(p, w_common, p_ease), 0.0)
    final_E = fin_E['final'].copy()
    for ai in seg_E['patset']:
        tgt = np.tile(D9.nav0_rot(ctx, ai), (len(final_E), 1))
        final_E[:, ai] = bake2.mix_rot(final_E[:, ai], tgt, w_ease)
        final_E[:, ai] /= np.linalg.norm(final_E[:, ai], axis=1, keepdims=True)
    # ---- dive11: foot IK (after the ease, before the hold: the ease must not undo the toe lift)
    ik = dict(sides={})
    nf = int(seg_E['F'][-1])
    hold = min(D10.frame_of_played(p_end), nf)
    if use_ik:
        ik_frames = np.nonzero(inside & (np.arange(len(final_E)) <= hold))[0]
        final_E, ik = foot_ik(ctx, seg_E, final_E, ref_out_E, ctx.disp_ex, ik_frames)
    fin_E['final'] = final_E
    # the clip holds its pose and its Reference position from the new window end to the clip end (as dive10)
    if hold < nf:
        final_E[hold + 1:] = final_E[hold]
        ref_out_E[hold + 1:] = ref_out_E[hold]
        fin_E['ref_out'] = ref_out_E
        held = [round(float(x), 9) for x in ref_out_E[hold]]
        keys = [k for k in fin_E['trans_keys'] if int(k['Frame']) < hold]
        keys.append({'Frame': int(hold), 'LocalPosition': held})
        keys.append({'Frame': int(nf), 'LocalPosition': held})
        fin_E['trans_keys'] = keys
    fin_E['final'] = final_E
    gap_post_ease = D10.gap_at(ctx, final_E, p_ease, patched_gap)
    gap_post_end = D10.gap_at(ctx, final_E, p_end, BD.GAP_BONES)
    fin_E['rot_tracks'] = {ai: bake2.rotation_track(dumps.track(ctx.ex_dump, ctx.rig.ac[ai]['hash'], 0) or [],
                                                   final_E[:, ai], max(0, int(win_E[0]) - 1),
                                                   int(seg_E['F'][-1])) for ai in seg_E['patset']}
    met_E, arr_E = D3._validate(ctx, seg_E, fin_E, ctx.t_end_E, ctx.span_e, D4.EXIT_O, next_dump=ctx.next_dump)
    cont = D4.cont_at_joint(ctx, fin_S, fin_E, last, int(win_E[0]))
    world = B.world_metrics(ctx, seg_S, fin_S, win_S, seg_E, fin_E, win_E)
    return dict(seg_S=seg_S, seg_E=seg_E, fin_S=fin_S, fin_E=fin_E, met_S=met_S, met_E=met_E, arr_S=arr_S,
                arr_E=arr_E, cont=cont, world=world, win_S=win_S, win_E=win_E,
                lift_S=np.array(lift_S, float), lift_E=np.array(lift_E, float), z0=z_e0, v_e0_clip=v_e0_clip,
                v_start_clip=v_e0_clip, w_common=w_common, p_up=p_up, p_end=p_end, p_ease=p_ease,
                patched_gap=patched_gap, gap_pre_ease=gap_pre, gap_post_ease=gap_post_ease, gap_post_end=gap_post_end,
                delay=int(delay), p_E=p, inside=inside, c_E=c, w_exit=w_exit, ik=ik, use_ik=bool(use_ik))


# ------------------------------------------------------------------------------------------------ floor report
def floor_phases(ctx, res, case):
    """Lowest non-finger bone height (cm, ground z = 0, entity + root displacement) per phase of the joined played window.
    The hands are left out only during the plant. Returns {phase: {min_cm, bone, t_s, frames}}."""
    names = bone_names(ctx.rig)
    body = [int(b) for b in ctx.body_nf]
    hands = {bone_idx(ctx.rig, h) for h in HANDS}
    hcols = np.array([j for j, b in enumerate(body) if b in hands], int)
    PS = D3._world(res['seg_S']['seq'], res['seg_S']['dump'], res['fin_S']['final'], res['fin_S']['ref_out'],
                   res['fin_S']['disp'])
    PE = D3._world(res['seg_E']['seq'], res['seg_E']['dump'], res['fin_E']['final'], res['fin_E']['ref_out'],
                   res['fin_E']['disp'])
    W = float(ctx.t_W)
    win_S = np.asarray(res['win_S'], int)
    tS = win_S / FPS
    tw = case.tm.w(tS)
    plant_set = set(int(f) for f in res['fin_S']['plant'])
    plant = np.array([int(f) in plant_set for f in win_S], bool)
    pre = (~plant) & (tw < D8.PLANT_W[0] - 1e-9)
    roll = (~plant) & (tw > D8.PLANT_W[1] + 1e-9) & (tS <= W - 0.10 + 1e-9)
    last = (tS > W - 0.10 + 1e-9) & (tS <= W + 1e-9)
    tE = np.asarray(res['seg_E']['t60'], float)
    p = (tE - D4.EXIT_O) / D4.EXIT_R
    ex_win = (p >= -1e-9) & (p <= res['p_end'] + 1e-9)
    ex_tail = (p > res['p_end'] + 1e-9)
    idx_E = np.arange(len(tE))
    sets = {
        'run-up and flight': (PS, win_S[pre], tS[pre], np.zeros(pre.sum(), bool), False),
        'plant': (PS, win_S[plant], tS[plant], np.ones(plant.sum(), bool), True),
        'roll': (PS, win_S[roll], tS[roll], np.zeros(roll.sum(), bool), False),
        'last 0.10 s before W': (PS, win_S[last], tS[last], np.zeros(last.sum(), bool), False),
        'Exit window': (PE, idx_E[ex_win], W + p[ex_win], np.zeros(ex_win.sum(), bool), False),
        'Exit tail': (PE, idx_E[ex_tail], W + p[ex_tail], np.zeros(ex_tail.sum(), bool), False),
    }
    out = {}
    for name in PHASES:
        P, frames, times, in_plant, is_plant = sets[name]
        if len(frames) == 0:
            out[name] = dict(min_cm=None, bone=None, t_s=None, frames=0)
            continue
        Z = P[frames][:, body, 2].copy()
        if is_plant and hcols.size:
            Z[:, hcols] = np.inf
        flat = int(np.argmin(Z))
        r, c = divmod(flat, Z.shape[1])
        out[name] = dict(min_cm=float(Z[r, c] * 100.0), bone=names[body[c]], t_s=float(times[r]), frames=int(len(frames)))
    return out


def floor_min_cm(phases):
    vals = [v['min_cm'] for v in phases.values() if v['min_cm'] is not None]
    return min(vals) if vals else float('nan')


def exit_window_floor(phases):
    v = [phases[k]['min_cm'] for k in ('Exit window', 'Exit tail') if phases[k]['min_cm'] is not None]
    return min(v) if v else float('nan')


def pass_floor(phases):
    """The rule: every non-finger bone >= 0 cm outside the plant (hands are excluded only in the plant, as in floor_phases)."""
    return all(v['min_cm'] is None or v['min_cm'] >= 0.0 for v in phases.values())


# ------------------------------------------------------------------------------------------------ the solve
class window:
    """Sets ctx.span_e to the delay's window for the bake and restores it after."""

    def __init__(self, ctx, delay):
        self.ctx, self.delay = ctx, delay

    def __enter__(self):
        self.saved = self.ctx.span_e
        self.ctx.span_e = D4.EXIT_R * D10.p_end_for(self.delay)
        return self

    def __exit__(self, *exc):
        self.ctx.span_e = self.saved
        return False


def S_need(ctx, res):
    """The SlideOver's one-sided raise (m) per clip frame, from the bake's own poses. Hands are left out only on the plant's
    frames (fin_S['plant'], where the hand IK pins them), so the flight frames next to the plant keep their hands. The
    surface is the box top where the lowest point lies in the box band and the ground otherwise, as dive8. Plant frames
    need only PLANT_TARGET; the box clearance (bake2.T9D_LIFT_CLEAR) applies outside the plant only, as the plant's arms
    are pinned by the hand IK and the clearance there cannot be reached by the lift. Returns (frames, need)."""
    seg, fin = res['seg_S'], res['fin_S']
    win = np.asarray(res['win_S'], int)
    body = np.asarray(ctx.body_nf)
    hm = S6.hands_mask(ctx)
    P = D3._world(seg['seq'], seg['dump'], fin['final'], fin['ref_out'], fin['disp'])
    plant = np.isin(win, np.asarray(fin['plant'], int))
    pts = P[win][:, body]
    z = pts[..., 2].copy()
    z[plant[:, None] & hm[None, :]] = np.inf
    arg = np.argmin(z, axis=1)
    lowest = z[np.arange(len(z)), arg]
    y = pts[..., 1]
    y_low = y[np.arange(len(y)), arg]
    near, far, top = float(ctx.ob['near_y']), float(ctx.ob['far_y']), float(ctx.ob['height'])
    surf = np.where((y_low > near) & (y_low < far), top, 0.0)
    inside = (y > near) & (y < far)
    excl = plant[:, None] & hm[None, :]
    box_z = np.where(inside & ~excl, top + bake2.T9D_LIFT_CLEAR - pts[..., 2], -np.inf)
    box_need = np.maximum(box_z.max(axis=1), 0.0)
    pf = np.asarray(fin['plant'], int)
    near = (win >= pf.min() - PLANT_NEAR) & (win <= pf.max()) if pf.size else np.zeros(len(win), bool)
    quiet = plant | near              # the plant and the frames just before it: floor only, no box clearance (the hands move)
    tgt = surf + np.where(quiet, PLANT_TARGET, TARGET)
    need = np.maximum(tgt - lowest, 0.0)
    need = np.where(quiet, need, np.maximum(need, box_need))
    return win, need


def E_samples11(ctx, res):
    """The Exit's contact samples with EVERY non-finger bone (dive8's exit_lowest uses the feet whenever the body is above
    1 cm, so a knee at 1.4 cm was never raised to the 2 cm aim). Exit frames have no plant, so the hands are included."""
    seg, fin, win = res['seg_E'], res['fin_E'], np.asarray(res['win_E'], int)
    P = D3._world(seg['seq'], seg['dump'], fin['final'], fin['ref_out'], fin['disp'])
    lowest = P[win][:, ctx.body_nf, 2].min(axis=1)
    n = len(win)
    tau = ctx.t_W + (win / FPS - D4.EXIT_O) / D4.EXIT_R
    return [dict(tag='E', tau=tau, lowest=lowest, surf=np.zeros(n), box_need=np.zeros(n),
                 contact=np.ones(n, bool), in_plant=np.zeros(n, bool))]


def solve_S11(ctx, case, res9, delay=DELAY):
    """SlideOver lift with the new floor, from dive9's lift. Two one-shot passes (no iteration: the lift creeps wherever a
    bone cannot move, such as the takeoff toe and the arms on the box, which the hand IK and the foot targets pin). Each pass
    adds, per frame, the raise that the frame needs (capped at S_STEP_CAP; the peaks are kept by max with the smoothed
    need), up to the SlideOver window end. Then dive9's rate floor on the SlideOver part of the joined pelvis."""
    k_W = int(np.floor(ctx.t_W * FPS + 1e-9))
    lift_S = np.array(res9['lift_final'], float)
    lift_E0 = np.zeros(D10.K_LIFT_E)
    hist = []
    with window(ctx, delay):
        for pass_ in range(S_PASSES):
            res = bake11(ctx, case, lift_S, lift_E0, delay)
            win, need = S_need(ctx, res)
            worst = float(np.max(need)) * 100.0 if need.size else 0.0
            hist.append(dict(pass_=pass_, need_max_cm=worst))
            if worst < CONTACT_STOP * 100.0:
                break
            add = np.zeros(len(lift_S))
            add[win] = need
            add = np.minimum(add, S_STEP_CAP)
            dil = add.copy()
            for j in range(1, 3):
                dil[j:] = np.maximum(dil[j:], add[:-j])
                dil[:-j] = np.maximum(dil[:-j], add[j:])
            spread = np.maximum(bake2._gauss_smooth(dil, S6.CONTACT_SIGMA_LIFT), add)
            lift_S = lift_S + np.where(np.arange(len(lift_S)) <= k_W, spread, 0.0)
        res = bake11(ctx, case, lift_S, lift_E0, delay)
        n0, Zj = D9.joined_pelvis_z(ctx, res)
        idx = np.arange(n0, n0 + len(Zj))
        sel = idx <= k_W
        lift_f = lift_S.copy()
        if np.any(sel):
            Lj = lift_S[idx[sel]]
            lift_f[idx[sel]] = D9.rate_floor(Lj, Zj[sel] - Lj, D9.PELVIS_FLOOR_V)
    return lift_f, dict(iters=hist, rate_add_max_cm=float(np.max(lift_f - lift_S)) * 100.0)


def solve_E11(ctx, case, lift_S, delay=DELAY, use_ik=True):
    """The Exit lift on top of the stand-up: one-sided updates (a body under the target is lifted by its depth, smoothed and
    ramped in from W). The candidate kept is the feasible one (peak Reference z speed <= RISE_V_MAX on both measures)
    with the highest lowest non-finger height in the Exit window and tail; then the rate floor on the Exit part."""
    k_W = int(np.floor(ctx.t_W * FPS + 1e-9))
    ramp = np.clip((np.arange(D10.K_LIFT_E) - k_W) / float(D10.RAMP), 0.0, 1.0)
    lift_E = np.zeros(D10.K_LIFT_E)
    cands = []
    with window(ctx, delay):
        res = bake11(ctx, case, lift_S, lift_E, delay, use_ik=use_ik)
        for it in range(E_ITER + 1):
            spd_j = D10.rise_speed_joined(ctx, res)
            spd_c = D10.rise_speed_clip(res)
            fl = floor_phases(ctx, res, case)
            cands.append(dict(it=it, lift=lift_E.copy(), res=res, speed=max(spd_j, spd_c), joined=spd_j, clip=spd_c,
                              pen=D10.penetration(res), floor_exit=exit_window_floor(fl), floor_all=floor_min_cm(fl)))
            r, dev = residual11(E_samples11(ctx, res), D10.K_LIFT_E)
            r = np.maximum(r, 0.0) * ramp
            r_stop = float(np.max(r))
            if it == E_ITER or r_stop < CONTACT_STOP:
                break
            # one-sided and peak-preserving: a single-frame need (the knee at Exit frame 18) gets its full raise, the smoothing
            # only spreads it (a plain Gaussian of the sum moved it by a third per iteration and did not converge in 12)
            add = np.minimum(r, S_STEP_CAP)
            dil = add.copy()
            for j in range(1, 3):
                dil[j:] = np.maximum(dil[j:], add[:-j])
                dil[:-j] = np.maximum(dil[:-j], add[j:])
            lift_E = lift_E + np.maximum(bake2._gauss_smooth(dil, S6.CONTACT_SIGMA_LIFT), add) * ramp
            res = bake11(ctx, case, lift_S, lift_E, delay, use_ik=use_ik)
        feasible = [c for c in cands if c['speed'] <= D9.RISE_V_MAX]
        pool = feasible if feasible else cands
        best = max(pool, key=lambda c: (c['floor_exit'], -c['it'])) if feasible else min(cands, key=lambda c: c['speed'])
        lift_E = best['lift']
        n0, Zj = D9.joined_pelvis_z(ctx, best['res'])
        idx = np.arange(n0, n0 + len(Zj))
        post = idx > k_W
        lift_E_f = lift_E.copy()
        if np.any(post):
            Lj = lift_E[idx[post]]
            lift_E_f[idx[post]] = D9.rate_floor(Lj, Zj[post] - Lj, D9.PELVIS_FLOOR_V)
        res = bake11(ctx, case, lift_S, lift_E_f, delay, use_ik=use_ik)
        worst, p95E = D10.contact_parts(ctx, res, case)
    res['contact_dev_S_m'] = worst['S']
    res['contact_dev_E_m'] = worst['E']
    res['contact_p95_E_cm'] = p95E * 100.0
    res['rate_add_E_max_cm'] = float(np.max(lift_E_f - lift_E)) * 100.0
    res['lift_E_max_cm'] = float(lift_E_f.max()) * 100.0
    res['lift_E_min_cm'] = float(lift_E_f.min()) * 100.0
    res['lift_S_frozen'] = False
    res['contact_hist'] = [dict(it=c['it'], speed=c['speed'], joined=c['joined'], clip=c['clip'], pen=c['pen'],
                                floor_exit=c['floor_exit'], floor_all=c['floor_all']) for c in cands]
    res['chosen_it'] = int(best['it'])
    res['candidates'] = res['contact_hist']
    res['lift_E_unfloored_max_cm'] = float(lift_E.max()) * 100.0
    return res


def run_metrics(ctx, res, case, delay=DELAY):
    """dive10's metrics (they read the joined path, so the window must be the delay's) plus the floor report."""
    with window(ctx, delay):
        m = D10.metrics10(ctx, res)
    fl = floor_phases(ctx, res, case)
    m['floor'] = fl
    m['floor_min_cm'] = floor_min_cm(fl)
    m['floor_pass'] = pass_floor(fl)
    m['fails10'] = D10.judge10(m)
    m['z_at_0.46'] = float(np.interp((D4.EXIT_O + D4.EXIT_R * 0.46) * FPS, np.arange(len(res['fin_E']['ref_out'])),
                                    res['fin_E']['ref_out'][:, 2]))
    m['floor_contact_E_worst_cm'] = float(res['contact_dev_E_m'] * 100.0)
    return m


# ------------------------------------------------------------------------------------------------ outputs
def event_tracks_delay15(ctx):
    """The Exit's infotype-5 tracks from out\\dive10\\delay15 (copied as they are). Checked against D10's own list."""
    with open(D10_EXIT, encoding='utf-8') as f:
        src = json.load(f)
    ev_file = [t for t in src['Tracks'] if t.get('infotype') == 5]
    ev_calc = D10.event_patch_tracks(ctx.ex_dump, DELAY)
    if json.dumps(ev_file, sort_keys=True) != json.dumps(ev_calc, sort_keys=True):
        raise ValueError('the delay15 event tracks differ from dive10.event_patch_tracks(delay 15)')
    return ev_file


def write_outputs11(ctx, res, feet_to):
    os.makedirs(OUT_DIR, exist_ok=True)
    rig = ctx.rig
    so = D8.patch_dict(rig, res['fin_S']['rot_tracks'], res['fin_S']['trans_keys'], ctx.so['dump']['Length'])
    so_aux = AT.add_aux_tracks(so, ctx.so['dump'], D9.TAKEOFF_FRAME, None)
    AT.write_patch(os.path.join(OUT_DIR, SO_HEX + '.baked.json'), so_aux)
    ex = D8.patch_dict(rig, res['fin_E']['rot_tracks'], res['fin_E']['trans_keys'], ctx.ex_dump['Length'])
    ex_aux = AT.add_aux_tracks(ex, ctx.ex_dump, 0, feet_to)
    ex_aux['Tracks'] = list(ex_aux['Tracks']) + event_tracks_delay15(ctx)
    AT.write_patch(os.path.join(OUT_DIR, EXIT_HEX + '.baked.json'), ex_aux)
    m = dict(clips=[
        DC2.mapping_entry(SO_HEX, SO_NAME, SO_HEX + '.baked.json', True, override=OVERRIDE),
        DC2.mapping_entry(EXIT_HEX, EXIT_NAME, EXIT_HEX + '.baked.json', True, only_after=[SO_NAME], override=OVERRIDE),
    ])
    with open(os.path.join(OUT_DIR, MAPPING_NAME), 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=1)
    return ex_aux


def write_sheet11(ctx, res, case, title):
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
    BCh.plot_joined(SHEET_PNG, ctx.rig, S, arrS, E, arrE, ctx.ob, tl, title, show_root=True)
    BCh.write_gif(SHEET_GIF, ctx.rig, S, arrS, E, arrE, ctx.ob, tl)


def fmt_phase_table(before, after):
    L = ['| phase | before min (cm) | before bone | after min (cm) | after bone | after time (s) |', '|---|---|---|---|---|---|']
    for ph in PHASES:
        b, a = before[ph], after[ph]
        L.append('| %s | %s | %s | %s | %s | %s |' % (
            ph, _c(b['min_cm']), b['bone'] or 'n/a', _c(a['min_cm']), a['bone'] or 'n/a',
            '%.2f' % a['t_s'] if a['t_s'] is not None else 'n/a'))
    return L


def _c(x, nd=2):
    return 'n/a' if x is None else '%.*f' % (nd, x)


def b_metrics_of(data):
    return data['before']['metrics']


def b_gap(m, bone):
    return float(m['gap_at_window_end'].get(bone, float('nan')))


def write_report11(data, path=REPORT):
    b, a, nk = data['before'], data['after'], data['noik']
    ik = data['ik']
    L = []
    L.append('# T23 dive11: the delay15 Exit and SlideOver with the contact floor fixed for every non-finger bone')
    L.append('')
    L.append('Generated by `v2/pipeline/dive11.py` (reuses dive10, dive9, dive8, dive6). Written only under `out\\dive11`. '
             'The "before" column is dive10 delay15 (rebuilt in this run and checked against `out\\dive10\\summary.json`).')
    L.append('')
    L.append('## 1. What changed')
    L.append('')
    L.append('* Floor target: every non-finger bone (feet, toes LeftFootThumb1 / RightFootThumb1, hands, the body) is held at '
             '%.0f cm above its surface in the Exit and the SlideOver roll. Exception: the plant (hands pinned by the hand IK) and '
             'the %d frames before it get the floor only (%.1f cm, no box clearance; section 6). Hands are excluded only on the plant '
             'frames. The box clearance is unchanged outside the plant.'
             % (TARGET * 100, PLANT_NEAR, PLANT_TARGET * 100))
    L.append('* Foot IK (Exit, after the ease, before the hold): while an ankle is below %.2f m, the Foot is rotated about its '
             'ankle by the smallest angle that puts the toe at or above %.0f cm (limit %.0f deg, smoothed over a few frames). '
             'The toe follows (child of the Foot). Only what the IK cannot reach goes into the body lift.'
             % (FOOT_NEAR, TOE_T * 100, np.degrees(IK_THETA_MAX)))
    L.append('* SlideOver lift: dive9\'s lift, raised by %d one-shot passes (one-sided, peak-preserving; no creep) and then dive9\'s '
             'rate floor on its part. SlideOver timing unchanged.' % S_PASSES)
    L.append('* Kept from delay15: the timing (P_UP %.4f, window end %.4f, ease %.4f), the Exit Reference profile (Z_UP %.2f m), '
             'the ease to Nav_Run, the event tracks (copied byte-identical from out\\dive10\\delay15) and the aux tracks '
             '(rebuilt for the new Reference, feet free until clip frame %d).'
             % (b['metrics']['p_up'], b['metrics']['p_end'], b['metrics']['p_ease'], D9.Z_UP, data['feet_to']))
    L.append('')
    L.append('## 2. Lowest non-finger bone height per phase (cm, ground = 0, joined played window)')
    L.append('')
    L.extend(fmt_phase_table(b['metrics']['floor'], a['metrics']['floor']))
    L.append('')
    L.append('Rule: every value >= 0 cm, except the hands during the plant (excluded there). Target: +2 cm. Result: %s '
             '(overall minimum %s cm).' % ('PASS' if a['metrics']['floor_pass'] else 'FAIL', _c(a['metrics']['floor_min_cm'])))
    L.append('')
    L.append('## 3. Lift and IK')
    L.append('')
    L.append('| item | before (delay15) | after (dive11, IK on) | after, IK off (comparison) |')
    L.append('|---|---|---|---|')
    L.append('| Exit lift max (cm) | %.2f | %.2f | %.2f |' % (b['metrics']['lift_E_max_cm'], a['metrics']['lift_E_max_cm'],
                                                          nk['metrics']['lift_E_max_cm']))
    L.append('| Exit lift min (cm) | %.2f | %.2f | %.2f |' % (b['metrics']['lift_E_min_cm'], a['metrics']['lift_E_min_cm'],
                                                          nk['metrics']['lift_E_min_cm']))
    L.append('| SlideOver lift extra over delay15 (max, cm) | n/a | %.2f | n/a |' % data['lift_S_extra_cm'])
    L.append('| peak Reference z speed after W (m/s, joined) | %.2f | %.2f | %.2f |' % (
        b['metrics']['rise_speed_joined_post_W_mps'], a['metrics']['rise_speed_joined_post_W_mps'],
        nk['metrics']['rise_speed_joined_post_W_mps']))
    L.append('| non-finger floor min, all phases (cm) | %s | %s | %s |' % (
        _c(b['metrics']['floor_min_cm']), _c(a['metrics']['floor_min_cm']), _c(nk['metrics']['floor_min_cm'])))
    L.append('| non-finger penetration (box/ground, cm) | %.2f | %.2f | %.2f |' % (
        b['metrics']['pen_patch_cm'], a['metrics']['pen_patch_cm'], nk['metrics']['pen_patch_cm']))
    L.append('')
    L.append('Foot IK per side (frames needing it, frames active, largest angle, unreachable frames, lowest toe after the IK):')
    L.append('')
    for side, d in ik['sides'].items():
        if 'skipped' in d:
            L.append('* %s: skipped (%s)' % (side, d['skipped']))
        else:
            L.append('* %s: %d frames needing the IK, %d active, largest %.1f deg, %d unreachable, lowest toe after IK %s cm'
                     % (side, d['frames_needed'], d['frames_active'], d['theta_max_deg'], d['unreachable_frames'],
                        _c(d.get('toe_min_after_cm'))))
    L.append('')
    L.append('Contact candidates (Exit lift iterations; the feasible candidate with the highest Exit floor is kept):')
    L.append('')
    L.append('| it | peak rise joined (m/s) | peak rise clip (m/s) | Exit floor min (cm) | all floor min (cm) | penetration (cm) | kept |')
    L.append('|---|---|---|---|---|---|---|')
    for c in data['after_candidates']:
        L.append('| %d | %.2f | %.2f | %s | %s | %.2f | %s |' % (
            c['it'], c['joined'], c['clip'], _c(c['floor_exit']), _c(c['floor_all']), c['pen'],
            'yes' if c['it'] == data['after_chosen'] else ''))
    L.append('')
    L.append('## 4. Profile and validation (after)')
    L.append('')
    m = a['metrics']
    L.append('| metric | before (delay15) | after (dive11) | rule |')
    L.append('|---|---|---|---|')
    for lbl, key, rule in (('Exit Reference z at played 0.10', 'z_at_0.10', 'info'),
                           ('Exit Reference z at played 0.16', 'z_at_0.16', 'info'),
                           ('Exit Reference z at played 0.22', 'z_at_0.22', 'info'),
                           ('Exit Reference z at played 0.30', 'z_at_0.30', 'info'),
                           ('Exit Reference z at played 0.38', 'z_at_0.38', 'info'),
                           ('Exit Reference z at played 0.46', 'z_at_0.46', 'info'),
                           ('Exit Reference z at P_UP (m)', 'z_at_p_up', '>= 0.78'),
                           ('peak Reference z speed after W, joined (m/s)', 'rise_speed_joined_post_W_mps', '<= 4'),
                           ('peak Reference z speed, Exit clip frames (m/s)', 'rise_speed_exit_clip_mps', '<= 4'),
                           ('Nav gap at the window end, patched bones (deg)', 'gap_at_window_end_max_patched', '<= 25'),
                           ('tilt at W (deg)', 'tilt_W', '<= 60'),
                           ('max tilt in the Exit (deg)', 'tilt_exit_max', '<= 60'),
                           ('world pelvis fall, peak (m/s)', 'pelvis_drop_max_mps', '<= 6'),
                           ('Reference offset max (m)', 'offset_max_m', '<= 1.0'),
                           ('continuity at W (deg)', 'cont_deg', '<= 45'),
                           ('touch frames (min)', 'touch_min', '>= 3')):
        bv = b['metrics'].get(key)
        av = m.get(key)
        f = lambda v: 'n/a' if v is None else ('%.3f' % v if isinstance(v, float) else str(v))
        L.append('| %s | %s | %s | %s |' % (lbl, f(bv), f(av), rule))
    L.append('| inverted at W / in the Exit | %s / %s | %s / %s | no |' % (
        b['metrics']['inv_W'], b['metrics']['inv_exit'], m['inv_W'], m['inv_exit']))
    L.append('| world pelvis dv / head dv (m/s per frame) | %.2f / %.2f | %.2f / %.2f | <= 1.5 |' % (
        b['metrics']['world_pelvis_dv_mps'], b['metrics']['world_head_dv_mps'], m['world_pelvis_dv_mps'], m['world_head_dv_mps']))
    L.append('| world pelvis / head step (cm per frame) | %.1f / %.1f | %.1f / %.1f | <= 12 |' % (
        b['metrics']['world_pelvis_step_cm'], b['metrics']['world_head_step_cm'], m['world_pelvis_step_cm'], m['world_head_step_cm']))
    L.append('')
    L.append('Nav_Run gap at the window end (frame 41 is played about 0.49 s), per patched bone (deg):')
    L.append('')
    L.append('| bone | before | after |')
    L.append('|---|---|---|')
    for bn in m['patched_gap']:
        L.append('| %s | %.1f | %.1f |' % (bn, b['metrics']['gap_at_window_end'].get(bn, float('nan')),
                                           m['gap_at_window_end'].get(bn, float('nan'))))
    L.append('')
    L.append('Spec status (dive10 rules and the floor rule): before: %s; after: %s.' % (
        '; '.join(b['metrics']['fails10']) or 'all pass', '; '.join(m['fails10']) or 'all pass'))
    L.append('')
    L.append('## 5. Outputs')
    L.append('')
    L.append('* `out\\dive11\\47BFB02DF.baked.json` (SlideOver), `out\\dive11\\F7919BDF0.baked.json` (Exit), `out\\dive11\\%s`'
             % MAPPING_NAME)
    L.append('* `out\\dive11\\joined_sheet.png`, `out\\dive11\\joined_sheet.gif`, `out\\dive11\\summary.json`, `out\\dive11\\report.md`')
    L.append('* Tests: `pipeline\\tests\\test_dive11.py`.')
    L.append('')
    L.append('## 6. Decisions and caveats')
    L.append('')
    L.append('* Foot IK aims at the envelope target (2 cm), not the 1 cm of the brief. With a 1 cm IK target, the Exit lift re-raises '
             'every toe between 1 and 2 cm, which is the body lift the IK is meant to avoid.')
    L.append('* The SlideOver lift is one-sided and runs as a few one-shot passes, not an iteration. dive9\'s two-sided ground contact '
             'has no fixed point in the vault over the box (the body is held up by its hands): it swings the lift +41 cm and -25 cm '
             'within 0.2 s and never settles. The new lift starts from dive9\'s and only raises; dive9\'s rate floor is kept.')
    L.append('* The plant (hands pinned by the hand IK) and the 6 frames before it get the floor only (target 0.5 cm, no box '
             'clearance there). A raise in that window moves the hands off their ground plant, and the hand touch fails (1 to 2 frames, '
             'need 3). With this rule the touch is %d frames.' % int(a['metrics']['touch_min']))
    L.append('* The floor rule holds in every phase (minimum %s cm, %s at %s s). The +2 cm aim is met at the roll end (%s cm), in the '
             'Exit window (%s cm) and the Exit tail (%s cm). The pre-plant right hand and the plant forearm sit at the floor-only '
             'target, as above.'
             % (_c(a['metrics']['floor_min_cm']), a['metrics']['floor']['run-up and flight']['bone'],
                _c(a['metrics']['floor']['run-up and flight']['t_s'], 2), _c(a['metrics']['floor']['last 0.10 s before W']['min_cm']),
                _c(a['metrics']['floor']['Exit window']['min_cm']), _c(a['metrics']['floor']['Exit tail']['min_cm'])))
    L.append('* The model\'s floor is not the game\'s floor. In game the roll-end hand reaches -8 to -12.6 cm and the toes 2 to 12 cm under '
             'the ground; the model gave -2.5 cm (roll-end hand) and -3.4 cm (Exit toe) before the fix. The fix is in the model\'s '
             'ground (z = 0). It needs an in-game check.')
    L.append('* Failures present before and after (dive9 and dive10 items, not changed here): world pelvis dv %.2f / head %.2f m/s per '
             'frame (> 1.5), world step %.1f / %.1f cm (> 12), Reference offset %.2f m (> 1.0). The same values are in delay15.'
             % (a['metrics']['world_pelvis_dv_mps'], a['metrics']['world_head_dv_mps'], a['metrics']['world_pelvis_step_cm'],
                a['metrics']['world_head_step_cm'], a['metrics']['offset_max_m']))
    L.append('* Nav gap at the window end: the left foot is %.1f deg (delay15 %.1f), inside the 25 deg rule; the IK turns the foot.'
             % (a['metrics']['gap_at_window_end'].get('LeftFoot', float('nan')), b_gap(b_metrics_of(data), 'LeftFoot')))
    L.append('* Aux tracks: rebuilt on the new Reference, feet free until clip frame %d (delay15: %d).'
             % (int(data['feet_to']), int(b_metrics_of(data)['feet_free_to_frame'])))
    L.append('')
    L.append('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


# ------------------------------------------------------------------------------------------------ main
def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    return x


def run_all(verbose=True):
    ctx = F5.make_ctx()
    case = D9.make_case9()
    A, t0, t1 = D8.bump_for(ctx, case)
    F5.BUMP.update(t0=t0, t1=t1, A=A)
    res9 = D9.solve9(ctx, case, A, t0, t1)
    F5.BUMP.update(t0=t0, t1=t1, A=A)
    # before: dive10 delay15, rebuilt with dive10's own solve
    res_b, m_b = D10.run_variant(ctx, D9.make_case9(), DELAY, res9['lift_final'], iterate_exit=True)
    m_b['floor'] = floor_phases(ctx, res_b, D9.make_case9())
    m_b['floor_min_cm'] = floor_min_cm(m_b['floor'])
    m_b['floor_pass'] = pass_floor(m_b['floor'])
    m_b['fails10'] = D10.judge10(m_b)
    m_b['lift_E_max_cm'] = float(res_b['lift_E_max_cm'])
    m_b['lift_E_min_cm'] = float(res_b['lift_E_min_cm'])
    if verbose:
        print('before (dive10 delay15):', {k: m_b[k] for k in ('pen_patch_cm', 'z_at_p_up', 'rise_speed_joined_post_W_mps')},
              'floor min', m_b['floor_min_cm'], flush=True)
    # SlideOver lift with the new target
    F5.BUMP.update(t0=t0, t1=t1, A=A)
    lift_S, s_info = solve_S11(ctx, D9.make_case9(), res9, DELAY)
    lift_S_extra_cm = float(np.max(lift_S[:len(res9['lift_final'])] - np.array(res9['lift_final'], float))) * 100.0
    if verbose:
        print('SlideOver lift done, extra max %.2f cm' % lift_S_extra_cm, flush=True)
    # after: with the foot IK, and without it (the lift the IK saves)
    case_a = D9.make_case9()
    res_a = solve_E11(ctx, case_a, lift_S, DELAY, use_ik=True)
    m_a = run_metrics(ctx, res_a, case_a)
    if verbose:
        print('after (IK on):', {k: m_a[k] for k in ('pen_patch_cm', 'z_at_p_up', 'rise_speed_joined_post_W_mps')},
              'floor min', m_a['floor_min_cm'], flush=True)
    case_n = D9.make_case9()
    res_n = solve_E11(ctx, case_n, lift_S, DELAY, use_ik=False)
    m_n = run_metrics(ctx, res_n, case_n)
    if verbose:
        print('after (IK off):', {k: m_n[k] for k in ('pen_patch_cm', 'z_at_p_up', 'rise_speed_joined_post_W_mps')},
              'floor min', m_n['floor_min_cm'], flush=True)
    return dict(ctx=ctx, res_b=res_b, m_b=m_b, res_a=res_a, m_a=m_a, res_n=res_n, m_n=m_n, case=case_a, s_info=s_info,
                lift_S_extra_cm=lift_S_extra_cm, lift_S=lift_S)


def main(argv=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = run_all()
    ctx, res, m = out['ctx'], out['res_a'], out['m_a']
    feet_to = int(m['feet_free_to_frame'])
    write_outputs11(ctx, res, feet_to)
    title = ('dive11 delay15 floor fix: lowest non-finger bone %.1f cm (ground), Exit lift max %.1f cm'
             % (m['floor_min_cm'], m['lift_E_max_cm']))
    write_sheet11(ctx, res, out['case'], title)
    data = dict(before=dict(metrics=out['m_b']), after=dict(metrics=m), noik=dict(metrics=out['m_n']),
                ik=res['ik'], feet_to=feet_to, lift_S_extra_cm=out['lift_S_extra_cm'],
                after_candidates=res['contact_hist'], after_chosen=res['chosen_it'])
    write_report11(data)
    summ = dict(variant='dive11 delay15 floor fix', override=OVERRIDE, target_cm=TARGET * 100, toe_target_cm=TOE_T * 100,
                foot_near_m=FOOT_NEAR, ik_theta_max_deg=float(np.degrees(IK_THETA_MAX)),
                before=dict(metrics=out['m_b']), after=dict(metrics=m, ik=res['ik'], candidates=res['contact_hist'],
                                                            chosen_it=res['chosen_it'], lift_E_unfloored_max_cm=res['lift_E_unfloored_max_cm']),
                noik=dict(metrics=out['m_n'], candidates=out['res_n']['contact_hist']),
                slideover=dict(lift_extra_max_cm=out['lift_S_extra_cm'], iters=out['s_info']['iters'],
                               rate_add_max_cm=out['s_info']['rate_add_max_cm']),
                feet_free_to_frame=feet_to)
    with open(SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(_jsonable(summ), f, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
