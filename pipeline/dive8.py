"""T20 (dive8): two variants for in-game A/B testing of the dive roll on the 1 m SlideOver.

B_fastroll
  The WD2 roll is faster so that it finishes as close to W as the hand rule allows (see the report: finishing exactly
  at W needs the same speed-up as ~2.8x in the plant and the roll, which is not a moderate change). The plant is
  compressed to k_plant = 1.5 and the roll runs at k_roll = 2.5 from the plant end until the Exit window end (the roll
  continues into the Exit, so the Exit carries the roll tail and the stand-up: WD2 1.03 -> 1.60).
  In the Exit the Reference converges to the original standing Reference by the end of the Exit window (smoothstep over
  the window). Contact: the body (lowest non-finger point, hands out during the plant) in the SlideOver, the feet in the
  Exit. Both patches carry the aux tracks: the SlideOver with the feet free from the takeoff, the Exit with the feet free
  for the whole clip.

C_noexit
  The dive7 SlideOver unchanged (it already carries the aux tracks). The Exit is a NO-OP: the original F7919BDF0 values,
  sampled every frame for every rotation track that dive7's Exit has, plus the Reference translation. No aux changes.

Outputs (only under out\\dive8):
  B_fastroll\\47BFB02DF.baked.json, B_fastroll\\F7919BDF0.baked.json, B_fastroll\\mapping_dive8_B.json, B_fastroll\\joined_sheet.*
  C_noexit\\47BFB02DF.baked.json, C_noexit\\F7919BDF0.baked.json, C_noexit\\mapping_dive8_C.json
  report.md, summary.json
Run: python dive8.py
"""
import json
import os
import shutil
import sys
import zlib

import numpy as np

import acuclip
import aux_tracks as AT
import bake2
import bake_chain as BC
import config
import dive3 as D3
import dive4 as D4
import dive4b as B
import dive4c as C
import dive5 as F5
import dive6 as S6
import dumps
import dive_chain2 as DC2

FPS = 60
T_TO, S_SCALE, SHIFT = 0.74, 1.03, 0.02
PLANT_W = (0.75, 0.92)
FEET_W = 1.2
K_PLANT_B, K_ROLL_B = 1.5, 2.5
TAKEOFF_FRAME = int(round(0.45 * FPS))         # ACU takeoff of the Up100 clip (0.45 s)
OUT_DIR = os.path.join(config.OUT, 'dive8')
B_DIR = os.path.join(OUT_DIR, 'B_fastroll')
C_DIR = os.path.join(OUT_DIR, 'C_noexit')
DIVE6_PATCHES = os.path.join(config.OUT, 'dive6', 'patches')
DIVE7_PATCHES = os.path.join(config.OUT, 'dive7', 'patches')
SO_NAME, SO_HEX = F5.SO_NAME, F5.SO_HEX
EXIT_NAME, EXIT_HEX = F5.EXIT_NAME, F5.EXIT_HEX
REF_ID = zlib.crc32(b'Reference')
FOOT_BONES = ('LeftFoot', 'RightFoot', 'LeftFootThumb1', 'RightFootThumb1')
PAD = 2.0 / FPS
BODY_FLOOR = bake2.T9D_LIFT_CLEAR       # m: the lowest non-finger point is kept at or above this in the Exit


# ------------------------------------------------------------------------------------------------ time map
class Timing8:
    """WD2 time at ACU clip time t. Before T_to the run-up (clamped at 0.2 s); T_to to T_to + 0.30 at unit speed (the flight);
    the plant (0.75 .. 0.92) at k_plant; the roll (0.92 onwards, into the Exit) at k_roll."""

    def __init__(self, T_to, k_plant, k_roll):
        self.T_to = float(T_to)
        self.k_plant = float(k_plant)
        self.k = float(k_roll)                  # the Exit uses tm.k (bake code)
        self.t_pl0 = self.T_to + (PLANT_W[0] - 0.45)
        self.t_pe = self.t_pl0 + (PLANT_W[1] - PLANT_W[0]) / self.k_plant
        self.t_feet = self.t_pe + (FEET_W - PLANT_W[1]) / self.k

    def w(self, t):
        t = np.asarray(t, float)
        pre = np.maximum(0.2, 0.45 + (t - self.T_to))
        plant = PLANT_W[0] + self.k_plant * (t - self.t_pl0)
        post = PLANT_W[1] + self.k * (t - self.t_pe)
        return np.where(t <= self.t_pl0, pre, np.where(t <= self.t_pe, plant, post))


def make_case(k_plant, k_roll):
    c = B.Case(T_TO, S_SCALE, SHIFT, k_roll)
    c.tm = Timing8(T_TO, k_plant, k_roll)
    return c


# ------------------------------------------------------------------------------------------------ one bake
def bake8(ctx, case, lift_k):
    """dive6.bake6 with the case's timing: the Exit Reference converges to the original standing Reference by the end
    of the Exit window (smoothstep over the window), not after it."""
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
    plant_S = win_S[(tw >= PLANT_W[0] - 1e-9) & (tw <= PLANT_W[1] + 1e-9)]
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
    off_W = fin_S['ref_out'][last] - seg_E['ref0_orig'][int(win_E[0])]
    inside = tE >= D4.EXIT_O - 1e-9
    c = np.where(inside, 1.0, 0.0) * (1.0 - B.smoothstep((tE - D4.EXIT_O) / max(ctx.t_end_E - D4.EXIT_O, 1e-9)))
    tau_E = ctx.t_W + (tE - D4.EXIT_O) / D4.EXIT_R
    ref_out_E = seg_E['ref0_orig'] + c[:, None] * off_W[None, :]
    ref_out_E[:, 2] += np.where(inside, B.lift_eval(lift_k, tau_E), 0.0)
    fin_E = B.bake_clip5(ctx, seg_E, pl, win_E, np.array([], int), None, ctx.disp_ex, ctx.ground_ob, None,
                         hand_targets=False, ref_out_fixed=ref_out_E)
    fin_E['win'] = win_E
    fin_E['plant'] = np.array([], int)
    fin_E['targets'] = {}
    fin_E['disp'] = ctx.disp_ex
    fW_E = min(int(round(ctx.t_end_E * FPS)), len(tE) - 1)
    gap_pre = D3._gap_to_next(ctx, fin_E['final'], fW_E)              # ease is off: reported only
    fin_E['rot_tracks'] = {ai: bake2.rotation_track(dumps.track(ctx.ex_dump, ctx.rig.ac[ai]['hash'], 0) or [],
                                                   fin_E['final'][:, ai], max(0, int(win_E[0]) - 1),
                                                   int(seg_E['F'][-1])) for ai in seg_E['patset']}
    met_E, arr_E = D3._validate(ctx, seg_E, fin_E, ctx.t_end_E, ctx.span_e, D4.EXIT_O, next_dump=ctx.next_dump)
    cont = D4.cont_at_joint(ctx, fin_S, fin_E, last, int(win_E[0]))
    world = B.world_metrics(ctx, seg_S, fin_S, win_S, seg_E, fin_E, win_E)
    return dict(case=case, seg_S=seg_S, seg_E=seg_E, fin_S=fin_S, fin_E=fin_E, met_S=met_S, met_E=met_E, arr_S=arr_S,
                arr_E=arr_E, cont=cont, world=world, gap_pre=gap_pre, ease_used=False, win_S=win_S, win_E=win_E,
                off_W=off_W, w_exit=w_exit, w_W=w_W, lift_k=np.array(lift_k, float), c_E=c,
                dS=fin_S['ref_out'] - seg_S['ref0_orig'], dE=fin_E['ref_out'] - seg_E['ref0_orig'])


# ------------------------------------------------------------------------------------------------ contact
def exit_lowest(lowest_body, lowest_feet):
    """The Exit's contact point per sample: the lowest foot, unless the lowest non-finger point is below the body floor
    (BODY_FLOOR), in which case the body is the point that sits on the ground and the feet float."""
    return np.where(np.asarray(lowest_body) < BODY_FLOOR, lowest_body, lowest_feet)


def contact_samples8(ctx, res, case):
    """Per played sample: the SlideOver's lowest non-finger point (hands out during the plant, the surface under it) and
    the Exit's lowest FOOT (the ground, two-sided). Same record layout as dive6.contact_samples."""
    tm = case.tm
    lo, hi = tm.t_pl0 - PAD, tm.t_pe + PAD
    rig = ctx.rig
    hm = S6.hands_mask(ctx)
    feet = [rig.ac_by_hash[zlib.crc32(n.encode())] for n in FOOT_BONES]
    out = []
    for tag, seg, fin, win in (('S', res['seg_S'], res['fin_S'], res['win_S']),
                               ('E', res['seg_E'], res['fin_E'], res['win_E'])):
        P = D3._world(seg['seq'], seg['dump'], fin['final'], fin['ref_out'], fin['disp'])
        win = np.asarray(win, int)
        tau = win / FPS
        if tag == 'E':
            tau = ctx.t_W + (tau - D4.EXIT_O) / D4.EXIT_R
            # feet on the ground (two-sided), but the body floor (lowest non-finger point >= 0) always wins: where the
            # rolled pose would put the body underground, the feet float instead
            lowest_feet = P[win][:, feet, 2].min(axis=1)
            lowest_body = P[win][:, ctx.body_nf, 2].min(axis=1)
            lowest = exit_lowest(lowest_body, lowest_feet)
            n = len(win)
            out.append(dict(tag=tag, tau=tau, lowest=lowest, surf=np.zeros(n), box_need=np.zeros(n),
                            contact=np.ones(n, bool), in_plant=np.zeros(n, bool)))
            continue
        pts = P[win][:, ctx.body_nf]
        in_plant = (tau >= lo - 1e-9) & (tau <= hi + 1e-9)
        z = pts[..., 2].copy()
        z[in_plant[:, None] & hm[None, :]] = np.inf
        arg = np.argmin(z, axis=1)
        lowest = z[np.arange(len(z)), arg]
        y = pts[..., 1]
        y_low = y[np.arange(len(y)), arg]
        surf = np.where((y_low > float(ctx.ob['near_y'])) & (y_low < float(ctx.ob['far_y'])), float(ctx.ob['height']), 0.0)
        inside = (y > float(ctx.ob['near_y'])) & (y < float(ctx.ob['far_y']))
        excl = in_plant[:, None] & hm[None, :]
        box_z = np.where(inside & ~excl, float(ctx.ob['height']) + bake2.T9D_LIFT_CLEAR - pts[..., 2], -np.inf)
        box_need = np.maximum(box_z.max(axis=1), 0.0)
        out.append(dict(tag=tag, tau=tau, lowest=lowest, surf=surf, box_need=box_need, contact=tau >= lo - 1e-9,
                        in_plant=in_plant))
    return out


def bump_for(ctx, case):
    """dive5's rule (the bump from the box need of a zero-bump bake, the middle of the span); the timing of this case."""
    F5.BUMP.update(t0=None, t1=None, A=0.0)
    res0 = bake8(ctx, case, np.zeros(B.LIFT_K))
    t, need = F5.box_need(ctx, res0)
    t0 = case.T_to - F5.PRE_TO
    t1 = case.T_to + F5.PLANT_LEAD
    shape = F5.raised_cosine_span(t, t0, t1)
    mid = shape >= 0.5
    A = float(np.max(need[mid] / shape[mid])) if np.any(need[mid] > 0) else 0.0
    return A, t0, t1


def bake_contact8(ctx, case):
    A, t0, t1 = bump_for(ctx, case)
    F5.BUMP.update(t0=t0, t1=t1, A=A)
    lift = np.zeros(B.LIFT_K)
    hist = []
    res = None
    dev = 0.0
    for it in range(S6.CONTACT_ITER + 1):
        res = bake8(ctx, case, lift)
        r, dev = S6.residual_from_samples(contact_samples8(ctx, res, case), len(lift))
        hist.append(dict(it=it, worst_dev_cm=dev * 100.0, resid_max_cm=float(np.max(np.abs(r))) * 100.0,
                         lift_max_cm=float(lift.max()) * 100.0))
        if it == S6.CONTACT_ITER or (dev <= S6.CONTACT_TOL and np.max(np.abs(r)) < 0.003):
            break
        lift = bake2._gauss_smooth(lift + r, S6.CONTACT_SIGMA_LIFT)
    devs = []
    for s in contact_samples8(ctx, res, case):
        devs.append(np.abs(s['surf'] + S6.CONTACT_TARGET - s['lowest'])[s['contact']])
    devs = np.concatenate(devs)
    res['contact_hist'] = hist
    res['contact_dev_m'] = dev
    res['contact_frac_tol'] = float(np.mean(devs <= S6.CONTACT_TOL))
    res['contact_p95_cm'] = float(np.percentile(devs, 95) * 100.0)
    res['lift_min_cm'] = float(lift.min()) * 100.0
    res['lift_max_cm'] = float(lift.max()) * 100.0
    res['bump'] = dict(A=A, t0=t0, t1=t1)
    return res


# ------------------------------------------------------------------------------------------------ patches
def patch_dict(rig, rot_tracks, trans_keys, length):
    """The same content as dive_chain2.write_patch, as a dict (rotation tracks and the Reference translation)."""
    tr = []
    for ai in sorted(rot_tracks, key=lambda a: rig.ac[a]['hash']):
        kf = [{'Frame': int(k['Frame']), 'LocalRotation': [round(float(x), 9) for x in k['LocalRotation']]}
              for k in rot_tracks[ai]]
        tr.append({'TrackID': str(rig.ac[ai]['hash']), 'codec': 1, 'infotype': 0, 'Keyframes': kf})
    tr.append({'TrackID': str(bake2.REF_TID), 'codec': 27, 'infotype': 3,
               'Keyframes': [{'Frame': int(k['Frame']), 'LocalPosition': [round(float(x), 9) for x in k['LocalPosition']]}
                             for k in trans_keys]})
    return {'Length': round(float(length), 6), 'Tracks': tr}


def noop_exit_patch(dump, ref_patch):
    """The original clip, written as a patch: every rotation track that the reference patch has (and the dump has), with
    the original values sampled at every frame, plus the Reference translation sampled at every frame."""
    last = int(round(float(dump['Length']) * FPS))
    ids = sorted({int(t['TrackID']) for t in ref_patch['Tracks'] if t['codec'] == 1 and t['infotype'] == 0})
    tracks, skipped = [], []
    for tid in ids:
        keys = dumps.track(dump, tid, 0)
        if not keys:
            skipped.append(tid)
            continue
        kf = [{'Frame': f, 'LocalRotation': [float(x) for x in dumps.sample_rotation(keys, f)]} for f in range(last + 1)]
        tracks.append({'TrackID': str(tid), 'codec': 1, 'infotype': 0, 'Keyframes': kf})
    pk = dumps.track(dump, REF_ID, 3)
    kf = [{'Frame': f, 'LocalPosition': [float(x) for x in dumps.sample_position(pk, f)]} for f in range(last + 1)]
    tracks.append({'TrackID': str(REF_ID), 'codec': 27, 'infotype': 3, 'Keyframes': kf})
    return {'Length': dump['Length'], 'Tracks': tracks}, skipped


def sanity_noop(dump, patch):
    """Max error of the no-op Exit against the original: at the frames and at the half frames, rotations in degrees and
    positions in metres, for every track of the patch. Sampling is the same function for both (the game's sampling is
    linear for positions and slerp for rotations; the half frames show the effect of the per-frame keys)."""
    worst = dict(rot_frame_deg=0.0, rot_half_deg=0.0, pos_frame_m=0.0, pos_half_m=0.0, tracks=0)
    for t in patch['Tracks']:
        tid = int(t['TrackID'])
        if t['infotype'] == 0:
            ok = dumps.track(dump, tid, 0)
            if not ok:
                continue
            worst['tracks'] += 1
            for f in np.arange(0, round(dump['Length'] * FPS) + 1e-9, 0.5):
                a = dumps.sample_rotation(t['Keyframes'], float(f))
                b = dumps.sample_rotation(ok, float(f))
                err = np.degrees(2.0 * np.arccos(min(1.0, abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))))))
                key = 'rot_frame_deg' if float(f) == int(f) else 'rot_half_deg'
                worst[key] = max(worst[key], float(err))
        else:
            ok = dumps.track(dump, tid, 3)
            if not ok:
                continue
            for f in np.arange(0, round(dump['Length'] * FPS) + 1e-9, 0.5):
                a = dumps.sample_position(t['Keyframes'], float(f))
                b = dumps.sample_position(ok, float(f))
                key = 'pos_frame_m' if float(f) == int(f) else 'pos_half_m'
                worst[key] = max(worst[key], float(np.abs(a - b).max()))
    return worst


# ------------------------------------------------------------------------------------------------ outputs
def build_B(ctx, res):
    rig = ctx.rig
    so = patch_dict(rig, res['fin_S']['rot_tracks'], res['fin_S']['trans_keys'], ctx.so['dump']['Length'])
    ex = patch_dict(rig, res['fin_E']['rot_tracks'], res['fin_E']['trans_keys'], ctx.ex_dump['Length'])
    so_aux = AT.add_aux_tracks(so, ctx.so['dump'], TAKEOFF_FRAME, None)
    ex_aux = AT.add_aux_tracks(ex, ctx.ex_dump, 0, None)
    os.makedirs(B_DIR, exist_ok=True)
    AT.write_patch(os.path.join(B_DIR, SO_HEX + '.baked.json'), so_aux)
    AT.write_patch(os.path.join(B_DIR, EXIT_HEX + '.baked.json'), ex_aux)
    return so_aux, ex_aux


def write_mapping(path, override, tag):
    m = dict(clips=[
        DC2.mapping_entry(SO_HEX, SO_NAME, SO_HEX + '.baked.json', True, override=override),
        DC2.mapping_entry(EXIT_HEX, EXIT_NAME, EXIT_HEX + '.baked.json', True, only_after=[SO_NAME], override=override),
    ])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=1)
    return m


def build_C(ctx):
    os.makedirs(C_DIR, exist_ok=True)
    shutil.copyfile(os.path.join(DIVE7_PATCHES, SO_HEX + '.baked.json'), os.path.join(C_DIR, SO_HEX + '.baked.json'))
    ref_exit = json.load(open(os.path.join(DIVE7_PATCHES, EXIT_HEX + '.baked.json'), encoding='utf-8'))
    exit_patch, skipped = noop_exit_patch(ctx.ex_dump, ref_exit)
    AT.write_patch(os.path.join(C_DIR, EXIT_HEX + '.baked.json'), exit_patch)
    return exit_patch, skipped


def write_sheet_B(ctx, res, path_png, path_gif):
    import bake_chain as BCh
    PfS = res['world']['_PS']
    PfE = D3._world(res['seg_E']['seq'], res['seg_E']['dump'], res['fin_E']['final'], res['fin_E']['ref_out'],
                    res['fin_E']['disp'])
    PoS = acuclip.world_positions(ctx.so['seq'], ctx.so['dump'])
    PoE = acuclip.world_positions(ctx.ex_seq, ctx.ex_dump)
    arrS = dict(Pf=PfS, Po=PoS)
    arrE = dict(Pf=PfE, Po=PoE)
    tl = B.Tl(ctx, res['case'], res['w_exit'])
    S = dict(F=res['seg_S']['F'], disp=ctx.disp_so, name=SO_NAME)
    E = dict(F=res['seg_E']['F'], disp=ctx.disp_ex, name=EXIT_NAME)
    c = res['case']
    title = ('dive8 B fastroll: T_to %.2f, k_plant %.1f, k_roll %.1f, bump %.2f m, contact dev %.1f cm' % (
        c.T_to, c.tm.k_plant, c.tm.k, res['bump']['A'], res['contact_dev_m'] * 100.0))
    BCh.plot_joined(path_png, ctx.rig, S, arrS, E, arrE, ctx.ob, tl, title, show_root=True)
    BCh.write_gif(path_gif, ctx.rig, S, arrS, E, arrE, ctx.ob, tl)


def report(ctx, res, st, heights, feet_info, sanity, exit_skipped, k_needed, times):
    mS, mE, w = res['met_S'], res['met_E'], res['world']
    L = []
    L.append('# T20 dive8: two variants for in-game A/B testing (B_fastroll, C_noexit)')
    L.append('')
    L.append('Generated by `v2/pipeline/dive8.py` from dive6 (bake), dive7 (aux tracks) and aux_tracks.py. Written only under `out\\dive8`.')
    L.append('')
    L.append('## 1. Timing: can the roll finish before W?')
    L.append('')
    L.append('Takeoff T_to %.2f s, plant start %.3f s (clip), W = %.3f s. With the hand rule the plant end is at T_to + 0.47 = '
             '%.3f s at unit speed, after W. The roll (0.92 -> 1.20 WD2 s, 0.28 s) then cannot finish before W at moderate speed.'
             % (T_TO, T_TO + 0.30, ctx.t_W, T_TO + 0.47))
    L.append('')
    L.append('| variant | k_plant | k_roll | plant end (clip s) | feet-down (clip s) | feet-down minus W (s) |')
    L.append('|---|---|---|---|---|---|')
    for row in times:
        L.append('| %s | %.2f | %.2f | %.3f | %.3f | %+.3f |' % row)
    L.append('')
    L.append('Speeds that would put feet-down exactly at W: %s. Those are not moderate, so B keeps k_plant 1.5, k_roll 2.5 '
             '(feet-down %+.3f s after W) and lets the roll tail run into the Exit.' % (k_needed, times[1][5]))
    L.append('')
    L.append('## 2. B_fastroll: validation')
    L.append('')
    L.append('Enable state (coordinator A/B): both clips written enabled; measured spec status: SlideOver **%s**, Exit **%s**.'
             % (st['S']['status'], st['E']['status']))
    L.append('Spec fails: %s' % ('; '.join(st['S']['fails']) or 'none'))
    L.append('')
    L.append('| metric | SlideOver | Exit | rule |')
    L.append('|---|---|---|---|')
    L.append('| non-finger penetration, box / ground (cm) | %.1f / %.1f | %.1f / %.1f | <= 5 |'
             % (mS['pen_box_cm'], mS['pen_ground_cm'], mE['pen_box_cm'], mE['pen_ground_cm']))
    L.append('| hand touch (frames, min) | %d | n/a | >= 4 |' % mS['touch']['min'])
    L.append('| contact: %% of contact samples within 3 cm / 95th pct (cm) | %.0f%% / %.1f | | info (surface: body in SlideOver, feet in Exit) |'
             % (res['contact_frac_tol'] * 100.0, res['contact_p95_cm']))
    L.append('| world pelvis: step cm/frame / dv m/s per frame | %.2f / %.2f | joined | <= 12 / 1.5 |'
             % (w['pelvis']['step_cm'], w['pelvis']['dv_ms']))
    L.append('| world head: step cm/frame / dv m/s per frame | %.2f / %.2f | joined | <= 12 / 1.5 |'
             % (w['head']['step_cm'], w['head']['dv_ms']))
    L.append('| Reference offset max (m) | %.3f | | <= 1.0 |' % w['offset_max_m'])
    L.append('| continuity at W (deg) | %.1f (%s) | | <= 45 |' % (res['cont']['max_deg'], res['cont']['worst']))
    L.append('| Exit Reference converges by the window end (clip s) | | %.3f | by window end |' % ctx.t_end_E)
    L.append('')
    L.append('Feet and body through the roll and the Exit (entity-local heights, clip time):')
    L.append('')
    L.append('| joined time (s) | Ref z (hips) | head z | lowest foot z | world hips z |')
    L.append('|---|---|---|---|---|')
    for r in heights:
        L.append('| %.2f | %.3f | %.3f | %.3f | %.3f |' % (r['tau'], r['ref_z'], r['head_z'], r['feet_z'], r['ref_world_z']))
    L.append('')
    L.append('Contact: %.0f%% of the samples within 3 cm; worst %.1f cm (see the sheet for the box-edge samples). Lift %+.1f .. %+.1f cm; %d iterations.'
             % (res['contact_frac_tol'] * 100.0, res['contact_dev_m'] * 100.0, res['lift_min_cm'], res['lift_max_cm'],
                len(res['contact_hist']) - 1))
    L.append('')
    L.append('Aux tracks: SlideOver feet free from frame %d (takeoff) to the clip end; Exit feet free for the whole clip. '
             'Inverse-Reference tracks added to both.' % TAKEOFF_FRAME)
    L.append('')
    L.append('## 3. C_noexit: sanity check (the Exit equals the original)')
    L.append('')
    L.append('Exit patch = original F7919BDF0 values, sampled every frame for %d rotation tracks (the ones dive7\'s Exit has that the '
             'dump has; skipped: %s) and the Reference translation. No aux tracks.' % (sanity['tracks'], exit_skipped or 'none'))
    L.append('')
    L.append('Max error against the original, sampled by the same functions: rotations %.2e deg at the frames, %.3f deg at the '
             'half frames; positions %.2e m at the frames, %.2e m at the half frames.'
             % (sanity['rot_frame_deg'], sanity['rot_half_deg'], sanity['pos_frame_m'], sanity['pos_half_m']))
    L.append('')
    L.append('The SlideOver of C is dive7\'s, unchanged.')
    L.append('')
    with open(os.path.join(OUT_DIR, 'report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


def main(argv=None):
    ctx = F5.make_ctx()
    case = make_case(K_PLANT_B, K_ROLL_B)
    # the feet and plant timing table
    times = []
    times.append(('dive6 timing (k 1 plant and roll)', 1.0, 1.0, T_TO + 0.47, T_TO + 0.75, (T_TO + 0.75) - ctx.t_W))
    tb = Timing8(T_TO, K_PLANT_B, K_ROLL_B)
    times.append(('B_fastroll', K_PLANT_B, K_ROLL_B, tb.t_pe, tb.t_feet, tb.t_feet - ctx.t_W))
    span_pl = (PLANT_W[1] - PLANT_W[0])
    t_plant_start = T_TO + 0.30
    need_k = (span_pl + (FEET_W - PLANT_W[1])) / max(ctx.t_W - t_plant_start, 1e-9)
    k_needed = 'k_plant = k_roll = %.2f (plant start %.3f s, W %.3f s)' % (need_k, t_plant_start, ctx.t_W)
    res = bake_contact8(ctx, case)
    st = B.judge(res)
    so_aux, ex_aux = build_B(ctx, res)
    write_sheet_B(ctx, res, os.path.join(B_DIR, 'joined_sheet.png'), os.path.join(B_DIR, 'joined_sheet.gif'))
    write_mapping(os.path.join(B_DIR, 'mapping_dive8_B.json'), 'preview: dive8 B fastroll', 'B')
    exit_patch, skipped = build_C(ctx)
    write_mapping(os.path.join(C_DIR, 'mapping_dive8_C.json'), 'preview: dive8 C noexit', 'C')
    sanity = sanity_noop(ctx.ex_dump, exit_patch)
    heights = S6.heights_table(ctx, res)
    report(ctx, res, st, heights, None, sanity, skipped, k_needed, times)
    summary = dict(B=dict(status=st, contact_frac=res['contact_frac_tol'], contact_dev_cm=res['contact_dev_m'] * 100.0,
                          world=dict((k, v) for k, v in res['world'].items() if not k.startswith('_')),
                          touch=res['met_S']['touch'], cont=res['cont']['max_deg'], bump=res['bump']),
                   C=dict(sanity=sanity, skipped=skipped))
    with open(os.path.join(OUT_DIR, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps(dict(B_S=st['S']['status'], B_E=st['E']['status'], touch=res['met_S']['touch']['min'],
                          contact_dev_cm=res['contact_dev_m'] * 100.0, C_rot_frame=sanity['rot_frame_deg'])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
