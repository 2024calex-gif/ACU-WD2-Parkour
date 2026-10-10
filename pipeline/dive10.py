"""T22 (dive10): the dive9 hover with the Exit's events and its stand-up stretched by a delay D (8 and 15 frames).

Reuses dive9 (the SlideOver bake, the timing, the bump, the rate floor, the aux rules and the validation helpers).
dive9's own default behaviour is not changed: dive9 is imported and its functions are called, never modified.

For each delay D (frames at 60 Hz, clip time) the Exit changes in four ways:

  1. Event tracks (infotype 5). Every key with Frame >= EVENT_FROM (13: the Exit's played start is clip frame 12.48)
     moves D frames later. A frame past the last frame (round(Length * 60) = 52) is clamped to it, and the LAST value is
     kept per frame. Frame-0 keys stay. Each track is written with "Slot" (its index in the dump's Tracks array, which is
     the plugin's rawTracks index), the original TrackID and codec, and infotype 5.
  2. Stand-up in ENTITY Reference z (the quantity the spec measures). The Exit starts at the SlideOver's entity z at W
     (its contact lift included) with the SlideOver's entity vertical speed at W, and rises by a Hermite (C1 at W, zero
     slope at the end) to Z_UP = 0.80 m at played P_UP = 0.16 + D / (60 * EXIT_R), then holds it. The start speed is the
     SlideOver's own (about 3.2 m/s), so the peak speed is about that value. dive9 built the profile on the lift-free z
     and added its contact lift back on top; that lift rises by about 0.25 m after W, and the sum peaks at 6.45 m/s.
  3. Contact after W. The Exit lift (added to the stand-up) starts at 0 at W and is raised by one-sided updates: a body
     whose lowest point is under the ground is lifted by its depth (smoothed, ramped in from W). Each update trades speed
     for clearance, and the rule keeps the candidate with the least penetration whose peak Reference z speed is at most
     4 m/s on both the joined played grid and the clip frames. The 6 m/s pelvis rate floor is then applied after W, as in
     dive9. The lift up to W is dive9's, frozen, so the SlideOver is unchanged.
  4. Rotations and hold. The ease toward Nav_Run's first frame is smoothstep over played 0 .. P_EASE, where P_EASE is the
     new window end minus EASE_LEAD (0.05 s). The common weight is the smallest that puts every patched bone within
     GAP_AIM of Nav_Run at P_EASE. The new window end (played) is dive9's 0.228 + D / (60 * EXIT_R). After the window end
     the Exit holds its pose and its Reference position (the game blends out there), so nothing drifts to the clip end.

The SlideOver patch is dive9's file, copied as it is. Outputs (only under out\\dive10):
  delay08\\F7919BDF0.baked.json, delay08\\47BFB02DF.baked.json, delay08\\mapping_dive10.json   (and delay15\\...)
  summary.json, report.md
Run: python dive10.py   (about 5 minutes: dive9's solve for the frozen SlideOver lift, then the two Exit variants)
"""
import json
import os
import shutil
import sys
import zlib

import numpy as np

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
import dive_chain2 as DC2
import dumps

FPS = 60
DELAYS = (8, 15)
EVENT_FROM = 13                       # clip frame: keys at or after this frame move (the played start is 12.48)
EXIT_R = D4.EXIT_R                    # 0.9665 clip s per played s
EXIT_WIN = D4.EXIT_WIN                # dive9's played window end (0.228 s): clip frame 25.7
EASE_LEAD = 0.05                      # s (clip): the ease completes this long before the new window end
K_LIFT_E = 140                        # grid of the Exit lift (joined time, 60 Hz): covers the longest window (~1.69 s)
RAMP = 6                              # lift indices over which the Exit contact correction ramps in after W
CONTACT_STOP = 0.002                  # m: stop the Exit contact iteration when its residual is below this
EXIT_CONTACT_ITER = 6                # most updates of the Exit contact correction (one-sided: it only raises the body)
OUT_DIR = os.path.join(config.OUT, 'dive10')
DIR_FMT = 'delay%02d'
SO_HEX, SO_NAME = D9.SO_HEX, D9.SO_NAME
EXIT_HEX, EXIT_NAME = D9.EXIT_HEX, D9.EXIT_NAME
SO_PATCH_DIR9 = os.path.join(config.OUT, 'dive9', 'patches')
MAPPING_NAME = 'mapping_dive10.json'
REPORT = os.path.join(OUT_DIR, 'report.md')
SUMMARY = os.path.join(OUT_DIR, 'summary.json')


def override_for(delay):
    return 'preview: dive10 exit events +%d' % int(delay)


def out_dir_for(delay):
    return os.path.join(OUT_DIR, DIR_FMT % int(delay))


# ------------------------------------------------------------------------------------------------ timing of a variant
def p_up_for(delay):
    """Played time (s) at which the stand-up reaches Z_UP: dive9's 0.16 plus the delay in played seconds."""
    return D9.P_UP + float(delay) / (FPS * EXIT_R)


def p_end_for(delay):
    """Played time (s) of the new window end: dive9's window end plus the delay."""
    return EXIT_WIN + float(delay) / (FPS * EXIT_R)


def p_ease_for(delay):
    """Played time (s) at which the rotation ease completes: the new window end minus EASE_LEAD (clip seconds)."""
    return p_end_for(delay) - EASE_LEAD / EXIT_R


def last_frame(length):
    return int(round(float(length) * FPS))


def frame_of_played(p):
    return int(round((D4.EXIT_O + EXIT_R * float(p)) * FPS))


# ------------------------------------------------------------------------------------------------ event tracks
def event_tracks_of(dump):
    """(slot, track) for every infotype-5 track of a dump. The slot is the track's index in dump['Tracks']: DumpAnimation.inl
    writes one entry per rawTracks index, in index order, so this index is the plugin's slot."""
    return [(i, t) for i, t in enumerate(dump['Tracks']) if t.get('infotype') == 5]


def shift_event_keys(keys, delay, last, from_frame=EVENT_FROM):
    """Shifted integer event keys of one track.

    Keys with Frame >= from_frame move by `delay` frames; a frame past `last` is clamped to `last`. Where keys land on
    the same frame, the LAST value (the later source key) is kept. The result is strictly increasing in Frame.
    """
    frames = [int(k['Frame']) for k in keys]
    if any(b <= a for a, b in zip(frames, frames[1:])):
        raise ValueError('source event keys must be strictly increasing in Frame')
    by_frame = {}
    for k in keys:
        f = int(k['Frame'])
        if f >= from_frame:
            f = min(f + int(delay), int(last))
        by_frame[f] = int(k['Int'])
    return [{'Frame': f, 'Int': by_frame[f]} for f in sorted(by_frame)]


def event_patch_tracks(dump, delay):
    """The shifted event tracks of a dump as patch tracks: TrackID and codec of the original, infotype 5, and "Slot"."""
    last = last_frame(dump['Length'])
    out = []
    for slot, t in event_tracks_of(dump):
        out.append({'TrackID': str(int(t['TrackID'])), 'codec': int(t['codec']), 'infotype': 5, 'Slot': int(slot),
                    'Keyframes': shift_event_keys(t['Keyframes'], delay, last)})
    return out


def event_table(dump, delay):
    """Before and after per event track: slot, TrackID, codec and the keys (frame, value)."""
    last = last_frame(dump['Length'])
    rows = []
    for slot, t in event_tracks_of(dump):
        before = [(int(k['Frame']), int(k['Int'])) for k in t['Keyframes']]
        after = [(k['Frame'], k['Int']) for k in shift_event_keys(t['Keyframes'], delay, last)]
        rows.append(dict(slot=slot, TrackID=str(int(t['TrackID'])), codec=int(t['codec']), before=before, after=after))
    return rows


# ------------------------------------------------------------------------------------------------ one bake
def gap_at(ctx, final, p, bones):
    return D9.gap_to_nav(ctx, final, frame_of_played(p), bones)


def bake10(ctx, case, lift_S, lift_E, delay):
    """dive9.bake9 with the stand-up in entity z, the ease timed to P_EASE, and the window to the new end.

    lift_S: the lift for the SlideOver (dive9's, frozen). lift_E: the lift added to the Exit's stand-up (zero at W).
    ctx.span_e must already be the new window (EXIT_R * p_end); the caller sets and restores it.
    """
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
    p = (tE - D4.EXIT_O) / D4.EXIT_R                       # played seconds into the Exit (p = 0 at W)
    tau_E = ctx.t_W + p
    p_up = p_up_for(delay)
    p_end = p_end_for(delay)
    p_ease = p_ease_for(delay)
    # ---- the stand-up in entity z: from the SlideOver's entity height and speed at W (its lift included)
    zS = fin_S['ref_out'][:, 2]
    z_e0 = float(zS[last])
    v_e0_clip = float((zS[last] - zS[last - 1]) * FPS)                  # m/s (clip time) at W, SlideOver's own
    v_e0_played = v_e0_clip * EXIT_R                                     # the profile slope is per played second
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
    # ---- rotations: common ease weight toward Nav_Run's first frame, completing at p_ease
    patched_gap = [bn for bn in BD.GAP_BONES if BD._acu(ctx.rig, bn) in set(seg_E['patset'])]
    gap_pre = gap_at(ctx, fin_E['final'], p_ease, patched_gap)
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
    fin_E['final'] = final_E
    # the clip holds its pose and its Reference position from the new window end to the clip end (the game blends out
    # at the window end; nothing after it may drift): the keys after the hold frame are replaced by the hold values
    nf = int(seg_E['F'][-1])
    hold = min(frame_of_played(p_end), nf)
    if hold < nf:
        final_E[hold + 1:] = final_E[hold]
        ref_out_E[hold + 1:] = ref_out_E[hold]
        fin_E['ref_out'] = ref_out_E
        held = [round(float(x), 9) for x in ref_out_E[hold]]
        keys = [k for k in fin_E['trans_keys'] if int(k['Frame']) < hold]
        keys.append({'Frame': int(hold), 'LocalPosition': held})
        keys.append({'Frame': int(nf), 'LocalPosition': held})
        fin_E['trans_keys'] = keys
    gap_post_ease = gap_at(ctx, final_E, p_ease, patched_gap)
    gap_post_end = gap_at(ctx, final_E, p_end, BD.GAP_BONES)
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
                delay=int(delay), p_E=p, inside=inside, c_E=c)


# ------------------------------------------------------------------------------------------------ contact and floor
def contact_parts(ctx, res, case):
    """Worst contact deviation (m) of the SlideOver and of the Exit, and the Exit's 95th percentile (m), as dive8 defines
    the contact samples (lowest non-finger point on the box for the SlideOver, lowest point for the Exit)."""
    dev = {'S': [], 'E': []}
    for s in D8.contact_samples8(ctx, res, case):
        dev[s['tag']].append(np.abs(s['surf'] + S6.CONTACT_TARGET - s['lowest'])[s['contact']])
    worst = {k: float(np.max(np.concatenate(v))) for k, v in dev.items()}
    p95E = float(np.percentile(np.concatenate(dev['E']), 95))
    return worst, p95E


def rise_speed_joined(ctx, res):
    """Peak Reference z speed (m/s) after W on the joined played grid, as dive9 measures it."""
    n0, zr = D9.joined_ref_z(ctx, res)
    tau_r = (np.arange(len(zr)) + n0) / FPS
    dzr = np.diff(zr) * FPS
    post = np.where(tau_r[1:] > ctx.t_W + 1e-9)[0]
    return float(np.max(np.abs(dzr[post]))) if post.size else float('nan')


def rise_speed_clip(res):
    """Peak Reference z speed (m/s per clip second) over the patched Exit frames, the clip frames as the Exit plays them."""
    zE = res['fin_E']['ref_out'][:, 2]
    dze = np.diff(zE) * FPS
    fe = np.arange(int(res['win_E'][0]), len(dze))
    return float(np.max(np.abs(dze[fe])))


def penetration(res):
    """The worst non-finger penetration (cm) of the variant's two segments (dive9's rule, as in the judge)."""
    return float(max(res['met_S']['pen_patch_cm'], res['met_E']['pen_patch_cm']))


def solve10(ctx, case, delay, lift0, iterate_exit=True, max_iter=EXIT_CONTACT_ITER):
    """The variant's lift. Up to W it is dive9's (lift0, frozen). After W, lift_E starts at 0 and is raised by one-sided
    updates (a body below the ground is lifted by its depth, smoothed and ramped in from W). Each update trades speed for
    clearance: the rule keeps the candidate with the least penetration whose peak Reference z speed is at most
    RISE_V_MAX on BOTH measures (the joined played grid, as dive9, and the clip frames). The first candidate (lift_E = 0)
    is always one of them if it is feasible. The rate floor is applied to lift_E after W.
    The bump must already be set (F5.BUMP). ctx.span_e must already be the new window."""
    k_W = int(np.floor(ctx.t_W * FPS + 1e-9))                            # the last SlideOver lift index
    ramp = np.clip((np.arange(K_LIFT_E) - k_W) / float(RAMP), 0.0, 1.0)
    lift_S = np.array(lift0, float)
    lift_E = np.zeros(K_LIFT_E)
    res = bake10(ctx, case, lift_S, lift_E, delay)
    cands, hist = [], []
    if iterate_exit:
        for it in range(max_iter + 1):
            spd, pen = rise_speed_joined(ctx, res), penetration(res)
            spd_clip = rise_speed_clip(res)
            cands.append(dict(it=it, lift=lift_E.copy(), res=res, speed=max(spd, spd_clip), joined=spd, clip=spd_clip, pen=pen))
            samples = [s for s in D8.contact_samples8(ctx, res, case) if s['tag'] == 'E']
            r, dev = S6.residual_from_samples(samples, K_LIFT_E)
            r = np.maximum(r, 0.0)                                       # one-sided: only raise a body under the ground
            r_stop = float(np.max(np.abs(r * (ramp >= 1.0))))
            hist.append(dict(it=it, rise_mps=spd, rise_clip_mps=spd_clip, pen_cm=pen, resid_up_cm=r_stop * 100.0,
                             lift_E_max_cm=float(lift_E.max()) * 100.0))
            if it == max_iter or r_stop < CONTACT_STOP:
                break
            lift_E = bake2._gauss_smooth(lift_E + r * ramp, S6.CONTACT_SIGMA_LIFT) * ramp
            res = bake10(ctx, case, lift_S, lift_E, delay)
        feasible = [c for c in cands if c['speed'] <= D9.RISE_V_MAX]
        best = min(feasible, key=lambda c: (c['pen'], c['it'])) if feasible else min(cands, key=lambda c: c['speed'])
        lift_E = best['lift']
    else:
        cands = [dict(it=0, lift=lift_E.copy(), res=res, speed=rise_speed_joined(ctx, res), joined=rise_speed_joined(ctx, res),
                      clip=rise_speed_clip(res), pen=penetration(res))]
        best = cands[0]
    # the rate floor on the Exit's part of the joined pelvis path (causal; the SlideOver part is dive9's, already floored)
    n0, Zj = D9.joined_pelvis_z(ctx, best['res'])
    idx = np.arange(n0, n0 + len(Zj))
    post = np.where(idx > k_W)[0]
    Lj = lift_E[idx]
    zbase = Zj - Lj
    lift_E_f = lift_E.copy()
    if post.size:
        lift_E_f[idx[post]] = D9.rate_floor(Lj[post], zbase[post], D9.PELVIS_FLOOR_V)
    res = bake10(ctx, case, lift_S, lift_E_f, delay)
    worst, p95E = contact_parts(ctx, res, case)
    res['contact_hist'] = hist
    res['candidates'] = [dict(it=c['it'], speed=c['speed'], joined=c['joined'], clip=c['clip'], pen=c['pen']) for c in cands]
    res['chosen_it'] = int(best['it'])
    res['contact_dev_S_m'] = worst['S']
    res['contact_dev_E_m'] = worst['E']
    res['contact_p95_E_cm'] = p95E * 100.0
    res['rate_add_E_max_cm'] = float(np.max(lift_E_f - lift_E)) * 100.0
    res['lift_E_max_cm'] = float(lift_E_f.max()) * 100.0
    res['lift_E_min_cm'] = float(lift_E_f.min()) * 100.0
    res['lift_S_frozen'] = bool(np.array_equal(lift_S, np.array(lift0, float)))
    return res


# ------------------------------------------------------------------------------------------------ metrics
def metrics10(ctx, res):
    """Exit profile, speeds, uprightness over the patched Exit, the Nav gap at the window end, the feet and the pelvis."""
    chain = D9.chain_idx(ctx)
    P_S = D3._world(res['seg_S']['seq'], res['seg_S']['dump'], res['fin_S']['final'], res['fin_S']['ref_out'],
                    res['fin_S']['disp'])
    P_E = D3._world(res['seg_E']['seq'], res['seg_E']['dump'], res['fin_E']['final'], res['fin_E']['ref_out'],
                    res['fin_E']['disp'])
    W = int(res['win_S'][-1])
    tilt_S, inv_S, _ = D9.upright(P_S, chain)
    tilt_E, inv_E, up_E = D9.upright(P_E, chain)
    eframes = np.arange(int(res['win_E'][0]), len(tilt_E))                # every patched Exit frame to the clip end
    out = dict(tilt_W=float(tilt_S[W]), inv_W=bool(inv_S[W]),
               tilt_exit_max=float(tilt_E[eframes].max()), inv_exit=bool(inv_E[eframes].any()),
               up_exit=bool(up_E[eframes].all()), exit_frames=[int(eframes[0]), int(eframes[-1])])
    zE = res['fin_E']['ref_out'][:, 2]
    for pz in (0.10, 0.16, 0.22, 0.30, 0.38):
        out['z_at_%.2f' % pz] = float(np.interp((D4.EXIT_O + EXIT_R * pz) * FPS, np.arange(len(zE)), zE))
    out['z_at_W'] = float(res['z0'])
    out['p_up'] = float(res['p_up'])
    out['z_at_p_up'] = float(np.interp((D4.EXIT_O + EXIT_R * res['p_up']) * FPS, np.arange(len(zE)), zE))
    out['p_end'] = float(res['p_end'])
    out['window_end_frame'] = frame_of_played(res['p_end'])
    out['p_ease'] = float(res['p_ease'])
    out['ease_end_clip_s'] = float(D4.EXIT_O + EXIT_R * res['p_ease'])
    out['z_window_end'] = float(np.interp((D4.EXIT_O + EXIT_R * res['p_end']) * FPS, np.arange(len(zE)), zE))
    out['z_clip_end'] = float(zE[-1])
    # rise speeds: on the joined played grid (as dive9) and on the Exit clip alone
    n0, zr = D9.joined_ref_z(ctx, res)
    tau_r = (np.arange(len(zr)) + n0) / FPS
    dzr = np.diff(zr) * FPS
    post = np.where(tau_r[1:] > ctx.t_W + 1e-9)[0]
    out['rise_speed_joined_post_W_mps'] = float(np.max(np.abs(dzr[post]))) if post.size else float('nan')
    out['rise_speed_joined_at_s'] = float(tau_r[1:][post][np.argmax(np.abs(dzr[post]))]) if post.size else float('nan')
    dze = np.diff(zE) * FPS
    fe = np.arange(int(res['win_E'][0]), len(dze))
    out['rise_speed_exit_clip_mps'] = float(np.max(np.abs(dze[fe])))
    out['v_start_clip'] = float(res['v_start_clip'])
    # the pose gap to Nav_Run
    pg = list(res['patched_gap'])
    out['patched_gap'] = pg
    out['gap_pre_ease_max'] = float(max(res['gap_pre_ease'][b] for b in pg))
    out['gap_at_ease_end_max'] = float(max(res['gap_post_ease'][b] for b in pg))
    out['gap_at_window_end'] = {b: float(v) for b, v in res['gap_post_end'].items()}
    out['gap_at_window_end_max_patched'] = float(max(res['gap_post_end'][b] for b in pg))
    p_clip_end = (len(zE) - 1 - D4.EXIT_O * FPS) / (FPS * EXIT_R)           # played time of the last clip frame
    out['gap_at_clip_end_max_patched'] = float(max(gap_at(ctx, res['fin_E']['final'], p_clip_end, pg).values()))
    out['w_common'] = float(res['w_common'])
    # feet down (both, upright) and the feet_free_to frame, as dive9
    fl = [ctx.rig.ac_by_hash[zlib.crc32(n.encode())] for n in ('LeftFoot', 'RightFoot')]
    thr = D9.foot_rest_z(ctx, res) + D9.DOWN_MARGIN
    out['foot_down_thr_m'] = float(thr)
    p_down = None
    for p in np.arange(0.0, 0.30 + 1e-9, 1.0 / FPS):
        f = frame_of_played(p)
        if bool(np.all(P_E[f, fl, 2] <= thr) and tilt_E[f] <= D9.TILT_MAX):
            p_down = float(p)
            break
    out['p_feet_down_upright'] = p_down
    if p_down is None:
        out['feet_free_to_frame'] = int(np.ceil((D4.EXIT_O + EXIT_R * 0.30) * FPS))
    else:
        out['feet_free_to_frame'] = int(np.ceil((D4.EXIT_O + EXIT_R * p_down) * FPS - 1e-9))
    # the world pelvis fall, the joined world metrics, the contact and the lift
    n0w, zw = D9.joined_pelvis_z(ctx, res)
    dzw = np.diff(zw) * FPS
    out['pelvis_drop_max_mps'] = float(-np.min(dzw)) if dzw.size else float('nan')
    w = res['world']
    out['world_pelvis_step_cm'] = float(w['pelvis']['step_cm'])
    out['world_pelvis_dv_mps'] = float(w['pelvis']['dv_ms'])
    out['world_head_step_cm'] = float(w['head']['step_cm'])
    out['world_head_dv_mps'] = float(w['head']['dv_ms'])
    out['offset_max_m'] = float(w['offset_max_m'])
    out['cont_deg'] = float(res['cont']['max_deg'])
    out['cont_worst'] = str(res['cont']['worst'])
    out['contact_dev_S_cm'] = float(res['contact_dev_S_m'] * 100.0)
    out['contact_dev_E_cm'] = float(res['contact_dev_E_m'] * 100.0)
    out['contact_p95_E_cm'] = float(res['contact_p95_E_cm'])
    out['rate_add_E_max_cm'] = float(res['rate_add_E_max_cm'])
    out['lift_E_max_cm'] = float(res['lift_E_max_cm'])
    out['lift_E_min_cm'] = float(res['lift_E_min_cm'])
    out['lift_S_frozen'] = bool(res['lift_S_frozen'])
    out['pen_patch_cm'] = float(max(res['met_S']['pen_patch_cm'], res['met_E']['pen_patch_cm']))
    out['touch_min'] = int(res['met_S']['touch']['min'])
    out['contact_iters'] = len(res.get('contact_hist', []))
    return out


def judge10(m):
    """The checks of a variant: dive9's rules where they still apply, plus the spec's stand-up and gap targets."""
    fails = []
    if m['pen_patch_cm'] > D9.PEN_MAX:
        fails.append('non-finger penetration %.1f cm (> %.0f)' % (m['pen_patch_cm'], D9.PEN_MAX))
    if m['touch_min'] < D9.TOUCH_MIN:
        fails.append('hand touch %d frames (need %d)' % (m['touch_min'], D9.TOUCH_MIN))
    if m['world_pelvis_dv_mps'] > D9.DV_MAX or m['world_head_dv_mps'] > D9.DV_MAX:
        fails.append('world dv pelvis %.2f / head %.2f m/s per frame (> %.1f)' % (
            m['world_pelvis_dv_mps'], m['world_head_dv_mps'], D9.DV_MAX))
    if m['world_pelvis_step_cm'] > D9.STEP_MAX or m['world_head_step_cm'] > D9.STEP_MAX:
        fails.append('world step pelvis %.1f / head %.1f cm (> %.0f)' % (
            m['world_pelvis_step_cm'], m['world_head_step_cm'], D9.STEP_MAX))
    if m['offset_max_m'] > D9.OFF_MAX:
        fails.append('Reference offset %.2f m (> %.1f)' % (m['offset_max_m'], D9.OFF_MAX))
    if m['cont_deg'] > D9.CONT_MAX:
        fails.append('continuity %.0f deg (> %.0f)' % (m['cont_deg'], D9.CONT_MAX))
    if m['tilt_W'] > D9.TILT_MAX or m['inv_W']:
        fails.append('upright at W: tilt %.0f deg, inverted %s' % (m['tilt_W'], m['inv_W']))
    if m['tilt_exit_max'] > D9.TILT_MAX or m['inv_exit']:
        fails.append('upright in the Exit: max tilt %.0f deg, inverted %s' % (m['tilt_exit_max'], m['inv_exit']))
    if m['z_at_p_up'] < D9.Z_UP_MIN:
        fails.append('Exit Reference z at P_UP %.3f m (< %.2f)' % (m['z_at_p_up'], D9.Z_UP_MIN))
    if m['rise_speed_joined_post_W_mps'] > D9.RISE_V_MAX:
        fails.append('stand-up speed %.2f m/s (> %.0f)' % (m['rise_speed_joined_post_W_mps'], D9.RISE_V_MAX))
    if m['gap_at_window_end_max_patched'] > D9.GAP_GOAL:
        fails.append('gap to Nav_Run at the window end %.1f deg (> %.0f)' % (m['gap_at_window_end_max_patched'], D9.GAP_GOAL))
    if m['pelvis_drop_max_mps'] > D9.PELVIS_V_MAX + 1e-6:
        fails.append('world pelvis drop %.2f m/s (> %.0f)' % (m['pelvis_drop_max_mps'], D9.PELVIS_V_MAX))
    return fails


# ------------------------------------------------------------------------------------------------ outputs
def write_variant(ctx, res, delay, feet_to, outdir):
    """The variant's folder: the SlideOver copy (dive9's file as it is), the Exit patch (body + aux + shifted events) and
    the mapping (dive9 schema, with this variant's override)."""
    os.makedirs(outdir, exist_ok=True)
    shutil.copyfile(os.path.join(SO_PATCH_DIR9, SO_HEX + '.baked.json'), os.path.join(outdir, SO_HEX + '.baked.json'))
    ex = D8.patch_dict(ctx.rig, res['fin_E']['rot_tracks'], res['fin_E']['trans_keys'], ctx.ex_dump['Length'])
    ex_aux = AT.add_aux_tracks(ex, ctx.ex_dump, 0, feet_to)
    ex_aux['Tracks'] = list(ex_aux['Tracks']) + event_patch_tracks(ctx.ex_dump, delay)
    AT.write_patch(os.path.join(outdir, EXIT_HEX + '.baked.json'), ex_aux)
    ov = override_for(delay)
    m = dict(clips=[
        DC2.mapping_entry(SO_HEX, SO_NAME, SO_HEX + '.baked.json', True, override=ov),
        DC2.mapping_entry(EXIT_HEX, EXIT_NAME, EXIT_HEX + '.baked.json', True, only_after=[SO_NAME], override=ov),
    ])
    with open(os.path.join(outdir, MAPPING_NAME), 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=1)
    return ex_aux


def baseline_metrics(ctx, res9, case):
    """dive9's hover result, shaped like a dive10 result, so that metrics10 reports it the same way."""
    p_end = EXIT_WIN
    gap_end = gap_at(ctx, res9['fin_E']['final'], p_end, BD.GAP_BONES)
    r = dict(res9)
    r.update(p_up=D9.P_UP, p_end=p_end, p_ease=D9.EASE_T, gap_pre_ease=res9['gap_pre'], gap_post_ease=res9['gap_post'],
             gap_post_end=gap_end, v_start_clip=float(res9['v0'] / EXIT_R), w_common=res9['w_common'],
             contact_hist=res9['contact_hist'], lift_S_frozen=True)
    worst, p95E = contact_parts(ctx, res9, case)
    r.update(contact_dev_S_m=worst['S'], contact_dev_E_m=worst['E'], contact_p95_E_cm=p95E * 100.0,
             rate_add_E_max_cm=0.0, lift_E_max_cm=float(res9['lift_final'].max()) * 100.0,
             lift_E_min_cm=float(res9['lift_final'].min()) * 100.0)
    m = metrics10(ctx, r)
    m['feet_free_to_frame'] = int(D9.metrics9(ctx, res9)['feet_free_to_frame'])
    m['gap_at_window_end'] = {b: float(v) for b, v in gap_end.items()}
    m['patched_gap'] = list(res9['patched_gap'])
    return m


def run_variant(ctx, case, delay, lift0, iterate_exit=True):
    """solve10 with the window of this delay (ctx.span_e is set for the bake and restored after)."""
    saved = ctx.span_e
    ctx.span_e = D4.EXIT_R * p_end_for(delay)
    try:
        res = solve10(ctx, case, delay, lift0, iterate_exit=iterate_exit)
        m = metrics10(ctx, res)
    finally:
        ctx.span_e = saved
    return res, m


def write_report(variants, base, events, fails_all, ctx):
    """report.md: the event tables, the Exit profiles, the validation per variant, the paths."""
    L = []
    L.append('# T22 dive10: the dive9 hover with the Exit events and the stand-up stretched by D = 8 and 15 frames')
    L.append('')
    L.append('Generated by `v2/pipeline/dive10.py` (reuses dive9 and its helpers). Written only under `out\\dive10`. '
             'The SlideOver patch (`47BFB02DF.baked.json`) is dive9\'s, copied as it is into each folder.')
    L.append('')
    L.append('## 1. What changes')
    L.append('')
    L.append('* Event tracks (infotype 5): each key at Frame >= %d moves +D frames. A frame past the last frame (%d) is clamped '
             'to it and the last value per frame is kept. Written with "Slot" (index in the dump\'s Tracks array), the dump\'s '
             'TrackID and codec.' % (EVENT_FROM, last_frame(ctx.ex_dump['Length'])))
    L.append('* Stand-up in entity Reference z: from the SlideOver\'s entity z at W (%.3f m) with its entity speed at W '
             '(%.2f m/s, clip time), a Hermite to %.2f m at P_UP = 0.16 + D / (60 * %.4f), then held. dive9 built its profile on '
             'the lift-free z (%.2f m/s at W, clip time) and added its contact lift on top; that lift rises by about 0.25 m after '
             'W, and the sum peaks at 6.45 m/s. dive10 keeps the stand-up in entity z (no lift in the profile) and adds only the '
             'Exit contact lift, from 0 at W.'
             % (variants[8]['z_at_W'], variants[8]['v_start_clip'], D9.Z_UP, EXIT_R, base['v_start_clip']))
    L.append('* Ease toward Nav_Run\'s first frame: smoothstep over played 0 .. P_EASE, P_EASE = the new window end minus %.2f s.'
             % EASE_LEAD)
    L.append('* SlideOver: dive9\'s patch, copied as it is. Its contact lift up to W is frozen; the Exit lift after W starts at 0.')
    L.append('')
    L.append('Timing (played s; clip time = %.3f + %.4f * played):' % (D4.EXIT_O, EXIT_R))
    L.append('')
    L.append('| variant | D (frames) | P_UP | window end (played / clip frame) | P_EASE (played / clip s) |')
    L.append('|---|---|---|---|---|')
    L.append('| dive9 hover | 0 | %.4f | %.4f / %.2f | %.4f / %.3f |'
             % (D9.P_UP, EXIT_WIN, (D4.EXIT_O + EXIT_R * EXIT_WIN) * FPS, D9.EASE_T, D4.EXIT_O + EXIT_R * D9.EASE_T))
    for d in DELAYS:
        pe = p_end_for(d)
        pz = p_ease_for(d)
        L.append('| delay%02d | %d | %.4f | %.4f / %.2f | %.4f / %.3f |' % (
            d, d, p_up_for(d), pe, (D4.EXIT_O + EXIT_R * pe) * FPS, pz, D4.EXIT_O + EXIT_R * pz))
    L.append('')
    L.append('## 2. Event tracks (before and after, per slot)')
    L.append('')
    for d in DELAYS:
        L.append('### D = %d frames (delay%02d)' % (d, d))
        L.append('')
        L.append('| slot | TrackID | codec | keys before (frame:value) | keys after (frame:value) |')
        L.append('|---|---|---|---|---|')
        for r in events[d]:
            b = ', '.join('%d:%d' % kv for kv in r['before'])
            a = ', '.join('%d:%d' % kv for kv in r['after'])
            L.append('| %d | %s | %d | %s | %s |' % (r['slot'], r['TrackID'], r['codec'], b, a))
        L.append('')
    L.append('## 3. Exit profiles (entity Reference z, m)')
    L.append('')
    L.append('| played (s) | dive9 hover | delay08 | delay15 |')
    L.append('|---|---|---|---|')
    for pz in (0.10, 0.16, 0.22, 0.30, 0.38):
        key = 'z_at_%.2f' % pz
        L.append('| %.2f | %.3f | %.3f | %.3f |' % (pz, base[key], variants[8][key], variants[15][key]))
    L.append('')
    L.append('Speeds and the start (m/s, clip time):')
    L.append('')
    L.append('| metric | dive9 hover | delay08 | delay15 | rule |')
    L.append('|---|---|---|---|---|')
    for key, lbl, rule in (('rise_speed_joined_post_W_mps', 'peak Reference z speed after W (joined played grid, real time)', '<= 4'),
                           ('rise_speed_exit_clip_mps', 'peak Reference z speed in the Exit clip frames (per clip second)', '<= 4'),
                           ('v_start_clip', 'entity speed at W (start of the stand-up)', 'info'),
                           ('pelvis_drop_max_mps', 'world pelvis, peak fall (m/s, joined)', '<= 6')):
        L.append('| %s | %.2f | %.2f | %.2f | %s |' % (lbl, base[key], variants[8][key], variants[15][key], rule))
    L.append('')
    L.append('## 4. Validation per variant')
    L.append('')
    L.append('| metric | dive9 hover | delay08 | delay15 | rule |')
    L.append('|---|---|---|---|---|')
    rows = [
        ('Nav gap at the window end, max over patched bones (deg)', 'gap_at_window_end_max_patched', '<= 25'),
        ('Nav gap at the ease end, max over patched bones (deg)', 'gap_at_ease_end_max', 'info'),
        ('Nav gap at the clip end, max over patched bones (deg)', 'gap_at_clip_end_max_patched', 'info'),
        ('tilt at W (deg)', 'tilt_W', '<= 60'),
        ('max tilt in the Exit, every patched frame (deg)', 'tilt_exit_max', '<= 60'),
        ('inverted in the Exit (any patched frame)', 'inv_exit', 'no'),
        ('inverted at W', 'inv_W', 'no'),
        ('feet free until (clip frame)', 'feet_free_to_frame', 'info'),
        ('world pelvis fall, peak (m/s)', 'pelvis_drop_max_mps', '<= 6'),
        ('world pelvis dv / head dv (m/s per frame)', 'DV', '<= 1.5'),
        ('world pelvis / head step (cm per frame)', 'STEP', '<= 12'),
        ('Reference offset max (m)', 'offset_max_m', '<= 1.0'),
        ('continuity at W (deg)', 'cont_deg', '<= 45'),
        ('contact dev, Exit (cm, worst)', 'contact_dev_E_cm', 'info'),
        ('contact dev, SlideOver (cm, worst)', 'contact_dev_S_cm', 'info'),
        ('non-finger penetration (cm)', 'pen_patch_cm', '<= 5'),
    ]
    for lbl, key, rule in rows:
        cells = []
        for x in (base, variants[8], variants[15]):
            if key == 'DV':
                cells.append('%.2f / %.2f' % (x['world_pelvis_dv_mps'], x['world_head_dv_mps']))
            elif key == 'STEP':
                cells.append('%.1f / %.1f' % (x['world_pelvis_step_cm'], x['world_head_step_cm']))
            else:
                v = x[key]
                if isinstance(v, bool):
                    cells.append('yes' if v else 'no')
                elif isinstance(v, float):
                    cells.append('%.2f' % v)
                else:
                    cells.append(str(v))
        L.append('| %s | %s | %s | %s | %s |' % (lbl, cells[0], cells[1], cells[2], rule))
    L.append('')
    L.append('Gap to Nav_Run at the window end, per patched bone (deg):')
    L.append('')
    L.append('| bone | dive9 hover | delay08 | delay15 |')
    L.append('|---|---|---|---|')
    for bn in variants[8]['patched_gap']:
        L.append('| %s | %.1f | %.1f | %.1f |' % (bn, base['gap_at_window_end'].get(bn, float('nan')),
                                                 variants[8]['gap_at_window_end'][bn], variants[15]['gap_at_window_end'][bn]))
    L.append('')
    L.append('Exit lift after W (contact correction; the SlideOver lift is dive9\'s and frozen): delay08 %.1f .. %.1f cm, '
             'delay15 %.1f .. %.1f cm; contact iterations delay08 %d, delay15 %d; SlideOver lift frozen: %s.'
             % (variants[8]['lift_E_min_cm'], variants[8]['lift_E_max_cm'], variants[15]['lift_E_min_cm'],
                variants[15]['lift_E_max_cm'], variants[8]['contact_iters'], variants[15]['contact_iters'],
                variants[8]['lift_S_frozen'] and variants[15]['lift_S_frozen']))
    L.append('')
    L.append('Exit contact updates (one-sided raise of the Exit lift). The rule keeps the least penetration whose peak rise is '
             '<= %.0f m/s on both measures.' % D9.RISE_V_MAX)
    L.append('')
    L.append('| delay | update | peak rise, joined (m/s) | peak rise, clip frames (m/s) | penetration (cm) | kept |')
    L.append('|---|---|---|---|---|---|')
    for d in DELAYS:
        for c in variants[d]['candidates']:
            L.append('| delay%02d | %d | %.2f | %.2f | %.2f | %s |' % (d, c['it'], c['joined'], c['clip'], c['pen'],
                                                                    'yes' if c['it'] == variants[d]['chosen_it'] else ''))
    L.append('')
    L.append('Spec status per variant:')
    L.append('')
    for d in DELAYS:
        L.append('* delay%02d: %s' % (d, '; '.join(fails_all[d]) if fails_all[d] else 'all dive10 checks pass'))
    L.append('')
    L.append('## 5. Paths')
    L.append('')
    for d in DELAYS:
        L.append('* `out\\dive10\\%s\\F7919BDF0.baked.json`, `out\\dive10\\%s\\47BFB02DF.baked.json`, `out\\dive10\\%s\\%s`'
                 % (DIR_FMT % d, DIR_FMT % d, DIR_FMT % d, MAPPING_NAME))
    L.append('* `out\\dive10\\summary.json`, `out\\dive10\\report.md`')
    L.append('')
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


# ------------------------------------------------------------------------------------------------ main
def run_all(ctx=None, case=None, verbose=True):
    ctx = F5.make_ctx() if ctx is None else ctx
    case = D9.make_case9() if case is None else case
    A, t0, t1 = D8.bump_for(ctx, case)
    F5.BUMP.update(t0=t0, t1=t1, A=A)
    res9 = D9.solve9(ctx, case, A, t0, t1)                  # dive9's own solve: the SlideOver's lift to freeze
    F5.BUMP.update(t0=t0, t1=t1, A=A)
    base = baseline_metrics(ctx, res9, case)
    variants, fails_all, events = {}, {}, {}
    for d in DELAYS:
        F5.BUMP.update(t0=t0, t1=t1, A=A)
        res, m = run_variant(ctx, D9.make_case9(), d, res9['lift_final'], iterate_exit=True)
        m['gap_at_window_end'] = {b: float(v) for b, v in res['gap_post_end'].items()}
        m['patched_gap'] = list(res['patched_gap'])
        m['candidates'] = res['candidates']
        m['chosen_it'] = res['chosen_it']
        variants[d] = m
        fails_all[d] = judge10(m)
        events[d] = event_table(ctx.ex_dump, d)
        write_variant(ctx, res, d, m['feet_free_to_frame'], out_dir_for(d))
        if verbose:
            print(json.dumps(dict(delay=d, fails=fails_all[d],
                                  z=[round(m['z_at_%.2f' % z], 4) for z in (0.10, 0.16, 0.22, 0.30, 0.38)],
                                  rise=round(m['rise_speed_joined_post_W_mps'], 3),
                                  gap_end=round(m['gap_at_window_end_max_patched'], 2),
                                  drop=round(m['pelvis_drop_max_mps'], 3), contact_E=round(m['contact_dev_E_cm'], 2),
                                  feet_to=m['feet_free_to_frame'], iters=m['contact_iters']), default=float))
    return dict(ctx=ctx, base=base, variants=variants, fails=fails_all, events=events)


def main(argv=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = run_all()
    with open(SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(dict(base=out['base'], variants={str(k): v for k, v in out['variants'].items()},
                       fails={str(k): v for k, v in out['fails'].items()},
                       ease_lead_s=EASE_LEAD, event_from=EVENT_FROM,
                       override={str(d): override_for(d) for d in DELAYS}),
                  f, indent=1, default=float)
    write_report(out['variants'], out['base'], out['events'], out['fails'], out['ctx'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
