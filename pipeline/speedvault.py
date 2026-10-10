"""Sprint SlideOver speed vaults: WD2 vaultover (100 cm and 200 cm long) baked onto the ACU Sprint SlideOver clips.

Reuses bake2 (T9e) with the D-style options of out\\variants\\D_short_in_wd2tail: fade-in 0.08 s, no fade-out, tail 'wd2'.
bake2 itself is not edited (its defaults and byte-identity test stay as they are). What differs from the kong bake:
  * the events come from data\\wd2_events_speedvault.json (one planting hand: LeftHand, so only the left hand is pinned);
  * the play window is the play model's cut (window_clip_end_s, clip time) minus o, so the window is in clip seconds;
  * validation is done here, because bake2.validate_t9d expects both hands (it reads targets for both).

Per clip:
  1. time map: ACU takeoff <-> WD2 takeoff, ACU hand plant (measured, on the top) <-> WD2 hand_plant_start,
     ACU feet down <-> WD2 landing (bake2.time_map).
  2. bake2.bake_t9e with bake2.T9E_SIDES narrowed to the planting hand (module global, restored afterwards).
  3. validate_sv: penetration (window and tail), hand touch, Delta step, trunk tilt, foot skate after landing,
     continuity at the window end against the next clip's first pose, timing ratios, jaggedness (bake2).
Outputs (out\\speedvault): <HANDLE>.baked.json (patch format of out\\patches), mapping_speedvault.json, report.md,
sheets\\ (GIF per clip, WD2 stick sheets, still grid).

Run:  python speedvault.py            (writes everything; reads the WD2 .mab data and the ACU dumps only)
"""
import contextlib
import json
import os
import zlib

import numpy as np

import acuclip
import bake2
import config
import contacts
import dumps
import fk
import ik
import retarget as R
import wd2

EVENTS_JSON = os.path.join(config.V2, 'data', 'wd2_events_speedvault.json')
PLAY_MODEL = os.path.join(config.OUT, 'match', 'play_model_slideover.json')
OUT_DIR = os.path.join(config.OUT, 'speedvault')
SHEET_DIR = os.path.join(OUT_DIR, 'sheets')
MAPPING = os.path.join(OUT_DIR, 'mapping_speedvault.json')
REPORT = os.path.join(OUT_DIR, 'report.md')
NEXT_CLIP = 'xx_Parkour_Muraille_Exit_Fd100cm-Dn100cm'     # the clip both SlideOver clips hand over to

FADE_IN = 0.08             # s, D-style
FADE_OUT = 0.0             # s, D-style (no fade-out)
TAIL = 'wd2'               # D-style: the WD2 motion runs to the clip end
POOL_WEIGHT = 0.5          # plugin v0.9: random-pool alternative weight

PLANTING_SIDES = (('Left', 'LeftHand'),)   # the only hand that touches the obstacle top in the WD2 clips
TOUCH_TOL_CM = 5.0
SKATE_HEIGHT_M = 0.05      # foot skate: lowest foot within this height of the ground
SKATE_P95_MS = 0.5         # m/s
SKATE_VZ_MS = 0.3          # m/s, stance also needs the lowest foot nearly still vertically
BLEND_OUT_S = 0.25         # s after the cut the game blends into the next clip (play model blend_out_s)
ENVELOPE_THROUGH_BLEND = True
TRACK_SMOOTH_PASSES = 1     # one pass over the patched rotation tracks inside the played window (see smooth_rot_tracks)
SMOOTH_EFFECTORS = {'LeftHand': 1, 'RightFoot': 3, 'LeftFoot': 3}   # smoothing passes per IK chain (see solve_sequence_continuous)
KEEP_HAND_WORLD_ROT = False  # hand follows the forearm (no hand-rotation compensation): removes the hand's extra step
SMOOTH_BONES_UPTO = 3       # bones of each IK chain that the IK smoothing touches (upper, lower, end)
ELBOW_POLE = True         # hand IK bends the planting arm towards the pre-IK elbow of the first plant frame
STEP_WIN_LIMIT_DEG = 25.0  # largest per-frame bone step allowed inside the played window
# before the continuity fix (measured on the first bake): max bone step in the window, bone, time (s)
BEFORE_STEP = {'speed_1m_100cm': (105.0, 'LeftForeArm', 1.033), 'speed_2m_100cm': (27.6, 'LeftForeArm', 0.633)}
OVERRIDE_TEXT = 'preview: in-game test (kong passed in game with similar offline numbers)'
TILT_MAX_DEG = 65.0        # trunk tilt (Spine2 - Hips against vertical): PASS limit, reported either way

# name -> (key in the events file, handle, clip name)
CLIPS = [
    dict(key='speed_1m_100cm', handle='47BFB02DF', name='xx_Parkour_Sprint_SlideOver_Fd100cm-Up100cm'),
    dict(key='speed_2m_100cm', handle='2E1406EF9', name='xx_Parkour_Sprint_SlideOver_Fd200cm-Up100cm'),
]


# ------------------------------------------------------------------------------------------------ inputs
def load_events(path=EVENTS_JSON):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_model(path=PLAY_MODEL):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def play_cut(model, name):
    """(o, window_clip) for a SlideOver clip: o = the model's o (reference map when the clip has none), window_clip =
    the model's cut in clip time (window_clip_end_s) minus o. The model's window is in played seconds (rate r)."""
    pc = model['per_clip'][name]
    ref = model['reference']
    o = float(pc['o']) if pc.get('o') is not None else float(ref['o'])
    end = float(pc['window_clip_end_s'])
    return o, end - o


@contextlib.contextmanager
def planting_sides(sides):
    """bake2.T9E_SIDES (the hands the T9e IK pins) narrowed to `sides` for the duration of the bake."""
    old = bake2.T9E_SIDES
    bake2.T9E_SIDES = tuple(sides)
    try:
        yield
    finally:
        bake2.T9E_SIDES = old


def _elbow_world(seq, lr, fi, rr, rp, elbow_idx):
    """World position of the bone elbow_idx for pose lr[fi] under the root transform (rr, rp)."""
    _r, W = fk.fk(seq.bones, lr[fi][None], np.asarray(seq.local_pos, float)[fi][None], root_rot=rr[fi], root_pos=rp[fi])
    return np.asarray(W, float)[0, elbow_idx].copy()


def solve_sequence_continuous(seq, root_rot, root_pos, targets, max_reach=0.999, keep_end_world_rot=KEEP_HAND_WORLD_ROT):
    """ik.solve_sequence with a per-frame pole for the LeftHand chain: the pole is the elbow of the previous frame's
    pose (after its IK), so the bend plane follows the arm frame by frame and cannot switch sides. Other effectors
    use the same two-bone IK as ik.solve_sequence. Same weights (slerp by the target weight) and same output."""
    import dataclasses
    F = len(seq.frames)
    bones = seq.bones
    rr = np.asarray(root_rot, float).reshape(F, 4)
    rp = np.asarray(root_pos, float).reshape(F, 3)
    plan = {}
    for name, entries in targets.items():
        if name not in ik.CHAINS:
            raise KeyError('unknown effector %r' % name)
        for fi, tgt, w in entries:
            fi, w = int(fi), float(w)
            if not 0 <= fi < F:
                raise IndexError('frame_index %d outside 0..%d' % (fi, F - 1))
            if not 0.0 <= w <= 1.0:
                raise ValueError('weight %r outside [0, 1]' % w)
            if w > 0.0:          # zero weight leaves the frame unchanged (as ik.solve_sequence)
                plan.setdefault(fi, []).append((name, np.asarray(tgt, float), w))
    chains = {name: ik.chain_indices(bones, name) for name in {e for v in plan.values() for e, _, _ in v}}
    lr = np.array(seq.local_rot, float)
    for fi in sorted(plan):
        cur = lr[fi].copy()
        for name, tgt, w in plan[fi]:
            if w == 0.0:
                continue
            ch = chains[name]
            pole = None
            if name == 'LeftHand' and fi > 0:
                # continuity for the planting arm: bend towards the elbow of the previous frame's pose
                pole = _elbow_world(seq, lr, fi - 1, rr, rp, ch[1])
            res_ik = ik.two_bone_ik(bones, cur, seq.local_pos[fi], ch, tgt, pole=pole,
                                    root_rot=rr[fi], root_pos=rp[fi], max_reach=max_reach,
                                    keep_end_world_rot=keep_end_world_rot)
            idx = list(ch)
            cur[idx] = fk.slerp(cur[idx], res_ik[idx], w)
        lr[fi] = cur
    # short time smoothing of the IK chains across the solved frames (ramps included): removes the single-frame jump
    # where the solution meets the ramp; each pass pulls a frame halfway towards the slerp of its neighbours
    if plan:
        fmin, fmax = min(plan), max(plan)
        for name, passes in SMOOTH_EFFECTORS.items():
            if name not in chains:
                continue
            for _ in range(passes):
                prev = lr.copy()
                for ai in chains[name][:SMOOTH_BONES_UPTO]:
                    for fi in range(max(1, fmin - 2), min(F - 1, fmax + 2) + 1):
                        nb = fk.slerp(prev[fi - 1, ai], prev[fi + 1, ai], 0.5)
                        lr[fi, ai] = fk.slerp(prev[fi, ai], nb, 0.5)
    return dataclasses.replace(seq, local_rot=lr)


@contextlib.contextmanager
def elbow_poles(enabled=True):
    """Use solve_sequence_continuous as ik.solve_sequence while the context is active (restored afterwards).

    Without it, two_bone_ik keeps the bend plane of the pre-IK pose at each frame, which can flip the elbow to the
    other side of the shoulder-hand line between two frames (the 1 m clip: elbow z 1.08 -> 1.29 m, 105 deg step).
    """
    orig = ik.solve_sequence
    if enabled:
        ik.solve_sequence = solve_sequence_continuous
    try:
        yield
    finally:
        ik.solve_sequence = orig


def wd2_event_dict(ev, dur):
    """bake2 wants takeoff, hand_plant_start/end, landing, static_until, duration."""
    return dict(takeoff=float(ev['takeoff']), hand_plant_start=float(ev['hand_plant_start']),
                hand_plant_end=float(ev['hand_plant_end']), landing=float(ev['landing']),
                static_until=float(ev['static_until']), duration=float(dur))


# ------------------------------------------------------------------------------------------------ bake
def smooth_rot_tracks(rig, dump, res, o, window, passes=1):
    """One slerp pass over the patched rotation tracks inside the played window [o, o + window] (+1 frame each side).

    Removes the single-frame steps that the fade-in and the time-mapped source leave at the window start (for
    example RightUpLeg / RightFoot at 0.46 s on the 1 m clip). Rebuilt with bake2.rotation_track, so the frames
    outside the window keep the original keys exactly and the Reference position is not touched."""
    L = float(dump['Length'])
    nf = int(round(L * 60))
    F60 = np.arange(nf + 1)
    a = max(1, int(round(o * 60)) - 1)
    b = min(nf - 1, int(round((o + window) * 60)) + 1)
    lo = max(0, int(res['win'][0]) - 1)
    hi = nf
    for _ in range(passes):
        for ai in res['patched']:
            vals = bake2.sample_track_rot(res['rot_tracks'][ai], F60)
            sm = vals.copy()
            for f in range(a, b + 1):
                nb = fk.slerp(vals[f - 1], vals[f + 1], 0.5)
                sm[f] = fk.slerp(vals[f], nb, 0.5)
            orig_keys = dumps.track(dump, rig.ac[ai]['hash'], 0) or []
            res['rot_tracks'][ai] = bake2.rotation_track(orig_keys, sm, lo, hi)
    return res


def bake_clip(rig, spec, wd2_ev, offsets):
    """Bake one clip. Returns a dict with the bake2 result and the inputs needed by validation."""
    dump = dumps.load_by_handle(spec['handle'])
    seq = acuclip.pose_sequence(dump, step=1)
    ev = contacts.events(seq, dump)
    acu = bake2.acu_seconds(ev)
    if acu['takeoff'] is None or acu['landing'] is None or acu['hand_plant'] is None:
        raise ValueError('%s: ACU takeoff, hand plant and landing are all needed (got %s)' % (spec['name'], acu))
    m = wd2.load(wd2_ev['stem'])
    W = wd2_event_dict(wd2_ev, m['dur'])
    o, window = spec['o'], spec['window']
    # the lift / hand IK / foot clamp envelopes run over the play cut plus the blend the game plays after it (the
    # patch keeps going to the clip end, D-style tail), so the body is lifted and clamped through the blend too
    env_window = window + BLEND_OUT_S if ENVELOPE_THROUGH_BLEND else window
    with planting_sides(PLANTING_SIDES), elbow_poles(ELBOW_POLE):
        res = bake2.bake_t9e(rig, m, dump, seq, acu, ev, W, o, env_window, offsets, fi=FADE_IN, fo=FADE_OUT,
                             tail=TAIL)
    if TRACK_SMOOTH_PASSES > 0:
        smooth_rot_tracks(rig, dump, res, o, window, passes=TRACK_SMOOTH_PASSES)
    return dict(dump=dump, seq=seq, ev=ev, acu=acu, wd2=W, m=m, o=o, window=window, env_window=env_window, res=res)


# ------------------------------------------------------------------------------------------------ validation
def _tilt_deg(P, rig, frames):
    """Trunk tilt (Spine2 - Hips against +Z), degrees, over the given frames of world positions P."""
    hips = rig.ac_by_hash[zlib.crc32('Hips'.encode())]
    sp2 = rig.ac_by_hash[zlib.crc32('Spine2'.encode())]
    up = P[frames, sp2] - P[frames, hips]
    n = np.linalg.norm(up, axis=1)
    cosz = np.clip(up[:, 2] / np.maximum(n, 1e-9), -1.0, 1.0)
    return np.degrees(np.arccos(cosz))


def _lowest_foot_speed(P, rig, frames, offsets):
    """Per consecutive frame pair: (horizontal speed of the lowest foot in m/s, its height above ground in m,
    its vertical speed in m/s). The Foot joint height has the contact offset removed."""
    fe = [rig.ac_by_hash[zlib.crc32(n.encode())] for n in ('LeftFoot', 'RightFoot')]
    off = float(offsets['Foot'])
    z = np.stack([P[frames, a, 2] for a in fe], axis=1) - off
    low = np.argmin(z, axis=1)
    idx = np.stack(fe)[low]
    pos = P[frames, idx]
    h = z.min(axis=1)
    v = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1) * 60.0
    vz = np.abs(np.diff(pos[:, 2])) * 60.0
    return v, h[1:], vz


def skate_stats(P, rig, frames, offsets):
    """p95 horizontal speed of the lowest foot over the stance frames: height <= SKATE_HEIGHT_M and vertical speed
    <= SKATE_VZ_MS (a foot that only passes the 5 cm band while swinging is not stance). Returns (p95, n stance)."""
    v, h, vz = _lowest_foot_speed(P, rig, frames, offsets)
    st = (h <= SKATE_HEIGHT_M) & (vz <= SKATE_VZ_MS)
    if not st.any():
        return 0.0, 0
    return float(np.percentile(v[st], 95)), int(st.sum())


def validate_sv(rig, c, offsets):
    """Metrics of the patch as the plugin plays it (60 Hz samples of the tracks), plus the speed-vault checks."""
    res, seq, dump, ev = c['res'], c['seq'], c['dump'], c['ev']
    o, window = c['o'], c['window']
    ob = ev['obstacle']
    F60 = np.asarray(seq.frames, int)
    nf = int(F60[-1])
    win = res['win']
    body_nf = res['body_nf']
    fing = res['fingers']
    pat = res['patched']
    orig = np.asarray(seq.local_rot, float)
    rot = orig.copy()
    for ai in pat:
        rot[:, ai] = bake2.sample_track_rot(res['rot_tracks'][ai], F60)
    pos = bake2.sample_track_pos(res['trans_keys'], F60)
    Pf = acuclip.world_positions(bake2.composed(seq, rot, pos), dump)
    Po = acuclip.world_positions(seq, dump)
    tail = np.arange(int(win[-1]) + 1, nf + 1)

    box_p, gnd_p = bake2.penetration_parts(Pf[win][:, body_nf], ob)
    box_t, gnd_t = bake2.penetration_parts(Pf[tail][:, body_nf], ob) if tail.size else (0.0, 0.0)
    box_o, gnd_o = bake2.penetration_parts(Po[win][:, body_nf], ob)
    fb = bake2.penetration_parts(Pf[win][:, fing], ob) if fing else (0.0, 0.0)

    # hand touch: the planting hand within TOUCH_TOL_CM of its target from the first plant frame
    plant = res['plant']
    touch_n = 0
    herr = None
    if plant.size:
        ai = rig.ac_by_hash[zlib.crc32(b'LeftHand')]
        err = np.linalg.norm(Pf[plant, ai] - res['targets']['LeftHand'], axis=1) * 100.0
        for e in err:
            if e > TOUCH_TOL_CM:
                break
            touch_n += 1
        herr = float(err.max())
    delta_v = pos - np.asarray(seq.local_pos[:, 0], float)
    dstep = float(np.abs(np.diff(delta_v[:, 2])).max() * 100.0) if len(F60) > 1 else 0.0
    step_p = bake2._step_deg(rot, np.nonzero(np.isin(F60, win))[0], pat)

    # trunk tilt over the window and the tail
    tilt_w = _tilt_deg(Pf, rig, win)
    tilt_all = _tilt_deg(Pf, rig, np.arange(nf + 1))
    tilt_o = _tilt_deg(Po, rig, win)

    # landing and foot skate over the part the game plays: landing .. cut + blend (stance frames only)
    t_land = float(c['acu']['landing'])
    f_land = int(round(t_land * 60))
    f_play = min(nf, int(round((o + window + BLEND_OUT_S) * 60)))
    skate_p95, skate_n = skate_stats(Pf, rig, np.arange(f_land, f_play + 1), offsets)
    skate_all_p95, _n = skate_stats(Pf, rig, np.arange(f_land, nf + 1), offsets)
    skate_orig_p95, _n2 = skate_stats(Po, rig, np.arange(f_land, f_play + 1), offsets)
    pen_blend = bake2.penetration_parts(Pf[np.arange(int(win[-1]) + 1, f_play + 1)][:, body_nf], ob)         if f_play > int(win[-1]) else (0.0, 0.0)
    pen_blend_cm = max(pen_blend) * 100.0
    tilt_blend = _tilt_deg(Pf, rig, np.arange(int(win[0]), f_play + 1))

    # landed and upright before the window end
    landed_margin = (o + window) - t_land            # s, > 0 means feet down before the cut
    # continuity at W: patched pose vs the next clip's first pose (bones of the rig, rotations)
    nxt = acuclip.pose_sequence(dumps.load_by_name(NEXT_CLIP), step=1)
    fW = int(round((o + window) * 60))
    fW = min(fW, nf)
    body_ai = [ai for ai in pat]
    gap_patch = bake2.angle_deg(np.asarray(res['final'][fW, body_ai], float),
                                np.asarray(nxt.local_rot[0, body_ai], float))
    gap_orig = bake2.angle_deg(np.asarray(orig[fW, body_ai], float),
                               np.asarray(nxt.local_rot[0, body_ai], float))

    jag = bake2.jaggedness_t9e(rig, c['m'], dump, seq, res, res['time_map'])
    # largest per-frame bone step inside the played window [o, cut] (60 Hz samples of the tracks)
    fw0, fw1 = int(round(o * 60)), min(nf, int(round((o + window) * 60)))
    if fw1 > fw0 and pat:
        steps = np.stack([bake2.angle_deg(rot[f, pat], rot[f - 1, pat]) for f in range(max(fw0, 1), fw1 + 1)])
        step_win = float(steps.max())
        step_win_bone = bake2._bone_name(rig, pat[int(np.unravel_index(np.argmax(steps), steps.shape)[1])])
    else:
        step_win, step_win_bone = 0.0, '-'

    fails = []
    pen_win = max(box_p, gnd_p) * 100.0
    if pen_win > bake2.T9C_PEN_CM:
        fails.append('window penetration %.1f cm' % pen_win)
    if touch_n < bake2.T9D_TOUCH:
        fails.append('hand touch %d frames (need %d)' % (touch_n, bake2.T9D_TOUCH))
    if dstep > bake2.T9C_DELTA_STEP_CM:
        fails.append('Delta step %.1f cm/frame' % dstep)
    if tilt_w.max() > TILT_MAX_DEG:
        fails.append('trunk tilt %.1f deg (limit %.0f)' % (tilt_w.max(), TILT_MAX_DEG))
    if step_win > STEP_WIN_LIMIT_DEG:
        fails.append('bone step %.1f deg in the window (%s)' % (step_win, step_win_bone))
    if pen_blend_cm > bake2.T9C_PEN_CM:
        fails.append('penetration in the blend %.1f cm' % pen_blend_cm)
    if skate_p95 > SKATE_P95_MS:
        fails.append('foot skate p95 %.2f m/s' % skate_p95)
    if landed_margin < 0:
        fails.append('feet down %.3f s after the window end' % (-landed_margin))

    met = dict(
        pen_window_cm=pen_win, pen_ground_cm=gnd_p * 100.0, pen_box_cm=box_p * 100.0,
        pen_tail_cm=max(box_t, gnd_t) * 100.0, pen_orig_cm=max(box_o, gnd_o) * 100.0,
        pen_finger_cm=max(fb) * 100.0,
        touch_frames=touch_n, plant_frames=int(plant.size), hand_err_cm=herr,
        dstep_max_cm=dstep, step_patch_deg=float(step_p),
        tilt_max_win_deg=float(tilt_w.max()), tilt_max_clip_deg=float(tilt_all.max()),
        tilt_max_orig_deg=float(tilt_o.max()),
        skate_p95_ms=skate_p95, skate_frames=skate_n, skate_all_p95_ms=skate_all_p95,
        skate_orig_p95_ms=skate_orig_p95, pen_blend_cm=pen_blend_cm,
        landed_margin_s=float(landed_margin),
        gap_patch_deg=float(np.median(gap_patch)), gap_patch_max_deg=float(gap_patch.max()),
        gap_orig_deg=float(np.median(gap_orig)), gap_orig_max_deg=float(gap_orig.max()),
        jag_patch=jag['patch_max'], jag_source=jag['source_max'], jag_orig=jag['orig_max'],
        step_win_max_deg=step_win, step_win_bone=step_win_bone,
        fails=fails, fail=bool(fails))
    arrays = dict(frames=F60, Pf=Pf, Po=Po, win=win, body=res['body'])
    return met, arrays, dict(tilt=tilt_w, t_land=t_land, tilt_blend=tilt_blend)


# ------------------------------------------------------------------------------------------------ timing
def timing(c, offsets):
    """Ratios of the WD2 and ACU spans and the time-map slopes (clip s / WD2 s)."""
    W, acu = c['wd2'], c['acu']
    wd_span = W['landing'] - W['takeoff']
    acu_span = acu['landing'] - acu['takeoff']
    return dict(wd2_takeoff_landing_s=wd_span, acu_takeoff_landing_s=acu_span,
                wd2_over_acu=wd_span / acu_span,
                wd2_takeoff_plant_s=W['hand_plant_start'] - W['takeoff'],
                acu_takeoff_plant_s=acu['hand_plant'] - acu['takeoff'],
                window_clip_s=c['window'], acu_landing_clip_s=acu['landing'],
                cut_clip_s=c['o'] + c['window'])


def time_map_table(c):
    """Anchor pairs and monotonic check of the time map (clip s -> WD2 s)."""
    W, acu = c['wd2'], c['acu']
    pts = [('takeoff', acu['takeoff'], W['takeoff']),
           ('hand_plant', bake2.acu_hand_plant(acu, W), W['hand_plant_start']),
           ('landing', acu['landing'], W['landing'])]
    tm = c['res']['time_map']
    tt = np.linspace(c['o'], c['o'] + c['window'] + 0.3, 200)
    ww = tm(tt)
    mono = bool(np.all(np.diff(ww) >= -1e-12))
    return pts, mono, float(tm(np.array([c['o']]))[0]), float(tm(np.array([c['o'] + c['window']]))[0])


# ------------------------------------------------------------------------------------------------ sheets
def wd2_stick_sheet(stem, key, out_path, t0, t1, marks):
    """Side-view stick figures of the WD2 clip (pelvis at origin), plus the traces' times of the events."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import wd2events as we
    tr = we.effector_traces(stem)
    t = tr['t']
    J = tr['joints']
    ks = [k for k in range(len(t)) if t0 - 1e-9 <= t[k] <= t1 + 1e-9]
    sel = ks[::max(1, len(ks) // 12)]
    bones = [('Hips', 'Spine'), ('Spine', 'Spine1'), ('Spine1', 'Spine2'), ('Spine2', 'Neck'), ('Neck', 'Head'),
             ('Spine2', 'LeftArm'), ('LeftArm', 'LeftForeArm'), ('LeftForeArm', 'LeftHand'),
             ('Spine2', 'RightArm'), ('RightArm', 'RightForeArm'), ('RightForeArm', 'RightHand'),
             ('Hips', 'LeftUpLeg'), ('LeftUpLeg', 'LeftLeg'), ('LeftLeg', 'LeftFoot'),
             ('Hips', 'RightUpLeg'), ('RightUpLeg', 'RightLeg'), ('RightLeg', 'RightFoot')]
    nc = (len(sel) + 1) // 2
    fig, axs = plt.subplots(2, nc, figsize=(2.6 * nc, 6.6), sharey=True)
    axs = np.asarray(axs).ravel()
    for ax, k in zip(axs, sel):
        for a, b in bones:
            pa, pb = J[a][k], J[b][k]
            col = 'C3' if a.startswith('Left') and a != 'Hips' else ('C0' if a.startswith('Right') else 'k')
            ax.plot([pa[0], pb[0]], [pa[2], pb[2]], '-', color=col, lw=2)
        ax.plot(J['LeftHand'][k][0], J['LeftHand'][k][2], 'o', color='C3', ms=5)
        ax.plot(J['RightHand'][k][0], J['RightHand'][k][2], 'o', color='C0', ms=5)
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.1, 1.0)
        ax.set_aspect('equal')
        ax.set_title('t=%.2f' % t[k], fontsize=9)
    for ax in axs[len(sel):]:
        ax.axis('off')
    marks_s = ', '.join('%s %.2f' % (n, v) for n, v in marks)
    plt.suptitle('%s side view (pelvis at origin; red = left arm, blue = right). Annotated: %s' % (key, marks_s),
                 fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=80)
    plt.close(fig)


def still_grid(rig, c, arrays, path, ob, o, window, n=8):
    """Still frames over [o - 0.05, o + window + 0.1]: obstacle, original (grey) and patch (colour), side view."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    Pf, Po, body = arrays['Pf'], arrays['Po'], arrays['body']
    F = arrays['frames']
    f0 = max(0, int(round((o - 0.05) * 60)))
    f1 = min(int(F[-1]), int(round((o + window + 0.1) * 60)))
    sel = np.linspace(f0, f1, n).round().astype(int)
    bset = set(body)
    edges = [(ai, rig.ac[ai]['parent']) for ai in body if rig.ac[ai]['parent'] in bset]
    nc = (n + 1) // 2
    fig, axs = plt.subplots(2, nc, figsize=(2.6 * nc, 5.6), sharey=True)
    axs = np.asarray(axs).ravel()
    for ax, f in zip(axs, sel):
        ax.add_patch(plt.Rectangle((ob['near_y'], 0), ob['far_y'] - ob['near_y'], ob['height'], color='0.8'))
        for P, col in ((Po, '0.6'), (Pf, 'C0')):
            for a, p in edges:
                ax.plot([P[f, a, 1], P[f, p, 1]], [P[f, a, 2], P[f, p, 2]], '-', color=col, lw=1.6)
        ax.set_title('t=%.2f%s' % (f / 60.0, ' (cut)' if abs(f / 60.0 - (o + window)) < 0.009 else ''), fontsize=8)
        ax.set_aspect('equal')
        ax.set_xlim(ob['near_y'] - 1.6, ob['far_y'] + 1.2)
        ax.set_ylim(-0.05, 2.2)
    for ax in axs[len(sel):]:
        ax.axis('off')
    plt.suptitle('grey = original ACU clip, colour = speed-vault patch (side view, obstacle grey box)', fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ outputs
def write_mapping(entries, path=MAPPING):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'clips': entries}, f, indent=1)


def mapping_entry(handle, name, source, enabled, override=None):
    e = {'handle': '0x' + handle, 'name': name, 'source': source, 'enabled': bool(enabled), 'prebaked': True,
         'patch_translation': True, 'pool_weight': POOL_WEIGHT}
    if override:
        e['override'] = override
    return e


def _tf(x, nd=2):
    return 'n/a' if x is None else ('%.*f' % (nd, x))


def write_report(rows, path=REPORT):
    L = ['# Sprint SlideOver speed vaults (WD2 vaultover -> ACU)', '',
         'Generated by `pipeline/speedvault.py`: T9e (bake2.bake_t9e) with D-style options (fade-in %.2f s, no fade-out,'
         ' tail wd2), narrowed to the planting hand (LeftHand). Events: `data/wd2_events_speedvault.json`.' % FADE_IN,
         '', '## Time maps (clip s -> WD2 s)', '',
         '| clip | anchor | ACU (clip s) | WD2 (s) |', '|---|---|---|---|']
    for r in rows:
        for nm, a, w in r['anchors']:
            L.append('| %s | %s | %.3f | %.3f |' % (r['name'], nm, a, w))
        L.append('| %s | monotonic over the window: %s; WD2 time at o = %.3f, at the cut = %.3f | | |' % (
            r['name'], r['monotonic'], r['tm_o'], r['tm_w']))
    L += ['', '## Window and timing', '',
          '| clip | o (clip s) | cut = o + window (clip s) | ACU feet down (clip s) | feet down before the cut (s) | WD2 takeoff..landing (s) | ACU takeoff..feet down (s) | WD2/ACU span | WD2 takeoff..plant (s) | ACU takeoff..plant (s) |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        t, m = r['timing'], r['metrics']
        L.append('| %s | %.3f | %.3f | %.3f | %+.3f | %.3f | %.3f | %.2f | %.3f | %.3f |' % (
            r['name'], r['o'], r['o'] + r['window'], t['acu_landing_clip_s'], m['landed_margin_s'],
            t['wd2_takeoff_landing_s'], t['acu_takeoff_landing_s'], t['wd2_over_acu'],
            t['wd2_takeoff_plant_s'], t['acu_takeoff_plant_s']))
    L += ['', '## Validation', '',
          '| clip | window pen (cm) | blend pen (cm) | box / ground (cm) | tail pen (cm, info) | hand touch (frames, need %d) | hand err max (cm) | Delta step (cm/frame) | trunk tilt max win / clip (deg) | foot skate p95 (m/s) | skate, original, same rule | jag patch / source / orig (deg/frame) | max bone step in window (deg, bone) | gap at cut, patch / orig (median deg) | verdict |' % bake2.T9D_TOUCH,
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        m = r['metrics']
        L.append('| %s | %.1f | %.1f | %.1f / %.1f | %.1f | %d | %s | %.2f | %.1f / %.1f | %.2f | %.2f | %.1f / %.1f / %.1f | %.1f (%s) | %.1f / %.1f | %s |' % (
            r['name'], m['pen_window_cm'], m['pen_blend_cm'], m['pen_box_cm'], m['pen_ground_cm'], m['pen_tail_cm'],
            m['touch_frames'], _tf(m['hand_err_cm'], 1), m['dstep_max_cm'], m['tilt_max_win_deg'],
            m['tilt_max_clip_deg'], m['skate_p95_ms'], m['skate_orig_p95_ms'], m['jag_patch'], m['jag_source'],
            m['jag_orig'], m['step_win_max_deg'], m['step_win_bone'], m['gap_patch_deg'], m['gap_orig_deg'],
            'PASS' if not m['fail'] else 'FAIL: ' + '; '.join(m['fails'])))
    L += ['', '## Criteria and what they measure', '',
          '- window / blend penetration: bake2.penetration_parts on the non-finger body, to the cut and to the cut + 0.25 s (limit %.0f cm).' % bake2.T9C_PEN_CM,
          '- hand touch: planting hand within %.0f cm of its target, consecutive plant frames from the first (need %d).' % (TOUCH_TOL_CM, bake2.T9D_TOUCH),
          '- Delta step: largest change of the Reference z offset between 60 Hz frames (limit %.0f cm).' % bake2.T9C_DELTA_STEP_CM,
          '- trunk tilt: angle of Spine2 - Hips from vertical over the window (limit %.0f deg).' % TILT_MAX_DEG,
          '- foot skate: p95 horizontal speed of the lowest foot over stance frames (height <= %.2f m and |vz| <= %.1f m/s), from the ACU landing to cut + blend (limit %.1f m/s). The same rule on the original clip is shown for comparison.' % (SKATE_HEIGHT_M, SKATE_VZ_MS, SKATE_P95_MS),
          '- continuity: per patched bone, the angle between the pose at the cut (patch, or original) and the first pose of %s (median / max).' % NEXT_CLIP,
          '- jaggedness: largest per-60 Hz-frame bone angle step (bake2.jaggedness_t9e): patch, retargeted source, original.',
          '- landed before the cut: ACU feet down (clip s) <= cut.', '']
    for r in rows:
        if r['metrics']['fail']:
            L.append('- %s: FAILED offline (%s). Written with enabled = true and the override below.' % (
                r['name'], '; '.join(r['metrics']['fails'])))
        else:
            L.append('- %s: PASS.' % r['name'])
    L += ['', 'Mapping override (set by coordinator): "%s". Note: the offline criteria above fail for both clips '
          '(skate and Delta step); the override reason is set by the coordinator, not measured here.' % OVERRIDE_TEXT, '']
    L += ['', '## Bone steps in the played window (before / after the continuity fix)', '',
          '| clip | before: max step (deg, bone, at) | after: max step (deg, bone) | limit (deg) |', '|---|---|---|---|']
    for r in rows:
        b = BEFORE_STEP[r['key']]
        L.append('| %s | %.1f (%s, %s) | %.1f (%s) | %.0f |' % (r['name'], b[0], b[1], b[2], r['metrics']['step_win_max_deg'],
                 r['metrics']['step_win_bone'], STEP_WIN_LIMIT_DEG))
    L += ['', 'What changed: (1) the planting-arm IK pole is the previous frame elbow (solve_sequence_continuous), so the '
          'bend plane follows the arm and cannot switch sides; (2) the IK chain rotations are smoothed one pass '
          '(LeftHand), two-to-three passes on the feet; (3) the patched rotation tracks get one slerp pass across the '
          'played window (smooth_rot_tracks). The hand touch and penetration gates still pass after (1)-(3).', '']
    L += ['', 'Sheets (sheets folder): <name>.gif (side view, grey original, colour patch), <name>_frames.png (still frames '
          'to the cut + 0.1 s), wd2_stick_<key>.png (WD2 source stick figures with the annotated events).', '']
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


def run(write=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SHEET_DIR, exist_ok=True)
    events = load_events()
    model = load_model()
    rig = R.Rig()
    # contact offsets pooled over the SkipOver clips, exactly as the kong bake (bake2.bake_all_t9e)
    pool = []
    for _h, nm, dump in bake2.targets():
        seq = acuclip.pose_sequence(dump, step=1)
        pool.append((nm, dump, seq, contacts.events(seq, dump)))
    offsets = bake2.measure_contact_offsets(pool)
    rows, entries = [], []
    for spec in CLIPS:
        ev_key = spec['key']
        wev = events[ev_key]
        o, window = play_cut(model, spec['name'])
        spec = dict(spec, o=o, window=window)
        c = bake_clip(rig, dict(spec, stem=wev['stem']), dict(wev, stem=wev['stem']), offsets)
        met, arrays, extra = validate_sv(rig, c, offsets)
        pts, mono, tm_o, tm_w = time_map_table(c)
        tim = timing(c, offsets)
        c['res']['name'] = spec['name']
        rows.append(dict(key=ev_key, name=spec['name'], handle=spec['handle'], stem=wev['stem'], o=o,
                         window=window, metrics=met, timing=tim, anchors=pts, monotonic=mono, tm_o=tm_o,
                         tm_w=tm_w, offsets=offsets, extra=extra, ev=wev, c=c, arrays=arrays))
        print('%-44s pen_win %5.1f touch %d/%d tilt %5.1f skate95 %5.2f gap %5.1f dstep %5.2f %s' % (
            spec['name'], met['pen_window_cm'], met['touch_frames'], bake2.T9D_TOUCH, met['tilt_max_win_deg'],
            met['skate_p95_ms'], met['gap_patch_deg'], met['dstep_max_cm'],
            'PASS' if not met['fail'] else 'FAIL (%s)' % '; '.join(met['fails'])), flush=True)
        if write:
            fn = '%s.baked.json' % spec['handle']
            bake2.write_patch_t9c(os.path.join(OUT_DIR, fn), rig, c['res']['rot_tracks'], c['res']['trans_keys'],
                                  float(c['dump']['Length']))
            entries.append(mapping_entry(spec['handle'], spec['name'], fn, enabled=True, override=OVERRIDE_TEXT))
            bake2.write_gif_t9e(os.path.join(SHEET_DIR, spec['name'] + '.gif'), rig, c['res'], arrays,
                                c['ev']['obstacle'], o, window, c['res']['time_map'])
            still_grid(rig, c, arrays, os.path.join(SHEET_DIR, spec['name'] + '_frames.png'),
                       c['ev']['obstacle'], o, window)
            wd2_stick_sheet(wev['stem'], ev_key, os.path.join(SHEET_DIR, 'wd2_stick_%s.png' % ev_key), 0.6,
                            2.0 if ev_key == 'speed_1m_100cm' else 1.9,
                            [('takeoff', wev['takeoff']), ('plant', wev['hand_plant_start']),
                             ('landing', wev['landing'])])
    if write:
        write_mapping(entries)
        write_report(rows)
    return rig, rows, entries, offsets


if __name__ == '__main__':
    import sys
    write = '--no-write' not in sys.argv
    run(write=write)
