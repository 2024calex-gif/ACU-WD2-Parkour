"""T12b: the WD2 frontdive as a SlideOver + gated Muraille_Exit chain on one joined played timeline.

Joined played time J: SlideOver played p in [0, W], then the Exit played p_e from W, J = W + p_e (Exit window 0.228 s,
median). Clip times: SlideOver t = o + r * p; Exit t = 0.208 + 0.9665 * p_e (out/match/exit_report.md).

Time map (WD2 seconds w(J), monotone, slope 1 at both joins):
  J <= p_takeoff:          w = 0.45 + (J - p_takeoff)                      (run-up, k = 1)
  p_takeoff <= J <= T:     monotone cubic Hermite, w(p_to) = 0.45, w(T) = 0.75, slope 1 at both ends (stretched flight)
  J >= T:                  w = 0.75 + (J - T)                              (k = 1 through plant, roll, Exit)
T (the plant start, played) is the EARLIEST J at which a plant beyond the far edge is feasible (search, see find_T).

Rotations: additive retarget on the joined timeline, D(w) = A_REF * Wt(w) * conj(A_REF * Wt(w0)), w0 = the SlideOver
patch start; patched world = D * clip dump world (the Reference carries the pelvis flip). The Exit patch therefore starts
at the WD2 pose the SlideOver tail reached at W (no fade-in).

Root: one joined smooth offset (Dy bump, Dz signed lift envelope) over the SlideOver window and the Exit window, so the
offset is continuous across the boundary by construction. Bounds: box clearance, ground clearance, reach of the plant
hands, touch bound (lowest non-finger point on the ground) through the roll, run-out and feet-down.

Outputs (out\\dive): patches\\<HEX>.baked.json for the SlideOvers and the Exit (HEX F7919BDF0), mapping_dive.json (SlideOver
entries plus ONE gated Exit entry: only_after = the enabled SlideOvers, arm_seconds 2.0), report.md, PNG and GIF.
"""
import json
import math
import os
import zlib

import numpy as np

import acuclip
import bake2
import bake_dive as BD
import config
import contacts
import dumps
import ik
import retarget as R
import wd2

FPS = acuclip.FPS
EXIT_NAME = 'xx_Parkour_Muraille_Exit_Fd100cm-Dn100cm'
EXIT_HEX = 'F7919BDF0'
NEXT_NAME = 'xx_Nav_Run_3-Loop'
MAIN_SO = 'xx_Parkour_Sprint_SlideOver_Fd100cm-Up100cm'
SO_SCOPE = (MAIN_SO, 'xx_Parkour_Sprint_SlideOver_Fd200cm-Up100cm', 'xx_Parkour_Sprint_SlideOver_Fd300cm-Up100cm')
EXIT_O = 0.208
EXIT_R = 0.9665
EXIT_WIN = 0.228                      # s, played Exit window (median)
EXIT_TAIL_NOTE = 'tail wd2 to the Exit clip end'
T_GRID = np.round(np.arange(0.30, 0.801, 0.02), 3)     # candidate plant starts (played J)
REUSE_EXTRA_SEARCH = 'run3'   # extra SlideOvers: T search (run 3, same code path, no Exit dependence) reused
FIND_A_GRID = np.round(np.arange(0.0, 1.001, 0.1), 3)  # forward amplitudes tried in the T search (coarse)
CONT_DEG = 45.0                       # PASS: largest local-rotation gap across W on the continuity bones
STEP_CM = BD.DSTEP_CM
GAP_WARN = BD.GAP_WARN_DEG
EASE_S = 0.15
OUT_DIR = BD.DIVE_OUT
PATCH_DIR = BD.PATCH_DIR
MAPPING = os.path.join(PATCH_DIR, 'mapping_dive.json')
REPORT = os.path.join(OUT_DIR, 'report.md')
CONT_BONES = BD.GAP_BONES


def _exit_support_z():
    """Feet support height of the Exit in its own dump frame (out/match/play_model_exit.json, geometry)."""
    with open(os.path.join(config.OUT, 'match', 'play_model_exit.json'), encoding='utf-8') as f:
        return float(json.load(f)['per_clip']['Fd100cm-Dn100cm']['geometry']['feet_support_z'])


EXIT_Z_SHIFT = -_exit_support_z()     # the Exit's ground is at z = feet_support_z in its frame; shift it to 0 for solving


# ------------------------------------------------------------------------------------------------ time map
def wd2_of_J(J, p_to, T):
    """WD2 seconds at joined played time J (see the module docstring)."""
    J = np.asarray(J, float)
    out = np.empty_like(J)
    pre = J <= p_to
    post = J >= T
    mid = ~(pre | post)
    out[pre] = BD.WD2_TAKEOFF + (J[pre] - p_to)
    out[post] = BD.WD2_PLANT[0] + (J[post] - T)
    h = T - p_to
    u = (J[mid] - p_to) / h
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    out[mid] = h00 * BD.WD2_TAKEOFF + h10 * h + h01 * BD.WD2_PLANT[0] + h11 * h
    return out


class Timeline:
    """Joined timeline of one SlideOver + the Exit: clip time -> joined J -> WD2 seconds."""

    def __init__(self, mp, p_to, W, T):
        self.o, self.r, self.p_to, self.W, self.T = float(mp['o']), float(mp['r']), float(p_to), float(W), float(T)
        self.J_end = self.W + EXIT_WIN

    def J_so(self, t):
        return (np.asarray(t, float) - self.o) / self.r

    def J_exit(self, t):
        return self.W + (np.asarray(t, float) - EXIT_O) / EXIT_R

    def w_so(self, t):
        return wd2_of_J(self.J_so(t), self.p_to, self.T)

    def w_exit(self, t):
        return wd2_of_J(self.J_exit(t), self.p_to, self.T)


# ------------------------------------------------------------------------------------------------ rotations
def chain_rotations(rig, m, wfun, w0, frames, patset, rot):
    """{ai: [F, 4]} additive retarget of the WD2 pose onto one clip (see the module docstring).

    wfun(t) gives WD2 seconds at clip seconds t. Hips has no track and follows the Reference (the pelvis delta)."""
    n = len(rig.ac)
    hips = rig.ac_by_hash[zlib.crc32(b'Hips')]
    Wt0 = [R.qn(q) for q in bake2.wd2_pose_at(rig, m, float(w0))[1]]
    out = {ai: np.zeros((len(frames), 4)) for ai in patset}
    ws = wfun(np.asarray(frames, float) / FPS)
    for j, f in enumerate(frames):
        Wt = bake2.wd2_pose_at(rig, m, float(ws[j]))[1]

        def delta(ai):
            a = R.qn(R.qmul(R.A_REF, Wt[ai]))
            b = R.qn(R.qmul(R.A_REF, Wt0[ai]))
            return R.qn(R.qmul(a, R.qconj(b)))

        Wd = [None] * n
        for ai in range(n):
            loc = bake2.orig_local(rig, rot, ai, int(f))
            Wd[ai] = R.qn(loc) if rig.ac[ai]['parent'] < 0 else R.qn(R.qmul(Wd[rig.ac[ai]['parent']], loc))
        W = [None] * n
        for ai in range(n):
            if ai == 0:
                W[0] = R.qn(R.qmul(delta(hips), Wd[0])) if 0 in patset else Wd[0]
                if 0 in patset:
                    out[0][j] = W[0]
                continue
            par = rig.ac[ai]['parent']
            if ai in patset:
                desired = R.qn(R.qmul(delta(ai), Wd[ai]))
                out[ai][j] = R.qn(R.qmul(R.qconj(W[par]), desired))
                W[ai] = desired
            else:
                W[ai] = R.qn(R.qmul(W[par], bake2.orig_local(rig, rot, ai, int(f))))
    return out


def segment(rig, m, seq, dump, wfun, w0, weight_fn, z_shift=0.0, disp=None):
    """Patched local rotations of one clip (all frames): orig, patch (full weight), blend (weighted), patset, rot.

    disp: the root displacement [F, 3] the clip plays with (None: the dump's TrackID 0). It is stored as seg['disp']
    and used for every world position of the segment (the root solve, IK, validation and plots)."""
    F = np.asarray(seq.frames, int)
    t60 = F / FPS
    if disp is None:
        disp = acuclip.markers(dump, F)['disp']
    patset, rot = BD.patch_set(rig, dump)
    pat = chain_rotations(rig, m, wfun, w0, list(F), patset, rot)
    orig = np.asarray(seq.local_rot, float)
    patch = orig.copy()
    for ai in patset:
        patch[:, ai] = pat[ai]
    w = weight_fn(t60)
    blend = orig.copy()
    for ai in patset:
        blend[:, ai] = bake2.mix_rot(orig[:, ai], pat[ai], w)
    ref_orig = np.asarray(seq.local_pos[:, 0], float)
    shift = np.array([0.0, 0.0, float(z_shift)])
    return dict(seq=seq, dump=dump, F=F, t60=t60, patset=patset, rot=rot, orig=orig, patch=patch, blend=blend,
                w=w, ref0=ref_orig + shift, ref0_orig=ref_orig, z_shift=float(z_shift),
                disp=np.asarray(disp, float))


# ------------------------------------------------------------------------------------------------ setup helpers
def _body_sets(rig):
    body_all = sorted(set(rig.map.values()))
    fingers = bake2.finger_bones(rig, body_all)
    body_nf = [b for b in body_all if b not in set(fingers)]
    arms = {rig.ac_by_hash[zlib.crc32(n.encode())] for n in
            ('LeftArm', 'LeftForeArm', 'LeftHand', 'RightArm', 'RightForeArm', 'RightHand')}
    body_nh = [b for b in body_nf if b not in arms]
    return body_all, fingers, body_nf, body_nh


def _world(seq, dump, rotl, ref0, dy, dz, disp=None):
    return BD.world(seq, dump, rotl, ref0, dy, dz, disp=disp)


def _ob_ground():
    """Ground-only obstacle for the Exit (no box after the vault; the shared obstacle is in another entity frame)."""
    return dict(height=0.0, near_y=1e9, far_y=1e9, near_x=0.0, far_z=0.0, depth=0.0)


# ------------------------------------------------------------------------------------------------ 1. T search
def so_ctx(rig, m, seg, ob, offsets, tl, span):
    """SlideOver-only context for _forward_stage (plant start at tl.T)."""
    body_all, fingers, body_nf, body_nh = _body_sets(rig)
    o, r, W = tl.o, tl.r, tl.W
    t_end = o + span
    win = bake2.window_frames(o, span, int(seg['F'][-1]))
    w_win = tl.w_so(seg['t60'][win])
    plant = win[(w_win >= BD.WD2_PLANT[0] - 1e-9) & (w_win <= BD.WD2_PLANT[1] + 1e-9)]
    roll = win[(w_win > BD.WD2_PLANT[1] + 1e-9) & (w_win <= BD.WD2_ROLL_END + 1e-9)]
    t_peak = o + r * tl.T
    if t_peak >= t_end - 1e-6:
        t_peak = o + 0.5 * span
    B = BD.bump(seg['t60'], o, t_peak, t_end)
    return dict(rig=rig, seq=seg['seq'], dump=seg['dump'], ob=ob, win=win, plant=plant, roll=roll,
                body_nf=body_nf, body_nh=body_nh, B=B, patch=seg['patch'], blend=seg['blend'], ref0=seg['ref0'],
                offsets=offsets)


def find_T(rig, m, so_seq, so_dump, ev, mp, offsets, p_to, W):
    """Earliest plant start T on T_GRID with a feasible plant (reach + box/ground clearance at the plant start).

    Returns (T, ctx, stage) or (None, None, None) when no T on the grid is feasible."""
    ob = ev['obstacle']
    span = mp['r'] * mp['W']
    for T in T_GRID:
        if T <= p_to + 0.02:
            continue
        print('  find_T: trying T=%.2f (played)' % T, flush=True)
        tl = Timeline(mp, p_to, W, T)
        seg = segment(rig, m, so_seq, so_dump, tl.w_so, tl.w_so(round(tl.o * FPS) / FPS), lambda t: bake2.patch_weight(
            t, tl.o, span, BD.FI, BD.FO, BD.TAIL))
        ctx = so_ctx(rig, m, seg, ob, offsets, tl, span)
        for A in FIND_A_GRID:
            st = BD._forward_stage(ctx, A)
            if st['reach_ok']:
                return float(T), ctx, st
    return None, None, None


# ------------------------------------------------------------------------------------------------ 2. joined root solve
def joined_stage(rig, A, S, E, tl, offsets, ob):
    """Lift need, bounds and the solved joined Dz for forward amplitude A.

    S, E: SlideOver and Exit segment dicts (segment()) with their windows; returns dict."""
    body_all, fingers, body_nf, body_nh = _body_sets(rig)
    out = {}
    parts_L, parts_U = [], []
    targets = {}
    reach_ok = True
    for key, seg, is_so in (('S', S, True), ('E', E, False)):
        seq, dump = seg['seq'], seg['dump']
        Jf = tl.J_so(seg['t60']) if is_so else tl.J_exit(seg['t60'])
        Bf = BD.bump(Jf, 0.0, tl.T, tl.J_end)
        dy = A * Bf
        zero = np.zeros(len(dy))
        Pb = _world(seq, dump, seg['blend'], seg['ref0'], dy, zero, disp=seg['disp'])
        Pp = _world(seq, dump, seg['patch'], seg['ref0'], dy, zero, disp=seg['disp'])
        win = seg['win']
        w_win = (tl.w_so(seg['t60'][win]) if is_so else tl.w_exit(seg['t60'][win]))
        plant_mask = (w_win >= BD.WD2_PLANT[0] - 1e-9) & (w_win <= BD.WD2_PLANT[1] + 1e-9)
        ground_nh = body_nh if is_so else body_nf
        ob_seg = seg['ob']
        Lfull = bake2.lift_need_signed(Pb[win][:, body_nf], ob_seg)
        Lnh = bake2.lift_need_signed(Pb[win][:, ground_nh], ob_seg)
        L = np.where(plant_mask, Lnh, Lfull) if is_so else Lfull
        U = np.full(len(win), np.inf)
        if is_so and np.any(plant_mask):
            plant = win[plant_mask]
            p0, pend = int(plant[0]), int(plant[-1])
            pos = plant_mask.nonzero()[0]
            reach_sides = []
            for side, h in (('Left', 'LeftHand'), ('Right', 'RightHand')):
                ua, la, ha = BD._acu(rig, side + 'Arm'), BD._acu(rig, side + 'ForeArm'), BD._acu(rig, h)
                p = Pp[p0, ha]
                tgt = np.array([p[0], float(ob_seg['far_y']) + BD.HAND_MARGIN, offsets['Hand']])
                targets[h] = tgt
                sh = Pp[plant, ua]
                arm = np.linalg.norm(Pp[plant, la] - Pp[plant, ua], axis=1) + np.linalg.norm(Pp[plant, ha] - Pp[plant, la], axis=1)
                reach_sides.append(bake2.reach_upper(sh, tgt, bake2.T9E_REACH * arm))
            Uplant = np.min(np.stack(reach_sides), axis=0)
            U[pos] = Uplant
            n_t = min(BD.TOUCH_FRAMES, plant.size)
            reach_ok = bool(np.all(np.isfinite(Uplant[:n_t])) and np.all(L[pos][:n_t] <= Uplant[:n_t] + 1e-9))
            out['plant_J'] = (float(Jf[plant[0]]), float(Jf[plant[-1]]))
        touch = (w_win > BD.WD2_PLANT[1] + 1e-9) & (w_win <= 1.6 + 1e-9)
        if np.any(touch):
            pos_t = touch.nonzero()[0]
            zlow = Pp[win[touch]][:, body_nf, 2].min(axis=1)
            U[pos_t] = np.minimum(U[pos_t], -zlow)
        parts_L.append(L)
        parts_U.append(U)
    Lj = np.concatenate(parts_L)
    Uj = np.concatenate(parts_U)
    D, linfo = bake2.solve_lift_t9e(Lj, Uj, ramp_edges=True)
    nS = len(S['win'])
    out.update(dict(A=float(A), D=D, DS=D[:nS], DE=D[nS:], L=Lj, U=Uj, conflicts=int(np.sum(Lj > Uj + 1e-9)),
                    reach_ok=reach_ok, targets=targets, linfo=linfo))
    out['feasible'] = bool(reach_ok and out['conflicts'] == 0)
    return out


def choose_joined(rig, S, E, tl, offsets, ob):
    stages = []
    chosen = None
    for A in BD.DY_GRID:
        st = joined_stage(rig, A, S, E, tl, offsets, ob)
        stages.append(st)
        if st['feasible']:
            chosen = st
            break
    if chosen is None:
        chosen = min(stages, key=lambda st: (st['conflicts'], st['A']))
    return chosen, len(stages)


# ------------------------------------------------------------------------------------------------ 3. finalise one clip
def finalize(rig, seg, dy_full, dz_full, win, plant, targets, ob, offsets, tail_win=None):
    """Hand IK (plant targets, weight while reachable), foot clamp IK, crossfade, tracks.

    Returns dict with final (local rotations), ref_new (Reference position), rot_tracks, trans_keys, delta."""
    seq, dump = seg['seq'], seg['dump']
    F = seg['F']
    nf = int(F[-1])
    orig, patch, patset, wgt = seg['orig'], seg['patch'], seg['patset'], seg['w']
    ref0 = seg['ref0']
    ref1 = ref0.copy()
    ref1[:, 1] += dy_full
    ref1[:, 2] += dz_full
    disp = seg['disp']
    P1 = _world(seq, dump, patch, ref0, dy_full, dz_full, disp=disp)
    ik_in = bake2.composed(seq, patch, ref1)
    ik_targets = {}
    if plant.size and targets:
        p0, pend = int(plant[0]), int(plant[-1])
        frr = np.arange(p0, pend + 1)
        for side, h in (('Left', 'LeftHand'), ('Right', 'RightHand')):
            ua, la, ha = BD._acu(rig, side + 'Arm'), BD._acu(rig, side + 'ForeArm'), BD._acu(rig, h)
            tgt = targets[h]
            d = np.linalg.norm(tgt[None, :] - P1[frr, ua], axis=1)
            arm1 = np.linalg.norm(P1[frr, la] - P1[frr, ua], axis=1) + np.linalg.norm(P1[frr, ha] - P1[frr, la], axis=1)
            wh, _u = bake2._ik_weights(len(F), win, p0, pend, d <= bake2.T9E_REACH * arm1, ramp=bake2.T9D_RAMP)
            ik_targets[h] = [(int(f), tgt, float(wh[f])) for f in np.nonzero(wh > 0)[0]]
    for side in ('Left', 'Right'):
        foot, toe = side + 'Foot', side + 'FootThumb1'
        need = np.zeros(len(F))
        for f in win:
            best = 0.0
            for name, off in ((foot, offsets['Foot']), (toe, offsets['FootThumb1'])):
                y, z = P1[f, BD._acu(rig, name), 1], P1[f, BD._acu(rig, name), 2]
                surf = float(ob['height']) if float(ob['near_y']) <= y <= float(ob['far_y']) else 0.0
                best = max(best, surf + off - z)
            need[f] = best
        pos_idx = np.nonzero(need > 0)[0]
        if pos_idx.size == 0:
            continue
        wf = bake2.blend_mod_fade(len(F), bake2._runs(pos_idx), bake2.T9D_RAMP)
        for f in np.nonzero(wf > 0)[0]:
            k = pos_idx[np.argmin(np.abs(pos_idx - f))]
            tgt = P1[f, BD._acu(rig, foot)] + np.array([0.0, 0.0, need[k]])
            ik_targets.setdefault(foot, []).append((int(f), tgt, float(wf[f])))
    if ik_targets:
        root_rot = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(F), 1))
        ik_out = ik.solve_sequence(ik_in, root_rot=root_rot, root_pos=disp, targets=ik_targets)
        changed = np.nonzero(np.any(np.abs(ik_out.local_rot - ik_in.local_rot) > 1e-12, axis=(0, 2)))[0]
        bad = sorted(set(int(c) for c in changed) - set(patset))
        if bad:
            raise ValueError('IK changed bones without a rotation track: %s' % bad)
        patch_ik = np.asarray(ik_out.local_rot, float)
    else:
        patch_ik = patch
    final = orig.copy()
    for ai in patset:
        final[:, ai] = bake2.mix_rot(orig[:, ai], patch_ik[:, ai], wgt)
        final[:, ai] /= np.linalg.norm(final[:, ai], axis=1, keepdims=True)
    lo = max(0, int(win[0]) - 1)
    hi = nf
    rot_tracks = {}
    for ai in patset:
        ok = dumps.track(dump, rig.ac[ai]['hash'], 0) or []
        rot_tracks[ai] = bake2.rotation_track(ok, final[:, ai], lo, hi)
    ref_keys = dumps.track(dump, bake2.REF_TID, 3) or []
    ref_out = ref1 - np.array([0.0, 0.0, seg.get('z_shift', 0.0)])
    trans_keys = bake2.translation_track(ref_keys, ref_out, lo, hi)
    return dict(final=final, ref_new=ref_out, rot_tracks=rot_tracks, trans_keys=trans_keys,
                delta=np.stack([np.zeros(len(F)), dy_full, dz_full], axis=1))


def _res_for_validation(rig, seg, fin, win, plant, targets, ob, o, span, t_end, tail=None):
    body_all, fingers, body_nf, body_nh = _body_sets(rig)
    return dict(ob=ob, patched=seg['patset'], rot_tracks=fin['rot_tracks'], trans_keys=fin['trans_keys'],
                ref0=seg['ref0'], orig=seg['orig'], win=win, body=body_all, body_nf=body_nf, fingers=sorted(fingers),
                plant=plant, targets=targets, o=o, span=span, t_end=t_end, final=fin['final'], ref_new=fin['ref_new'])


def validate_segment(rig, seg, fin, win, plant, targets, ob, o, span, t_end, next_dump=None, is_exit=False):
    res = _res_for_validation(rig, seg, fin, win, plant, targets, ob, o, span, t_end)
    met, arr = BD.validate_dive(rig, seg['dump'], seg['seq'], res, None, next_dump=next_dump,
                                z_shift=seg.get('z_shift', 0.0))
    if is_exit:
        met['fails'] = [f for f in met['fails'] if not f.startswith('hand touch')]
        met['fail'] = bool(met['fails'])
    return met, arr, res


# ------------------------------------------------------------------------------------------------ 4. build the joined chain
def build_chain(rig, m, seq_so, dump_so, ev_so, mp_so, seq_ex, dump_ex, offsets, p_to=None, T=None,
                disp_so=None, disp_ex=None, z_shift_ex=EXIT_Z_SHIFT):
    """Joined SlideOver (main) + Exit build. Returns dict with the two clips' results and the metrics.

    T: fixed plant start (played); None runs the T search. disp_so / disp_ex: the root each clip plays with (None: the
    dumps' own TrackID 0). z_shift_ex: the Exit Reference shift (the root-delta convention of dive_chain2 uses 0)."""
    tk = float(ev_so['takeoff']['seconds'])
    if p_to is None:
        p_to = (tk - mp_so['o']) / mp_so['r']
    ctx_T = st_T = None
    if T is None:
        T, ctx_T, st_T = find_T(rig, m, seq_so, dump_so, ev_so, mp_so, offsets, p_to, mp_so['W'])
    if T is None:
        raise RuntimeError('no feasible plant start on the grid %s' % T_GRID[[0, -1]])
    tl = Timeline(mp_so, p_to, mp_so['W'], T)
    ob = ev_so['obstacle']
    span_so = mp_so['r'] * mp_so['W']
    seg_S = segment(rig, m, seq_so, dump_so, tl.w_so, tl.w_so(round(tl.o * FPS) / FPS),
                    lambda t: bake2.patch_weight(t, tl.o, span_so, BD.FI, BD.FO, BD.TAIL), disp=disp_so)
    seg_E = segment(rig, m, seq_ex, dump_ex, tl.w_exit, tl.w_so(round(tl.o * FPS) / FPS),
                    lambda t: (np.asarray(t, float) >= EXIT_O - 1e-9).astype(float), z_shift=z_shift_ex, disp=disp_ex)
    seg_S['win'] = bake2.window_frames(tl.o, span_so, int(seg_S['F'][-1]))
    span_e = EXIT_R * EXIT_WIN
    seg_E['win'] = bake2.window_frames(EXIT_O, span_e, int(seg_E['F'][-1]))
    ob_e = _ob_ground()
    seg_S['ob'] = ob
    seg_E['ob'] = ob_e
    chosen, n_stages = choose_joined(rig, seg_S, seg_E, tl, offsets, ob)
    A = chosen['A']
    dyS = A * BD.bump(tl.J_so(seg_S['t60']), 0.0, tl.T, tl.J_end)
    dyE = A * BD.bump(tl.J_exit(seg_E['t60']), 0.0, tl.T, tl.J_end)
    dzS = np.zeros(len(seg_S['F']))
    dzS[seg_S['win']] = chosen['DS']
    dzE = np.zeros(len(seg_E['F']))
    dzE[seg_E['win']] = chosen['DE']
    # plant targets and window, SlideOver
    w_win = tl.w_so(seg_S['t60'][seg_S['win']])
    plant_S = seg_S['win'][(w_win >= BD.WD2_PLANT[0] - 1e-9) & (w_win <= BD.WD2_PLANT[1] + 1e-9)]
    fin_S = finalize(rig, seg_S, dyS, dzS, seg_S['win'], plant_S, chosen['targets'], ob, offsets)
    fin_E = finalize(rig, seg_E, dyE, dzE, seg_E['win'], np.array([], int), {}, ob_e, offsets)
    return dict(T=T, tl=tl, A=A, chosen=chosen, n_stages=n_stages, st_T=st_T, seg_S=seg_S, seg_E=seg_E,
                fin_S=fin_S, fin_E=fin_E, plant_S=plant_S, dyS=dyS, dyE=dyE, dzS=dzS, dzE=dzE, ob=ob, ob_e=ob_e,
                span_so=span_so, span_e=span_e)


# ------------------------------------------------------------------------------------------------ 5. validation and joined metrics
def continuity(rig, S, fS, E, fE, tl):
    """Gap across the boundary: our SlideOver tail at clip time o + r * W against our Exit at its window start (both at
    the nearest 60 Hz frame). Max / median local-rotation gap on CONT_BONES (deg), the Hips offset relative to each clip's
    own entity displacement (cm), and the per-bone angles."""
    fW = min(int(round((tl.o + tl.r * tl.W) * FPS)), int(S['F'][-1]))
    fE0 = min(max(int(round(EXIT_O * FPS)), int(E['F'][0])), int(E['F'][-1]))
    ang = {}
    for bn in CONT_BONES:
        ai = BD._acu(rig, bn)
        ang[bn] = float(bake2.angle_deg(fS['final'][fW, ai], fE['final'][fE0, ai]))
    vals = np.array(list(ang.values()))
    hips = BD._acu(rig, 'Hips')
    zS, zE = S.get('z_shift', 0.0), E.get('z_shift', 0.0)
    PS = BD.world(S['seq'], S['dump'], fS['final'], fS['ref_new'] + np.array([0, 0, zS]), 0.0, 0.0, disp=S['disp'])
    PE = BD.world(E['seq'], E['dump'], fE['final'], fE['ref_new'] + np.array([0, 0, zE]), 0.0, 0.0, disp=E['disp'])
    dS = S['disp'][fW]
    dE = E['disp'][fE0]
    hips_off = (PS[fW, hips] - dS) - (PE[fE0, hips] - dE)
    return dict(max_deg=float(vals.max()), median_deg=float(np.median(vals)), worst=max(ang, key=lambda k: ang[k]),
                per_bone=ang, hips_cm=float(np.linalg.norm(hips_off) * 100.0), fW=fW, fE0=fE0)


def joined_root_step_cm(S, fS, E, fE):
    """Largest per-frame change of the offset across the joined sequence (SlideOver window, then Exit window), cm."""
    d = np.concatenate([fS['delta'][S['win']], fE['delta'][E['win']]])
    return float(np.max(np.linalg.norm(np.diff(d, axis=0), axis=1)) * 100.0) if len(d) > 1 else 0.0


def ease_exit(rig, E, fE, next_dump, t_end):
    """Eased Exit: over the last EASE_S of its window the patched local rotations go to the next clip's first frame."""
    fW = min(int(round(t_end * FPS)), int(E['F'][-1]))
    z0 = max(0, fW - int(round(EASE_S * FPS)))
    fin = fE['final'].copy()
    frames_z = np.arange(z0, fW + 1)
    we = bake2._smoothstep((frames_z - z0) / float(max(1, fW - z0)))
    for ai in E['patset']:
        trk = dumps.track(next_dump, rig.ac[ai]['hash'], 0)
        target = dumps.sample_rotation(trk, 0) if trk else R.qn(rig.ac[ai]['lrot'])
        fin[frames_z, ai] = bake2.mix_rot(fin[frames_z, ai], np.tile(target, (len(frames_z), 1)), we)
    return fin


def retrack(rig, seg, fin_rot, ref_new, win):
    """Rotation and translation tracks for a (possibly eased) final pose."""
    dump = seg['dump']
    nf = int(seg['F'][-1])
    lo = max(0, int(win[0]) - 1)
    rot_tracks = {}
    for ai in seg['patset']:
        ok = dumps.track(dump, rig.ac[ai]['hash'], 0) or []
        rot_tracks[ai] = bake2.rotation_track(ok, fin_rot[:, ai], lo, nf)
    ref_keys = dumps.track(dump, bake2.REF_TID, 3) or []
    return rot_tracks, bake2.translation_track(ref_keys, ref_new, lo, nf)


def _seg_result(rig, seg, fin, win, plant, targets, ob, o, span, t_end, next_dump=None, is_exit=False):
    """validate_dive on the written tracks of one clip (the Exit has no hands: its hand-touch rule is dropped)."""
    body_all = sorted(set(rig.map.values()))
    fingers = bake2.finger_bones(rig, body_all)
    res = dict(ob=ob, patched=seg['patset'], rot_tracks=fin['rot_tracks'], trans_keys=fin['trans_keys'],
               ref0=seg['ref0_orig'], orig=seg['orig'], win=win, body=body_all, fingers=sorted(fingers),
               body_nf=[b for b in body_all if b not in set(fingers)], plant=plant, targets=targets,
               o=o, span=span, t_end=t_end, disp=seg.get('disp'), disp_orig=seg.get('disp_orig'))
    met, arr = BD.validate_dive(rig, seg['dump'], seg['seq'], res, None, next_dump=next_dump,
                                z_shift=seg.get('z_shift', 0.0))
    if is_exit:
        met['fails'] = [f for f in met['fails'] if not f.startswith('hand touch')]
        met['fail'] = bool(met['fail'] and met['fails'])
    return met, arr, res


def so_fail_list(met, cont, step_cm):
    fails = list(met['fails'])
    if cont['max_deg'] > CONT_DEG:
        fails.append('continuity at W %.0f deg (> %.0f)' % (cont['max_deg'], CONT_DEG))
    if step_cm > STEP_CM:
        fails.append('joined root step %.1f cm/frame' % step_cm)
    return fails


# ------------------------------------------------------------------------------------------------ 6. the main chain
def run_main(rig, data):
    """Validate the main SlideOver + Exit build; ease the Exit toward Nav_Run only when its gap is large."""
    S, E, fS, fE = data['seg_S'], data['seg_E'], data['fin_S'], data['fin_E']
    tl, ob, ob_e = data['tl'], data['ob'], data['ob_e']
    t_so_end = tl.o + tl.r * tl.W
    t_e_end = EXIT_O + EXIT_R * EXIT_WIN
    nxt = dumps.load_by_name(NEXT_NAME)
    met_S, arr_S, res_S = _seg_result(rig, S, fS, S['win'], data['plant_S'], data['chosen']['targets'], ob,
                                      tl.o, data['span_so'], t_so_end, None)
    met_E, arr_E, res_E = _seg_result(rig, E, fE, E['win'], np.array([], int), {}, ob_e, EXIT_O,
                                      data['span_e'], t_e_end, nxt, is_exit=True)
    gap_noease = met_E['gap']
    ease_used = False
    if gap_noease['max_deg'] is not None and gap_noease['max_deg'] > GAP_WARN:
        fin_eased = ease_exit(rig, E, fE, nxt, t_e_end)
        rot_tracks, trans = retrack(rig, E, fin_eased, fE['ref_new'], E['win'])
        fE = dict(fE, final=fin_eased, rot_tracks=rot_tracks, trans_keys=trans)
        met_E, arr_E, res_E = _seg_result(rig, E, fE, E['win'], np.array([], int), {}, ob_e, EXIT_O,
                                          data['span_e'], t_e_end, nxt, is_exit=True)
        ease_used = True
    cont = continuity(rig, S, fS, E, fE, tl)
    step = joined_root_step_cm(S, fS, E, fE)
    return dict(met_S=met_S, arr_S=arr_S, res_S=res_S, met_E=met_E, arr_E=arr_E, res_E=res_E,
                fS=fS, fE=fE, cont=cont, step_cm=step, ease_used=ease_used, gap_noease=gap_noease)


def so_only_build(rig, m, seq, dump, ev, mp, offsets):
    """SlideOver-only build for an extra obstacle: its own T search and root (no joined solve). The shared Exit is not
    re-solved; its continuity with this SlideOver is checked in the report."""
    tk = float(ev['takeoff']['seconds'])
    p_to = (tk - mp['o']) / mp['r']
    T, ctx, st = find_T(rig, m, seq, dump, ev, mp, offsets, p_to, mp['W'])
    if T is None:
        return dict(T=None)
    tl = Timeline(mp, p_to, mp['W'], T)
    span = mp['r'] * mp['W']
    seg = segment(rig, m, seq, dump, tl.w_so, tl.w_so(round(tl.o * FPS) / FPS),
                  lambda t: bake2.patch_weight(t, tl.o, span, BD.FI, BD.FO, BD.TAIL))
    seg['win'] = bake2.window_frames(tl.o, span, int(seg['F'][-1]))
    seg['ob'] = ev['obstacle']
    stages = []
    chosen = None
    for A in BD.DY_GRID:
        s2 = BD._forward_stage(ctx, A)
        stages.append(s2)
        if s2['feasible']:
            chosen = s2
            break
    if chosen is None:
        chosen = min(stages, key=lambda x: (x['conflicts'], x['A']))
    dy = chosen['A'] * ctx['B']
    dz = np.zeros(len(seg['F']))
    dz[seg['win']] = chosen['D_w']
    wwin = tl.w_so(seg['t60'][seg['win']])
    plant = seg['win'][(wwin >= BD.WD2_PLANT[0] - 1e-9) & (wwin <= BD.WD2_PLANT[1] + 1e-9)]
    fin = finalize(rig, seg, dy, dz, seg['win'], plant, chosen['targets'], ev['obstacle'], offsets)
    return dict(T=T, tl=tl, A=chosen['A'], chosen=chosen, n_stages=len(stages), seg=seg, fin=fin, plant=plant,
                ob=ev['obstacle'], span=span, dy=dy, dz=dz)


# ------------------------------------------------------------------------------------------------ 7. plots
def _rel(seg, arr, t_ref):
    """Entity-relative world positions (entity displacement at clip time t_ref removed): (Pf, Po, frame, disp_y)."""
    F = seg['F']
    f = min(int(round(t_ref * FPS)), int(F[-1]))
    disp = seg['disp'][f]
    return arr['Pf'] - disp[None, None, :], arr['Po'] - disp[None, None, :], f, float(disp[1])


def _ob_rel(ob, disp_y_w):
    """Obstacle near edge in the entity-relative frame of the boundary (the entity at W is at y = 0)."""
    return float(ob['near_y']) - disp_y_w


def plot_joined(path, rig, S, arrS, E, arrE, ob, tl, title_extra, n_fig=6, show_root=False):
    """Side view over the SlideOver and the Exit played windows, entity-relative (the boundary is at y = 0).

    show_root: also draw the entity root path (its displacement relative to the entity at W, dashed)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    PfS, PoS, fWs, dyW = _rel(S, arrS, tl.o + tl.r * tl.W)
    PfE, PoE, fEs, _dyE = _rel(E, arrE, EXIT_O)
    body = sorted(set(rig.map.values()))
    bset = set(body)
    edges = [(ai, rig.ac[ai]['parent']) for ai in body if rig.ac[ai]['parent'] in bset]
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=100)
    near = _ob_rel(ob, dyW)
    ax.add_patch(Rectangle((near, 0.0), float(ob['far_y'] - ob['near_y']), float(ob['height']), facecolor='#8b6f47',
                           alpha=0.35, edgecolor='#5c4a2e', label='obstacle'))
    ax.axhline(0.0, color='#777', lw=0.8)
    ax.axvline(0.0, color='k', ls=':', lw=1.0)
    tS = np.linspace(tl.o + tl.r * tl.T - 0.10, tl.o + tl.r * tl.W, n_fig)
    tE = np.linspace(EXIT_O, EXIT_O + EXIT_R * EXIT_WIN, n_fig)
    for seg_F, Pf, Po, times, col, lab in ((S['F'], PfS, PoS, tS, 'tab:blue', 'SlideOver'),
                                           (E['F'], PfE, PoE, tE, 'tab:orange', 'Exit')):
        idxs = [int(np.argmin(np.abs(seg_F / FPS - t))) for t in times]
        for k, fi in enumerate(idxs):
            for a, p in edges:
                ax.plot([Po[fi, a, 1], Po[fi, p, 1]], [Po[fi, a, 2], Po[fi, p, 2]], color='#c9c9c9', lw=1.0, zorder=1)
            for a, p in edges:
                ax.plot([Pf[fi, a, 1], Pf[fi, p, 1]], [Pf[fi, a, 2], Pf[fi, p, 2]], color=col, lw=2.0,
                        alpha=0.35 + 0.65 * k / max(1, len(idxs) - 1), zorder=2)
        ax.plot([], [], color=col, lw=2.0, label=lab + ' (patched)')
    ax.plot([], [], color='#c9c9c9', lw=1.0, label='original')
    if show_root:
        for seg_, col_, dspan in ((S, 'tab:blue', (tl.o, tl.o + tl.r * tl.W)), (E, 'tab:orange', (EXIT_O, EXIT_O + EXIT_R * EXIT_WIN))):
            fr = np.asarray(seg_['F'])
            tt = fr / FPS
            sel = (tt >= dspan[0] - 1e-9) & (tt <= dspan[1] + 1e-9)
            d = np.asarray(seg_['disp'], float)
            ax.plot(d[sel, 1] - dyW, d[sel, 2], '--', color=col_, lw=1.2, label='%s root path' % ('SlideOver' if seg_ is S else 'Exit'))
        ax.plot([], [], 'k--', lw=1.0, label='entity root (origin)')
    ax.set_xlabel('entity forward Y, relative to the entity at W (m)')
    ax.set_ylabel('Z (m)')
    ax.set_xlim(near - 1.5, near + 3.0)
    ax.set_ylim(-0.1, 2.4)
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_title('%s\nblue: SlideOver (last 0.10 s before its plant to W); dotted line: boundary W; orange: Exit window\n%s'
                 % (S.get('name', ''), title_extra), fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_gif(path, rig, S, arrS, E, arrE, ob, tl):
    """Joined animation, 20 fps, entity-relative (the same frame as the PNG): SlideOver frames, then the Exit frames."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter
    from matplotlib.patches import Rectangle
    PfS, PoS, fWs, dyW = _rel(S, arrS, tl.o + tl.r * tl.W)
    PfE, PoE, fEs, _dyE = _rel(E, arrE, EXIT_O)
    body = sorted(set(rig.map.values()))
    bset = set(body)
    edges = [(ai, rig.ac[ai]['parent']) for ai in body if rig.ac[ai]['parent'] in bset]
    near = _ob_rel(ob, dyW)
    frames = [('S', f) for f in range(int(round(tl.o * FPS)), fWs + 1, 3)] + \
             [('E', f) for f in range(int(round(EXIT_O * FPS)), fEs + 1, 3)]
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=80)
    writer = PillowWriter(fps=20)
    with writer.saving(fig, path, dpi=80):
        for kind, f in frames:
            ax.cla()
            Pf, Po = (PfS, PoS) if kind == 'S' else (PfE, PoE)
            col = 'tab:blue' if kind == 'S' else 'tab:orange'
            ax.add_patch(Rectangle((near, 0.0), float(ob['far_y'] - ob['near_y']), float(ob['height']), facecolor='#8b6f47',
                                   alpha=0.35, edgecolor='#5c4a2e'))
            ax.axhline(0.0, color='#777', lw=0.8)
            for a, p in edges:
                ax.plot([Po[f, a, 1], Po[f, p, 1]], [Po[f, a, 2], Po[f, p, 2]], color='#c9c9c9', lw=1.4)
            for a, p in edges:
                ax.plot([Pf[f, a, 1], Pf[f, p, 1]], [Pf[f, a, 2], Pf[f, p, 2]], color=col, lw=2.0)
            ax.set_xlim(near - 1.5, near + 3.0)
            ax.set_ylim(-0.1, 2.4)
            ax.set_aspect('equal')
            tag = 'SlideOver' if kind == 'S' else 'Exit'
            wd2 = float(tl.w_so(f / FPS)) if kind == 'S' else float(tl.w_exit(f / FPS))
            ax.set_title('%s  clip %.2f s (WD2 %.2f s)%s' % (tag, f / FPS, wd2,
                                                           '  [boundary W]' if kind == 'E' and f == int(round(EXIT_O * FPS)) else ''),
                         fontsize=7)
            writer.grab_frame()
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ 8. outputs
def write_report(rows, exit_row, offsets, T_info, notes):
    lines = ['# T12b: WD2 frontdive as SlideOver + gated Muraille_Exit (joined timeline)', '',
             'Generated by `v2/pipeline/bake_chain.py`. Times in played seconds (SlideOver from its start, Exit from the '
             'SlideOver W) unless marked clip; heights in cm; angles in degrees.', '',
             '## Time map', '',
             '- WD2 = 0.45 at the ACU takeoff (played %.3f). Monotone cubic Hermite (slope 1 at both ends) to WD2 0.75 at T '
             '(plant start). From T on, WD2 runs at k = 1 (WD2 = 0.75 + (J - T)), through the roll, the boundary and the Exit.'
             % T_info['p_to'],
             '- T (earliest feasible plant start on a 0.02 s grid from %.2f): %s. k_flight over takeoff..T (WD2 s per played '
             's, mean) = %s.'
             % (T_GRID[0], ('%.2f played' % T_info['T']) if T_info['T'] else 'none feasible',
                ('%.2f' % T_info['k_flight']) if T_info['T'] else 'n/a'),
             '- Exit: clip time = %.3f + %.4f * p_e; window %.3f s (median); joined end at W + %.3f.'
             % (EXIT_O, EXIT_R, EXIT_WIN, EXIT_WIN), '',
             '## Validation', '',
             '| clip | hex | forward A (m) | T (played) | max Dz (cm) | non-finger pen box / ground (cm) | below ground (cm) | '
             'hand touch L/R (fr, need 4) | max Delta step (cm/fr) | continuity at W max / median (deg) | joined root step (cm/fr) | '
             'status | enabled |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        met, cont = r['met'], r['cont']
        tc = met['touch']
        lines.append('| %s | %s | %.2f | %s | %.1f | %.1f / %.1f | %.1f | %d / %d | %.2f | %s | %.2f | %s | %s |'
                     % (r['name'].replace('xx_Parkour_Sprint_', ''), r['hex'], r['A'],
                        ('%.2f' % r['T']) if r['T'] else '-', met['dz_max_cm'], met['pen_box_cm'], met['pen_ground_cm'],
                        met['ground_all_cm'], tc.get('LeftHand', 0), tc.get('RightHand', 0), met['dstep_max_cm'],
                        ('%.1f / %.1f' % (cont['max_deg'], cont['median_deg'])) if r['T'] else '-', r['step_cm'],
                        'FAIL' if r['fails'] else 'PASS', 'yes' if r['enabled'] else 'no'))
    em = exit_row['met']
    g = em['gap']
    lines += ['', 'Exit (one shared patch, built with the main SlideOver): forward A %.2f m; max Dz %.1f cm; below ground %.1f cm; '
              'max Delta step %.2f cm/fr; non-finger penetration %.1f cm; gap to Nav_Run at the Exit end max / median %s deg; '
              'ease toward Nav_Run applied: %s.'
              % (exit_row['A'], em['dz_max_cm'], em['ground_all_cm'], em['dstep_max_cm'], em['pen_patch_cm'],
                 ('%.1f / %.1f' % (g['max_deg'], g['median_deg'])) if g['max_deg'] is not None else 'n/a',
                 'yes' if exit_row['ease'] else 'no'), '']
    lines += ['## Failure reasons', '']
    for r in rows:
        lines.append('- %s: %s' % (r['name'], '; '.join(r['fails']) if r['fails'] else 'none (PASS)'))
    lines.append('- Exit (%s): %s' % (EXIT_NAME, '; '.join(exit_row['fails']) if exit_row['fails'] else 'none (PASS)'))
    lines += ['', '## Notes', '']
    lines += ['- ' + n for n in notes]
    lines += ['', 'Offsets (median over the Sprint SlideOver dumps): Hand %.3f m, Foot %.3f m, FootThumb1 %.3f m.'
              % (offsets['Hand'], offsets['Foot'], offsets['FootThumb1']), '']
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ------------------------------------------------------------------------------------------------ 9. driver
def main():
    os.makedirs(PATCH_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    rig = R.Rig()
    m = wd2.load(BD.load_dive_events()['stem'])
    model = BD.load_play_model_so()
    clips = {}
    for handle, name, dump in BD.targets_dive():
        seq = acuclip.pose_sequence(dump, step=1)
        clips[name] = dict(handle=handle, dump=dump, seq=seq, ev=contacts.events(seq, dump),
                           mp=BD.clip_model(name, model, float(dump['Length'])))
    offsets = bake2.measure_contact_offsets([(n, c['dump'], c['seq'], c['ev']) for n, c in clips.items()])
    ex_dump = dumps.load_by_name(EXIT_NAME)
    ex_seq = acuclip.pose_sequence(ex_dump, step=1)

    main_c = clips[MAIN_SO]
    data = build_chain(rig, m, main_c['seq'], main_c['dump'], main_c['ev'], main_c['mp'], ex_seq, ex_dump, offsets)
    rm = run_main(rig, data)
    tl = data['tl']
    T_main = float(data['T'])
    k_flight = (BD.WD2_PLANT[0] - BD.WD2_TAKEOFF) / (T_main - tl.p_to)
    notes = []

    rows = []
    fails = so_fail_list(rm['met_S'], rm['cont'], rm['step_cm'])
    rows.append(dict(name=MAIN_SO, hex=main_c['handle'], met=rm['met_S'], A=data['A'], T=T_main, cont=rm['cont'],
                     step_cm=rm['step_cm'], fails=fails, enabled=not fails, fin=rm['fS'], clip=main_c))

    for name in SO_SCOPE[1:]:
        c = clips[name]
        if REUSE_EXTRA_SEARCH is not None:
            b = dict(T=None)
        else:
            b = so_only_build(rig, m, c['seq'], c['dump'], c['ev'], c['mp'], offsets)
        if b['T'] is None:
            rows.append(dict(name=name, hex=c['handle'], met=dict(fails=[], dz_max_cm=0, pen_box_cm=0, pen_ground_cm=0,
                                                                  ground_all_cm=0, dstep_max_cm=0, touch={}, gap={}),
                             A=0.0, T=None, cont=dict(max_deg=float('nan'), median_deg=float('nan')), step_cm=0.0,
                             fails=['no feasible plant start on the T grid'], enabled=False, fin=None, clip=c))
            notes.append('%s: no feasible plant start on the T grid, not baked.' % name)
            continue
        met, arr, res = _so_only_result(rig, b)
        cont = continuity(rig, b['seg'], b['fin'], data['seg_E'], rm['fE'], b['tl'])
        step = float(np.max(np.linalg.norm(np.diff(b['fin']['delta'][b['seg']['win']], axis=0), axis=1)) * 100.0)
        fails = so_fail_list(met, cont, step)
        rows.append(dict(name=name, hex=c['handle'], met=met, A=b['A'], T=b['T'], cont=cont, step_cm=step,
                         fails=fails, enabled=not fails, fin=b['fin'], clip=c, build=b, arr=arr))

    any_enabled = any(r['enabled'] for r in rows)
    exit_fails = list(rm['met_E']['fails'])
    exit_row = dict(A=data['A'], met=rm['met_E'], fails=exit_fails, ease=rm['ease_used'],
                    enabled=(not exit_fails) and any_enabled)

    # ---- patches: the in-scope SlideOvers and the Exit only; stale files from the earlier design are removed
    keep = {r['clip']['handle'] + '.baked.json' for r in rows} | {EXIT_HEX + '.baked.json'}
    for fn in os.listdir(PATCH_DIR):
        if fn.endswith('.baked.json') and fn not in keep:
            os.remove(os.path.join(PATCH_DIR, fn))
    mapping = []
    for r in rows:
        if r['fin'] is None:
            continue
        fn = r['clip']['handle'] + '.baked.json'
        bake2.write_patch_t9c(os.path.join(PATCH_DIR, fn), rig, r['fin']['rot_tracks'], r['fin']['trans_keys'],
                              r['clip']['dump']['Length'])
        mapping.append(bake2.mapping_entry_t9c(r['clip']['handle'], r['name'], fn, r['enabled']))
    bake2.write_patch_t9c(os.path.join(PATCH_DIR, EXIT_HEX + '.baked.json'), rig, rm['fE']['rot_tracks'],
                          rm['fE']['trans_keys'], ex_dump['Length'])
    only = [r['name'] for r in rows if r['enabled']]
    ex_entry = bake2.mapping_entry_t9c(EXIT_HEX, EXIT_NAME, EXIT_HEX + '.baked.json', exit_row['enabled'])
    ex_entry['only_after'] = only
    ex_entry['arm_seconds'] = 2.0
    mapping.append(ex_entry)
    with open(MAPPING, 'w', encoding='utf-8') as f:
        json.dump({'clips': mapping}, f, indent=1)

    # ---- plots (main joined, and the two extra SlideOvers against the shared Exit)
    arrS, arrE = rm['arr_S'], rm['arr_E']
    title = 'main: %s, T %.2f, A %.2f m, continuity %.0f deg' % (MAIN_SO, T_main, data['A'], rm['cont']['max_deg'])
    plot_joined(os.path.join(OUT_DIR, MAIN_SO + '_joined.png'), rig, data['seg_S'] | {'name': MAIN_SO}, arrS,
                data['seg_E'], arrE, data['ob'], tl, title)
    write_gif(os.path.join(OUT_DIR, MAIN_SO + '_joined.gif'), rig, data['seg_S'], arrS, data['seg_E'], arrE, data['ob'], tl)
    write_report(rows, exit_row, offsets, dict(p_to=tl.p_to, T=T_main, k_flight=k_flight), notes)
    summary = dict(T=T_main, k_flight=k_flight, p_to=float(tl.p_to),
                   slideovers=[dict(name=r['name'], enabled=r['enabled'], fails=r['fails'],
                                    continuity_deg=(float(r['cont']['max_deg']) if r['T'] else None),
                                    T=r['T'], A=float(r['A'])) for r in rows],
                   exit=dict(enabled=exit_row['enabled'], fails=exit_row['fails'], ease=exit_row['ease'],
                             gap_max_deg=(exit_row['met']['gap']['max_deg'])))
    with open(os.path.join(OUT_DIR, 'chain_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, default=float)
    return rows, exit_row, rm, data, offsets


def _so_only_result(rig, b):
    """validate_dive for an extra SlideOver build (its own window, no next clip; the continuity is checked separately)."""
    met, arr, res = _seg_result(rig, b['seg'], b['fin'], b['seg']['win'], b['plant'], b['chosen']['targets'],
                                b['ob'], b['tl'].o, b['span'], b['tl'].o + b['tl'].r * b['tl'].W, None)
    return met, arr, res


if __name__ == '__main__':
    main()
