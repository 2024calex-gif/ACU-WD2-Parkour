"""T15 (dive4): the WD2 frontdive body path placed on AC Unity's LOW-wall SlideOver xx_Parkour_Sprint_SlideOver_Fd100cm-Up050cm
(dump 7FC9C7405, obstacle 0.5 m high), with the Muraille_Exit Fd100cm-Dn100cm hand-off (F7919BDF0, gated).

Changes from dive3 (the brief's fixes 1..6):
  1. Rotations are the ABSOLUTE WD2 orientations (world = A_REF * Wt, bake_dive.ROT_ABSOLUTE). The Reference rotation is
     patched. Fade-in from the original pose over FI = 0.15 s at o, no fade-out (tail wd2). The Reference translation
     uses the same 0.15 s ramp.
  2. Placement: a scan of the scale s in [0.9, 1.5] and a forward shift in [0, 0.4] m, with the hand rule (closer hand at
     or beyond the far edge + 0.20 m). The takeoff foot is ANCHORED at the measured ACU takeoff foot (LeftFootThumb1 at
     the takeoff event), so the WD2 stance offset (0.652 m behind the takeoff pelvis) scales with s. The cost is
     shift^2 + 0.5 (s-1)^2 + 0.5 dW^2 (dW = pelvis offset at W). Scan table in scan.csv / scan.md.
  3. Clearance: a raised-cosine bump on the Reference z from WD2 0.35 s (0.1 s before the takeoff) to the plant start
     (0.75 s); minimum amplitude that clears the non-finger body by 5 cm over the box top.
  4. Timing: k = 1. WD2 time w(t) = 0.45 + (t - T_A) with T_A the ACU takeoff (dump event, 0.4667 clip s). W = o + r * W_played
     with the capture window (0.6305 played s, one occurrence, session4 cap_20261008_201039_19.json). The capture shows the
     Exit start Fd100cm-Dn500cm (F7919BE5E) replaced 14 us later by Fd100cm-Dn100cm (F7919BDF0), which plays. So the gated
     Exit is F7919BDF0 with the dive3 Exit time map (o 0.208, r 0.9665).
  5. The Exit continues WD2 at k = 1 from w(W). Its Reference offset converges to the original across the Exit window and the
     remaining clip (smoothstep, per-frame step checked). Ease to Nav_Run_3-Loop's first frame over the last 0.1 s of the
     window only when the pre-ease gap is above 45 deg.
  6. Hand IK pins the hands during the plant; the foot clamp lifts feet after the feet-down (dive3's helpers).

Validation (as dive3): non-finger penetration <= 5 cm, hand touch >= 4 frames, Reference step <= 4 cm/frame,
continuity at W <= 45 deg. The narrow fail (<= 10 cm, >= 2 frames, <= 8 cm/frame, <= 60 deg) enables with an override.

Outputs (only under out\\dive4):
  patches\\7FC9C7405.baked.json   rotations incl. the Reference, Reference translation; NO TrackID 0
  patches\\F7919BDF0.baked.json   Exit: rotations and Reference translation
  patches\\mapping_dive4.json     SlideOver + gated Exit (only_after SlideOver, arm 2.0 s), prebaked, patch_translation
  joined_sheet.png / joined_sheet.gif, scan.csv, scan.md, report.md, summary.json
Run:  python dive4.py            (writes out\\dive4)
      python dive4.py --scan     (prints the scan summary, writes nothing)
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
import contacts
import dive3 as D3
import dive_chain2 as DC2
import dumps
import ik
import retarget as R
import wd2
import wd2_path as WP

FPS = acuclip.FPS
OUT_DIR = os.path.join(config.OUT, 'dive4')
PATCH_DIR = os.path.join(OUT_DIR, 'patches')
MAPPING = os.path.join(PATCH_DIR, 'mapping_dive4.json')
REPORT = os.path.join(OUT_DIR, 'report.md')
SCAN_CSV = os.path.join(OUT_DIR, 'scan.csv')
SCAN_MD = os.path.join(OUT_DIR, 'scan.md')
SUMMARY = os.path.join(OUT_DIR, 'summary.json')
SHEET_PNG = os.path.join(OUT_DIR, 'joined_sheet.png')
SHEET_GIF = os.path.join(OUT_DIR, 'joined_sheet.gif')
WD2_SIDE_PNG = os.path.join(OUT_DIR, 'wd2_path_side.png')

SO_NAME = 'xx_Parkour_Sprint_SlideOver_Fd100cm-Up050cm'
SO_HEX = '7FC9C7405'
EXIT_NAME = BC.EXIT_NAME                  # xx_Parkour_Muraille_Exit_Fd100cm-Dn100cm (played, see the capture check)
EXIT_HEX = BC.EXIT_HEX                    # F7919BDF0
NEXT_NAME = BC.NEXT_NAME                  # xx_Nav_Run_3-Loop
EXIT_O = BC.EXIT_O                        # 0.208 clip s: the Exit played window start
EXIT_R = BC.EXIT_R                        # 0.9665 clip s per played s
EXIT_WIN = BC.EXIT_WIN                    # 0.228 played s (median)
FI = 0.15                                 # s, rotation fade-in from the original pose at o (fix 1)
TFI = 0.15                                # s, Reference translation ramp (same as the rotations)
FO = 0.0
TAIL = bake2.TAIL_WD2
EASE_S = 0.10                             # s, ease to Nav_Run over the last part of the Exit window
GAP_DEG = 45.0
CONT_DEG = D3.CONT_DEG
STEP_CM = D3.STEP_CM
PEN_CM = D3.PEN_CM
TOUCH = D3.TOUCH
NARROW = D3.NARROW
S_GRID = np.round(np.arange(0.90, 1.5001, 0.01), 2)
SHIFT_GRID = np.round(np.arange(0.0, 0.4001, 0.01), 2)
FAR_CLEAR = 0.20                          # m, the closer hand plants at or beyond the far edge + this
BUMP_T0, BUMP_T1 = 0.35, 0.75             # WD2 s: 0.1 s before the takeoff up to the plant start
BUMP_CLEAR = 0.05                         # m, clearance of the non-finger body above the box top
COST_W_S = 0.5                            # weight of (s - 1)^2 in the cost
COST_W_W = 0.5                            # weight of dW^2 in the cost
TAKEOFF_FOOT = 'LeftFootThumb1'           # the ACU foot that leaves the ground at the takeoff event


def w_of_t(t, t_acu):
    """WD2 seconds at clip seconds t (k = 1): the ACU takeoff t_acu maps to the WD2 takeoff 0.45 s."""
    return WP.TAKEOFF + (np.asarray(t, float) - float(t_acu))


def _smoothstep(u):
    return D3._smoothstep(u)


# ------------------------------------------------------------------------------------------------ setup
class Ctx:
    """Everything that does not depend on the placement (loaded once)."""

    def __init__(self):
        self.rig = R.Rig()
        self.ev = WP.load_events()
        self.m = wd2.load(self.ev['stem'])
        self.sol = WP.solve_path(self.rig, self.m, self.ev)
        model = BD.load_play_model_so()
        clips = {}
        for handle, name, dump in BD.targets_dive():
            seq = acuclip.pose_sequence(dump, step=1)
            clips[name] = dict(handle=handle, dump=dump, seq=seq, ev=contacts.events(seq, dump),
                               mp=BD.clip_model(name, model, float(dump['Length'])))
        self.offsets = bake2.measure_contact_offsets([(n, c['dump'], c['seq'], c['ev']) for n, c in clips.items()])
        so = clips[SO_NAME]
        assert so['handle'] == SO_HEX, so['handle']
        self.so = so
        self.mp = so['mp']
        self.ob = so['ev']['obstacle']
        self.ob_source = so['ev']
        self.T_A = float(so['ev']['takeoff']['seconds'])           # ACU takeoff, clip s (dump event)
        self.takeoff_foot = so['ev']['takeoff']['foot']
        self.o = float(self.mp['o'])
        self.r = float(self.mp['r'])
        self.W_played = float(self.mp['W'])
        self.span_so = self.r * self.W_played
        self.t_W = self.o + self.span_so                           # clip s of the SlideOver window end
        self.w_W = float(self.w(self.t_W))
        self.ex_dump = dumps.load_by_name(EXIT_NAME)
        self.ex_seq = acuclip.pose_sequence(self.ex_dump, step=1)
        self.next_dump = dumps.load_by_name(NEXT_NAME)
        self.span_e = EXIT_R * EXIT_WIN
        self.t_end_E = EXIT_O + self.span_e
        self.hips = self.rig.ac_by_hash[zlib.crc32(b'Hips')]
        self.lpos_hips = np.array(self.rig.ac[self.hips]['lpos'], float)
        self.body = sorted(set(self.rig.map.values()))
        self.fingers = sorted(bake2.finger_bones(self.rig, self.body))
        self.body_nf = [b for b in self.body if b not in set(self.fingers)]
        self.world_so = acuclip.world_positions(so['seq'], so['dump'])
        self.disp_so = acuclip.markers(so['dump'], so['seq'].frames)['disp']
        self.hips_orig_so = self.world_so[:, self.hips]
        fr = np.arange(len(self.hips_orig_so))
        self.orig_takeoff_y = float(np.interp(self.T_A * FPS, fr, self.hips_orig_so[:, 1]))
        self.orig_W = np.array([np.interp(self.t_W * FPS, fr, self.hips_orig_so[:, c]) for c in range(3)])
        foot_i = self.rig.ac_by_hash[zlib.crc32(TAKEOFF_FOOT.encode())]
        self.foot_take_y = float(np.interp(self.T_A * FPS, fr, self.world_so[:, foot_i, 1]))
        self.foot_take_z = float(np.interp(self.T_A * FPS, fr, self.world_so[:, foot_i, 2]))
        # Exit: joined root = the SlideOver root at W plus the Exit's own displacement change since its played start
        disp_E = acuclip.markers(self.ex_dump, self.ex_seq.frames)['disp']
        cut = np.array([np.interp(EXIT_O * FPS, np.arange(len(disp_E)), disp_E[:, c]) for c in range(3)])
        root_W = np.array([np.interp(self.t_W * FPS, fr, self.disp_so[:, c]) for c in range(3)])
        self.disp_ex = root_W[None, :] + (disp_E - cut[None, :])
        self.disp_ex[:, 0] = 0.0
        self.root_W = root_W
        self.ground_ob = BC._ob_ground()
        # WD2 placement constants (relative to the takeoff pelvis at WD2 0.45 s)
        self.p0 = WP.path_at(self.sol, [WP.TAKEOFF])[0]
        a = self.sol['anchors']
        self.stance_rel_y = float(a['stance_xy'][1] - self.p0[1])        # -0.652 m
        self.hand_rel_y = {s: float(a['hand_%s_xy' % s][1] - self.p0[1]) for s in ('L', 'R')}
        self.P_W_rel_y = float(WP.path_at(self.sol, [self.w_W])[0][1] - self.p0[1])

    def w(self, t):
        return w_of_t(t, self.T_A)

    @property
    def T_plant0(self):
        """Clip seconds of the WD2 hand-plant start."""
        return float(self.T_A + (self.ev['hand_plant_start'] - WP.TAKEOFF))


# ------------------------------------------------------------------------------------------------ placement
class Place:
    """WD2 world -> ACU world: x = x_wd2 - x_takeoff (no scale), y = y_take + s (y_wd2 - y_takeoff), z = z_wd2 + bump.
    The bump is a raised cosine over WD2 [b0, b1] (Reference z, the body is lifted over the box top)."""

    def __init__(self, sol, s, y_take, bump_amp=0.0, b0=BUMP_T0, b1=BUMP_T1):
        self.sol = sol
        self.s = float(s)
        self.y_take = float(y_take)
        self.bump_amp = float(bump_amp)
        self.b0, self.b1 = float(b0), float(b1)
        self.p0 = WP.path_at(sol, [WP.TAKEOFF])[0]

    def map_points(self, p):
        p = np.asarray(p, float)
        out = np.empty_like(p)
        out[..., 0] = p[..., 0] - self.p0[0]
        out[..., 1] = self.y_take + self.s * (p[..., 1] - self.p0[1])
        out[..., 2] = p[..., 2]
        return out

    def bump(self, w):
        return self.bump_amp * WP.raised_cosine(np.asarray(w, float), self.b0, self.b1)

    def pelvis(self, w):
        q = self.map_points(WP.path_at(self.sol, w))
        q[..., 2] = q[..., 2] + self.bump(w)
        return q

    def hand_anchor(self, side):
        a = self.sol['anchors']['hand_%s_xy' % side]
        q = self.map_points(np.array([a[0], a[1], WP.ACU_HAND_Z]))
        q[2] = WP.ACU_HAND_Z
        return q

    def hand_y(self):
        return {side: float(self.hand_anchor(side)[1]) for side in ('L', 'R')}


def y_take_for(ctx, s, shift, anchor='foot', foot_y=None):
    """Forward position of the takeoff pelvis (ACU Hips world y).

    anchor 'foot' (spec arithmetic): the WD2 stance foot lands on the measured ACU takeoff foot, so y_take = foot_y -
    s * stance_rel_y + shift. anchor 'pelvis' (dive3 convention): y_take = original Hips y at takeoff + shift."""
    if anchor == 'foot':
        fy = ctx.foot_take_y if foot_y is None else float(foot_y)
        return fy - s * ctx.stance_rel_y + shift
    if anchor == 'pelvis':
        return ctx.orig_takeoff_y + shift
    raise ValueError(anchor)


def scan_rows(ctx, s_grid=S_GRID, shift_grid=SHIFT_GRID, anchor='foot', foot_y=None, w_s=COST_W_S, w_w=COST_W_W):
    """One row per (s, shift): hand clearance, the W offset and the cost (feasible = hand rule holds)."""
    far = float(ctx.ob['far_y'])
    rows = []
    for s in s_grid:
        for sh in shift_grid:
            yt = y_take_for(ctx, s, sh, anchor, foot_y)
            hy = {side: yt + s * ctx.hand_rel_y[side] for side in ('L', 'R')}
            clear = min(hy.values()) - far
            pW = yt + s * ctx.P_W_rel_y
            dW = pW - float(ctx.orig_W[1])
            feas = clear >= FAR_CLEAR - 1e-9
            cost = float(sh) ** 2 + w_s * (float(s) - 1.0) ** 2 + w_w * dW ** 2
            rows.append(dict(s=float(s), shift=float(sh), y_take=float(yt), hand_L=hy['L'], hand_R=hy['R'],
                             hand_clear=float(clear), dW=float(dW), cost=float(cost), feasible=bool(feas),
                             ref_y_takeoff=float(yt - ctx.orig_takeoff_y)))
    return rows


def pick_placement(rows):
    """Lowest cost among the feasible rows (None when no row satisfies the hand rule)."""
    ok = [r for r in rows if r['feasible']]
    if not ok:
        return None
    return min(ok, key=lambda r: (r['cost'], r['s'], r['shift']))


def lift_profile(ctx, s, y_take, ws):
    """Lift (m) each WD2 sample ws needs before any bump: the non-finger body above the box top + clearance."""
    sol = ctx.sol
    ws = np.asarray(ws, float)
    pel = Place(sol, s, y_take, 0.0).pelvis(ws)
    body = sol['body']
    rel = np.array([[np.interp(ws, sol['t'], sol['rel'][:, b, c]) for c in range(3)] for b in body])   # [nb, 3, F]
    pts = pel[:, None, :] + rel.transpose(2, 0, 1)                                                     # [F, nb, 3]
    y, z = pts[..., 1], pts[..., 2]
    near, far, top = float(ctx.ob['near_y']), float(ctx.ob['far_y']), float(ctx.ob['height'])
    in_box = (y > near) & (y < far)
    need = np.where(in_box, top + BUMP_CLEAR - z, 0.0)
    return np.maximum(need.max(axis=1), 0.0)


def bump_amplitude(ctx, s, y_take):
    """Smallest amplitude A of the raised cosine on [BUMP_T0, BUMP_T1] with A * shape(w) >= lift(w) for every w.
    Returns (A or None, info). None when a needed lift falls where the shape is zero."""
    ws = np.round(np.arange(BUMP_T0, BUMP_T1 + 1e-9, 1.0 / FPS), 6)
    L = lift_profile(ctx, s, y_take, ws)
    return amplitude_from_lift(ws, L)


def amplitude_from_lift(ws, L, t0=BUMP_T0, t1=BUMP_T1):
    """Minimum amplitude A with A * raised_cosine(ws; t0, t1) >= L at every sample (None when some lift falls where the
    shape is zero). Returns (A, info) with the lift at the WD2 takeoff, the maximum lift and where it occurs."""
    ws = np.asarray(ws, float)
    L = np.asarray(L, float)
    shape = WP.raised_cosine(ws, t0, t1)
    info = dict(L_takeoff=float(np.interp(WP.TAKEOFF, ws, L)), L_max=float(L.max()), w_max=float(ws[int(np.argmax(L))]))
    bad = (shape <= 1e-12) & (L > 1e-9)
    if np.any(bad):
        info['bad_w'] = [float(x) for x in ws[bad]]
        return None, info
    pos = shape > 1e-12
    amp = float(np.max(L[pos] / shape[pos])) if np.any(L > 0) else 0.0
    return amp, info


def post_bump_clearance(ctx, s, y_take, amp):
    """Smallest clearance (m) of the non-finger body above the box top over the flight with the bump, and where."""
    ws = np.round(np.arange(WP.TAKEOFF, 0.92 + 1e-9, 1.0 / FPS), 6)
    sol = ctx.sol
    pl = Place(sol, s, y_take, amp)
    pel = pl.pelvis(ws)
    body = sol['body']
    rel = np.array([[np.interp(ws, sol['t'], sol['rel'][:, b, c]) for c in range(3)] for b in body])
    pts = pel[:, None, :] + rel.transpose(2, 0, 1)
    y, z = pts[..., 1], pts[..., 2]
    near, far = float(ctx.ob['near_y']), float(ctx.ob['far_y'])
    in_box = (y > near) & (y < far)
    clr = np.where(in_box, z - float(ctx.ob['height']), np.inf)
    k = np.unravel_index(int(np.argmin(clr)), clr.shape)
    return float(clr.min()), float(ws[k[0]])


# ------------------------------------------------------------------------------------------------ one case
def window_step_cm(d, win):
    """Largest change of the Reference offset d [F, 3] between two consecutive 60 Hz frames of the played window (cm)."""
    win = np.asarray(win, int)
    if win.size < 2:
        return 0.0
    return float(np.max(np.linalg.norm(np.diff(d[win], axis=0), axis=1)) * 100.0)


def cont_at_joint(ctx, fS, fE, fS_last, fE_first):
    """Continuity of the joined rotations: the last played SlideOver frame against the first played Exit frame (the
    dive3 check used round(t_W * 60) and round(0.208 * 60) = 12, which is 0.200 s, before the Exit's step)."""
    ang = {}
    for bn in BD.GAP_BONES:
        ai = BD._acu(ctx.rig, bn)
        ang[bn] = float(bake2.angle_deg(fS['final'][fS_last, ai], fE['final'][fE_first, ai]))
    vals = np.array(list(ang.values()))
    return dict(max_deg=float(vals.max()), median_deg=float(np.median(vals)), worst=max(ang, key=lambda k: ang[k]),
                per_bone=ang)


def bake_clip4(ctx, seg, pl, win, plant, ref_times, disp, ob, ref_w, hand_targets=True, ref_override=None):
    """dive3.bake_clip with an optional explicit Reference position (ref_override [F, 3]): the Reference translation is
    ref0 + ref_w * (ref_full - ref0). Without the override ref_full is the placed pelvis (target) minus root and FK."""
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
    if ref_override is None:
        target = pl.pelvis(ref_times)
        ref_full = np.asarray(target, float) - np.asarray(disp, float) - D3._hips_offset(ctx, rref)
    else:
        ref_full = np.asarray(ref_override, float)
    targets = {}
    ik_targets = {}
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
    ref_out = ref0 + np.asarray(ref_w, float)[:, None] * (ref_full - ref0)
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


def bake_case(ctx, s, y_take, amp, fi=FI, tfi=TFI, mode='absolute', ex_conv_end=None):
    """Bake the joined SlideOver + Exit for one placement. Returns a dict with both clips and the metrics.

    mode 'absolute' is the spec (WD2 orientations); 'additive' is the dive3 scheme (diagnostic only).
    ex_conv_end: clip s at which the Exit's Reference offset reaches 0 (default: the Exit clip end)."""
    sol = ctx.sol
    pl = Place(sol, s, y_take, bump_amp=amp)
    o, span = ctx.o, ctx.span_so
    wfun = ctx.w
    # ---- SlideOver
    seg_S = BC.segment(ctx.rig, ctx.m, ctx.so['seq'], ctx.so['dump'], wfun, float(wfun(o)),
                       lambda t: bake2.patch_weight(t, o, span, fi, FO, TAIL), disp=ctx.disp_so)
    seg_S['win'] = bake2.window_frames(o, span, int(seg_S['F'][-1]))
    seg_S['ob'] = ctx.ob
    if mode == 'absolute':
        seg_S['patch'] = D3.absolute_patch(ctx, seg_S, wfun, o)
    t_S = seg_S['t60']
    win_S = seg_S['win']
    tw = wfun(win_S / FPS)
    plant_S = win_S[(tw >= ctx.ev['hand_plant_start'] - 1e-9) & (tw <= ctx.ev['hand_plant_end'] + 1e-9)]
    wT_S = bake2.patch_weight(t_S, o, span, tfi, FO, TAIL)
    fin_S = bake_clip4(ctx, seg_S, pl, win_S, plant_S, wfun(t_S), ctx.disp_so, ctx.ob, wT_S)
    fin_S['win'] = win_S
    fin_S['plant'] = plant_S
    met_S, arr_S = D3._validate(ctx, seg_S, fin_S, ctx.t_W, span, o)

    # ---- Exit (gated step at its played start, WD2 continues at k = 1 from w(W))
    w_W = float(wfun(ctx.t_W))

    def w_exit(t):
        return w_W + (np.asarray(t, float) - EXIT_O) / EXIT_R

    seg_E = BC.segment(ctx.rig, ctx.m, ctx.ex_seq, ctx.ex_dump, w_exit, float(w_exit(EXIT_O)),
                       lambda t: (np.asarray(t, float) >= EXIT_O - 1e-9).astype(float), disp=ctx.disp_ex)
    seg_E['win'] = bake2.window_frames(EXIT_O, ctx.span_e, int(seg_E['F'][-1]))
    seg_E['ob'] = ctx.ground_ob
    if mode == 'absolute':
        seg_E['patch'] = D3.absolute_patch(ctx, seg_E, w_exit, EXIT_O)
    tE = seg_E['t60']
    L_E = float(ctx.ex_dump['Length'])
    conv_end = L_E if ex_conv_end is None else float(ex_conv_end)
    # the offset the SlideOver ends with (its last played frame) carries over and converges to the Exit's own original
    # Reference across the Exit window and the rest of the clip (smoothstep): no jump at the cut (fix 5)
    off_W = fin_S['ref_out'][int(win_S[-1])] - seg_S['ref0_orig'][int(win_S[-1])]
    wE_tr = np.where(tE >= EXIT_O - 1e-9, 1.0, 0.0) * (1.0 - _smoothstep((tE - EXIT_O) / max(conv_end - EXIT_O, 1e-9)))
    win_E = seg_E['win']
    ref_over = seg_E['ref0_orig'] + off_W[None, :]
    fin_E = bake_clip4(ctx, seg_E, pl, win_E, np.array([], int), w_exit(tE), ctx.disp_ex, ctx.ground_ob, wE_tr,
                       hand_targets=False, ref_override=ref_over)
    fin_E['win'] = win_E
    fin_E['plant'] = np.array([], int)
    fin_E['targets'] = {}
    fin_E['disp'] = ctx.disp_ex
    fW_E = min(int(round(ctx.t_end_E * FPS)), len(tE) - 1)
    gap_pre = D3._gap_to_next(ctx, fin_E['final'], fW_E)
    ease_used = False
    if gap_pre['max_deg'] > GAP_DEG:
        fin_E['final'] = D3._ease(ctx, fin_E['final'], seg_E['patset'], fW_E)
        ease_used = True
    fin_E['rot_tracks'] = {ai: bake2.rotation_track(dumps.track(ctx.ex_dump, ctx.rig.ac[ai]['hash'], 0) or [],
                                                   fin_E['final'][:, ai], max(0, int(win_E[0]) - 1),
                                                   int(seg_E['F'][-1])) for ai in seg_E['patset']}
    met_E, arr_E = D3._validate(ctx, seg_E, fin_E, ctx.t_end_E, ctx.span_e, EXIT_O, next_dump=ctx.next_dump)
    cont = cont_at_joint(ctx, fin_S, fin_E, int(win_S[-1]), int(win_E[0]))
    dS = fin_S['ref_out'] - seg_S['ref0_orig']
    dE = fin_E['ref_out'] - seg_E['ref0_orig']
    seq_d = np.concatenate([dS[win_S], dE[win_E]])
    joined_step = float(np.max(np.linalg.norm(np.diff(seq_d, axis=0), axis=1)) * 100.0)
    # the step rule is on the PLAYED window: the Reference samples outside it (before o, after the window end) are not
    # played, so their frame-to-frame change is not counted (dive3 counted the whole clip; kept as dstep_clip_cm)
    for met, d, win in ((met_S, dS, win_S), (met_E, dE, win_E)):
        met['dstep_clip_cm'] = met['dstep_max_cm']
        met['dstep_max_cm'] = window_step_cm(d, win)
    return dict(s=float(s), y_take=float(y_take), bump=float(amp), pl=pl, seg_S=seg_S, seg_E=seg_E, off_W=off_W,
                fin_S=fin_S, fin_E=fin_E, met_S=met_S, arr_S=arr_S, met_E=met_E, arr_E=arr_E, cont=cont,
                gap_pre=gap_pre, ease_used=ease_used, joined_step_cm=joined_step, w_exit=w_exit, w_W=w_W,
                dS=dS, dE=dE, win_S=win_S, win_E=win_E, mode=mode, fi=fi, tfi=tfi)


def statuses(res):
    return D3.statuses(res)


def case_row(res, st):
    """One line of the diagnostics table."""
    mS, mE = res['met_S'], res['met_E']
    return dict(s=res['s'], y_take=res['y_take'], bump=res['bump'], mode=res['mode'], fi=res['fi'], tfi=res['tfi'],
                pen_box_S=mS['pen_box_cm'], pen_ground_S=mS['pen_ground_cm'], touch=mS['touch']['min'],
                step_S=mS['dstep_max_cm'], step_joined=res['joined_step_cm'], cont=res['cont']['max_deg'],
                pen_E=mE['pen_patch_cm'], step_E=mE['dstep_max_cm'], gap_pre=res['gap_pre']['max_deg'],
                S=st['S']['status'], E=st['E']['status'])


# ------------------------------------------------------------------------------------------------ outputs
class Tl4:
    """Timeline for the BC plot helpers (o, r, W played, T played plant start, w_so, w_exit)."""

    def __init__(self, ctx, w_exit):
        self.o, self.r, self.W = ctx.o, ctx.r, ctx.W_played
        self.T = (ctx.T_plant0 - ctx.o) / ctx.r
        self._w_exit = w_exit
        self._ctx = ctx

    def w_so(self, t):
        return float(self._ctx.w(t))

    def w_exit(self, t):
        return float(self._w_exit(t))


def write_outputs(ctx, res, st):
    os.makedirs(PATCH_DIR, exist_ok=True)
    rig = ctx.rig
    DC2.write_patch(os.path.join(PATCH_DIR, SO_HEX + '.baked.json'), rig, res['fin_S']['rot_tracks'],
                    res['fin_S']['trans_keys'], None, ctx.so['dump']['Length'])
    DC2.write_patch(os.path.join(PATCH_DIR, EXIT_HEX + '.baked.json'), rig, res['fin_E']['rot_tracks'],
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
    rig = ctx.rig
    PfS = D3._world(res['seg_S']['seq'], res['seg_S']['dump'], res['fin_S']['final'], res['fin_S']['ref_out'],
                    res['fin_S']['disp'])
    PoS = acuclip.world_positions(ctx.so['seq'], ctx.so['dump'])
    PfE = D3._world(res['seg_E']['seq'], res['seg_E']['dump'], res['fin_E']['final'], res['fin_E']['ref_out'],
                    res['fin_E']['disp'])
    PoE = acuclip.world_positions(ctx.ex_seq, ctx.ex_dump)
    arrS = dict(Pf=PfS, Po=PoS)
    arrE = dict(Pf=PfE, Po=PoE)
    tl = Tl4(ctx, res['w_exit'])
    S = dict(F=res['seg_S']['F'], disp=ctx.disp_so, name=SO_NAME)
    E = dict(F=res['seg_E']['F'], disp=ctx.disp_ex, name=EXIT_NAME)
    title = 'dive4: s %.2f, y_take %.3f, bump %.3f m, continuity %.0f deg' % (res['s'], res['y_take'], res['bump'],
                                                                             res['cont']['max_deg'])
    BC.plot_joined(SHEET_PNG, rig, S, arrS, E, arrE, ctx.ob, tl, title, show_root=True)
    BC.write_gif(SHEET_GIF, rig, S, arrS, E, arrE, ctx.ob, tl)


def _table(rows, headers):
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(out)


def write_scan(rows, pick):
    os.makedirs(OUT_DIR, exist_ok=True)
    keys = ['s', 'shift', 'y_take', 'ref_y_takeoff', 'hand_L', 'hand_R', 'hand_clear', 'dW', 'cost', 'feasible']
    with open(SCAN_CSV, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(keys)
        for r in rows:
            wr.writerow([('%.4f' % r[k]) if isinstance(r[k], float) else r[k] for k in keys])
    # the compact table: the feasible boundary and a few scales at the chosen shift
    lines = ['# dive4 placement scan', '',
             'Full grid (%d rows): scan.csv. Feasible = closer hand at or beyond the far edge + %.2f m. '
             'Cost = shift^2 + %.2f (s-1)^2 + %.2f dW^2.' % (len(rows), FAR_CLEAR, COST_W_S, COST_W_W), '']
    head = ['s', 'shift (m)', 'Reference y at takeoff (m)', 'closer hand y (m)', 'hand clearance (m)', 'dW at W (m)',
            'cost', 'feasible']
    picks = []
    for s in [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        at_s = [r for r in rows if abs(r['s'] - s) < 1e-9 and abs(r['shift']) < 1e-9]
        if at_s:
            picks.append(at_s[0])
    rows_md = []
    for r in picks:
        rows_md.append(['%.2f' % r['s'], '%.2f' % r['shift'], '%+.3f' % r['ref_y_takeoff'],
                        '%.3f' % min(r['hand_L'], r['hand_R']), '%+.3f' % r['hand_clear'], '%+.3f' % r['dW'],
                        '%.4f' % r['cost'], 'yes' if r['feasible'] else 'no'])
    lines.append('## Shift 0 at each scale (the foot-anchored baseline)')
    lines.append('')
    lines.append(_table(rows_md, head))
    lines.append('')
    best_reach = max(rows, key=lambda r: r['hand_clear'])
    lines.append('Best reach in the box: s %.2f, shift %.2f, closer hand %.3f m (clearance %+.3f m, feasible: %s).'
                 % (best_reach['s'], best_reach['shift'], min(best_reach['hand_L'], best_reach['hand_R']),
                    best_reach['hand_clear'], 'yes' if best_reach['feasible'] else 'no'))
    if pick is None:
        lines.append('No row is feasible: no pick.')
    else:
        lines.append('Pick (lowest cost among the feasible rows): s %.2f, shift %.2f, Reference y at takeoff %+.3f m, '
                     'dW %+.3f m, cost %.4f.' % (pick['s'], pick['shift'], pick['ref_y_takeoff'], pick['dW'],
                                                 pick['cost']))
    lines.append('')
    with open(SCAN_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    feas = [r for r in rows if r['feasible']]
    return dict(best=best_reach, n_feasible=len(feas), min_offset=min(r['ref_y_takeoff'] for r in feas) if feas else float('nan'))


def _mode_row(r):
    return ['%.1f' % r['pen_box_S'], '%.1f' % r['pen_ground_S'], str(r['touch']), '%.2f' % r['step_S'],
            '%.2f' % r['step_joined'], '%.0f' % r['cont'], '%.1f' % r['pen_E'], '%.2f' % r['step_E'],
            '%.0f' % r['gap_pre']]


def write_report(ctx, res, st, pick, scan_info, diags, bump_info, clear_info, lift_end):
    mS, mE = res['met_S'], res['met_E']
    A = []
    A.append('# T15 dive4: WD2 frontdive body path on the Up050cm SlideOver (LOW wall), Exit Fd100cm-Dn100cm\n')
    A.append('Generated by `v2/pipeline/dive4.py` (reuses `dive3.py` helpers). Written only under `out\\dive4`.\n')
    A.append('## Status\n')
    A.append('| clip | status | enabled | override | fails |')
    A.append('|---|---|---|---|---|')
    A.append('| SlideOver %s | %s | %s | %s | %s |' % (SO_NAME, st['S']['status'], 'yes' if st['S']['enabled'] else 'no',
                                                   st['S']['override'] or '-', '; '.join(st['S']['fails']) or '-'))
    A.append('| Exit %s | %s | %s | %s | %s |' % (EXIT_NAME, st['E']['status'], 'yes' if st['E']['enabled'] else 'no',
                                               st['E']['override'] or '-', '; '.join(st['E']['fails']) or '-'))
    A.append('')
    A.append('Enabled only if the enable rule passes (or the narrow override applies). The result is in section 3.\n')

    A.append('## 1. Inputs checked against the dumps and captures\n')
    A.append('| item | value | source |')
    A.append('|---|---|---|')
    A.append('| SlideOver | %s (%s) | dump, obstacle: height %.3f m, near edge y %.3f, far edge y %.3f, landing y %.3f |'
             % (SO_NAME, SO_HEX, ctx.ob['height'], ctx.ob['near_y'], ctx.ob['far_y'], ctx.ob['landing_y']))
    A.append('| ACU takeoff (clip s) | %.4f, foot %s (on the ground before, off after) | contacts.events on 7FC9C7405 (frame 28) |'
             % (ctx.T_A, ctx.takeoff_foot))
    A.append('| takeoff foot y at takeoff / Hips y at takeoff | %.3f / %.3f m | dump world positions |'
             % (ctx.foot_take_y, ctx.orig_takeoff_y))
    A.append('| root y at 0.48 s (the brief\'s "takeoff foot 2.71") | %.3f m | dump disp, root rise begins 0.47 s |'
             % float(np.interp(0.48 * FPS, np.arange(len(ctx.disp_so)), ctx.disp_so[:, 1])))
    A.append('| o / r / W played | %.4f / %.4f / %.4f | play model (reference map, ASSUMED: this clip has no own fit) and the '
             'capture window (1 occurrence) |' % (ctx.o, ctx.r, ctx.W_played))
    A.append('| SlideOver W (clip s) | %.4f | o + r W |' % ctx.t_W)
    A.append('| WD2 time at W | %.4f s | k = 1: w = 0.45 + (t - %.4f) |' % (ctx.w_W, ctx.T_A))
    A.append('| next clip after the SlideOver (capture session4 cap_20261008_201039_19) | Muraille_Exit Fd100cm-Dn500cm '
             '(F7919BE5E) at 1.718873 s, then Fd100cm-Dn100cm (F7919BDF0) at 1.718887 s | 1 ms rule: the last member plays |')
    A.append('| Exit that plays | Fd100cm-Dn100cm (F7919BDF0), o %.3f, r %.4f | exit_report.md (42 usable fits) |'
             % (EXIT_O, EXIT_R))
    A.append('| after the Exit | group of 5 Nav starts within 40 us at 1.934 s: Nav_Run_3-Loop among them (BankLeft_Light, '
             'SlopeDn30Deg, SlopeSprint_Dn30Deg, Sprint_V02) | the played one is not resolvable from this capture; ease target '
             'is Nav_Run_3-Loop as specified |')
    A.append('')

    A.append('## 2. Placement (fix 2): scan and choice\n')
    A.append('Anchor: the WD2 stance foot (-0.652 m behind the takeoff pelvis) lands on the measured ACU takeoff foot '
             '(%s y %.3f m); the placed pelvis at takeoff is then %.3f m + s*0.652 + shift. The hand rule is the closer hand '
             '(R) at or beyond the far edge + %.2f m = %.3f m.\n'
             % (TAKEOFF_FOOT, ctx.foot_take_y, ctx.foot_take_y, FAR_CLEAR, ctx.ob['far_y'] + FAR_CLEAR))
    A.append('Pick: s %.2f, shift %.2f m (Reference y at takeoff %+.3f m), dW at W %+.3f m, cost %.4f.\n'
             % (pick['s'], pick['shift'], pick['ref_y_takeoff'], pick['dW'], pick['cost']) if pick else
             'No feasible placement in the box (see scan.md).\n')
    A.append('Scan summary (full grid in scan.csv; see scan.md): %d of %d rows feasible. Best reach in the box: s %.2f, shift %.2f, '
             'closer hand %.3f m (clearance %+.3f m).\n'
             % (scan_info['n_feasible'], len(S_GRID) * len(SHIFT_GRID), scan_info['best']['s'], scan_info['best']['shift'],
                min(scan_info['best']['hand_L'], scan_info['best']['hand_R']), scan_info['best']['hand_clear']))
    A.append('The Reference y at takeoff is the placed target (%+.3f m). The applied offset is smaller at the takeoff (see '
             'section 3): the 0.15 s fade starts at o, 0.04 s before the takeoff, so the offset is partly ramped in there.\n'
             % (pick['ref_y_takeoff'] if pick else float('nan')))
    A.append('Bump (fix 3): raised cosine on the Reference z over WD2 %.2f .. %.2f s (0.1 s before the takeoff to the plant start). '
             'Amplitude %s m (minimum for %.2f m clearance of the non-finger body over the box top). The lift needed at the takeoff '
             '(WD2 0.45 s) is %.3f m, at its worst %.3f m (WD2 %.2f s). Post-bump lowest clearance over the box: %+.3f m at WD2 %.2f s.\n'
             % (BUMP_T0, BUMP_T1, ('%.3f' % res['bump']) if res['bump'] else 'none', BUMP_CLEAR,
                bump_info['L_takeoff'], bump_info['L_max'], bump_info['w_max'], clear_info[0], clear_info[1]))

    A.append('## 3. Validation (fixes 1, 4, 5, 6)\n')
    A.append('| metric | SlideOver | Exit | rule |')
    A.append('|---|---|---|---|')
    A.append('| non-finger penetration, box / ground / max (cm) | %.1f / %.1f / %.1f | %.1f / %.1f / %.1f | <= %.0f |'
             % (mS['pen_box_cm'], mS['pen_ground_cm'], mS['pen_patch_cm'], mE['pen_box_cm'], mE['pen_ground_cm'],
                mE['pen_patch_cm'], PEN_CM))
    A.append('| bone below ground, all bones incl. fingers (cm) | %.1f | %.1f | info |' % (mS['ground_all_cm'], mE['ground_all_cm']))
    A.append('| hand touch frames (L / R, min) | %d / %d (%d) | n/a | >= %d |'
             % (mS['touch'].get('LeftHand', 0), mS['touch'].get('RightHand', 0), mS['touch']['min'], TOUCH))
    A.append('| Reference offset max step per 60 Hz frame (cm) | %.2f | %.2f (joined %.2f) | <= %.0f |'
             % (mS['dstep_max_cm'], mE['dstep_max_cm'], res['joined_step_cm'], STEP_CM))
    A.append('| continuity at W, max / median (deg) | %.1f / %.1f (%s) | | <= %.0f |'
             % (res['cont']['max_deg'], res['cont']['median_deg'], res['cont']['worst'], CONT_DEG))
    A.append('| pre-ease gap to Nav_Run_3-Loop, max / median (deg) | | %.1f / %.1f (%s) | ease over the last %.2f s when > %.0f |'
             % (res['gap_pre']['max_deg'], res['gap_pre']['median_deg'], res['gap_pre']['worst'], EASE_S, GAP_DEG))
    A.append('| Exit ease applied | | %s | |' % ('yes' if res['ease_used'] else 'no'))
    dS, dE = res['dS'], res['dE']
    A.append('| Reference offset (x, y, z; m) at o / takeoff / W | %s | | |'
             % ' ; '.join('%s (%.2f, %.2f, %.2f)' % ((lbl,) + tuple(dS[int(round(tt * FPS))]))
                          for lbl, tt in (('o', ctx.o), ('takeoff', ctx.T_A), ('W', ctx.t_W))))
    A.append('')
    A.append('Hand IK channels: %s. Foot clamp channels: %s.\n' % (res['fin_S']['ik_hands'], res['fin_S']['ik_feet']))
    A.append('Pre-takeoff float: the bump starts at WD2 0.35 s; the stance foot is lifted by up to the bump at the takeoff (the '
             'specified span; not clamped to the ground before the takeoff).\n')

    A.append('## 4. Enable decision\n')
    A.append('Rule: non-finger penetration <= 5 cm, hand touch >= 4 frames, Reference step <= 4 cm/frame, continuity <= 45 deg. '
             'Narrow override: <= 10 cm, >= 2 frames, <= 8 cm/frame, <= 60 deg.\n')
    A.append('SlideOver: **%s**. Exit: **%s**.\n' % (st['S']['status'], st['E']['status']))

    A.append('## 5. Diagnostic bakes (not written)\n')
    A.append('| bake | mode | fade rot / trans (s) | s | Reference y at takeoff (m) | bump (m) | box pen (cm) | ground pen (cm) | '
             'hand touch | SO step (cm/f) | joined step (cm/f) | continuity (deg) | Exit pen (cm) | Exit step (cm/f) | '
             'pre-ease gap (deg) | status SO / Exit |')
    A.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for name, row in diags:
        if 'note' in row:
            A.append('| %s | - | - | - | - | - | - | - | - | - | - | - | - | - | - | %s |' % (name, row['note']))
            continue
        A.append('| %s | %s | %.2f / %.2f | %.2f | %+.3f | %.3f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s / %s |'
                 % (name, row['mode'], row['fi'], row['tfi'], row['s'], row['y_take'] - ctx.orig_takeoff_y, row['bump'],
                    '%.1f' % row['pen_box_S'], '%.1f' % row['pen_ground_S'], row['touch'], '%.2f' % row['step_S'],
                    '%.2f' % row['step_joined'], '%.0f' % row['cont'], '%.1f' % row['pen_E'], '%.2f' % row['step_E'],
                    '%.0f' % row['gap_pre'], row['S'], row['E']))
    A.append('')
    A.append('## 6. What it shows and what is needed\n')
    m0 = scan_info['min_offset']
    ramp = lambda off: 1.5 * abs(off) / (STEP_CM / 100.0 * FPS)
    diag_map = dict((n, r) for n, r in diags if 'note' not in r)
    A.append('* **The hand rule forces a large takeoff offset.** The smallest Reference y offset at takeoff over all feasible '
             'placements is %+.3f m (the pick: %+.3f m). A smoothstep ramp of length T moves at most 1.5 D / T per second, so '
             '4 cm/frame needs T >= 1.5 D / 2.4 = %.2f s for D = %.2f m, against the specified 0.15 s (0.25 s in the fade '
             'diagnostic). The step rule fails for every feasible placement in the box.\n'
             % (m0, pick['ref_y_takeoff'] if pick else float('nan'), ramp(m0), m0))
    A.append('* **The WD2 roll is too low for this rig.** At W the WD2 pelvis is %.2f m high (WD2 %.2f s), the original ACU '
             'Hips %.2f m. Carrying the SlideOver offset into the Exit puts its lowest non-finger point at %+.2f m (clip %.2f s), '
             'so the Exit penetrates the ground by %.0f cm. The SlideOver lowest point is %+.2f m (clip %.2f s): ground '
             'penetration %.0f cm. Both exceed the 5 cm limit.\n'
             % (float(WP.path_at(ctx.sol, [ctx.w_W])[0][2]), ctx.w_W, float(ctx.orig_W[2]), res['minz_E'][0],
                res['minz_E'][1], -100.0 * res['minz_E'][0], res['minz_S'][0], res['minz_S'][1],
                -100.0 * res['minz_S'][0]))
    A.append('* **Hands and steps.** The hand plant lands %.3f m beyond the far edge, with the closer hand at %.3f m. Hand IK '
             'holds the hands for %d frames. The Exit\'s own played Reference step is %.2f cm/frame; the joined step (%.2f '
             'cm/frame) is set by the SlideOver ramp.\n'
             % (pick['hand_clear'] if pick else float('nan'), min(pick['hand_L'], pick['hand_R']) if pick else float('nan'),
                res['met_S']['touch']['min'], res['met_E']['dstep_max_cm'], res['joined_step_cm']))
    A.append('* **Diagnostics (section 5).** The 0.25 s fade passes the box check (%.1f cm) but not the ground or step rules. '
             'A dW weight of 2.0 moves the pick to s 1.29 and changes nothing material. The brief figure of a takeoff foot at '
             '2.71 m gives s 1.23 and an offset of +1.14 m. The dive3 pelvis anchor cannot reach the hand rule (best %.3f m '
             'against %.2f m). A 0.6 s translation fade removes the step but loses the hands (touch 0) and the box clearance. '
             'Additive rotations give continuity %.0f deg, as dive3 found.\n'
             % (diag_map.get('fade 0.25 s', {}).get('pen_box_S', float('nan')), 4.308, ctx.ob['far_y'] + FAR_CLEAR,
                diag_map.get('additive rotations (dive3 scheme)', {}).get('cont', float('nan'))))
    A.append('* **What the sheet shows.** The takeoff leaves over the near edge, the arms reach for the far corner of the box, '
             'and the body rises to about 1.7 m. The roll lands at the far edge in a crumpled pose, and the lowest point of the Exit is '
             'under the ground. It does not read as run, dive, hands, forward roll, up: the roll is too low and too far forward '
             'for this obstacle with k = 1.\n')
    A.append('* **Not done (needs a decision).** Plant the hands on the box top instead of beyond the far edge (the takeoff '
             'offset then falls toward +0.45 m, see the scan), or change the run-up so the takeoff comes later, or add a roll '
             'height lift. None of these is in the brief, so the patches stay disabled.\n')
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(A) + '\n')


def wd2_side(ctx):
    WP.plot_side(WD2_SIDE_PNG, ctx.sol)


# ------------------------------------------------------------------------------------------------ main
def min_body_z(ctx, res, which):
    """Lowest non-finger body point (m, entity world) over the played window of one clip, and its WD2 time."""
    seg, fin, win = (res['seg_S'], res['fin_S'], res['win_S']) if which == 'S' else (res['seg_E'], res['fin_E'], res['win_E'])
    P = D3._world(seg['seq'], seg['dump'], fin['final'], fin['ref_out'], fin['disp'])
    z = P[win][:, ctx.body_nf, 2]
    k = np.unravel_index(int(np.argmin(z)), z.shape)
    return float(z.min()), float(win[k[0]] / FPS)


def run_case(ctx, anchor='foot', foot_y=None, w_s=COST_W_S, w_w=COST_W_W, fi=FI, tfi=TFI, mode='absolute'):
    """Scan -> pick -> bump -> bake for one set of options. Returns (row dict, res, st, pick, bump_info, clear_info) or
    (None, reason) when no placement is feasible / no bump amplitude exists."""
    rows = scan_rows(ctx, anchor=anchor, foot_y=foot_y, w_s=w_s, w_w=w_w)
    pick = pick_placement(rows)
    if pick is None:
        return None, 'no feasible placement (best reach %.3f m closer hand)' % max(
            min(r['hand_L'], r['hand_R']) for r in rows)
    s, y_take = pick['s'], pick['y_take']
    amp, binfo = bump_amplitude(ctx, s, y_take)
    if amp is None:
        return None, 'no bump amplitude (lift needed where the shape is zero): %s' % binfo
    clear = post_bump_clearance(ctx, s, y_take, amp)
    res = bake_case(ctx, s, y_take, amp, fi=fi, tfi=tfi, mode=mode)
    st = statuses(res)
    res.update(pick=pick, bump_info=binfo, clear_info=clear, rows=rows, anchor=anchor, foot_y=foot_y, w_s=w_s, w_w=w_w)
    res['minz_S'] = min_body_z(ctx, res, 'S')
    res['minz_E'] = min_body_z(ctx, res, 'E')
    return case_row(res, st), res, st, pick, binfo, clear


DIAGS = [
    ('spec', dict()),
    ('fade 0.25 s', dict(fi=0.25, tfi=0.25)),
    ('cost: dW weight 2.0', dict(w_w=2.0)),
    ('takeoff foot at 2.71 (brief figure)', dict(foot_y=2.71)),
    ('pelvis anchor (dive3 convention)', dict(anchor='pelvis')),
    ('translation fade 0.60 s (outside 0.10-0.25, step check)', dict(tfi=0.60)),
    ('additive rotations (dive3 scheme)', dict(mode='additive')),
]


def run(verbose=True):
    ctx = Ctx()
    rows = scan_rows(ctx)
    pick = pick_placement(rows)
    scan_info = write_scan(rows, pick)
    out = run_case(ctx)
    if out[0] is None:
        raise RuntimeError('spec placement failed: %s' % out[1])
    spec_row, res, st, pick, binfo, clear = out
    diags = []
    for name, kw in DIAGS[1:]:
        o = run_case(ctx, **kw)
        if o[0] is None:
            diags.append((name, dict(note=o[1])))
        else:
            diags.append((name, dict(o[0], pick=o[3], minz_S=o[1]['minz_S'], minz_E=o[1]['minz_E'])))
        if verbose:
            print(name, ('->' if o[0] is not None else 'no pick:'), (o[0] or {}).get('S', o[1]), (o[0] or {}).get('E', ''))
    return ctx, rows, pick, scan_info, res, st, spec_row, binfo, clear, diags


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--scan' in argv:
        ctx = Ctx()
        rows = scan_rows(ctx)
        pick = pick_placement(rows)
        print('feasible rows', sum(1 for r in rows if r['feasible']), 'of', len(rows), 'pick', pick)
        return 0
    ctx, rows, pick, scan_info, res, st, spec_row, binfo, clear, diags = run()
    os.makedirs(OUT_DIR, exist_ok=True)
    wd2_side(ctx)
    mapping = write_outputs(ctx, res, st)
    write_sheet(ctx, res)
    write_report(ctx, res, st, pick, scan_info, [('spec', spec_row)] + diags, binfo, clear, None)
    summary = dict(s=res['s'], y_take=res['y_take'], bump=res['bump'], status=st, pick=pick, bump_info=binfo,
                   clear_info=clear, spec=spec_row, diags=diags, minz_S=res['minz_S'], minz_E=res['minz_E'],
                   off_W=[float(x) for x in res['off_W']])
    with open(SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps(dict(s=res['s'], y_take=res['y_take'], bump=res['bump'], S=st['S']['status'],
                          E=st['E']['status'])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
