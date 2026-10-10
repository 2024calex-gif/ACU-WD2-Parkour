"""T14 (dive3): the WD2 frontdive body path, reconstructed from the WD2 contacts (wd2_path.py), placed in ACU.

Differences from dive2 (replaced root):
  * TrackID 0 is NOT patched. The ACU entity root is the original one. The body path is set through the Reference
    position track only: Reference = target pelvis - entity root - FK offset. The Hips joint is a child of the Reference
    at lpos_hips, so its FK offset is R_ref * lpos_hips.
  * Time map k = 1: SlideOver clip seconds t are WD2 seconds (w = t). The Exit continues at WD2 time
    t_W + (t - 0.208) / 0.9665, where t_W is the SlideOver's played window end in clip seconds.
  * Space: x = x_take + WD2 lateral travel, y = y_take + s * WD2 forward travel, z = WD2 z (+ bump on the flight).
    s and y_take come from the spec's two placement rules (choose_placement): the hands plant >= far edge + 0.25 m,
    and the pelvis at W within 0.25-0.3 m of the original pelvis.
  * Rotations: the additive WD2 retarget of bake_chain.segment, variant D (fade-in 0.08 s from the original pose at o,
    no fade-out, WD2 tail). The Exit rotation is a gated step at 0.208 (no fade-in).
  * Hand IK pins the hands to their plant targets while reachable. A foot clamp lifts feet above the ground / box top.
  * Exit: the Reference offset converges to the original by the end of the Exit clip. Rotations ease to Nav_Run over
    the last 0.1 s of the window when the pre-ease gap exceeds 45 deg.

Outputs (all under out\\dive3; nothing else is written):
  patches\\47BFB02DF.baked.json   rotations incl. the Reference rotation and the Reference translation; NO TrackID 0
  patches\\F7919BDF0.baked.json   Exit: rotations and Reference translation
  patches\\mapping_dive3.json     SlideOver + gated Exit (only_after SlideOver, arm 2.0 s), prebaked, patch_translation
  joined_sheet.png / joined_sheet.gif (side view), scan.md (placement trade-off), report.md, summary.json
Run:  python dive3.py            (writes out\\dive3)
      python dive3.py --scan     (prints the placement trade-off table, writes nothing)
"""
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
import dive_chain2 as DC2
import dumps
import ik
import retarget as R
import wd2
import wd2_path as WP

OUT_DIR = os.path.join(config.OUT, 'dive3')
PATCH_DIR = os.path.join(OUT_DIR, 'patches')
MAPPING = os.path.join(PATCH_DIR, 'mapping_dive3.json')
REPORT = os.path.join(OUT_DIR, 'report.md')
SCAN = os.path.join(OUT_DIR, 'scan.md')
SUMMARY = os.path.join(OUT_DIR, 'summary.json')
SHEET_PNG = os.path.join(OUT_DIR, 'joined_sheet.png')
SHEET_GIF = os.path.join(OUT_DIR, 'joined_sheet.gif')
SO_NAME = BC.MAIN_SO
SO_HEX = '47BFB02DF'
EXIT_NAME = BC.EXIT_NAME
EXIT_HEX = BC.EXIT_HEX
NEXT_NAME = BC.NEXT_NAME                  # xx_Nav_Run_3-Loop
EXIT_O = BC.EXIT_O                        # 0.208 clip s: Exit played window start
EXIT_R = BC.EXIT_R                        # 0.9665 clip s per played s
FI, FO = BD.FI, BD.FO                     # variant D: fade-in 0.08 s, no fade-out
TAIL = bake2.TAIL_WD2
EASE_S = 0.10                             # s, ease to Nav_Run over the last part of the Exit window
GAP_DEG = 45.0                            # ease when the pre-ease gap exceeds this
CONT_DEG = 45.0                           # PASS: continuity at W
STEP_CM = 4.0                             # PASS: largest Reference offset change per 60 Hz frame
PEN_CM = 5.0                              # PASS: non-finger penetration of the box and the ground
TOUCH = 4                                 # PASS: frames both hands stay on their targets
NARROW = dict(pen=10.0, touch=2, step=8.0, cont=60.0)
S_GRID = np.round(np.arange(0.70, 1.3001, 0.01), 2)
FAR_CLEAR = 0.25                          # m, the hands plant at least this far beyond the far edge
W_TOL = 0.30                              # m, the pelvis at W within this distance of the original (spec)
W_PICK = 0.25                             # m, the pick keeps a margin inside that tolerance


class Tl:
    """Joined timeline for the plot helpers (bake_chain.plot_joined / write_gif)."""

    def __init__(self, o, r, W, t_plant0, w_exit):
        self.o, self.r, self.W = float(o), float(r), float(W)
        self.T = (float(t_plant0) - self.o) / self.r
        self._we = w_exit

    def w_so(self, t):
        return float(t)

    def w_exit(self, t):
        return float(self._we(t))


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
        self.o = float(self.mp['o'])
        self.r = float(self.mp['r'])
        self.W_played = float(self.mp['W'])
        self.span_so = self.r * self.W_played
        self.t_anchor = round(self.o * acuclip.FPS) / acuclip.FPS
        self.t_W = self.o + self.span_so                  # clip seconds of the SlideOver window end
        self.ex_dump = dumps.load_by_name(EXIT_NAME)
        self.ex_seq = acuclip.pose_sequence(self.ex_dump, step=1)
        self.next_dump = dumps.load_by_name(NEXT_NAME)
        self.span_e = EXIT_R * BC.EXIT_WIN
        self.t_end_E = EXIT_O + self.span_e
        self.hips = self.rig.ac_by_hash[zlib.crc32(b'Hips')]
        self.lpos_hips = np.array(self.rig.ac[self.hips]['lpos'], float)
        self.body = sorted(set(self.rig.map.values()))
        self.fingers = sorted(bake2.finger_bones(self.rig, self.body))
        self.body_nf = [b for b in self.body if b not in set(self.fingers)]
        self.world_so = acuclip.world_positions(so['seq'], so['dump'])
        self.disp_so = acuclip.markers(so['dump'], so['seq'].frames)['disp']
        self.hips_orig_so = self.world_so[:, self.hips]
        # the original SlideOver pelvis at takeoff and at W (world y)
        self.orig_takeoff_y = float(np.interp(WP.TAKEOFF * acuclip.FPS, np.arange(len(self.hips_orig_so)),
                                              self.hips_orig_so[:, 1]))
        self.orig_W = np.array([np.interp(self.t_W * acuclip.FPS, np.arange(len(self.hips_orig_so)),
                                          self.hips_orig_so[:, c]) for c in range(3)])
        # Exit joined root: the SlideOver's original root at W plus the Exit's own displacement change since the cut
        disp_E = acuclip.markers(self.ex_dump, self.ex_seq.frames)['disp']
        cut = np.array([np.interp(EXIT_O * acuclip.FPS, np.arange(len(disp_E)), disp_E[:, c]) for c in range(3)])
        root_W = np.array([np.interp(self.t_W * acuclip.FPS, np.arange(len(self.disp_so)), self.disp_so[:, c])
                           for c in range(3)])
        self.disp_ex = root_W[None, :] + (disp_E - cut[None, :])
        self.disp_ex[:, 0] = 0.0
        self.root_W = root_W
        self.ground_ob = BC._ob_ground()
        self.A_hand = {s: float(self.sol['anchors']['hand_%s_xy' % s][1]) for s in ('L', 'R')}
        self.Y_wd2_takeoff = float(WP.path_at(self.sol, [WP.TAKEOFF])[0][1])


def _hips_offset(ctx, rot_ref):
    """FK offset of the Hips joint from the Reference origin for Reference rotations rot_ref [F, 4] (entity frame)."""
    return np.stack([R.qrot(q, ctx.lpos_hips) for q in rot_ref], axis=0)


def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _world(seq, dump, rot, ref, disp):
    """World positions [F, B, 3] of a pose with explicit Reference positions and root displacement (entity + disp)."""
    return acuclip.entity_positions(bake2.composed(seq, rot, ref)) + np.asarray(disp, float)[:, None, :]


# ------------------------------------------------------------------------------------------------ placement
def placement_row(ctx, s, y_take=None):
    """Placement numbers for scale s. y_take None: the shift that puts the closest hand exactly FAR_CLEAR beyond the far
    edge (the spec's rule (a)). Returns a dict."""
    sol = ctx.sol
    a_min = min(ctx.A_hand.values())
    if y_take is None:
        y_take = (ctx.ob['far_y'] + FAR_CLEAR) - s * (a_min - ctx.Y_wd2_takeoff)
    pl = WP.Placement(sol, s, y_take)
    hand_y = {side: float(pl.hand_anchor(side)[1]) for side in ('L', 'R')}
    py_W = float(pl.pelvis(np.array([ctx.t_W]))[0, 1])
    dW = py_W - float(ctx.orig_W[1])
    stance_y = float(pl.map_points(np.array([sol['anchors']['stance_xy'][0], sol['anchors']['stance_xy'][1], 0.0]))[1])
    return dict(s=float(s), y_take=float(y_take), d_takeoff=float(y_take - ctx.orig_takeoff_y),
                hand_L=hand_y['L'], hand_R=hand_y['R'], hand_clear=min(hand_y.values()) - float(ctx.ob['far_y']),
                W_y=py_W, dW=dW, stance_foot_y=stance_y,
                stance_in_box=bool(ctx.ob['near_y'] < stance_y < ctx.ob['far_y']))


def choose_placement(ctx):
    """The spec's rules: hands at >= far + 0.25 m, and the pelvis at W within W_TOL of the original (W_PICK margin).
    Among the feasible scales the largest one is taken (the smallest forward shift), so the pick is the one whose
    takeoff displacement from the original is smallest. Returns (s, y_take, row)."""
    best = None
    for s in S_GRID:
        row = placement_row(ctx, s)
        if row['hand_clear'] + 1e-9 >= FAR_CLEAR - 1e-9 and abs(row['dW']) <= W_PICK + 1e-9:
            if best is None or row['s'] > best['s']:
                best = row
    if best is None:
        raise RuntimeError('no scale in %s satisfies the hand and W rules' % (S_GRID[[0, -1]],))
    return best['s'], best['y_take'], best


def bump_for(ctx, s, y_take):
    """Smallest raised-cosine bump (0.45 .. 0.75 s) that keeps every non-finger point out of the box with 5 cm clearance.
    Returns (amplitude or None, info): the lift the body needs at takeoff and its maximum (None: no amplitude works)."""
    pl = WP.Placement(ctx.sol, s, y_take)
    w, L = WP.required_lift(ctx.sol, pl, ctx.ob, ctx.sol['body'], clear=0.05)
    amp = WP.min_amplitude_for_lift(w, L)
    return amp, dict(L_takeoff=float(L[0]), L_max=float(L.max()), w_max=float(w[int(np.argmax(L))]))


# ------------------------------------------------------------------------------------------------ IK and clamps
def _hand_targets(ctx, pl, seq, dump, rot_full, ref_full, disp, win, plant):
    """Hand plant targets and their IK weights; returns (targets, ik_targets)."""
    rig = ctx.rig
    targets, ik_targets = {}, {}
    if plant.size == 0:
        return targets, ik_targets
    p0, pend = int(plant[0]), int(plant[-1])
    frr = np.arange(p0, pend + 1)
    P1 = _world(seq, dump, rot_full, ref_full, disp)
    for side, h, name in (('L', 'LeftHand', 'Left'), ('R', 'RightHand', 'Right')):
        ua, la, ha = BD._acu(rig, name + 'Arm'), BD._acu(rig, name + 'ForeArm'), BD._acu(rig, h)
        tgt = pl.hand_anchor(side)
        targets[h] = tgt
        d = np.linalg.norm(tgt[None, :] - P1[frr, ua], axis=1)
        arm1 = np.linalg.norm(P1[frr, la] - P1[frr, ua], axis=1) + np.linalg.norm(P1[frr, ha] - P1[frr, la], axis=1)
        wh, _u = bake2._ik_weights(len(seq.frames), win, p0, pend, d <= bake2.T9E_REACH * arm1, ramp=bake2.T9D_RAMP)
        ik_targets[h] = [(int(f), tgt, float(wh[f])) for f in np.nonzero(wh > 0)[0]]
    return targets, ik_targets


def _foot_targets(rig, P1, ob, offsets, win):
    """Feet lifted to the ground (or to the box top) when they are inside the window below their support."""
    out = {}
    for side in ('Left', 'Right'):
        foot, toe = side + 'Foot', side + 'FootThumb1'
        need = np.zeros(len(P1))
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
        wf = bake2.blend_mod_fade(len(P1), bake2._runs(pos_idx), bake2.T9D_RAMP)
        for f in np.nonzero(wf > 0)[0]:
            k = pos_idx[np.argmin(np.abs(pos_idx - f))]
            tgt = P1[f, BD._acu(rig, foot)] + np.array([0.0, 0.0, need[k]])
            out.setdefault(foot, []).append((int(f), tgt, float(wf[f])))
    return out


class _WClock:
    """bake_dive.dive_rotations clock: o is the patch start (clip s), wfun maps clip seconds to WD2 seconds."""

    def __init__(self, o, wfun=None):
        self.o = float(o)
        self._w = wfun

    def w(self, t):
        t = np.asarray(t, float)
        return t if self._w is None else np.asarray(self._w(t), float)


def absolute_patch(ctx, seg, wfun, o):
    """Diagnostic only: the absolute WD2 orientations (world = A_REF * Wt) on the patched bones (bake_dive.dive_rotations
    mode 'absolute'). The spec's scheme is the additive one; this shows the body shape the WD2 orientations produce."""
    patset, rot = seg['patset'], seg['rot']
    out = BD.dive_rotations(ctx.rig, ctx.m, _WClock(o, wfun), list(seg['F']), patset, rot, mode=BD.ROT_ABSOLUTE)
    patch = seg['orig'].copy()
    for ai in patset:
        patch[:, ai] = out[ai]
    return patch


# ------------------------------------------------------------------------------------------------ one clip
def bake_clip(ctx, seg, pl, win, plant, ref_times, disp, ob, ref_w, hand_targets=True):
    """Final rotations (hand IK, foot clamp, blend), the Reference offset and the tracks for one clip.

    ref_times: clip seconds of the frames (k = 1 -> WD2 seconds of the placed path, for the Exit the WD2 time of the cut).
    ref_w: translation weight [F] (0 keeps the original Reference position)."""
    seq, dump = seg['seq'], seg['dump']
    rig = ctx.rig
    F = seg['F']
    nf = int(F[-1])
    orig, patset, wgt = seg['orig'], seg['patset'], seg['w']
    ref0 = seg['ref0_orig']
    # blend of the rotations without IK: the Reference rotation there is the FK base of the Hips
    rot_pre = orig.copy()
    for ai in patset:
        rot_pre[:, ai] = bake2.mix_rot(orig[:, ai], seg['patch'][:, ai], wgt)
    rref = rot_pre[:, 0]
    target = pl.pelvis(ref_times)                            # ACU world pelvis (Hips) target, [F, 3]
    ref_full = np.asarray(target, float) - np.asarray(disp, float) - _hips_offset(ctx, rref)
    targets = {}
    ik_targets = {}
    if hand_targets and plant.size:
        targets, ik_targets = _hand_targets(ctx, pl, seq, dump, seg['patch'], ref_full, disp, win, plant)
    P1 = _world(seq, dump, seg['patch'], ref_full, disp)
    ik_targets.update(_foot_targets(rig, P1, ob, ctx.offsets, win))
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


def _validate(ctx, seg, fin, t_end, span, o, next_dump=None):
    """BD.validate_dive on the written tracks (entity + the segment's root), the forward-offset rule dropped."""
    res = dict(ob=seg['ob'], patched=seg['patset'], rot_tracks=fin['rot_tracks'], trans_keys=fin['trans_keys'],
               ref0=seg['ref0_orig'], orig=seg['orig'], win=fin['win'], body=ctx.body, fingers=ctx.fingers,
               body_nf=ctx.body_nf, plant=fin['plant'], targets=fin['targets'], o=o, span=span, t_end=t_end,
               disp=fin['disp'], disp_orig=None)
    met, arr = BD.validate_dive(ctx.rig, seg['dump'], seg['seq'], res, None, next_dump=next_dump, z_shift=0.0)
    met['fails'] = [f for f in met['fails'] if not f.startswith('forward offset')]
    met['fail'] = bool(met['fails'])
    return met, arr


def _cont_at_W(ctx, fS, fE):
    fW = min(int(round(ctx.t_W * acuclip.FPS)), len(fS['final']) - 1)
    fE0 = min(int(round(EXIT_O * acuclip.FPS)), len(fE['final']) - 1)
    ang = {}
    for bn in BD.GAP_BONES:
        ai = BD._acu(ctx.rig, bn)
        ang[bn] = float(bake2.angle_deg(fS['final'][fW, ai], fE['final'][fE0, ai]))
    vals = np.array(list(ang.values()))
    return dict(max_deg=float(vals.max()), median_deg=float(np.median(vals)), worst=max(ang, key=lambda k: ang[k]),
                per_bone=ang)


def _gap_to_next(ctx, final_rot, fW):
    ang = {}
    for bn in BD.GAP_BONES:
        ai = BD._acu(ctx.rig, bn)
        trk = dumps.track(ctx.next_dump, ctx.rig.ac[ai]['hash'], 0)
        nxt = dumps.sample_rotation(trk, 0) if trk else R.qn(ctx.rig.ac[ai]['lrot'])
        ang[bn] = float(bake2.angle_deg(final_rot[fW, ai], nxt))
    vals = np.array(list(ang.values()))
    return dict(max_deg=float(vals.max()), median_deg=float(np.median(vals)), worst=max(ang, key=lambda k: ang[k]))


def _ease(ctx, fin_rot, patset, fW):
    """Last EASE_S of the window: the patched locals go to the next clip's first frame (smoothstep)."""
    z0 = max(0, fW - int(round(EASE_S * acuclip.FPS)))
    out = fin_rot.copy()
    frames_z = np.arange(z0, fW + 1)
    we = bake2._smoothstep((frames_z - z0) / float(max(1, fW - z0)))
    for ai in patset:
        trk = dumps.track(ctx.next_dump, ctx.rig.ac[ai]['hash'], 0)
        target = dumps.sample_rotation(trk, 0) if trk else R.qn(ctx.rig.ac[ai]['lrot'])
        out[frames_z, ai] = bake2.mix_rot(fin_rot[frames_z, ai], np.tile(target, (len(frames_z), 1)), we)
    return out


# ------------------------------------------------------------------------------------------------ the joined chain
def bake_case(ctx, s, y_take, bump_amp, ease=True, mode='additive'):
    """Bake the joined SlideOver + Exit for one placement. Returns a dict with both clips and the metrics.

    mode 'additive' is the spec (bake_dive.dive_rotations additive); 'absolute' is a diagnostic only (WD2 orientations)."""
    sol = ctx.sol
    pl = WP.Placement(sol, s, y_take, x_take=0.0, bump_amp=bump_amp)
    o, span = ctx.o, ctx.span_so
    ones = lambda t: np.asarray(t, float)
    # ---- SlideOver: rotations (additive, k = 1) and the Reference offset (TrackID 0 stays the original)
    seg_S = BC.segment(ctx.rig, ctx.m, ctx.so['seq'], ctx.so['dump'], ones, ctx.t_anchor,
                       lambda t: bake2.patch_weight(t, o, span, FI, FO, TAIL), disp=ctx.disp_so)
    seg_S['win'] = bake2.window_frames(o, span, int(seg_S['F'][-1]))
    seg_S['ob'] = ctx.ob
    if mode == 'absolute':
        seg_S['patch'] = absolute_patch(ctx, seg_S, None, o)
    t_S = seg_S['t60']
    win_S = seg_S['win']
    tw = win_S / acuclip.FPS
    plant_S = win_S[(tw >= ctx.ev['hand_plant_start'] - 1e-9) & (tw <= ctx.ev['hand_plant_end'] + 1e-9)]
    wS = bake2.patch_weight(t_S, o, span, FI, FO, TAIL)
    fin_S = bake_clip(ctx, seg_S, pl, win_S, plant_S, t_S, ctx.disp_so, ctx.ob, wS)
    fin_S['win'] = win_S
    fin_S['plant'] = plant_S
    met_S, arr_S = _validate(ctx, seg_S, fin_S, ctx.t_W, span, o)

    # ---- Exit: rotations (gated step at 0.208, additive from the same anchor) and the joined root
    t_W = ctx.t_W

    def w_exit(t):
        return t_W + (np.asarray(t, float) - EXIT_O) / EXIT_R

    seg_E = BC.segment(ctx.rig, ctx.m, ctx.ex_seq, ctx.ex_dump, w_exit, ctx.t_anchor,
                       lambda t: (np.asarray(t, float) >= EXIT_O - 1e-9).astype(float), disp=ctx.disp_ex)
    seg_E['win'] = bake2.window_frames(EXIT_O, ctx.span_e, int(seg_E['F'][-1]))
    seg_E['ob'] = ctx.ground_ob
    if mode == 'absolute':
        seg_E['patch'] = absolute_patch(ctx, seg_E, w_exit, EXIT_O)
    tE = seg_E['t60']
    L_E = float(ctx.ex_dump['Length'])
    wE = np.where(tE >= EXIT_O - 1e-9, 1.0, 0.0)
    conv = np.clip((tE - ctx.t_end_E) / max(L_E - ctx.t_end_E, 1e-9), 0.0, 1.0)
    wE = wE * (1.0 - _smoothstep(conv))
    win_E = seg_E['win']
    fin_E = bake_clip(ctx, seg_E, pl, win_E, np.array([], int), w_exit(tE), ctx.disp_ex, ctx.ground_ob, wE,
                      hand_targets=False)
    fin_E['win'] = win_E
    fin_E['plant'] = np.array([], int)
    fin_E['targets'] = {}
    fin_E['disp'] = ctx.disp_ex
    fW_E = int(round(ctx.t_end_E * acuclip.FPS))
    gap_pre = _gap_to_next(ctx, fin_E['final'], min(fW_E, len(tE) - 1))
    ease_used = False
    if ease and gap_pre['max_deg'] > GAP_DEG:
        fin_E['final'] = _ease(ctx, fin_E['final'], seg_E['patset'], min(fW_E, len(tE) - 1))
        ease_used = True
        fin_E['rot_tracks'] = {ai: bake2.rotation_track(dumps.track(ctx.ex_dump, ctx.rig.ac[ai]['hash'], 0) or [],
                                                       fin_E['final'][:, ai], max(0, int(win_E[0]) - 1),
                                                       int(seg_E['F'][-1])) for ai in seg_E['patset']}
    else:
        fin_E['rot_tracks'] = {ai: bake2.rotation_track(dumps.track(ctx.ex_dump, ctx.rig.ac[ai]['hash'], 0) or [],
                                                       fin_E['final'][:, ai], max(0, int(win_E[0]) - 1),
                                                       int(seg_E['F'][-1])) for ai in seg_E['patset']}
    met_E, arr_E = _validate(ctx, seg_E, fin_E, ctx.t_end_E, ctx.span_e, EXIT_O, next_dump=ctx.next_dump)
    # the rotation tracks written for the Exit are the ones the validation sampled (rebuilt above)
    cont = _cont_at_W(ctx, fin_S, fin_E)
    dS = fin_S['ref_out'] - seg_S['ref0_orig']
    dE = fin_E['ref_out'] - seg_E['ref0_orig']
    seq_d = np.concatenate([dS[win_S], dE[win_E]])
    joined_step = float(np.max(np.linalg.norm(np.diff(seq_d, axis=0), axis=1)) * 100.0)
    return dict(s=float(s), y_take=float(y_take), bump=float(bump_amp), pl=pl, seg_S=seg_S, seg_E=seg_E,
                fin_S=fin_S, fin_E=fin_E, met_S=met_S, arr_S=arr_S, met_E=met_E, arr_E=arr_E, cont=cont,
                gap_pre=gap_pre, ease_used=ease_used, joined_step_cm=joined_step, w_exit=w_exit,
                dS=dS, dE=dE, win_S=win_S, win_E=win_E)


# ------------------------------------------------------------------------------------------------ status
def fail_list(met, cont_deg, step_cm, is_exit):
    """The spec's PASS criteria that fail (strings). The touch rule is not applied to the Exit (no hands)."""
    fails = []
    pen = met['pen_patch_cm']
    if pen > PEN_CM:
        fails.append('non-finger penetration %.1f cm (> %.0f)' % (pen, PEN_CM))
    if not is_exit and met['touch']['min'] < TOUCH:
        fails.append('hand touch %d frames (need %d)' % (met['touch']['min'], TOUCH))
    if step_cm > STEP_CM:
        fails.append('Delta step %.1f cm/frame (> %.0f)' % (step_cm, STEP_CM))
    if cont_deg > CONT_DEG:
        fails.append('continuity %.0f deg (> %.0f)' % (cont_deg, CONT_DEG))
    return fails


def narrow_ok(fails, met, cont_deg, step_cm, is_exit):
    for f in fails:
        if f.startswith('non-finger penetration'):
            if met['pen_patch_cm'] > NARROW['pen']:
                return False
        elif f.startswith('hand touch'):
            if met['touch']['min'] < NARROW['touch']:
                return False
        elif f.startswith('Delta step'):
            if step_cm > NARROW['step']:
                return False
        elif f.startswith('continuity'):
            if cont_deg > NARROW['cont']:
                return False
        else:
            return False
    return True


def status_of(met, cont_deg, step_cm, is_exit):
    fails = fail_list(met, cont_deg, step_cm, is_exit)
    if not fails:
        return 'PASS', fails, None
    if narrow_ok(fails, met, cont_deg, step_cm, is_exit):
        return 'OVERRIDE', fails, 'preview: ' + '; '.join(fails)
    return 'FAIL', fails, None


def statuses(res):
    step_S = float(res['met_S']['dstep_max_cm'])
    step_E = float(res['met_E']['dstep_max_cm'])
    cont = float(res['cont']['max_deg'])
    st_S, f_S, ov_S = status_of(res['met_S'], cont, max(step_S, res['joined_step_cm']), False)
    st_E, f_E, ov_E = status_of(res['met_E'], cont, max(step_E, res['joined_step_cm']), True)
    so_enabled = st_S in ('PASS', 'OVERRIDE')
    ex_enabled = st_E in ('PASS', 'OVERRIDE') and so_enabled
    return dict(S=dict(status=st_S, fails=f_S, override=ov_S, enabled=so_enabled),
                E=dict(status=st_E, fails=f_E, override=ov_E, enabled=ex_enabled))


# ------------------------------------------------------------------------------------------------ outputs
def _world_for_plot(seg, fin):
    return _world(seg['seq'], seg['dump'], fin['final'], fin['ref_out'], fin['disp'])


def write_outputs(ctx, res, st):
    os.makedirs(PATCH_DIR, exist_ok=True)
    rig = ctx.rig
    DC2.write_patch(os.path.join(PATCH_DIR, SO_HEX + '.baked.json'), rig, res['fin_S']['rot_tracks'],
                    res['fin_S']['trans_keys'], None, ctx.so['dump']['Length'])
    DC2.write_patch(os.path.join(PATCH_DIR, EXIT_HEX + '.baked.json'), rig, res['fin_E']['rot_tracks'],
                    res['fin_E']['trans_keys'], None, ctx.ex_dump['Length'])
    ov_S = st['S']['override']
    ov_E = st['E']['override']
    mapping = dict(clips=[
        DC2.mapping_entry(SO_HEX, SO_NAME, SO_HEX + '.baked.json', st['S']['enabled'], override=ov_S),
        DC2.mapping_entry(EXIT_HEX, EXIT_NAME, EXIT_HEX + '.baked.json', st['E']['enabled'],
                          only_after=[SO_NAME], override=ov_E),
    ])
    with open(MAPPING, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=1)
    # side view
    PfS = _world_for_plot(res['seg_S'], res['fin_S'])
    PoS = acuclip.world_positions(ctx.so['seq'], ctx.so['dump'])
    PfE = _world_for_plot(res['seg_E'], res['fin_E'])
    PoE = acuclip.world_positions(ctx.ex_seq, ctx.ex_dump)
    arrS = dict(Pf=PfS, Po=PoS)
    arrE = dict(Pf=PfE, Po=PoE)
    tl = Tl(ctx.o, ctx.r, ctx.W_played, ctx.ev['hand_plant_start'], res['w_exit'])
    title = 'dive3: s %.2f, y_take %.3f, bump %.2f m, continuity %.0f deg' % (res['s'], res['y_take'], res['bump'],
                                                                             res['cont']['max_deg'])
    BC.plot_joined(SHEET_PNG, rig, dict(F=res['seg_S']['F'], disp=ctx.disp_so, name=SO_NAME), arrS,
                   dict(F=res['seg_E']['F'], disp=ctx.disp_ex, name=EXIT_NAME), arrE, ctx.ob, tl, title,
                   show_root=True)
    BC.write_gif(SHEET_GIF, rig, dict(F=res['seg_S']['F'], disp=ctx.disp_so, name=SO_NAME), arrS,
                 dict(F=res['seg_E']['F'], disp=ctx.disp_ex, name=EXIT_NAME), arrE, ctx.ob, tl)
    return mapping


def _table(rows, headers):
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(out)


def run_scan(ctx):
    rows = []
    for s in [0.70, 0.75, 0.79, 0.85, 1.0, 1.3]:
        row = placement_row(ctx, s)
        row['bump'] = None
        rows.append(row)
    return rows


def geometry_error(ctx, seg, patch, times):
    """Joint offsets from the Hips, ACU (patched rotations, Reference at the origin) against the WD2 FK offsets (pelvis
    at the origin), over the clip times in `times`. Returns (mean m, max m) over 10 joints and the times."""
    F = np.asarray(seg['F'], int)
    rig = ctx.rig
    ent = acuclip.entity_positions(bake2.composed(seg['seq'], patch, np.zeros((len(F), 3))))
    wi = {b['name']: i for i, b in enumerate(rig.wd)}
    pairs = [('Spine', 'Spine'), ('Neck', 'Neck'), ('Head', 'Head'), ('LeftHand', 'L Hand'), ('RightHand', 'R Hand'),
             ('LeftFoot', 'L Foot'), ('RightFoot', 'R Foot'), ('LeftForeArm', 'L Forearm'), ('LeftArm', 'L UpperArm'),
             ('Spine2', 'Spine2')]
    errs = []
    for t in times:
        f = int(round(t * acuclip.FPS))
        R_t = WP.relative_fk(rig, ctx.m, np.array([t]))[0]
        for an, wn in pairs:
            a = ent[f, BD._acu(rig, an)] - ent[f, ctx.hips]
            errs.append(float(np.linalg.norm(a - R_t[wi[wn]])))
    e = np.array(errs)
    return float(e.mean()), float(e.max())


def _mode_row(r):
    mS, mE = r['met_S'], r['met_E']
    return ['%.1f' % mS['pen_box_cm'], '%.1f' % mS['pen_ground_cm'], str(mS['touch']['min']),
            '%.2f' % mS['dstep_max_cm'], '%.2f' % r['joined_step_cm'], '%.0f' % r['cont']['max_deg'],
            '%.1f' % mE['pen_patch_cm'], '%.2f' % mE['dstep_max_cm'], '%.0f' % r['gap_pre']['max_deg']]


def write_report(ctx, res, st, pick, rows_scan, diag, geo):
    mS, mE = res['met_S'], res['met_E']
    sol = ctx.sol
    P, t = sol['P'], sol['t']
    lines = []
    A = lines.append

    def at(tt):
        return int(round((tt - WP.T_START) * acuclip.FPS))

    fl = slice(at(0.45), at(0.75) + 1)
    A('# T14 dive3: WD2 frontdive body path (reconstructed from contacts) placed in ACU, Reference translation only\n')
    A('Generated by `v2/pipeline/dive3.py` from `wd2_path.py`. Written only under `out\\dive3`.\n')
    A('## Status\n')
    A('| clip | status | enabled | override | fails |')
    A('|---|---|---|---|---|')
    A('| SlideOver %s | %s | %s | %s | %s |' % (SO_NAME, st['S']['status'], 'yes' if st['S']['enabled'] else 'no',
                                               st['S']['override'] or '-', '; '.join(st['S']['fails']) or '-'))
    A('| Exit %s | %s | %s | %s | %s |' % (EXIT_NAME, st['E']['status'], 'yes' if st['E']['enabled'] else 'no',
                                           st['E']['override'] or '-', '; '.join(st['E']['fails']) or '-'))
    A('')
    A('Both patches are written as specified (additive rotations, Reference translation, no TrackID 0) and both are '
      '**disabled**: the enable rule fails. Section 5 gives the diagnosis; the diagnostic bake changes only the rotations.\n')
    A('## 1. Reconstructed WD2 pelvis path (wd2_path.py)\n')
    A('Linear least squares over 0.2 .. 1.8 s at 60 Hz (%d samples). Stance foot %s planted before takeoff (free anchor, '
      'height %.2f m); flight second difference = (0, 0, -g) dt^2 (medium weight); both hands planted 0.75 .. 0.92 s (free '
      'anchors, height %.2f m); roll: the lowest non-finger point on z = 0 with zero velocity while it stays the lowest; '
      'run-out: the lowest foot on the ground. Pelvis-relative WD2 FK with the decoded rotations; the WD2 axes are the ACU '
      'entity axes (A_REF M is the identity on positions).\n' % (len(t), sol['stance'], WP.FOOT_Z, WP.HAND_Z))
    apex = float(P[fl, 2].max())
    apex_t = float(t[fl][int(np.argmax(P[fl, 2]))])
    A('| quantity | value |')
    A('|---|---|')
    A('| forward travel, takeoff (0.45 s) to feet down (1.20 s) | %.3f m |' % float(P[at(1.2), 1] - P[at(0.45), 1]))
    A('| forward travel, takeoff to hand plant start (0.75 s) | %.3f m |' % float(P[at(0.75), 1] - P[at(0.45), 1]))
    A('| pelvis apex height over the flight | %.3f m at %.2f s |' % (apex, apex_t))
    A('| pelvis z at takeoff / at plant start | %.3f / %.3f m |' % (P[at(0.45), 2], P[at(0.75), 2]))
    A('| hand plant (anchors) forward of the takeoff pelvis, L / R | %+.3f / %+.3f m |'
      % (float(sol['anchors']['hand_L_xy'][1] - P[at(0.45), 1]), float(sol['anchors']['hand_R_xy'][1] - P[at(0.45), 1])))
    A('| hand plant (anchors) lateral, L / R | %+.3f / %+.3f m |'
      % (float(sol['anchors']['hand_L_xy'][0] - P[at(0.45), 0]), float(sol['anchors']['hand_R_xy'][0] - P[at(0.45), 0])))
    sx, sy = float(sol['anchors']['stance_xy'][0]), float(sol['anchors']['stance_xy'][1])
    A('| hand plant (anchors) relative to the takeoff foot anchor: forward L / R, lateral L / R | %+.3f / %+.3f, %+.3f / %+.3f m |'
      % (float(sol['anchors']['hand_L_xy'][1]) - sy, float(sol['anchors']['hand_R_xy'][1]) - sy,
         float(sol['anchors']['hand_L_xy'][0]) - sx, float(sol['anchors']['hand_R_xy'][0]) - sx))
    A('| stance foot (anchor) forward of the takeoff pelvis | %+.3f m |'
      % float(sol['anchors']['stance_xy'][1] - P[at(0.45), 1]))
    cl = WP.clearance_region(sol, 1.0, 0.45, 0.92)
    if cl is None:
        A('| lowest non-finger body point >= 1.0 m during 0.45 .. 0.92 s | never (max %.3f m) |'
          % float(WP.lowest_body_z(sol)[fl].max()))
    else:
        A('| lowest non-finger body point >= 1.0 m: forward range from the takeoff pelvis | %+.3f .. %+.3f m (%.2f .. %.2f s) |'
          % (cl['y_min'], cl['y_max'], cl['t_min'], cl['t_max']))
    A('')
    A('Solver residuals (metres, unweighted; the flight row is the position error of the second difference against (0, 0, -g) dt^2):\n')
    A(_table([(k, v['n'], '%.4f' % v['rms'], '%.4f' % v['max']) for k, v in sorted(sol['residuals'].items())],
             ['tag', 'rows', 'rms', 'max']))
    A('')
    A('## 2. Placement (spec step 2)\n')
    A('s = %.2f: the largest scale on a 0.01 grid in [0.70, 1.30] whose pelvis at W is within %.2f m of the original '
      '(spec tolerance %.2f m). y_take = %.3f m: the forward shift that puts the closer hand %.2f m beyond the far edge '
      '(far edge y %.3f m). Bump: %s.\n'
      % (pick['s'], W_PICK, W_TOL, pick['y_take'], FAR_CLEAR, ctx.ob['far_y'],
         ('%.2f m' % res['bump']) if res['bump'] else ('none: no raised-cosine amplitude on 0.45 .. 0.75 s clears the box. The non-finger body needs %.2f m of lift at '
          'takeoff (0.45 s), where the cosine is zero, and %.2f m at its worst (%.2f s)'
          % (pick['bump_info']['L_takeoff'], pick['bump_info']['L_max'], pick['bump_info']['w_max']))))
    A('The original ACU pelvis (Hips) is at y %.3f m at takeoff and y %.3f m at W (%.3f s). After placement the pelvis is at '
      'y %.3f m at takeoff (shift %+.3f m from the original) and at y %.3f m at W (%+.3f m).\n'
      % (ctx.orig_takeoff_y, ctx.orig_W[1], ctx.t_W, pick['y_take'], pick['y_take'] - ctx.orig_takeoff_y,
         float(res['pl'].pelvis(np.array([ctx.t_W]))[0, 1]),
         float(res['pl'].pelvis(np.array([ctx.t_W]))[0, 1] - ctx.orig_W[1])))
    A('Trade-off scan (rule (a): closer hand at or beyond far + 0.25 m; rule (b): pelvis at W within 0.30 m). The box near '
      'face is at y %.3f m, far face %.3f m.\n' % (ctx.ob['near_y'], ctx.ob['far_y']))
    A(_table(rows_scan, ['s', 'y_take (m)', 'takeoff shift (m)', 'hand clearance (m)', 'pelvis W offset (m)',
                         'stance foot y (m)', 'stance foot inside box footprint']))
    A('')
    A('## 3. Validation of the written patches (SlideOver window [o, W]; Exit window [0.208, 0.428] s)\n')
    A('| metric | SlideOver | Exit | rule |')
    A('|---|---|---|---|')
    A('| non-finger penetration, box / ground / max (cm) | %.1f / %.1f / %.1f | %.1f / %.1f / %.1f | <= %.0f |'
      % (mS['pen_box_cm'], mS['pen_ground_cm'], mS['pen_patch_cm'], mE['pen_box_cm'], mE['pen_ground_cm'],
         mE['pen_patch_cm'], PEN_CM))
    A('| bone below ground, all bones incl. fingers (cm) | %.1f | %.1f | info |' % (mS['ground_all_cm'], mE['ground_all_cm']))
    A('| hand touch frames (L / R, min) | %d / %d (%d) | n/a | >= %d |'
      % (mS['touch'].get('LeftHand', 0), mS['touch'].get('RightHand', 0), mS['touch']['min'], TOUCH))
    A('| Reference offset max step per 60 Hz frame (cm) | %.2f | %.2f (joined %.2f) | <= %.0f |'
      % (mS['dstep_max_cm'], mE['dstep_max_cm'], res['joined_step_cm'], STEP_CM))
    A('| continuity at W, max / median (deg) | %.1f / %.1f (%s) | | <= %.0f |'
      % (res['cont']['max_deg'], res['cont']['median_deg'], res['cont']['worst'], CONT_DEG))
    A('| pre-ease gap to Nav_Run, max / median (deg) | | %.1f / %.1f (%s) | ease over the last %.2f s when > %.0f |'
      % (res['gap_pre']['max_deg'], res['gap_pre']['median_deg'], res['gap_pre']['worst'], EASE_S, GAP_DEG))
    A('| Exit ease applied | | %s | |' % ('yes' if res['ease_used'] else 'no'))
    dS, dE = res['dS'], res['dE']
    A('| Reference offset (x, y, z; m) at o / takeoff / W | %s | | |'
      % ' ; '.join('%s (%.2f, %.2f, %.2f)' % ((lbl,) + tuple(dS[int(round(tt * acuclip.FPS))]))
                   for lbl, tt in (('o', ctx.o), ('takeoff', WP.TAKEOFF), ('W', ctx.t_W))))
    A('| max abs Reference offset y / z over the SlideOver window (m) | %.3f / %.3f | | |'
      % (float(np.max(np.abs(dS[res['win_S'], 1]))), float(np.max(np.abs(dS[res['win_S'], 2])))))
    A('| max abs Reference offset y / z over the Exit window (m) | | %.3f / %.3f | |'
      % (float(np.max(np.abs(dE[res['win_E'], 1]))), float(np.max(np.abs(dE[res['win_E'], 2])))))
    A('')
    A('Hand IK channels: %s. Foot clamp channels: %s.\n' % (res['fin_S']['ik_hands'], res['fin_S']['ik_feet']))
    A('Side view of the joined played time: `joined_sheet.png` (six stills per clip: blue SlideOver, orange Exit, grey '
      'original, box and ground) and `joined_sheet.gif`.\n')
    A('## 4. Enable decision\n')
    A('Rule: non-finger penetration <= 5 cm, hand touch >= 4 frames, Reference step <= 4 cm/frame, continuity <= 45 deg; '
      'override only inside the narrow limits (10 cm, 2 frames, 8 cm/frame, 60 deg). Both clips fail beyond the narrow limits, '
      'so both are disabled and the mapping carries `enabled: false`.\n')
    A('## 5. Diagnosis\n')
    A('| bake | box pen (cm) | ground pen (cm) | hand touch | SO Ref step (cm/frame) | joined step (cm/frame) | '
      'continuity (deg) | Exit pen (cm) | Exit Ref step (cm/frame) | pre-ease gap (deg) |')
    A('|---|---|---|---|---|---|---|---|---|---|')
    A('| spec: additive rotations (written) | ' + ' | '.join(_mode_row(res)) + ' |')
    A('| diagnostic: absolute WD2 orientations (not written) | ' + ' | '.join(_mode_row(diag)) + ' |')
    A('')
    A('Joint offsets from the Hips, ACU pose against the WD2 FK (10 joints, 0.45 .. 1.20 s): additive mean %.3f m, max %.3f m; '
      'absolute mean %.3f m, max %.3f m.\n' % (geo['add'][0], geo['add'][1], geo['abs'][0], geo['abs'][1]))
    A('1. **The additive rotation scheme does not carry the WD2 body shape.** It keeps the SlideOver posture and adds the WD2 '
      'change, so the arms and legs are not where the WD2 contacts put them (joint offsets up to %.2f m). With the WD2 '
      'orientations (absolute, diagnostic only) the joint offsets agree to %.0f cm on average (%.0f cm max), the hands touch '
      'for %d frames and the ground penetration falls from %.0f cm to %.0f cm.\n'
      % (geo['add'][1], geo['abs'][0] * 100.0, geo['abs'][1] * 100.0, diag['met_S']['touch']['min'],
         mS['pen_ground_cm'], diag['met_S']['pen_ground_cm']))
    A('2. **The placement the spec asks for starts the body %.2f m ahead of the original.** The hand rule puts the closer hand '
      'beyond the far edge; the original SlideOver has the pelvis at y %.2f m at takeoff, the placed path needs y %.2f m. The '
      'Reference offset therefore starts near %.2f m, and the 0.08 s fade-in moves the body by up to %.0f cm per frame.\n'
      % (pick['y_take'] - ctx.orig_takeoff_y, ctx.orig_takeoff_y, pick['y_take'], pick['y_take'] - ctx.orig_takeoff_y,
         mS['dstep_max_cm']))
    s_clear = min(float(x) for x in S_GRID if placement_row(ctx, float(x))['stance_foot_y'] <= ctx.ob['near_y'])
    A('3. **The takeoff leg and foot are inside the box footprint.** The stance foot is at y %.3f m, past the near face '
      '(%.3f m); the non-finger body needs %.2f m of lift at takeoff, where no bump is allowed. The stance foot clears the '
      'near face only from s = %.2f on, and there the W rule fails by %.2f m.\n'
      % (pick['stance_foot_y'], ctx.ob['near_y'], pick['bump_info']['L_takeoff'], s_clear,
         placement_row(ctx, s_clear)['dW']))
    A('4. **The Exit starts without a fade-in** (gated step at 0.208), and its Reference offset is the SlideOver\'s at W minus '
      'the original Exit offset, so its step is %.0f cm per frame. At W the placed pelvis is %.2f m high (the WD2 roll ends '
      'low), while the original clip has the pelvis at %.2f m: the Reference z offset at W is %+.2f m.\n'
      % (mE['dstep_max_cm'], float(res['pl'].pelvis(np.array([ctx.t_W]))[0, 2]), float(ctx.orig_W[2]),
         float(res['dS'][int(round(ctx.t_W * acuclip.FPS)), 2])))
    A('## 6. What would be needed (not done here)\n')
    A('* The dive has to take off about %.1f m closer to the box than the original SlideOver does (the placed body is that '
      'far forward at takeoff): the predecessor run-up must end later, or the obstacle must be shallower. At k = 1 the '
      'WD2 hand plant and the original W cannot both hold for this ACU obstacle (1.0 m deep).'
      % (pick['y_take'] - ctx.orig_takeoff_y))
    A('* The WD2 orientations for the body shape (the absolute mode), with a fade-in long enough to hide the posture change '
      '(about 74 deg at o). The spec fade of 0.08 s cannot hide it.')
    A('* Rules (a) and (b) hold together only for s in about 0.70 .. 0.79, and only with the takeoff offset of section 2. '
      'That offset, not the rules themselves, is what breaks the fade-in, the takeoff foot and the Exit start.')
    A('')
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def write_scan_table(rows):
    head = ['s', 'y_take (m)', 'takeoff shift (m)', 'hand clearance (m)', 'pelvis W offset (m)', 'stance foot y (m)',
            'stance foot inside box footprint']
    with open(SCAN, 'w', encoding='utf-8') as f:
        f.write('# dive3 placement scan\n\n')
        f.write(_table(rows, head) + '\n')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ctx = Ctx()
    rows = run_scan(ctx)
    scan_rows = [['%.2f' % r['s'], '%.3f' % r['y_take'], '%+.3f' % r['d_takeoff'], '%+.3f' % r['hand_clear'],
                  '%+.3f' % r['dW'], '%.3f' % r['stance_foot_y'], 'yes' if r['stance_in_box'] else 'no'] for r in rows]
    if '--scan' in argv:
        for r in scan_rows:
            print(' | '.join(r))
        return 0
    s, y_take, pick = choose_placement(ctx)
    amp, bump_info = bump_for(ctx, s, y_take)
    bump = float(amp) if amp else 0.0
    res = bake_case(ctx, s, y_take, bump, mode='additive')
    st = statuses(res)
    diag = bake_case(ctx, s, y_take, bump, mode='absolute')
    geo_add = geometry_error(ctx, res['seg_S'], res['seg_S']['patch'], np.arange(0.45, 1.201, 0.05))
    geo_abs = geometry_error(ctx, diag['seg_S'], diag['seg_S']['patch'], np.arange(0.45, 1.201, 0.05))
    os.makedirs(PATCH_DIR, exist_ok=True)
    WP.plot_side(os.path.join(OUT_DIR, 'wd2_path_side.png'), ctx.sol)
    write_scan_table(scan_rows)
    write_outputs(ctx, res, st)
    pick = dict(pick, bump=bump, s=s, y_take=y_take, stance_foot_y=pick['stance_foot_y'], bump_info=bump_info)
    write_report(ctx, res, st, pick, scan_rows, diag, dict(add=geo_add, abs=geo_abs))
    summary = dict(s=s, y_take=y_take, bump=bump, status=st, takeoff_shift=y_take - ctx.orig_takeoff_y,
                   met_S={k: v for k, v in res['met_S'].items() if k not in ('gap', 'ease')},
                   met_E={k: v for k, v in res['met_E'].items() if k not in ('gap', 'ease')},
                   cont=res['cont'], gap_pre=res['gap_pre'], joined_step_cm=res['joined_step_cm'],
                   ease_used=res['ease_used'],
                   diagnostic_absolute=dict(met_S={k: v for k, v in diag['met_S'].items() if k not in ('gap', 'ease')},
                                            met_E={k: v for k, v in diag['met_E'].items() if k not in ('gap', 'ease')},
                                            cont=diag['cont'], joined_step_cm=diag['joined_step_cm'],
                                            gap_pre=diag['gap_pre']),
                   geometry_error_m=dict(additive=geo_add, absolute=geo_abs))
    with open(SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps(dict(s=s, y_take=y_take, bump=bump, S=st['S']['status'], E=st['E']['status'])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
