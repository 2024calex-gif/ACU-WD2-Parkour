"""Bake the WD2 frontdive ("diveroll", data/wd2_events.json) onto the ACU Sprint SlideOver clips.

Variant D of the T9e family (the in-game favourite): fade-in 0.08 s, no fade-out, tail 'wd2'. Per clip, with clip seconds t,
played seconds p (t = o + r * p, out/match/play_model_slideover.json) and the played window W:

  1. time map     WD2 takeoff <-> ACU takeoff. WD2 seconds w = 0.45 + (p - p_takeoff) / k, with
                  k = min(k_acu, k_roll), k_acu = (ACU feet-down - ACU takeoff, played) / 0.75,
                  k_roll = (W - p_takeoff) / 0.73.
  2. rotations    additive retarget (default): each patched bone's world orientation is the WD2 world change since o
                  (A_REF * Wt(t) * conj(A_REF * Wt(o))) applied to the clip's own world orientation. At o the patch IS the
                  original clip, and the dive's inversion is carried by the Reference (Hips has no track and rides on it), as
                  the game's own landing-roll clip does (its Reference swings 174 deg). The bake2-style absolute form
                  (world = A_REF * Wt) is rot_mode='absolute'; it is about 74 deg off the clip's own posture at o + 0.08 s
                  (reference clip) once the fade-in is done, so it is reported for comparison only.
  3. root         Reference position = original + Delta, Delta = (0, Dy, Dz):
                  Dy  forward bump of amplitude A, peaking at the plant start and back to 0 at the window end. A is searched
                      on DY_GRID (0 to 1 m): the smallest A with no lift conflict and reachable hands at the plant start. When
                      no A is feasible the least-conflict A is used and the clip fails. A is reported.
                  Dz  signed lift envelope (bake2 solve_lift_t9e) with: clearance of the non-finger body from the obstacle box
                      and the ground (hands and arm chain excluded in the plant: their IK owns them), the reach bound of the
                      plant (shoulders reach the hand targets), and a touch bound in the roll (the lowest non-finger point
                      rests on the ground).
  4. hands        plant targets beyond the far edge (y = far + HAND_MARGIN, z = ground + hand offset); hand IK with a weight
                  while the target is reachable (bake2 T9e IK helpers).
  5. feet         foot clamp IK (bake2 T9e).
  6. crossfade    variant D weight: smoothstep 0.08 s from o, then 1 to the clip end.
  7. validation   the composed pose the written tracks produce (60 Hz, as the plugin samples them).

Writes only under out\\dive (patches\\, report.md). Imports bake2 read-only; bake2.py is not modified.
Run:  python bake_dive.py       (all Sprint SlideOver clips; needs the WD2 .mab data and the ACU dumps)
"""
import json
import math
import os
import zlib

import numpy as np

import acuclip
import bake2
import config
import contacts
import dumps
import ik
import retarget as R
import wd2

FPS = acuclip.FPS
DIVE_KEY = 'diveroll_frontdive'
PLAY_MODEL_SO = os.path.join(config.OUT, 'match', 'play_model_slideover.json')
DIVE_OUT = os.path.join(config.OUT, 'dive')
PATCH_DIR = os.path.join(DIVE_OUT, 'patches')
REPORT = os.path.join(DIVE_OUT, 'report.md')
MAPPING_DIVE = os.path.join(PATCH_DIR, 'mapping_dive.json')
REF_CLIP = 'xx_Parkour_Sprint_SlideOver_Fd100cm-Up100cm'     # the clip whose o / r / W give the reference map
FALLBACK_NEXT = 'xx_Parkour_Muraille_Exit_Fd100cm-Dn100cm'

# WD2 frontdive events (seconds of the WD2 clip, data/wd2_events.json "diveroll_frontdive")
WD2_TAKEOFF = 0.45
WD2_PLANT = (0.75, 0.92)
WD2_ROLL_END = 1.18
WD2_FEET = 1.2
K_ACU_SPAN = WD2_FEET - WD2_TAKEOFF         # 0.75 s
K_ROLL_SPAN = WD2_ROLL_END - WD2_TAKEOFF    # 0.73 s

# variant D
FI = 0.08                       # s, fade-in from the original pose at o
FO = 0.0                        # s, no fade-out
TAIL = bake2.TAIL_WD2

HAND_MARGIN = 0.10              # m, plant targets lie this far beyond the far edge
DY_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 3)    # m, candidate forward amplitudes
DY_SMALL = 0.25                 # m, PASS needs max |Dy| <= this (a "small" forward offset)
TOUCH_FRAMES = 4                # frames both hands stay on their targets (bake2 T9D_TOUCH)
HAND_TOL_CM = 5.0
PEN_CM = 5.0                    # PASS: non-finger body inside the obstacle box or below ground
GROUND_CM = 3.0                 # PASS: any bone below z = 0 by more than this, inside the played window
DSTEP_CM = 3.0                  # PASS: largest change of Delta between two 60 Hz frames
GAP_BONES = ('Reference', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftUpLeg', 'LeftLeg', 'LeftFoot',
             'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftArm', 'LeftForeArm', 'RightArm', 'RightForeArm')
GAP_WARN_DEG = 45.0             # report flag: largest local-rotation gap at W on these bones
EASE_S = 0.15                   # s, last part of the played window of the ease-to-exit variant (report only)


# ------------------------------------------------------------------------------------------------ inputs
def load_dive_events():
    with open(bake2.EVENTS_JSON, encoding='utf-8') as f:
        return json.load(f)[DIVE_KEY]


def load_play_model_so(path=PLAY_MODEL_SO):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def targets_dive():
    """[(handle, name, dump)] for the Sprint SlideOver clips, in dump-file order."""
    out = []
    for handle, name, _path in dumps.list_dumps():
        if name.startswith('xx_Parkour_Sprint_SlideOver_'):
            out.append((handle, name, dumps.load_by_name(name)))
    return out


def clip_model(name, model, length):
    """{'o', 'r', 'W', 'source', 'capture'} for a Sprint SlideOver clip.

    A clip with a play-model entry uses that entry's map (own when map_used.source == 'own', otherwise the reference map
    marked ASSUMED) and its measured played window W. A clip with no entry (no capture) uses the reference map and the
    reference window, marked ASSUMED, and is never enabled."""
    ref = model['per_clip'][REF_CLIP]
    pc = model['per_clip'].get(name)
    if pc is None:
        return dict(o=float(ref['map_used']['o']), r=float(ref['map_used']['r']), W=float(ref['window']),
                    source='no capture: reference map and window (ASSUMED)', capture=False, length=float(length))
    mu = pc['map_used']
    src = 'own map' if mu.get('source') == 'own' else 'reference map (ASSUMED)'
    return dict(o=float(mu['o']), r=float(mu['r']), W=float(pc['window']), source=src, capture=True,
                length=float(length), next_counts=dict(pc.get('next_clip_counts', {})))


def next_clip_for(mp):
    """Exit clip used for the gap at W: the most frequent next clip in the captures (Muraille_Exit only), else default."""
    counts = {k: v for k, v in mp.get('next_counts', {}).items() if k.startswith('xx_Parkour_Muraille_Exit_')}
    if not counts:
        return FALLBACK_NEXT
    return max(sorted(counts), key=lambda k: counts[k])


# ------------------------------------------------------------------------------------------------ 1. time map
class Clock:
    """Clip seconds <-> WD2 seconds for one clip. p = (t - o) / r played; w = 0.45 + (p - p_takeoff) / k."""

    def __init__(self, o, r, p_takeoff, p_feet, W):
        self.o = float(o)
        self.r = float(r)
        self.W = float(W)
        self.p_takeoff = float(p_takeoff)
        self.p_feet = float(p_feet)
        self.k_acu = (self.p_feet - self.p_takeoff) / K_ACU_SPAN
        self.k_roll = (self.W - self.p_takeoff) / K_ROLL_SPAN
        self.k = min(self.k_acu, self.k_roll)

    def w(self, t):
        """WD2 seconds at clip seconds t (array in, array out)."""
        p = (np.asarray(t, float) - self.o) / self.r
        return WD2_TAKEOFF + (p - self.p_takeoff) / self.k

    def t_of_w(self, w_s):
        """Clip seconds at which the WD2 time is w_s."""
        return self.o + self.r * (self.p_takeoff + (float(w_s) - WD2_TAKEOFF) * self.k)

    def feet_wd2(self):
        """WD2 time at the ACU feet-down (played p_feet): WD2_FEET when k = k_acu, earlier/later otherwise."""
        return float(self.w(self.o + self.r * self.p_feet))


def make_clock(ev, mp):
    if ev.get('takeoff') is None or ev.get('landing') is None:
        raise ValueError('ACU takeoff or landing event missing')
    tk = float(ev['takeoff']['seconds'])
    fd = float(ev['landing']['seconds'])
    return Clock(mp['o'], mp['r'], (tk - mp['o']) / mp['r'], (fd - mp['o']) / mp['r'], mp['W'])


def bump(t, o, t_peak, t_end):
    """Forward bump B(t): smoothstep from 0 at o to 1 at t_peak, back to 0 at t_end; 0 outside [o, t_end]."""
    t = np.asarray(t, float)
    up = np.clip((t - o) / max(t_peak - o, 1e-9), 0.0, 1.0)
    dn = np.clip((t_end - t) / max(t_end - t_peak, 1e-9), 0.0, 1.0)
    b = np.where(t <= t_peak, up, dn)
    b = np.where((t < o) | (t > t_end), 0.0, b)
    return b * b * (3.0 - 2.0 * b)


# ------------------------------------------------------------------------------------------------ 2. rotations
def patch_set(rig, dump):
    """Bone indices that get a rotation track: mapped bones with a track in the dump, and the Reference (0) if it has one."""
    rot = bake2.rot_tracks(rig, dump)
    ps = {ai for ai in set(rig.map.values()) if ai in rot}
    if 0 in rot:
        ps.add(0)
    return sorted(ps), rot


ROT_ADDITIVE = 'additive'
ROT_ABSOLUTE = 'absolute'


def dive_rotations(rig, m, clock, frames, patset, rot, mode=ROT_ADDITIVE):
    """{ai: [F, 4]} patched local rotations at the 60 Hz frames for the bones in patset (ai 0 = Reference).

    Additive retarget. For each bone, D(t) = (A_REF * Wt(t)) * conj(A_REF * Wt(w0)) is the WD2 world rotation change since
    the patch start w0 = clock.w(o), in ACU axes (Rb cancels). The patched world orientation is D(t) * W_dump(t), where
    W_dump is the ACU clip's own world orientation. So at o the patch equals the original clip exactly, the clip keeps its
    base lean (its Reference is about 29 deg off A_REF), and the WD2 motion is added on top. Hips (no track) follows the
    Reference: D_hips drives the Reference, so the pelvis carries the dive's flip. Each patched child's local is
    conj(actual parent world) * desired, so its world equals the desired one.
    """
    if mode not in (ROT_ADDITIVE, ROT_ABSOLUTE):
        raise ValueError('rotation mode must be additive or absolute, got %r' % (mode,))
    n = len(rig.ac)
    hips = rig.ac_by_hash[zlib.crc32(b'Hips')]
    out = {ai: np.zeros((len(frames), 4)) for ai in patset}
    t = np.asarray(frames, float) / FPS
    w_s = clock.w(t)
    t_anchor = round(clock.o * FPS) / FPS          # the first frame of the patch on the 60 Hz grid
    _L0, Wt0 = bake2.wd2_pose_at(rig, m, float(clock.w(t_anchor)))
    Wt0 = [R.qn(q) for q in Wt0]

    def delta(Wt_now, ai):
        if mode == ROT_ABSOLUTE:
            return R.qn(R.qmul(R.A_REF, Wt_now[ai]))
        a = R.qn(R.qmul(R.A_REF, Wt_now[ai]))
        b = R.qn(R.qmul(R.A_REF, Wt0[ai]))
        return R.qn(R.qmul(a, R.qconj(b)))

    for j, f in enumerate(frames):
        _Ls, Wt = bake2.wd2_pose_at(rig, m, float(w_s[j]))
        Wd = [None] * n                    # the dump's own world orientation of every bone at this frame
        for ai in range(n):
            loc = bake2.orig_local(rig, rot, ai, int(f))
            Wd[ai] = R.qn(loc) if rig.ac[ai]['parent'] < 0 else R.qn(R.qmul(Wd[rig.ac[ai]['parent']], loc))
        W = [None] * n                     # the patched world orientation of every bone at this frame
        for ai in range(n):
            par = rig.ac[ai]['parent']
            if ai == 0:
                if mode == ROT_ABSOLUTE:
                    W[0] = R.qn(R.qmul(R.A_REF, _Ls[0]))          # the WD2 pelvis delta on the bind (Reference = Rp)
                else:
                    W[0] = R.qn(R.qmul(delta(Wt, hips), Wd[0]))   # the pelvis change since o on the clip's own Reference
                if 0 in patset:
                    out[0][j] = W[0]
                continue
            if ai in patset:
                base = R.qn(R.qmul(R.A_REF, Wt[ai])) if mode == ROT_ABSOLUTE else Wd[ai]
                desired = R.qn(R.qmul(delta(Wt, ai), base)) if mode == ROT_ADDITIVE else base
                out[ai][j] = R.qn(R.qmul(R.qconj(W[par]), desired))
                W[ai] = desired
            else:
                loc = bake2.orig_local(rig, rot, ai, int(f))
                W[ai] = R.qn(R.qmul(W[par], loc))
    return out


# ------------------------------------------------------------------------------------------------ 3. bake
def _acu(rig, name):
    return rig.ac_by_hash[zlib.crc32(name.encode())]


def world(seq, dump, rotl, ref0, dy, dz, disp=None):
    """Composed world positions [F, B, 3] with the Reference at ref0 + (0, dy, dz).

    disp: the root displacement [F, 3] to use instead of the dump's TrackID 0 (dive_chain2 replaces the root); None
    keeps the dump's own displacement."""
    delta = np.zeros((len(ref0), 3))
    delta[:, 1] = dy
    delta[:, 2] = dz
    composed = bake2.composed(seq, rotl, ref0 + delta)
    if disp is None:
        return acuclip.world_positions(composed, dump)
    return acuclip.entity_positions(composed) + np.asarray(disp, float)[:, None, :]


def _forward_stage(ctx, A):
    """Everything the root solve needs for forward amplitude A: lift need, reach bound, touch bound, Dz envelope.

    Hands are IK-controlled to their plant targets, so during the plant they are left out of the lift and touch bounds
    (their clearance is what the IK establishes); the reach bound is what keeps them on the targets.
    """
    rig, seq, dump, ob = ctx['rig'], ctx['seq'], ctx['dump'], ctx['ob']
    win, plant, roll = ctx['win'], ctx['plant'], ctx['roll']
    body_nf, body_nh = ctx['body_nf'], ctx['body_nh']
    dy_full = A * ctx['B']
    zero = np.zeros(len(dy_full))
    Pb = world(seq, dump, ctx['blend'], ctx['ref0'], dy_full, zero)
    Pp = world(seq, dump, ctx['patch'], ctx['ref0'], dy_full, zero)
    Lfull = bake2.lift_need_signed(Pb[win][:, body_nf], ob)
    Lnh = bake2.lift_need_signed(Pb[win][:, body_nh], ob)
    Lw = Lfull.copy()
    in_plant = np.isin(win, plant)
    Lw[in_plant] = Lnh[in_plant]
    U_win = np.full(len(win), np.inf)
    targets = {}
    reach_ok = True
    if plant.size:
        p0 = int(plant[0])
        pos = plant - win[0]
        reach_sides = []
        for side, h in (('Left', 'LeftHand'), ('Right', 'RightHand')):
            ua, la, ha = _acu(rig, side + 'Arm'), _acu(rig, side + 'ForeArm'), _acu(rig, h)
            p = Pp[p0, ha]
            tgt = np.array([p[0], float(ob['far_y']) + HAND_MARGIN, ctx['offsets']['Hand']])
            targets[h] = tgt
            sh = Pp[plant, ua]
            arm = np.linalg.norm(Pp[plant, la] - Pp[plant, ua], axis=1) + np.linalg.norm(Pp[plant, ha] - Pp[plant, la], axis=1)
            reach_sides.append(bake2.reach_upper(sh, tgt, bake2.T9E_REACH * arm))
        U_plant = np.min(np.stack(reach_sides), axis=0)
        U_win[pos] = U_plant
        for d in range(1, bake2.T9E_RELAX_FRAMES + 1):
            for jj, edge in ((int(pos[0]) - d, U_plant[0]), (int(pos[-1]) + d, U_plant[-1])):
                if 0 <= jj < len(win) and np.isinf(U_win[jj]) and U_win[jj] > 0:
                    U_win[jj] = edge + bake2.T9E_RELAX_M * d / bake2.T9E_RELAX_FRAMES if np.isfinite(edge) else edge
        n_t = min(TOUCH_FRAMES, plant.size)
        reach_ok = bool(np.all(np.isfinite(U_plant[:n_t])) and np.all(Lw[pos][:n_t] <= U_plant[:n_t] + 1e-9))
    if roll.size:
        pos_r = roll - win[0]
        z_low = Pp[roll][:, body_nf, 2].min(axis=1)
        U_win[pos_r] = np.minimum(U_win[pos_r], -z_low)
    D_w, linfo = bake2.solve_lift_t9e(Lw, U_win, ramp_edges=True)
    conflicts = int(np.sum(Lw > U_win + 1e-9))
    feasible = bool(reach_ok and conflicts == 0)
    plant_first = None
    if plant.size:
        plant_first = (float(Lw[plant - win[0]][0]), float(U_win[plant - win[0]][0]))
    return dict(A=float(A), dy_full=dy_full, D_w=D_w, linfo=linfo, U=U_win, L=Lw, targets=targets, feasible=feasible,
                conflicts=conflicts, reach_ok=reach_ok, plant_first=plant_first)


def bake_clip_dive(rig, m, dump, seq, ev, mp, offsets, fi=FI, fo=FO, tail=TAIL, rot_mode=ROT_ADDITIVE):
    """One clip. Returns res (keys in the return statement at the end)."""
    L_ = float(dump['Length'])
    ob = ev['obstacle']
    F = np.asarray(seq.frames, int)
    nf = int(F[-1])
    t60 = F / FPS
    o, r, W = mp['o'], mp['r'], mp['W']
    span = r * W                              # clip seconds of the played window
    t_end = o + span
    clock = make_clock(ev, mp)
    patset, rot = patch_set(rig, dump)
    orig = np.asarray(seq.local_rot, float)
    ref0 = np.asarray(seq.local_pos[:, 0], float)

    pat_rot = dive_rotations(rig, m, clock, list(F), patset, rot, mode=rot_mode)
    patch = orig.copy()
    for ai in patset:
        patch[:, ai] = pat_rot[ai]
    wgt = bake2.patch_weight(t60, o, span, fi, fo, tail)
    blend = orig.copy()
    for ai in patset:
        blend[:, ai] = bake2.mix_rot(orig[:, ai], pat_rot[ai], wgt)

    body = sorted(set(rig.map.values()))
    fingers = bake2.finger_bones(rig, body)
    body_nf = [bi for bi in body if bi not in set(fingers)]
    win = bake2.window_frames(o, span, nf)
    w_win = clock.w(t60[win])
    plant = win[(w_win >= WD2_PLANT[0] - 1e-9) & (w_win <= WD2_PLANT[1] + 1e-9)]
    t_peak = clock.t_of_w(WD2_PLANT[0])
    if t_peak >= t_end - 1e-6:
        t_peak = o + 0.5 * span
    B = bump(t60, o, t_peak, t_end)
    disp = acuclip.markers(dump, F)['disp']

    # plant: the hands and the arm chain are IK-controlled (targets and elbow), so they leave the lift bounds
    arm_chain = {_acu(rig, n) for n in ('LeftArm', 'LeftForeArm', 'LeftHand', 'RightArm', 'RightForeArm', 'RightHand')}
    body_nh = [bi for bi in body_nf if bi not in arm_chain]
    roll = win[(w_win > WD2_PLANT[1] + 1e-9) & (w_win <= WD2_ROLL_END + 1e-9)]
    ctx = dict(rig=rig, seq=seq, dump=dump, ob=ob, win=win, plant=plant, roll=roll, body_nf=body_nf, body_nh=body_nh,
               B=B, patch=patch, blend=blend, ref0=ref0, offsets=offsets)

    # forward amplitude: smallest A on the grid for which the plant is feasible (reach + no conflict)
    stages = []
    chosen = None
    for A in DY_GRID:
        st = _forward_stage(ctx, A)
        stages.append(st)
        if st['feasible']:
            chosen = st
            break
    if chosen is None:
        # no feasible amplitude on the grid: keep the one with the fewest conflicts (the smallest A on a tie)
        chosen = min(stages, key=lambda st: (st['conflicts'], st['A']))
    A = chosen['A']
    dy_full = chosen['dy_full']
    dz_full = np.zeros(len(F))
    dz_full[win] = chosen['D_w']
    ref1 = ref0.copy()
    ref1[:, 1] += dy_full
    ref1[:, 2] += dz_full
    delta = np.stack([np.zeros(len(F)), dy_full, dz_full], axis=1)

    # hand IK toward the plant targets (weight while reachable), after the offset
    P1 = world(seq, dump, patch, ref0, dy_full, dz_full)
    ik_in = bake2.composed(seq, patch, ref1)
    targets = {}
    hand_targets = chosen['targets']
    reach_info = {}
    if plant.size:
        p0, pend = int(plant[0]), int(plant[-1])
        frr = np.arange(p0, pend + 1)
        for side, h in (('Left', 'LeftHand'), ('Right', 'RightHand')):
            ua, la, ha = _acu(rig, side + 'Arm'), _acu(rig, side + 'ForeArm'), _acu(rig, h)
            tgt = hand_targets[h]
            d = np.linalg.norm(tgt[None, :] - P1[frr, ua], axis=1)
            arm1 = np.linalg.norm(P1[frr, la] - P1[frr, ua], axis=1) + np.linalg.norm(P1[frr, ha] - P1[frr, la], axis=1)
            reach = d <= bake2.T9E_REACH * arm1
            wh, u = bake2._ik_weights(len(F), win, p0, pend, reach, ramp=bake2.T9D_RAMP)
            reach_info[h] = dict(first_unreachable=int(u) if u <= pend else None, reach_frames=int(reach.sum()))
            targets[h] = [(int(f), tgt, float(wh[f])) for f in np.nonzero(wh > 0)[0]]

    # foot clamp (bake2 T9e): the lower of Foot / FootThumb1 below its support is raised to it
    foot_info = {}
    for side in ('Left', 'Right'):
        foot, toe = side + 'Foot', side + 'FootThumb1'
        need = np.zeros(len(F))
        for f in win:
            best = 0.0
            for name, off in ((foot, offsets['Foot']), (toe, offsets['FootThumb1'])):
                y, z = P1[f, _acu(rig, name), 1], P1[f, _acu(rig, name), 2]
                surf = float(ob['height']) if float(ob['near_y']) <= y <= float(ob['far_y']) else 0.0
                best = max(best, surf + off - z)
            need[f] = best
        pos_idx = np.nonzero(need > 0)[0]
        foot_info[foot] = dict(frames=int(pos_idx.size), max_raise_cm=float(need.max() * 100.0))
        if pos_idx.size == 0:
            continue
        wf = bake2.blend_mod_fade(len(F), bake2._runs(pos_idx), bake2.T9D_RAMP)
        for f in np.nonzero(wf > 0)[0]:
            k = pos_idx[np.argmin(np.abs(pos_idx - f))]
            tgt = P1[f, _acu(rig, foot)] + np.array([0.0, 0.0, need[k]])
            targets.setdefault(foot, []).append((int(f), tgt, float(wf[f])))

    if targets:
        root_rot = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(F), 1))
        ik_out = ik.solve_sequence(ik_in, root_rot=root_rot, root_pos=disp, targets=targets)
        changed = np.nonzero(np.any(np.abs(ik_out.local_rot - ik_in.local_rot) > 1e-12, axis=(0, 2)))[0]
        bad = sorted(set(int(c) for c in changed) - set(patset))
        if bad:
            raise ValueError('IK changed bones without a rotation track: %s' % [rig.ac[b]['name'] for b in bad])
        patch_ik = np.asarray(ik_out.local_rot, float)
    else:
        patch_ik = patch

    final = orig.copy()
    for ai in patset:
        final[:, ai] = bake2.mix_rot(orig[:, ai], patch_ik[:, ai], wgt)
        final[:, ai] /= np.linalg.norm(final[:, ai], axis=1, keepdims=True)   # IK output can drift ~1e-4 off unit length

    lo = max(0, int(win[0]) - 1)
    hi = nf if tail != bake2.TAIL_ORIG else min(nf, int(win[-1]) + 1)
    rot_tracks = {}
    for ai in patset:
        ok = dumps.track(dump, rig.ac[ai]['hash'], 0) or []
        rot_tracks[ai] = bake2.rotation_track(ok, final[:, ai], lo, hi)
    ref_keys = dumps.track(dump, bake2.REF_TID, 3) or []
    trans_keys = bake2.translation_track(ref_keys, ref1, lo, hi)

    lin = chosen['linfo']
    solve_info = dict(lift_max_cm=lin['max_D_cm'], lift_min_cm=lin['min_D_cm'], lift_step_cm=lin['max_step_cm'],
                      lift_conflict=lin['conflict'], lift_reach_limited=lin['reach_limited'],
                      forced_start=lin['forced_start'], forced_end=lin['forced_end'], A_m=A,
                      feasible=chosen['feasible'], stages_tried=len(stages), conflicts=chosen['conflicts'],
                      reach_ok=chosen['reach_ok'],
                      dy_max_cm=float(np.max(np.abs(dy_full))) * 100.0, dz_max_cm=float(np.max(np.abs(dz_full))) * 100.0)
    stages_rec = [dict(A=st['A'], plant_first=st['plant_first'], conflicts=st['conflicts']) for st in stages]
    return dict(name=None, ob=ob, stages=stages_rec, patched=patset, rot_tracks=rot_tracks, trans_keys=trans_keys, final=final, orig=orig, patch=patch,
                ref0=ref0, ref_new=ref1, delta=delta, win=win, plant=plant, roll=roll,
                targets=hand_targets, body=body, body_nf=body_nf, fingers=sorted(fingers), clock=clock,
                o=float(o), span=float(span), t_end=float(t_end), fades=(fi, fo), tail=tail, B=B,
                solve_info=solve_info, reach=reach_info, foot=foot_info, disp=disp, L=chosen['L'], U=chosen['U'],
                pure=dict(clip=(o + fi, t_end), wd2=(float(clock.w(o + fi)), float(clock.w(t_end)))),
                time_map=clock)


# ------------------------------------------------------------------------------------------------ 4. validation
def _rot_full(orig, rot60, patset):
    rf = orig.copy()
    for ai in patset:
        rf[:, ai] = rot60[ai]
    return rf


def validate_dive(rig, dump, seq, res, mp, next_dump=None, z_shift=0.0):
    """Metrics of the written tracks (sampled as the plugin samples them), plus arrays for the plots."""
    F = np.asarray(seq.frames, int)
    ob = res['ob']
    win = res['win']
    body, body_nf, fingers = res['body'], res['body_nf'], res['fingers']
    patset = res['patched']
    orig = res['orig']
    rot60 = {ai: bake2.sample_track_rot(res['rot_tracks'][ai], F) for ai in patset}
    rf = _rot_full(orig, rot60, patset)
    pos = bake2.sample_track_pos(res['trans_keys'], F)
    delta = pos - res['ref0']
    shift = np.array([0.0, 0.0, float(z_shift)])
    disp_w = res.get('disp')           # the root that plays (None: the dump's own TrackID 0)
    disp_o = res.get('disp_orig')      # the original root for the comparison (None: the dump's own)
    if disp_w is None:
        Pf = acuclip.world_positions(bake2.composed(seq, rf, pos + shift), dump)
    else:
        Pf = acuclip.entity_positions(bake2.composed(seq, rf, pos + shift)) + np.asarray(disp_w, float)[:, None, :]
    if disp_o is None:
        Po = acuclip.world_positions(seq, dump) + shift
    else:
        Po = acuclip.entity_positions(seq) + np.asarray(disp_o, float)[:, None, :] + shift

    box_nf, gnd_nf = bake2.penetration_parts(Pf[win][:, body_nf], ob)
    box_f, gnd_f = bake2.penetration_parts(Pf[win][:, fingers], ob) if fingers else (0.0, 0.0)
    _b, gnd_all = bake2.penetration_parts(Pf[win][:, body], ob)
    box_o, gnd_o = bake2.penetration_parts(Po[win][:, body_nf], ob)

    plant = res['plant']
    hands = ('LeftHand', 'RightHand')
    hand_err = hand_z = touch = None
    if plant.size:
        touch = {}
        errs, zs = [], []
        for h in hands:
            ai = _acu(rig, h)
            errs.append(np.linalg.norm(Pf[plant, ai] - res['targets'][h], axis=1))
            zs.append(Pf[plant, ai, 2] * 100.0)
        hand_err = float(np.max(np.stack(errs)) * 100.0)
        zz = np.concatenate(zs)
        hand_z = dict(min=float(zz.min()), median=float(np.median(zz)), max=float(zz.max()))
        touch = bake2.touch_frames(Pf, plant, res['targets'], rig, HAND_TOL_CM)
    else:
        touch = dict(LeftHand=0, RightHand=0, min=0)

    dstep = np.linalg.norm(np.diff(delta, axis=0), axis=1) * 100.0
    dstep_max = float(dstep.max()) if dstep.size else 0.0
    dy_max = float(np.max(np.abs(delta[:, 1]))) * 100.0
    dz_max = float(np.max(np.abs(delta[:, 2]))) * 100.0

    t = F / FPS
    even = np.where((F % 2) == 0)[0]
    in_win = even[np.isin(even, win)]
    o, span = res['o'], res['span']
    b1 = even[(t[even] >= o - FI) & (t[even] <= o + FI)]
    b2 = even[(t[even] >= o + span - FI) & (t[even] <= o + span + FI)]
    step_p = bake2._step_deg(rf, in_win, patset)
    bnd_p = max(bake2._step_deg(rf, b1, patset), bake2._step_deg(rf, b2, patset))
    bnd_o = max(bake2._step_deg(orig, b1, patset), bake2._step_deg(orig, b2, patset))

    gap = dict(max_deg=None, median_deg=None, worst=None, per_bone={}, next=None)
    if next_dump is not None:
        fW = int(round(res['t_end'] * FPS))
        ang = {}
        for bn in GAP_BONES:
            ai = _acu(rig, bn)
            mine = rf[min(fW, len(F) - 1), ai]
            trk = dumps.track(next_dump, rig.ac[ai]['hash'], 0)
            nxt = dumps.sample_rotation(trk, 0) if trk else R.qn(rig.ac[ai]['lrot'])
            ang[bn] = float(bake2.angle_deg(mine, nxt))
        vals = np.array(list(ang.values()))
        worst = max(ang, key=lambda k: ang[k])
        gap = dict(max_deg=float(vals.max()), median_deg=float(np.median(vals)), worst=worst, per_bone=ang,
                   next=next_dump['Name'])

    ease = None
    if next_dump is not None:
        fW = int(round(res['t_end'] * FPS))
        nfz = int(round(EASE_S * FPS))
        z0 = max(0, fW - nfz)
        exit_loc = {}
        for ai in patset:
            trk = dumps.track(next_dump, rig.ac[ai]['hash'], 0)
            exit_loc[ai] = dumps.sample_rotation(trk, 0) if trk else R.qn(rig.ac[ai]['lrot'])
        frames_z = np.arange(z0, min(fW, len(F) - 1) + 1)
        we = bake2._smoothstep((frames_z - z0) / float(max(1, fW - z0)))
        eased = rf.copy()
        for ai in patset:
            eased[frames_z, ai] = bake2.mix_rot(rf[frames_z, ai], np.tile(exit_loc[ai], (len(frames_z), 1)), we)
        zone = np.arange(max(0, z0 - 2), min(fW, len(F) - 1) + 1)
        zone = zone[zone % 2 == 0]                 # 30 Hz pairs, as the other step metrics
        ease = dict(zone_s=float(EASE_S), default_step_deg=bake2._step_deg(rf, zone, patset),
                    eased_step_deg=bake2._step_deg(eased, zone, patset))

    fails = []
    if max(box_nf, gnd_nf) * 100.0 > PEN_CM:
        fails.append('non-finger penetration %.1f cm' % (max(box_nf, gnd_nf) * 100.0))
    if gnd_all * 100.0 > GROUND_CM:
        fails.append('bone below ground %.1f cm' % (gnd_all * 100.0))
    if touch['min'] < TOUCH_FRAMES:
        fails.append('hand touch %d frames (need %d)' % (touch['min'], TOUCH_FRAMES))
    if dstep_max > DSTEP_CM:
        fails.append('Delta step %.1f cm/frame' % dstep_max)
    if dy_max / 100.0 > DY_SMALL + 1e-9:
        fails.append('forward offset %.2f m (> %.2f m)' % (dy_max / 100.0, DY_SMALL))
    metrics = dict(pen_box_cm=box_nf * 100.0, pen_ground_cm=gnd_nf * 100.0, pen_patch_cm=max(box_nf, gnd_nf) * 100.0,
                   pen_orig_box_cm=box_o * 100.0, pen_orig_ground_cm=gnd_o * 100.0,
                   pen_orig_cm=max(box_o, gnd_o) * 100.0, pen_finger_cm=max(box_f, gnd_f) * 100.0,
                   ground_all_cm=gnd_all * 100.0, hand_err_cm=hand_err, hand=hand_z, touch=touch,
                   dy_max_cm=dy_max, dz_max_cm=dz_max, dstep_max_cm=dstep_max,
                   step_patch_deg=step_p, boundary_patch_deg=bnd_p, boundary_orig_deg=bnd_o, gap=gap, ease=ease,
                   fails=fails, fail=bool(fails))
    arrays = dict(frames=F, Pf=Pf, Po=Po, win=win, body=body)
    return metrics, arrays


def body_tilts(rig, seq, dump, res, t_start, t_end, fade_in=FI):
    """Head-from-hips tilt (deg from vertical). 'o': the UN-FADED patch and the clip at o + fade_in (the moment the patch
    is fully in effect; the faded pose equals the clip at o by construction). 'W': the faded (final) pose and the clip
    at the window end. Returns {'o': (patch, clip), 'W': (patch, clip)}."""
    F = np.asarray(seq.frames, int)
    orig_world = acuclip.world_positions(seq, dump)
    hips = _acu(rig, 'Hips')
    head = _acu(rig, 'Head')

    def tilt(P, f):
        up = P[f, head] - P[f, hips]
        return math.degrees(math.acos(max(-1.0, min(1.0, float(up[2] / np.linalg.norm(up))))))

    unfaded = acuclip.world_positions(bake2.composed(seq, res['patch'], res['ref_new']), dump)
    faded = acuclip.world_positions(bake2.composed(seq, res['final'], res['ref_new']), dump)
    f_in = min(int(round((t_start + fade_in) * FPS)), int(F[-1]))
    f_w = min(int(round(t_end * FPS)), int(F[-1]))
    return {'o': (tilt(unfaded, f_in), tilt(orig_world, f_in)), 'W': (tilt(faded, f_w), tilt(orig_world, f_w))}


# ------------------------------------------------------------------------------------------------ 5. outputs
def _table(rows):
    lines = ['| clip | source | o / r / W (played) | k (acu / roll / used) | WD2 s at: takeoff / plant start / roll end / ACU feet-down | '
             'forward A (m; least-conflict if none feasible) | max abs Dz (cm) | non-finger pen box / ground (cm) | any bone below ground (cm) | '
             'hand touch L / R (fr) | max Delta step (cm/fr) | gap at W max / median (deg, worst bone) | status | enabled |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        mt, mp, c = r['metrics'], r['mp'], r['clock']
        tc = mt['touch']
        g = mt['gap']
        gtxt = 'n/a' if g['max_deg'] is None else '%.1f / %.1f (%s)' % (g['max_deg'], g['median_deg'], g['worst'])
        lines.append('| %s | %s | %.3f / %.3f / %.3f | %.3f / %.3f / %.3f | %.2f / %.2f / %.2f / %.2f | %.2f | %.1f | '
                     '%.1f / %.1f | %.1f | %d / %d | %.2f | %s | %s | %s |'
                     % (r['name'], mp['source'], mp['o'], mp['r'], mp['W'], c.k_acu, c.k_roll, c.k,
                        WD2_TAKEOFF, WD2_PLANT[0], WD2_ROLL_END, c.feet_wd2(), r['solve_info']['A_m'],
                        mt['dz_max_cm'], mt['pen_box_cm'], mt['pen_ground_cm'], mt['ground_all_cm'],
                        tc.get('LeftHand', 0), tc.get('RightHand', 0), mt['dstep_max_cm'], gtxt,
                        'FAIL' if mt['fail'] else 'PASS', 'yes' if r['enabled'] else 'no'))
    return lines


def _reference_bounds_text(rows):
    """Text for the reference clip's plant-start lift need and reach bound at a few amplitudes (from the stages stored on
    the reference row, when present)."""
    ref = next((r for r in rows if r.get('stages') and r['name'] == REF_CLIP), None)
    if ref is None:
        return ('(not recorded)', '(not recorded)')
    need, reach = [], []
    for st in ref['stages']:
        if st['A'] in (0.5, 0.75, 1.0) and st['plant_first'] is not None:
            need.append('%.2f m at %.2f' % (st['plant_first'][0], st['A']))
            reach.append('%s at %.2f' % ('unreachable' if not np.isfinite(st['plant_first'][1])
                                         else '%.2f m' % st['plant_first'][1], st['A']))
    return (', '.join(need) or '(not recorded)', ', '.join(reach) or '(not recorded)')


def write_report(path, rows, offsets):
    n_on = sum(1 for r in rows if r['enabled'])
    lines = ['# Dive: WD2 frontdive (diveroll) onto the ACU Sprint SlideOver clips (variant D)', '',
             'Generated by `v2/pipeline/bake_dive.py`. Times in clip seconds unless marked played; heights in cm; angles in '
             'degrees. Variant D: fade-in %.2f s from o, no fade-out, tail "wd2" (the WD2 motion runs to the clip end; the '
             'game blends out after the played window).' % FI, '',
             '## Method', '',
             '- Time map: played p = (t - o) / r. WD2 time w = %.2f + (p - p_takeoff) / k (WD2 takeoff <-> ACU takeoff). '
             'k = min(k_acu, k_roll), k_acu = (ACU feet-down - takeoff, played) / %.2f, k_roll = (W - p_takeoff) / %.2f.'
             % (WD2_TAKEOFF, K_ACU_SPAN, K_ROLL_SPAN),
             '- Rotations: additive (default): patched world = (A_REF * Wt(t) * conj(A_REF * Wt(o))) * clip world, so the patch '
             'is the original clip at o and carries only the WD2 change; the Reference is patched (bake2 kept it) because the '
             'dive inverts the body and Hips has no track. The bake2-style absolute form is reported below.',
             '- Root: Reference = original + (0, Dy, Dz). Dy = A * bump, the bump peaks at the plant start and is back at 0 at '
             'the window end. A is the smallest value on a %.2f m grid (0 to %.1f m) with no lift conflict and reachable hands '
             'at the plant start (hands beyond the far edge, +%.2f m); if none is feasible, the least-conflict A is used. '
             'Dz = bake2 signed lift (clearance of the non-finger body from the box and the ground; hands and arms are excluded '
             'in the plant because their IK owns them), reach bound on the plant, touch bound (lowest non-finger point on the '
             'ground) in the roll.' % (DY_GRID[1] - DY_GRID[0], DY_GRID[-1], HAND_MARGIN),
             '- Hands: IK to the plant targets (z = ground + hand offset %.3f m, y beyond the far edge) with weight while '
             'reachable. Feet: foot clamp IK as bake2.' % offsets['Hand'],
             '- PASS = non-finger penetration <= %.0f cm; no bone below the ground by > %.0f cm (window); both hands within '
             '%.0f cm of their targets for >= %d frames from the plant start; Delta step <= %.0f cm/frame; max |Dy| <= %.2f m '
             '(a small forward offset). The gap at W is reported (flag above %.0f deg), not a PASS rule.'
             % (PEN_CM, GROUND_CM, HAND_TOL_CM, TOUCH_FRAMES, DSTEP_CM, DY_SMALL, GAP_WARN_DEG), '',
             'Contact offsets (median over the Sprint SlideOver dumps): Hand %.3f m, Foot %.3f m, FootThumb1 %.3f m.'
             % (offsets['Hand'], offsets['Foot'], offsets['FootThumb1']), '',
             '## Validation', '']
    main = [r for r in rows if not r['name'].endswith(' [absolute]')]
    lines += _table(main)
    ref_row = next((r for r in rows if r.get('tilt')), None)
    abs_row = next((r for r in rows if 'absolute' in (r.get('comparison') or '')), None)
    main_rows = [r for r in rows if not (r.get('comparison') or '').startswith('absolute')]
    gmax = [r['metrics']['gap']['max_deg'] for r in main_rows if r['metrics']['gap']['max_deg'] is not None]
    gmed = [r['metrics']['gap']['median_deg'] for r in main_rows if r['metrics']['gap']['median_deg'] is not None]
    conf = [r['solve_info']['conflicts'] for r in main_rows]
    ref_need_text, ref_reach_text = _reference_bounds_text(rows)
    lines += ['', '## Findings', '']
    lines += ['- No clip has a feasible plant on the forward grid (0 to %.2f m): the least-conflict amplitude leaves %d to %d '
              'lift conflicts per clip. In the ACU clip the pelvis reaches the far edge at clip ~0.75 s (played ~0.55); the '
              'WD2 hand plant is at played 0.29 to 0.44 s. The root would have to move forward by about 1 m (roughly 0.2 s at '
              'the ~5 m/s sprint) for the hands to land beyond the edge. Reference clip, first plant frame: the lift the body '
              'needs to clear the box corner and the ground is %s; the reach bound (the largest lift that still lets the '
              'shoulders reach the ground beyond the edge) is %s. No amplitude gives both.'
              % (DY_GRID[-1], min(conf), max(conf), ref_need_text, ref_reach_text)]
    lines += ['- Window end: our body is not upright at W. The Muraille_Exit clip starts upright, so the gap at W is %.0f to %.0f deg '
              '(largest bone, Reference in most clips) and %.0f to %.0f deg median. That comes from the WD2 timing: the '
              'landing at W leaves no time to stand up. Easing the last %.2f s to the exit pose removes the gap at W by construction, '
              'but the largest per-pair step in that zone rises (reference clip, see the convention table).'
              % (min(gmax), max(gmax), min(gmed), max(gmed), EASE_S)]
    if ref_row is not None:
        to, tw = ref_row['tilt']['o'], ref_row['tilt']['W']
        lines.append('- Reference clip, head-from-hips tilt from vertical: at o + %.2f s (the fade-in done) the patch %.0f deg, '
                     'the clip %.0f; at W the patch %.0f deg, the clip %.0f.' % (FI, to[0], to[1], tw[0], tw[1]))
    if abs_row is not None and abs_row.get('tilt'):
        to = abs_row['tilt']['o']
        lines.append('- The bake2-style absolute rotations are %.0f deg away from the clip at o + %.2f s (patch %.0f, clip %.0f), '
                     'so the fade-in has to swing that much posture; the additive form is the default.'
                     % (abs(to[0] - to[1]), FI, to[0], to[1]))
    lines.append('- The hand heights in the PNG title read "hand above top" (bake2 plot helper); here they are heights above '
                 'the ground z = 0.')
    lines += ['', 'Enabled: %d of %d. Enabled clips: %s.'
              % (n_on, len(main), ', '.join(r['name'] for r in main if r['enabled']) or 'none'), '',
              '## Failure reasons', '']
    for r in main:
        why = '; '.join(r['metrics']['fails']) if r['metrics']['fails'] else 'none (PASS)'
        lines.append('- %s: %s' % (r['name'], why))
    lines += ['', '## Gap at the window end (our pose at W vs the first frame of the next Muraille_Exit clip)', '',
              '| clip | next clip | max / median (deg) | worst bone | bones over %.0f deg |' % GAP_WARN_DEG,
              '|---|---|---|---|---|']
    for r in main:
        g = r['metrics']['gap']
        if g['max_deg'] is None:
            lines.append('| %s | (none) | n/a | n/a | n/a |' % r['name'])
            continue
        over = ', '.join(b for b, v in g['per_bone'].items() if v > GAP_WARN_DEG) or 'none'
        lines.append('| %s | %s | %.1f / %.1f | %s | %s |' % (r['name'], g['next'], g['max_deg'], g['median_deg'],
                                                             g['worst'], over))
    cmp_rows = [r for r in rows if r.get('comparison')]
    if cmp_rows:
        lines += ['', '## Rotation convention, reference clip (both runs use the same time map, solver and metrics)', '',
                  '| convention | A (m) | conflicts | non-finger pen box / ground (cm) | touch min (fr) | max Delta step (cm/fr) | '
                  'gap at W max / median (deg, worst) | ease zone step default -> eased (deg/pair) |',
                  '|---|---|---|---|---|---|---|---|']
        for r in cmp_rows:
            mt, si = r['metrics'], r['solve_info']
            g, e = mt['gap'], mt['ease']
            lines.append('| %s | %.2f | %d | %.1f / %.1f | %d | %.1f | %.1f / %.1f (%s) | %.1f -> %.1f |'
                         % (r['comparison'], si['A_m'], si['conflicts'], mt['pen_box_cm'], mt['pen_ground_cm'],
                            mt['touch']['min'], mt['dstep_max_cm'], g['max_deg'], g['median_deg'], g['worst'],
                            e['default_step_deg'], e['eased_step_deg']))
    lines += ['', '## Notes', '']
    for r in main:
        si = r['solve_info']
        lines.append('- %s: plant %s; forward A %.2f m (%s); Dz max %.1f cm; lift conflicts %d; reach-limited frames %d; '
                     'foot raises %s.'
                     % (r['name'], 'none' if r['plant_s'] is None else '%.2f to %.2f s clip' % r['plant_s'],
                        si['A_m'], ('feasible: no conflicts, plant reachable' if si['feasible'] else
                                    'no feasible A on the 0-%.2f m grid after %d stages; the least-conflict A is shown'
                                    % (DY_GRID[-1], si['stages_tried'])),
                        si['dz_max_cm'], si['lift_conflict'], si['lift_reach_limited'],
                        ', '.join('%s %d frames' % (k, v['frames']) for k, v in r['foot'].items())))
    lines.append('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def bake_all_dive(write=True, plots=True):
    """All Sprint SlideOver clips. Writes patches\\<HEX>.baked.json, mapping_dive.json, PNG/GIF and report.md under out\\dive."""
    os.makedirs(PATCH_DIR, exist_ok=True)
    os.makedirs(DIVE_OUT, exist_ok=True)
    rig = R.Rig()
    ev_json = load_dive_events()
    m = wd2.load(ev_json['stem'])
    model = load_play_model_so()
    clips = []
    for handle, name, dump in targets_dive():
        seq = acuclip.pose_sequence(dump, step=1)
        ev = contacts.events(seq, dump)
        mp = clip_model(name, model, float(dump['Length']))
        clips.append(dict(handle=handle, name=name, dump=dump, seq=seq, ev=ev, mp=mp))
    offsets = bake2.measure_contact_offsets([(c['name'], c['dump'], c['seq'], c['ev']) for c in clips])
    rows, mapping = [], []
    for c in clips:
        nxt_name = next_clip_for(c['mp'])
        nxt = dumps.load_by_name(nxt_name)
        res = bake_clip_dive(rig, m, c['dump'], c['seq'], c['ev'], c['mp'], offsets)
        res['name'] = c['name']
        metrics, arrays = validate_dive(rig, c['dump'], c['seq'], res, c['mp'], next_dump=nxt)
        enabled = (not metrics['fail']) and c['mp']['capture']
        fn = '%s.baked.json' % c['handle']
        if write:
            bake2.write_patch_t9c(os.path.join(PATCH_DIR, fn), rig, res['rot_tracks'], res['trans_keys'],
                                  c['dump']['Length'])
            mapping.append(bake2.mapping_entry_t9c(c['handle'], c['name'], fn, enabled))
            if plots:
                plot_path = os.path.join(DIVE_OUT, c['name'] + '.png')
                bake2.plot_clip_t9c(plot_path, c['name'], rig, res, arrays, metrics, res['o'], res['span'],
                                    c['ev']['obstacle'])
                bake2.write_gif_t9e(os.path.join(DIVE_OUT, c['name'] + '.gif'), rig, res, arrays, c['ev']['obstacle'],
                                    res['o'], res['span'], res['clock'].w)
        pl = None
        if res['plant'].size:
            pl = (float(res['plant'][0]) / FPS, float(res['plant'][-1]) / FPS)
        ps = (res['clock'].t_of_w(WD2_PLANT[0]), res['clock'].t_of_w(WD2_PLANT[1])) if pl else None
        rows.append(dict(name=c['name'], handle=c['handle'], mp=c['mp'], clock=res['clock'], metrics=metrics,
                         solve_info=res['solve_info'], foot=res['foot'], plant_s=ps, enabled=enabled,
                         stages=res['stages'], comparison='additive (default)' if c['name'] == REF_CLIP else None))
        if c['name'] == REF_CLIP:
            res_a = bake_clip_dive(rig, m, c['dump'], c['seq'], c['ev'], c['mp'], offsets, rot_mode=ROT_ABSOLUTE)
            res_a['name'] = c['name']
            met_a, _arr = validate_dive(rig, c['dump'], c['seq'], res_a, c['mp'], next_dump=nxt)
            rows.append(dict(name=c['name'] + ' [absolute]', handle=c['handle'], mp=c['mp'], clock=res_a['clock'],
                             metrics=met_a, solve_info=res_a['solve_info'], foot=res_a['foot'], plant_s=None,
                             enabled=False, comparison='absolute (bake2-style, comparison only)',
                             tilt=body_tilts(rig, c['seq'], c['dump'], res_a, res_a['o'], res_a['t_end'])))
        if c['name'] == REF_CLIP:
            rows[-2]['tilt'] = body_tilts(rig, c['seq'], c['dump'], res, res['o'], res['t_end'])
        print('%-48s A %.2f m  Dz %.1f cm  pen %.1f/%.1f cm  touch %s  dstep %.2f  gap %s  %s' % (
            c['name'], res['solve_info']['A_m'], metrics['dz_max_cm'], metrics['pen_box_cm'], metrics['pen_ground_cm'],
            metrics['touch'], metrics['dstep_max_cm'],
            'n/a' if metrics['gap']['max_deg'] is None else '%.1f' % metrics['gap']['max_deg'],
            'FAIL (%s)' % '; '.join(metrics['fails']) if metrics['fail'] else 'PASS'))
    if write:
        with open(MAPPING_DIVE, 'w', encoding='utf-8') as f:
            json.dump({'clips': mapping}, f, indent=1)
        write_report(REPORT, rows, offsets)
    return rows, offsets


if __name__ == '__main__':
    bake_all_dive(write=True)
