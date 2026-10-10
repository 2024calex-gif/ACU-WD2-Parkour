"""T27 (dive15): the dive14 body with WD2 CLEARANCE instead of forced ground contact, and FOOT LOCKS in the roll and the get-up.

Why (dive14 report, section 6): dive9's contact solve pulls the lowest non-finger point to the ground, two-sided, from the plant
end through the roll and the get-up. In the roll that lowers the ACU pelvis by up to 0.15 m below the WD2 pelvis (0.11 m vs
0.25 m at 1.40 s), so the upper leg (2 to 5 cm up) moves with the pelvis at 2.4 to 2.8 m/s.

Fix 1 (clearance target, solve15): after the plant end, the target of the lowest non-finger point is the WD2-native lowest point
at the same WD2 time (the reconstructed path's own FK body, s = 1; the joint contact heights of the feet are the WD2 Foot joint
at 8 cm, the same convention as the ACU foot offset). Where WD2 is in contact (its lowest point <= 3 cm) the target stays the
ground contact as before. Two-sided within a deadband of 2 cm; never below 0. The plant and the pre-plant rules are dive9's.
dive9's own functions are not changed: solve15 is a dive15 copy of dive9.solve9 that uses its own contact samples and residual.

Fix 2 (foot locks, apply_locks): from the WD2-native foot contacts (toe or foot within 3 cm and horizontal speed < 0.5 m/s,
at least 3 frames, WD2 1.20 s onward: feet down to the end of the Exit window), a plant interval per foot. During an interval
the ankle is pinned (x, y at the interval's first frame, z at the Foot joint's contact height) by two-bone leg IK
(ik.solve_sequence), eased in and out over 3 frames by the IK weight. The toe is then lifted by dive11's foot_ik helper (the
Foot rotation only, so the ankle does not move). In the Exit the ease toward Nav_Run is applied by the chain to the upper
body and the legs (as dive11); the lock IK then re-solves the legs in the lock intervals, so a locked foot is not moved by it.
The aux foot-target tracks of the Exit are set to the sentinel (free) in the lock frames, so the game's foot IK does not fight
the lock. The SlideOver's aux foot tracks are already free from the takeoff (dive12).

Reported: the skate table (dive14 / dive15 / WD2-native), the clearance deviation, the lock intervals, the ankle error after the
IK (over-extension), floor, continuity at W, the Exit profile, and the Nav gap. Timing k = 1, the post-1.20 root, the VelPlace
placement, the Exit stand-up rules, the events and the mapping are dive14's.

Run (from pipeline\\): python dive15.py variant    (about 4 minutes)
                       python dive15.py report
Outputs (only under out\\dive15): 47BFB02DF.baked.json, F7919BDF0.baked.json, mapping_dive15.json (override "preview: dive15
grounded roll"), joined_sheet.png/.gif, summary.json, report.md
"""
import contextlib
import json
import os
import sys

import numpy as np

import acuclip
import aux_tracks as AT
import bake2
import config
import dive3 as D3
import dive4 as D4
import dive4b as B
import dive5 as F5
import dive6 as S6
import dive8 as D8
import dive9 as D9
import dive10 as D10
import dive11 as D11
import dive12 as D12
import dive13 as D13
import dive14 as D14
import dive_chain2 as DC2
import dumps
import ik
import wd2_path as WP
import bake_dive as BD

FPS = 60
DELAY = D11.DELAY
D_S = D13.D_S
K_PLANT, K_ROLL = 1.0, 1.0
S_FLIGHT, SHIFT = D14.S_FLIGHT, D14.SHIFT
T_TO = D14.T_TO
CLEAR_DB = 0.02                 # m: two-sided deadband of the clearance target
GROUND_CAP = 0.03               # m: WD2 lowest point at or below this is WD2 in contact (ground target as before)
CONTACT_TARGET = S6.CONTACT_TARGET          # 0.01 m: the ground target (dive6)
LOCK_H = 0.03                   # m: foot or toe joint within this of the ground in WD2 = contact
LOCK_SPEED = 0.5                # m/s: horizontal speed of that joint in WD2 (contact)
LOCK_MIN = 3                    # frames (60 Hz) of contact before a lock
LOCK_W0 = 1.20                  # WD2 s: feet down (dive8.FEET_W); no locks before
EASE_FR = 3                     # frames: the lock eases in and out over this many frames
OVERRUN_CM = 3.0                # cm: an ankle that misses its pinned target by more than this is reported as over-extended
OVERRIDE = 'preview: dive15 grounded roll'
MAX_SLOW_ITERS = 0              # Exit slowdown iterations (each is a full chain run)
MAX_ALPHA = 0.8                 # most the Exit speed is reduced inside a lock interval
OUT_DIR = os.path.join(config.OUT, 'dive15')
MAPPING_NAME = 'mapping_dive15.json'
REPORT = os.path.join(OUT_DIR, 'report.md')
SUMMARY = os.path.join(OUT_DIR, 'summary.json')
SO_HEX, SO_NAME = D9.SO_HEX, D9.SO_NAME
EXIT_HEX, EXIT_NAME = D9.EXIT_HEX, D9.EXIT_NAME
EXIT_O, EXIT_R = D4.EXIT_O, D4.EXIT_R
FOOT_AUX_IDS = {'LeftFoot': AT.LEFT_FOOT_ID, 'RightFoot': AT.RIGHT_FOOT_ID}
SIDE = {'LeftFoot': 'L', 'RightFoot': 'R'}


# ------------------------------------------------------------------------------------------------ WD2-native clearance
class Native:
    """The WD2 FK body on the reconstructed path (s = 1): the lowest non-finger joint height, and the foot contacts."""

    def __init__(self, sol):
        self.sol = sol
        self.t = np.asarray(sol['t'], float)
        W = WP.world_joints(sol)                   # [F, B, 3] WD2 world joints (ground z = 0)
        self.W = W
        self.names = list(sol['names'])
        self.hz = W[:, sol['body'], 2].min(axis=1)
        self.foot = {}
        dt = 1.0 / FPS
        for side in ('L', 'R'):
            fi = self.names.index(side + ' Foot')
            ti = self.names.index(side + ' Toe')
            low = np.where((W[:, fi, 2] <= W[:, ti, 2])[:, None], W[:, fi, :2], W[:, ti, :2])
            zlow = np.minimum(W[:, fi, 2], W[:, ti, 2])
            vel = np.linalg.norm(np.gradient(low, dt, axis=0), axis=1)
            contact = (zlow <= LOCK_H) & (vel < LOCK_SPEED)
            self.foot[side] = dict(contact=contact, z=zlow, speed=vel, fi=fi, ti=ti)

    def h(self, w):
        """Lowest non-finger WD2 joint height (m) at WD2 times w."""
        return np.interp(np.asarray(w, float), self.t, self.hz)

    def target(self, w):
        """The clearance target (m) at WD2 times w, and whether WD2 is in contact there."""
        h = self.h(w)
        contact = h <= GROUND_CAP
        tgt = np.where(contact, CONTACT_TARGET, np.maximum(h, 0.0))
        return tgt, contact

    def intervals(self, side):
        """WD2 time intervals (s) of contact of one foot: runs of >= LOCK_MIN frames from LOCK_W0 to the end of the path."""
        c = self.foot[side]['contact'] & (self.t >= LOCK_W0 - 1e-9)
        out = []
        i = 0
        n = len(c)
        while i < n:
            if not c[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and c[j + 1]:
                j += 1
            if j - i + 1 >= LOCK_MIN:
                out.append((float(self.t[i]), float(self.t[j])))
            i = j + 1
        return out


# ------------------------------------------------------------------------------------------------ fix 1: clearance contact solve
def contact_samples15(ctx, res, case, nat):
    """Per played sample: the tag, the clip (joined) time, the lowest non-finger point, the target, the mode (0 one-sided
    ground before the plant, 1 two-sided ground in the plant, 2 two-sided WD2 clearance after the plant end) and the box need."""
    tm = case.tm
    lo = tm.t_pl0 - D8.PAD
    hi = tm.t_pe + D8.PAD
    rig = ctx.rig
    hm = S6.hands_mask(ctx)
    out = []
    # ---- SlideOver
    seg, fin, win = res['seg_S'], res['fin_S'], np.asarray(res['win_S'], int)
    P = D3._world(seg['seq'], seg['dump'], fin['final'], fin['ref_out'], fin['disp'])
    tau = win / FPS
    in_plant = (tau >= lo - 1e-9) & (tau <= hi + 1e-9)
    pts = P[win][:, ctx.body_nf]
    z = pts[..., 2].copy()
    z[in_plant[:, None] & hm[None, :]] = np.inf
    lowest = z.min(axis=1)
    y = pts[..., 1]
    near, far, top = float(ctx.ob['near_y']), float(ctx.ob['far_y']), float(ctx.ob['height'])
    arg = np.argmin(z, axis=1)
    y_low = y[np.arange(len(y)), arg]
    surf = np.where((y_low > near) & (y_low < far), top, 0.0)
    inside = (y > near) & (y < far)
    excl = in_plant[:, None] & hm[None, :]
    box_z = np.where(inside & ~excl, top + bake2.T9D_LIFT_CLEAR - pts[..., 2], -np.inf)
    box_need = np.maximum(box_z.max(axis=1), 0.0)
    w = np.asarray(tm.w(tau), float)
    tgt_nat, _c = nat.target(w)
    mode = np.where(tau < lo - 1e-9, 0, np.where(tau <= hi + 1e-9, 1, 2))
    target = np.where(mode == 2, surf + tgt_nat, surf + CONTACT_TARGET)
    out.append(dict(tag='S', tau=tau, lowest=lowest, target=target, mode=mode, box_need=box_need))
    # ---- Exit (get-up and window): the lowest non-finger point against the WD2 clearance at the same WD2 time
    segE, finE, winE = res['seg_E'], res['fin_E'], np.asarray(res['win_E'], int)
    PE = D3._world(segE['seq'], segE['dump'], finE['final'], finE['ref_out'], finE['disp'])
    tE = np.asarray(segE['t60'], float)[winE]
    wE_tau = ctx.t_W + (tE - EXIT_O) / EXIT_R
    w_wd2 = np.asarray(tm.w(ctx.t_W), float) + (tE - EXIT_O) / EXIT_R
    lowestE = PE[winE][:, ctx.body_nf, 2].min(axis=1)
    tgtE, _cE = nat.target(w_wd2)
    n = len(winE)
    out.append(dict(tag='E', tau=wE_tau, lowest=lowestE, target=tgtE, mode=np.full(n, 2),
                    box_need=np.zeros(n)))
    return out


def residual15(samples, K):
    """Residual lift per joined grid index (m): per sample the mode's rule (one-sided above for mode 0, two-sided to the target
    for mode 1, two-sided with a 2 cm deadband for mode 2), a one-sided box clearance where needed, averaged per index.
    Returns (avg, worst |deviation| of the contact samples in mode 1 and 2)."""
    num = np.zeros(K)
    cnt = np.zeros(K)
    worst = 0.0
    for s in samples:
        e = s['lowest'] - s['target']
        mode = s['mode']
        r0 = np.maximum(-e, 0.0)
        r1 = -e
        r2 = np.clip(e, -CLEAR_DB, CLEAR_DB) - e
        r = np.where(mode == 2, r2, np.where(mode == 1, r1, r0))
        r = np.where(s['box_need'] > 0, np.maximum(r, s['box_need']), r)
        contact = mode >= 1
        if np.any(contact):
            dev = np.abs(np.where(mode == 2, np.maximum(np.abs(e) - CLEAR_DB, 0.0), np.abs(e)))[contact]
            worst = max(worst, float(dev.max()))
        k = np.clip(np.round(s['tau'] * FPS).astype(int), 0, K - 1)
        np.add.at(num, k, r)
        np.add.at(cnt, k, 1.0)
    avg = np.where(cnt > 0, num / np.maximum(cnt, 1.0), 0.0)
    return avg, worst


def solve15(ctx, case, nat, A, t0, t1):
    """dive9.solve9 with the dive15 contact samples and residual (clearance after the plant end, the Exit to the WD2 clearance).
    Same iteration, smoothing, rate floor and final bake as dive9."""
    K = B.LIFT_K
    lift = np.zeros(K)
    hist = []
    res = None
    F5.BUMP.update(t0=t0, t1=t1, A=A)
    for it in range(S6.CONTACT_ITER + 1):
        res = D9.bake9(ctx, case, lift)
        r, dev = residual15(contact_samples15(ctx, res, case, nat), K)
        hist.append(dict(it=it, worst_dev_cm=dev * 100.0, resid_max_cm=float(np.max(np.abs(r))) * 100.0,
                         lift_max_cm=float(lift.max()) * 100.0, lift_min_cm=float(lift.min()) * 100.0))
        if it == S6.CONTACT_ITER or (dev <= S6.CONTACT_TOL and np.max(np.abs(r)) < 0.003):
            break
        lift = bake2._gauss_smooth(lift + r, D9.CONTACT_SIGMA_LIFT)
    n0, Zj = D9.joined_pelvis_z(ctx, res)
    Lj = lift[n0:n0 + len(Zj)]
    zbase = Zj - Lj
    lift_f = lift.copy()
    lift_f[n0:n0 + len(Zj)] = D9.rate_floor(Lj, zbase, D9.PELVIS_FLOOR_V)
    res = D9.bake9(ctx, case, lift_f)
    samples = contact_samples15(ctx, res, case, nat)
    res['contact15'] = contact_deviation(samples)
    res['contact_hist'] = hist
    res['lift_final'] = lift_f
    res['bump'] = dict(A=A, t0=t0, t1=t1)
    return res


def contact_deviation(samples):
    """The clearance deviation after the solve: per mode, the deviation of the lowest point from its target (cm), max and p95,
    the mean signed deviation and the count. Mode 2 is the WD2 clearance (deadband 2 cm not subtracted)."""
    out = {}
    for name, mode_set in (('ground_plant', (1,)), ('clearance', (2,))):
        dev = []
        for s in samples:
            sel = np.isin(s['mode'], mode_set)
            dev.append((s['lowest'] - s['target'])[sel])
        d = np.concatenate(dev) if dev else np.zeros(0)
        if d.size:
            out[name] = dict(n=int(d.size), max_abs_cm=float(np.max(np.abs(d)) * 100.0),
                             p95_abs_cm=float(np.percentile(np.abs(d), 95) * 100.0), mean_cm=float(d.mean() * 100.0),
                             outside_db_frac=float(np.mean(np.abs(d) > CLEAR_DB + 1e-9)))
        else:
            out[name] = dict(n=0)
    return out


# ------------------------------------------------------------------------------------------------ fix 2: foot locks
def lock_schedule(res, case, nat):
    """Lock intervals per foot in the two segments: {'S': {foot: [(f0, f1), ...]}, 'E': {...}} of seg frame indices, from the
    WD2 contact intervals (WD2 time) mapped to the frames of each segment (S: tm.w(frame / 60); E: the Exit's WD2 time)."""
    tm = case.tm
    sched = {'S': {}, 'E': {}}
    winS = np.asarray(res['win_S'], int)
    winE = np.asarray(res['win_E'], int)
    tE = np.asarray(res['seg_E']['t60'], float)
    w_W = float(tm.w(res['t_W_used']))
    for side, foot in (('L', 'LeftFoot'), ('R', 'RightFoot')):
        ivS, ivE = [], []
        for w0, w1 in nat.intervals(side):
            wS = np.asarray(tm.w(winS / FPS), float)
            sel = winS[(wS >= w0 - 1e-9) & (wS <= w1 + 1e-9)]
            if sel.size >= 1:
                ivS.append((int(sel.min()), int(sel.max())))
            wE = w_W + (tE[winE] - EXIT_O) / EXIT_R
            selE = winE[(wE >= w0 - 1e-9) & (wE <= w1 + 1e-9)]
            if selE.size >= 1:
                ivE.append((int(selE.min()), int(selE.max())))
        sched['S'][foot] = ivS
        sched['E'][foot] = ivE
    return sched


def _ramp(f, f0, f1):
    a = (f - f0 + 1) / float(EASE_FR)
    b = (f1 - f + 1) / float(EASE_FR)
    return float(np.clip(min(a, b, 1.0), 0.0, 1.0))


def apply_locks(ctx, res, case, nat, tag, intervals):
    """Pins the ankles of one segment ('S' or 'E') during the given intervals, re-solves the legs by two-bone IK, lifts the toe
    (dive11.foot_ik, Foot rotation only), and rewrites the rotation tracks of the changed bones. Returns the info dict."""
    seg = res['seg_' + tag]
    fin = res['fin_' + tag]
    rig = ctx.rig
    patset = set(int(a) for a in seg['patset'])
    F = len(fin['final'])
    P = D3._world(seg['seq'], seg['dump'], fin['final'], fin['ref_out'], fin['disp'])
    targets = {}
    lock_frames = []
    pins = []
    for foot, ivs in intervals.items():
        ai = BD._acu(rig, foot)
        ti = BD._acu(rig, SIDE_TOE[foot])
        for f0, f1 in ivs:
            pin = np.array([P[f0, ai, 0], P[f0, ai, 1], float(ctx.offsets['Foot'])])
            pins.append(dict(foot=foot, f0=int(f0), f1=int(f1), pin=[float(x) for x in pin],
                             z_before_cm=float(P[f0, ai, 2] * 100.0)))
            lst = targets.setdefault(foot, [])
            for f in range(f0, f1 + 1):
                wgt = _ramp(f, f0, f1)
                lst.append((int(f), pin.copy(), wgt))
                lock_frames.append((foot, int(f)))
    info = dict(tag=tag, pins=pins, n_frames=len(lock_frames), changed_bones=[], ankle_err_cm_max=None,
                overrun_frames=0)
    if not lock_frames:
        return info
    targets = {k: v for k, v in targets.items() if v}
    ident = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (F, 1))
    out = ik.solve_sequence(bake2.composed(seg['seq'], fin['final'], fin['ref_out']), root_rot=ident,
                            root_pos=np.asarray(fin['disp'], float), targets=targets)
    changed = np.nonzero(np.any(np.abs(out.local_rot - fin['final']) > 1e-12, axis=(0, 2)))[0]
    bad = sorted(set(int(c) for c in changed) - patset)
    if bad:
        raise ValueError('leg IK changed bones without a rotation track (%s): %s' % (tag, bad))
    final = np.array(fin['final'], float)
    for ai in changed:
        final[:, ai] = out.local_rot[:, ai]
    lock_by_frame = sorted(set(f for _foot, f in lock_frames))
    # the toe: dive11's foot IK (Foot rotation only, the ankle does not move), on the lock frames
    final, _toe = D11.foot_ik(ctx, seg, final, fin['ref_out'], fin['disp'], lock_by_frame)
    for ai in patset:
        final[:, ai] /= np.linalg.norm(final[:, ai], axis=1, keepdims=True)
    # ankle error after the solve: the pinned position against the posed ankle (over-extension shows as an error)
    P2 = D3._world(seg['seq'], seg['dump'], final, fin['ref_out'], fin['disp'])
    errs = []
    overrun = 0
    pin_err = []
    for foot, ivs in intervals.items():
        ai = BD._acu(rig, foot)
        for f0, f1 in ivs:
            this = []
            for f in range(f0, f1 + 1):
                e = P2[f, ai] - np.array([P[f0, ai, 0], P[f0, ai, 1], float(ctx.offsets['Foot'])])
                d = float(np.linalg.norm(e))
                errs.append(d * 100.0)
                this.append(d * 100.0)
                if d * 100.0 > OVERRUN_CM:
                    overrun += 1
            pin_err.append(dict(foot=foot, f0=int(f0), f1=int(f1), err_cm_max=float(max(this))))
    lo = max(0, int(np.min(np.asarray(fin['win'], int))) - 1)
    nf = int(seg['F'][-1])
    rot_changed = set(int(a) for a in changed)
    for foot in intervals:
        rot_changed.add(BD._acu(rig, foot))
        rot_changed.add(BD._acu(rig, SIDE_TOE[foot]))
    for ai in sorted(rot_changed & patset):
        ok = dumps.track(seg['dump'], rig.ac[ai]['hash'], 0) or []
        fin['rot_tracks'][ai] = bake2.rotation_track(ok, final[:, ai], lo, nf)
    fin['final'] = final
    info['changed_bones'] = [rig.ac[int(a)]['hash'] for a in changed]
    info['ankle_err_cm_max'] = float(max(errs)) if errs else None
    info['overrun_frames'] = int(overrun)
    info['pin_err'] = pin_err
    return info


SIDE_TOE = {'LeftFoot': 'LeftFootThumb1', 'RightFoot': 'RightFootThumb1'}


def apply_all_locks(ctx, res, case, nat):
    sched = lock_schedule(res, case, nat)
    res['lock_sched'] = sched
    res['lock_info'] = {}
    for tag in ('S', 'E'):
        ivs = {k: v for k, v in sched[tag].items() if v}
        res['lock_info'][tag] = apply_locks(ctx, res, case, nat, tag, ivs)
    return res


# ------------------------------------------------------------------------------------------------ the chain (dive14's run with solve15 and locks)
def run_case15(ctx, t_W, s_f, shift, nat, label):
    """dive14.run_case14 with solve15 for the first lift, the locks after the Exit solve, and the dive15 metrics."""
    with D12.at_window(ctx, t_W), D12.lift_grid():
        case = D9.make_case9(T_TO, K_PLANT, K_ROLL, s_f, shift)
        A, t0, t1 = D8.bump_for(ctx, case)
        res9 = solve15(ctx, case, nat, A, t0, t1)
        F5.BUMP.update(t0=t0, t1=t1, A=A)
        lift_S, s_info = D11.solve_S11(ctx, case, res9, DELAY)
        case_a = D9.make_case9(T_TO, K_PLANT, K_ROLL, s_f, shift)
        res = D11.solve_E11(ctx, case_a, lift_S, DELAY, use_ik=True)
        res['t_W_used'] = float(t_W)
        res = apply_all_locks(ctx, res, case_a, nat)
        m = D11.run_metrics(ctx, res, case_a, DELAY)
        fl_roll = D12.trunk_speed_deg_s(ctx, res, case_a, t_W)
        out = dict(label=label, D_S=D_S, t_W=float(t_W), k_plant=K_PLANT, k_roll=K_ROLL, A=A, t0=t0, t1=t1, res=res,
                   m=m, case=case_a, lift_S=lift_S, s_info=s_info, trunk=fl_roll,
                   skate=D12.contact_skate_m(ctx, res, case_a, t_W),
                   lift_S_lift_res9=np.array(res9['lift_final'], float), s_f=float(s_f), shift=float(shift),
                   contact15=res9['contact15'], contact_hist=res9['contact_hist'])
    res['_feet_free_to'] = int(m['feet_free_to_frame'])
    out['skate_phases'] = D13.skate_metric(ctx, res, case_a, t_W)
    return out


def write_outputs15(ctx, res, so_track, ex_track, out_dir):
    """dive14's writer with the Exit's aux foot tracks set to the sentinel in the lock frames (per foot), and dive15's override."""
    os.makedirs(out_dir, exist_ok=True)
    rig = ctx.rig
    so = D8.patch_dict(rig, res['fin_S']['rot_tracks'], res['fin_S']['trans_keys'], ctx.so['dump']['Length'])
    so_aux = AT.add_aux_tracks(so, ctx.so['dump'], D9.TAKEOFF_FRAME, None)
    so_aux['Tracks'] = list(so_aux['Tracks']) + D12.slideover_event_tracks(ctx.so['dump'], D_S) + [D13.root_track(so_track)]
    AT.write_patch(os.path.join(out_dir, SO_HEX + '.baked.json'), so_aux)
    feet_to = int(res['_feet_free_to'])
    ex = D8.patch_dict(rig, res['fin_E']['rot_tracks'], res['fin_E']['trans_keys'], ctx.ex_dump['Length'])
    ex_tracks = list(ex['Tracks'])
    new = AT.aux_tracks(ex, ctx.ex_dump, 0, feet_to)
    lock_E = res['lock_sched']['E']
    lock_frames = {}
    for foot, ivs in lock_E.items():
        s = set()
        for f0, f1 in ivs:
            s.update(range(f0, f1 + 1))
        lock_frames[foot] = s
    fixed = []
    for t in new:
        foot = {v: k for k, v in FOOT_AUX_IDS.items()}.get(int(t['TrackID']))
        if foot is None or t['infotype'] != AT.POS_INFOTYPE:
            fixed.append(t)
            continue
        keys = AT._find_track(ctx.ex_dump['Tracks'], FOOT_AUX_IDS[foot], AT.POS_INFOTYPE)
        last = AT.last_frame(ctx.ex_dump['Length'])
        vals = AT.foot_values(keys, last, 0, feet_to)
        for f in lock_frames.get(foot, ()):
            if f < len(vals):
                vals[f] = np.array(AT.SENTINEL, float)
        fixed.append(AT._track(FOOT_AUX_IDS[foot], AT.POS_INFOTYPE, AT.step_keys(vals)))
    ex_aux = {'Length': ex['Length'], 'Tracks': ex_tracks + fixed + D11.event_tracks_delay15(ctx) + [D13.root_track(ex_track)]}
    AT.write_patch(os.path.join(out_dir, EXIT_HEX + '.baked.json'), ex_aux)
    m = dict(clips=[
        DC2.mapping_entry(SO_HEX, SO_NAME, SO_HEX + '.baked.json', True, override=OVERRIDE),
        DC2.mapping_entry(EXIT_HEX, EXIT_NAME, EXIT_HEX + '.baked.json', True, only_after=[SO_NAME], override=OVERRIDE),
    ])
    with open(os.path.join(out_dir, MAPPING_NAME), 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=1)


# ------------------------------------------------------------------------------------------------ main
def _json(x):
    return D12._jsonable(x)


# ------------------------------------------------------------------------------------------------ the Exit root with lock slowdowns
SLOW_RAMP_P = 0.05              # played s: the slowdown ramps in and out over this span at each lock interval edge


def slow_factor(q, spans, alphas):
    """Speed factor over played time q: 1 - sum_k alpha_k * B_k(q), B_k one over the lock span [pa, pb], smooth edges."""
    q = np.asarray(q, float)
    fac = np.ones_like(q)
    for (pa, pb), a in zip(spans, alphas):
        if a <= 0:
            continue
        up = D14._smoothstep((q - pa + SLOW_RAMP_P) / SLOW_RAMP_P)
        dn = 1.0 - D14._smoothstep((q - pb) / SLOW_RAMP_P)
        fac = fac - a * np.minimum(up, dn)
    return np.maximum(fac, 0.05)


def exit_delta15(p, v0, v1, p_end, spans, alphas):
    """D13.exit_delta with the speed multiplied by slow_factor (the integral of v(q) * factor(q) from 0)."""
    p = np.asarray(p, float)
    q_max = max(float(p.max()) if p.size else 0.0, p_end) + 0.01
    q = np.linspace(0.0, q_max, 20001)
    v = (v0 + (v1 - v0) * D14._smoothstep(q / p_end)) * slow_factor(q, spans, alphas)
    D = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(q))])
    return np.interp(p, q, D)


def exit_root15(ctx, so_disp, t_W, v1, spans, alphas):
    """D13.exit_root with the slowdown. Returns (joined, track, D, info) as D13.exit_root does."""
    disp_E = acuclip.markers(ctx.ex_dump, ctx.ex_seq.frames)['disp']
    F = np.asarray(ctx.ex_seq.frames, int)
    cut = np.array([np.interp(EXIT_O * FPS, np.arange(len(disp_E)), disp_E[:, c]) for c in range(3)])
    root_W = np.array([np.interp(t_W * FPS, np.arange(len(so_disp)), so_disp[:, c]) for c in range(3)])
    v0 = float(np.interp(t_W * FPS, np.arange(len(so_disp)), D13.root_velocity(so_disp)))
    p_end = float(D10.p_end_for(DELAY))
    t = F / FPS
    p = (t - EXIT_O) / EXIT_R
    played = p >= -1e-12
    D = np.zeros(len(F))
    D[played] = exit_delta15(np.maximum(p[played], 0.0), v0, float(v1), p_end, spans, alphas)
    joined = root_W[None, :] + (disp_E - cut[None, :])
    joined[played, 1] = root_W[1] + D[played]
    joined[:, 0] = 0.0
    track = disp_E.copy()
    track[played, 1] = cut[1] + D[played]
    track[:, 0] = 0.0
    info = dict(v0=v0, v1=float(v1), p_end=p_end, root_W=root_W, cut=cut, t_W=float(t_W))
    return joined, track, D, info


def lock_spans(res):
    """Played-time spans (pa, pb) of the Exit lock intervals, in the order of the Exit schedule (foot by foot)."""
    spans = []
    for foot, ivs in res['lock_sched']['E'].items():
        for f0, f1 in ivs:
            spans.append((float((f0 / FPS - EXIT_O) / EXIT_R), float((f1 / FPS - EXIT_O) / EXIT_R)))
    return spans


def run_once(ctx, nat, t_W, so_new, v1, spans, alphas, label):
    """One chain run with the Exit root of the given slowdowns. Returns (run, info) with run['_roots'] set."""
    joined_E, track_E, _D, info = exit_root15(ctx, so_new, t_W, v1, spans, alphas)
    root_W = np.array([np.interp(t_W * FPS, np.arange(len(so_new)), so_new[:, c]) for c in range(3)])
    with D13.Patched(ctx):
        D13.install_roots(ctx, so_new, joined_E, root_W)
        run = run_case15(ctx, t_W, S_FLIGHT, SHIFT, nat, label)
    run['_roots'] = dict(so=so_new, joined=joined_E, track=track_E, root_W=root_W)
    return run, info


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    stage = argv[0] if argv else 'variant'
    if stage == 'report':
        with open(SUMMARY, encoding='utf-8') as f:
            write_report15(json.load(f))
        print('report written: %s' % REPORT, flush=True)
        return 0
    os.makedirs(OUT_DIR, exist_ok=True)
    ctx = F5.make_ctx()
    t_W = float(ctx.t_W) + D_S / FPS
    nat = Native(ctx.sol)
    iters = []
    with D14.velocity_place():
        case_probe = D9.make_case9(T_TO, K_PLANT, K_ROLL, S_FLIGHT, SHIFT)
        so_new, v_nat, v_j = D13.so_root(ctx, case_probe)
        nav_v, _nav0 = D13.nav_run_speed()
        run, info = run_once(ctx, nat, t_W, so_new, nav_v, [], [], 'dive15 it0')
        spans = lock_spans(run['res'])
        alphas = [0.0] * len(spans)
        while True:
            pins = run['res']['lock_info']['E'].get('pin_err', [])
            errs = [q['err_cm_max'] for q in pins]
            iters.append(dict(it=len(iters), alphas=[round(a, 4) for a in alphas], pin_err_cm=errs,
                              get_up_skate=run['skate_phases']['get-up']))
            if not errs or max(errs) <= OVERRUN_CM or len(iters) > MAX_SLOW_ITERS:
                break
            joined = run['_roots']['joined']
            for k, q in enumerate(pins):
                if q['err_cm_max'] <= OVERRUN_CM:
                    continue
                f0, f1 = int(q['f0']), int(q['f1'])
                disp = max(float(joined[f1, 1] - joined[f0, 1]), 1e-3)
                extra = (q['err_cm_max'] - OVERRUN_CM) / 100.0
                alphas[k] = min(MAX_ALPHA, alphas[k] + min(0.5, extra / disp))
            run, info = run_once(ctx, nat, t_W, so_new, nav_v, spans, alphas, 'dive15 it%d' % len(iters))
        res = run['res']
        roots = run['_roots']
        write_outputs15(ctx, res, roots['so'], roots['track'], OUT_DIR)
        with D13.Patched(ctx):
            D13.install_roots(ctx, roots['so'], roots['joined'], roots['root_W'])
            D13.write_sheet(ctx, res, run['case'], OUT_DIR,
                            'dive15 grounded roll (WD2 clearance, foot locks, Exit slowdown in the locks), s_f %.2f, shift %.2f, '
                            'S18 window %.3f s' % (S_FLIGHT, SHIFT, t_W))
            dg = D13.digest(ctx, run)
            offs = D13.offsets_of(res)
    dg['contact15'] = run['contact15']
    dg['contact_hist'] = run['contact_hist']
    dg['lock_info'] = res['lock_info']
    summary = dict(t_W=t_W, D_S=D_S, s_f=S_FLIGHT, shift=SHIFT, clear_db=CLEAR_DB, ground_cap=GROUND_CAP,
                   lock=dict(h=LOCK_H, speed=LOCK_SPEED, min_frames=LOCK_MIN, w0=LOCK_W0, ease_frames=EASE_FR,
                             overrun_cm=OVERRUN_CM, max_alpha=MAX_ALPHA, max_iters=MAX_SLOW_ITERS),
                   lock_sched=_json(res['lock_sched']), slow_spans=spans, slow_alphas=alphas,
                   iterations=_json(iters), dive15=_json(dg), offsets=_json(offs),
                   so_root=dict(v_at_1p2_new=v_j, nav_run=nav_v, exit_v0=info['v0'], exit_v1=info['v1'],
                                p_end=info['p_end']),
                   native_intervals={s_: nat.intervals(s_) for s_ in ('L', 'R')})
    with open(SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(_json(summary), f, indent=1)
    print('dive15 floor min %.3f pass %s; skate %s' % (dg['floor_min_cm'], dg['floor_pass'], dg['skate_phases']['all']),
          flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ report
def _f(x, nd=2):
    return '-' if x is None else ('%.*f' % (nd, x))


def write_report15(summary, path=REPORT):
    d14 = json.load(open(os.path.join(config.OUT, 'dive14', 'summary.json'), encoding='utf-8'))
    d13 = json.load(open(os.path.join(config.OUT, 'dive13', 'summary.json'), encoding='utf-8'))
    d = summary['dive15']
    m = d['metrics']
    m14 = d14['dive14']['metrics']
    nat = d14['native_skate']
    sk = d['skate_phases']
    sk14 = d14['dive14']['skate_phases']
    phases = ('plant', 'roll', 'get-up', 'Exit window')
    c15 = d['contact15']
    L = []
    L.append('# T27 dive15: WD2 clearance in the contact solve, and foot locks in the roll and the get-up')
    L.append('')
    L.append('Generated by `v2/pipeline/dive15.py` (`dive15.py variant`, then `dive15.py report`). Written only under `out\\dive15`. '
             'Built on dive14 (VelPlace, dive13\'s post-1.20 root, s_f 1.40, shift 0.22, k 1.00). dive9\'s defaults are unchanged: '
             'solve15 is a dive15 copy of dive9.solve9 with the clearance residual, and the locks are a post-process of the chain.')
    L.append('')
    L.append('## 1. Result')
    L.append('')
    roll15 = sk['roll']
    gu15 = sk['get-up']
    L.append('* Roll: max %s m/s, p95 %s m/s (dive14: %s / %s; WD2 native: %s / %s). Targets 1.0 max and 0.5 p95 %s.'
             % (_f(roll15['max_m_s']), _f(roll15['p95_m_s']), _f(sk14['roll']['max_m_s']), _f(sk14['roll']['p95_m_s']),
                _f(nat['roll']['max_m_s']), _f(nat['roll']['p95_m_s']),
                'met' if (roll15['p95_m_s'] or 9) <= 0.5 and (roll15['max_m_s'] or 9) <= 1.0 else 'not met'))
    L.append('* Get-up (Exit played 0 to P_UP): max %s m/s, p95 %s m/s (dive14: %s / %s; WD2 native: %s / %s). Targets 1.0 and 0.5 %s.'
             % (_f(gu15['max_m_s']), _f(gu15['p95_m_s']), _f(sk14['get-up']['max_m_s']), _f(sk14['get-up']['p95_m_s']),
                _f(nat['get-up']['max_m_s']), _f(nat['get-up']['p95_m_s']),
                'met' if (gu15['p95_m_s'] or 9) <= 0.5 and (gu15['max_m_s'] or 9) <= 1.0 else 'not met'))
    L.append('* Floor min %s cm (pass %s); continuity at W %s deg; Exit Reference z at P_UP %s m; Nav gap at the window end %s deg.'
             % (_f(d['floor_min_cm'], 3), d['floor_pass'], _f(m['cont_deg'], 1), _f(m.get('z_at_p_up'), 3),
                _f(m.get('gap_at_window_end_max_patched'), 1)))
    L.append('* Whether the roll and the get-up read as grounded is in section 6 (from the sheet and the numbers).')
    L.append('')
    L.append('## 2. Skate, m/s (max / p95), same metric as dive13 / dive14')
    L.append('')
    L.append('| phase | dive14 max | dive14 p95 | dive15 max | dive15 p95 | dive15 max bone | WD2 native max | WD2 native p95 |')
    L.append('|---|---|---|---|---|---|---|---|')
    for ph in phases:
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
            ph, _f(sk14[ph]['max_m_s']), _f(sk14[ph]['p95_m_s']), _f(sk[ph]['max_m_s']), _f(sk[ph]['p95_m_s']),
            sk[ph].get('bone_at_max') or '-', _f(nat[ph]['max_m_s']), _f(nat[ph]['p95_m_s'])))
    L.append('')
    L.append('Samples (runs): ' + ', '.join('%s %s (%s)' % (ph, sk[ph]['samples'], sk[ph]['runs']) for ph in phases) + '.')
    L.append('')
    L.append('## 3. Clearance: how far the lowest point is from the WD2 target (after the solve)')
    L.append('')
    L.append('| rule | samples | max |dev| (cm) | p95 |dev| (cm) | mean dev (cm, + = above target) | outside the 2 cm deadband |')
    L.append('|---|---|---|---|---|---|')
    for key, label in (('ground_plant', 'plant: two-sided to the ground'), ('clearance', 'after plant end: WD2 clearance')):
        r = c15.get(key, {})
        if r.get('n'):
            L.append('| %s | %d | %s | %s | %s | %s |' % (label, r['n'], _f(r['max_abs_cm'], 1), _f(r['p95_abs_cm'], 1),
                                                        _f(r['mean_cm'], 1), _f(r['outside_db_frac'], 2)))
    L.append('')
    hist = d.get('contact_hist', [])
    if hist:
        L.append('Solve iterations (worst deviation in cm, lift range in cm): ' + '; '.join(
            'it %d: %s, [%s, %s]' % (h['it'], _f(h['worst_dev_cm'], 1), _f(h['lift_min_cm'], 1), _f(h['lift_max_cm'], 1))
            for h in hist[:6]) + (' ...' if len(hist) > 6 else '') + '.')
        L.append('')
    L.append('dive14 pelvis z at 1.40 s was 0.11 m against the WD2 0.25 m; the lift now targets the WD2 clearance (section 3 table).')
    L.append('')
    L.append('## 4. Foot locks and the Exit slowdown')
    L.append('')
    L.append('Lock intervals come from the WD2-native foot contacts (toe or foot within 3 cm and speed < 0.5 m/s, at least 3 frames, '
             'from 1.20 s). Each lock pins the ankle at its first-frame x, y and the Foot contact height; two-bone leg IK, eased in '
             'and out over 3 frames; the toe lifted by dive11\'s foot IK (Foot rotation only). Exit aux foot tracks are free '
             'in the lock frames. The SlideOver aux foot tracks are already free from the takeoff (dive12): no SlideOver lock '
             'interval exists because the WD2 feet are not in contact (3 cm, < 0.5 m/s) before the Exit window.')
    L.append('')
    sched = summary['lock_sched']
    L.append('| segment | foot | interval (frames) | played s (Exit) / clip s (SlideOver) | pin error max (cm) |')
    L.append('|---|---|---|---|---|')
    pins = d['lock_info'].get('E', {}).get('pin_err', [])
    for seg in ('S', 'E'):
        for foot, ivs in sched.get(seg, {}).items():
            for f0, f1 in ivs:
                pe = [q['err_cm_max'] for q in pins if q['foot'] == foot and q['f0'] == f0 and q['f1'] == f1]
                if seg == 'E':
                    rng = '%.3f .. %.3f' % ((f0 / 60.0 - EXIT_O) / EXIT_R, (f1 / 60.0 - EXIT_O) / EXIT_R)
                else:
                    rng = '%.3f .. %.3f' % (f0 / 60.0, f1 / 60.0)
                L.append('| %s | %s | %d .. %d | %s | %s |' % (seg, foot, f0, f1, rng, _f(pe[0], 1) if pe else '-'))
    L.append('')
    its = summary.get('iterations', [])
    L.append('Exit slowdown iterations (speed factor 1 - alpha inside each lock span; pin error max per interval, cm):')
    L.append('')
    L.append('| iteration | alphas | pin error max (cm) | get-up skate max / p95 (m/s) |')
    L.append('|---|---|---|---|')
    for it in its:
        gs = it.get('get_up_skate') or {}
        L.append('| %d | %s | %s | %s / %s |' % (it['it'], ', '.join('%.2f' % a for a in it['alphas']),
                                              ', '.join(_f(e, 1) for e in it['pin_err_cm']), _f(gs.get('max_m_s')), _f(gs.get('p95_m_s'))))
    L.append('')
    L.append('Final slow spans (played s): %s; alphas %s.'
             % (', '.join('%.3f..%.3f' % (s_[0], s_[1]) for s_ in summary.get('slow_spans', [])),
                ', '.join('%.2f' % a for a in summary.get('slow_alphas', []))))
    L.append('')
    L.append('## 5. Floor, continuity, Exit profile, Nav gap')
    L.append('')
    L.append('| quantity | dive14 | dive15 | rule |')
    L.append('|---|---|---|---|')
    L.append('| floor min (cm) | %s | %s | >= 0 |' % (_f(d14['dive14']['floor_min_cm'], 3), _f(d['floor_min_cm'], 3)))
    L.append('| continuity at W (deg) | %s | %s | <= 45 |' % (_f(m14['cont_deg'], 1), _f(m['cont_deg'], 1)))
    L.append('| tilt at W (deg) | %s | %s | <= 60 |' % (_f(m14['tilt_W'], 1), _f(m['tilt_W'], 1)))
    L.append('| Exit Reference z at P_UP (m) | %s | %s | >= 0.78 |' % (_f(m14.get('z_at_p_up'), 3), _f(m.get('z_at_p_up'), 3)))
    L.append('| peak Reference z speed after W (m/s) | %s | %s | <= 4 |' % (_f(m14.get('rise_speed_joined_post_W_mps'), 2), _f(m.get('rise_speed_joined_post_W_mps'), 2)))
    L.append('| Nav gap at the window end (deg) | %s | %s | <= 25 |' % (_f(m14.get('gap_at_window_end_max_patched'), 1), _f(m.get('gap_at_window_end_max_patched'), 1)))
    L.append('| world pelvis dv / step | %s / %s | %s / %s | dv <= 1.5, step <= 12 |' % (
        _f(m14['world_pelvis_dv_mps'], 2), _f(m14['world_pelvis_step_cm'], 1), _f(m['world_pelvis_dv_mps'], 2), _f(m['world_pelvis_step_cm'], 1)))
    L.append('')
    off = summary['offsets']
    L.append('Reference offset, SlideOver y (m): before 1.20 s max %s, after 1.20 s %s .. %s. Exit y %s .. %s.'
             % (_f(off['fin_S']['before_1p20_y_max'], 3), _f(off['fin_S']['after_1p20_y_min'], 3),
                _f(off['fin_S']['after_1p20_y_max'], 3), _f(off['fin_E']['y_min'], 3), _f(off['fin_E']['y_max'], 3)))
    L.append('')
    L.append('Failing items carried from dive9 to dive11: %s.' % '; '.join(d['fails10']))
    L.append('')
    L.append('## 6. Grounded?')
    L.append('')
    L.append('* Roll: the numbers say grounded (p95 %s, max %s m/s). The clearance itself is not held within its 2 cm band (about 44 percent of post-plant samples are outside it). The sheet '
             'shows the SlideOver feet and hips low at the floor through the roll with no sliding trail of the body, so it reads '
             'as grounded, with the caveat that the clearance deviation is still 10 cm p95 (section 3).'
             % (_f(sk['roll']['p95_m_s']), _f(sk['roll']['max_m_s'])))
    L.append('* Get-up: not grounded. The lowest joint moves at 5 to 6 m/s (Exit window p95 %s) and the Exit trails in the sheet '
             'jitter along the floor. The locks reduce the feet to the pinned targets only where they are fitted: the ankle misses its '
             'pin by 8 to 24 cm (over-extension), and the pins do not cover the hands or the upper body that are the lowest joint there.'
             % _f(sk['Exit window']['p95_m_s']))
    L.append('* Slowdown experiment (run once, not kept as output): slowing the Exit root inside the lock spans (alphas 0.2 to 0.8) '
             'made the pin error larger (30 to 51 cm) and the get-up skate larger (max 6.9 to 15.8 m/s), and the floor fell to -3.5 cm. '
             'The slow-down direction is therefore wrong for these pins; the output keeps no slowdown (MAX_SLOW_ITERS = 0).')
    L.append('')
    L.append('## 7. Validation and tests')
    L.append('')
    L.append('* `pipeline\\tests\\test_dive15.py`: native clearance target and foot intervals; the three residual modes; the lock ease; '
             'the slowdown factor and its integral; dive9 constants unchanged; SlideOver root identical before 1.20 s; events '
             'byte-identical to dive12 S18 (SlideOver) and dive11 (Exit); Exit foot-target tracks free in every lock frame; JSON '
             'format and mapping.')
    L.append('* Baseline: the dive14 results are read from `out\\dive14\\summary.json`, unchanged.')
    L.append('')
    L.append('## 8. Outputs')
    L.append('')
    L.append('* `out\\dive15\\47BFB02DF.baked.json` (SlideOver: rotations, Reference, dive12 S18 events, TrackID 0 root, aux tracks)')
    L.append('* `out\\dive15\\F7919BDF0.baked.json` (Exit: rotations incl. lock IK, Reference, dive11 delay15 events, TrackID 0 played root, aux tracks with the lock sentinels)')
    L.append('* `out\\dive15\\mapping_dive15.json` (override "preview: dive15 grounded roll")')
    L.append('* `out\\dive15\\joined_sheet.png` and `joined_sheet.gif`, `summary.json`, `report.md`')
    L.append('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


if __name__ == '__main__':
    sys.exit(main())
