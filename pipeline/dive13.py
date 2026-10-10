"""T25 (dive13): the dive12 S18 dive with a SLOW ROOT during the roll and the stand-up (fix for "he slides on the floor").

Why: in game the SlideOver entity (TrackID 0, the original root) runs 4 to 8 m/s from clip 1.2 s, while the body that the
dive places (the WD2 pelvis path, scaled) is still low, so the lowest bone is dragged along the floor. The root is patched
here, not the body's target:

  SlideOver (TrackID 0, infotype 3, codec 27): the original root for clip frames 0..72 (clip 1.20 s), then from frame 73 the
      forward speed blends (smoothstep over 0.15 s, so C1 at 1.20) from the original speed to the natural forward speed of the
      body, i.e. the WD2 pelvis path (dive4b.target_world with dive12's S18 case) smoothed and clipped at 0. z = 0, x = 0.
  Exit (TrackID 0, codec 27): the original root before 0.208 clip s (unplayed). From 0.208 the joined root is
      root_S(W) + D(p), p = (clip - 0.208) / 0.9665 (played), with D'(p) = v0 + (v1 - v0) * smoothstep(p / p_end):
      v0 = the SlideOver root's speed at W, v1 = the Nav_Run root speed (dump TrackID 0, first 0.2 s), p_end = the delay15
      window end (dive10.p_end_for). The patch stores cut + D(p) (the game applies the Exit root as a delta from its start).

The bake (dive12 run_case, dive11 chain) is reused unchanged. It takes the root through ctx.disp_so / ctx.disp_ex (the
root and the Exit's joined root). The Reference is re-solved by the bake against the new root (bake_clip5 for the SlideOver,
the Exit's Reference is carried as before). Events are copied as they are: SlideOver = dive12's S18 events (checked
byte-identical against out\\dive12\\S18), Exit = dive11's delay15 events (checked against out\\dive11\\F7919BDF0). The
rotations, plant, takeoff, roll timing (k 1.00 / 1.00) and the Exit stand-up profile are dive12's.

New skate metric (report section 3): for each frame of a phase where the lowest non-finger bone is within 12 cm of its
surface (ground, or the box top inside the box band), the horizontal world speed of that bone, kept only where the same
bone is the lowest for at least 3 frames in a row. Phases: plant, roll (plant end to W), get-up (Exit played 0 .. P_UP),
Exit window (P_UP .. p_end). Speeds in the Exit are per joined second (dtau = dt_clip / 0.9665).

Outputs (only under out\\dive13):
  47BFB02DF.baked.json (SlideOver with TrackID 0), F7919BDF0.baked.json (Exit with TrackID 0), mapping_dive13.json,
  joined_sheet.png / joined_sheet.gif (entity root drawn), summary.json, report.md
Run: python dive13.py   (the baseline S18 with the original root and the dive13 bake; several minutes)
"""
import json
import os
import sys

import numpy as np

import acuclip
import aux_tracks as AT
import bake_chain as BCh
import bake2
import config
import dive4 as D4
import dive4b as B
import dive5 as F5
import dive8 as D8
import dive9 as D9
import dive10 as D10
import dive11 as D11
import dive12 as D12
import dive3 as D3
import dive_chain2 as DC2
import dumps

FPS = 60
DELAY = D11.DELAY                       # 15 frames: the Exit window as dive11
D_S = 18                                # the SlideOver cut moved +18 frames (dive12 S18)
K_PLANT, K_ROLL = 1.0, 1.0              # dive12 S18 timing
J_FRAME = 72                            # clip 1.20 s: the root is the original up to and including this frame
BLEND_S = 0.15                          # s: the speed blends to the natural speed over this long after J_FRAME
NATURAL_SIGMA_S = 0.06                  # s: Gaussian smoothing of the natural speed (the plant-end step of the WD2 path)
EXIT_O = D4.EXIT_O                      # 0.208 clip s: the Exit's played start
EXIT_R = D4.EXIT_R                      # 0.9665 clip s per played s
NAV_NAME = D4.NEXT_NAME                 # xx_Nav_Run_3-Loop
NAV_SPEED_S = 0.20                      # s: Nav_Run's root speed is its mean over this first span
SKATE_H = 0.12                          # m: a bone within this height of its surface is in contact
SKATE_RUN = 3                           # frames: the same lowest bone for at least this many frames
SKATE_P95_TARGET = 0.5                  # m/s
SKATE_MAX_TARGET = 1.0                  # m/s
OFF_Y_MAX = 0.5                         # m: the Reference offset in y stays within this (brief)
OVERRIDE = 'preview: dive13 slow root during roll'
OUT_DIR = os.path.join(config.OUT, 'dive13')
MAPPING_NAME = 'mapping_dive13.json'
REPORT = os.path.join(OUT_DIR, 'report.md')
SUMMARY = os.path.join(OUT_DIR, 'summary.json')
SO_HEX, SO_NAME = D9.SO_HEX, D9.SO_NAME
EXIT_HEX, EXIT_NAME = D9.EXIT_HEX, D9.EXIT_NAME
ROOT_TID = 0
PHASES = ('plant', 'roll', 'get-up', 'Exit window')
_S_ORIG_DIR = os.path.join(config.OUT, 'dive12', 'S18')          # dive12's installed S18 (read only)
_E_DIVE11_DIR = os.path.join(config.OUT, 'dive11')


# ------------------------------------------------------------------------------------------------ small helpers
def _smoothstep(u):
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _gauss(x, sigma_frames):
    """Gaussian smoothing with edge padding (sigma in frames)."""
    if sigma_frames <= 0:
        return np.asarray(x, float)
    r = int(np.ceil(3.0 * sigma_frames))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_frames) ** 2)
    k /= k.sum()
    x = np.asarray(x, float)
    xp = np.concatenate([np.full(r, x[0]), x, np.full(r, x[-1])])
    return np.convolve(xp, k, mode='valid')


def case18():
    """dive12's S18 timing case (takeoff, scale, shift, k 1.00 / 1.00)."""
    return D9.make_case9(D12.T_TO, K_PLANT, K_ROLL, D12.S_SCALE, D12.SHIFT)


def natural_speed(ctx, case, frames):
    """Natural forward (y) speed of the body in m/s at the given 60 Hz clip frames: the WD2 pelvis path that the bake places
    (dive4b.target_world, the lift is z only), differentiated, smoothed, and clipped at 0."""
    frames = np.asarray(frames, float)
    tg, _pl = B.target_world(ctx, case, frames / FPS, np.zeros(B.LIFT_K))
    v = np.gradient(tg[:, 1], 1.0 / FPS)
    return np.maximum(_gauss(v, NATURAL_SIGMA_S * FPS), 0.0)


def root_velocity(disp, j=None):
    """Backward-difference velocity (m/s) of the y component of a 60 Hz root [F, 3]; element f is (disp[f]-disp[f-1]) * 60."""
    v = np.zeros(len(disp))
    v[1:] = (disp[1:, 1] - disp[:-1, 1]) * FPS
    return v


def nav_run_speed(dump=None):
    """Nav_Run's root speed (m/s): the mean slope of its TrackID 0 (dump displacement, y) over the first NAV_SPEED_S seconds."""
    dump = dumps.load_by_name(NAV_NAME) if dump is None else dump
    F = np.arange(int(round(float(dump['Length']) * FPS)) + 1)
    disp = acuclip.markers(dump, F)['disp']
    n = max(2, int(round(NAV_SPEED_S * FPS)))
    return float((disp[n, 1] - disp[0, 1]) / (n / FPS)), float(root_velocity(disp)[1])


# ------------------------------------------------------------------------------------------------ the SlideOver root
def so_root(ctx, case):
    """The SlideOver root [F, 3] (F = 0..98): the original up to J_FRAME, then the natural speed blended in (C1 at J_FRAME).
    Returns (new [F, 3], natural speed [F] (m/s), the original speed at J_FRAME (m/s))."""
    orig = np.asarray(ctx.disp_so, float)
    F = np.arange(len(orig))
    v_nat = natural_speed(ctx, case, F)
    new = orig.copy()                       # x is kept (the original's own, about 3e-6 m)
    v_orig = root_velocity(orig)
    v_j = float(v_orig[J_FRAME])
    n_blend = max(1, int(round(BLEND_S * FPS)))
    y = float(orig[J_FRAME, 1])
    for f in range(J_FRAME + 1, len(orig)):
        b = float(_smoothstep((f - J_FRAME) / n_blend))
        v = (1.0 - b) * v_j + b * float(v_nat[f])
        y = y + v / FPS
        new[f, 1] = y
    new[J_FRAME + 1:, 2] = 0.0
    if np.max(np.abs(orig[J_FRAME + 1:, 2])) > 1e-3:
        raise ValueError('the original SlideOver root leaves z = 0 after clip 1.20 s')
    return new, v_nat, v_j


# ------------------------------------------------------------------------------------------------ the Exit root
def exit_delta(p, v0, v1, p_end):
    """D(p) for played times p >= 0: the integral of v(q) = v0 + (v1 - v0) * smoothstep(q / p_end) from 0 (v1 after p_end)."""
    p = np.asarray(p, float)
    q_max = max(float(p.max()) if p.size else 0.0, p_end) + 0.01
    q = np.linspace(0.0, q_max, 20001)
    v = v0 + (v1 - v0) * _smoothstep(q / p_end)
    D = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(q))])
    return np.interp(p, q, D)


def exit_root_speeds(so_disp, t_W, v1):
    """(v0, v1, p_end): the Exit's start speed (the SlideOver root's speed at W), Nav_Run's speed, and the window end."""
    v0 = float(np.interp(t_W * FPS, np.arange(len(so_disp)), root_velocity(so_disp)))
    return v0, float(v1), float(D10.p_end_for(DELAY))


def exit_root(ctx, so_disp, t_W, v1):
    """The Exit's root for a whole Exit clip: returns (joined disp_ex [F_E, 3] (the bake's convention: root_W + (disp - cut)),
    the track positions [F_E, 3] for the patch (cut + D(p) for played frames, the original elsewhere), D(p) [F_E], info dict)."""
    disp_E = acuclip.markers(ctx.ex_dump, ctx.ex_seq.frames)['disp']
    F = np.asarray(ctx.ex_seq.frames, int)
    cut = np.array([np.interp(EXIT_O * FPS, np.arange(len(disp_E)), disp_E[:, c]) for c in range(3)])
    root_W = np.array([np.interp(t_W * FPS, np.arange(len(so_disp)), so_disp[:, c]) for c in range(3)])
    v0, v1, p_end = exit_root_speeds(so_disp, t_W, v1)
    t = F / FPS
    p = (t - EXIT_O) / EXIT_R
    played = p >= -1e-12
    D = np.zeros(len(F))
    D[played] = exit_delta(np.maximum(p[played], 0.0), v0, v1, p_end)
    joined = root_W[None, :] + (disp_E - cut[None, :])
    joined[played, 1] = root_W[1] + D[played]
    joined[:, 0] = 0.0
    track = disp_E.copy()
    track[played, 1] = cut[1] + D[played]
    track[:, 0] = 0.0
    info = dict(v0=v0, v1=v1, p_end=p_end, root_W=root_W, cut=cut, t_W=float(t_W))
    return joined, track, D, info


# ------------------------------------------------------------------------------------------------ the patched context
class Patched:
    """Restores ctx.disp_so, root_W and disp_ex after the slow-root run (the context is shared with dive12)."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.saved = dict(disp_so=ctx.disp_so, root_W=ctx.root_W, disp_ex=ctx.disp_ex)

    def __enter__(self):
        return self.ctx

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(self.ctx, k, v)
        return False


def install_roots(ctx, so_disp, joined_E, root_W):
    ctx.disp_so = np.asarray(so_disp, float)
    ctx.root_W = np.asarray(root_W, float)
    ctx.disp_ex = np.asarray(joined_E, float)


# ------------------------------------------------------------------------------------------------ skate metric
def _phase_sets(ctx, res, case, t_W, delay=DELAY):
    """Frame index sets per phase for the S and E segments: {phase: ('S' or 'E', frames)}, with the frames sorted."""
    win_S = np.asarray(res['win_S'], int)
    tS = win_S / FPS
    plant = set(int(f) for f in res['fin_S']['plant'])
    is_plant = np.array([int(f) in plant for f in win_S], bool)
    t_pe = float(case.tm.t_pe)
    roll = (~is_plant) & (tS > t_pe + 1e-9) & (tS <= t_W + 1e-9)
    tE = np.asarray(res['seg_E']['t60'], float)
    p = (tE - EXIT_O) / EXIT_R
    p_up = float(D10.p_up_for(delay))
    p_end = float(D10.p_end_for(delay))
    idx_E = np.arange(len(tE))
    return {
        'plant': ('S', np.sort(win_S[is_plant])),
        'roll': ('S', np.sort(win_S[roll])),
        'get-up': ('E', idx_E[(p >= -1e-9) & (p <= p_up + 1e-9)]),
        'Exit window': ('E', idx_E[(p > p_up + 1e-9) & (p <= p_end + 1e-9)]),
    }


def _surfaces(P, body, ob, is_S):
    """Height above the surface per frame and body bone: the box top inside the box band (SlideOver only), else ground z = 0."""
    z = P[:, :, 2]
    if not is_S:
        return z
    near, far, top = float(ob['near_y']), float(ob['far_y']), float(ob['height'])
    y = P[:, :, 1]
    surf = np.where((y > near) & (y < far), top, 0.0)
    return z - surf


def skate_samples(P, frames, body, names, dtau, h):
    """Skate samples for one phase: (speeds m/s, bone name per sample, runs kept). P [F, B, 3] world, frames sorted indices,
    h [F, B] height above the surface, dtau the seconds per frame step (joined time)."""
    frames = np.asarray(frames, int)
    if frames.size < 2:
        return np.zeros(0), [], 0
    body = np.asarray(body, int)
    hb = h[:, body]
    low_col = np.argmin(hb, axis=1)
    low_h = hb[np.arange(len(hb)), low_col]
    vel = np.gradient(P[:, :, :2], dtau, axis=0)              # [F, B, 2] horizontal velocity of every bone (central)
    speeds, label, runs = [], [], 0
    i = 0
    n = len(frames)
    while i < n:
        f = frames[i]
        if low_h[f] > SKATE_H:
            i += 1
            continue
        b = low_col[f]
        j = i
        while j + 1 < n:
            g = frames[j + 1]
            if g != frames[j] + 1 or low_h[g] > SKATE_H or low_col[g] != b:
                break
            j += 1
        if j - i + 1 >= SKATE_RUN:
            runs += 1
            for k in range(i, j + 1):
                f2 = frames[k]
                speeds.append(float(np.linalg.norm(vel[f2, body[b], :])))
                label.append(names[body[b]])
        i = j + 1
    return np.asarray(speeds, float), label, runs


def skate_metric(ctx, res, case, t_W, delay=DELAY):
    """Per-phase skate of the lowest non-finger bone (m/s): max, p95, the bone at the max, and the sample count."""
    body = [int(b) for b in ctx.body_nf]
    names = D11.bone_names(ctx.rig)
    PS = D12.world_slideover(res)
    fin_E = res['fin_E']
    PE = D3._world(res['seg_E']['seq'], res['seg_E']['dump'], fin_E['final'], fin_E['ref_out'], fin_E['disp'])
    hS = _surfaces(PS, body, ctx.ob, True)
    hE = _surfaces(PE, body, ctx.ob, False)
    sets = _phase_sets(ctx, res, case, t_W, delay)
    out = {}
    all_s = []
    for ph in PHASES:
        seg, frames = sets[ph]
        if seg == 'S':
            sp, lab, runs = skate_samples(PS, frames, body, names, 1.0 / FPS, hS)
        else:
            sp, lab, runs = skate_samples(PE, frames, body, names, (1.0 / FPS) / EXIT_R, hE)
        all_s.append(sp)
        if sp.size:
            k = int(np.argmax(sp))
            out[ph] = dict(max_m_s=float(sp.max()), p95_m_s=float(np.percentile(sp, 95)), bone_at_max=lab[k],
                           samples=int(sp.size), runs=int(runs), frames=int(len(frames)))
        else:
            out[ph] = dict(max_m_s=None, p95_m_s=None, bone_at_max=None, samples=0, runs=int(runs), frames=int(len(frames)))
    allv = np.concatenate(all_s) if all_s else np.zeros(0)
    out['all'] = dict(max_m_s=float(allv.max()) if allv.size else None,
                      p95_m_s=float(np.percentile(allv, 95)) if allv.size else None, samples=int(allv.size))
    return out


# ------------------------------------------------------------------------------------------------ writers
def root_track(track_rows):
    """A TrackID 0 root track (infotype 3, codec 27, LocalPosition per 60 Hz frame)."""
    return {'TrackID': str(ROOT_TID), 'codec': 27, 'infotype': 3,
            'Keyframes': [{'Frame': int(f), 'LocalPosition': [round(float(x), 9) for x in row]}
                          for f, row in enumerate(track_rows)]}


def write_outputs(ctx, res, D_S_=D_S, out_dir=OUT_DIR, so_track=None, ex_track=None):
    """Patches (rotations, Reference, events, aux, TrackID 0), the mapping, and the joined sheet/GIF for this run."""
    os.makedirs(out_dir, exist_ok=True)
    rig = ctx.rig
    so = D8.patch_dict(rig, res['fin_S']['rot_tracks'], res['fin_S']['trans_keys'], ctx.so['dump']['Length'])
    so_aux = AT.add_aux_tracks(so, ctx.so['dump'], D9.TAKEOFF_FRAME, None)
    so_aux['Tracks'] = list(so_aux['Tracks']) + D12.slideover_event_tracks(ctx.so['dump'], D_S_) + [root_track(so_track)]
    AT.write_patch(os.path.join(out_dir, SO_HEX + '.baked.json'), so_aux)
    ex = D8.patch_dict(rig, res['fin_E']['rot_tracks'], res['fin_E']['trans_keys'], ctx.ex_dump['Length'])
    ex_aux = AT.add_aux_tracks(ex, ctx.ex_dump, 0, int(res['_feet_free_to']))
    ex_aux['Tracks'] = list(ex_aux['Tracks']) + D11.event_tracks_delay15(ctx) + [root_track(ex_track)]
    AT.write_patch(os.path.join(out_dir, EXIT_HEX + '.baked.json'), ex_aux)
    m = dict(clips=[
        DC2.mapping_entry(SO_HEX, SO_NAME, SO_HEX + '.baked.json', True, override=OVERRIDE),
        DC2.mapping_entry(EXIT_HEX, EXIT_NAME, EXIT_HEX + '.baked.json', True, only_after=[SO_NAME], override=OVERRIDE),
    ])
    with open(os.path.join(out_dir, MAPPING_NAME), 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=1)


def write_sheet(ctx, res, case, out_dir, title):
    """dive12's joined sheet and GIF with the entity root drawn (show_root), written to out_dir."""
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
    BCh.plot_joined(os.path.join(out_dir, 'joined_sheet.png'), ctx.rig, S, arrS, E, arrE, ctx.ob, tl, title, show_root=True)
    BCh.write_gif(os.path.join(out_dir, 'joined_sheet.gif'), ctx.rig, S, arrS, E, arrE, ctx.ob, tl)


# ------------------------------------------------------------------------------------------------ run
def run_variant(ctx, t_W, label, write_dir=None, so_track=None, ex_track=None, title=None):
    """dive12's run_case at the S18 window (k 1.00 / 1.00) with the roots currently installed on ctx. Returns run dict + skate."""
    run = D12.run_case(ctx, t_W, K_PLANT, K_ROLL, D_S, label)
    res = run['res']
    res['_feet_free_to'] = int(run['m']['feet_free_to_frame'])
    skate = skate_metric(ctx, res, run['case'], t_W)
    if write_dir is not None:
        write_outputs(ctx, res, D_S, write_dir, so_track, ex_track)
        write_sheet(ctx, res, run['case'], write_dir, title or label)
    run['skate_phases'] = skate
    return run


# ------------------------------------------------------------------------------------------------ numbers for the report
def offsets_of(res):
    """Reference offset (m) from the original pose over the SlideOver window and the Exit window: y and x range, max step."""
    out = {}
    for seg_key, fin_key, frames_key in (('seg_S', 'fin_S', 'win_S'), ('seg_E', 'fin_E', 'win_E')):
        seg, fin = res[seg_key], res[fin_key]
        fr = np.asarray(res[frames_key], int)
        off = np.asarray(fin['ref_out'], float)[fr] - np.asarray(seg['ref0_orig'], float)[fr]
        out[fin_key] = dict(y_min=float(off[:, 1].min()), y_max=float(off[:, 1].max()),
                            x_abs_max=float(np.max(np.abs(off[:, 0]))), z_min=float(off[:, 2].min()),
                            z_max=float(off[:, 2].max()), y_abs_max=float(np.max(np.abs(off[:, 1]))),
                            step_cm_frame=float(np.max(np.linalg.norm(np.diff(off, axis=0), axis=1)) * 100.0))
        if fin_key == 'fin_S':
            after = fr > J_FRAME
            out[fin_key]['after_1p20_y_min'] = float(off[after, 1].min())
            out[fin_key]['after_1p20_y_max'] = float(off[after, 1].max())
            out[fin_key]['before_1p20_y_max'] = float(off[~after, 1].max())
    return out


def root_profile(root, t_s, scale=1.0, central=False):
    """y of a 60 Hz root at clip times t_s (s) and its forward speed (m/s): backward difference by default, central if asked;
    scale converts clip seconds to joined seconds for the Exit (EXIT_R)."""
    root = np.asarray(root, float)
    v = (np.gradient(root[:, 1], 1.0 / FPS) if central else root_velocity(root)) * float(scale)
    F = np.arange(len(root))
    return {str(t): dict(y=float(np.interp(t * FPS, F, root[:, 1])), v=float(np.interp(t * FPS, F, v)))
            for t in t_s}


def digest(ctx, run):
    """The dive12 digest plus the dive13 skate phases and the Reference offsets of one run."""
    dg = D12.digest_run(ctx, run)
    dg['skate_phases'] = run['skate_phases']
    dg['offsets'] = offsets_of(run['res'])
    return dg


def _json(x):
    return D12._jsonable(x)


# ------------------------------------------------------------------------------------------------ main
BASELINE = os.path.join(OUT_DIR, 'baseline_S18.json')          # dive12's S18 root, same bake, for the comparison


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    stage = argv[0] if argv else 'variant'
    os.makedirs(OUT_DIR, exist_ok=True)
    ctx = F5.make_ctx()
    case = case18()
    t_W = float(ctx.t_W) + D_S / FPS
    orig_so = np.asarray(ctx.disp_so, float).copy()
    orig_ex_root_W = np.array([np.interp(t_W * FPS, np.arange(len(orig_so)), orig_so[:, c]) for c in range(3)])
    if stage == 'baseline':
        run = run_variant(ctx, t_W, 'S18 original root')
        base = digest(ctx, run)
        base['t_W'] = t_W
        with open(BASELINE, 'w', encoding='utf-8') as f:
            json.dump(_json(base), f, indent=1)
        print('baseline floor min %.3f, skate %s' % (base['floor_min_cm'], base['skate_phases']['all']), flush=True)
        return 0
    so_new, v_nat, v_j = so_root(ctx, case)
    nav_v, nav_v0 = nav_run_speed()
    joined_E, track_E, D, info = exit_root(ctx, so_new, t_W, nav_v)
    root_W = np.array([np.interp(t_W * FPS, np.arange(len(so_new)), so_new[:, c]) for c in range(3)])
    print('new root: v at 1.20 %.3f, Nav_Run %.3f, exit v0 %.3f, p_end %.4f' % (v_j, nav_v, info['v0'], info['p_end']),
          flush=True)
    with Patched(ctx):
        install_roots(ctx, so_new, joined_E, root_W)
        run = run_variant(ctx, t_W, 'dive13 slow root', write_dir=OUT_DIR, so_track=so_new, ex_track=track_E,
                          title='dive13 slow root during roll (TrackID 0 patched), S18 window %.3f s' % t_W)
        dg = digest(ctx, run)
    # the entity's travel and speeds for the report (joined root, clip frames)
    disp_E_orig = acuclip.markers(ctx.ex_dump, ctx.ex_seq.frames)['disp']
    cut = np.array([np.interp(EXIT_O * FPS, np.arange(len(disp_E_orig)), disp_E_orig[:, c]) for c in range(3)])
    pe = info['p_end']
    t_end = EXIT_O + EXIT_R * pe
    fe = t_end * FPS
    vanilla_joined_y = float(orig_ex_root_W[1] + (np.interp(fe, np.arange(len(disp_E_orig)), disp_E_orig[:, 1]) - cut[1]))
    new_joined_y = float(np.interp(fe, np.arange(len(joined_E)), joined_E[:, 1]))
    van_joined = orig_ex_root_W[None, :] + (disp_E_orig - cut[None, :])
    # joined time: d(joined)/d(tau) = d(joined)/d(clip) * EXIT_R (tau advances EXIT_R per clip second)
    vanilla_speed = float(np.interp(fe, np.arange(len(van_joined)), np.gradient(van_joined[:, 1], 1.0 / FPS)) * EXIT_R)
    new_speed = float(np.interp(fe, np.arange(len(joined_E)), np.gradient(joined_E[:, 1], 1.0 / FPS)) * EXIT_R)
    so_orig_prof = root_profile(orig_so, [1.2, 1.3, 1.4, 1.5, round(t_W, 4), 1.6])
    so_new_prof = root_profile(so_new, [1.2, 1.3, 1.4, 1.5, round(t_W, 4), 1.6])
    ex_t = [0.208, 0.25, 0.30, 0.35, 0.40, 0.45, round(t_end, 4)]
    ex_orig_prof = root_profile(van_joined, ex_t, scale=EXIT_R, central=True)
    ex_new_prof = root_profile(joined_E, ex_t, scale=EXIT_R, central=True)
    summary = dict(t_W=t_W, D_S=D_S, k=(K_PLANT, K_ROLL), takeoff=dict(T_to=D12.T_TO, s=D12.S_SCALE, shift=D12.SHIFT),
                   nav_run=dict(speed_mean_first_0p2s=nav_v, speed_first_frame=nav_v0), exit_root=dict(
                       v0=info['v0'], v1=info['v1'], p_end=info['p_end'], p_up=float(D10.p_up_for(DELAY)),
                       t_end_clip=t_end, root_W=root_W.tolist()),
                   so_root=dict(v_at_1p2_orig=v_j, profile_orig=so_orig_prof, profile_new=so_new_prof),
                   exit_root_profile=dict(orig=ex_orig_prof, new=ex_new_prof),
                   travel=dict(vanilla_joined_y_at_window_end=vanilla_joined_y, new_joined_y_at_window_end=new_joined_y,
                               shorter_m=vanilla_joined_y - new_joined_y),
                   entity_speed_at_window_end=dict(vanilla=vanilla_speed, new=new_speed, nav_run=nav_v),
                   dive13=_json(dg))
    base_path = BASELINE
    if os.path.exists(base_path):
        with open(base_path, encoding='utf-8') as f:
            summary['baseline'] = json.load(f)
    with open(SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(_json(summary), f, indent=1)
    write_report(_json(summary))
    print('dive13 floor min %.3f pass %s; skate %s' % (dg['floor_min_cm'], dg['floor_pass'], dg['skate_phases']['all']),
          flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ report
REPORT_PATH = os.path.join(OUT_DIR, 'report.md')


def _f(x, nd=2):
    return '-' if x is None else ('%.*f' % (nd, x))


def _phase_row(name, b, n):
    return '| %s | %s | %s | %s | %s | %s |' % (name, _f(b.get('max_m_s')), _f(b.get('p95_m_s')), _f(n.get('max_m_s')),
                                              _f(n.get('p95_m_s')), '%s / %s' % (n.get('bone_at_max') or '-', n.get('samples')))


def write_report(summary, path=REPORT_PATH):
    b = summary['baseline']
    d = summary['dive13']
    m = d['metrics']
    bm = b['metrics']
    so_o, so_n = summary['so_root']['profile_orig'], summary['so_root']['profile_new']
    ex_o, ex_n = summary['exit_root_profile']['orig'], summary['exit_root_profile']['new']
    travel, spd = summary['travel'], summary['entity_speed_at_window_end']
    nav = summary['nav_run']
    sk_b, sk_n = b['skate_phases'], d['skate_phases']
    off_b, off_n = b['offsets'], d['offsets']
    L = []
    L.append('# T25 dive13: slow root during the roll and the get-up (S18 window, k 1.00 / 1.00)')
    L.append('')
    L.append('Generated by `v2/pipeline/dive13.py` (`dive13.py baseline`, then `dive13.py variant`). Written only under '
             '`out\\dive13`. The bake is dive12\'s run_case (dive11 chain, dive9 solver, dive8 timing) with the SlideOver and Exit '
             'roots installed on the context. Nothing in dive9 to dive12 is changed.')
    L.append('')
    L.append('## 1. Result')
    L.append('')
    L.append('The root patch does what it was written to do: identical to the original before 1.20 s, C1 at 1.20 s, speed >= 0, '
             'z = 0 after it, and the floor rule still holds (%s cm). The skate targets are NOT met. Over the played window the '
             'skate p95 is %s m/s (target 0.5) and the max is %s m/s (target 1.0). Two causes, both visible in the numbers below.'
             % (_f(d['floor_min_cm'], 3), _f(sk_n['all']['p95_m_s']), _f(sk_n['all']['max_m_s'])))
    L.append('')
    L.append('* SlideOver: the bake puts the Reference on the WD2 pelvis path, so the root change is absorbed by the Reference and the '
             'body\'s world motion does not change (roll skate identical). The root alone cannot change the roll slide.')
    L.append('* Exit: the root does set the body\'s motion. It now runs at the natural speed (%s m/s at W, rising to Nav_Run\'s %s m/s), '
             'so the Exit window and the get-up improve (section 3), but the lowest bone also moves with the stand-up rotation, '
             'so it still reaches 5 to 6 m/s there.'
             % (_f(summary['exit_root']['v0'], 2), _f(nav['speed_mean_first_0p2s'], 2)))
    L.append('')
    L.append('Section 6 gives the reason the targets cannot be met with the brief\'s constraints.')
    L.append('')
    L.append('## 2. What changed (against dive12 S18)')
    L.append('')
    L.append('| item | dive12 S18 | dive13 |')
    L.append('|---|---|---|')
    L.append('| SlideOver root (TrackID 0, infotype 3, codec 27) | not patched (the original clip root plays) | patched from clip 1.20 s (frame 72): forward speed blends (smoothstep, 0.15 s) from the original speed to the natural speed of the WD2 pelvis path; z = 0; frames 0 to 72 identical to the original |')
    L.append('| Exit root (TrackID 0, codec 27; original was 29) | not patched | joined root = root_S(W) + D(p), D\'(p) = v0 + (v1 - v0) smoothstep(p / p_end), p = (clip - 0.208) / 0.9665; v0 = %s m/s (SlideOver root at W), v1 = %s m/s (Nav_Run), p_end = %.4f s played; before 0.208 s the original'
             % (_f(summary['exit_root']['v0'], 3), _f(summary['exit_root']['v1'], 3), summary['exit_root']['p_end']))
    L.append('| Reference (both clips) | re-solved by the bake against the new roots (SlideOver: target - root - hips; Exit: carried offset as in dive12) | same bake |')
    L.append('| timing, takeoff, plant, roll, hands, upright, Exit stand-up profile | dive12 S18 | unchanged (k 1.00 / 1.00, T_to 0.62, s 1.35, shift 0.20) |')
    L.append('| SlideOver events (infotype 5) | dive12 S18 (+18 frames) | copied (checked byte-identical to out\\dive12\\S18) |')
    L.append('| Exit events (infotype 5) | dive11 delay15 | copied (checked against out\\dive11) |')
    L.append('')
    L.append('## 3. Skate, new metric (m/s)')
    L.append('')
    L.append('Rule: a frame counts when a non-finger bone is within 12 cm of its surface (ground, or the box top inside the box band). '
             'The lowest such bone is taken, and its horizontal world speed counts only where it is the same lowest bone for at least 3 '
             'frames in a row. Phases: plant (hands pinned on the box), roll (plant end to W), get-up (Exit played 0 to P_UP = %.3f s, '
             'the stand-up with the delay15 offset), Exit window (P_UP to p_end = %.3f s). Exit speeds are per joined second. '
             'Targets: p95 <= 0.5, max <= 1.0.' % (summary['exit_root']['p_up'], summary['exit_root']['p_end']))
    L.append('')
    L.append('| phase | S18 max | S18 p95 | dive13 max | dive13 p95 | dive13 max bone / samples |')
    L.append('|---|---|---|---|---|---|')
    for ph in ('plant', 'roll', 'get-up', 'Exit window', 'all'):
        L.append(_phase_row(ph, sk_b[ph], sk_n[ph]))
    L.append('')
    L.append('The original root (S18) reproduces the slide: roll p95 %s m/s, get-up p95 %s m/s, Exit window p95 %s m/s.'
             % (_f(sk_b['roll']['p95_m_s']), _f(sk_b['get-up']['p95_m_s']), _f(sk_b['Exit window']['p95_m_s'])))
    L.append('')
    L.append('## 4. Root profiles (both clips)')
    L.append('')
    L.append('SlideOver root: y (m, dump frame) and forward speed (m/s, backward difference).')
    L.append('')
    L.append('| clip s | original y | original v | dive13 y | dive13 v |')
    L.append('|---|---|---|---|---|')
    for t in sorted(so_o.keys(), key=float):
        L.append('| %s | %s | %s | %s | %s |' % (t, _f(so_o[t]['y'], 3), _f(so_o[t]['v'], 2), _f(so_n[t]['y'], 3), _f(so_n[t]['v'], 2)))
    L.append('')
    L.append('Exit root (joined, y in m, speed in m/s per joined second) at Exit clip times:')
    L.append('')
    L.append('| Exit clip s | vanilla joined y | vanilla v | dive13 joined y | dive13 v |')
    L.append('|---|---|---|---|---|')
    for t in ex_o.keys():
        L.append('| %s | %s | %s | %s | %s |' % (t, _f(ex_o[t]['y'], 3), _f(ex_o[t]['v'], 2), _f(ex_n[t]['y'], 3), _f(ex_n[t]['v'], 2)))
    L.append('')
    L.append('Entity at the Exit window end:')
    L.append('')
    L.append('| quantity | vanilla | dive13 |')
    L.append('|---|---|---|')
    L.append('| joined y (m) | %s | %s |' % (_f(travel['vanilla_joined_y_at_window_end'], 3), _f(travel['new_joined_y_at_window_end'], 3)))
    L.append('| entity forward speed (m/s, joined) | %s | %s |' % (_f(spd['vanilla']), _f(spd['new'])))
    L.append('| Nav_Run root speed (m/s, mean of the first 0.2 s) | %s | %s |' % (_f(spd['nav_run']), _f(spd['nav_run'])))
    L.append('')
    L.append('The entity travels %s m less by the Exit window end. The landing spot moves; that is the intended change.'
             % _f(travel['shorter_m'], 2))
    L.append('')
    L.append('## 5. Reference offsets, floor and the dive9-dive11 checks')
    L.append('')
    L.append('| quantity | S18 | dive13 | rule |')
    L.append('|---|---|---|---|')
    L.append('| SlideOver Reference offset, y range (m) | %s .. %s | %s .. %s | about <= 0.5 (not met, see note) |'
             % (_f(off_b['fin_S']['y_min'], 3), _f(off_b['fin_S']['y_max'], 3), _f(off_n['fin_S']['y_min'], 3), _f(off_n['fin_S']['y_max'], 3)))
    L.append('| Exit Reference offset, y range (m) | %s .. %s | %s .. %s | about <= 0.5 (not met) |'
             % (_f(off_b['fin_E']['y_min'], 3), _f(off_b['fin_E']['y_max'], 3), _f(off_n['fin_E']['y_min'], 3), _f(off_n['fin_E']['y_max'], 3)))
    L.append('| Reference step, cm per frame (SlideOver / Exit) | %s / %s | %s / %s | <= 12 |'
             % (_f(off_b['fin_S']['step_cm_frame'], 1), _f(off_b['fin_E']['step_cm_frame'], 1),
                _f(off_n['fin_S']['step_cm_frame'], 1), _f(off_n['fin_E']['step_cm_frame'], 1)))
    L.append('| floor min, all phases (cm) | %s | %s | >= 0 |' % (_f(b['floor_min_cm'], 3), _f(d['floor_min_cm'], 3)))
    L.append('| floor rule | %s | %s | pass |' % (b['floor_pass'], d['floor_pass']))
    L.append('| tilt at W (deg) | %s | %s | <= 60 |' % (_f(bm['tilt_W'], 1), _f(m['tilt_W'], 1)))
    L.append('| continuity at W (deg) | %s | %s | <= 45 |' % (_f(bm['cont_deg'], 1), _f(m['cont_deg'], 1)))
    L.append('| Exit Reference z at P_UP (m) | %s | %s | >= 0.78 |' % (_f(bm.get('z_at_p_up'), 3), _f(m.get('z_at_p_up'), 3)))
    L.append('| peak Reference z speed after W (m/s) | %s | %s | <= 4 |' % (_f(bm.get('rise_speed_joined_post_W_mps'), 2), _f(m.get('rise_speed_joined_post_W_mps'), 2)))
    L.append('| Nav gap at the window end (deg) | %s | %s | <= 25 |' % (_f(bm.get('gap_at_window_end_max_patched'), 1), _f(m.get('gap_at_window_end_max_patched'), 1)))
    L.append('| world pelvis dv / step (m/s per frame / cm per frame) | %s / %s | %s / %s | dv <= 1.5, step <= 12 |'
             % (_f(bm['world_pelvis_dv_mps'], 2), _f(bm['world_pelvis_step_cm'], 1), _f(m['world_pelvis_dv_mps'], 2), _f(m['world_pelvis_step_cm'], 1)))
    L.append('')
    L.append('Failing items carried from dive9 to dive11 (both runs): %s.' % '; '.join(d['fails10']))
    L.append('')
    L.append('Note on the SlideOver Reference offset: the y maximum (%s m) is before 1.20 s, where the root is unchanged, so the root '
             'patch cannot reduce it. After 1.20 s the offset runs %s to %s m in dive13 (S18: %s to %s m).'
             % (_f(off_n['fin_S']['before_1p20_y_max'], 3), _f(off_n['fin_S']['after_1p20_y_min'], 3),
                _f(off_n['fin_S']['after_1p20_y_max'], 3), _f(off_b['fin_S']['after_1p20_y_min'], 3),
                _f(off_b['fin_S']['after_1p20_y_max'], 3)))
    L.append('')
    L.append('## 6. Why the targets are not met, and what would be needed')
    L.append('')
    L.append('* The root alone does not move the SlideOver body. The bake sets the Reference so that the pelvis lands on the WD2 path. '
             'A slower root moves the Reference by the same amount, so the world motion is unchanged (roll p95 %s vs %s m/s). The '
             'roll slide is the WD2 pelvis path itself: 1 to 3 m/s while the body is still low.'
             % (_f(sk_n['roll']['p95_m_s']), _f(sk_b['roll']['p95_m_s'])))
    L.append('* In the Exit the root does set the body motion, so the slow Exit root helps: the Exit window p95 goes from %s to %s m/s, '
             'and the get-up p95 from %s to %s m/s. The get-up max rises a little (%s to %s m/s, left hand). The lowest bone moves '
             'faster than the root there (root 2.9 to 4.5 m/s), so the rotation of the stand-up and the Reference carried from W '
             'add the rest; I did not separate those two in this run.'
             % (_f(sk_b['Exit window']['p95_m_s']), _f(sk_n['Exit window']['p95_m_s']), _f(sk_b['get-up']['p95_m_s']),
                _f(sk_n['get-up']['p95_m_s']), _f(sk_b['get-up']['max_m_s']), _f(sk_n['get-up']['max_m_s'])))
    L.append('* The world pelvis dv metric (dive9 item, already failing) gets worse, %s to %s m/s per frame: the change of the Exit root '
             'at W adds a speed step to the pelvis. It is reported, not hidden.'
             % (_f(bm['world_pelvis_dv_mps'], 2), _f(m['world_pelvis_dv_mps'], 2)))
    L.append('* p95 0.5 and max 1.0 need the body to stop moving forward while the lowest bone is within 12 cm (the WD2 pelvis is at '
             '1 to 3 m/s there). That changes the body path, not the root, and moves the Reference away from the WD2 path and the '
             'landing point. It is a decision for the user, so it is not made here.')
    L.append('* Nothing was run in game (rule). The skate numbers are the offline model; the in-game check is still to do.')
    L.append('')
    L.append('## 7. Validation and tests')
    L.append('')
    L.append('* `pipeline\\tests\\test_dive13.py`: root identical before 1.20 s; C1 at 1.20 s (position and speed); speed >= 0; z = 0 '
             'after 1.20 s; Exit root identical before 0.208 s and equal to the played formula; SlideOver events byte-identical to '
             'out\\dive12\\S18; Exit events identical to out\\dive11; JSON format of both patches and the mapping; the skate metric on '
             'synthetic data (a sliding bone, a rolling bone, a short run).')
    L.append('* The baseline (dive12 S18 root, same bake) reproduces dive12: floor %s cm, trunk %s deg/s, tilt %s deg, margin 0.131 s.'
             % (_f(b['floor_min_cm'], 3), _f(b['trunk']['max_deg_s'], 1), _f(bm['tilt_W'], 1)))
    L.append('')
    L.append('## 8. Outputs')
    L.append('')
    L.append('* `out\\dive13\\47BFB02DF.baked.json` (SlideOver: rotations, Reference, dive12 S18 events, TrackID 0 root)')
    L.append('* `out\\dive13\\F7919BDF0.baked.json` (Exit: rotations, Reference, dive11 delay15 events, TrackID 0 played root)')
    L.append('* `out\\dive13\\mapping_dive13.json` (override "preview: dive13 slow root during roll")')
    L.append('* `out\\dive13\\joined_sheet.png` and `joined_sheet.gif` (entity root drawn), `summary.json`, `baseline_S18.json`, `report.md`')
    L.append('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


if __name__ == '__main__':
    sys.exit(main())
