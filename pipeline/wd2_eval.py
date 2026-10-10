"""Evaluate Watch Dogs 2 vault clips as replacements for ACU vault clips (analysis only: nothing is baked or installed).

For each WD2 clip (rotations only; positions are pelvis-relative FK, plus a contact-anchored pelvis path):
  events    takeoff, hand plant and landing from the candidate contacts of wd2events (body speed from the run-in).
            Clips without a hand plant use the largest gap between foot contacts as the flight.
  trunk     max Hips->Head tilt from upright in the flight, inversion class, max trunk angular speed.
  hands     hand depth below the pelvis during the flight (a plant on a top when > 0.3 m) and when.
  path      pelvis path from one least-squares problem: foot contacts pin the pelvis height and xy travel, hand
            contacts on the flight pin hands to a free surface height, flight is ballistic, origin at takeoff.
            Gives the forward travel takeoff->landing, the hips height profile, and the measured top height.
  recovery  time from landing until the hips are at >= 85 % of the standing height for 0.25 s.

For each ACU target (the dumps in config.ACU_DUMPS): takeoff, hand plant, landing and obstacle from contacts.events,
root displacement travel, and the played window (out\\match\\play_model*.json) that bounds the clip after landing.

Score: 0..100 per (WD2 clip, ACU target), see SCORE_WEIGHTS. Verdict: strong fit / possible / poor fit.

Run:  python wd2_eval.py      writes v2\\out\\wd2_eval\\{report.md, summary.json, sheets\\wd2_<stem>.png}
Nothing else is written. Existing outputs (out\\sheets, out\\match, patches) are read, never modified.
"""
import json
import math
import os
import re
import sys
import zlib

import numpy as np

import acuclip
import config
import contacts as acu_contacts
import dumps
import retarget
import wd2
import wd2events as we
import wd2_path as wp

DT_SRC = 1.0 / 30.0                  # s, WD2 source tick
DT_PATH = 1.0 / 60.0                 # s, path grid
G = 9.81
FOOT_Z = wp.FOOT_Z                   # m, WD2 Foot joint above the ground (contact)
HAND_Z = wp.HAND_Z                   # m, WD2 Hand joint above the surface it is planted on
W_CONTACT = 1000.0                   # heavy
W_BALLISTIC = 100.0                  # medium
W_VELOCITY = 100.0                   # medium
LAM_SMOOTH = 0.1
MERGE_GAP = we.HAND_MERGE_TICKS * DT_SRC + 1e-9
MIN_FLIGHT_S = 0.2                   # s, smallest foot gap accepted as the flight when there is no hand plant
HAND_DEPTH_M = 0.3                   # m, hands this far below the pelvis in flight = planted on a top
UPRIGHT_HIPS = 0.85                  # fraction of the standing hips height that counts as upright-running
SUSTAIN_S = 0.25                     # s, hips must stay above UPRIGHT_HIPS this long
TILT_MILD_DEG = 90.0                 # tilt past vertical: head lower than hips
TILT_SEVERE_DEG = 120.0

HAND_EFFS = ('LeftHand', 'RightHand')
FOOT_EFFS = ('LeftFoot', 'RightFoot', 'LeftFootThumb1', 'RightFootThumb1')
WD_OF = {'LeftHand': 'L Hand', 'RightHand': 'R Hand', 'LeftFoot': 'L Foot', 'RightFoot': 'R Foot'}
FINGER_WORDS = wp.FINGER_WORDS

ANALYSIS_DIR = os.path.join(config.OUT, 'wd2_eval')
SHEETS_DIR = os.path.join(ANALYSIS_DIR, 'sheets')
REPORT_MD = os.path.join(ANALYSIS_DIR, 'report.md')
SUMMARY_JSON = os.path.join(ANALYSIS_DIR, 'summary.json')
PLAY_SKIP = os.path.join(config.OUT, 'match', 'play_model.json')
PLAY_SLIDE = os.path.join(config.OUT, 'match', 'play_model_slideover.json')

KONG = 'b00-000-00_vaultover-kong_mplyr_000f_norm_nowep'
FRONTDIVE = 'a00-000-00_vaultover-sprint-100cm-long-frontdive_male_000f_norm_nowep'

CANDIDATES = [
    'a00-000-00_vaultover-sprint-100cm-long-vaultover_male_000f_norm_nowep',
    'a00-000-00_vaultover-sprint-200cm-long-vaultover_male_000f_norm_nowep',
    'a00-000-00_vaultover-sprint-100cm-long-sideflip_male_000f_norm_nowep',
    'b00-000-00_vaultover-jumpover-sprint-200cm_mplyr_000f_norm_nowep',
    'b07-001-03_climbover-100cm-2handtap_mplyr_000f_norm_nowep',
    'b07-001-03_climbover-100cm-rollflip_mplyr_000f_norm_nowep',
    'b07-008-06_vaultover-sprint-200cm-long-handstand_male_000f_norm_nowep',
    'b07-008-06_vaultover-sprint-200cm-long-vaultkick_male_000f_norm_nowep',
    'b07-024-02_vaultover-sprint-100cm-long-jumponflip_male_000f_norm_nowep',
    'b07-024-03_climbover-100cm-minikong_mplyr_000f_norm_nowep',
    'a00-000-00_sprint-50-cm-left-foot-climbover_male_000f_norm_nowep',
    'a00-000-00_sprint-min-80cm-left-foot-climbover_male_000f_norm_nowep',
    'a00-000-00_sprint-max-140cm-left-foot-climbover_male_000f_norm_nowep',
    'a00-000-00_ers-climbover-sprint-100cm-climbover_male_000f_norm_nowep',
]
REFERENCES = {'kong': KONG, 'frontdive': FRONTDIVE}

# ACU targets: (clip name, group). 'free' = not used by the kong; 'taken' = reference only (kong bake).
ACU_TARGETS = [
    ('xx_Parkour_Sprint_SlideOver_Fd100cm-Up100cm', 'free'),
    ('xx_Parkour_Sprint_SlideOver_Fd200cm-Up100cm', 'free'),
    ('xx_Parkour_Sprint_SlideOver_Fd300cm-Up100cm', 'free'),
    ('xx_Parkour_Sprint_SlideOver_Fd100cm-Up050cm', 'free'),
    ('xx_Parkour_Sprint_SlideOver_Fd200cm-Up050cm', 'free'),
    ('xx_Parkour_Sprint_SlideOver_Fd300cm-Up050cm', 'free'),
    ('xx_Parkour_Sprint_SlideOver_Fd100cm-Up150cm', 'free'),
    ('xx_Parkour_Sprint_SlideOver_Fd200cm-Up150cm', 'free'),
    ('xx_Parkour_Sprint_SlideOver_Fd300cm-Up150cm', 'free'),
    ('xx_Parkour_Sprint_SlideOver_Fd200cm-Up200cm', 'extra'),   # not in the spec's free list: reference only
    ('xx_Parkour_Sprint_SkipOver_Fd100cm-Up050cm', 'free'),
    ('xx_Parkour_Sprint_SkipOver_Fd200cm-Up050cm', 'free'),
    ('xx_Parkour_Sprint_SkipOver_Fd100cm-Up150cm', 'free'),
    ('xx_Parkour_Sprint_SkipOver_Fd200cm-Up150cm', 'free'),
    ('xx_Parkour_Sprint_SkipOver_Fd300cm-Up150cm', 'free'),
    ('xx_Parkour_Sprint_SkipOver_Fd010cm-Up200cm', 'free'),
    ('xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm', 'taken'),
    ('xx_Parkour_Sprint_SkipOver_Fd200cm-Up100cm', 'taken'),
    ('xx_Parkour_Sprint_SkipOver_Fd010cm-Up100cm', 'taken'),
]
DEFAULT_PLAY_O = 0.38

SCORE_WEIGHTS = dict(timing=15, upright=25, recovery=20, hands=10, height=15, distance=15)
VERDICT_STRONG = 70                  # strong fit: score >= 70, no severe inversion, upright by the cut
VERDICT_POSSIBLE = 55                # possible: score >= 55. The kong reference (80) is strong, the frontdive (53) poor.
TIMING_IDEAL = (0.85, 1.2)
ACU_GROUP = dict(ACU_TARGETS)


# ------------------------------------------------------------------------------------------------ small helpers
def merge_intervals(ivs, gap):
    out = []
    for a, b in sorted(ivs):
        if out and a <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def name_cm(stem):
    """Height class in cm from the clip name ('...-100cm...'), or None. Only a hint: the WD2 name is not verified."""
    m = re.search(r'(\d+)[- ]?cm', stem)
    return int(m.group(1)) if m else None


def _angle_deg(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


# ------------------------------------------------------------------------------------------------ 1. events
def detect_events(contacts):
    """takeoff, landing, hand plant (start, end) and the rule used, from contact intervals (seconds, ACU geometry).

    Hand plant rule: first merged hand run; takeoff = end of the last foot run ending before it; landing = start of the
    first foot run starting after it. No hand run: the flight is the largest gap between merged foot runs (>= 0.2 s).
    """
    hands = merge_intervals([iv for h in HAND_EFFS for iv in contacts.get(h, [])], MERGE_GAP)
    feet_all = sorted(iv for f in FOOT_EFFS for iv in contacts.get(f, []))
    feet = merge_intervals(feet_all, MERGE_GAP)
    ev = dict(takeoff=None, landing=None, hand_plant_start=None, hand_plant_end=None, rule=None, gap_s=None)
    if hands and feet_all:
        hs, he = hands[0]
        before = [b for _a, b in feet_all if b < hs]
        after = [a for a, _b in feet_all if a > he]
        if before and after:
            ev.update(takeoff=max(before), landing=min(after), hand_plant_start=hs, hand_plant_end=he,
                      rule='hand plant', gap_s=float(min(after) - max(before)))
            return ev
    best = None
    for i in range(len(feet) - 1):
        g = feet[i + 1][0] - feet[i][1]
        if best is None or g > best[0]:
            best = (g, i)
    if best is not None and best[0] >= MIN_FLIGHT_S:
        i = best[1]
        ev.update(takeoff=feet[i][1], landing=feet[i + 1][0], rule='foot gap', gap_s=float(best[0]))
        if hands:
            ev.update(hand_plant_start=hands[0][0], hand_plant_end=hands[0][1])
    return ev


def wd2_traces(stem):
    """Pelvis-relative effector traces (wd2events.effector_traces) plus candidate contacts and detected events."""
    tr = we.effector_traces(stem)
    c = we.candidate_contacts(tr)
    ev = detect_events(c)
    return tr, c, ev


def trunk_and_hands(tr, ev):
    """Trunk tilt, inversion class and angular speed, and hand depth, over the flight (30 Hz source ticks)."""
    t = tr['t']
    J = tr['joints']
    up = np.array([0.0, 0.0, 1.0])
    vec = J['Head'] - J['Hips']
    nrm = np.linalg.norm(vec, axis=1)
    unit = vec / np.maximum(nrm, 1e-9)[:, None]
    tilt = np.degrees(np.arccos(np.clip(unit @ up, -1.0, 1.0)))
    ang_speed = np.zeros(len(t))
    for k in range(1, len(t)):
        ang_speed[k] = _angle_deg(unit[k - 1], unit[k]) / DT_SRC
    out = dict(tilt_max_all=float(tilt.max()))
    to, ld = ev.get('takeoff'), ev.get('landing')
    if to is None or ld is None:
        out.update(tilt_max_flight=None, inversion='unknown', head_below_hips_s=None, ang_speed_max=None)
        return out, tilt, ang_speed
    fl = (t >= to - 1e-9) & (t <= ld + 1e-9)
    tmax = float(tilt[fl].max()) if fl.any() else float('nan')
    below = float(np.sum(vec[fl, 2] < 0.0) * DT_SRC)
    if tmax > TILT_SEVERE_DEG:
        inv = 'severe'
    elif tmax >= TILT_MILD_DEG:
        inv = 'mild'
    else:
        inv = 'none'
    out.update(tilt_max_flight=tmax, tilt_at_s=float(t[fl][np.argmax(tilt[fl])]), inversion=inv,
               head_below_hips_s=below, ang_speed_max=float(ang_speed[fl].max()) if fl.any() else None,
               ang_speed_p95=float(np.percentile(ang_speed[fl], 95)) if fl.any() else None)
    hz = np.minimum(tr['eff']['LeftHand']['z'], tr['eff']['RightHand']['z'])
    deep = fl & (hz <= -HAND_DEPTH_M)
    if deep.any():
        out.update(hand_depth_max=float(-hz[fl].min()), hands_below_0p3_from=float(t[deep][0]),
                   hands_below_0p3_to=float(t[deep][-1]), hands_on_top=True)
    else:
        out.update(hand_depth_max=float(-hz[fl].min()) if fl.any() else None, hands_below_0p3_from=None,
                   hands_below_0p3_to=None, hands_on_top=False)
    return out, tilt, ang_speed


def exit_speed(tr, c, ev):
    """Body speed after landing from the foot runs that start after the landing (m/s), or None."""
    ld = ev.get('landing')
    if ld is None:
        return None
    vals = []
    for name in FOOT_EFFS:
        vf = np.asarray(tr['eff'][name]['vf'])
        for a, b in c.get(name, []):
            if a < ld - 1e-9:
                continue
            i0, i1 = int(round(a / DT_SRC)), int(round(b / DT_SRC))
            seg = vf[i0:i1 + 1]
            if seg.size and float(np.median(seg)) < 0:
                vals.append(-float(np.median(seg)))
    return float(np.median(vals)) if vals else None


def speed_class(s):
    if s is None:
        return 'unknown'
    if s >= 4.5:
        return 'sprint'
    if s >= 2.0:
        return 'jog'
    return 'walk/idle'


# ------------------------------------------------------------------------------------------------ 2. path
def _contiguous(ks):
    """Split a sorted index array into runs of consecutive indices."""
    if ks.size == 0:
        return []
    cut = np.nonzero(np.diff(ks) > 1)[0]
    return np.split(ks, cut + 1)


def _runs_on_grid(contacts, t, takeoff, landing):
    """Contact runs on the path grid: dict(name=WD2 bone, kind, ks (frame indices), top, a, b).

    A candidate run is split at the flight boundary: the frames before takeoff and after landing are ground runs
    (z pinned to the ground for feet, to HAND_Z for hands); inside the flight a hand run is one top run (z free) and a
    foot run is dropped (feet are airborne in the flight). Without a flight every run is a ground run.
    """
    runs = []
    eps = 1e-9
    have_flight = takeoff is not None and landing is not None
    for ac in WD_OF:                      # hands and Foot joints only (toes are not pinned)
        for a, b in contacts.get(ac, []):
            ks = np.nonzero((t >= a - 1e-9) & (t <= b + 1e-9))[0]
            if ks.size == 0:
                continue
            kind = 'hand' if ac in HAND_EFFS else 'foot'
            if have_flight:
                inside = (t[ks] > takeoff + eps) & (t[ks] < landing - eps)
            else:
                inside = np.zeros(ks.size, bool)
            for seg in _contiguous(ks[~inside]):
                runs.append(dict(name=WD_OF[ac], kind=kind, ks=seg, top=False, a=float(a), b=float(b)))
            if inside.any() and kind == 'hand':          # feet are airborne in the flight: inside foot frames dropped
                runs.append(dict(name=WD_OF[ac], kind=kind, ks=ks[inside], top=True, a=float(a), b=float(b)))
    return runs


PRUNE_M = 0.06                       # m, a ground run whose pelvis-implied height deviates more than this is dropped
PRUNE_PASSES = 3


def solve_path(rig, m, contacts, takeoff, landing, dur, ground_windows=(), lam=LAM_SMOOTH):
    """Contact-anchored pelvis path (WD2 world, metres) on the 60 Hz grid 0 .. dur.

    Candidate contacts that do not fit the rest (a ground run whose implied height is more than PRUNE_M off the
    ground, e.g. an arm sweep that matched the speed band) are dropped and the problem is solved again (3 passes).
    The dropped runs are returned in 'dropped'.
    """
    t = np.arange(0.0, float(dur) + 1e-9, DT_PATH)
    rel = wp.relative_fk(rig, m, t)
    runs = _runs_on_grid(contacts, t, takeoff, landing)
    dropped = []
    sol = None
    for _ in range(PRUNE_PASSES):
        sol = _solve_runs(rig, t, rel, runs, takeoff, landing, ground_windows, lam)
        bad = [r for r in sol['runs'] if (not r['top']) and r['dev_m'] > PRUNE_M]
        if not bad:
            break
        dropped += [dict(name=r['name'], t0=round(float(t[r['ks'][0]]), 3), t1=round(float(t[r['ks'][-1]]), 3),
                         dev_m=round(float(r['dev_m']), 3)) for r in bad]
        runs = [r for r in runs if not any(r is b for b in bad)]
        sol = None
    if sol is None:
        sol = _solve_runs(rig, t, rel, runs, takeoff, landing, ground_windows, lam)
    sol['dropped'] = dropped
    return sol


def _solve_runs(rig, t, rel, runs, takeoff, landing, ground_windows, lam):
    """One least-squares solve for the given contact runs (see solve_path)."""
    F = len(t)
    jt = wp.joint_sets(rig)
    idx, body = jt['idx'], jt['body']

    nvar = 3 * F
    for r in runs:
        r['a_xy'] = nvar
        nvar += 2
    for r in runs:
        if r['top']:
            r['a_z'] = nvar
            nvar += 1
    sysm = wp._System(nvar)

    contact_frames = np.zeros(F, bool)
    for r in runs:
        contact_frames[r['ks']] = True
        i = idx[r['name']]
        for k in r['ks']:
            for c in (0, 1):
                sysm.add([(3 * k + c, 1.0), (r['a_xy'] + c, -1.0)], -rel[k, i, c], W_CONTACT, 'contact_xy')
            if r['top']:
                sysm.add([(3 * k + 2, 1.0), (r['a_z'], -1.0)], -rel[k, i, 2], W_CONTACT, 'top_z')
            else:
                z0 = FOOT_Z if r['kind'] == 'foot' else HAND_Z
                sysm.add([(3 * k + 2, 1.0)], z0 - rel[k, i, 2], W_CONTACT, 'ground_z')

    if takeoff is not None and landing is not None:
        ga = np.array([0.0, 0.0, -G]) * DT_PATH * DT_PATH
        flight = (t >= takeoff - 1e-9) & (t <= landing + 1e-9)
        for k in range(1, F - 1):
            if flight[k] and not contact_frames[k - 1:k + 2].any():
                for c in range(3):
                    sysm.add([(3 * (k - 1) + c, 1.0), (3 * k + c, -2.0), (3 * (k + 1) + c, 1.0)], ga[c],
                             W_BALLISTIC, 'flight')

    for a, b in ground_windows:
        ks = np.nonzero((t >= a - 1e-9) & (t <= b + 1e-9))[0]
        prev_b = None
        for k in ks:
            bk = body[int(np.argmin(rel[k, body, 2]))]
            sysm.add([(3 * k + 2, 1.0)], -rel[k, bk, 2], W_CONTACT, 'roll_z')
            if k - 1 in ks and prev_b == bk:
                for c in range(3):
                    sysm.add([(3 * k + c, 1.0), (3 * (k - 1) + c, -1.0)], -(rel[k, bk, c] - rel[k - 1, bk, c]),
                             W_VELOCITY, 'roll_vel')
            prev_b = bk

    k_to = int(np.argmin(np.abs(t - (takeoff if takeoff is not None else 0.0))))
    for c in (0, 1):
        sysm.add([(3 * k_to + c, 1.0)], 0.0, W_CONTACT, 'origin')
    for k in range(1, F - 1):
        for c in range(3):
            sysm.add([(3 * (k - 1) + c, 1.0), (3 * k + c, -2.0), (3 * (k + 1) + c, 1.0)], 0.0, lam, 'smooth')

    x, res, tags = sysm.solve()
    P = x[:3 * F].reshape(F, 3)
    tags = np.asarray(tags)
    resid = {}
    for tag in sorted(set(tags)):
        r = res[tags == tag]
        resid[str(tag)] = dict(rms=float(np.sqrt(np.mean(r ** 2))), max=float(np.max(np.abs(r))), n=int(r.size))
    tops_h = [float(x[r['a_z']] ) for r in runs if r['top'] and r['kind'] == 'hand']
    tops_f = [float(x[r['a_z']]) for r in runs if r['top'] and r['kind'] == 'foot']
    for r in runs:
        r['anchor_xy'] = [float(x[r['a_xy']]), float(x[r['a_xy'] + 1])]
        if r['top']:
            r['anchor_z'] = float(x[r['a_z']])
    for r in runs:
        i = idx[r['name']]
        if r['top']:
            r['dev_m'] = 0.0
        else:
            z0 = FOOT_Z if r['kind'] == 'foot' else HAND_Z
            r['dev_m'] = float(np.mean(np.abs(P[r['ks'], 2] + rel[r['ks'], i, 2] - z0)))
    return dict(t=t, P=P, rel=rel, idx=idx, body=body, runs=runs, residuals=resid,
                hand_top_z=float(np.median(tops_h)) if tops_h else None,
                foot_top_z=float(np.median(tops_f)) if tops_f else None,
                n_ground=int(sum(1 for r in runs if not r['top'] and r['kind'] == 'foot')))


def path_metrics(sol, takeoff, landing):
    """Forward travel (m, WD2 +Y) takeoff->landing, standing hips, hips profile and recovery time after landing."""
    t, P = sol['t'], sol['P']
    out = {}
    if takeoff is None or landing is None:
        return dict(travel_m=None, standing_hips_m=None, recovery_s=None)
    kt = int(np.argmin(np.abs(t - takeoff)))
    kl = int(np.argmin(np.abs(t - landing)))
    out['travel_m'] = float(P[kl, 1] - P[kt, 1])
    ground_mask = np.zeros(len(t), bool)
    for r in sol['runs']:
        if r['kind'] == 'foot' and not r['top']:
            ground_mask[r['ks']] = True
    pre = ground_mask & (t < takeoff - 1e-9)
    std = float(np.median(P[pre, 2])) if pre.any() else None
    out['standing_hips_m'] = std
    out['hips_z_at_landing_m'] = float(P[kl, 2])
    out['recovery_s'] = recovery_time(P[:, 2], t, landing, std)
    return out


def recovery_time(z, t, landing, standing):
    """Seconds from landing until z >= UPRIGHT_HIPS * standing for SUSTAIN_S (None when not reached in the clip)."""
    if standing is None or standing <= 0:
        return None
    need = UPRIGHT_HIPS * standing
    n_sus = int(round(SUSTAIN_S / (t[1] - t[0]))) if len(t) > 1 else 1
    start = int(np.argmin(np.abs(t - landing)))
    ok = z >= need
    for k in range(start, len(z) - n_sus + 1):
        if ok[k:k + n_sus].all():
            return float(t[k] - landing)
    return None


def foot_hips_recovery(tr, ev):
    """Recovery from foot contact only (no path): hips = FOOT_Z - lowest foot z (pelvis-relative), ACU geometry."""
    t = tr['t']
    lowest = np.minimum(tr['eff']['LeftFoot']['z'], tr['eff']['RightFoot']['z'])
    hips = FOOT_Z - lowest
    to, ld = ev.get('takeoff'), ev.get('landing')
    if to is None or ld is None:
        return None, None
    pre = t < to - 1e-9
    std = float(np.median(hips[pre])) if pre.any() else None
    return recovery_time(hips, t, ld, std), std


# ------------------------------------------------------------------------------------------------ 3. one WD2 clip
def analyse_wd2(stem, rig=None):
    """All measures for one WD2 clip. Returns (summary dict, arrays dict for the sheet)."""
    rig = rig or retarget.Rig()
    m = wd2.load(stem)
    tr, c, ev = wd2_traces(stem)
    dur = float(m['dur'])
    res = dict(stem=stem, duration_s=dur, speed_in_mps=float(tr['speed']), speed_in_class=speed_class(tr['speed']),
               events=ev, contacts={k: [[round(a, 4), round(b, 4)] for a, b in v] for k, v in c.items()})
    trunk, tilt, angsp = trunk_and_hands(tr, ev)
    res.update(trunk)
    res['speed_out_mps'] = exit_speed(tr, c, ev)
    res['speed_out_class'] = speed_class(res['speed_out_mps'])
    if ev['takeoff'] is None or ev['landing'] is None:
        res.update(status='no flight detected', flight_s=None, detection_confidence='low',
                   confidence_notes=['no flight detected'])
        return res, dict(tr=tr, tilt=tilt, angsp=angsp, sol=None)
    T = float(ev['landing'] - ev['takeoff'])
    res['flight_s'] = T
    res['detect_rule'] = ev['rule']
    sol = solve_path(rig, m, c, ev['takeoff'], ev['landing'], dur)
    pm = path_metrics(sol, ev['takeoff'], ev['landing'])
    fr_rec, fr_std = foot_hips_recovery(tr, ev)
    rr = sol['residuals']
    path_ok = (sol['n_ground'] >= 1 and rr.get('flight', {}).get('rms', 0.0) < 0.05
               and rr.get('contact_xy', {}).get('rms', 0.0) < 0.05 and rr.get('ground_z', {}).get('rms', 0.0) < 0.05)
    res['path'] = dict(ok=bool(path_ok), travel_m=pm['travel_m'], standing_hips_m=pm['standing_hips_m'],
                       hips_z_at_landing_m=pm.get('hips_z_at_landing_m'), recovery_s=pm['recovery_s'],
                       hand_top_z_m=sol['hand_top_z'], foot_top_z_m=sol['foot_top_z'],
                       residuals=sol['residuals'], n_ground_runs=sol['n_ground'],
                       dropped=sol.get('dropped', []))
    res['recovery_foot_s'] = fr_rec
    res['standing_hips_foot_m'] = fr_std
    res['recovery_s'], res['recovery_source'] = (fr_rec, 'feet') if fr_rec is not None else (pm['recovery_s'], 'path')
    res['status'] = 'ok' if path_ok else 'ok (path not reliable)'
    res['detection_confidence'], res['confidence_notes'] = detection_confidence(res, ev)
    return res, dict(tr=tr, tilt=tilt, angsp=angsp, sol=sol, c=c, ev=ev)


def detection_confidence(res, ev):
    """'high' / 'low' with the reasons. Low when the run-in speed estimate is below a sprint-like 2 m/s, when the
    flight came from the foot-gap rule with a short gap, when the flight is shorter than a vault (0.4 s), or when hand
    contacts come before the takeoff."""
    notes = []
    if res.get('speed_in_mps', 0.0) < 2.0:
        notes.append('run-in speed %.2f m/s (too slow for a sprint clip)' % res.get('speed_in_mps', 0.0))
    if ev.get('rule') == 'foot gap' and (ev.get('gap_s') or 0.0) < 0.5:
        notes.append('flight from a %.2f s foot gap (no hand plant)' % (ev.get('gap_s') or 0.0))
    if res.get('flight_s') is not None and res['flight_s'] < 0.4:
        notes.append('flight %.2f s (shorter than a vault)' % res['flight_s'])
    hands = [iv for h in HAND_EFFS for iv in res.get('contacts', {}).get(h, [])]
    if ev.get('takeoff') is not None and any(b < ev['takeoff'] for _a, b in hands):
        notes.append('hand contact before takeoff')
    return ('low' if notes else 'high'), notes


def analyse_roll_reference(stem, rig=None, ground_windows=((0.92, 1.18),)):
    """Dive reference: same as analyse_wd2 but with the annotated roll (body on the ground) as a ground window."""
    rig = rig or retarget.Rig()
    res, arr = analyse_wd2(stem, rig)
    if arr.get('sol') is None:
        return res, arr
    m = wd2.load(stem)
    ev = arr['ev']
    sol = solve_path(rig, m, arr['c'], ev['takeoff'], ev['landing'], float(m['dur']), ground_windows=ground_windows)
    pm = path_metrics(sol, ev['takeoff'], ev['landing'])
    res['path_with_roll_window'] = dict(travel_m=pm['travel_m'], recovery_s=pm['recovery_s'],
                                        residuals=sol['residuals'], standing_hips_m=pm['standing_hips_m'])
    arr['sol_roll'] = sol
    return res, arr


# ------------------------------------------------------------------------------------------------ 4. ACU targets
def _acu_dump_by_name(name):
    for handle, nm, _path in dumps.list_dumps():
        if nm == name:
            return handle, dumps.load_by_name(name)
    return None, None


def _play_for(name, length):
    """(o, r, window_s, source, measured) for a target clip, from the match models.

    measured is False when the window is a default or the clip length (the cut is then not known)."""
    if 'SkipOver' in name and os.path.exists(PLAY_SKIP):
        d = json.load(open(PLAY_SKIP, encoding='utf-8')).get('per_clip', {})
        if name in d and 'window' in d[name]:
            e = d[name]
            src = str(e.get('src', ''))
            measured = not ('default' in src)
            return (float(e['o']), float(e.get('r', 1.0)), float(e['window']), 'play_model.json: ' + src,
                    measured)
    if 'SlideOver' in name and os.path.exists(PLAY_SLIDE):
        d = json.load(open(PLAY_SLIDE, encoding='utf-8')).get('per_clip', {})
        if name in d and d[name].get('window') is not None:
            e = d[name]
            mu = e.get('map_used') or {}
            o = e.get('o') if e.get('o') is not None else mu.get('o', DEFAULT_PLAY_O)
            r = e.get('r') if e.get('r') is not None else mu.get('r', 1.0)
            return (float(o), float(r), float(e['window']), 'play_model_slideover.json: ' + str(mu.get('source', 'fit')),
                    True)
    return (DEFAULT_PLAY_O, 1.0, float(length) - DEFAULT_PLAY_O, 'unmeasured: clip length (optimistic)', False)


def analyse_acu(name, group='free'):
    handle, dump = _acu_dump_by_name(name)
    if dump is None:
        return None
    seq = acuclip.pose_sequence(dump)
    ev = acu_contacts.events(seq, dump)
    ob = ev['obstacle']
    length = float(dump['Length'])
    out = dict(name=name, group=group, handle=handle, length_s=length, height_m=ob['height'], depth_m=ob['depth'],
               takeoff_s=None, landing_s=None, flight_s=None, hand_plant=ev['hand_plant'] is not None)
    if ev['takeoff'] is None or ev['landing'] is None:
        out['status'] = 'no takeoff/landing'
        return out
    to, ld = float(ev['takeoff']['seconds']), float(ev['landing']['seconds'])
    frames = np.array([ev['takeoff']['frame'], ev['landing']['frame']])
    disp = acuclip.markers(dump, frames)['disp']
    o, r, window, src, measured = _play_for(name, length)
    cut = o + r * window
    runin = np.array([0, ev['takeoff']['frame']])
    runin_disp = acuclip.markers(dump, runin)['disp']
    out.update(takeoff_s=to, landing_s=ld, flight_s=ld - to, travel_m=float(disp[1, 1] - disp[0, 1]),
               runin_speed_mps=float((runin_disp[1, 1] - runin_disp[0, 1]) / max(to, 1e-6)),
               play_o=o, play_r=r, play_window_s=window, play_source=src, cut_measured=measured, cut_s=cut,
               remaining_after_landing_s=float(max(0.0, cut - ld)))
    if ev['hand_plant'] is not None:
        out['hand_plant_s'] = float(ev['hand_plant']['seconds'])
    out['status'] = 'ok'
    return out


# ------------------------------------------------------------------------------------------------ 5. scoring
def _timing_score(ratio):
    lo, hi = TIMING_IDEAL
    dist = max(0.0, lo - ratio, ratio - hi)
    return max(0.0, 1.0 - dist / 0.35)


def score_pair(wd, acu):
    """Score (0..100) of one WD2 clip against one ACU target, with the parts. None when a measure is missing."""
    if wd.get('flight_s') is None or acu.get('flight_s') is None:
        return None
    parts = {}
    ratio = wd['flight_s'] / acu['flight_s']
    parts['timing'] = _timing_score(ratio)
    inv = wd.get('inversion')
    parts['upright'] = {'none': 1.0, 'mild': 0.5, 'severe': 0.0}.get(inv, 0.0)
    rec = wd.get('recovery_s')
    rem = acu.get('remaining_after_landing_s', 0.0)
    if rec is None:
        parts['recovery'] = 0.0
    elif not acu.get('cut_measured', False):
        # the cut is not measured: only a landing that is upright at once counts in full
        parts['recovery'] = 1.0 if rec <= 0.05 else 0.5
    elif rec <= rem:
        parts['recovery'] = 1.0
    elif rec <= rem + 0.25:
        parts['recovery'] = 0.5
    else:
        parts['recovery'] = 0.0
    # both sides by hand contact in the flight (ACU: contacts on the obstacle top; WD2: candidate hand run in the flight)
    wd_plant = wd.get('events', {}).get('hand_plant_start') is not None and wd.get('flight_s') is not None
    parts['hands'] = 1.0 if bool(wd_plant) == bool(acu.get('hand_plant')) else 0.0
    # the WD2 obstacle top is not identified by the path (see METHOD_NOTES): only the clip-name hint is used
    cm = name_cm(wd['stem'])
    dh = abs(cm / 100.0 - acu['height_m']) if cm is not None else None
    height_src = 'name hint (unverified)' if dh is not None else 'unknown'
    if dh is None:
        parts['height'] = 0.5
    else:
        parts['height'] = 1.0 if dh <= 0.15 else (0.5 if dh <= 0.30 else 0.0)
    trav = wd.get('path', {}).get('travel_m') if wd.get('path', {}).get('ok') else None
    if trav is None or acu.get('travel_m') in (None, 0):
        parts['distance'] = 0.5
        dist_ratio = None
    else:
        dist_ratio = trav / acu['travel_m']
        if 0.8 <= dist_ratio <= 1.25:
            parts['distance'] = 1.0
        else:
            d = max(0.0, 0.8 - dist_ratio, dist_ratio - 1.25)
            parts['distance'] = max(0.0, 1.0 - d / 0.5)
    total = sum(SCORE_WEIGHTS[k] * parts[k] for k in SCORE_WEIGHTS)
    if parts['timing'] < 0.3:
        verdict = 'poor fit'                       # flight length far from the ACU flight: no matter the rest
    elif total >= VERDICT_STRONG and inv != 'severe' and parts['recovery'] >= 1.0:
        verdict = 'strong fit'
    elif total >= VERDICT_POSSIBLE:
        verdict = 'possible'
    else:
        verdict = 'poor fit'
    return dict(score=round(total, 1), parts={k: round(v, 3) for k, v in parts.items()}, verdict=verdict,
                timing_ratio=round(ratio, 3), distance_ratio=None if dist_ratio is None else round(dist_ratio, 3),
                height_src=height_src, recovery_used_s=rec, remaining_s=round(rem, 3),
                cut_measured=bool(acu.get('cut_measured', False)))


# ------------------------------------------------------------------------------------------------ 6. sheets
def draw_sheet(path, res, arr, bones):
    """Side view: stick figures at 10 times around the flight, then tilt and hand/foot heights with the events."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tr, sol, ev = arr['tr'], arr.get('sol'), res['events']
    fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 11), gridspec_kw=dict(height_ratios=[2.4, 1.0, 1.0]))
    if sol is not None:
        _stick_panel(ax, sol, res, bones)
    ax.set_title('%s  (%s)' % (res['stem'], _panel_note(res)), fontsize=9)
    ax.set_xlabel('forward (m): WD2 world Y relative to the takeoff pelvis, or pelvis-relative with offsets')
    ax.set_ylabel('z (m)')
    ax.set_ylim(-0.1, 2.2)
    ax.grid(alpha=0.2)
    ax.set_aspect('equal', adjustable='datalim')

    t = tr['t']
    ax2.plot(t, arr['tilt'], 'k-', lw=1.1, label='trunk tilt (deg)')
    ax2.axhline(TILT_MILD_DEG, color='0.5', ls=':', lw=0.8)
    ax2.axhline(TILT_SEVERE_DEG, color='0.5', ls='--', lw=0.8)
    ax2.set_ylabel('tilt (deg)', fontsize=8)
    ax2.set_ylim(0, 180)
    ax2b = ax2.twinx()
    ax2b.plot(t, arr['angsp'], color='tab:purple', lw=0.8, alpha=0.6, label='trunk ang. speed (deg/s)')
    ax2b.set_ylabel('deg/s', fontsize=8, color='tab:purple')
    ax3.plot(t, np.minimum(tr['eff']['LeftHand']['z'], tr['eff']['RightHand']['z']), color='tab:red', lw=1.1,
             label='lowest hand, rel. pelvis (m)')
    ax3.plot(t, np.minimum(tr['eff']['LeftFoot']['z'], tr['eff']['RightFoot']['z']), color='tab:blue', lw=1.1,
             label='lowest foot, rel. pelvis (m)')
    ax3.axhline(-HAND_DEPTH_M, color='tab:red', ls=':', lw=0.8)
    ax3.axhline(0.0, color='0.5', lw=0.6)
    ax3.set_xlabel('WD2 time (s)')
    ax3.set_ylabel('z rel. pelvis (m)', fontsize=8)
    ax3.set_ylim(-1.3, 1.3)
    for key, col in (('takeoff', 'k'), ('hand_plant_start', 'tab:green'), ('hand_plant_end', 'tab:olive'),
                     ('landing', 'tab:red')):
        v = ev.get(key)
        if v is None:
            continue
        for a in (ax2, ax3):
            a.axvline(v, color=col, ls='--', lw=1.0)
    if res.get('recovery_s') is not None and ev.get('landing') is not None:
        for a in (ax2, ax3):
            a.axvline(ev['landing'] + res['recovery_s'], color='tab:brown', ls='-.', lw=1.0)
    for a in (ax2, ax3):
        a.grid(alpha=0.2)
        a.set_xlim(0, t[-1])
    ax2.legend(fontsize=7, loc='upper left')
    ax3.legend(fontsize=7, loc='upper left')
    fig.suptitle('takeoff black, hand plant green/olive, landing red, upright-running brown dash-dot; run-in speed '
                 '%.2f m/s (%s)' % (res['speed_in_mps'], res['speed_in_class']), fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _panel_note(res):
    if res.get('path', {}).get('ok'):
        return 'pelvis path from contacts, WD2 world; flight height not identified'
    return 'pelvis-relative drawings (path contacts not consistent)'


def _stick_panel(ax, sol, res, bones):
    import matplotlib.pyplot as plt
    t = sol['t']
    ev = res['events']
    par = [int(b['parent']) for b in bones]
    names = [b['name'] for b in bones]
    keep = [i for i, n in enumerate(names) if not any(w in n for w in FINGER_WORDS)]
    kset = set(keep)
    head = names.index('Head')
    ok = bool(res.get('path', {}).get('ok'))
    to, ld = ev.get('takeoff'), ev.get('landing')
    picks = np.linspace(to - 0.25, ld + 0.25, 10)
    k0 = int(np.argmin(np.abs(t - to)))
    cmap = plt.get_cmap('viridis')
    for j, tp in enumerate(picks):
        k = int(np.argmin(np.abs(t - tp)))
        if ok:
            W = sol['P'][k][None, :] + sol['rel'][k]
            xs = W[:, 1] - sol['P'][k0, 1]
        else:
            W = sol['rel'][k]
            xs = W[:, 1] + j * 1.6
        zs = W[:, 2]
        col = cmap(j / 9.0)
        for i in keep:
            p = par[i]
            if p >= 0 and p in kset:
                ax.plot([xs[i], xs[p]], [zs[i], zs[p]], '-', color=col, lw=1.4)
        ax.text(xs[head], zs[head] + 0.12, '%.2f' % tp, fontsize=6, color=col, ha='center')
    ax.axhline(0.0, color='0.4', lw=0.8)


# ------------------------------------------------------------------------------------------------ 7. outputs
def run_all(write=True, sheets=True):
    rig = retarget.Rig()
    acu_rows = {}
    for name, grp in ACU_TARGETS:
        a = analyse_acu(name, grp)
        if a is not None:
            acu_rows[name] = a
    wd_rows, arrays = {}, {}
    for stem in CANDIDATES + [KONG, FRONTDIVE]:
        if stem == FRONTDIVE:
            res, arr = analyse_roll_reference(stem, rig)
        else:
            res, arr = analyse_wd2(stem, rig)
        wd_rows[stem] = res
        arrays[stem] = arr
    for stem, res in wd_rows.items():
        scored = {}
        for name, a in acu_rows.items():
            sc = score_pair(res, a)
            if sc is not None:
                scored[name] = sc
        res['scores'] = scored
        # candidates are ranked on the free targets only (taken = kong bake, extra = not in the free list)
        free = {n: v for n, v in scored.items() if acu_rows[n]['group'] == 'free'}
        measured = {n: v for n, v in free.items() if v['cut_measured']}
        if measured or free:
            pool = measured or free
            best = max(pool.items(), key=lambda kv: kv[1]['score'])
            res['best_target'], res['best_score'], res['best_verdict'] = best[0], best[1]['score'], best[1]['verdict']
        else:
            res['best_target'], res['best_score'], res['best_verdict'] = None, None, 'not analysable'
        if scored:
            ub = max(scored.items(), key=lambda kv: kv[1]['score'])
            res['best_any_target'], res['best_any_score'] = ub[0], ub[1]['score']
        else:
            res['best_any_target'], res['best_any_score'] = None, None
    refs = {k: wd_rows[v] for k, v in REFERENCES.items()}
    ranking = sorted(CANDIDATES, key=lambda s: -(wd_rows[s]['best_score'] or 0.0))
    ranking_hi = [s for s in ranking if wd_rows[s].get('detection_confidence') == 'high']
    out = dict(acu_targets=acu_rows, wd2=wd_rows, ranking=ranking, ranking_high_confidence=ranking_hi,
               references={k: v['stem'] for k, v in refs.items()}, weights=SCORE_WEIGHTS,
               thresholds=dict(strong=70, possible=45), method_notes=METHOD_NOTES)
    if sheets:
        for stem in CANDIDATES + [KONG, FRONTDIVE]:
            draw_sheet(os.path.join(SHEETS_DIR, 'wd2_%s.png' % stem), wd_rows[stem], arrays[stem], rig.wd)
    if write:
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
        with open(SUMMARY_JSON, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=1, default=_json_default)
        with open(REPORT_MD, 'w', encoding='utf-8') as f:
            f.write(build_report(out))
    return out


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def _f(x, nd=2, unit=''):
    if x is None:
        return '-'
    return ('%.' + str(nd) + 'f') % x + unit


def _short(stem):
    return stem.replace('_male_000f_norm_nowep', '').replace('_mplyr_000f_norm_nowep', '')


def _target_short(name):
    return name.replace('xx_Parkour_Sprint_', '') if name else '-'


METHOD_NOTES = [
    'Flight = takeoff (end of the last foot contact before the first hand plant) to landing (start of the first foot '
    'contact after it). Without a hand plant it is the largest gap between foot contacts (>= 0.2 s). Contacts are the '
    'candidate runs of wd2events: a foot or hand moving backward within 35 % of the run-in body speed for >= 0.1 s.',
    'Event accuracy on the hand-annotated references: takeoff and landing within 0.03 s. The hand plant start is '
    'within 0.02 s for the kong but 0.25 s early for the frontdive (an arm sweep matches the speed band). The kong '
    'plant end is 0.15 s early: the body decelerates while the hands are down, so the band loses them.',
    'Pelvis path: one least-squares problem on a 60 Hz grid. Foot contacts before takeoff and after landing pin the '
    'pelvis height (ground) and xy; hand contacts in the flight have a free surface height; inside the flight the '
    'pelvis is ballistic (second difference (0, 0, -g) dt^2) on frames without contact; origin at takeoff. Candidate '
    'ground runs whose implied height is more than 6 cm off are dropped (up to 3 passes; listed as dropped runs).',
    'What the path identifies: the forward travel (pinned by the feet before takeoff and after landing), the standing '
    'hips height, and the hips height at the ground contacts. Between the contacts it is not identified (the path drifts; '
    'the kong path falls below the ground after 2 s), so no hips profile is drawn in the flight or in the run-out. It '
    'does NOT identify the flight height: the top surface height comes out at 0 m for the kong, so no measured obstacle '
    'height is used. A non-ballistic flight (a roll) cannot be '
    'detected by the residual (free hand anchors absorb it): the frontdive with and without its annotated roll window '
    'gives travel 1.88 m vs 1.80 m, so the travel uncertainty is about 0.1 m.',
    'Trunk tilt = angle of Hips->Head from +Z (ACU skeleton, pelvis-relative, 30 Hz). Inversion in the flight: none '
    '< 90 deg, mild 90-120 deg, severe > 120 deg. Trunk angular speed = max rate of change of that direction.',
    'Hand use: hands are on top when a hand is more than 0.3 m below the pelvis during the flight (reported). The '
    'hands score compares hand contacts in the flight on both sides (WD2 candidate run, ACU contacts on the top), since '
    'the WD2 top is not visible.',
    'Recovery: time from landing until the hips are at >= 85 % of their standing height for 0.25 s. Primary measure: '
    'hips = Foot height minus the lowest foot (pelvis-relative FK, no path needed). Fallback: the path hips.',
    'ACU cut = match-model play window (o + r * window, clip seconds) when measured; otherwise the clip length '
    '(optimistic, flagged). remaining = cut - ACU landing. The recovery test asks whether the body is upright-running '
    'before that cut (within 0.25 s counts half).',
    'Obstacle height class: the WD2 clip names carry a NNcm tag. It is used as an unverified hint only (the frontdive '
    'annotation puts its 100cm obstacle top at 1.0 m, which supports the reading); clips without a tag score neutral '
    '(0.5).',
    'Distance: WD2 travel (pelvis, takeoff->landing, path, WD2 +Y) vs ACU root displacement over the same flight. '
    'Ratio 0.8-1.25 scores full.',
    'Speed class: run-in body speed from the backward foot speed (>= 4.5 m/s sprint, >= 2.0 jog). This estimator reads '
    'about half the ACU sprint run-in speed: the ACU targets run 4-7 m/s (root displacement before takeoff, summary.json '
    'runin_speed_mps), the WD2 clips 1.0-3.4 m/s, so the speed classes are not comparable to ACU and the candidate '
    'contact band is biased on the slow-reading clips. Those clips are marked detection LOW; check their events by hand.',
    'Verdicts: strong fit = score >= 70, no severe inversion, upright by the cut (full recovery credit); possible = score >= 55; poor fit '
    'otherwise, and always poor when the timing score is below 0.3 (the flight is far from the ACU flight). The cut '
    'is the match-model play window; where it is not measured, a landing upright at once counts in full and anything '
    'later counts half. not analysable = no flight detected.',
]


def build_report(out):
    """Markdown report: ranked table, scores matrix, per-clip notes, top 3, method notes."""
    wd = out['wd2']
    refs = out['references']
    ref_stems = set(refs.values())
    lines = []
    lines.append('# WD2 vault clips as replacements for ACU vault clips')
    lines.append('')
    lines.append('Analysis only: no baking, nothing installed, no game run. Score 0-100 per (WD2 clip, ACU target); '
                 'weights %s. Verdict: strong fit >= 70 (no severe inversion, full recovery), possible >= 55, else '
                 'poor fit; not analysable when no flight is found. The two references calibrate the scale: kong 80 '
                 '(strong), frontdive 53 (poor). Best target = best score among targets whose play cut is measured '
                 '(unmeasured targets are scored and listed, but a clip is only ranked on measured cuts).' % json.dumps(SCORE_WEIGHTS))
    lines.append('')
    lines.append('## Ranked table (best ACU target per clip)')
    lines.append('')
    lines.append('| # | WD2 clip | verdict | detection | score | best ACU target | flight s | timing ratio | tilt max deg '
                 '| inversion | hands > 0.3 m below pelvis | recovery s | travel m | speed in / out |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    order = list(out['ranking']) + [refs['kong'], refs['frontdive']]
    for i, stem in enumerate(order, 1):
        r = wd[stem]
        tag = ' (reference)' if stem in ref_stems else ''
        bs = r.get('scores', {}).get(r.get('best_target') or '', {})
        trav = r.get('path', {}).get('travel_m') if r.get('path', {}).get('ok') else None
        hands = ('yes, %s m, %s-%s s' % (_f(r.get('hand_depth_max')), _f(r.get('hands_below_0p3_from')),
                                         _f(r.get('hands_below_0p3_to')))) if r.get('hands_on_top') else 'no'
        lines.append('| %s | %s%s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s / %s |' % (
            i if not tag else '-', _short(stem), tag, r.get('best_verdict', '-'),
            r.get('detection_confidence', 'none'), _f(r.get('best_score'), 1),
            _target_short(r.get('best_target')), _f(r.get('flight_s')), _f(bs.get('timing_ratio')),
            _f(r.get('tilt_max_flight'), 0), r.get('inversion', '-'), hands, _f(r.get('recovery_s')), _f(trav),
            r.get('speed_in_class', '-'), r.get('speed_out_class', '-')))
    lines.append('')
    lines.append('## Score of each clip against each ACU target')
    lines.append('')
    tnames = [n for n, _g in ACU_TARGETS if n in out['acu_targets']]
    lines.append('| WD2 clip | ' + ' | '.join(_target_short(n) for n in tnames) + ' |')
    lines.append('|---|' + '---|' * len(tnames))
    for stem in order:
        r = wd[stem]
        cells = ['-' if r.get('scores', {}).get(n) is None else _f(r['scores'][n]['score'], 0) for n in tnames]
        lines.append('| %s | %s |' % (_short(stem), ' | '.join(cells)))
    lines.append('')
    lines.append('ACU targets (flight, travel, hand plant, cut, remaining after landing, play source):')
    lines.append('')
    for n in tnames:
        a = out['acu_targets'][n]
        lines.append('- `%s` (%s): height %s m, depth %s m, flight %s s, travel %s m, hand plant %s, cut %s s, '
                     'remaining %s s, %s' % (
                         _target_short(n), a.get('group'), _f(a.get('height_m')), _f(a.get('depth_m')),
                         _f(a.get('flight_s')), _f(a.get('travel_m')), 'yes' if a.get('hand_plant') else 'no',
                         _f(a.get('cut_s')), _f(a.get('remaining_after_landing_s')), a.get('play_source', '-')))
    lines.append('')
    lines.append('## Per-clip notes')
    lines.append('')
    for i, stem in enumerate(order, 1):
        lines.append(_clip_paragraph(stem, wd[stem], i))
        lines.append('')
    lines.append('## Top 3 recommendations')
    lines.append('')
    for j, stem in enumerate(out['ranking'][:3], 1):
        r = wd[stem]
        tgt = r.get('best_target')
        lines.append('%d. **%s** -> ACU `%s` (score %s, %s; detection %s). %s' % (
            j, _short(stem), _target_short(tgt), _f(r.get('best_score'), 1), r.get('best_verdict'),
            r.get('detection_confidence', '-'), _reason(r, tgt)))
    hi = [s for s in out.get('ranking_high_confidence', []) if wd[s].get('best_score') is not None]
    lines.append('')
    if hi:
        lines.append('Firm candidates (high detection confidence): ' + '; '.join(
            '%s (%s, %s)' % (_short(s), _f(wd[s].get('best_score'), 1), wd[s].get('best_verdict')) for s in hi) + '.')
    else:
        lines.append('No candidate has high detection confidence, so no recommendation is firm.')
    low = [s for s in out['ranking'] if wd[s].get('detection_confidence') == 'low' and wd[s].get('best_score') is not None]
    if low:
        lines.append('')
        lines.append('Low detection confidence (scores shown, check the events by hand before use): ' +
                     '; '.join('%s (%s, %s)' % (_short(s), _f(wd[s].get('best_score'), 1), _target_short(wd[s].get('best_target')))
                               for s in low[:6]) + '.')
    lines.append('')
    lines.append('## Method notes and caveats')
    lines.append('')
    for note in METHOD_NOTES:
        lines.append('- ' + note)
    lines.append('')
    lines.append('## Files')
    lines.append('')
    lines.append('- report: `%s`' % REPORT_MD)
    lines.append('- summary: `%s`' % SUMMARY_JSON)
    lines.append('- sheets: `%s` (one per clip, wd2_<stem>.png)' % SHEETS_DIR)
    lines.append('- module: `%s`' % os.path.abspath(__file__))
    lines.append('- tests: `%s`' % os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests',
                                              'test_wd2_eval.py'))
    return '\n'.join(lines) + '\n'


def _reason(r, tgt):
    s = r.get('scores', {}).get(tgt or '', {})
    p = s.get('parts', {})
    bits = ['timing ratio %s' % _f(s.get('timing_ratio')), 'inversion %s' % r.get('inversion'),
            'recovery %s s vs %s s before the cut' % (_f(s.get('recovery_used_s')), _f(s.get('remaining_s'))),
            'hand plant %s' % ('match' if p.get('hands') == 1.0 else 'mismatch'),
            'height %s' % s.get('height_src', '-')]
    return '; '.join(bits) + '.'


def _clip_paragraph(stem, r, i):
    ev = r.get('events', {})
    to, ld = ev.get('takeoff'), ev.get('landing')
    p = r.get('path', {})
    head = '### %d. %s' % (i, _short(stem))
    if to is None or ld is None:
        return head + ('\n\nNo flight detected (run-in %s m/s, %s). Not analysable with this detector.'
                       % (_f(r.get('speed_in_mps')), r.get('speed_in_class')))
    hp = ('%s-%s s' % (_f(ev.get('hand_plant_start')), _f(ev.get('hand_plant_end')))
          if ev.get('hand_plant_start') is not None else 'none')
    if r.get('hands_on_top'):
        hands = ('on top: %s m below the pelvis at its deepest, below 0.3 m from %s to %s s'
                 % (_f(r.get('hand_depth_max')), _f(r.get('hands_below_0p3_from')), _f(r.get('hands_below_0p3_to'))))
    else:
        hands = 'never 0.3 m below the pelvis in the flight (deepest %s m)' % _f(r.get('hand_depth_max'))
    txt = ('Duration %s s. Takeoff %s s, hand plant %s, landing %s s: flight %s s (rule: %s). Run-in %s m/s (%s), '
           'run-out %s m/s (%s). Trunk tilt max %s deg in the flight (%s; head below hips for %s s), trunk angular '
           'speed max %s deg/s (95th pct %s). Hands %s. Recovery to upright-running %s s after landing (source: %s). '
           'Path travel %s m (%s), standing hips %s m, hips at landing %s m.' % (
               _f(r.get('duration_s')), _f(to), hp, _f(ld), _f(r.get('flight_s')), ev.get('rule'),
               _f(r.get('speed_in_mps')), r.get('speed_in_class'), _f(r.get('speed_out_mps')),
               r.get('speed_out_class'), _f(r.get('tilt_max_flight'), 0), r.get('inversion'),
               _f(r.get('head_below_hips_s')), _f(r.get('ang_speed_max'), 0), _f(r.get('ang_speed_p95'), 0), hands,
               _f(r.get('recovery_s')), r.get('recovery_source', '-'), _f(p.get('travel_m')),
               'consistent' if p.get('ok') else 'contacts inconsistent', _f(p.get('standing_hips_m')),
               _f(p.get('hips_z_at_landing_m'))))
    if r.get('confidence_notes'):
        txt += ' Detection confidence LOW: %s.' % '; '.join(r['confidence_notes'])
    if p.get('dropped'):
        txt += ' Dropped contact runs: %s.' % '; '.join('%s %s-%s s (%s m)' % (d['name'], d['t0'], d['t1'], d['dev_m'])
                                                      for d in p['dropped'])
    sc = {n: v for n, v in r.get('scores', {}).items() if ACU_GROUP.get(n) == 'free'}
    if sc:
        top = sorted(sc.items(), key=lambda kv: -kv[1]['score'])[:3]
        txt += ' Best free ACU targets: ' + '; '.join('%s %s (%s, cut %s)' % (
            _target_short(n), v['score'], v['verdict'], 'measured' if v.get('cut_measured') else 'not measured')
            for n, v in top) + '.'
    return head + '\n\n' + txt


if __name__ == '__main__':
    result = run_all(write=True, sheets=True)
    print('wrote', REPORT_MD, SUMMARY_JSON, SHEETS_DIR, sep='\n')
    for stem in result['ranking'] + [result['references']['kong'], result['references']['frontdive']]:
        r = result['wd2'][stem]
        print('%-62s %-12s %6s %s' % (_short(stem)[:62], r.get('best_verdict'), r.get('best_score'),
                                      r.get('best_target')))
    sys.exit(0)
