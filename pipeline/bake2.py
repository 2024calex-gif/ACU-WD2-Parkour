"""Bake the WD2 catpass (kong) motion onto the ten ACU SkipOver clips (task T9b, PLAN_v2 section 5).

Per target clip:
  1. time_map          clip time -> WD2 time through the event anchors (takeoff, hand plant, landing)
  2. wd2_pose_at       retargeted ACU locals and world rotations of the WD2 clip at a WD2 time
  3. compensated_rotations  local rotations of the patched bones such that, with the game's own Reference rotation
                       (kept untouched) and the un-patched bones, the world orientation equals the retargeted one
  4. bake_clip         30 Hz keys over the whole clip, blended back to the original pose outside the play window
  5. reduce_keys, write_patch, mapping.json (only these ten entries)
  6. validate_clip     composed pose in world space: penetration, hand height, angular steps; PNG + report.md

Rotation tracks only. Reference position (translation) is not patched here (patch_translation false).
Output: config.PATCHES (patch files + mapping.json), config.OUT\\bake2 (PNGs, report.md).
Run:  python bake2.py          (writes everything; needs the WD2 .mab data and the ACU dumps)
"""
import dataclasses
import json
import math
import os
import re
import sys
import zlib

import numpy as np

import acuclip
import blend
import config
import contacts
import dumps
import fk
import ik
import retarget as R
import rootsolve
import wd2

EVENTS_JSON = os.path.join(config.V2, 'data', 'wd2_events.json')
PLAY_MODEL = os.path.join(config.OUT, 'match', 'play_model.json')
MEASURED = os.path.join(config.ROOT, 'tools', 'measured_play.json')
BAKE_OUT = os.path.join(config.OUT, 'bake2')
PATCH_OUT = config.PATCHES
OBSERVATIONS = os.path.join(BAKE_OUT, 'observations.md')   # optional hand-written notes, appended to report.md

TARGET_RX = r'^xx_Parkour_Sprint_SkipOver_'
SOURCE_EVENT = 'catpass_kong'
STEP_FRAMES = 2            # 30 Hz keys, like the game's own clips
FPS = 60
FADE_IN = 0.10             # s, original -> patch ramp after o
FADE_OUT = 0.15            # s, patch -> original ramp before o + window
DEFAULT_O = 0.0            # s, used when no play model exists
PLAY_FRACTION = 0.38       # window fallback = PLAY_FRACTION * clip length
REDUCE_TOL_DEG = 0.15
PEN_FAIL_CM = 5.0
HAND_BONES = ('LeftHand', 'RightHand')


# ------------------------------------------------------------------------------------------------ small helpers
def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def angle_deg(a, b):
    """Angle in degrees between quaternions (x, y, z, w), sign-insensitive; broadcasts over leading dims."""
    d = np.abs(np.sum(np.asarray(a, float) * np.asarray(b, float), axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(d, 0.0, 1.0)))


def _continuous(quats):
    """Flip signs so consecutive quaternions of one bone lie in the same hemisphere (same rotations)."""
    out = []
    prev = None
    for q in quats:
        q = np.asarray(q, float)
        if prev is not None and float(prev @ q) < 0:
            q = -q
        out.append(q)
        prev = q
    return out


def sample_keys(keys, frames):
    """Rotations at frames [N] from (frame, quat) keys: slerp between keys, clamped at both ends (like the plugin)."""
    kf = np.array([k[0] for k in keys], float)
    kq = np.array([k[1] for k in keys], float)
    fr = np.asarray(frames, float)
    if len(kf) == 1:
        return np.repeat(kq[:1], len(fr), axis=0)
    idx = np.clip(np.searchsorted(kf, fr, side='right') - 1, 0, len(kf) - 2)
    span = kf[idx + 1] - kf[idx]
    u = np.clip((fr - kf[idx]) / span, 0.0, 1.0)
    return fk.slerp(kq[idx], kq[idx + 1], u)


def load_events():
    with open(EVENTS_JSON, encoding='utf-8') as f:
        return json.load(f)[SOURCE_EVENT]


def acu_seconds(ev):
    """{'takeoff', 'hand_plant', 'landing'} in seconds from contacts.events() output (hand_plant may be None)."""
    def sec(k):
        return None if ev.get(k) is None else float(ev[k]['seconds'])
    return {'takeoff': sec('takeoff'), 'hand_plant': sec('hand_plant'), 'landing': sec('landing')}


# ------------------------------------------------------------------------------------------------ 1. time map
def acu_hand_plant(acu_ev, wd2_ev):
    """ACU hand plant in seconds: measured if present, else placed by the WD2 hand-plant fraction of takeoff..landing."""
    if acu_ev.get('hand_plant') is not None:
        return float(acu_ev['hand_plant'])
    to, ld = float(acu_ev['takeoff']), float(acu_ev['landing'])
    frac = (wd2_ev['hand_plant_start'] - wd2_ev['takeoff']) / (wd2_ev['landing'] - wd2_ev['takeoff'])
    return to + frac * (ld - to)


def time_map(acu_ev, wd2_ev, clip_len):
    """Function t_clip_seconds -> t_wd2_seconds (array in, array out).

    Piecewise linear through the anchors (ACU takeoff <-> WD2 takeoff, ACU hand plant <-> WD2 hand_plant_start,
    ACU landing <-> WD2 landing). Outside the anchors the slope is 1. The result is clamped to
    [wd2_ev['static_until'], wd2_ev['duration']].
    """
    if acu_ev.get('takeoff') is None or acu_ev.get('landing') is None:
        raise ValueError('ACU takeoff and landing are required for the time map')
    a_t = np.array([float(acu_ev['takeoff']), acu_hand_plant(acu_ev, wd2_ev), float(acu_ev['landing'])])
    w_t = np.array([float(wd2_ev['takeoff']), float(wd2_ev['hand_plant_start']), float(wd2_ev['landing'])])
    if a_t[0] < 0 or a_t[-1] > float(clip_len):
        raise ValueError('ACU anchors %s lie outside the clip (0..%.3f s)' % (a_t, clip_len))
    if np.any(np.diff(a_t) <= 0) or np.any(np.diff(w_t) <= 0):
        raise ValueError('anchors must be strictly increasing: ACU %s, WD2 %s' % (a_t, w_t))
    lo = float(wd2_ev['static_until'])
    hi = float(wd2_ev['duration'])

    def f(t):
        t = np.asarray(t, float)
        y = np.interp(t, a_t, w_t)
        y = np.where(t < a_t[0], w_t[0] + (t - a_t[0]), y)
        y = np.where(t > a_t[-1], w_t[-1] + (t - a_t[-1]), y)
        return np.clip(y, lo, hi)
    return f


def clip_interval_for_wd2(tm, a, b, clip_len):
    """(start, end) ACU clip seconds whose mapped WD2 time lies in [a, b], or None when no clip time maps there."""
    tt = np.linspace(0.0, float(clip_len), int(round(float(clip_len) * FPS)) * 4 + 1)
    ww = tm(tt)
    sel = tt[(ww >= a - 1e-9) & (ww <= b + 1e-9)]
    return (float(sel.min()), float(sel.max())) if sel.size else None


# ------------------------------------------------------------------------------------------------ 2. WD2 pose
def wd2_pose_at(rig, m, t_wd2):
    """(retargeted ACU locals [B,4], retargeted ACU world rotations [B,4]) of the WD2 clip m at WD2 time t_wd2 (s)."""
    t = min(max(float(t_wd2), 0.0), float(m['dur']))
    return rig.target_locals(rig.source_locals(m, t * 30.0))


# ------------------------------------------------------------------------------------------------ 3. compensation
def rot_tracks(rig, dump):
    """{ac bone index: rotation keys} for every ACU bone that has a rotation track in the dump."""
    out = {}
    for ai, b in enumerate(rig.ac):
        k = dumps.track(dump, b['hash'], 0)
        if k:
            out[ai] = k
    return out


def orig_local(rig, rot, ai, frame):
    """The dump's own local rotation of bone ai at a clip frame (bind pose when the bone has no track)."""
    k = rot.get(ai)
    return dumps.sample_rotation(k, frame) if k else R.qn(rig.ac[ai]['lrot'])


def patchable_bones(rig, rot):
    """Mapped ACU bones that have a rotation track in the dump. Hips (no track in the game's clips) is excluded."""
    hips = rig.ac_by_hash[zlib.crc32(b'Hips')]
    return sorted(ai for ai in set(rig.map.values()) if ai != hips and ai != 0 and ai in rot)


def compensated_rotations(rig, rot, frame, wt, patched):
    """Local rotations {ai: quat} for the patched bones at one clip frame.

    Walks the skeleton from the root. The game's Reference rotation (bone 0) is kept as it is in the dump. A patched
    bone's local is computed against its parent's ACTUAL orientation, so that its world orientation equals the
    retargeted one:
      * a direct child of Reference: parent = dump Reference rotation (rref)
      * a child of Hips: parent = rref * Hips bind local (Hips has no track in the game's clip)
      * other parents: their actual world orientation (patched parents equal their retargeted world)
    Desired world orientation in the game's animation space is R.A_REF * Wt, exactly as in tools/bake.py
    (`desired = R.qn(R.qmul(R.A_REF, Wt[ai]))`): A_REF converts the skeleton space of the retargeting into the
    animation space the game's clips use (the Reference carries that base turn; measured in v1).
    Un-patched bones keep their dump local (or bind local) and only contribute their actual orientation.
    """
    W = [None] * len(rig.ac)
    out = {}
    for ai, b in enumerate(rig.ac):
        par = b['parent']
        if ai == 0:
            W[0] = orig_local(rig, rot, 0, frame)
            continue
        if ai in patched:
            desired = R.qn(R.qmul(R.A_REF, wt[ai]))
            loc = R.qn(R.qmul(R.qconj(W[par]), desired))
            out[ai] = loc
            W[ai] = desired
        else:
            loc = orig_local(rig, rot, ai, frame)
            W[ai] = R.qn(R.qmul(W[par], loc))
    return out


# ------------------------------------------------------------------------------------------------ 4. bake
def bake_clip(rig, m, dump, acu_s, wd2_ev, o, window, step=STEP_FRAMES):
    """30 Hz keys for every patchable bone, crossfaded to the original outside [o, o + window].

    Returns dict: frames (60 Hz frame numbers), patched (ac indices), raw (patch keys before the crossfade),
    orig (dump keys at the same frames), blended (after the crossfade), time_map.
    """
    L = float(dump['Length'])
    nframes = int(round(L * FPS))
    frames = list(range(0, nframes + 1, step))
    if frames[-1] != nframes:
        frames.append(nframes)
    tm = time_map(acu_s, wd2_ev, L)
    rot = rot_tracks(rig, dump)
    pat = patchable_bones(rig, rot)
    patset = set(pat)
    raw = {ai: [] for ai in pat}
    orig = {ai: [] for ai in pat}
    t_wd2 = tm(np.asarray(frames, float) / FPS)
    for f, tw in zip(frames, t_wd2):
        _, wt = wd2_pose_at(rig, m, float(tw))
        comp = compensated_rotations(rig, rot, f, wt, patset)
        for ai in pat:
            raw[ai].append((f, comp[ai]))
            orig[ai].append((f, orig_local(rig, rot, ai, f)))
    blended = crossfade_to_original(orig, raw, frames, o, window)
    return dict(frames=frames, patched=pat, raw=raw, orig=orig, blended=blended, time_map=tm, nframes=nframes)


def fade_weight(t, o, window, fi=None, fo=None):
    """Weight of the patch at clip time t (s): 0 before o, smoothstep up over [o, o+fi], 1 in between,
    smoothstep down over [o+window-fo, o+window], 0 after o+window. fi / fo default to FADE_IN / FADE_OUT."""
    fi = FADE_IN if fi is None else fi
    fo = FADE_OUT if fo is None else fo
    t = np.asarray(t, float)
    w_in = np.clip((t - o) / fi, 0.0, 1.0)
    w_out = np.clip((o + window - t) / fo, 0.0, 1.0)
    return np.minimum(_smoothstep(w_in), _smoothstep(w_out))


TAIL_ORIG = 'orig'     # after the window: back to the ACU original pose (fade-out, default)
TAIL_WD2 = 'wd2'       # after the window: keep following the time-mapped WD2 motion to the clip end (no fade-out)
TAIL_HOLD = 'hold'     # after the window: hold the pose of the window end (o + window) to the clip end
TAILS = (TAIL_ORIG, TAIL_WD2, TAIL_HOLD)


def patch_weight(t, o, window, fi, fo, tail=TAIL_ORIG):
    """Patch weight at clip time t (s) for the T9e options. fi / fo are the fade lengths (s); fi = 0 is a step at o,
    fo = 0 a step at o + window (tail orig only). For tail orig with fi, fo > 0 this equals fade_weight exactly.
    tail wd2 / hold: no fade-out, the weight stays 1 after the fade-in (the tail decides what is played after the window).
    """
    t = np.asarray(t, float)
    if fi > 0:
        w_in = _smoothstep(np.clip((t - o) / fi, 0.0, 1.0))
    else:
        w_in = (t >= o - 1e-9).astype(float)
    if tail != TAIL_ORIG:
        w_out = np.ones_like(t)
    elif fo > 0:
        w_out = _smoothstep(np.clip((o + window - t) / fo, 0.0, 1.0))
    else:
        w_out = (t <= o + window + 1e-9).astype(float)
    return np.minimum(w_in, w_out)


def crossfade_to_original(orig, patch, frames, o, window):
    """Per bone: slerp(original, patch, w(t)) at each key frame, with w from fade_weight. Sign-continuous output.

    orig, patch: {ai: [(frame, quat), ...]} with identical frame lists. Where w == 0 the output is the original
    quaternion itself (exactly), so entry and exit match the game's own clip.
    """
    if window <= 0:
        raise ValueError('window must be positive')
    t = np.asarray(frames, float) / FPS
    w = fade_weight(t, o, window)
    out = {}
    for ai in patch:
        res = []
        for (f1, qp), (f2, qo), wi in zip(patch[ai], orig[ai], w):
            if f1 != f2:
                raise ValueError('frame lists differ')
            qo = R.qn(qo)
            if wi <= 0.0:
                q = qo
            elif wi >= 1.0:
                q = R.qn(qp)
            else:
                q = R.slerp(qo, qp, float(wi))
            res.append((f1, q))
        out[ai] = [(f, q) for (f, _), q in zip(res, _continuous([q for _, q in res]))]
    return out


def reduce_keys(keys, tol_deg=REDUCE_TOL_DEG):
    """Drop keys so that the slerp between kept keys stays within tol_deg of every dropped key.

    Same idea as tools/bake.py's reduce_keys, but the check is against the final kept interpolation: a key is
    dropped only when the slerp from the last kept key to the next candidate reproduces every key it skips.
    (bake.py checked each key only against its successor, so errors could build up along a run.)
    """
    fixed = [keys[0]]
    for t, q in keys[1:]:
        pq = fixed[-1][1]
        fixed.append((t, q if float(pq @ q) >= 0 else -q))
    n = len(fixed)
    if n <= 2:
        return fixed
    keep = [0]
    last = 0
    i = 2
    while i < n:
        t0, q0 = fixed[last]
        t1, q1 = fixed[i]
        ok = True
        for m in range(last + 1, i):
            t, q = fixed[m]
            pred = R.slerp(q0, q1, (t - t0) / max(1, t1 - t0))
            if math.degrees(2 * math.acos(min(1.0, abs(float(pred @ q))))) > tol_deg:
                ok = False
                break
        if ok:
            i += 1
        else:
            last = i - 1
            keep.append(last)
            i = last + 2
    keep.append(n - 1)
    return [fixed[k] for k in sorted(set(keep))]


# ------------------------------------------------------------------------------------------------ 5. write
def patch_dict(rig, tracks, length):
    """Patch file content: rotation tracks only (codec 1, infotype 0), Frame in 60 Hz clip frames."""
    tr = []
    for ai in sorted(tracks, key=lambda a: rig.ac[a]['hash']):
        kf = [{'Frame': int(f), 'LocalRotation': [round(float(x), 6) for x in q]} for f, q in tracks[ai]]
        tr.append({'TrackID': str(rig.ac[ai]['hash']), 'codec': 1, 'infotype': 0, 'Keyframes': kf})
    return {'Length': round(float(length), 6), 'Tracks': tr}


def write_patch(path, rig, tracks, length):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(patch_dict(rig, tracks, length), f)


def mapping_entry(handle, name, source):
    return {'handle': '0x' + handle, 'name': name, 'source': source, 'enabled': True, 'prebaked': True,
            'patch_translation': False}


# ------------------------------------------------------------------------------------------------ play model
def load_play_model(path=PLAY_MODEL):
    """Per-clip {'o', 'window'} from out\\match\\play_model.json (written by another task), or {} when missing."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    clips = d.get('per_clip', d.get('clips', d)) if isinstance(d, dict) else {}
    out = {k: v for k, v in clips.items() if isinstance(v, dict) and 'o' in v and 'window' in v}
    for k, v in out.items():
        if abs(float(v.get('r', 1.0)) - 1.0) > 1e-9:
            raise ValueError('play model rate r=%s for %s: the bake assumes r = 1' % (v.get('r'), k))
    return out


def load_measured(path=MEASURED):
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def play_for(name, length, model, measured):
    """(o, window, source) for a clip: play model if it has the clip, else o = 0 and the measured window, else 0.38 * len."""
    if name in model:
        return float(model[name]['o']), float(model[name]['window']), 'play_model'
    if name in measured:
        return DEFAULT_O, float(measured[name]), 'measured_play.json'
    return DEFAULT_O, PLAY_FRACTION * float(length), 'fallback 0.38*len'


# ------------------------------------------------------------------------------------------------ 6. validation
def penetration_parts(P, ob):
    """(box, ground): largest depth (m) of any point P [..., 3] inside the obstacle box, and below the ground.

    The box is y in (near_y, far_y), z in (0, height), unbounded in x. Depth inside the box is the distance
    to the nearest face; below ground it is -z.
    """
    z = P[..., 2]
    y = P[..., 1]
    below = np.maximum(0.0, -z)
    inside = (y > ob['near_y']) & (y < ob['far_y']) & (z > 0.0) & (z < ob['height'])
    depth = np.minimum(np.minimum(ob['height'] - z, y - ob['near_y']), ob['far_y'] - y)
    return float(np.where(inside, depth, 0.0).max()), float(below.max())


def penetration_m(P, ob):
    """Largest depth (m) of any point in the obstacle box or below the ground (see penetration_parts)."""
    return max(penetration_parts(P, ob))


def _step_deg(lr, idx, bones):
    """Largest angle (deg) between consecutive 60 Hz samples idx (only pairs exactly 2 frames apart)."""
    idx = np.asarray(idx, int)
    if len(idx) < 2:
        return 0.0
    keep = np.diff(idx) == 2
    if not np.any(keep):
        return 0.0
    a = lr[idx[:-1][keep]][:, bones]
    b = lr[idx[1:][keep]][:, bones]
    return float(np.max(angle_deg(a, b)))


def validate_clip(rig, dump, bake, o, window, hp_interval, ob):
    """Composed pose (dump Reference + patched locals) in world space, plus the report metrics.

    Returns (metrics dict, arrays dict for the plot).
    """
    seq = acuclip.pose_sequence(dump, step=1)           # 60 Hz, the dump's own locals
    frames = seq.frames.astype(float)
    t = frames / FPS
    tracks = bake['tracks']
    lr = seq.local_rot.copy()
    for ai in bake['patched']:
        lr[:, ai] = sample_keys(tracks[ai], frames)
    pseq = dataclasses.replace(seq, local_rot=lr)
    Pw = acuclip.world_positions(pseq, dump)
    Po = acuclip.world_positions(seq, dump)
    win = (t >= o - 1e-9) & (t <= o + window + 1e-9)
    body = sorted(set(rig.map.values()))
    box_p, gnd_p = penetration_parts(Pw[win][:, body], ob)
    pen_p = max(box_p, gnd_p) * 100.0
    pen_o = penetration_m(Po[win][:, body], ob) * 100.0
    box_o, gnd_o = penetration_parts(Po[win][:, body], ob)

    hands = [rig.ac_by_hash[zlib.crc32(n.encode())] for n in HAND_BONES]
    hand_stats = None
    if hp_interval is not None:
        sel = (t >= hp_interval[0] - 1e-9) & (t <= hp_interval[1] + 1e-9) & win
        if np.any(sel):
            hz = (Pw[sel][:, hands, 2] - ob['height']) * 100.0
            hand_stats = dict(min=float(hz.min()), median=float(np.median(hz)), max=float(hz.max()))

    even = np.where((np.asarray(frames, int) % 2) == 0)[0]
    in_win = even[win[even]]
    pat = bake['patched']
    step_p = _step_deg(lr, in_win, pat)
    step_o = _step_deg(seq.local_rot, in_win, pat)
    b1 = even[(t[even] >= o - FADE_IN) & (t[even] <= o + FADE_IN)]
    b2 = even[(t[even] >= o + window - FADE_OUT) & (t[even] <= o + window + FADE_IN)]
    bnd_p = max(_step_deg(lr, b1, pat), _step_deg(lr, b2, pat))
    bnd_o = max(_step_deg(seq.local_rot, b1, pat), _step_deg(seq.local_rot, b2, pat))

    # error of the key reduction: reduced keys (sampled like the plugin) against the blended 30 Hz keys
    red_err = 0.0
    for ai in pat:
        bl_f = [k[0] for k in bake['blended'][ai]]
        bl_q = np.array([k[1] for k in bake['blended'][ai]])
        red_err = max(red_err, float(np.max(angle_deg(sample_keys(tracks[ai], bl_f), bl_q))))

    metrics = dict(pen_patch_cm=pen_p, pen_orig_cm=pen_o, pen_box_cm=box_p * 100.0, pen_ground_cm=gnd_p * 100.0,
                   pen_orig_box_cm=box_o * 100.0, pen_orig_ground_cm=gnd_o * 100.0, hand=hand_stats, step_patch_deg=step_p,
                   step_orig_deg=step_o, boundary_patch_deg=bnd_p, boundary_orig_deg=bnd_o,
                   reduce_err_deg=red_err, n_patched=len(pat), n_keys=sum(len(v) for v in tracks.values()),
                   fail=pen_p > PEN_FAIL_CM)
    arrays = dict(frames=frames, Pw=Pw, Po=Po, win=win, body=body)
    return metrics, arrays


def plot_clip(path, name, rig, arrays, o, window, ob, metrics):
    """Side view (entity forward Y, height Z): obstacle box, ~12 stick figures across the play window."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    frames = arrays['frames']
    Pw, Po, body = arrays['Pw'], arrays['Po'], arrays['body']
    bset = set(body)
    edges = [(ai, rig.ac[ai]['parent']) for ai in body if rig.ac[ai]['parent'] in bset]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=110)
    ax.add_patch(Rectangle((ob['near_y'], 0.0), ob['far_y'] - ob['near_y'], ob['height'],
                           facecolor='#8b6f47', alpha=0.35, edgecolor='#5c4a2e', label='obstacle'))
    ax.axhline(0.0, color='#777', lw=0.8)
    sel_t = np.linspace(o, o + window, 12)
    idxs = [int(np.argmin(np.abs(frames / FPS - s))) for s in sel_t]
    colours = plt.cm.viridis(np.linspace(0.05, 0.95, len(idxs)))
    for k, fi in enumerate(idxs):
        for a, p in edges:
            ax.plot([Po[fi, a, 1], Po[fi, p, 1]], [Po[fi, a, 2], Po[fi, p, 2]], color='#bdbdbd', lw=1.2, zorder=1)
        for a, p in edges:
            ax.plot([Pw[fi, a, 1], Pw[fi, p, 1]], [Pw[fi, a, 2], Pw[fi, p, 2]], color=colours[k], lw=2.0, zorder=2)
    for ai in (rig.ac_by_hash[zlib.crc32(n.encode())] for n in HAND_BONES):
        ax.plot(Pw[idxs, ai, 1], Pw[idxs, ai, 2], 'o', color='#d62728', ms=3, zorder=3)
    ys = Pw[idxs][:, body, 1]
    ax.set_xlim(float(ys.min()) - 0.6, float(ys.max()) + 0.6)
    ax.set_ylim(-0.1, max(ob['height'] + 1.0, float(Pw[idxs][:, body, 2].max()) + 0.2))
    ax.set_aspect('equal')
    ax.set_xlabel('entity forward Y (m)')
    ax.set_ylabel('Z (m)')
    hs = metrics['hand']
    hand_txt = 'n/a' if hs is None else '%+.0f / %+.0f / %+.0f cm' % (hs['min'], hs['median'], hs['max'])
    status = 'FAIL' if metrics['fail'] else 'PASS'
    ax.set_title('%s  [%s]\ngrey = original, colour = patch (%d figures, o=%.2f s, window=%.2f s)\n'
                 'penetration patch %.1f cm (orig %.1f cm); hand above top min/med/max %s'
                 % (name, status, len(idxs), o, window, metrics['pen_patch_cm'], metrics['pen_orig_cm'], hand_txt),
                 fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(path, rows, play_model_path, notes_path=OBSERVATIONS):
    lines = ['# T9b bake2: WD2 catpass (kong) onto ACU SkipOver clips', '',
             'Generated by `v2/pipeline/bake2.py`. Times in seconds of the clip; heights in cm; angles in degrees.',
             '', 'Play model used: ' + (play_model_path if play_model_path and os.path.exists(play_model_path)
                                       else 'none found (defaults: o = 0, window from tools/measured_play.json, '
                                            'else 0.38 * length)'), '',
             '## Results', '',
             '| clip | len | o / window (source) | ACU takeoff / hand plant / landing | pen patch box / ground (cm) | pen orig box / ground (cm) | '
             'hand above top min / med / max | max step patch / orig | boundary step patch / orig | reduce err | keys | status |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        hs = r['metrics']['hand']
        hand = 'n/a (plant outside window)' if hs is None else '%+.1f / %+.1f / %+.1f' % (hs['min'], hs['median'], hs['max'])
        anc = r['anchors']
        mt = r['metrics']
        lines.append('| %s | %.2f | %.2f / %.2f (%s) | %s / %s%s / %.2f | %.1f / %.1f | %.1f / %.1f | %s | %.1f / %.1f | %.1f / %.1f | '
                     '%.2f | %d | %s |'
                     % (r['name'], r['length'], r['o'], r['window'], r['source'],
                        '%.3f' % anc['takeoff'], '%.3f' % anc['hand_plant'],
                        '' if anc['hp_measured'] else ' (placed)', anc['landing'],
                        mt['pen_box_cm'], mt['pen_ground_cm'], mt['pen_orig_box_cm'], mt['pen_orig_ground_cm'], hand,
                        r['metrics']['step_patch_deg'], r['metrics']['step_orig_deg'],
                        r['metrics']['boundary_patch_deg'], r['metrics']['boundary_orig_deg'],
                        r['metrics']['reduce_err_deg'], r['metrics']['n_keys'],
                        'FAIL' if r['metrics']['fail'] else 'PASS'))
    lines += ['', '## WD2 anchors (wd2_events.json, catpass_kong)', '',
              'takeoff %.3f, hand plant %.3f to %.3f, landing %.3f, static until %.3f, WD2 duration %.3f s.'
              % (rows[0]['wd2']['takeoff'], rows[0]['wd2']['hand_plant_start'], rows[0]['wd2']['hand_plant_end'],
                 rows[0]['wd2']['landing'], rows[0]['wd2']['static_until'], rows[0]['wd2']['duration']), '',
              'Pen = deepest body joint inside the obstacle box (box) or below ground (ground), over the play window '
              '[o, o + window]. '
              'FAIL when pen patch > %.0f cm. Hand = Hand joint height minus obstacle top over the mapped hand plant '
              'inside the window. Max step = largest angle between consecutive 30 Hz samples of the patched bones in '
              'the window; boundary step = same at the two crossfades (+-0.10 s around o and o + window). '
              'Reduce err = max deviation of the reduced keys from the raw 30 Hz keys.' % PEN_FAIL_CM, '',
              '## Clips', '']
    for r in rows:
        lines.append('- %s: hand plant %s; window source %s; time map anchors ACU %s -> WD2 %s.'
                     % (r['name'], 'measured' if r['anchors']['hp_measured'] else 'placed (no measured hand plant)',
                        r['source'], '(%.3f, %.3f, %.3f)' % (r['anchors']['takeoff'], r['anchors']['hand_plant'],
                                                              r['anchors']['landing']),
                        '(%.3f, %.3f, %.3f)' % (r['wd2']['takeoff'], r['wd2']['hand_plant_start'],
                                                r['wd2']['landing'])))
    lines += ['', '## Observations', '']
    if os.path.exists(notes_path):
        with open(notes_path, encoding='utf-8') as f:
            lines.append(f.read().rstrip())
    else:
        lines.append('(no observations file)')
    lines.append('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ------------------------------------------------------------------------------------------------ main
def targets():
    """[(handle_hex, clip_name, dump)] for the SkipOver clips, in dump-file order."""
    out = []
    for handle, name, _path in dumps.list_dumps():
        if re.match(TARGET_RX, name):
            out.append((handle, name, dumps.load_by_name(name)))
    return out


def vault_window(name, length, takeoff, landing):
    """Diagnostic window: from 0.05 s before take-off to 0.30 s after landing (not used for the patches)."""
    o = max(0.0, takeoff - 0.05)
    return o, min(float(length), landing + 0.30) - o, 'diagnostic: take-off..landing'


def bake_all(write=True, window_fn=None, diagnostic=False):
    os.makedirs(PATCH_OUT, exist_ok=True)
    os.makedirs(BAKE_OUT, exist_ok=True)
    rig = R.Rig()
    ev_json = load_events()
    m = wd2.load(ev_json['stem'])
    wd2_ev = dict(takeoff=ev_json['takeoff'], hand_plant_start=ev_json['hand_plant_start'],
                  hand_plant_end=ev_json['hand_plant_end'], landing=ev_json['landing'],
                  static_until=ev_json['static_until'], duration=float(m['dur']))
    model = load_play_model()
    measured = load_measured()
    rows = []
    mapping = []
    for handle, name, dump in targets():
        seq = acuclip.pose_sequence(dump, step=1)
        ev = contacts.events(seq, dump)
        acu_s = acu_seconds(ev)
        ob = ev['obstacle']
        L = float(dump['Length'])
        if window_fn is None:
            o, window, source = play_for(name, L, model, measured)
        else:
            o, window, source = window_fn(name, L, acu_s['takeoff'], acu_s['landing'])
        bake = bake_clip(rig, m, dump, acu_s, wd2_ev, o, window)
        tracks = {ai: reduce_keys(bake['blended'][ai]) for ai in bake['patched']}
        bake['tracks'] = tracks
        hp_interval = clip_interval_for_wd2(bake['time_map'], wd2_ev['hand_plant_start'],
                                            wd2_ev['hand_plant_end'], L)
        metrics, arrays = validate_clip(rig, dump, bake, o, window, hp_interval, ob)
        fn = '%s.baked.json' % handle
        if write:
            write_patch(os.path.join(PATCH_OUT, fn), rig, tracks, L)
            mapping.append(mapping_entry(handle, name, fn))
            plot_clip(os.path.join(BAKE_OUT, name + '.png'), name, rig, arrays, o, window, ob, metrics)
        anchors = dict(takeoff=acu_s['takeoff'], hand_plant=acu_hand_plant(acu_s, wd2_ev),
                       landing=acu_s['landing'], hp_measured=acu_s['hand_plant'] is not None)
        rows.append(dict(name=name, handle=handle, length=L, o=o, window=window, source=source,
                         metrics=metrics, anchors=anchors, wd2=wd2_ev))
        print('%-48s o %.2f win %.3f (%s)  pen %.1f/%.1f cm  %s' % (
            name, o, window, source, metrics['pen_patch_cm'], metrics['pen_orig_cm'],
            'FAIL' if metrics['fail'] else 'PASS'))
    if write:
        with open(os.path.join(PATCH_OUT, 'mapping.json'), 'w', encoding='utf-8') as f:
            json.dump({'clips': mapping}, f, indent=1)
        write_report(os.path.join(BAKE_OUT, 'report.md'), rows, PLAY_MODEL)
    if diagnostic:
        write_diagnostic(os.path.join(BAKE_OUT, 'diagnostic.md'), rows)
    return rows


def write_diagnostic(path, rows):
    lines = ['# T9b diagnostic: the same patch over take-off..landing', '',
             'Not the shipped patch. Same bake, but the crossfade window is 0.05 s before take-off to 0.30 s after '
             'landing, so the whole vault is inside the patch. Shows whether the motion itself penetrates the wall, '
             'independent of the play window.', '',
             '| clip | window (s) | pen box / ground (cm) | hand above top min / med / max (cm) | status |',
             '|---|---|---|---|---|']
    for r in rows:
        mt = r['metrics']
        hs = mt['hand']
        hand = 'n/a' if hs is None else '%+.1f / %+.1f / %+.1f' % (hs['min'], hs['median'], hs['max'])
        lines.append('| %s | %.2f to %.2f | %.1f / %.1f | %s | %s |'
                     % (r['name'], r['o'], r['o'] + r['window'], mt['pen_box_cm'], mt['pen_ground_cm'], hand,
                        'FAIL' if mt['fail'] else 'PASS'))
    lines.append('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ------------------------------------------------------------------------------------------------ 7. T9c
# Root offset (Reference position track) and hand IK on top of the T9b rotation bake (PLAN_v2 section 5, T9c).
# Pipeline per clip (60 Hz frames, all inside the play window [o, o + window] unless stated):
#   a. patch rotations from the T9b raw keys; crossfade them to the original (blend, as the game would see it)
#   b. root solve: rootsolve.solve_offset on the blended pose -> Delta (zero outside the window)
#   c. hand IK on the un-faded patch with Reference + Delta: fixed targets at the plant midpoint, then the window
#      crossfade is applied again (so the boundaries equal the original)
#   d. tracks: original keys outside the window kept exactly, new keys inside (2-frame grid + original keys),
#      reduced with 2 mm (translation) and 0.15 deg (rotation)
#   e. validate the composed pose that the tracks produce (sampled like the plugin does)

T9C_HEIGHT_WEIGHT = 1.0       # weight of the Height rows (hands and foot contacts), as specified
T9C_LAM_SMOOTH = 200.0        # second-difference penalty of the root solve (T9a: 50 was too jerky)
T9C_LAM_SMALL = (5.0, 0.1)    # (Y, Z) penalty on the offset itself
T9C_INEQ_WEIGHT = rootsolve.INEQ_WEIGHT  # weight of the Above (no penetration) equalities
T9C_PIN_FRAMES = 4            # Delta = 0 on the first and last frames of the window
T9C_IK_RAMP = 3               # frames (60 Hz) of IK weight ramp outside the plant interval
T9C_IK_MARGIN = 0.05          # m, hand target kept this far inside the near and far edge of the top
T9C_TRANS_TOL = 0.002         # m, Reference translation key reduction tolerance
T9C_HAND_TOL_CM = 5.0         # PASS: hands within this distance of their target during the plant
T9C_DELTA_STEP_CM = 3.0       # PASS: largest change of Delta between two 60 Hz frames
T9C_PEN_CM = 5.0              # PASS: wall box and ground penetration (window)
T9C_HANDS = ('LeftHand', 'RightHand')
T9C_PLANT_SOURCE = 'wd2'        # 'wd2' (spec) or 'acu' (diagnostic: the original clip's own hand-on-top frames)
T9C_SOLVE_EXCLUDE = ()          # bone names left out of the root solve only (diagnostic; spec run: none)
T9C_LEGS = (('LeftFoot', 'LeftFootThumb1'), ('RightFoot', 'RightFootThumb1'))
REF_TID = zlib.crc32(b'Reference')


def _bone_name(rig, ai):
    """Name of ACU bone ai (from the bone-name table; '?' when unknown)."""
    return dumps.bone_names().get(int(rig.ac[ai]['hash']), '?')


def _ptype(name):
    """LeftFoot -> Foot, RightFootThumb1 -> FootThumb1, LeftHand -> Hand."""
    return name.replace('Left', '', 1).replace('Right', '', 1)


def measure_contact_offsets(clips):
    """Median height (m) of the Hand, Foot and FootThumb1 joints above their support over all contact frames.

    clips: iterable of (name, dump, seq, ev) for the ORIGINAL vault clips (seq with step 1, ev = contacts.events).
    The support is the obstacle top for 'top' contacts and the ground (z = 0) for 'ground' contacts. Both sides
    of a point type are pooled. Returns {'Hand', 'Foot', 'FootThumb1': metres, 'n': {type: samples}}.
    """
    acc = {'Hand': [], 'Foot': [], 'FootThumb1': []}
    for _name, dump, seq, ev in clips:
        ob = ev['obstacle']
        W = acuclip.world_positions(seq, dump)
        for pt, runs in ev['contacts'].items():
            idx = seq.hash_index[zlib.crc32(pt.encode())]
            for s, e, kind in runs:
                surf = ob['height'] if kind == 'top' else 0.0
                acc[_ptype(pt)].extend((W[int(s):int(e) + 1, idx, 2] - surf).tolist())
    out = {k: float(np.median(np.asarray(v, float))) for k, v in acc.items()}
    out['n'] = {k: len(v) for k, v in acc.items()}
    return out


def window_frames(o, window, nframes):
    """60 Hz frame numbers (= indices, the sequence starts at frame 0) inside [o, o + window]."""
    t = np.arange(int(nframes) + 1) / FPS
    return np.nonzero((t >= o - 1e-9) & (t <= o + window + 1e-9))[0]


def plant_in_window(hp_iv, win):
    """Frames of `win` inside the WD2 hand-plant interval mapped to clip time (seconds), or an empty array."""
    win = np.asarray(win, int)
    if hp_iv is None or win.size == 0:
        return np.array([], int)
    t = win / FPS
    return win[(t >= hp_iv[0] - 1e-9) & (t <= hp_iv[1] + 1e-9)]


def _flip_continuous(q):
    """Negate quaternions whose dot with the previous one is negative (same rotations, continuous signs)."""
    q = np.array(q, float)
    for i in range(1, len(q)):
        if float(q[i - 1] @ q[i]) < 0:
            q[i] = -q[i]
    return q


def mix_rot(orig, patch, w):
    """slerp(orig, patch, w) per frame ([F, 4] arrays, w [F]); exactly orig where w == 0, patch where w == 1."""
    orig = np.asarray(orig, float)
    patch = np.asarray(patch, float)
    w = np.asarray(w, float)
    out = fk.slerp(orig, patch, w)
    out = np.where((w <= 0)[:, None], orig, out)
    out = np.where((w >= 1)[:, None], patch, out)
    return _flip_continuous(out)


def composed(seq, local_rot, ref_pos):
    """PoseSeq with the given local rotations [F, B, 4] and the Reference (bone 0) position replaced."""
    lp = np.array(seq.local_pos, float)
    lp[:, 0] = np.asarray(ref_pos, float)
    return dataclasses.replace(seq, local_rot=np.asarray(local_rot, float), local_pos=lp)


def patch_rotations60(raw, pat, frames):
    """{ai: [F, 4]} patch rotations at the given 60 Hz frames, sampled from the raw 30 Hz keys (slerp, sign-continuous)."""
    out = {}
    for ai in pat:
        keys = raw[ai]
        q = _flip_continuous([qq for _f, qq in keys])
        out[ai] = sample_keys([(f, qq) for (f, _), qq in zip(keys, q)], frames)
    return out


def _kind_at(runs, f):
    """'top' / 'ground' when frame f lies in one of the contact runs [s, e, kind], else None."""
    for s, e, kind in runs:
        if s <= f <= e:
            return kind
    return None


def root_constraints(P, win, plant_j, contacts_by_name, point_of, ob, offsets):
    """rootsolve constraints for one clip window.

    P: [Fw, K, 3] body points over the window (index j <-> frame win[j]); point_of: name -> column of P.
    plant_j: window positions of the hand plant. Hands: Height(top + Hand offset) on the plant (weight T9C_HEIGHT_WEIGHT).
    Feet: per leg, every frame in an ACU contact of Foot or FootThumb1 constrains only the lower of the two
    (Height at its surface + its measured offset). Above('all') against the obstacle box (and ground).
    Pin(0) on the first and last T9C_PIN_FRAMES window frames, minus the plant frames (shrunk, reported).
    Returns (constraints, info).
    """
    Fw = int(P.shape[0])
    box = rootsolve.Box(float(ob['height']), float(ob['near_y']), float(ob['far_y']))
    cons = []
    if len(plant_j):
        for h in T9C_HANDS:
            cons.append(rootsolve.Height(point=point_of[h], frames=np.asarray(plant_j, int),
                                         z=float(ob['height']) + offsets['Hand'], weight=T9C_HEIGHT_WEIGHT))
    feet_frames = {}
    for foot, toe in T9C_LEGS:
        groups = {}
        for j, f in enumerate(win):
            cands = []
            for name in (foot, toe):
                kind = _kind_at(contacts_by_name[name], int(f))
                if kind is not None:
                    cands.append((name, kind))
            if not cands:
                continue
            name, kind = min(cands, key=lambda c: float(P[j, point_of[c[0]], 2]))
            surf = float(ob['height']) if kind == 'top' else 0.0
            z = round(surf + offsets[_ptype(name)], 9)
            groups.setdefault((name, z), []).append(j)
        for (name, z), js in groups.items():
            cons.append(rootsolve.Height(point=point_of[name], frames=np.asarray(js, int), z=z,
                                         weight=T9C_HEIGHT_WEIGHT))
            feet_frames[name] = feet_frames.get(name, 0) + len(js)
    cons.append(rootsolve.Above(point='all', frames=np.arange(Fw), surface=box))
    plant_set = set(int(j) for j in plant_j)
    pins_all = sorted(set(range(min(T9C_PIN_FRAMES, Fw))) | set(range(max(0, Fw - T9C_PIN_FRAMES), Fw)))
    pins = [j for j in pins_all if j not in plant_set]
    if pins:
        cons.append(rootsolve.Pin(frames=np.asarray(pins, int), value=0.0, weight=rootsolve.WEIGHT))
    info = dict(pin_frames=len(pins), pin_frames_wanted=len(pins_all), foot_height_frames=feet_frames)
    return cons, info


def inside_key_frames(win, orig_frames):
    """Key frames inside the window: every 60 Hz frame of the window (the IK and the root solve are evaluated at
    60 Hz, so the tracks must not be coarser than that; reduce_keys / reduce_positions then drop what is redundant)."""
    lo, hi = int(win[0]), int(win[-1])
    s = set(range(lo, hi + 1)) | {lo, hi}
    s |= {int(f) for f in orig_frames if lo <= int(f) <= hi}
    return sorted(s)


def reduce_positions(keys, tol):
    """Greedy key reduction for (frame, [3] position) keys: every dropped key stays within tol (m) of the
    straight interpolation between the kept keys (same scheme as reduce_keys)."""
    keys = [(int(f), np.asarray(p, float)) for f, p in keys]
    n = len(keys)
    if n <= 2:
        return keys
    keep = [0]
    last = 0
    i = 2
    while i < n:
        t0, p0 = keys[last]
        t1, p1 = keys[i]
        ok = True
        for m in range(last + 1, i):
            t, p = keys[m]
            pred = p0 + (p1 - p0) * ((t - t0) / max(1, t1 - t0))
            if float(np.linalg.norm(pred - p)) > tol:
                ok = False
                break
        if ok:
            i += 1
        else:
            last = i - 1
            keep.append(last)
            i = last + 2
    keep.append(n - 1)
    return [keys[k] for k in sorted(set(keep))]


def _kf_frame(k):
    return int(k['Frame'])


def rotation_track(orig_keys, values60, lo, hi):
    """Keys of one rotation track. Frames lo..hi (the window plus one frame on each side) are rebuilt from values60
    at every 60 Hz frame and reduced; the frames lo and hi keep the original value (w = 0 there), so every 60 Hz frame
    outside the window plays the original motion exactly. Original keys outside [lo, hi] are copied unchanged."""
    before = [k for k in orig_keys if _kf_frame(k) < lo]
    after = [k for k in orig_keys if _kf_frame(k) > hi]
    frames_in = inside_key_frames(np.arange(lo, hi + 1), [_kf_frame(k) for k in orig_keys])
    inside = [(f, values60[f]) for f in frames_in]
    red = reduce_keys(inside, REDUCE_TOL_DEG)
    mid = [{'Frame': int(f), 'LocalRotation': [float(x) for x in q]} for f, q in red]
    return before + mid + after


def translation_track(orig_keys, values60, lo, hi):
    """Keys of the Reference position track, same frame range rule as rotation_track."""
    before = [k for k in orig_keys if _kf_frame(k) < lo]
    after = [k for k in orig_keys if _kf_frame(k) > hi]
    frames_in = inside_key_frames(np.arange(lo, hi + 1), [_kf_frame(k) for k in orig_keys])
    inside = [(f, values60[f]) for f in frames_in]
    red = reduce_positions(inside, T9C_TRANS_TOL)
    mid = [{'Frame': int(f), 'LocalPosition': [float(x) for x in p]} for f, p in red]
    return before + mid + after


def bake_t9c(rig, m, dump, seq, acu_s, ev, wd2_ev, o, window, offsets):
    """Root solve + hand IK + tracks for one clip. Returns a dict (see the keys at the end)."""
    L = float(dump['Length'])
    ob = ev['obstacle']
    F60 = np.asarray(seq.frames, int)
    nf = int(F60[-1])
    base = bake_clip(rig, m, dump, acu_s, wd2_ev, o, window)
    pat = base['patched']
    tm = base['time_map']
    orig = np.asarray(seq.local_rot, float)
    ref0 = np.asarray(seq.local_pos[:, 0], float)
    w60 = fade_weight(F60 / FPS, o, window)
    p60 = patch_rotations60(base['raw'], pat, F60)
    patch = orig.copy()
    blended = orig.copy()
    for ai in pat:
        patch[:, ai] = p60[ai]
        blended[:, ai] = mix_rot(orig[:, ai], p60[ai], w60)

    body = sorted(set(rig.map.values()))
    # the root solve sees the body minus T9C_SOLVE_EXCLUDE (diagnostic only; the spec run excludes nothing)
    solve_body = [bi for bi in body if _bone_name(rig, bi) not in T9C_SOLVE_EXCLUDE]
    point_of = {}
    for name in T9C_HANDS + tuple(n for leg in T9C_LEGS for n in leg):
        ai = rig.ac_by_hash[zlib.crc32(name.encode())]
        if ai not in body:
            raise KeyError('%s is not a mapped body bone' % name)
        point_of[name] = solve_body.index(ai)
    hand_ac = {h: rig.ac_by_hash[zlib.crc32(h.encode())] for h in T9C_HANDS}

    win = window_frames(o, window, nf)
    hp_iv = clip_interval_for_wd2(tm, wd2_ev['hand_plant_start'], wd2_ev['hand_plant_end'], L)
    plant = plant_in_window(hp_iv, win)
    if T9C_PLANT_SOURCE == 'acu':          # diagnostic only: the original clip's own hand-on-top frames
        acu_hand = set()
        for h in T9C_HANDS:
            for s0, e0, kind in ev['contacts'][h]:
                if kind == 'top':
                    acu_hand.update(range(int(s0), int(e0) + 1))
        plant = np.array(sorted(f for f in win if int(f) in acu_hand), int)
    plant_j = np.nonzero(np.isin(win, plant))[0]

    # b. root solve on the blended pose (what the game shows before the offset)
    P0 = acuclip.world_positions(composed(seq, blended, ref0), dump)
    Pw = P0[win][:, solve_body]
    cons, cinfo = root_constraints(Pw, win, plant_j, ev['contacts'], point_of, ob, offsets)
    delta_w, sinfo = rootsolve.solve_offset(Pw, cons, lam_smooth=T9C_LAM_SMOOTH, lam_small=T9C_LAM_SMALL,
                                            ineq_weight=T9C_INEQ_WEIGHT)
    delta = np.zeros((len(F60), 3))
    delta[win] = delta_w
    ref_new = ref0 + delta

    # c. hand IK on the un-faded patch, fixed targets at the plant midpoint
    disp = acuclip.markers(dump, F60)['disp']
    ik_in = composed(seq, patch, ref_new)
    targets = {}
    hand_targets = {}
    if plant.size:
        Wp = acuclip.world_positions(ik_in, dump)
        mid = int(plant[len(plant) // 2])
        wgt = blend.fade_weights(len(F60), [(int(plant[0]), int(plant[-1]))], T9C_IK_RAMP)
        for h in T9C_HANDS:
            p = Wp[mid, hand_ac[h]]
            tgt = np.array([p[0],
                            min(max(p[1], float(ob['near_y']) + T9C_IK_MARGIN), float(ob['far_y']) - T9C_IK_MARGIN),
                            float(ob['height']) + offsets['Hand']])
            hand_targets[h] = tgt
            targets[h] = [(int(f), tgt, float(wgt[f])) for f in np.nonzero(wgt > 0)[0]]
    if targets:
        root_rot = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(F60), 1))
        ik_out = ik.solve_sequence(ik_in, root_rot=root_rot, root_pos=disp, targets=targets)
        changed = np.nonzero(np.any(np.abs(ik_out.local_rot - ik_in.local_rot) > 1e-12, axis=(0, 2)))[0]
        bad = sorted(set(int(c) for c in changed) - set(pat))
        if bad:
            raise ValueError('IK changed bones without a rotation track: %s' % [rig.ac[b]['name'] for b in bad])
        patch_ik = np.asarray(ik_out.local_rot, float)
    else:
        patch_ik = patch

    final = orig.copy()
    for ai in pat:
        final[:, ai] = mix_rot(orig[:, ai], patch_ik[:, ai], w60)

    # d. tracks (rebuilt over the window plus one frame on each side, see rotation_track)
    lo = max(0, int(win[0]) - 1)
    hi = min(nf, int(win[-1]) + 1)
    rot_tracks = {}
    for ai in pat:
        tid = rig.ac[ai]['hash']
        ok = dumps.track(dump, tid, 0) or []
        rot_tracks[ai] = rotation_track(ok, final[:, ai], lo, hi)
    ref_keys = dumps.track(dump, REF_TID, 3) or []
    trans_keys = translation_track(ref_keys, ref_new, lo, hi)

    return dict(patched=pat, rot_tracks=rot_tracks, trans_keys=trans_keys, final=final, ref_new=ref_new,
                delta=delta, win=win, plant=plant, plant_j=plant_j, hp_iv=hp_iv, targets=hand_targets,
                body=body, solve_info=sinfo, constraint_info=cinfo, time_map=tm, o=float(o), window=float(window),
                disp=disp)


def sample_track_rot(keys, frames):
    """[F, 4] rotations at 60 Hz frames from a rotation track's keys (slerp, clamped), like the plugin samples them."""
    pairs = [(_kf_frame(k), np.asarray(k['LocalRotation'], float)) for k in keys]
    return sample_keys(pairs, frames)


def sample_track_pos(keys, frames):
    """[F, 3] positions at 60 Hz frames from a translation track's keys (lerp, clamped)."""
    kf = np.array([_kf_frame(k) for k in keys], float)
    kv = np.array([k['LocalPosition'] for k in keys], float)
    fr = np.asarray(frames, float)
    return np.stack([np.interp(fr, kf, kv[:, c]) for c in range(3)], axis=-1)


def validate_t9c(rig, dump, seq, res, o, window, ob):
    """Metrics of the tracks as the plugin would play them (60 Hz samples of the composed pose)."""
    F60 = np.asarray(seq.frames, int)
    win = res['win']
    body = res['body']
    pat = res['patched']
    orig = np.asarray(seq.local_rot, float)
    rot = orig.copy()
    for ai in pat:
        rot[:, ai] = sample_track_rot(res['rot_tracks'][ai], F60)
    pos = sample_track_pos(res['trans_keys'], F60)
    Pf = acuclip.world_positions(composed(seq, rot, pos), dump)
    Po = acuclip.world_positions(seq, dump)
    box_p, gnd_p = penetration_parts(Pf[win][:, body], ob)
    box_o, gnd_o = penetration_parts(Po[win][:, body], ob)

    plant = res['plant']
    hand_ac = {h: rig.ac_by_hash[zlib.crc32(h.encode())] for h in T9C_HANDS}
    hand_err = None
    hand_z = None
    if plant.size:
        errs = []
        zs = []
        for h in T9C_HANDS:
            d = Pf[plant, hand_ac[h]] - res['targets'][h]
            errs.append(np.linalg.norm(d, axis=1))
            zs.append((Pf[plant, hand_ac[h], 2] - ob['height']) * 100.0)
        hand_err = float(np.max(np.stack(errs)) * 100.0)
        zz = np.concatenate(zs)
        hand_z = dict(min=float(zz.min()), median=float(np.median(zz)), max=float(zz.max()))

    delta = pos - np.asarray(seq.local_pos[:, 0], float)
    outside = np.ones(len(F60), bool)
    outside[win] = False
    d_out = float(np.max(np.abs(delta[outside]))) if np.any(outside) else 0.0
    dstep = np.linalg.norm(np.diff(delta, axis=0), axis=1) * 100.0
    dstep_max = float(dstep.max()) if dstep.size else 0.0
    dy_max = float(np.max(np.abs(delta[:, 1])) * 100.0)
    dz_max = float(np.max(np.abs(delta[:, 2])) * 100.0)

    # outside the window the rotations and the translation must be the original
    rot_out = 0.0
    for ai in pat:
        if np.any(outside):
            rot_out = max(rot_out, float(np.max(angle_deg(rot[outside, ai], orig[outside, ai]))))
    trans_err = float(np.max(np.abs(pos[win] - res['ref_new'][win])) * 100.0)  # mm if *1000; reported in mm below
    rot_err = 0.0
    for ai in pat:
        rot_err = max(rot_err, float(np.max(angle_deg(rot[win, ai], res['final'][win, ai]))))

    # steps of the patched bones at 30 Hz inside the window and at the two crossfades (as in T9b)
    even = np.where((F60 % 2) == 0)[0]
    in_win = even[np.isin(even, win)]
    t = F60 / FPS
    b1 = even[(t[even] >= o - FADE_IN) & (t[even] <= o + FADE_IN)]
    b2 = even[(t[even] >= o + window - FADE_OUT) & (t[even] <= o + window + FADE_IN)]
    step_p = _step_deg(rot, in_win, pat)
    bnd_p = max(_step_deg(rot, b1, pat), _step_deg(rot, b2, pat))

    pen_p = max(box_p, gnd_p) * 100.0
    fails = []
    if pen_p > T9C_PEN_CM:
        fails.append('penetration %.1f cm' % pen_p)
    if hand_err is not None and hand_err > T9C_HAND_TOL_CM:
        fails.append('hand %.1f cm from target' % hand_err)
    if dstep_max > T9C_DELTA_STEP_CM:
        fails.append('Delta step %.1f cm/frame' % dstep_max)
    metrics = dict(pen_box_cm=box_p * 100.0, pen_ground_cm=gnd_p * 100.0, pen_patch_cm=pen_p,
                   pen_orig_box_cm=box_o * 100.0, pen_orig_ground_cm=gnd_o * 100.0,
                   pen_orig_cm=max(box_o, gnd_o) * 100.0,
                   hand_err_cm=hand_err, hand=hand_z, dy_max_cm=dy_max, dz_max_cm=dz_max,
                   dstep_max_cm=dstep_max, delta_outside_m=d_out, rot_outside_deg=rot_out,
                   trans_reduce_err_mm=trans_err * 10.0, rot_reduce_err_deg=rot_err,
                   step_patch_deg=step_p, boundary_patch_deg=bnd_p, n_patched=len(pat),
                   n_keys=sum(len(v) for v in res['rot_tracks'].values()) + len(res['trans_keys']),
                   fails=fails, fail=bool(fails))
    arrays = dict(frames=F60, Pf=Pf, Po=Po, win=win, body=body)
    return metrics, arrays


def write_patch_t9c(path, rig, rot_tracks, trans_keys, length):
    """Patch file: rotation tracks (codec 1, infotype 0) and the Reference translation track (codec 27, infotype 3)."""
    tr = []
    for ai in sorted(rot_tracks, key=lambda a: rig.ac[a]['hash']):
        kf = [{'Frame': int(k['Frame']), 'LocalRotation': [round(float(x), 9) for x in k['LocalRotation']]}
              for k in rot_tracks[ai]]
        tr.append({'TrackID': str(rig.ac[ai]['hash']), 'codec': 1, 'infotype': 0, 'Keyframes': kf})
    tr.append({'TrackID': str(REF_TID), 'codec': 27, 'infotype': 3,
               'Keyframes': [{'Frame': int(k['Frame']), 'LocalPosition': [round(float(x), 9) for x in k['LocalPosition']]}
                             for k in trans_keys]})
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'Length': round(float(length), 6), 'Tracks': tr}, f)


def mapping_entry_t9c(handle, name, source, enabled):
    return {'handle': '0x' + handle, 'name': name, 'source': source, 'enabled': bool(enabled), 'prebaked': True,
            'patch_translation': True}


def plot_clip_t9c(path, name, rig, res, arrays, metrics, o, window, ob):
    """Side view: obstacle, original (grey) and patched (colour) figures, pelvis paths, hand targets (crosses)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    frames = arrays['frames']
    Pf, Po, body = arrays['Pf'], arrays['Po'], arrays['body']
    win = arrays['win']
    bset = set(body)
    edges = [(ai, rig.ac[ai]['parent']) for ai in body if rig.ac[ai]['parent'] in bset]
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=110)
    ax.add_patch(Rectangle((ob['near_y'], 0.0), ob['far_y'] - ob['near_y'], ob['height'],
                           facecolor='#8b6f47', alpha=0.35, edgecolor='#5c4a2e', label='obstacle'))
    ax.axhline(0.0, color='#777', lw=0.8)
    sel_t = np.linspace(o, o + window, 12)
    idxs = [int(np.argmin(np.abs(frames / FPS - s))) for s in sel_t]
    colours = plt.cm.viridis(np.linspace(0.05, 0.95, len(idxs)))
    for k, fi in enumerate(idxs):
        for a, p in edges:
            ax.plot([Po[fi, a, 1], Po[fi, p, 1]], [Po[fi, a, 2], Po[fi, p, 2]], color='#c9c9c9', lw=1.0, zorder=1)
        for a, p in edges:
            ax.plot([Pf[fi, a, 1], Pf[fi, p, 1]], [Pf[fi, a, 2], Pf[fi, p, 2]], color=colours[k], lw=2.0, zorder=2)
    ax.plot(Po[win, 0, 1], Po[win, 0, 2], '--', color='#888', lw=1.2, label='pelvis original')
    ax.plot(Pf[win, 0, 1], Pf[win, 0, 2], '-', color='#1f77b4', lw=2.0, label='pelvis solved')
    for h in T9C_HANDS:
        ai = rig.ac_by_hash[zlib.crc32(h.encode())]
        ax.plot(Pf[win, ai, 1], Pf[win, ai, 2], '.', color='#d62728', ms=3, zorder=3)
        if h in res['targets']:
            tg = res['targets'][h]
            ax.plot(tg[1], tg[2], 'x', color='k', ms=10, mew=2.2, zorder=4)
    ax.plot([], [], 'x', color='k', ms=8, mew=2, label='hand target')
    ys = Pf[idxs][:, body, 1]
    ax.set_xlim(float(ys.min()) - 0.6, float(ys.max()) + 0.6)
    ax.set_ylim(-0.1, max(ob['height'] + 1.0, float(Pf[idxs][:, body, 2].max()) + 0.2))
    ax.set_aspect('equal')
    ax.set_xlabel('entity forward Y (m)')
    ax.set_ylabel('Z (m)')
    ax.legend(fontsize=7, loc='upper right')
    hs = metrics['hand']
    hand_txt = 'n/a' if hs is None else '%+.0f / %+.0f / %+.0f cm' % (hs['min'], hs['median'], hs['max'])
    herr = 'n/a' if metrics['hand_err_cm'] is None else '%.1f cm' % metrics['hand_err_cm']
    status = 'FAIL' if metrics['fail'] else 'PASS'
    ax.set_title('%s  [%s]\npen patch %.1f cm (orig %.1f); hand above top min/med/max %s; hand to target max %s;\n'
                 'Delta max |dY| %.1f cm, |dZ| %.1f cm, step %.2f cm/frame; enabled = %s'
                 % (name, status, metrics['pen_patch_cm'], metrics['pen_orig_cm'], hand_txt, herr,
                    metrics['dy_max_cm'], metrics['dz_max_cm'], metrics['dstep_max_cm'],
                    'no' if metrics['fail'] else 'yes'), fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _table_before(rows):
    lines = ['| clip | o / window (s) | pen patch box / ground (cm) | pen orig box / ground (cm) | '
             'hand above top min / med / max (cm) | status |',
             '|---|---|---|---|---|---|']
    for r in rows:
        mt = r['metrics']
        hs = mt['hand']
        hand = 'n/a' if hs is None else '%+.1f / %+.1f / %+.1f' % (hs['min'], hs['median'], hs['max'])
        lines.append('| %s | %.2f / %.2f | %.1f / %.1f | %.1f / %.1f | %s | %s |'
                     % (r['name'], r['o'], r['window'], mt['pen_box_cm'], mt['pen_ground_cm'],
                        mt['pen_orig_box_cm'], mt['pen_orig_ground_cm'], hand,
                        'FAIL' if mt['fail'] else 'PASS'))
    return lines


def _table_after(rows):
    lines = ['| clip | o / window (s) | plant in window (s) | pen box / ground (cm) | hand to target max (cm) | '
             'hand above top min / med / max (cm) | max abs dY / dZ (cm) | max Delta step (cm/frame) | '
             'pins kept / wanted | status | enabled |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        mt = r['metrics']
        hs = mt['hand']
        hand = 'n/a' if hs is None else '%+.1f / %+.1f / %+.1f' % (hs['min'], hs['median'], hs['max'])
        herr = 'n/a' if mt['hand_err_cm'] is None else '%.1f' % mt['hand_err_cm']
        pl = r['plant_s']
        plant = 'none' if pl is None else '%.2f to %.2f' % pl
        ci = r['constraint_info']
        lines.append('| %s | %.2f / %.2f | %s | %.1f / %.1f | %s | %s | %.1f / %.1f | %.2f | %d / %d | %s | %s |'
                     % (r['name'], r['o'], r['window'], plant, mt['pen_box_cm'], mt['pen_ground_cm'], herr, hand,
                        mt['dy_max_cm'], mt['dz_max_cm'], mt['dstep_max_cm'], ci['pin_frames'],
                        ci['pin_frames_wanted'], 'FAIL' if mt['fail'] else 'PASS',
                        'no' if mt['fail'] else 'yes'))
    return lines


def write_report_t9c(path, rows, before_rows, offsets, play_model_path, notes_path=OBSERVATIONS):
    n_on = sum(1 for r in rows if not r['metrics']['fail'])
    lines = ['# T9c: WD2 catpass (kong) onto ACU SkipOver clips, with root offset and hand IK', '',
             'Generated by `v2/pipeline/bake2.py` (bake_all_t9c). Times in seconds of the clip; heights in cm; '
             'angles in degrees. Play model: ' + (play_model_path if os.path.exists(play_model_path)
                                                  else 'none (fallback windows)') + '.', '',
             '## Method', '',
             '- Contact offsets (median of joint height above its support over the ORIGINAL SkipOver dumps, all contact '
             'frames of contacts.events): Hand %.3f m, Foot %.3f m, FootThumb1 (toe) %.3f m. Samples: %s.'
             % (offsets['Hand'], offsets['Foot'], offsets['FootThumb1'], offsets['n']),
             '- Hand plant: WD2 hand_plant_start..end mapped through the inverse of the time map, intersected with the play window.',
             '- Feet: ACU contacts of the original clip (Foot or FootThumb1, the lower one per frame), inside the window.',
             '- Root solve: `rootsolve.solve_offset` over the window only, lam_smooth %g, lam_small %s, Height weight %g, '
             'Above(all) against the box and ground, Pin(0) on the first and last %d window frames (minus the plant). '
             'Delta is exactly 0 outside the window.' % (T9C_LAM_SMOOTH, T9C_LAM_SMALL, T9C_HEIGHT_WEIGHT, T9C_PIN_FRAMES),
             '- Hand IK (`ik.solve_sequence`): on the un-faded patch with Reference + Delta; each hand to a fixed target = '
             'its position at the plant midpoint with z = top + Hand offset and y clamped to the top (margin %.2f m); '
             'weight 1 in the plant, %d-frame ramp outside. The window crossfade is then applied again.'
             % (T9C_IK_MARGIN, T9C_IK_RAMP),
             '- Reference position track: original keys outside the window, original + Delta inside (2-frame grid plus '
             'original keys), reduced with %.0f mm tolerance; `patch_translation: true`.' % (T9C_TRANS_TOL * 1000),
             '- PASS = penetration <= %.0f cm (box and ground, window) AND hands within %.0f cm of target during the plant '
             'AND largest Delta change <= %.0f cm per 60 Hz frame. FAIL clips have `enabled: false`.'
             % (T9C_PEN_CM, T9C_HAND_TOL_CM, T9C_DELTA_STEP_CM), '',
             '## Before (T9b: rotations only, same play windows)', ''] + _table_before(before_rows) + [
             '', '## After (T9c: root offset + hand IK + rotations)', ''] + _table_after(rows) + ['',
             'Enabled: %d of %d. Enabled clips: %s.'
             % (n_on, len(rows), ', '.join(r['name'] for r in rows if not r['metrics']['fail']) or 'none'), '',
             '## Failure reasons', '']
    for r in rows:
        why = '; '.join(r['metrics']['fails']) if r['metrics']['fails'] else 'none (PASS)'
        lines.append('- %s: %s' % (r['name'], why))
    lines += ['', '## Observations', '']
    if os.path.exists(notes_path):
        with open(notes_path, encoding='utf-8') as f:
            lines.append(f.read().rstrip())
    else:
        lines.append('(no observations file)')
    lines.append('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def bake_all_t9c(write=True):
    """All ten SkipOver clips with the T9c pipeline. Writes patches, mapping.json, PNGs and report.md when write."""
    os.makedirs(PATCH_OUT, exist_ok=True)
    os.makedirs(BAKE_OUT, exist_ok=True)
    rig = R.Rig()
    ev_json = load_events()
    m = wd2.load(ev_json['stem'])
    wd2_ev = dict(takeoff=ev_json['takeoff'], hand_plant_start=ev_json['hand_plant_start'],
                  hand_plant_end=ev_json['hand_plant_end'], landing=ev_json['landing'],
                  static_until=ev_json['static_until'], duration=float(m['dur']))
    model = load_play_model()
    measured = load_measured()
    clips = []
    for handle, name, dump in targets():
        seq = acuclip.pose_sequence(dump, step=1)
        ev = contacts.events(seq, dump)
        clips.append(dict(handle=handle, name=name, dump=dump, seq=seq, ev=ev, acu=acu_seconds(ev),
                          L=float(dump['Length'])))
    offsets = measure_contact_offsets([(c['name'], c['dump'], c['seq'], c['ev']) for c in clips])
    before_rows = bake_all(write=False)
    rows = []
    mapping = []
    for c in clips:
        o, window, source = play_for(c['name'], c['L'], model, measured)
        res = bake_t9c(rig, m, c['dump'], c['seq'], c['acu'], c['ev'], wd2_ev, o, window, offsets)
        metrics, arrays = validate_t9c(rig, c['dump'], c['seq'], res, o, window, c['ev']['obstacle'])
        pl = None
        if res['plant'].size:
            pl = (float(res['plant'][0]) / FPS, float(res['plant'][-1]) / FPS)
        rows.append(dict(name=c['name'], handle=c['handle'], length=c['L'], o=o, window=window, source=source,
                         metrics=metrics, constraint_info=res['constraint_info'], solve_info=res['solve_info'],
                         plant_s=pl))
        fn = '%s.baked.json' % c['handle']
        if write:
            write_patch_t9c(os.path.join(PATCH_OUT, fn), rig, res['rot_tracks'], res['trans_keys'], c['L'])
            mapping.append(mapping_entry_t9c(c['handle'], c['name'], fn, not metrics['fail']))
            plot_clip_t9c(os.path.join(BAKE_OUT, c['name'] + '.png'), c['name'], rig, res, arrays, metrics, o,
                          window, c['ev']['obstacle'])
        si = res['solve_info']
        print('%-48s o %.2f win %.2f  pen %.1f cm  hand %s  dstep %.2f  solve iters %d conv %s  %s' % (
            c['name'], o, window, metrics['pen_patch_cm'],
            'n/a' if metrics['hand_err_cm'] is None else '%.1f' % metrics['hand_err_cm'],
            metrics['dstep_max_cm'], si['iterations'], si['converged'],
            'FAIL (%s)' % '; '.join(metrics['fails']) if metrics['fail'] else 'PASS'))
    if write:
        with open(os.path.join(PATCH_OUT, 'mapping.json'), 'w', encoding='utf-8') as f:
            json.dump({'clips': mapping}, f, indent=1)
        write_report_t9c(os.path.join(BAKE_OUT, 'report.md'), rows, before_rows, offsets, PLAY_MODEL)
    return rows, before_rows, offsets


# ------------------------------------------------------------------------------------------------ 8. T9d
# Method change (orchestrator, T9d): no least-squares root solve. Instead
#   1. ROOT LIFT: required lift per window frame from the non-finger body points, smooth upper envelope, taper.
#   2. HAND IK: fixed target at the first plant frame after the lift, IK weight while reachable.
#   3. FOOT CLAMP IK: raise the lower of Foot / FootThumb1 to its support where it is below.
#   4. window crossfade (as T9b/T9c); outside the window the patch equals the original.
# The T9c solver path (bake_t9c) is kept behind --t9c.
T9D_LIFT_CLEAR = 0.01        # m, clearance the lift aims for above the surface
T9D_LIFT_SIGMA = 5.0         # frames, Gaussian of the envelope
T9D_LIFT_ITER = 40           # envelope iterations
T9D_TAPER = 6                # frames at each window end where the lift is tapered to 0
T9D_RAMP = 3                 # frames of IK weight ramp in / out
T9D_REACH = 0.95             # hand IK on while shoulder->target <= REACH * arm length
T9D_TOUCH = 4                # frames both hands must stay within T9C_HAND_TOL_CM at the start of the plant
T9D_FINGER_WORDS = ('Index', 'Middle', 'Pinky', 'Ring', 'Thumb')
T9D_LIFT_NO_FEET = False   # diagnostic only: leave Foot / FootThumb1 out of the lift (spec run: False)


def finger_bones(rig, body):
    """ACU bone indices of the hand finger joints among `body` (LeftHand*/RightHand* except the hand itself)."""
    out = []
    for bi in body:
        n = _bone_name(rig, bi)
        if n.startswith(('LeftHand', 'RightHand')) and n not in ('LeftHand', 'RightHand') \
                and any(w in n for w in T9D_FINGER_WORDS):
            out.append(bi)
    return out


def lift_need(P, ob):
    """Required lift (m) per frame: max(0, max_k(surface(y_k) + clearance - z_k)) over the points P [F, K, 3].

    surface = obstacle top for points inside the band [near_y, far_y] (laterally inside the box), ground otherwise.
    """
    P = np.asarray(P, float)
    if P.shape[1] == 0:
        return np.zeros(P.shape[0])
    y = P[..., 1]
    z = P[..., 2]
    surf = np.where((y >= ob['near_y']) & (y <= ob['far_y']), ob['height'], 0.0)
    return np.maximum(0.0, (surf + T9D_LIFT_CLEAR - z).max(axis=1))


def _gauss_smooth(x, sigma):
    """1-D Gaussian smoothing with edge-replicating ('nearest') padding, numpy only (no scipy here)."""
    r = int(math.ceil(4.0 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    xp = np.concatenate([np.full(r, x[0]), x, np.full(r, x[-1])])
    return np.convolve(xp, k, mode='valid')


def smooth_envelope(L):
    """Upper envelope: D = L; repeat T9D_LIFT_ITER times: D = max(L, gauss(D, T9D_LIFT_SIGMA))."""
    L = np.asarray(L, float)
    if L.size == 0 or not np.any(L > 0):
        return np.zeros_like(L)
    D = L.copy()
    for _ in range(T9D_LIFT_ITER):
        D = np.maximum(L, _gauss_smooth(D, T9D_LIFT_SIGMA))
    return D


def lift_taper(D, L):
    """Multiply D by a smoothstep that is 0 at the window start / end frame and 1 from T9D_TAPER frames in.

    A 6-frame end zone in which the required lift L is > 0 is NOT tapered (reported). Returns (D, report).
    """
    D = np.asarray(D, float)
    L = np.asarray(L, float)
    n = len(D)
    j = np.arange(n, dtype=float)

    def ss(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    w_s = ss(j / T9D_TAPER)
    w_e = ss((n - 1 - j) / T9D_TAPER)
    start_zone = j < T9D_TAPER
    end_zone = j > n - 1 - T9D_TAPER
    rep = dict(start_untapered=False, end_untapered=False)
    if np.any(L[start_zone] > 0):
        w_s = np.where(start_zone, 1.0, w_s)
        rep['start_untapered'] = True
    if np.any(L[end_zone] > 0):
        w_e = np.where(end_zone, 1.0, w_e)
        rep['end_untapered'] = True
    return D * np.minimum(w_s, w_e), rep


def _ik_weights(nF, win, start, plant_end, reach, ramp=None):
    """IK weight per 60 Hz frame. Ramp in over T9D_RAMP frames before `start`; 1 while `reach` (bool per frame
    from start) holds and the plant lasts; then fade to 0 over T9D_RAMP frames from the first frame that is out
    of reach or after plant_end. Frames outside the window get 0."""
    ramp = T9D_RAMP if ramp is None else int(ramp)
    w = np.zeros(nF)
    inwin = np.zeros(nF, bool)
    inwin[np.asarray(win, int)] = True
    frames = np.arange(start, plant_end + 1)
    bad = np.nonzero(~np.asarray(reach, bool))[0]
    u = int(frames[bad[0]]) if bad.size else int(plant_end) + 1
    for f in range(max(0, start - ramp), nF):
        if f < start:
            v = (f - (start - ramp)) / float(ramp)
        elif f < u:
            v = 1.0
        else:
            v = max(0.0, 1.0 - (f - u + 1) / float(ramp))
        w[f] = v
    w[~inwin] = 0.0
    return w, u


def bake_t9d(rig, m, dump, seq, acu_s, ev, wd2_ev, o, window, offsets):
    """T9d: lift, hand IK, foot clamp, window crossfade. Returns a dict with the same keys as bake_t9c (plus info)."""
    L_ = float(dump['Length'])
    ob = ev['obstacle']
    F60 = np.asarray(seq.frames, int)
    nf = int(F60[-1])
    base = bake_clip(rig, m, dump, acu_s, wd2_ev, o, window)
    pat = base['patched']
    tm = base['time_map']
    orig = np.asarray(seq.local_rot, float)
    ref0 = np.asarray(seq.local_pos[:, 0], float)
    w60 = fade_weight(F60 / FPS, o, window)
    p60 = patch_rotations60(base['raw'], pat, F60)
    patch = orig.copy()
    for ai in pat:
        patch[:, ai] = p60[ai]

    body = sorted(set(rig.map.values()))
    fingers = set(finger_bones(rig, body))
    body_nf = [bi for bi in body if bi not in fingers]
    feet_ac = {rig.ac_by_hash[zlib.crc32(n.encode())] for leg in T9C_LEGS for n in leg}
    body_lift = [bi for bi in body_nf if not (T9D_LIFT_NO_FEET and bi in feet_ac)]
    win = window_frames(o, window, nf)
    hp_iv = clip_interval_for_wd2(tm, wd2_ev['hand_plant_start'], wd2_ev['hand_plant_end'], L_)
    plant = plant_in_window(hp_iv, win)
    ac = lambda name: rig.ac_by_hash[zlib.crc32(name.encode())]

    # 1. root lift from the BLENDED pose (what the game plays: the un-faded patch dips into the ground during the
    #    fade-in, which the crossfade hides; lifting for it gave a 17.7 cm step at the window start)
    blend_rot = orig.copy()
    for ai in pat:
        blend_rot[:, ai] = mix_rot(orig[:, ai], patch[:, ai], w60)
    P_blend = acuclip.world_positions(composed(seq, blend_rot, ref0), dump)
    Lw = lift_need(P_blend[win][:, body_lift], ob)
    Dw = smooth_envelope(Lw)
    Dw, taper_rep = lift_taper(Dw, Lw)
    Dfull = np.zeros(len(F60))
    Dfull[win] = Dw
    delta = np.zeros((len(F60), 3))
    delta[:, 2] = Dfull
    ref1 = ref0 + delta
    P1 = acuclip.world_positions(composed(seq, patch, ref1), dump)   # post-lift, pre-IK pose
    disp = acuclip.markers(dump, F60)['disp']
    ik_in = composed(seq, patch, ref1)

    # 2. hand IK: fixed targets at the first plant frame after the lift
    targets = {}
    hand_targets = {}
    reach_info = {}
    if plant.size:
        p0 = int(plant[0])
        pend = int(plant[-1])
        for h, side in (('LeftHand', 'Left'), ('RightHand', 'Right')):
            ua, la, ha = ac(side + 'Arm'), ac(side + 'ForeArm'), ac(h)
            p = P1[p0, ha]
            tgt = np.array([p[0],
                            min(max(p[1], float(ob['near_y']) + T9C_IK_MARGIN), float(ob['far_y']) - T9C_IK_MARGIN),
                            float(ob['height']) + offsets['Hand']])
            hand_targets[h] = tgt
            fr = np.arange(p0, pend + 1)
            sh = P1[fr, ua]
            d = np.linalg.norm(tgt[None, :] - sh, axis=1)
            arm = np.linalg.norm(P1[fr, la] - P1[fr, ua], axis=1) + np.linalg.norm(P1[fr, ha] - P1[fr, la], axis=1)
            reach = d <= T9D_REACH * arm
            wh, u = _ik_weights(len(F60), win, p0, pend, reach)
            reach_info[h] = dict(first_unreachable=u if u <= pend else None, reach_frames=int(reach.sum()))
            targets[h] = [(int(f), tgt, float(wh[f])) for f in np.nonzero(wh > 0)[0]]

    # 3. foot clamp: the lower of Foot / FootThumb1 below its support (+ offset) is raised to it, every window frame
    foot_targets = {}
    foot_info = {}
    for side in ('Left', 'Right'):
        foot, toe = side + 'Foot', side + 'FootThumb1'
        need = np.zeros(len(F60))
        for f in win:
            best = 0.0
            for name, off in ((foot, offsets['Foot']), (toe, offsets['FootThumb1'])):
                y, z = P1[f, ac(name), 1], P1[f, ac(name), 2]
                surf = float(ob['height']) if float(ob['near_y']) <= y <= float(ob['far_y']) else 0.0
                best = max(best, surf + off - z)
            need[f] = best
        pos = P1[:, ac(foot)]
        pos_idx = np.nonzero(need > 0)[0]
        n_clamp = int(pos_idx.size)
        foot_info[foot] = dict(frames=n_clamp, max_raise_cm=float(need.max() * 100.0))
        if n_clamp == 0:
            continue
        # weights: 1 on the clamp frames, 3-frame linear ramps around them; target uses the nearest clamp need
        wf = blend.fade_weights(len(F60), [(int(a), int(b)) for a, b in _runs(pos_idx)], T9D_RAMP)
        for f in np.nonzero(wf > 0)[0]:
            k = pos_idx[np.argmin(np.abs(pos_idx - f))]
            tgt = pos[f] + np.array([0.0, 0.0, need[k]])
            foot_targets.setdefault(foot, []).append((int(f), tgt, float(wf[f])))

    all_targets = {**targets, **foot_targets}
    if all_targets:
        root_rot = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(F60), 1))
        ik_out = ik.solve_sequence(ik_in, root_rot=root_rot, root_pos=disp, targets=all_targets)
        changed = np.nonzero(np.any(np.abs(ik_out.local_rot - ik_in.local_rot) > 1e-12, axis=(0, 2)))[0]
        bad = sorted(set(int(c) for c in changed) - set(pat))
        if bad:
            raise ValueError('IK changed bones without a rotation track: %s' % [rig.ac[b]['name'] for b in bad])
        patch_ik = np.asarray(ik_out.local_rot, float)
    else:
        patch_ik = patch

    # 4. window crossfade and tracks (same as T9c)
    final = orig.copy()
    for ai in pat:
        final[:, ai] = mix_rot(orig[:, ai], patch_ik[:, ai], w60)
    lo = max(0, int(win[0]) - 1)
    hi = min(nf, int(win[-1]) + 1)
    rot_tracks = {}
    for ai in pat:
        ok = dumps.track(dump, rig.ac[ai]['hash'], 0) or []
        rot_tracks[ai] = rotation_track(ok, final[:, ai], lo, hi)
    ref_keys = dumps.track(dump, REF_TID, 3) or []
    trans_keys = translation_track(ref_keys, ref1, lo, hi)

    return dict(patched=pat, rot_tracks=rot_tracks, trans_keys=trans_keys, final=final, ref_new=ref1,
                delta=delta, win=win, plant=plant, plant_j=np.nonzero(np.isin(win, plant))[0], hp_iv=hp_iv,
                targets=hand_targets, body=body, body_nf=body_nf, fingers=sorted(fingers),
                solve_info=dict(iterations=0, converged=True, max_residual={}, max_penetration=0.0,
                                n_violations=0, active=[], lift_max_cm=float(Dw.max() * 100.0) if Dw.size else 0.0,
                                lift_taper=taper_rep, lift_frames=int(np.sum(Lw > 0))),
                constraint_info=dict(pin_frames=0, pin_frames_wanted=0, reach=reach_info, foot=foot_info),
                time_map=tm, o=float(o), window=float(window), disp=disp)


def _runs(idx):
    """Contiguous runs of a sorted int array as (first, last) pairs."""
    out = []
    if len(idx) == 0:
        return out
    s = int(idx[0])
    p = int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v == p + 1:
            p = v
            continue
        out.append((s, p))
        s = p = v
    out.append((s, p))
    return out


def touch_frames(Pf, plant, targets, rig, tol_cm):
    """Consecutive 60 Hz frames from the first plant frame during which each hand is within tol_cm of its target.

    Returns {'LeftHand': n, 'RightHand': n, 'min': n} (n counts frames; 0 when there is no plant)."""
    out = {}
    if plant.size == 0:
        return dict(LeftHand=0, RightHand=0, min=0)
    for h in T9C_HANDS:
        ai = rig.ac_by_hash[zlib.crc32(h.encode())]
        err = np.linalg.norm(Pf[plant, ai] - targets[h], axis=1) * 100.0
        n = 0
        for e in err:
            if e > tol_cm:
                break
            n += 1
        out[h] = int(n)
    out['min'] = min(out[h] for h in T9C_HANDS)
    return out


def validate_t9d(rig, dump, seq, res, o, window, ob, fi=None, fo=None):
    """Metrics of the tracks as the plugin plays them. Penetration uses the non-finger body; fingers listed apart."""
    fi_ = FADE_IN if fi is None else fi
    fo_ = FADE_OUT if fo is None else fo
    F60 = np.asarray(seq.frames, int)
    win = res['win']
    body_nf = res['body_nf']
    fing = res['fingers']
    pat = res['patched']
    orig = np.asarray(seq.local_rot, float)
    rot = orig.copy()
    for ai in pat:
        rot[:, ai] = sample_track_rot(res['rot_tracks'][ai], F60)
    pos = sample_track_pos(res['trans_keys'], F60)
    Pf = acuclip.world_positions(composed(seq, rot, pos), dump)
    Po = acuclip.world_positions(seq, dump)
    box_p, gnd_p = penetration_parts(Pf[win][:, body_nf], ob)
    box_o, gnd_o = penetration_parts(Po[win][:, body_nf], ob)
    fb_p, fg_p = penetration_parts(Pf[win][:, fing], ob) if fing else (0.0, 0.0)

    plant = res['plant']
    hand_ac = {h: rig.ac_by_hash[zlib.crc32(h.encode())] for h in T9C_HANDS}
    hand_err = None
    hand_z = None
    if plant.size:
        errs, zs = [], []
        for h in T9C_HANDS:
            errs.append(np.linalg.norm(Pf[plant, hand_ac[h]] - res['targets'][h], axis=1))
            zs.append((Pf[plant, hand_ac[h], 2] - ob['height']) * 100.0)
        hand_err = float(np.max(np.stack(errs)) * 100.0)
        zz = np.concatenate(zs)
        hand_z = dict(min=float(zz.min()), median=float(np.median(zz)), max=float(zz.max()))
    touch = touch_frames(Pf, plant, res['targets'], rig, T9C_HAND_TOL_CM)

    delta_v = pos - np.asarray(seq.local_pos[:, 0], float)
    outside = np.ones(len(F60), bool)
    outside[win] = False
    d_out = float(np.max(np.abs(delta_v[outside]))) if np.any(outside) else 0.0
    dstep = np.abs(np.diff(delta_v[:, 2])) * 100.0
    dstep_max = float(dstep.max()) if dstep.size else 0.0
    rot_out = 0.0
    for ai in pat:
        if np.any(outside):
            rot_out = max(rot_out, float(np.max(angle_deg(rot[outside, ai], orig[outside, ai]))))
    trans_err = float(np.max(np.abs(pos[win] - res['ref_new'][win])) * 100.0)
    rot_err = 0.0
    for ai in pat:
        rot_err = max(rot_err, float(np.max(angle_deg(rot[win, ai], res['final'][win, ai]))))

    even = np.where((F60 % 2) == 0)[0]
    in_win = even[np.isin(even, win)]
    t = F60 / FPS
    b1 = even[(t[even] >= o - fi_) & (t[even] <= o + fi_)]
    b2 = even[(t[even] >= o + window - fo_) & (t[even] <= o + window + fi_)]
    step_p = _step_deg(rot, in_win, pat)
    bnd_p = max(_step_deg(rot, b1, pat), _step_deg(rot, b2, pat))

    pen_nf = max(box_p, gnd_p) * 100.0
    fails = []
    if pen_nf > T9C_PEN_CM:
        fails.append('non-finger penetration %.1f cm' % pen_nf)
    if touch['min'] < T9D_TOUCH:
        fails.append('hand touch %d frames (need %d)' % (touch['min'], T9D_TOUCH))
    if dstep_max > T9C_DELTA_STEP_CM:
        fails.append('Delta step %.1f cm/frame' % dstep_max)
    metrics = dict(pen_box_cm=box_p * 100.0, pen_ground_cm=gnd_p * 100.0, pen_patch_cm=pen_nf,
                   pen_orig_box_cm=box_o * 100.0, pen_orig_ground_cm=gnd_o * 100.0,
                   pen_orig_cm=max(box_o, gnd_o) * 100.0,
                   pen_finger_cm=max(fb_p, fg_p) * 100.0,
                   hand_err_cm=hand_err, hand=hand_z, touch=touch,
                   dy_max_cm=float(np.max(np.abs(delta_v[:, 1])) * 100.0),
                   dz_max_cm=float(np.max(np.abs(delta_v[:, 2])) * 100.0),
                   dstep_max_cm=dstep_max, delta_outside_m=d_out, rot_outside_deg=rot_out,
                   trans_reduce_err_mm=trans_err * 10.0, rot_reduce_err_deg=rot_err,
                   step_patch_deg=step_p, boundary_patch_deg=bnd_p, n_patched=len(pat),
                   n_keys=sum(len(v) for v in res['rot_tracks'].values()) + len(res['trans_keys']),
                   fails=fails, fail=bool(fails))
    arrays = dict(frames=F60, Pf=Pf, Po=Po, win=win, body=res['body'])
    return metrics, arrays


def source_steps(rig, m, dump, seq, res, o, window, tm):
    """Largest angle per 60 Hz frame (deg) of each patched bone: patch (final tracks) and retargeted WD2 source
    (wd2_pose_at through the time map, compensated to the dump's Reference, no lift / IK / crossfade)."""
    lo = max(0, int(res['win'][0]) - 1)
    hi = min(len(seq.frames) - 1, int(res['win'][-1]) + 1)
    frames = np.arange(lo, hi + 1)
    F60 = np.asarray(seq.frames, int)
    pat = res['patched']
    patch = np.stack([sample_track_rot(res['rot_tracks'][ai], F60)[frames] for ai in pat], axis=1)
    rot_t = rot_tracks(rig, dump)
    patset = set(pat)
    src = np.zeros((len(frames), len(pat), 4))
    for i, f in enumerate(frames):
        _, wt = wd2_pose_at(rig, m, float(tm(np.array([f / FPS]))[0]))
        comp = compensated_rotations(rig, rot_t, int(f), wt, patset)
        for k, ai in enumerate(pat):
            src[i, k] = comp[ai]
    def steps(arr):
        if len(arr) < 2:
            return np.zeros(len(pat))
        return np.max(angle_deg(arr[1:], arr[:-1]), axis=0)
    sp = steps(patch)
    ss = steps(src)
    return dict(bones=[rig.ac[ai]['name'] if 'name' in rig.ac[ai] else _bone_name(rig, ai) for ai in pat],
                patch=sp, source=ss, patch_max=float(sp.max()), source_max=float(ss.max()))


def write_report_t9d(path, rows, t9c_rows, before_rows, offsets, jag, play_model_path, notes_path=OBSERVATIONS):
    n_on = sum(1 for r in rows if not r['metrics']['fail'])
    lines = ['# T9d: WD2 catpass (kong) onto ACU SkipOver clips: root lift, hand IK, foot clamp', '',
             'Generated by `v2/pipeline/bake2.py` (bake_all_t9d). Times in seconds of the clip; heights in cm; '
             'angles in degrees. Play model: ' + (play_model_path if os.path.exists(play_model_path)
                                                  else 'none (fallback windows)') + '.', '',
             '## Method (T9d)', '',
             '- Offsets (median over the original SkipOver dumps): Hand %.3f m, Foot %.3f m, FootThumb1 %.3f m.'
             % (offsets['Hand'], offsets['Foot'], offsets['FootThumb1']),
             '- Root lift: required lift per window frame L(f) = max(0, max_k(surface(y_k) + 0.01 - z_k)) over the '
             'non-finger body points of the un-faded patch (surface = obstacle top inside [near_y, far_y], ground '
             'otherwise). Upper envelope D = max(L, gauss(D, sigma %g frames)), %d iterations (numpy Gaussian, '
             'no scipy). Taper: smoothstep 0 at the window ends to 1 after %d frames; a zone with L > 0 is not tapered '
             '(reported). Delta = (0, 0, D) on the Reference position; zero outside the window.'
             % (T9D_LIFT_SIGMA, T9D_LIFT_ITER, T9D_TAPER),
             '- Hand IK: target per hand = its position at the first plant frame after the lift, z = top + Hand offset, '
             'y clamped to [near + %.2f, far - %.2f]. Weight 1 while the shoulder-to-target distance <= %.2f x arm '
             'length, %d-frame ramp in before the plant, %d-frame fade out from the first frame out of reach or after the '
             'plant.' % (T9C_IK_MARGIN, T9C_IK_MARGIN, T9D_REACH, T9D_RAMP, T9D_RAMP),
             '- Foot clamp: where the lower of Foot / FootThumb1 is below its support plus its offset, the Foot joint is '
             'raised by the deficit (IK, end world rotation kept), %d-frame ramps, every window frame.' % T9D_RAMP,
             '- Window crossfade as before; tracks rebuilt over the window plus one frame each side (exact outside).',
             '- PASS = non-finger penetration <= %.0f cm (box and ground, window) AND each hand within %.0f cm of its target '
             'for at least %d consecutive 60 Hz frames from the first plant frame (the touch) AND largest Delta change '
             '<= %.0f cm per 60 Hz frame. Fingers are reported separately and are not in PASS.'
             % (T9C_PEN_CM, T9C_HAND_TOL_CM, T9D_TOUCH, T9C_DELTA_STEP_CM), '',
             '## Before (T9b: rotations only)', ''] + _table_before(before_rows) + [
             '', '## T9c (root solve, for comparison)', ''] + _table_after(t9c_rows) + [
             '', '## T9d', ''] + _table_t9d(rows) + ['',
             'Enabled: %d of %d. Enabled clips: %s.'
             % (n_on, len(rows), ', '.join(r['name'] for r in rows if not r['metrics']['fail']) or 'none'), '',
             '## Jaggedness (largest angle per 60 Hz frame, patch vs retargeted WD2 source)', ''] + _jag_lines(jag) + [
             '', '## Failure reasons', '']
    for r in rows:
        why = '; '.join(r['metrics']['fails']) if r['metrics']['fails'] else 'none (PASS)'
        lines.append('- %s: %s' % (r['name'], why))
    lines += ['', '## Observations', '']
    if os.path.exists(notes_path):
        with open(notes_path, encoding='utf-8') as f:
            lines.append(f.read().rstrip())
    else:
        lines.append('(no observations file)')
    lines.append('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _table_t9d(rows):
    lines = ['| clip | o / window (s) | plant (s) | non-finger pen box / ground (cm) | finger pen (cm) | '
             'hand touch L / R (frames) | hand to target max (cm) | lift max (cm) | max Delta step (cm/frame) | '
             'taper start / end | status | enabled |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        mt = r['metrics']
        pl = r['plant_s']
        plant = 'none' if pl is None else '%.2f to %.2f' % pl
        tc = mt['touch']
        tp = r['solve_info']['lift_taper']
        herr = 'n/a' if mt['hand_err_cm'] is None else '%.1f' % mt['hand_err_cm']
        lines.append('| %s | %.2f / %.2f | %s | %.1f / %.1f | %.1f | %d / %d | %s | %.1f | %.2f | %s / %s | %s | %s |'
                     % (r['name'], r['o'], r['window'], plant, mt['pen_box_cm'], mt['pen_ground_cm'],
                        mt['pen_finger_cm'], tc.get('LeftHand', 0), tc.get('RightHand', 0), herr,
                        r['solve_info']['lift_max_cm'], mt['dstep_max_cm'],
                        'untapered' if tp['start_untapered'] else 'tapered',
                        'untapered' if tp['end_untapered'] else 'tapered',
                        'FAIL' if mt['fail'] else 'PASS', 'yes' if not mt['fail'] else 'no'))
    return lines


def _jag_lines(jag):
    """jag: {clip name: source_steps dict}. Per-clip maxima and the five worst bones over all clips."""
    lines = ['| clip | patch max step (deg) | source max step (deg) |', '|---|---|---|']
    worst_p = {}
    worst_s = {}
    for name, j in jag.items():
        lines.append('| %s | %.1f | %.1f |' % (name, j['patch_max'], j['source_max']))
        for b, p, s in zip(j['bones'], j['patch'], j['source']):
            worst_p[b] = max(worst_p.get(b, 0.0), float(p))
            worst_s[b] = max(worst_s.get(b, 0.0), float(s))
    lines += ['', 'Five worst bones (max over clips), patch: ' + ', '.join(
        '%s %.1f (source %.1f)' % (b, v, worst_s[b]) for b, v in sorted(worst_p.items(), key=lambda kv: -kv[1])[:5]) + '.',
              'Five worst bones (max over clips), source: ' + ', '.join(
        '%s %.1f (patch %.1f)' % (b, v, worst_p[b]) for b, v in sorted(worst_s.items(), key=lambda kv: -kv[1])[:5]) + '.']
    return lines


def bake_all_t9d(write=True):
    """All ten SkipOver clips with T9d. With write: patches, mapping.json (enabled only for PASS), PNGs, report.md."""
    os.makedirs(PATCH_OUT, exist_ok=True)
    os.makedirs(BAKE_OUT, exist_ok=True)
    rig = R.Rig()
    ev_json = load_events()
    m = wd2.load(ev_json['stem'])
    wd2_ev = dict(takeoff=ev_json['takeoff'], hand_plant_start=ev_json['hand_plant_start'],
                  hand_plant_end=ev_json['hand_plant_end'], landing=ev_json['landing'],
                  static_until=ev_json['static_until'], duration=float(m['dur']))
    model = load_play_model()
    measured = load_measured()
    clips = []
    for handle, name, dump in targets():
        seq = acuclip.pose_sequence(dump, step=1)
        ev = contacts.events(seq, dump)
        clips.append(dict(handle=handle, name=name, dump=dump, seq=seq, ev=ev, acu=acu_seconds(ev),
                          L=float(dump['Length'])))
    offsets = measure_contact_offsets([(c['name'], c['dump'], c['seq'], c['ev']) for c in clips])
    before_rows = bake_all(write=False)
    t9c_rows, _b, _o = bake_all_t9c(write=False)
    rows = []
    mapping = []
    jag = {}
    for c in clips:
        o, window, source = play_for(c['name'], c['L'], model, measured)
        res = bake_t9d(rig, m, c['dump'], c['seq'], c['acu'], c['ev'], wd2_ev, o, window, offsets)
        metrics, arrays = validate_t9d(rig, c['dump'], c['seq'], res, o, window, c['ev']['obstacle'])
        pl = None
        if res['plant'].size:
            pl = (float(res['plant'][0]) / FPS, float(res['plant'][-1]) / FPS)
        j = source_steps(rig, m, c['dump'], c['seq'], res, o, window, res['time_map'])
        jag[c['name']] = j
        rows.append(dict(name=c['name'], handle=c['handle'], length=c['L'], o=o, window=window, source=source,
                         metrics=metrics, constraint_info=res['constraint_info'], solve_info=res['solve_info'],
                         plant_s=pl))
        fn = '%s.baked.json' % c['handle']
        if write:
            write_patch_t9c(os.path.join(PATCH_OUT, fn), rig, res['rot_tracks'], res['trans_keys'], c['L'])
            mapping.append(mapping_entry_t9c(c['handle'], c['name'], fn, not metrics['fail']))
            plot_clip_t9c(os.path.join(BAKE_OUT, c['name'] + '.png'), c['name'], rig, res, arrays, metrics, o,
                          window, c['ev']['obstacle'])
        print('%-44s o %.2f win %.2f  pen(nf) %.1f cm  touch %s  dstep %.2f  lift %.1f cm  jag patch %.1f src %.1f  %s' % (
            c['name'], o, window, metrics['pen_patch_cm'], metrics['touch'], metrics['dstep_max_cm'],
            res['solve_info']['lift_max_cm'], j['patch_max'], j['source_max'],
            'FAIL (%s)' % '; '.join(metrics['fails']) if metrics['fail'] else 'PASS'))
    if write:
        with open(os.path.join(PATCH_OUT, 'mapping.json'), 'w', encoding='utf-8') as f:
            json.dump({'clips': mapping}, f, indent=1)
        write_report_t9d(os.path.join(BAKE_OUT, 'report.md'), rows, t9c_rows, before_rows, offsets, jag, PLAY_MODEL)
    return rows, t9c_rows, before_rows, offsets, jag


# ------------------------------------------------------------------------------------------------ 9. T9e
# Orchestrator decisions (T9e), default method:
#   (a) the lift sees the non-finger body WITHOUT Foot / FootThumb1 (both sides); feet are handled by the foot clamp IK only;
#   (b) two-sided lift: L(f) signed (not clamped at 0), upper bound U(f) on the plant (reach of both shoulders to the
#       hand targets within T9E_REACH x arm length, +-3-frame relaxation), preferred D0 rule, smoothing that keeps
#       D >= L and D <= U where feasible, taper towards 0 at the window ends (forced to L where L > 0);
#   (c) per-clip fades: fi = min(0.20 s, 0.30 w), fo = min(0.25 s, 0.35 w); pure-WD2 time reported;
#   (d) PASS unchanged (validate_t9d): non-finger penetration incl. feet <= 5 cm, touch >= 4 frames per hand, Delta step <= 3 cm.
# T9d (bake_t9d) stays behind --t9d; T9c behind --t9c; T9b behind --t9b.
T9E_FADE_IN_MAX = 0.20
T9E_FADE_IN_FRAC = 0.30
T9E_FADE_OUT_MAX = 0.25
T9E_FADE_OUT_FRAC = 0.35
T9E_RELAX_FRAMES = 3         # frames over which the upper bound is relaxed outside the plant
T9E_RELAX_M = 10.0           # m, relaxation at the end of the ramp (effectively no bound)
T9E_REACH = 0.95             # both shoulders within this fraction of the arm length of their hand targets
T9E_FEET = ('LeftFoot', 'LeftFootThumb1', 'RightFoot', 'RightFootThumb1')
T9E_SIDES = (('Left', 'LeftHand'), ('Right', 'RightHand'))


def fades_for(window):
    """(fade_in, fade_out) in seconds for a play window of `window` seconds."""
    return (min(T9E_FADE_IN_MAX, T9E_FADE_IN_FRAC * float(window)),
            min(T9E_FADE_OUT_MAX, T9E_FADE_OUT_FRAC * float(window)))


def tail_fades(window, fi=None, fo=None, tail=TAIL_ORIG):
    """Validated (fade_in, fade_out, span) for the T9e variants. None = the default of fades_for(window).

    The fade-out returns the patch to the ORIGINAL pose, so it only exists with tail 'orig'; with tail 'wd2' or 'hold'
    the default fade-out is 0 and a non-zero one is refused (it would have no target). span = clip seconds the patch
    covers after o: the window, or None for 'wd2' (up to the clip end, decided by the caller).
    """
    if tail not in TAILS:
        raise ValueError('tail must be one of %s, got %r' % (TAILS, tail))
    fi_def, fo_def = fades_for(window)
    fi = fi_def if fi is None else float(fi)
    fo = (fo_def if tail == TAIL_ORIG else 0.0) if fo is None else float(fo)
    if fi < 0 or fo < 0:
        raise ValueError('fades must be >= 0 (fi %.3f, fo %.3f)' % (fi, fo))
    if tail != TAIL_ORIG and fo > 0:
        raise ValueError('fade-out %.3f s needs tail "orig" (it returns to the ACU pose); use fade-out 0 with tail %r'
                         % (fo, tail))
    if fi + fo > float(window) + 1e-12:
        raise ValueError('fade-in + fade-out (%.3f + %.3f s) longer than the window %.3f s' % (fi, fo, window))
    return fi, fo


def lift_points(rig, body):
    """Body bones the T9e lift sees: no finger joints and no feet (Foot / FootThumb1, both sides)."""
    feet = {rig.ac_by_hash[zlib.crc32(n.encode())] for n in T9E_FEET}
    fing = set(finger_bones(rig, body))
    return [bi for bi in body if bi not in feet and bi not in fing]


def lift_need_signed(P, ob):
    """L per frame = max_k(surface(y_k) + clearance - z_k), NOT clamped at 0 (negative = room to lower the body)."""
    P = np.asarray(P, float)
    y = P[..., 1]
    z = P[..., 2]
    surf = np.where((y >= ob['near_y']) & (y <= ob['far_y']), ob['height'], 0.0)
    return (surf + T9D_LIFT_CLEAR - z).max(axis=1)


def reach_upper(sh, tgt, R):
    """Largest lift dz for which |tgt - (sh + dz * z_hat)| <= R, per row (exact: the set of dz is an interval
    [vz - s, vz + s] with s = sqrt(R^2 - vx^2 - vy^2), v = tgt - sh). sh [n, 3], tgt [3], R [n].
    Returns -inf where R cannot reach the target at any height (lateral distance larger than R)."""
    sh = np.asarray(sh, float)
    v = np.asarray(tgt, float)[None, :] - sh
    s2 = np.asarray(R, float) ** 2 - v[:, 0] ** 2 - v[:, 1] ** 2
    return np.where(s2 < 0, -np.inf, v[:, 2] + np.sqrt(np.maximum(s2, 0.0)))


def d0_rule(L, U):
    """Preferred lift (orchestrator rule): 0 if L <= 0 <= U; L if L > 0; U if U < 0 and U >= L; otherwise L
    (conflict: penetration wins, the touch fails there)."""
    L = np.asarray(L, float)
    U = np.asarray(U, float)
    return np.where(L > 0, L,
                    np.where((U < 0) & (U >= L), U,
                             np.where((L <= 0) & (U >= 0), 0.0, L)))


def solve_lift_t9e(L, U, ramp_edges=False):
    """D over the window frames. D0 = d0_rule; then D = clip(gauss(D, sigma), L, upper) for T9D_LIFT_ITER iterations,
    where upper = U where U >= L and +inf where U < L (penetration wins there). Then the end taper towards 0 with
    D >= L and D <= upper kept (the frames where that forces a value are counted). Returns (D, info).

    ramp_edges=True (used when the fades or the tail are not the default): the taper is applied as a plain ramp, so
    D goes 0 -> D smoothly at both window ends and nothing is forced to L there. The need that the ramp does not
    cover is then left as penetration (reported by validate_t9d) instead of a step in Delta."""
    L = np.asarray(L, float)
    U = np.asarray(U, float)
    n = len(L)
    if n == 0:
        return np.zeros(0), dict(conflict=0, forced_start=0, forced_end=0, reach_limited=0, max_D_cm=0.0,
                                 min_D_cm=0.0, max_step_cm=0.0)
    upper = np.where(U >= L, U, np.inf)
    D = d0_rule(L, U)
    for _ in range(T9D_LIFT_ITER):
        D = np.clip(_gauss_smooth(D, T9D_LIFT_SIGMA), L, upper)
    j = np.arange(n, dtype=float)

    def ss(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    taper = np.minimum(ss(j / T9D_TAPER), ss((n - 1 - j) / T9D_TAPER))
    if ramp_edges:
        Dt = D * taper
        forced = np.zeros(n, bool)
    else:
        Dt = np.clip(D * taper, L, upper)
        forced = np.abs(Dt - D * taper) > 1e-12
    start_zone = j < T9D_TAPER
    end_zone = j > n - 1 - T9D_TAPER
    info = dict(conflict=int(np.sum(U < L)),
                forced_start=int(np.sum(forced & start_zone)),
                forced_end=int(np.sum(forced & end_zone)),
                reach_limited=int(np.sum(np.isfinite(U) & (U >= L) & (np.abs(Dt - U) < 1e-9))),
                max_D_cm=float(Dt.max() * 100.0), min_D_cm=float(Dt.min() * 100.0),
                max_step_cm=float(np.max(np.abs(np.diff(Dt))) * 100.0) if n > 1 else 0.0)
    return Dt, info


def jaggedness_t9e(rig, m, dump, seq, res, tm):
    """Largest angle per 60 Hz frame (deg), per patched bone, over the window +-1 frame: final patch, retargeted WD2
    source (wd2_pose_at through the time map, compensated, no lift / IK / crossfade) and the original dump."""
    lo = max(0, int(res['win'][0]) - 1)
    hi = min(len(seq.frames) - 1, int(res['win'][-1]) + 1)
    frames = np.arange(lo, hi + 1)
    F60 = np.asarray(seq.frames, int)
    pat = res['patched']
    patch = np.stack([sample_track_rot(res['rot_tracks'][ai], F60)[frames] for ai in pat], axis=1)
    orig = np.stack([np.asarray(seq.local_rot, float)[frames, ai] for ai in pat], axis=1)
    rot_t = rot_tracks(rig, dump)
    patset = set(pat)
    src = np.zeros((len(frames), len(pat), 4))
    for i, f in enumerate(frames):
        _, wt = wd2_pose_at(rig, m, float(tm(np.array([f / FPS]))[0]))
        comp = compensated_rotations(rig, rot_t, int(f), wt, patset)
        for k, ai in enumerate(pat):
            src[i, k] = comp[ai]

    def steps(arr):
        if len(arr) < 2:
            return np.zeros(len(pat))
        return np.max(angle_deg(arr[1:], arr[:-1]), axis=0)
    sp, ss_, so = steps(patch), steps(src), steps(orig)
    return dict(bones=[_bone_name(rig, ai) for ai in pat], patch=sp, source=ss_, orig=so,
                patch_max=float(sp.max()), source_max=float(ss_.max()), orig_max=float(so.max()))


def bake_t9e(rig, m, dump, seq, acu_s, ev, wd2_ev, o, window, offsets, fi=None, fo=None, tail=TAIL_ORIG):
    """T9e: two-sided lift (no feet), hand IK (targets fixed at the first plant frame, z = top + Hand offset), foot
    clamp IK, per-clip crossfade. Same result keys as bake_t9d, plus fades and pure-WD2 times.

    Variant options (defaults = the shipped T9e, bit for bit):
      fi, fo  fade-in / fade-out lengths in s (None = fades_for(window)); fo must be 0 unless tail == 'orig'.
      tail    'orig': fade back to the ACU pose over [o + window - fo, o + window] (default).
              'wd2' : no fade-out; the patch (time-mapped WD2 motion, lift, IK) runs from o to the clip end; the WD2
                      source holds its last pose once it runs out (wd2_pose_at clamps the time).
              'hold': as the default up to o + window, then every patched rotation and the Reference position are
                      frozen at their values at o + window until the clip end.
    Envelopes: when the fades or the tail are not the default, the lift taper is a plain ramp at both ends (ramp_edges),
    so the lift does not step at the window edges. The hand IK and foot clamp weights already ramp (T9D_RAMP frames).
    """
    L_ = float(dump['Length'])
    ob = ev['obstacle']
    F60 = np.asarray(seq.frames, int)
    nf = int(F60[-1])
    fi_def, fo_def = fades_for(window)
    fi, fo = tail_fades(window, fi, fo, tail)
    ramp_edges = not (tail == TAIL_ORIG and fi == fi_def and fo == fo_def)
    ik_ramp = T9D_TAPER if ramp_edges else T9D_RAMP     # IK in / out ramps: longer when the fades are not the default
    span = float(window)      # envelopes (lift, IK, foot clamp) stay on the window for every tail (see docstring)
    base = bake_clip(rig, m, dump, acu_s, wd2_ev, o, window)
    pat = base['patched']
    tm = base['time_map']
    orig = np.asarray(seq.local_rot, float)
    ref0 = np.asarray(seq.local_pos[:, 0], float)
    w60 = patch_weight(F60 / FPS, o, window, fi, fo, tail)
    p60 = patch_rotations60(base['raw'], pat, F60)
    patch = orig.copy()
    blend = orig.copy()
    for ai in pat:
        patch[:, ai] = p60[ai]
        blend[:, ai] = mix_rot(orig[:, ai], p60[ai], w60)

    body = sorted(set(rig.map.values()))
    fingers = set(finger_bones(rig, body))
    body_nf = [bi for bi in body if bi not in fingers]
    lift_pts = lift_points(rig, body)
    win = window_frames(o, span, nf)
    hp_iv = clip_interval_for_wd2(tm, wd2_ev['hand_plant_start'], wd2_ev['hand_plant_end'], L_)
    plant = plant_in_window(hp_iv, win)
    ac = lambda name: rig.ac_by_hash[zlib.crc32(name.encode())]
    disp = acuclip.markers(dump, F60)['disp']

    # 1. signed lift from the blended pose; upper bound from the un-lifted un-faded patch (the lift moves no x or y)
    P_blend0 = acuclip.world_positions(composed(seq, blend, ref0), dump)
    Lw = lift_need_signed(P_blend0[win][:, lift_pts], ob)
    P_patch0 = acuclip.world_positions(composed(seq, patch, ref0), dump)
    n_win = len(win)
    U_win = np.full(n_win, np.inf)
    hand_targets = {}
    if plant.size:
        p0 = int(plant[0])
        fr = plant
        hi_sides = []
        for side, h in T9E_SIDES:
            ua, la, ha = ac(side + 'Arm'), ac(side + 'ForeArm'), ac(h)
            p = P_patch0[p0, ha]
            tgt = np.array([p[0],
                            min(max(p[1], float(ob['near_y']) + T9C_IK_MARGIN), float(ob['far_y']) - T9C_IK_MARGIN),
                            float(ob['height']) + offsets['Hand']])
            hand_targets[h] = tgt
            sh = P_patch0[fr, ua]
            arm = (np.linalg.norm(P_patch0[fr, la] - P_patch0[fr, ua], axis=1)
                   + np.linalg.norm(P_patch0[fr, ha] - P_patch0[fr, la], axis=1))
            hi_sides.append(reach_upper(sh, tgt, T9E_REACH * arm))
        U_plant = np.min(np.stack(hi_sides), axis=0)
        pos = plant - win[0]
        U_win[pos] = U_plant
        for d in range(1, T9E_RELAX_FRAMES + 1):
            for jj, edge in ((int(pos[0]) - d, U_plant[0]), (int(pos[-1]) + d, U_plant[-1])):
                if 0 <= jj < n_win and np.isinf(U_win[jj]) and U_win[jj] > 0:
                    U_win[jj] = edge + T9E_RELAX_M * d / T9E_RELAX_FRAMES if np.isfinite(edge) else edge
    D_w, linfo = solve_lift_t9e(Lw, U_win, ramp_edges=ramp_edges)
    Dfull = np.zeros(len(F60))
    Dfull[win] = D_w
    delta = np.zeros((len(F60), 3))
    delta[:, 2] = Dfull
    ref1 = ref0 + delta
    P1 = acuclip.world_positions(composed(seq, patch, ref1), dump)   # post-lift, pre-IK
    ik_in = composed(seq, patch, ref1)

    # 2. hand IK: fixed target, weight 1 while both shoulder reach holds (post-lift pose), 3-frame ramps
    targets = {}
    reach_info = {}
    if plant.size:
        p0 = int(plant[0])
        pend = int(plant[-1])
        for side, h in T9E_SIDES:
            ua, la, ha = ac(side + 'Arm'), ac(side + 'ForeArm'), ac(h)
            tgt = hand_targets[h]
            frr = np.arange(p0, pend + 1)
            d = np.linalg.norm(tgt[None, :] - P1[frr, ua], axis=1)
            arm1 = (np.linalg.norm(P1[frr, la] - P1[frr, ua], axis=1)
                    + np.linalg.norm(P1[frr, ha] - P1[frr, la], axis=1))
            reach = d <= T9E_REACH * arm1
            wh, u = _ik_weights(len(F60), win, p0, pend, reach, ramp=ik_ramp)
            reach_info[h] = dict(first_unreachable=int(u) if u <= pend else None, reach_frames=int(reach.sum()))
            targets[h] = [(int(f), tgt, float(wh[f])) for f in np.nonzero(wh > 0)[0]]

    # 3. foot clamp: the lower of Foot / FootThumb1 below its support (+ offset) is raised, every window frame
    foot_info = {}
    for side in ('Left', 'Right'):
        foot, toe = side + 'Foot', side + 'FootThumb1'
        need = np.zeros(len(F60))
        for f in win:
            best = 0.0
            for name, off in ((foot, offsets['Foot']), (toe, offsets['FootThumb1'])):
                y, z = P1[f, ac(name), 1], P1[f, ac(name), 2]
                surf = float(ob['height']) if float(ob['near_y']) <= y <= float(ob['far_y']) else 0.0
                best = max(best, surf + off - z)
            need[f] = best
        pos_f = P1[:, ac(foot)]
        pos_idx = np.nonzero(need > 0)[0]
        foot_info[foot] = dict(frames=int(pos_idx.size), max_raise_cm=float(need.max() * 100.0))
        if pos_idx.size == 0:
            continue
        wf = blend_mod_fade(len(F60), _runs(pos_idx), ik_ramp)
        for f in np.nonzero(wf > 0)[0]:
            k = pos_idx[np.argmin(np.abs(pos_idx - f))]
            tgt = pos_f[f] + np.array([0.0, 0.0, need[k]])
            targets.setdefault(foot, []).append((int(f), tgt, float(wf[f])))

    if targets:
        root_rot = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(F60), 1))
        ik_out = ik.solve_sequence(ik_in, root_rot=root_rot, root_pos=disp, targets=targets)
        changed = np.nonzero(np.any(np.abs(ik_out.local_rot - ik_in.local_rot) > 1e-12, axis=(0, 2)))[0]
        bad = sorted(set(int(c) for c in changed) - set(pat))
        if bad:
            raise ValueError('IK changed bones without a rotation track: %s' % [rig.ac[b]['name'] for b in bad])
        patch_ik = np.asarray(ik_out.local_rot, float)
    else:
        patch_ik = patch

    # 4. per-clip crossfade and tracks (outside the window: original)
    final = orig.copy()
    for ai in pat:
        final[:, ai] = mix_rot(orig[:, ai], patch_ik[:, ai], w60)
    if tail == TAIL_HOLD and int(win[-1]) < nf:
        # hold: the pose (rotations and Reference position) at the window end stays until the clip end
        e_idx = int(win[-1])
        final[e_idx + 1:, pat] = final[e_idx, pat]
        ref1[e_idx + 1:] = ref1[e_idx]
        delta[e_idx + 1:] = delta[e_idx]
    lo = max(0, int(win[0]) - 1)
    hi = nf if tail != TAIL_ORIG else min(nf, int(win[-1]) + 1)
    rot_tracks_out = {}
    for ai in pat:
        ok = dumps.track(dump, rig.ac[ai]['hash'], 0) or []
        rot_tracks_out[ai] = rotation_track(ok, final[:, ai], lo, hi)
    ref_keys = dumps.track(dump, REF_TID, 3) or []
    trans_keys = translation_track(ref_keys, ref1, lo, hi)

    c0 = o + fi
    c1 = o + span - fo
    pure = dict(clip=(c0, c1), wd2=(float(tm(np.array([c0]))[0]), float(tm(np.array([c1]))[0])))
    solve_info = dict(iterations=0, converged=bool(linfo['conflict'] == 0), max_residual={}, max_penetration=0.0,
                      n_violations=int(linfo['conflict']), active=[],
                      lift_max_cm=linfo['max_D_cm'], lift_min_cm=linfo['min_D_cm'],
                      lift_step_cm=linfo['max_step_cm'], lift_conflict=linfo['conflict'],
                      lift_reach_limited=linfo['reach_limited'],
                      lift_taper=dict(start_untapered=linfo['forced_start'] > 0,
                                      end_untapered=linfo['forced_end'] > 0),
                      lift_frames=int(np.sum(Lw > 0)))
    return dict(patched=pat, rot_tracks=rot_tracks_out, trans_keys=trans_keys, final=final, ref_new=ref1,
                delta=delta, win=win, plant=plant, plant_j=np.nonzero(np.isin(win, plant))[0], hp_iv=hp_iv,
                targets=hand_targets, body=body, body_nf=body_nf, fingers=sorted(fingers),
                lift_points=lift_pts, solve_info=solve_info,
                constraint_info=dict(pin_frames=0, pin_frames_wanted=0, reach=reach_info, foot=foot_info),
                fades=(fi, fo), pure=pure, L=Lw, U=U_win, D=D_w,
                time_map=tm, o=float(o), window=float(window), disp=disp, tail=tail, span_end=float(o + span))


def blend_mod_fade(n, runs, ramp):
    """blend.fade_weights for runs given as (first, last) pairs (wrapper so the T9e code reads simply)."""
    return blend.fade_weights(n, [(int(a), int(b)) for a, b in runs], ramp)


def _table_t9e(rows):
    lines = ['| clip | o / window (s) | fades in / out (s) | pure WD2 part: clip s (WD2 s) | non-finger pen box / ground (cm) | '
             'finger pen (cm) | touch L / R (frames) | hand to target max (cm) | lift min .. max (cm) | '
             'conflict / forced start / forced end / reach-limited frames | max Delta step (cm/frame) | status | enabled |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        mt = r['metrics']
        si = r['solve_info']
        tc = mt['touch']
        herr = 'n/a' if mt['hand_err_cm'] is None else '%.1f' % mt['hand_err_cm']
        pu = r['pure']
        lines.append('| %s | %.2f / %.2f | %.2f / %.2f | %.2f-%.2f (%.2f-%.2f) | %.1f / %.1f | %.1f | %d / %d | %s | '
                     '%.1f .. %.1f | %d / %s / %s / %d | %.2f | %s | %s |'
                     % (r['name'], r['o'], r['window'], r['fades'][0], r['fades'][1],
                        pu['clip'][0], pu['clip'][1], pu['wd2'][0], pu['wd2'][1],
                        mt['pen_box_cm'], mt['pen_ground_cm'], mt['pen_finger_cm'],
                        tc.get('LeftHand', 0), tc.get('RightHand', 0), herr,
                        si['lift_min_cm'], si['lift_max_cm'], si['lift_conflict'],
                        'forced' if si['lift_taper']['start_untapered'] else 'tapered',
                        'forced' if si['lift_taper']['end_untapered'] else 'tapered', si['lift_reach_limited'],
                        mt['dstep_max_cm'], 'FAIL' if mt['fail'] else 'PASS', 'yes' if not mt['fail'] else 'no'))
    return lines


def _jag_lines_t9e(jag):
    lines = ['| clip | patch max step (deg) | WD2 source max step (deg) | original dump max step (deg) |', '|---|---|---|---|']
    worst = {'patch': {}, 'source': {}, 'orig': {}}
    for name, j in jag.items():
        lines.append('| %s | %.1f | %.1f | %.1f |' % (name, j['patch_max'], j['source_max'], j['orig_max']))
        for b, p, s, o_ in zip(j['bones'], j['patch'], j['source'], j['orig']):
            worst['patch'][b] = max(worst['patch'].get(b, 0.0), float(p))
            worst['source'][b] = max(worst['source'].get(b, 0.0), float(s))
            worst['orig'][b] = max(worst['orig'].get(b, 0.0), float(o_))
    for key, label in (('patch', 'patch'), ('source', 'WD2 source'), ('orig', 'original dump')):
        top = sorted(worst[key].items(), key=lambda kv: -kv[1])[:5]
        lines.append('')
        lines.append('Five worst bones, %s (max over clips): %s.' % (
            label, ', '.join('%s %.1f' % (b, v) for b, v in top)))
    return lines


def write_gif_t9e(path, rig, res, arrays, ob, o, window, tm):
    """Animated GIF, side view (entity forward Y, height Z) in world space: obstacle, original (grey) and patch
    (colour), about 20 fps over [o - 0.2, o + window + 0.2] clamped to the clip."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter
    from matplotlib.patches import Rectangle

    F = arrays['frames']
    Pf, Po, body = arrays['Pf'], arrays['Po'], arrays['body']
    f0 = max(0, int(round((o - 0.2) * FPS)))
    f1 = min(int(F[-1]), int(round((o + window + 0.2) * FPS)))
    sel = np.arange(f0, f1 + 1, 3)                       # 60 Hz / 3 = 20 fps
    bset = set(body)
    edges = [(ai, rig.ac[ai]['parent']) for ai in body if rig.ac[ai]['parent'] in bset]
    ys = Pf[sel][:, body, 1]
    zmax = max(float(ob['height']) + 0.6, float(Pf[sel][:, body, 2].max()) + 0.1)
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=80)
    writer = PillowWriter(fps=20)
    with writer.saving(fig, path, dpi=80):
        for f in sel:
            ax.cla()
            ax.add_patch(Rectangle((ob['near_y'], 0.0), ob['far_y'] - ob['near_y'], ob['height'],
                                   facecolor='#8b6f47', alpha=0.35, edgecolor='#5c4a2e'))
            ax.axhline(0.0, color='#777', lw=0.8)
            for a, p in edges:
                ax.plot([Po[f, a, 1], Po[f, p, 1]], [Po[f, a, 2], Po[f, p, 2]], color='#bdbdbd', lw=1.6)
            for a, p in edges:
                ax.plot([Pf[f, a, 1], Pf[f, p, 1]], [Pf[f, a, 2], Pf[f, p, 2]], color='#1f77b4', lw=2.0)
            ax.set_xlim(float(ys.min()) - 0.6, float(ys.max()) + 0.6)
            ax.set_ylim(-0.05, zmax)
            ax.set_aspect('equal')
            ax.set_xlabel('entity forward Y (m)', fontsize=7)
            ax.set_ylabel('Z (m)', fontsize=7)
            ax.tick_params(labelsize=6)
            t = f / FPS
            pure = (res['pure']['clip'][0] <= t <= res['pure']['clip'][1])
            ax.set_title('%s\nclip %.2f s (WD2 %.2f s)  %s' % (
                res['name'] if 'name' in res else '', t, float(tm(np.array([t]))[0]),
                'pure WD2' if pure else 'crossfade / original'), fontsize=7)
            writer.grab_frame()
    plt.close(fig)


def write_report_t9e(path, rows, t9d_rows, t9c_rows, before_rows, offsets, jag, play_model_path,
                     notes_path=OBSERVATIONS):
    n_on = sum(1 for r in rows if not r['metrics']['fail'])
    lines = ['# T9e: WD2 catpass (kong) onto ACU SkipOver clips: two-sided lift, hand IK, foot clamp, per-clip fades', '',
             'Generated by `v2/pipeline/bake2.py` (bake_all_t9e). Times in seconds of the clip; heights in cm; '
             'angles in degrees. Play model: ' + (play_model_path if os.path.exists(play_model_path)
                                                  else 'none (fallback windows)') + '.', '',
             '## Method (T9e)', '',
             '- Offsets (median over the original SkipOver dumps): Hand %.3f m, Foot %.3f m, FootThumb1 %.3f m.'
             % (offsets['Hand'], offsets['Foot'], offsets['FootThumb1']),
             '- Lift points: non-finger body WITHOUT Foot / FootThumb1 (both sides). Feet are handled by the foot clamp IK only.',
             '- L(f) = max_k(surface(y_k) + 0.01 - z_k), signed (not clamped at 0), on the blended pose.',
             '- Upper bound U(f) on the plant (+-%d frames relaxed linearly): the largest lift dz for which both shoulders '
             'reach their hand targets within %.2f x arm length (exact interval from the shoulder positions; hand target = '
             'x, y of the un-lifted hand at the first plant frame, y clamped, z = top + Hand offset). Off the plant U = +inf.'
             % (T9E_RELAX_FRAMES, T9E_REACH),
             '- D0 rule: 0 if L <= 0 <= U; L if L > 0; U if U < 0 and U >= L; L on conflict (U < L; penetration wins, touch fails).',
             '- Smoothing: D = clip(gauss(D, sigma %g frames), L, U) x %d, with U ignored where U < L. Taper towards 0 at the '
             'window ends with D >= L and D <= U kept (forced frames counted). Delta = (0, 0, D).'
             % (T9D_LIFT_SIGMA, T9D_LIFT_ITER),
             '- Fades per clip: fi = min(%.2f s, %.2f x window), fo = min(%.2f s, %.2f x window). Pure WD2 part = clip time '
             'where the patch weight is 1, mapped to WD2 time through the time map.'
             % (T9E_FADE_IN_MAX, T9E_FADE_IN_FRAC, T9E_FADE_OUT_MAX, T9E_FADE_OUT_FRAC),
             '- Hand IK (weight 1 while both hands keep their reach after the lift, 3-frame ramps) and foot clamp IK '
             '(lower of Foot / FootThumb1 raised to its support + offset, 3-frame ramps), applied before the crossfade.',
             '- PASS = non-finger penetration (incl. feet after foot IK) <= %.0f cm (box and ground, window) AND each hand '
             'within %.0f cm of its target for >= %d consecutive frames from the first plant frame AND largest Delta change '
             '<= %.0f cm per 60 Hz frame. Fingers are reported separately. FAIL clips have `enabled: false`.'
             % (T9C_PEN_CM, T9C_HAND_TOL_CM, T9D_TOUCH, T9C_DELTA_STEP_CM), '',
             '## T9e', ''] + _table_t9e(rows) + ['',
             'Enabled: %d of %d. Enabled clips: %s.'
             % (n_on, len(rows), ', '.join(r['name'] for r in rows if not r['metrics']['fail']) or 'none'), '',
             '## Jaggedness (largest angle per 60 Hz frame: final patch, retargeted WD2 source, original dump)', ''] + \
            _jag_lines_t9e(jag) + ['', '## Failure reasons', '']
    for r in rows:
        why = '; '.join(r['metrics']['fails']) if r['metrics']['fails'] else 'none (PASS)'
        lines.append('- %s: %s' % (r['name'], why))
    lines += ['', '## T9d (previous method, for comparison)', ''] + _table_t9d(t9d_rows)
    lines += ['', '## T9c (root solve, for comparison)', ''] + _table_after(t9c_rows)
    lines += ['', '## Before (T9b: rotations only)', ''] + _table_before(before_rows)
    lines += ['', '## Observations', '']
    if os.path.exists(notes_path):
        with open(notes_path, encoding='utf-8') as f:
            lines.append(f.read().rstrip())
    else:
        lines.append('(no observations file)')
    lines.append('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def enabled_from_mapping(path):
    """{HANDLE (no 0x, upper case): enabled} from a mapping.json: the flags a variant copies, not recomputes."""
    with open(path, encoding='utf-8') as f:
        clips = json.load(f)['clips']
    return {str(e['handle']).upper().replace('0X', ''): bool(e['enabled']) for e in clips}


def bake_all_t9e(write=True, fi=None, fo=None, tail=TAIL_ORIG, out_dir=None, enabled_from=None, compare=True):
    """All ten SkipOver clips with T9e.

    Default call: the shipped T9e. With write: patches and mapping.json (enabled = PASS) in config.PATCHES, PNGs, GIFs
    and report.md in out\\bake2.
    Variant call (out_dir given): patches and mapping.json only, written to out_dir. No plots or report. The enabled
    flags are copied from enabled_from (the baseline mapping.json). compare=False skips the T9b / T9c / T9d rows.
    The installed set (config.PATCHES) can never be the variant output directory.
    """
    if out_dir is not None and os.path.normcase(os.path.abspath(out_dir)) == os.path.normcase(os.path.abspath(PATCH_OUT)):
        raise ValueError('out_dir must not be the installed patch folder %s' % PATCH_OUT)
    patch_dir = PATCH_OUT if out_dir is None else os.path.abspath(out_dir)
    bake_dir = BAKE_OUT if out_dir is None else None
    os.makedirs(patch_dir, exist_ok=True)
    if bake_dir is not None:
        os.makedirs(bake_dir, exist_ok=True)
    rig = R.Rig()
    ev_json = load_events()
    m = wd2.load(ev_json['stem'])
    wd2_ev = dict(takeoff=ev_json['takeoff'], hand_plant_start=ev_json['hand_plant_start'],
                  hand_plant_end=ev_json['hand_plant_end'], landing=ev_json['landing'],
                  static_until=ev_json['static_until'], duration=float(m['dur']))
    model = load_play_model()
    measured = load_measured()
    clips = []
    for handle, name, dump in targets():
        seq = acuclip.pose_sequence(dump, step=1)
        ev = contacts.events(seq, dump)
        clips.append(dict(handle=handle, name=name, dump=dump, seq=seq, ev=ev, acu=acu_seconds(ev),
                          L=float(dump['Length'])))
    offsets = measure_contact_offsets([(c['name'], c['dump'], c['seq'], c['ev']) for c in clips])
    enabled = enabled_from_mapping(enabled_from) if enabled_from else None
    before_rows = t9c_rows = None
    t9d_rows = None
    if compare:
        before_rows = bake_all(write=False)
        t9c_rows, _b, _o = bake_all_t9c(write=False)
        t9d_rows = []
        for c in clips:
            o, window, source = play_for(c['name'], c['L'], model, measured)
            r9 = bake_t9d(rig, m, c['dump'], c['seq'], c['acu'], c['ev'], wd2_ev, o, window, offsets)
            met9, _a = validate_t9d(rig, c['dump'], c['seq'], r9, o, window, c['ev']['obstacle'])
            pl9 = None if not r9['plant'].size else (float(r9['plant'][0]) / FPS, float(r9['plant'][-1]) / FPS)
            t9d_rows.append(dict(name=c['name'], o=o, window=window, metrics=met9, solve_info=r9['solve_info'],
                                 plant_s=pl9, constraint_info=r9['constraint_info']))
    rows = []
    mapping = []
    jag = {}
    for c in clips:
        o, window, source = play_for(c['name'], c['L'], model, measured)
        res = bake_t9e(rig, m, c['dump'], c['seq'], c['acu'], c['ev'], wd2_ev, o, window, offsets,
                       fi=fi, fo=fo, tail=tail)
        fi_r, fo_r = res['fades']
        metrics, arrays = validate_t9d(rig, c['dump'], c['seq'], res, o, window, c['ev']['obstacle'], fi=fi_r, fo=fo_r)
        pl = None
        if res['plant'].size:
            pl = (float(res['plant'][0]) / FPS, float(res['plant'][-1]) / FPS)
        j = jaggedness_t9e(rig, m, c['dump'], c['seq'], res, res['time_map'])
        jag[c['name']] = j
        rows.append(dict(name=c['name'], handle=c['handle'], length=c['L'], o=o, window=window, source=source,
                         metrics=metrics, constraint_info=res['constraint_info'], solve_info=res['solve_info'],
                         plant_s=pl, fades=(fi_r, fo_r), pure=res['pure'], tail=tail, span_end=res['span_end']))
        fn = '%s.baked.json' % c['handle']
        if write:
            write_patch_t9c(os.path.join(patch_dir, fn), rig, res['rot_tracks'], res['trans_keys'], c['L'])
            en = (not metrics['fail']) if enabled is None else enabled[c['handle'].upper()]
            mapping.append(mapping_entry_t9c(c['handle'], c['name'], fn, en))
            if bake_dir is not None:
                plot_clip_t9c(os.path.join(bake_dir, c['name'] + '.png'), c['name'], rig, res, arrays, metrics, o,
                              window, c['ev']['obstacle'])
                res['name'] = c['name']
                write_gif_t9e(os.path.join(bake_dir, c['name'] + '.gif'), rig, res, arrays, c['ev']['obstacle'], o,
                              window, res['time_map'])
        print('%-44s pen_nf %5.1f touch %s herr %s dstep %6.2f lift %6.1f..%6.1f conflict %d  jag %.1f/%.1f/%.1f  %s' % (
            c['name'], metrics['pen_patch_cm'], metrics['touch']['min'],
            'n/a' if metrics['hand_err_cm'] is None else '%.1f' % metrics['hand_err_cm'], metrics['dstep_max_cm'],
            res['solve_info']['lift_min_cm'], res['solve_info']['lift_max_cm'], res['solve_info']['lift_conflict'],
            j['patch_max'], j['source_max'], j['orig_max'],
            'PASS' if not metrics['fail'] else 'FAIL (%s)' % '; '.join(metrics['fails'])))
    if write:
        with open(os.path.join(patch_dir, 'mapping.json'), 'w', encoding='utf-8') as f:
            json.dump({'clips': mapping}, f, indent=1)
        if bake_dir is not None:
            write_report_t9e(os.path.join(bake_dir, 'report.md'), rows, t9d_rows, t9c_rows, before_rows, offsets, jag,
                             PLAY_MODEL)
    return rows, t9d_rows, t9c_rows, before_rows, offsets, jag


def _arg(flag):
    """Value after a command-line flag, or None."""
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None


if __name__ == '__main__':
    if '--diagnostic' in sys.argv:
        bake_all(write=False, window_fn=vault_window, diagnostic=True)
    elif '--t9b' in sys.argv:
        bake_all(write=True)
    elif '--t9c' in sys.argv:
        bake_all_t9c(write=True)
    elif '--t9d' in sys.argv:
        bake_all_t9d(write=True)
    elif any(f in sys.argv for f in ('--fade-in', '--fade-out', '--tail', '--out')):
        # T9e variant: python bake2.py --fade-in 0.08 --fade-out 0.10 --tail orig --out <dir>
        out_a = _arg('--out')
        if out_a is None:
            raise SystemExit('a T9e variant needs --out <dir> (under out\\variants)')
        fi_a, fo_a = _arg('--fade-in'), _arg('--fade-out')
        bake_all_t9e(write=True, fi=None if fi_a is None else float(fi_a),
                     fo=None if fo_a is None else float(fo_a), tail=_arg('--tail') or TAIL_ORIG,
                     out_dir=out_a, enabled_from=os.path.join(config.PATCHES, 'mapping.json'), compare=False)
    else:
        bake_all_t9e(write=True)
