"""Transition analysis for Sprint SkipOver captures (WD2 patch vs original ACU), with the bake2 fade windows.

For every capture under data\\captures\\session1_20261008 and session3_20261008 that contains a Sprint SkipOver clip,
each SkipOver occurrence gets:
  t_in   start of the SkipOver clip
  t_out  start of the first clip that starts after it (exit / landing / run); None when there is none

Per frame, every derivative uses the real frame times (variable dt):
  speed    |dP/dt| of a bone's entity-local position (m/s), at the midpoint of each frame pair
  jerk     |d(velocity)/dt| / 60 (m/s per nominal 60 Hz frame): the velocity change a 60 Hz frame would show
  ang      angle between consecutive rotations / dt (deg/s), for Spine2 and Head
Root (entity) speed and jerk use entity_pos (world space) and are computed for every capture, pose or not.

Windows: in = [t_in - 0.15, t_in + 0.25], out = [t_out - 0.20, t_out + 0.25].
Cut steps compare the last frame before a cut with the first frame at or after it; a ratio well above 1 (boundary
rate over the median rate of the frames around it) means the pose jumps at the cut.

Bone analysis runs only where pose_ok is true. Clip times of patched occurrences use the play model's clock,
  clip_t = o + r * (t - t_in)
which is the model matchclip.fit_time_map fits (clip_time = o + r * (t - t_start), t_start = the clip-start event).
So at the capture's clip start the game is at clip time o. bake2 evaluates its fades on the same clip clock.
A capture time t therefore sits at clip time o + r * (t - t_in); the fade-in starts at t_in and the patch window ends
at o + window, which lands on the next clip start (t_out) for these clips.

Usage (from v2\\pipeline):  python transition_report.py [--out DIR] [--no-plots]
Outputs: DIR\\report.md and DIR\\*.png. The findings block in report.md (between the FINDINGS markers) is kept on rerun.
"""
import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np

import bake2
import capture
import config
import dumps

SESSIONS = ('session1_20261008', 'session3_20261008')
CAPTURE_ROOT = os.path.join(config.V2, 'data', 'captures')
OUT_DIR = os.path.join(config.OUT, 'transitions')
MAPPING = os.path.join(config.PATCHES, 'mapping.json')
SKIP_RE = re.compile(r'^xx_Parkour_Sprint_SkipOver_')

NOMINAL_FRAME = 1.0 / 60.0
WIN = {'in': (-0.15, 0.25), 'out': (-0.20, 0.25)}
JERK_BONES = ('Hips', 'LeftHand', 'RightHand', 'Head')
REPORT_JERK_ORDER = ('LeftHand', 'RightHand', 'Head', 'Hips')   # column order of the report tables
SPEED_BONES = ('Hips', 'Spine2', 'Head', 'LeftHand', 'RightHand', 'LeftFoot', 'RightFoot')
ANG_BONES = ('Spine2', 'Head')
BOUNDARY_BONES = ('Hips', 'Head', 'LeftHand', 'RightHand', 'LeftFoot', 'RightFoot')
BOUNDARY_NEIGH = 6          # frame pairs either side of a cut used for the typical rate

FINDINGS_BEGIN = '<!-- FINDINGS:BEGIN -->'
FINDINGS_END = '<!-- FINDINGS:END -->'


# ---------------------------------------------------------------------------------------------- derivatives

def mid_times(t):
    return 0.5 * (t[1:] + t[:-1])


def velocity(t, P):
    """Finite-difference velocity between consecutive frames with the real frame times.

    t [N], P [N,k] -> (tm [N-1] midpoint times, V [N-1,k])."""
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError('frame times must be strictly increasing')
    return mid_times(t), np.diff(P, axis=0) / dt[:, None]


def speed(t, P):
    tm, V = velocity(t, P)
    return tm, np.linalg.norm(V, axis=1)


def jerk(t, P):
    """Velocity change per 60 Hz frame (m/s per frame): |dV/dtm| / 60, with the midpoint spacing dtm."""
    tm, V = velocity(t, P)
    dtm = np.diff(tm)
    A = np.diff(V, axis=0) / dtm[:, None]
    return mid_times(tm), np.linalg.norm(A, axis=1) * NOMINAL_FRAME


def angular_speed(t, Q):
    """Angle between consecutive quaternions (xyzw) divided by the frame dt (deg/s), at frame-pair midpoints."""
    Qn = Q / np.linalg.norm(Q, axis=1)[:, None]
    d = np.abs(np.sum(Qn[:-1] * Qn[1:], axis=1))
    s = np.sqrt(np.clip(1.0 - d * d, 0.0, None))
    ang = 2.0 * np.arctan2(s, np.clip(d, 0.0, 1.0))
    return mid_times(t), np.degrees(ang) / np.diff(t)


def window_max(tt, vals, lo, hi):
    """(max, time of max, n samples) of vals whose time tt lies in [lo, hi]; (nan, nan, 0) when empty."""
    m = (tt >= lo) & (tt <= hi) & np.isfinite(vals)
    if not m.any():
        return float('nan'), float('nan'), 0
    idx = np.nonzero(m)[0]
    j = idx[int(np.argmax(vals[idx]))]
    return float(vals[j]), float(tt[j]), int(m.sum())


CONTROL_WIDTH = 0.40         # s, width of the control windows (about the in-window length)
CONTROL_PAD = (-0.15, 0.30)  # a control window may not touch [clip start + pad[0], clip start + pad[1]]


def control_median_max(tt, vals, clip_starts, width=CONTROL_WIDTH):
    """Median over clean control windows of the window maximum: the 'normal motion' level of a series in this capture.

    A control window starts at each sample time, lasts `width`, and is kept only when it touches no clip start
    (padded by CONTROL_PAD). Returns (median, n_windows); (nan, 0) when no window is clean."""
    tt = np.asarray(tt, float)
    vals = np.asarray(vals, float)
    starts = np.asarray(clip_starts, float)
    maxima = []
    for s in tt:
        e = s + width
        if e > tt[-1]:
            break
        if starts.size and np.any((starts + CONTROL_PAD[1] >= s) & (starts + CONTROL_PAD[0] <= e)):
            continue
        m = (tt >= s) & (tt <= e) & np.isfinite(vals)
        if m.sum() >= 3:
            maxima.append(float(vals[m].max()))
    if not maxima:
        return float('nan'), 0
    return float(np.median(maxima)), len(maxima)


def timing_corr(cap):
    """Correlation of root displacement per frame pair with the pair's dt (about 1 when positions follow the
    timestamps at constant speed). Low values mean the positions and the frame times disagree at frame level."""
    dt = np.diff(cap.t)
    d = np.linalg.norm(np.diff(cap.entity_pos, axis=0), axis=1)
    moving = d / dt > 2.0
    if moving.sum() < 5:
        return float('nan')
    return float(np.corrcoef(d[moving], dt[moving])[0, 1])


def boundary_step(t, X, tb, quat=False):
    """Pose jump across a cut at time tb: last frame before tb to first frame at or after it.

    X is [N,3] positions or [N,4] quaternions (xyzw, quat=True). Returns dict(step, rate, typical_rate, ratio, dt):
    step in m (deg for quaternions), rate in per second, typical_rate = median rate of the frame pairs around the cut
    (the boundary pair excluded), ratio = rate / typical_rate. None when the cut is too close to either end."""
    i1 = int(np.searchsorted(t, tb, side='left'))
    if i1 < 2 or i1 > len(t) - 2:
        return None
    i0 = i1 - 1
    dtb = float(t[i1] - t[i0])
    if quat:
        Qn = X / np.linalg.norm(X, axis=1)[:, None]
        d = min(1.0, abs(float(np.dot(Qn[i0], Qn[i1]))))
        step = math.degrees(2.0 * math.atan2(math.sqrt(max(0.0, 1.0 - d * d)), d))
        _, rate_series = angular_speed(t, X)
    else:
        step = float(np.linalg.norm(X[i1] - X[i0]))
        _, rate_series = speed(t, X)
    k = np.arange(max(0, i1 - BOUNDARY_NEIGH), min(len(t) - 1, i1 + BOUNDARY_NEIGH))
    k = k[k != i0]
    rates = rate_series[k]
    rates = rates[np.isfinite(rates)]
    typical = float(np.median(rates)) if rates.size else float('nan')
    rate = step / dtb
    ratio = rate / typical if np.isfinite(typical) and typical > 0 else float('nan')
    return dict(step=step, rate=rate, typical_rate=typical, ratio=ratio, dt=dtb)


# ---------------------------------------------------------------------------------------------- occurrences

@dataclass
class Occurrence:
    path: str
    session: str
    name: str
    handle: object
    t_in: float
    t_out: object            # float, or None when no clip starts after the SkipOver
    next_clip: object        # name (or handle) of the clip starting at t_out
    pose_ok: bool
    patched: bool            # the clip's own patched flag
    t_end: float             # last frame time of the capture
    clip_len: object         # ACU dump length (s), or None
    fades: object = None     # dict(o, window, r, fi, fo, source, enabled) for patched clips, else None
    win: dict = field(default_factory=dict)        # win['in' | 'out'] -> window metrics
    bnd: dict = field(default_factory=dict)        # bnd['in' | 'out'] -> boundary steps per bone
    truncated: dict = field(default_factory=dict)  # window name -> True when the capture ends inside it
    ctrl: dict = field(default_factory=dict)       # 'jerk <bone>' / 'root jerk' -> (control median, n windows)

    @property
    def base(self):
        return os.path.basename(self.path)

    @property
    def short(self):
        return self.name.replace('xx_Parkour_Sprint_SkipOver_', '')

    @property
    def group(self):
        return 'patched' if self.patched else 'original'


_LEN_CACHE = {}


def _clip_length(name, handle):
    key = (name, handle)
    if key not in _LEN_CACHE:
        d = dumps.load_by_name(name) if name else None
        if d is None and handle:
            d = dumps.load_by_handle(handle)
        _LEN_CACHE[key] = float(d['Length']) if d is not None else None
    return _LEN_CACHE[key]


def load_mapping(path=MAPPING):
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return {e['name']: bool(e.get('enabled')) for e in json.load(f).get('clips', [])}


def fades_for_clip(name, mapping):
    """Bake parameters for a clip exactly as bake2 uses them: play_for, the rate r, and fades_for (T9e)."""
    dump = dumps.load_by_name(name)
    if dump is None:
        return None
    model = bake2.load_play_model()
    o, window, source = bake2.play_for(name, float(dump['Length']), model, bake2.load_measured())
    r = float(model.get(name, {}).get('r', 1.0))
    fi, fo = bake2.fades_for(window)
    return dict(o=o, window=window, r=r, fi=fi, fo=fo, source=source, enabled=mapping.get(name))


def occurrences_of(cap, path, session, mapping):
    out = []
    for c in cap.clips:
        name = c.get('name') or ''
        if not SKIP_RE.match(name):
            continue
        t_in = float(c['t'])
        later = [d for d in cap.clips if d['t'] > t_in + 1e-9]
        nxt = later[0] if later else None
        occ = Occurrence(
            path=os.path.abspath(path), session=session, name=name, handle=c.get('handle'),
            t_in=t_in, t_out=float(nxt['t']) if nxt else None,
            next_clip=(nxt.get('name') or nxt.get('handle')) if nxt else None,
            pose_ok=bool(cap.pose_ok and cap.bones_space == 'entity'),
            patched=bool(c.get('patched', False)), t_end=float(cap.t[-1]),
            clip_len=_clip_length(name, c.get('handle')),
        )
        if occ.patched:
            occ.fades = fades_for_clip(name, mapping)
        out.append(occ)
    return out


# ---------------------------------------------------------------------------------------------- analysis

def analyse(cap, occ):
    """Fill occ.win, occ.bnd and occ.truncated from the capture."""
    series = {}
    if occ.pose_ok:
        for b in SPEED_BONES:
            arr = cap.bones.get(b)
            if arr is None:
                continue
            series[('speed', b)] = speed(cap.t, arr[:, 0:3])
            if b in JERK_BONES:
                series[('jerk', b)] = jerk(cap.t, arr[:, 0:3])
            if b in ANG_BONES:
                series[('ang', b)] = angular_speed(cap.t, arr[:, 3:7])
    series[('root_speed', 'root')] = speed(cap.t, cap.entity_pos)
    series[('root_jerk', 'root')] = jerk(cap.t, cap.entity_pos)

    starts = [c['t'] for c in cap.clips]
    for (kind, b), (tt, vals) in series.items():
        if kind == 'jerk':
            occ.ctrl['jerk %s' % b] = control_median_max(tt, vals, starts)
        elif kind == 'root_jerk':
            occ.ctrl['root jerk'] = control_median_max(tt, vals, starts)

    centres = {'in': occ.t_in, 'out': occ.t_out}
    for w, centre in centres.items():
        if centre is None:
            continue
        lo, hi = centre + WIN[w][0], centre + WIN[w][1]
        occ.truncated[w] = bool(hi > occ.t_end)
        m = {'jerk': {}, 'speed': {}, 'ang': {}, 'root_speed': None, 'root_jerk': None}
        for (kind, b), (tt, vals) in series.items():
            mx, tmx, n = window_max(tt, vals, lo, hi)
            rel = (tmx - centre) if np.isfinite(tmx) else None
            if kind in ('speed', 'jerk', 'ang'):
                m[kind][b] = (mx, rel, n)
            else:
                m[kind] = (mx, rel, n)
        occ.win[w] = m

        bnd = {}
        if occ.pose_ok:
            for b in BOUNDARY_BONES:
                arr = cap.bones.get(b)
                if arr is not None:
                    bnd[b] = boundary_step(cap.t, arr[:, 0:3], centre)
            for b in ANG_BONES:
                arr = cap.bones.get(b)
                if arr is not None:
                    bnd[b + ' rot'] = boundary_step(cap.t, arr[:, 3:7], centre, quat=True)
        bnd['root'] = boundary_step(cap.t, cap.entity_pos, centre)
        occ.bnd[w] = bnd
    return occ


def process_capture(path, session, mapping):
    cap = capture.load(path)
    occs = occurrences_of(cap, path, session, mapping)
    for occ in occs:
        analyse(cap, occ)
    return cap, occs


def discover(root=CAPTURE_ROOT, sessions=SESSIONS):
    """(path, session) for every capture JSON in the session folders, in order."""
    for s in sessions:
        d = os.path.join(root, s)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith('.json'):
                yield os.path.join(d, fn), s


def analyse_all(sessions=SESSIONS, root=CAPTURE_ROOT):
    """(summary, occurrences). summary: capture file name -> dict(session, pose_ok, n_frames, n_skip, ...)."""
    mapping = load_mapping()
    summary = {}
    occs = []
    for path, session in discover(root, sessions):
        cap, found = process_capture(path, session, mapping)
        summary[os.path.basename(path)] = dict(session=session, pose_ok=cap.pose_ok, n_frames=len(cap.t),
                                               n_skip=len(found), t_span=float(cap.t[-1] - cap.t[0]),
                                               timing_r=timing_corr(cap))
        occs.extend(found)
    return summary, occs


# ---------------------------------------------------------------------------------------------- clip time

def clip_time(occ, t):
    """Play-model clock at capture time t: o + r * (t - t_in). Needs occ.fades."""
    f = occ.fades
    return f['o'] + f['r'] * (t - occ.t_in)


def capture_rel(occ, c):
    """Capture time relative to t_in at which the play-model clock reads clip time c (inverse of clip_time)."""
    f = occ.fades
    return (c - f['o']) / f['r']


def fade_regions(occ):
    """Clip-time boundaries of the bake for a patched occurrence, or None."""
    f = occ.fades
    if not f:
        return None
    o, w = f['o'], f['window']
    return dict(o=o, fade_in_end=o + f['fi'], fade_out_start=o + w - f['fo'], end=o + w)


def region_of(clip_t, reg):
    if reg is None:
        return 'n/a'
    if clip_t < reg['o']:
        return 'before window (original)'
    if clip_t < reg['fade_in_end']:
        return 'fade-in'
    if clip_t < reg['fade_out_start']:
        return 'full patch'
    if clip_t <= reg['end']:
        return 'fade-out'
    return 'after window (original)'


def spike_clip_times(occ):
    """Window maxima of a patched occurrence as spikes on the play-model clock.

    Each dict: window, what, value, unit, rel (time from the window centre), t_from_in (capture time from t_in),
    clip_t (o + r * t_from_in), region, ratio (value / control median)."""
    if not occ.fades:
        return []
    reg = fade_regions(occ)
    rows = []
    for w in ('in', 'out'):
        m = occ.win.get(w)
        if not m:
            continue
        centre = occ.t_in if w == 'in' else occ.t_out
        if centre is None:
            continue
        cands = [('jerk %s' % b, v, rel, 'm/s/frame') for b, (v, rel, n) in m['jerk'].items()]
        cands += [('ang %s' % b, v, rel, 'deg/s') for b, (v, rel, n) in m['ang'].items()]
        if m['root_jerk']:
            v, rel, n = m['root_jerk']
            cands.append(('root jerk', v, rel, 'm/s/frame'))
        for what, v, rel, unit in cands:
            if rel is None or not np.isfinite(rel):
                continue
            t_from_in = (centre - occ.t_in) + rel
            ct = clip_time(occ, occ.t_in + t_from_in)
            rows.append(dict(window=w, what=what, value=v, unit=unit, rel=rel, t_from_in=t_from_in,
                             clip_t=ct, region=region_of(ct, reg), ratio=_ctrl_ratio(occ, what, v)))
    return rows


# ---------------------------------------------------------------------------------------------- report

def _f(x, nd=2):
    if x is None:
        return '-'
    if isinstance(x, (float, np.floating)) and not np.isfinite(x):
        return '-'
    return ('%.' + str(nd) + 'f') % x


def _rel(x):
    return '-' if x is None or not np.isfinite(x) else '%+.3f' % x


def _short_file(name):
    return name.replace('cap_20261008_', '')


def _ctrl_ratio(occ, key, value):
    """Window maximum divided by the capture's control median for the same series (nan when unavailable)."""
    c = occ.ctrl.get(key)
    if not c or not np.isfinite(c[0]) or c[0] <= 0 or value is None or not np.isfinite(value):
        return float('nan')
    return value / c[0]


def md_table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return out


def _median(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float('nan')


def _groups(occs):
    """{(clip name, group): [occurrences]} in sorted key order."""
    by = {}
    for o in occs:
        by.setdefault((o.name, o.group), []).append(o)
    return {k: by[k] for k in sorted(by)}


def aggregate(occs):
    """Per (clip name, group): medians of the window maxima over the occurrences that have them."""
    rows = {}
    for key, group in _groups(occs).items():
        pose = [o for o in group if o.pose_ok]
        rec = dict(n=len(group), n_pose=len(pose))
        for w in ('in', 'out'):
            for b in JERK_BONES:
                rec[('jerk', w, b)] = _median([o.win[w]['jerk'][b][0] for o in pose
                                               if w in o.win and b in o.win[w]['jerk']])
            for b in ANG_BONES:
                rec[('ang', w, b)] = _median([o.win[w]['ang'][b][0] for o in pose
                                              if w in o.win and b in o.win[w]['ang']])
            rec[('root_jerk', w)] = _median([o.win[w]['root_jerk'][0] for o in group
                                             if w in o.win and o.win[w]['root_jerk']])
            rec[('root_speed', w)] = _median([o.win[w]['root_speed'][0] for o in group
                                              if w in o.win and o.win[w]['root_speed']])
        rows[key] = rec
    return rows


def report_md(summary, occs, figs):
    L = []
    L.append('# Sprint SkipOver transitions: WD2 patch (session3) vs original ACU (session1)')
    L.append('')
    L.append('Generated by `v2\\pipeline\\transition_report.py`. Windows: in = [t_in-0.15, t_in+0.25] and '
             'out = [t_out-0.20, t_out+0.25] (s). Jerk = velocity change per 60 Hz frame (m/s per frame), computed with '
             'the real frame times. Angular speed in deg/s. "patched" means the clip\'s own patched flag is set.')
    L.append('')
    L.append('Scope and limits:')
    L.append('')
    L.append('- Captures: session1 (original ACU) and session3 (WD2 patch, plugin 0.4.1). session2 is not in the brief and is not included.')
    n_pose = sum(1 for o in occs if o.pose_ok)
    L.append('- Bone analysis runs only where `pose_ok` is true: %d of %d SkipOver occurrences. Root (entity) speed and '
             'jerk are reported for all occurrences.' % (n_pose, len(occs)))
    pat_pose = [o for o in occs if o.patched and o.pose_ok]
    L.append('- Patched occurrences with bone data: %d (%s). Any patched-vs-original bone comparison rests on that count.'
             % (len(pat_pose), ', '.join(sorted({_short_file(o.base) for o in pat_pose})) or 'none'))
    s3 = [s for s in summary.values() if s['session'].startswith('session3')]
    L.append('- session3: %d of %d captures have pose_ok false, and most of those have no bone frames at all.'
             % (sum(1 for s in s3 if not s['pose_ok']), len(s3)))
    L.append('')

    L.append('## 1. Inventory')
    L.append('')
    rows = []
    for fn, s in summary.items():
        if s['n_skip']:
            rows.append([s['session'][:8], _short_file(fn), s['n_frames'], '%.2f' % s['t_span'],
                         'yes' if s['pose_ok'] else 'no', s['n_skip'], _f(s['timing_r'], 2)])
    L += md_table(['session', 'capture', 'frames', 'span (s)', 'pose_ok', 'SkipOver clips',
                   'root dp-vs-dt corr'], rows)
    L.append('')
    L.append('"root dp-vs-dt corr": correlation of root displacement with frame dt over moving frame pairs. At constant '
             'speed it would be near 1 if positions followed the frame times. Values near 0 mean the positions and the '
             'frame times disagree frame to frame, so single-frame derivatives carry timing noise (see the control '
             'baseline below).')
    L.append('')
    rows = []
    for o in occs:
        rows.append([o.session[:8], _short_file(o.base), o.short, o.group, '%.3f' % o.t_in,
                     _f(o.t_out, 3) if o.t_out is not None else 'none', o.next_clip or '-',
                     _f(o.t_out - o.t_in, 3) if o.t_out is not None else '-',
                     _f(o.clip_len, 3) if o.clip_len is not None else '-',
                     'yes' if o.pose_ok else 'no',
                     'cut' if o.truncated.get('out') else ('-' if o.t_out is None else 'no')])
    L += md_table(['session', 'capture', 'SkipOver', 'group', 't_in', 't_out', 'next clip at t_out', 'in-to-out (s)',
                   'clip len (s)', 'pose_ok', 'out window cut by capture end'], rows)
    L.append('')

    L.append('## 2. Window maxima, bone analysis (pose_ok only)')
    L.append('')
    L.append('Jerk in m/s per 60 Hz frame. Each jerk cell is value (time of the maximum relative to the window centre, s) '
             'and "x" = value / control median, the window maximum over clean same-capture windows (see section 3 for the '
             'method). Angular speed in deg/s.')
    L.append('')
    for w, label in (('in', 'In-transition, centre t_in'), ('out', 'Out-transition, centre t_out')):
        rows = []
        for o in occs:
            if not o.pose_ok or w not in o.win:
                continue
            m = o.win[w]
            cells = []
            for b in REPORT_JERK_ORDER:
                v, rel, _n = m['jerk'].get(b, (float('nan'), None, 0))
                cells.append('%s (%s) x%s' % (_f(v, 3), _rel(rel), _f(_ctrl_ratio(o, 'jerk ' + b, v), 1)))
            for b in ANG_BONES:
                v, rel, _n = m['ang'].get(b, (float('nan'), None, 0))
                cells.append('%s (%s)' % (_f(v, 0), _rel(rel)))
            rows.append([_short_file(o.base), o.short, o.group] + cells)
        L.append('### ' + label)
        L.append('')
        L += md_table(['capture', 'SkipOver', 'group', 'jerk LeftHand', 'jerk RightHand', 'jerk Head', 'jerk Hips',
                       'ang Spine2', 'ang Head'], rows) if rows else ['(none)']
        L.append('')

    L.append('## 3. Root (entity) speed and jerk, all occurrences')
    L.append('')
    L.append('Control baseline: for each capture and series, the window maximum (same 0.40 s width) is taken over every '
             'window start whose window touches no clip start (padded -0.15 s / +0.30 s). The median of those maxima is '
             'the normal-motion level of that capture. "x" in the tables = window maximum / control median. A ratio near '
             '1 means the window is no noisier than ordinary motion in the same capture.')
    L.append('')
    for w, label in (('in', 'In-transition, centre t_in'), ('out', 'Out-transition, centre t_out')):
        rows = []
        for o in occs:
            if w not in o.win:
                continue
            rs = o.win[w]['root_speed']
            rj = o.win[w]['root_jerk']
            rows.append([_short_file(o.base), o.short, o.group, 'yes' if o.pose_ok else 'no',
                         '%s (%s)' % (_f(rs[0], 2), _rel(rs[1])),
                         '%s (%s) x%s' % (_f(rj[0], 3), _rel(rj[1]), _f(_ctrl_ratio(o, 'root jerk', rj[0]), 1))])
        L.append('### ' + label)
        L.append('')
        L += md_table(['capture', 'SkipOver', 'group', 'pose_ok', 'root speed m/s', 'root jerk m/s/frame'], rows)
        L.append('')

    L.append('## 4. Aggregate per clip name: patched vs original (median of window maxima)')
    L.append('')
    L.append('n = occurrences in the group, n_pose = those with bone data. A median over one value is that value. '
             '"-" means no bone data in the group.')
    L.append('')
    agg = aggregate(occs)
    rows = []
    for (name, group), r in agg.items():
        rows.append([name.replace('xx_Parkour_Sprint_SkipOver_', ''), group, r['n'], r['n_pose'],
                     _f(r[('jerk', 'in', 'LeftHand')], 3), _f(r[('jerk', 'in', 'RightHand')], 3),
                     _f(r[('jerk', 'in', 'Head')], 3), _f(r[('jerk', 'in', 'Hips')], 3),
                     _f(r[('jerk', 'out', 'LeftHand')], 3), _f(r[('jerk', 'out', 'RightHand')], 3),
                     _f(r[('jerk', 'out', 'Head')], 3), _f(r[('jerk', 'out', 'Hips')], 3)])
    L += md_table(['clip', 'group', 'n', 'n_pose', 'in LH', 'in RH', 'in Head', 'in Hips',
                   'out LH', 'out RH', 'out Head', 'out Hips'], rows)
    L.append('')
    rows = []
    for (name, group), r in agg.items():
        rows.append([name.replace('xx_Parkour_Sprint_SkipOver_', ''), group, r['n'], r['n_pose'],
                     _f(r[('ang', 'in', 'Spine2')], 0), _f(r[('ang', 'in', 'Head')], 0),
                     _f(r[('ang', 'out', 'Spine2')], 0), _f(r[('ang', 'out', 'Head')], 0),
                     _f(r[('root_jerk', 'in')], 3), _f(r[('root_jerk', 'out')], 3),
                     _f(r[('root_speed', 'in')], 2), _f(r[('root_speed', 'out')], 2)])
    L += md_table(['clip', 'group', 'n', 'n_pose', 'in ang Spine2', 'in ang Head', 'out ang Spine2', 'out ang Head',
                   'in root jerk', 'out root jerk', 'in root speed', 'out root speed'], rows)
    L.append('')

    L.append('Same groups, as ratios to each capture\'s control median (section 3). Median over occurrences.')
    L.append('')
    rows = []
    for (name, group), occ_group in _groups(occs).items():
        pose = [o for o in occ_group if o.pose_ok]

        def med_ratio(w, b, pool):
            vals = []
            for o in pool:
                if w in o.win and b in o.win[w]['jerk']:
                    vals.append(_ctrl_ratio(o, 'jerk ' + b, o.win[w]['jerk'][b][0]))
            return _median(vals)

        rr = {w: _median([_ctrl_ratio(o, 'root jerk', o.win[w]['root_jerk'][0]) for o in occ_group
                          if w in o.win and o.win[w]['root_jerk']]) for w in ('in', 'out')}
        rows.append([name.replace('xx_Parkour_Sprint_SkipOver_', ''), group, len(occ_group), len(pose),
                     _f(med_ratio('in', 'LeftHand', pose), 1), _f(med_ratio('in', 'RightHand', pose), 1),
                     _f(med_ratio('in', 'Head', pose), 1), _f(med_ratio('in', 'Hips', pose), 1),
                     _f(med_ratio('out', 'LeftHand', pose), 1), _f(med_ratio('out', 'RightHand', pose), 1),
                     _f(med_ratio('out', 'Head', pose), 1), _f(med_ratio('out', 'Hips', pose), 1),
                     _f(rr['in'], 1), _f(rr['out'], 1)])
    L += md_table(['clip', 'group', 'n', 'n_pose', 'in LH x', 'in RH x', 'in Head x', 'in Hips x',
                   'out LH x', 'out RH x', 'out Head x', 'out Hips x', 'in root x', 'out root x'], rows)
    L.append('')

    L.append('## 5. Pose at the cuts')
    L.append('')
    L.append('Step = pose change from the last frame before the cut to the first frame at or after it. ratio = '
             'boundary rate / median rate of the frames around the cut: about 1 is ordinary motion, well above 1 is a '
             'jump. Cells: step (cm for positions, deg for rotations) / ratio.')
    L.append('')
    cols = ('Hips', 'Head', 'Head rot', 'Spine2 rot', 'LeftHand', 'RightHand', 'LeftFoot', 'RightFoot', 'root')
    for w, label in (('in', 'At t_in (previous clip -> SkipOver)'), ('out', 'At t_out (SkipOver -> next clip)')):
        rows = []
        for o in occs:
            if w not in o.bnd:
                continue
            cells = []
            for b in cols:
                x = o.bnd[w].get(b)
                if x is None:
                    cells.append('-')
                    continue
                scale = 1.0 if b.endswith('rot') else 100.0
                cells.append('%s / x%s' % (_f(x['step'] * scale, 1), _f(x['ratio'], 1)))
            rows.append([_short_file(o.base), o.short, o.group, 'yes' if o.pose_ok else 'no'] + cells)
        L.append('### ' + label)
        L.append('')
        L += md_table(['capture', 'SkipOver', 'group', 'pose_ok'] + list(cols), rows) if rows else ['(none)']
        L.append('')

    L.append('## 6. Patched clips: bake fade windows and spike clip times')
    L.append('')
    L.append('Parameters from `bake2.play_for` (out/match/play_model.json), `bake2.fades_for` (T9e: fi = min(0.20, 0.30 w), '
             'fo = min(0.25, 0.35 w)) and out/patches/mapping.json (enabled). Clock: clip time = o + r * (t - t_in), the '
             'matchclip model. Fade-in = clip [o, o+fi], full patch = clip [o+fi, o+w-fo], fade-out = clip [o+w-fo, o+w]. '
             'The "from t_in" columns convert those to capture seconds after t_in.')
    L.append('')
    fade_rows = []
    seen = set()
    for o in occs:
        if not o.patched or not o.fades or o.name in seen:
            continue
        seen.add(o.name)
        f = o.fades
        reg = fade_regions(o)
        en = f['enabled']
        fi_s = (capture_rel(o, reg['o']), capture_rel(o, reg['fade_in_end']))
        fo_s = (capture_rel(o, reg['fade_out_start']), capture_rel(o, reg['end']))
        fade_rows.append([o.short, 'yes' if en else ('no' if en is False else 'not in mapping'), f['source'],
                          _f(f['o'], 3), _f(f['window'], 3), _f(f['r'], 3), _f(f['fi'], 3), _f(f['fo'], 3),
                          '%s to %s' % (_f(reg['o'], 3), _f(reg['fade_in_end'], 3)),
                          '%s to %s' % (_f(reg['fade_out_start'], 3), _f(reg['end'], 3)),
                          '%s to %s' % (_f(fi_s[0], 3), _f(fi_s[1], 3)),
                          '%s to %s' % (_f(fo_s[0], 3), _f(fo_s[1], 3)),
                          _f(o.t_out - o.t_in, 3) if o.t_out is not None else '-',
                          _f(o.clip_len, 3) if o.clip_len is not None else '-'])
    L += md_table(['clip', 'enabled', 'source', 'o (s)', 'window (s)', 'r', 'fi (s)', 'fo (s)',
                   'fade-in, clip s', 'fade-out, clip s', 'fade-in, s from t_in', 'fade-out, s from t_in',
                   't_out - t_in (s)', 'clip len (s)'], fade_rows) if fade_rows \
        else ['(no patched SkipOver)']
    L.append('')
    L.append('Spike clip times, patched occurrences (window maxima). rel = spike time minus its centre (t_in or t_out). '
             'clip_t = o + r * (t_spike - t_in), the play-model clock, placed in the fade regions by clip_t. '
             'x control = value / control median (section 3).')
    L.append('')
    rows = []
    for o in occs:
        if not o.patched:
            continue
        for s in spike_clip_times(o):
            rows.append([_short_file(o.base), o.short, s['window'], s['what'], _f(s['value'], 3), _f(s['ratio'], 1),
                         _rel(s['rel']), _f(s['clip_t'], 3), s['region']])
    L += md_table(['capture', 'SkipOver', 'window', 'spike', 'value', 'x control', 'rel (s)', 'clip_t',
                   'region'], rows) if rows else ['(no patched spikes)']
    L.append('')

    L.append('## 7. Plots')
    L.append('')
    if figs:
        for f in figs:
            L.append('- `%s`' % f)
    else:
        L.append('(plots skipped)')
    L.append('')
    L.append('## Findings')
    L.append('')
    return L


def _findings_block(existing_text):
    """The FINDINGS block from an earlier report.md, or a placeholder."""
    if existing_text:
        i = existing_text.find(FINDINGS_BEGIN)
        j = existing_text.find(FINDINGS_END)
        if i >= 0 and j > i:
            return existing_text[i:j + len(FINDINGS_END)]
    return FINDINGS_BEGIN + '\n(findings not written yet)\n' + FINDINGS_END


def write_report(path, lines):
    """Write report.md; the hand-written findings block between the FINDINGS markers is kept from an earlier run."""
    existing = None
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            existing = f.read()
    body = '\n'.join(lines) + '\n' + _findings_block(existing) + '\n'
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(body)


# ---------------------------------------------------------------------------------------------- plots

PLOT_CAPTURES = {
    'session1_20261008': ['cap_20261008_121824_6.json', 'cap_20261008_121833_7.json', 'cap_20261008_121939_13.json'],
    'session3_20261008': ['cap_20261008_191504_2.json', 'cap_20261008_191527_4.json', 'cap_20261008_191607_11.json'],
}


def plot_occurrence(occ, cap, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t_in, t_out = occ.t_in, occ.t_out
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True)
    colours = {'LeftHand': '#d55e00', 'RightHand': '#e69f00', 'Head': '#0072b2', 'Hips': '#009e73'}
    if occ.pose_ok:
        for b in ('LeftHand', 'RightHand', 'Head', 'Hips'):
            P = cap.bones[b][:, 0:3]
            tm, v = speed(cap.t, P)
            ax1.plot(tm - t_in, v, color=colours[b], lw=1.2, label='%s' % b)
            tj, j = jerk(cap.t, P)
            ax2.plot(tj - t_in, j, color=colours[b], lw=1.0, label='%s' % b)
        ax1.set_ylabel('speed, entity-local (m/s)')
        ax2.set_ylabel('jerk (m/s per 60 Hz frame)')
        sub = 'bone speeds (entity-local)'
    else:
        tm, v = speed(cap.t, cap.entity_pos)
        ax1.plot(tm - t_in, v, color='#444444', lw=1.2, label='entity root')
        tj, j = jerk(cap.t, cap.entity_pos)
        ax2.plot(tj - t_in, j, color='#444444', lw=1.0, label='entity root')
        ax1.set_ylabel('root speed (m/s)')
        ax2.set_ylabel('root jerk (m/s per frame)')
        sub = 'pose_ok false: no bone data, entity root only'
    f = occ.fades
    if f:
        o, w, fi, fo = f['o'], f['window'], f['fi'], f['fo']
        spans = [(capture_rel(occ, o), capture_rel(occ, o + w), '#7fb3d5', 0.18, 'patch window (clip o to o+w)'),
                 (capture_rel(occ, o), capture_rel(occ, o + fi), '#f5b041', 0.28, 'bake fade-in'),
                 (capture_rel(occ, o + w - fo), capture_rel(occ, o + w), '#ec7063', 0.28, 'bake fade-out')]
        for a, b_, col, alpha, lab in spans:
            for ax in (ax1, ax2):
                ax.axvspan(a, b_, color=col, alpha=alpha, lw=0)
            ax1.plot([], [], color=col, lw=6, alpha=alpha + 0.3, label=lab)
    for ax in (ax1, ax2):
        ax.axvline(0.0, color='k', ls='--', lw=1.0)
        if t_out is not None:
            ax.axvline(t_out - t_in, color='k', ls=':', lw=1.2)
        ax.grid(alpha=0.3)
    top = ax1.get_ylim()[1]
    ax1.text(0.0, top, ' t_in', fontsize=8, va='top')
    if t_out is not None:
        ax1.text(t_out - t_in, top, ' t_out', fontsize=8, va='top')
    x_hi = (t_out if t_out is not None else occ.t_end) - t_in + 0.6
    ax2.set_xlabel('time from t_in (s)')
    ax1.set_xlim(-0.6, x_hi)
    ax1.legend(fontsize=7, ncol=3, loc='upper right')
    ax2.legend(fontsize=7, ncol=4, loc='upper right')
    ax1.set_title('%s  %s  [%s]\n%s' % (occ.base, occ.short, occ.group, sub), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_plots(occs, out_dir):
    files = []
    for session, names in PLOT_CAPTURES.items():
        for fn in names:
            cand = [o for o in occs if o.base == fn and o.session == session]
            if not cand:
                continue
            occ = cand[0]
            cap = capture.load(occ.path)
            png = os.path.join(out_dir, '%s_%s.png' % (session.split('_')[0], fn.replace('cap_20261008_', '')[:-5]))
            plot_occurrence(occ, cap, png)
            files.append(os.path.basename(png))
    return files


# ---------------------------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description='Sprint SkipOver transition analysis (see module docstring).')
    ap.add_argument('--out', default=OUT_DIR, help='output folder (default v2\\out\\transitions)')
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args(argv)

    summary, occs = analyse_all()
    if not occs:
        print('no SkipOver occurrences under', CAPTURE_ROOT)
        return 1
    os.makedirs(args.out, exist_ok=True)
    figs = [] if args.no_plots else make_plots(occs, args.out)
    rep = os.path.join(args.out, 'report.md')
    write_report(rep, report_md(summary, occs, figs))

    print('captures with SkipOver: %d, occurrences: %d (pose_ok %d)' % (
        sum(1 for s in summary.values() if s['n_skip']), len(occs), sum(1 for o in occs if o.pose_ok)))
    for o in occs:
        bits = []
        if o.pose_ok:
            for w in ('in', 'out'):
                if w in o.win:
                    jj = o.win[w]['jerk']
                    bits.append('%s jerk LH %.2f RH %.2f Head %.2f Hips %.2f' % (
                        w, jj['LeftHand'][0], jj['RightHand'][0], jj['Head'][0], jj['Hips'][0]))
        print('%-24s %-8s %-30s t_in %7.3f  %s' % (_short_file(o.base), o.group, o.short, o.t_in,
                                                    '; '.join(bits) if bits else 'root only'))
    print('report:', rep)
    for f in figs:
        print('plot:', os.path.join(args.out, f))
    return 0


if __name__ == '__main__':
    sys.exit(main())
