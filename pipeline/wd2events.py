"""Contact and event candidates for WD2 clips, from rotations only (no pelvis translation is decoded).

Idea: the Reference (pelvis) is kept at the origin with identity root, so the effector positions are
pelvis-relative. A point planted in the world moves backward relative to the pelvis at the body's forward
speed s. Candidate contact = forward velocity v_f within TOL of -s, sustained for MIN_TICKS ticks.

Conventions:
  * the ACU world frame of retarget.py (X forward, Y left, Z up); positions come from fk.fk with the
    retargeted locals (retarget.Rig().target_locals), root identity.
  * ticks are 30 Hz; t = tick / 30.
  * forward axis: horizontal part of the pelvis heading (Reference rotation applied to +X) over the first
    0.5 s of the clip, normalised. The bind-pose toes point along +X (ACU) so this is the body's forward.
  * the body speed s is NOT known from the clip; it is estimated from the run-up (see estimate_body_speed)
    unless overridden.

CLI:  python wd2events.py <stem> [<stem> ...] [--speed M_PER_S] [--strip-speed M_PER_S]
Writes v2\\out\\sheets\\wd2_<stem>.png and prints contact intervals and suggested events per clip.
"""
import argparse
import os
import sys
import zlib

import numpy as np

import config
import fk
import retarget
import wd2

FPS = 30.0
DT = 1.0 / FPS
EFFECTORS = ['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot', 'LeftFootThumb1', 'RightFootThumb1']
HANDS = ['LeftHand', 'RightHand']
FEET = ['LeftFoot', 'RightFoot', 'LeftFootThumb1', 'RightFootThumb1']
JOINTS = ['Reference', 'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
          'LeftArm', 'LeftForeArm', 'LeftHand', 'RightArm', 'RightForeArm', 'RightHand',
          'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftFootThumb1',
          'RightUpLeg', 'RightLeg', 'RightFoot', 'RightFootThumb1']
TOL = 0.35          # relative band around -s
MIN_TICKS = 3       # minimum run length for a contact (0.1 s)
RUNUP_S = 1.0       # window for the body speed estimate
FORWARD_WIN_S = 0.5 # window for the forward axis (pelvis heading)
HAND_MERGE_TICKS = 3


# ---- traces ----------------------------------------------------------------------------------------

def _smooth3(x):
    """[1, 2, 1] / 4 smoothing along axis 0, end values kept."""
    x = np.asarray(x, float)
    y = x.copy()
    if len(x) >= 3:
        y[1:-1] = 0.25 * x[:-2] + 0.5 * x[1:-1] + 0.25 * x[2:]
    return y


def _index_runs(mask, min_len):
    """(i0, i1) inclusive index pairs of runs of True with length >= min_len."""
    mask = np.asarray(mask, bool)
    runs = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def forward_axis(heading, t, win_s=FORWARD_WIN_S):
    """Unit horizontal forward vector: median pelvis heading over the first win_s seconds."""
    sel = t < t[0] + win_s
    h = np.median(np.asarray(heading)[sel], axis=0)
    h = np.array([h[0], h[1], 0.0])
    n = np.linalg.norm(h)
    if n < 1e-9:
        raise ValueError('pelvis heading has no horizontal component in the run-up')
    return h / n


def effector_traces(stem):
    """Retarget one WD2 clip (rotations only) and return pelvis-relative effector traces.

    Returns dict: stem, dur, t (s), dt, forward (3,), speed (m/s, from estimate_body_speed),
      eff: {name: {'pos' [F,3], 'vf' [F] (forward velocity, smoothed), 'z' [F] (height rel. pelvis)}},
      joints: {name: [F,3]} for the stick figure, heading [F,3].
    """
    m = wd2.load(stem)
    rig = retarget.Rig()
    ticks = int(round(m['key_times'][-1])) + 1
    bones = rig.ac
    lpos = np.array([b['lpos'] for b in bones], float)
    LR = np.empty((ticks, len(bones), 4))
    heading = np.empty((ticks, 3))
    for k in range(ticks):
        L, _ = rig.target_locals(rig.source_locals(m, k))
        LR[k] = np.array(L)
        heading[k] = retarget.qrot(LR[k, 0], np.array([1.0, 0.0, 0.0]))
    LP = np.broadcast_to(lpos, (ticks,) + lpos.shape)
    _, WP = fk.fk(bones, LR, LP)
    idx = {n: rig.ac_by_hash[zlib.crc32(n.encode())] for n in JOINTS}
    t = np.arange(ticks) * DT
    f = forward_axis(heading, t)
    eff = {}
    for n in EFFECTORS:
        P = WP[:, idx[n], :]
        V = np.gradient(P, DT, axis=0)
        eff[n] = {'pos': P, 'vf': _smooth3(V @ f), 'z': P[:, 2].copy()}
    joints = {n: WP[:, idx[n], :] for n in JOINTS}
    traces = {'stem': stem, 'dur': float(m['dur']), 't': t, 'dt': DT, 'forward': f,
              'eff': eff, 'joints': joints, 'heading': heading}
    traces['speed'] = estimate_body_speed(traces)
    return traces


# ---- body speed and candidate contacts --------------------------------------------------------------

def estimate_body_speed(traces, window=RUNUP_S, min_run=MIN_TICKS):
    """Body speed (m/s, >= 0) from the run-up.

    For each foot effector: over the first `window` seconds, take the most negative sustained run
    (>= min_run ticks with v_f < 0) and its median. The body speed is minus the median of those values
    over the feet that have one. Returns 0.0 when no foot moves backward in a sustained way.
    """
    t = np.asarray(traces['t'])
    sel = t < t[0] + window
    vals = []
    for name in FEET:
        if name not in traces['eff']:
            continue
        vf = np.asarray(traces['eff'][name]['vf'])[sel]
        best = None
        for i0, i1 in _index_runs(vf < 0, min_run):
            med = float(np.median(vf[i0:i1 + 1]))
            if best is None or med < best:
                best = med
        if best is not None:
            vals.append(best)
    if not vals:
        return 0.0
    return float(max(0.0, -np.median(vals)))


def _runs_to_seconds(mask, t, min_ticks):
    return [(float(t[i0]), float(t[i1])) for i0, i1 in _index_runs(mask, min_ticks)]


def candidate_contacts(traces, speed=None, tol=TOL, min_ticks=MIN_TICKS):
    """Contact candidate intervals per effector: {name: [(t0, t1), ...]} in seconds (inclusive ends).

    A tick is a candidate when |v_f + s| <= tol * s (v_f within tol of -s), with s = speed, else
    traces['speed'], else estimate_body_speed(traces). Runs shorter than min_ticks are dropped.
    With s = 0 nothing is a candidate.
    """
    if speed is None:
        speed = traces.get('speed')
    if speed is None:
        speed = estimate_body_speed(traces)
    t = np.asarray(traces['t'])
    out = {}
    for name, e in traces['eff'].items():
        vf = np.asarray(e['vf'])
        if speed > 0:
            mask = np.abs(vf + speed) <= tol * speed
        else:
            mask = np.zeros(len(vf), bool)
        out[name] = _runs_to_seconds(mask, t, min_ticks)
    return out


# ---- suggested events -------------------------------------------------------------------------------

def suggest_events(contacts, dt=DT):
    """takeoff, hand_plant_start, hand_plant_end, landing (seconds or None).

    hand_plant: first merged run of hand contacts (both hands, gaps <= HAND_MERGE_TICKS ticks).
    takeoff: end of the last foot contact that ends before the hand plant starts.
    landing: start of the first foot contact that starts after the hand plant ends.
    """
    gap = HAND_MERGE_TICKS * dt + 1e-9
    hand = sorted(iv for h in HANDS for iv in contacts.get(h, []))
    merged = []
    for a, b in hand:
        if merged and a <= merged[-1][1] + gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    ev = {'takeoff': None, 'hand_plant_start': None, 'hand_plant_end': None, 'landing': None}
    if not merged:
        return ev
    hs, he = merged[0]
    ev['hand_plant_start'], ev['hand_plant_end'] = hs, he
    feet = sorted(iv for f in FEET for iv in contacts.get(f, []))
    before = [b for a, b in feet if b < hs]
    if before:
        ev['takeoff'] = max(before)
    after = [a for a, b in feet if a > he]
    if after:
        ev['landing'] = min(after)
    return ev


# ---- plot -------------------------------------------------------------------------------------------

def _limb_pairs():
    return [('Reference', 'Hips'), ('Reference', 'Spine'), ('Spine', 'Spine1'), ('Spine1', 'Spine2'),
            ('Spine2', 'Neck'), ('Neck', 'Head'),
            ('Spine2', 'LeftArm'), ('LeftArm', 'LeftForeArm'), ('LeftForeArm', 'LeftHand'),
            ('Spine2', 'RightArm'), ('RightArm', 'RightForeArm'), ('RightForeArm', 'RightHand'),
            ('Hips', 'LeftUpLeg'), ('LeftUpLeg', 'LeftLeg'), ('LeftLeg', 'LeftFoot'), ('LeftFoot', 'LeftFootThumb1'),
            ('Hips', 'RightUpLeg'), ('RightUpLeg', 'RightLeg'), ('RightLeg', 'RightFoot'), ('RightFoot', 'RightFootThumb1')]


def plot_sheet(traces, contacts, events, out_path, strip_m_per_s=1.0, stick_every_s=0.1):
    """Side-view stick strip (top) and v_f / z_rel per effector (bottom), saved as PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = traces['t']
    f = traces['forward']
    joints = traces['joints']
    dur = t[-1]
    colors = plt.cm.tab10(np.arange(len(EFFECTORS)) % 10)
    fig = plt.figure(figsize=(16, 4.0 + 1.5 * len(EFFECTORS)), constrained_layout=False)
    gs = fig.add_gridspec(1 + len(EFFECTORS), 2, height_ratios=[2.6] + [1] * len(EFFECTORS),
                          hspace=0.55, wspace=0.12, left=0.05, right=0.99, top=0.95, bottom=0.04)

    # top: stick figures, pelvis-anchored, forward horizontal
    ax = fig.add_subplot(gs[0, :])
    step = max(1, int(round(stick_every_s / traces['dt'])))
    for k in range(0, len(t), step):
        off = t[k] * strip_m_per_s
        pts = {n: (float(np.dot(joints[n][k], f)) + off, float(joints[n][k, 2])) for n in JOINTS}
        for a, b in _limb_pairs():
            xa, za = pts[a]
            xb, zb = pts[b]
            ax.plot([xa, xb], [za, zb], '-', lw=1.2, color='k', alpha=0.55)
        ax.text(pts['Head'][0], pts['Head'][1] + 0.12, '%.1f' % t[k], fontsize=6, ha='center')
    ax.set_title('%s  (pelvis-anchored side view, strip offset %.1f m/s; forward axis %s)'
                 % (traces['stem'], strip_m_per_s, np.array2string(f, precision=2)), fontsize=9)
    ax.set_xlabel('forward (m) + t * %.1f m' % strip_m_per_s)
    ax.set_ylabel('z rel. pelvis (m)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=0.2)
    ev_lines = [('takeoff', 'k'), ('hand_plant_start', 'tab:green'), ('hand_plant_end', 'tab:olive'),
                ('landing', 'tab:red')]

    for i, name in enumerate(EFFECTORS):
        e = traces['eff'][name]
        for col, key in ((0, 'vf'), (1, 'z')):
            a = fig.add_subplot(gs[1 + i, col])
            a.plot(t, e[key], color=colors[i], lw=1.0)
            for c0, c1 in contacts.get(name, []):
                a.axvspan(c0, c1, color=colors[i], alpha=0.25, lw=0)
            for evn, evc in ev_lines:
                if events.get(evn) is not None:
                    a.axvline(events[evn], color=evc, lw=1.0, ls='--')
            if key == 'vf':
                s = traces.get('speed', 0.0)
                a.axhline(-s, color='0.4', lw=0.8, ls=':')
                if s > 0:
                    a.axhspan(-(1 + TOL) * s, -(1 - TOL) * s, color='0.8', alpha=0.4, lw=0)
                a.set_ylabel('%s\nv_f (m/s)' % name, fontsize=7)
            else:
                a.set_ylabel('z rel pelvis', fontsize=7)
            a.tick_params(labelsize=6)
            a.set_xlim(0, dur)
            a.grid(alpha=0.2)
            if i == len(EFFECTORS) - 1:
                a.set_xlabel('time (s)', fontsize=7)
    fig.suptitle('contacts: shaded per effector, suggested events dashed (takeoff black, hand plant green/olive, '
                 'landing red); speed %.2f m/s' % traces.get('speed', 0.0), fontsize=9)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# ---- CLI --------------------------------------------------------------------------------------------

def _fmt_iv(ivs):
    if not ivs:
        return '-'
    return ', '.join('%.2f-%.2f' % (a, b) for a, b in ivs)


def run_clip(stem, speed=None, strip_m_per_s=1.0, out_dir=None):
    traces = effector_traces(stem)
    if speed is not None:
        traces['speed'] = float(speed)
    contacts = candidate_contacts(traces)
    events = suggest_events(contacts, traces['dt'])
    out_dir = out_dir or os.path.join(config.OUT, 'sheets')
    png = plot_sheet(traces, contacts, events, os.path.join(out_dir, 'wd2_%s.png' % stem),
                     strip_m_per_s=strip_m_per_s)
    lines = []
    lines.append('== %s' % stem)
    lines.append('duration %.2f s (%d ticks at 30 Hz)' % (traces['dur'], len(traces['t'])))
    lines.append('forward axis (ACU world) %s   body speed %.2f m/s%s'
                 % (np.array2string(traces['forward'], precision=2),
                    traces['speed'], '' if speed is None else ' (override)'))
    for name in EFFECTORS:
        lines.append('  %-16s %s' % (name, _fmt_iv(contacts[name])))

    def fmt(x):
        return '%.2f s' % x if x is not None else 'none'
    lines.append('suggested events: takeoff %s | hand_plant_start %s | hand_plant_end %s | landing %s'
                 % (fmt(events['takeoff']), fmt(events['hand_plant_start']),
                    fmt(events['hand_plant_end']), fmt(events['landing'])))
    lines.append('plot: %s' % png)
    return '\n'.join(lines), traces, contacts, events


def main(argv=None):
    p = argparse.ArgumentParser(description='WD2 clip contact candidates and suggested events (rotations only).')
    p.add_argument('stems', nargs='+')
    p.add_argument('--speed', type=float, default=None, help='override the body speed (m/s)')
    p.add_argument('--strip-speed', type=float, default=1.0, help='stick strip offset (m per s)')
    args = p.parse_args(argv)
    for stem in args.stems:
        text, _, _, _ = run_clip(stem, speed=args.speed, strip_m_per_s=args.strip_speed)
        print(text)
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
