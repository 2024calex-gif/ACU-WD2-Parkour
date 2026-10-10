"""Contact events from a pose sequence (hands / feet in world space).

A point is in contact with a support surface on a frame when:
  * its height above the support is within [-CONTACT_BELOW, CONTACT_ABOVE], and
  * its smoothed world speed is below SPEED_MAX.
The support is the obstacle top (z = height) when the point's world y lies in
[near_y - SUPPORT_BAND, far_y + SUPPORT_BAND], otherwise the ground (z = 0).

Hands are the Hand bone (the palm offset is not modelled). Feet use the Foot bone and the Toe bone
(LeftFootThumb1 / RightFootThumb1) as separate points.
"""
import zlib

import numpy as np

import acuclip

FPS = acuclip.FPS
SUPPORT_BAND = 0.15       # m, half-width of the y band around the obstacle that counts as "on the wall"
CONTACT_ABOVE = 0.12      # m, a point this high above its support is in contact
CONTACT_BELOW = 0.03      # m, tolerance for joint offsets below the surface
SPEED_MAX = 1.0           # m/s, contact requires the point to be (almost) still
SPEED_WINDOW = 3          # frames, moving-average window for the speed
MIN_CONTACT_FRAMES = 2    # shorter contact runs are dropped as noise

HANDS = ('LeftHand', 'RightHand')
FEET = ('LeftFoot', 'RightFoot', 'LeftFootThumb1', 'RightFootThumb1')
POINTS = HANDS + FEET


def smooth_speed(points, fps):
    """Speed (m/s) per frame of a [F, 3] trajectory, central differences then a moving average."""
    points = np.asarray(points, float)
    v = np.gradient(points, axis=0) * fps
    s = np.linalg.norm(v, axis=1)
    k = SPEED_WINDOW
    pad = k // 2
    sp = np.pad(s, pad, mode='edge')
    return np.convolve(sp, np.ones(k) / k, mode='valid')


def point_masks(points, near_y, far_y, height, fps):
    """(top, ground) boolean arrays [F] for one point trajectory [F, 3]."""
    points = np.asarray(points, float)
    y = points[:, 1]
    z = points[:, 2]
    in_band = (y >= near_y - SUPPORT_BAND) & (y <= far_y + SUPPORT_BAND)
    support_z = np.where(in_band, height, 0.0)
    h = z - support_z
    close = (h <= CONTACT_ABOVE) & (h >= -CONTACT_BELOW)
    still = smooth_speed(points, fps) < SPEED_MAX
    c = close & still
    return c & in_band, c & ~in_band


def runs(mask, frames):
    """Runs of True in mask as [start_frame, end_frame] (inclusive, frame numbers from `frames`)."""
    out = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m or i == len(mask) - 1) and start is not None:
            end = i if m else i - 1
            if end - start + 1 >= MIN_CONTACT_FRAMES:
                out.append([int(frames[start]), int(frames[end])])
            start = None
    return out


def _seconds(frame):
    return None if frame is None else round(float(frame) / FPS, 4)


def events(seq, dump):
    """Contacts and summary events for one pose sequence of a vault clip.

    Returns dict with:
      obstacle      acuclip.obstacle(dump)
      contacts      {point: [[start, end, 'top' | 'ground'], ...]} (frame numbers)
      hand_plant    {'frame', 'seconds', 'hand'} first hand contact on the obstacle top
      foot_on_top   {'frame', 'seconds', 'foot'} first foot contact on the obstacle top
      takeoff       {'frame', 'seconds', 'foot'} end of the last foot ground contact before the near edge
      landing       {'frame', 'seconds', 'foot'} first foot ground contact after the far edge
      near_frame / far_frame  first frame the lead foot reaches the near / far edge
    Each summary is None when the event does not occur.
    """
    ob = acuclip.obstacle(dump)
    world = acuclip.world_positions(seq, dump)
    idx = {name: seq.hash_index[zlib.crc32(name.encode())] for name in POINTS}
    fps = FPS / max(1, seq.step)
    frames = seq.frames

    contacts = {}
    top_runs = {}
    ground_runs = {}
    for name in POINTS:
        pts = world[:, idx[name]]
        top, ground = point_masks(pts, ob['near_y'], ob['far_y'], ob['height'], fps)
        t_runs = runs(top, frames)
        g_runs = runs(ground, frames)
        top_runs[name] = t_runs
        ground_runs[name] = g_runs
        contacts[name] = [[s, e, 'top'] for s, e in t_runs] + [[s, e, 'ground'] for s, e in g_runs]
        contacts[name].sort(key=lambda r: r[0])

    feet_y = np.max(np.stack([world[:, idx[n], 1] for n in FEET], axis=1), axis=1)
    near_hits = np.nonzero(feet_y >= ob['near_y'])[0]
    far_hits = np.nonzero(feet_y > ob['far_y'])[0]
    near_frame = int(frames[near_hits[0]]) if len(near_hits) else None
    far_frame = int(frames[far_hits[0]]) if len(far_hits) else None

    # hand plant: earliest top contact of any hand
    hp = [(top_runs[name][0][0], name) for name in HANDS if top_runs[name]]
    hand_plant = None
    if hp:
        f, name = min(hp, key=lambda t: (t[0], HANDS.index(t[1])))
        hand_plant = dict(frame=f, seconds=_seconds(f), hand=name)

    # foot on top: earliest top contact of any foot point
    ft = [(top_runs[name][0][0], name) for name in FEET if top_runs[name]]
    foot_on_top = None
    if ft:
        f, name = min(ft, key=lambda t: (t[0], FEET.index(t[1])))
        foot_on_top = dict(frame=f, seconds=_seconds(f), foot=name)

    # take-off: end of the last foot ground contact that starts before the lead foot reaches the near edge
    takeoff = None
    if near_frame is not None:
        cand = []
        for name in FEET:
            for s, e in ground_runs[name]:
                if s < near_frame:
                    cand.append((min(e, near_frame), name))
        if cand:
            f, name = max(cand, key=lambda t: t[0])
            takeoff = dict(frame=f, seconds=_seconds(f), foot=name)

    # landing: first foot ground contact that starts after the lead foot passes the far edge
    landing = None
    if far_frame is not None:
        cand = []
        for name in FEET:
            for s, _e in ground_runs[name]:
                if s >= far_frame:
                    cand.append((s, name))
        if cand:
            f, name = min(cand, key=lambda t: (t[0], FEET.index(t[1])))
            landing = dict(frame=f, seconds=_seconds(f), foot=name)

    return dict(obstacle=ob, contacts=contacts, hand_plant=hand_plant, foot_on_top=foot_on_top,
                takeoff=takeoff, landing=landing, near_frame=near_frame, far_frame=far_frame)

