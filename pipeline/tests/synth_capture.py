"""Synthetic WD2Vaults capture (format v2/CAPTURE_FORMAT.md) for the tests and for trying capture_report.py.

Scene (all times in s, 60 fps, 180 frames):
  - The entity sprints at 6 m/s along its +Y heading for 3 s. Z is up, the entity origin stays on z = 0.
  - The SkipOver clip starts at t = 0.5 (it is the trigger). The obstacle is placed from that clip's original dump:
    the marker tracks at frame 0 are mapped with the entity matrix at t = 0.5. That puts the near edge 3.59 m ahead
    of the entity at the clip start (not 4 m ahead of the capture start: 6 m/s over 0.5 s makes that impossible).
  - Feet alternate on the ground (50% duty, 0.4 s cycle). Both feet leave the ground at 0.9 s and land at 1.5 s.
  - Hands plant on the obstacle top 0.3 m past the near edge at 1.05 s, held for 0.2 s, moving in and out in 0.1 s.
  - The Muraille exit clip starts at 1.4 s.

write(path, ...) returns the ground truth dict used by the tests.
"""
import json
import math
import os
import zlib

import numpy as np

import capture
import clipnames
import dumps

FPS = 60
N_FRAMES = 180
SPEED = 6.0
ORIGIN = (10.0, -5.0, 0.0)
CLIP_SKIP = 'xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm'
CLIP_MURAILLE = 'xx_Parkour_Muraille_Exit_Fd100cm-Dn100cm'
T_SKIP = 0.5
T_MURAILLE = 1.4
T_TAKEOFF = 0.9
T_LAND = 1.5
T_PLANT = 1.05
PLANT_HOLD = 0.2
PLANT_RAMP = 0.1
HAND_REACH = 0.3        # hands plant this far past the near edge
HAND_LATERAL = 0.2
FOOT_LATERAL = 0.12
GROUND_FOOT = 0.06      # foot bone height above the ground while planted
STEP_PERIOD = 0.4


def _basis(heading_deg):
    h = math.radians(heading_deg)
    X = np.array([math.cos(h), math.sin(h), 0.0])
    Y = np.array([-math.sin(h), math.cos(h), 0.0])   # entity forward
    Z = np.array([0.0, 0.0, 1.0])
    return X, Y, Z


def _pack(X, Y, Z, T, convention):
    F = np.zeros(16)
    if convention == 'rows':
        F[0:3], F[4:7], F[8:11] = X, Y, Z
    elif convention == 'columns':
        F[[0, 4, 8]] = X
        F[[1, 5, 9]] = Y
        F[[2, 6, 10]] = Z
    else:
        raise ValueError(convention)
    F[12:15] = T
    F[15] = 1.0
    return F


def _smooth(u):
    u = min(1.0, max(0.0, u))
    return 0.5 - 0.5 * math.cos(math.pi * u)


def _ramp(t, t_a, t_b, ramp):
    """0 before t_a-ramp, linear to 1 at t_a, 1 until t_b, linear to 0 at t_b+ramp."""
    if t < t_a - ramp or t > t_b + ramp:
        return 0.0
    if t < t_a:
        return (t - (t_a - ramp)) / ramp
    if t <= t_b:
        return 1.0
    return 1.0 - (t - t_b) / ramp


def _marker_local(dump, tid):
    keys = dumps.track(dump, tid, 3)
    return np.asarray(dumps.sample_position(keys, 0), float)


def build(heading_deg=30.0, convention='rows', pose=True, patched=False, origin=ORIGIN):
    """(capture document dict, ground truth dict)."""
    X, Y, Z = _basis(heading_deg)
    O = np.asarray(origin, float)

    def along(t):                           # entity position at time t
        return O + SPEED * (t - T_SKIP) * Y

    skip_dump = dumps.load_by_handle('296F0C7D9')
    mur_handle = clipnames.load()[0][CLIP_MURAILLE]
    mur_dump = dumps.load_by_handle(mur_handle)

    # Obstacle: dump marker points at frame 0, mapped with the entity matrix at the clip start.
    near_l = _marker_local(skip_dump, capture.MARKER_NEAR)
    far_l = _marker_local(skip_dump, capture.MARKER_FAR)
    M_skip = _pack(X, Y, Z, along(T_SKIP), convention).reshape(4, 4)
    near_w = capture.apply_matrices(M_skip[None], near_l[None], row_major=(convention == 'rows'))[0]
    far_w = capture.apply_matrices(M_skip[None], far_l[None], row_major=(convention == 'rows'))[0]
    top = float(near_w[2])
    s_near = float(near_l[1])               # along-Y distance of the near edge from the clip-start entity position
    H_s = s_near + HAND_REACH               # hand plant point along the path (relative to the clip-start entity)

    def foot_s(side, t):
        """Position along +Y (relative to ORIGIN) of a foot in the ground phase."""
        phi0 = 0.0 if side == 'L' else STEP_PERIOD / 2
        c = phi0 + STEP_PERIOD * math.floor((t - phi0) / STEP_PERIOD)
        plant = lambda cs: SPEED * (cs - T_SKIP) + 0.5
        if t - c < STEP_PERIOD / 2:
            return plant(c), GROUND_FOOT
        u = (t - c - STEP_PERIOD / 2) / (STEP_PERIOD / 2)
        s0, s1 = plant(c), plant(c + STEP_PERIOD)
        return s0 + (s1 - s0) * _smooth(u), GROUND_FOOT + 0.25 * math.sin(math.pi * min(1.0, max(0.0, u)))

    def feet_world(t):
        out = {}
        for side in ('L', 'R'):
            lat = FOOT_LATERAL * X * (1 if side == 'L' else -1)
            if T_TAKEOFF <= t < T_LAND:
                u = (t - T_TAKEOFF) / (T_LAND - T_TAKEOFF)
                p = O + SPEED * (t - T_SKIP) * Y + lat
                p[2] = GROUND_FOOT + 1.4 * 4 * u * (1 - u)
            else:
                s, z = foot_s(side, t)
                p = O + s * Y + lat
                p[2] = z
            out[side] = p
        return out

    def hands_world(t):
        w = _ramp(t, T_PLANT, T_PLANT + PLANT_HOLD, PLANT_RAMP)
        out = {}
        for side in ('L', 'R'):
            lat = HAND_LATERAL * X * (1 if side == 'L' else -1)
            raised = O + (SPEED * (t - T_SKIP) + 0.6) * Y + lat + (top + 0.5) * Z
            planted = O + H_s * Y + lat + (top + 0.03) * Z
            out[side] = (1 - w) * raised + w * planted
        return out

    times = np.arange(N_FRAMES) / FPS
    local_bones = {n: [] for n in capture.BONE_NAMES}
    frames = []
    truth_pos = []
    truth_world = {n: [] for n in capture.BONE_NAMES}
    for t in times:
        P = along(t)
        P = P.copy()
        M = _pack(X, Y, Z, P, convention)
        truth_pos.append(P)
        F = {}
        F['Reference'] = P
        F['Hips'] = P + 0.95 * Z
        F['Spine2'] = P + 0.08 * Y + 1.25 * Z
        F['Head'] = P + 0.12 * Y + 1.6 * Z
        hw = hands_world(t)
        F['LeftHand'], F['RightHand'] = hw['L'], hw['R']
        fw = feet_world(t)
        F['LeftFoot'], F['RightFoot'] = fw['L'], fw['R']
        F['LeftFootThumb1'] = fw['L'] + 0.08 * Y
        F['RightFootThumb1'] = fw['R'] + 0.08 * Y
        frame = {'t': round(float(t), 6), 'entity_matrix': [float(v) for v in M]}
        if pose:
            bones = {}
            for n in capture.BONE_NAMES:
                w = F[n]
                truth_world[n].append(w)
                rel = w - P
                loc = np.array([rel @ X, rel @ Y, rel @ Z])
                bones[str(zlib.crc32(n.encode('ascii')))] = [float(v) for v in loc] + [0.0, 0.0, 0.0, 1.0]
            frame['bones'] = bones
        frames.append(frame)

    clip_skip = dict(t=T_SKIP, ptr='0x1A2B3C', handle='0x296F0C7D9', name=CLIP_SKIP,
                     length=float(skip_dump['Length']), patched=patched)
    clip_mur = dict(t=T_MURAILLE, ptr='0x1A2B5000', handle='0x' + mur_handle, name=CLIP_MURAILLE,
                    length=float(mur_dump['Length']) if mur_dump else 1.0, patched=False)
    doc = dict(format=capture.FORMAT_NAME, version=capture.FORMAT_VERSION, plugin_version='0.5.0',
               patching_enabled=bool(patched), variant='', trigger=CLIP_SKIP, pose_ok=bool(pose),
               pose_bones_missing=[] if pose else list(capture.BONE_NAMES),
               bones_space='entity' if pose else None, frames=frames, clips=[clip_skip, clip_mur])
    truth = dict(
        heading_deg=heading_deg, convention=convention, times=times,
        entity_pos=np.array(truth_pos),
        near=near_w, far=far_w, top=top, ground_z=0.0,
        hand_plant=T_PLANT, takeoff=T_TAKEOFF, landing=T_LAND,
        bones_world={n: np.array(v) for n, v in truth_world.items()} if pose else None,
    )
    return doc, truth


def write(path, **kwargs):
    """Write the synthetic capture JSON to path. Returns the ground truth dict."""
    doc, truth = build(**kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(doc, f)
    return truth


if __name__ == '__main__':
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synth.json')
    write(out)
    print('wrote', out)
