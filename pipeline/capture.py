"""Reader and offline analysis for WD2Vaults flight-recorder captures (format: v2/CAPTURE_FORMAT.md, version 1).

Entity matrix convention: the matrix is stored in memory order and reshaped to [4,4] (row-major).
Translation is always elements 12-14. The basis reading is selected by ROW_MAJOR_BASIS:
  True  -> axes are the rows: X = floats 0-2, Y = floats 4-6, Z = floats 8-10
  False -> axes are the columns: X = floats 0,4,8, Y = floats 1,5,9, Z = floats 2,6,10
check_convention(cap) tests both readings against the movement of the entity (forward = entity +Y).

Contact detection (events) allows the bone to sit CONTACT_BELOW under the support as well as CONTACT_ABOVE over it,
because animated bones jitter a few cm around the support plane.
"""
import json
import os
import zlib
from dataclasses import dataclass, field

import numpy as np

import clipnames

FORMAT_NAME = 'wd2vaults-capture'
FORMAT_VERSION = 1

# Convention used by to_world / bone_world / entity_axes when no explicit row_major is passed.
ROW_MAJOR_BASIS = True

BONE_NAMES = ('Reference', 'Hips', 'Spine2', 'Head', 'LeftHand', 'RightHand', 'LeftFoot', 'RightFoot',
              'LeftFootThumb1', 'RightFootThumb1')
BONE_IDS = {zlib.crc32(n.encode('ascii')): n for n in BONE_NAMES}

# Marker tracks of an ACU vault clip (frame 0, entity space): obstacle near and far edge.
MARKER_NEAR = 2307824085
MARKER_FAR = 277334127

HAND_BONES = ('LeftHand', 'RightHand')
FOOT_BONES = ('LeftFoot', 'RightFoot')

CONTACT_ABOVE = 0.12      # m above the support plane
CONTACT_BELOW = 0.05      # m below it (animation jitter tolerance)
CONTACT_SPEED = 1.0       # m/s, world speed of the bone
CONVENTION_MIN_SPEED = 2.0  # m/s, horizontal entity speed used by check_convention
CONVENTION_TOL_DEG = 30.0   # a reading "fits" when the median angle to movement is below this


@dataclass
class Capture:
    path: str
    trigger: str
    patching_enabled: bool
    pose_ok: bool            # every frame has all bones (plugin 0.5+); False for partial poses
    bones_space: object
    plugin_version: object
    t: np.ndarray            # [N]
    M: np.ndarray            # [N,4,4] entity matrices (memory order reshaped to 4x4)
    entity_pos: np.ndarray   # [N,3] translation (elements 12-14)
    bones: dict              # bone name -> [N,7] (px,py,pz,qx,qy,qz,qw), NaN in frames without that bone; None if never seen
    clips: list              # dicts: t, ptr, handle, name (filled from the index), length, patched
    variant: str = ''        # patch variant active when recorded ('' = top-level files; absent in older files)
    pose_bones_missing: list = field(default_factory=list)   # bone names missing in at least one frame


@dataclass
class Obstacle:
    near: np.ndarray   # world point of the near edge, (3,)
    far: np.ndarray    # world point of the far edge, (3,)
    top_z: float       # world z of the obstacle top
    ground_z: float    # entity z at the trigger frame
    height: float      # top_z - ground_z
    depth: float       # horizontal distance near -> far
    t0: float          # trigger clip start (s)
    clip: str
    handle: object


# ---------------------------------------------------------------------------------------------- loading

_CLIP_INDEX = None


def _clip_by_id():
    global _CLIP_INDEX
    if _CLIP_INDEX is None:
        _CLIP_INDEX = clipnames.load()[1]
    return _CLIP_INDEX


def _handle_int(h):
    if h is None:
        return None
    try:
        return int(str(h).strip(), 16)
    except ValueError:
        return None


def _load_clips(raw):
    by_id = _clip_by_id()
    out = []
    for c in raw:
        d = dict(c)
        d['t'] = float(c['t'])
        d['patched'] = bool(c.get('patched', False))
        if not d.get('name'):
            hid = _handle_int(d.get('handle'))
            d['name'] = by_id.get(hid) if hid is not None else None
        out.append(d)
    out.sort(key=lambda d: d['t'])
    return out


def load(path):
    """Read one capture JSON. Bones are keyed by name: [N,7] with NaN in the frames that lack the bone, or None when
    no frame has it. Partial poses (some bones missing) load fine; pose_ok only says whether every bone was there."""
    with open(path, encoding='utf-8') as f:
        doc = json.load(f)
    if doc.get('format') != FORMAT_NAME or doc.get('version') != FORMAT_VERSION:
        raise ValueError('%s is not a %s v%d file' % (path, FORMAT_NAME, FORMAT_VERSION))
    frames = doc.get('frames') or []
    if not frames:
        raise ValueError('%s has no frames' % path)
    n = len(frames)
    t = np.array([float(f['t']) for f in frames], float)
    M = np.empty((n, 4, 4), float)
    for i, f in enumerate(frames):
        m = f['entity_matrix']
        if len(m) != 16:
            raise ValueError('frame %d of %s: entity_matrix has %d values' % (i, path, len(m)))
        M[i] = np.asarray(m, float).reshape(4, 4)
    entity_pos = M[:, 3, 0:3].copy()

    # Partial poses: each bone is read from every frame that has it (a frame's "bones" may omit some bones).
    # A bone that is missing in some frames is NaN there; a bone that is never present is None.
    pose_ok = bool(doc.get('pose_ok', False))
    if 'pose_bones_missing' in doc:
        pose_bones_missing = [str(x) for x in (doc.get('pose_bones_missing') or [])]
    else:   # older files (no per-bone record): a pose-less capture misses every bone
        pose_bones_missing = [] if pose_ok else list(BONE_NAMES)
    bones = {}
    for name in BONE_NAMES:
        key = str(zlib.crc32(name.encode('ascii')))
        arr = np.full((n, 7), np.nan)
        seen = False
        for i, f in enumerate(frames):
            b = (f.get('bones') or {}).get(key)
            if b is not None:
                arr[i] = np.asarray(b, float)
                seen = True
        bones[name] = arr if seen else None

    return Capture(
        path=os.path.abspath(path),
        trigger=doc.get('trigger'),
        patching_enabled=bool(doc.get('patching_enabled', False)),
        pose_ok=pose_ok,
        bones_space=doc.get('bones_space'),
        plugin_version=doc.get('plugin_version'),
        variant=str(doc.get('variant') or ''),
        pose_bones_missing=pose_bones_missing,
        t=t, M=M, entity_pos=entity_pos, bones=bones,
        clips=_load_clips(doc.get('clips') or []),
    )


# ---------------------------------------------------------------------------------------------- matrices

def entity_axes(M, row_major=None):
    """(X, Y, Z, T) arrays [N,3] of an entity matrix array [N,4,4] (or one matrix [4,4]).

    row_major None -> ROW_MAJOR_BASIS. Translation is always elements 12-14.
    """
    if row_major is None:
        row_major = ROW_MAJOR_BASIS
    M = np.asarray(M, float)
    if M.ndim == 2:
        M = M[None]
    if row_major:
        X = M[:, 0, 0:3]
        Y = M[:, 1, 0:3]
        Z = M[:, 2, 0:3]
    else:
        X = M[:, 0:3, 0]
        Y = M[:, 0:3, 1]
        Z = M[:, 0:3, 2]
    T = M[:, 3, 0:3]
    return X, Y, Z, T


def apply_matrices(M, local_points, row_major=None):
    """World points [N,3] = local.x*X + local.y*Y + local.z*Z + T, one matrix per point."""
    X, Y, Z, T = entity_axes(M, row_major)
    p = np.asarray(local_points, float).reshape(-1, 3)
    return p[:, 0:1] * X + p[:, 1:2] * Y + p[:, 2:3] * Z + T


def to_world(cap, local_points, row_major=None):
    """Entity-space points [N,3] -> world points [N,3], using the entity matrix of each frame."""
    return apply_matrices(cap.M, local_points, row_major)


def bone_world(cap, name, row_major=None):
    """World positions [N,3] of a recorded bone, or None when the bone is not in the capture."""
    arr = cap.bones.get(name)
    if arr is None:
        return None
    if cap.bones_space != 'entity':
        raise ValueError('bone_world needs bones_space == "entity", capture has %r' % (cap.bones_space,))
    return to_world(cap, arr[:, 0:3], row_major)


def _angle_deg(a, b):
    """Angle in degrees between row vectors a [N,2] and b [N,2] (NaN where a norm is zero)."""
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        cos = np.sum(a * b, axis=1) / (na * nb)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def check_convention(cap, min_speed=CONVENTION_MIN_SPEED, tol_deg=CONVENTION_TOL_DEG):
    """Which matrix reading puts entity +Y along the horizontal movement.

    Uses frames where the entity moves faster than min_speed. Candidates: +Y as the matrix rows
    (floats 4-6) and +Y as the columns (floats 1,5,9). Verdict 'rows' or 'columns' when exactly one
    candidate is within tol_deg (median angle); 'ambiguous' when both are (heading near 0 degrees,
    where the two readings coincide); 'neither'; 'insufficient motion' when too few frames move.
    """
    res = dict(n_frames=0, rows_deg=None, cols_deg=None, verdict='insufficient motion',
               row_major_fits=False, matches_constant=False)
    v = np.gradient(cap.entity_pos[:, 0:2], cap.t, axis=0)
    sp = np.linalg.norm(v, axis=1)
    mask = np.isfinite(sp) & (sp > min_speed)
    res['n_frames'] = int(mask.sum())
    if mask.sum() < 3:
        return res
    mdir = v[mask] / sp[mask, None]
    _, Y_rows, _, _ = entity_axes(cap.M[mask], row_major=True)
    _, Y_cols, _, _ = entity_axes(cap.M[mask], row_major=False)
    rows_deg = float(np.nanmedian(_angle_deg(Y_rows[:, 0:2], mdir)))
    cols_deg = float(np.nanmedian(_angle_deg(Y_cols[:, 0:2], mdir)))
    good_r = rows_deg < tol_deg
    good_c = cols_deg < tol_deg
    if good_r and good_c:
        verdict = 'ambiguous'
    elif good_r:
        verdict = 'rows'
    elif good_c:
        verdict = 'columns'
    else:
        verdict = 'neither'
    res.update(rows_deg=rows_deg, cols_deg=cols_deg, verdict=verdict,
               row_major_fits=(verdict == 'rows'),
               matches_constant=(verdict == ('rows' if ROW_MAJOR_BASIS else 'columns')))
    return res


# ---------------------------------------------------------------------------------------------- clips

def trigger_clip(cap):
    """First clip whose name is the capture trigger, or None."""
    for c in cap.clips:
        if c['name'] == cap.trigger:
            return c
    return None


def clip_segments(cap):
    """[(name, t_start, t_end)] in start order; t_end = next clip start, last one ends at the capture end.

    Clips can overlap (blends); each segment still runs to the next start.
    """
    out = []
    end = float(cap.t[-1])
    clips = cap.clips
    for i, c in enumerate(clips):
        name = c['name'] or ('handle %s' % c.get('handle') if c.get('handle') else '<unknown>')
        t_end = clips[i + 1]['t'] if i + 1 < len(clips) else end
        out.append((name, c['t'], t_end))
    return out


def obstacle_from_clip(cap, dumps_module, row_major=None):
    """Obstacle in world space from the trigger clip's original dump, or None when there is no dump.

    Marker tracks at frame 0 are entity-space points; they are mapped with the entity matrix at the
    trigger start t0 (nearest captured frame).
    """
    clip = trigger_clip(cap)
    if clip is None:
        return None
    dump = None
    if clip.get('handle'):
        dump = dumps_module.load_by_handle(clip['handle'])
    if dump is None and clip.get('name'):
        dump = dumps_module.load_by_name(clip['name'])
    if dump is None:
        return None

    def marker(tid):
        keys = dumps_module.track(dump, tid, 3)
        if not keys:
            return None
        return np.asarray(dumps_module.sample_position(keys, 0), float)

    near_l = marker(MARKER_NEAR)
    far_l = marker(MARKER_FAR)
    if near_l is None or far_l is None:
        return None
    i = int(np.argmin(np.abs(cap.t - clip['t'])))
    world = apply_matrices(cap.M[[i, i]], np.vstack([near_l, far_l]), row_major)
    near, far = world[0], world[1]
    ground_z = float(cap.entity_pos[i, 2])
    top_z = float((near[2] + far[2]) / 2.0)
    return Obstacle(
        near=near, far=far, top_z=top_z, ground_z=ground_z,
        height=top_z - ground_z,
        depth=float(np.linalg.norm(far[0:2] - near[0:2])),
        t0=float(clip['t']), clip=clip['name'], handle=clip.get('handle'),
    )


# ---------------------------------------------------------------------------------------------- events

def _runs(mask):
    runs = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def contact_intervals(cap, name, obstacle, ground_z, row_major=None):
    """Contact intervals of one bone: dicts start, end, on_obstacle, u_start, bone.

    A frame is in contact when the bone is within [-CONTACT_BELOW, CONTACT_ABOVE] of its support and its
    world speed is below CONTACT_SPEED. Support = obstacle top when the bone is horizontally between the
    obstacle edges (projection on near->far in [0,1]), else ground_z. Returns None when the bone is missing.
    """
    P = bone_world(cap, name, row_major)
    if P is None or len(P) < 3:
        return None if P is None else []
    speed = np.linalg.norm(np.gradient(P, cap.t, axis=0), axis=1)
    n = len(P)
    if obstacle is not None:
        N2 = obstacle.near[0:2]
        d = obstacle.far[0:2] - N2
        u = ((P[:, 0:2] - N2) @ d) / float(d @ d)
        on = (u >= 0.0) & (u <= 1.0)
        support = np.where(on, obstacle.top_z, ground_z)
    else:
        u = np.full(n, np.nan)
        on = np.zeros(n, bool)
        support = np.full(n, ground_z)
    dz = P[:, 2] - support
    contact = (dz <= CONTACT_ABOVE) & (dz >= -CONTACT_BELOW) & (speed < CONTACT_SPEED)
    out = []
    for a, b in _runs(contact):
        out.append(dict(start=float(cap.t[a]), end=float(cap.t[b]),
                        on_obstacle=bool(on[a:b + 1].mean() >= 0.5),
                        u_start=float(u[a]), bone=name))
    return out


def events(cap, obstacle=None, row_major=None):
    """Timeline events in seconds.

    Always returns trigger_start, clip_starts (name, t). With a usable pose it also returns:
      hand_plant  first hand contact on the obstacle top
      hand_release  end of the last hand contact on the obstacle (within 0.5 s of the plant)
      takeoff     end of the last foot contact before the hand plant
      landing     start of the first foot contact after the hand release, beyond the far edge
    Without a pose (or without an obstacle for hand_plant) those are None.
    """
    tr = trigger_clip(cap)
    t0 = tr['t'] if tr is not None else float(cap.t[0])
    out = dict(trigger_start=t0,
               clip_starts=[(c['name'], c['t']) for c in cap.clips],
               pose=False, hand_plant=None, hand_release=None, takeoff=None, landing=None, contacts={})
    # Partial poses count: contacts are found for the hand and foot bones that were recorded (NaN frames never touch).
    if cap.bones_space != 'entity':
        return out
    if all(cap.bones.get(b) is None for b in HAND_BONES + FOOT_BONES):
        return out
    out['pose'] = True
    gi = int(np.argmin(np.abs(cap.t - t0)))
    ground_z = float(cap.entity_pos[gi, 2])

    hands, feet = [], []
    for name in HAND_BONES:
        iv = contact_intervals(cap, name, obstacle, ground_z, row_major) or []
        out['contacts'][name] = iv
        hands += iv
    for name in FOOT_BONES:
        iv = contact_intervals(cap, name, obstacle, ground_z, row_major) or []
        out['contacts'][name] = iv
        feet += iv

    on_obs = [h for h in hands if h['on_obstacle']]
    if not on_obs:
        return out
    plant = min(h['start'] for h in on_obs)
    release = max(h['end'] for h in on_obs if h['start'] <= plant + 0.5)
    out['hand_plant'] = plant
    out['hand_release'] = release
    before = [f for f in feet if f['end'] <= plant]
    out['takeoff'] = max(f['end'] for f in before) if before else None
    after = [f for f in feet if f['start'] > release and not f['on_obstacle'] and f['u_start'] > 1.0]
    out['landing'] = min(f['start'] for f in after) if after else None
    return out
