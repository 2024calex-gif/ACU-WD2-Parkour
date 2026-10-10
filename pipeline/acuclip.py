"""Original ACU vault clips as pose sequences, in entity space and in world space.

Track meaning (verified on the 20 vault dumps, see PLAN_v2 / T5):
  * Reference (bone hash of 'Reference') has a rotation and a position track: its transform in
    entity space. It is the root of the skeleton; the bone record itself is ignored.
  * TrackID 0 (position): root displacement of the character along +Y (0 -> ~10 m). World = entity + disp.
  * 2307824085: obstacle near top edge (z = height, y = near edge in entity space).
  * 277334127: obstacle far top edge (depth = far.y - near.y).
  * 1736505593: point on the ground (z = 0) about 1 m beyond the far edge (landing spot).
  * 2246150967, 3215835963: both equal the INVERSE of the Reference transform (checked numerically:
    position = -conj(R_ref) * P_ref, rotation = conj(R_ref) up to sign). Not skeleton bones.
"""
import re
import zlib
from dataclasses import dataclass, field

import numpy as np

import dumps
import fk as fkmod
import skel

FPS = 60
REF_HASH = zlib.crc32(b'Reference')

# name -> TrackID of the position track holding the marker (entity space)
MARKER_TRACKS = {
    'disp': 0,
    'near_edge': 2307824085,
    'far_edge': 277334127,
    'landing': 1736505593,
    'm2246150967': 2246150967,
    'm3215835963': 3215835963,
}

VAULT_TAGS = ('SkipOver', 'SlideOver')


@dataclass
class PoseSeq:
    name: str
    fps: int
    frames: np.ndarray            # [F] frame numbers (60 Hz)
    local_rot: np.ndarray         # [F, B, 4] local rotations (x, y, z, w); bone 0 = Reference track
    local_pos: np.ndarray         # [F, B, 3] local positions; bone 0 = Reference track
    bones: list                   # ACU skeleton (skel.load_acu())
    step: int = 1
    hash_index: dict = field(default_factory=dict)


def is_vault(name):
    return any(tag in name for tag in VAULT_TAGS)


def parse_name(name):
    """(depth_m, height_m) from a clip name like '...Fd100cm-Up150cm'; None where absent."""
    fd = re.search(r'Fd(\d+)cm', name)
    up = re.search(r'Up(\d+)cm', name)
    return (int(fd.group(1)) / 100.0 if fd else None,
            int(up.group(1)) / 100.0 if up else None)


def vault_dumps():
    """[(handle, name, dump)] for every vault clip (names containing SkipOver or SlideOver)."""
    out = []
    for handle, name, _path in dumps.list_dumps():
        if is_vault(name):
            out.append((handle, name, dumps.load_by_handle(handle)))
    return out


def _continuity(q):
    """Flip quaternion signs so that consecutive frames of each bone stay in the same hemisphere."""
    F, B = q.shape[0], q.shape[1]
    if F < 2:
        return q
    d = np.sum(q[1:] * q[:-1], axis=-1)                 # [F-1, B]
    s = np.where(d < 0, -1.0, 1.0)
    s = np.cumprod(np.concatenate([np.ones((1, B)), s], axis=0), axis=0)
    return q * s[..., None]


def pose_sequence(dump, step=1):
    """PoseSeq of local transforms sampled every `step` frames from 0 to round(Length * 60).

    Bones with a track use the sampled track; bones without one use the skeleton bind pose.
    Bone 0 (Reference) carries its track transform in local_rot / local_pos.
    """
    bones = skel.load_acu()
    B = len(bones)
    last = int(round(float(dump['Length']) * FPS))
    frames = np.arange(0, last + 1, step)
    F = len(frames)
    lr = np.zeros((F, B, 4))
    lp = np.zeros((F, B, 3))
    for i, b in enumerate(bones):
        rk = dumps.track(dump, b['hash'], 0)
        pk = dumps.track(dump, b['hash'], 3)
        bind_r = np.array(b['lrot'], float)
        bind_r = bind_r / np.linalg.norm(bind_r)
        bind_p = np.array(b['lpos'], float)
        for j, f in enumerate(frames):
            lr[j, i] = dumps.sample_rotation(rk, int(f)) if rk else bind_r
            lp[j, i] = dumps.sample_position(pk, int(f)) if pk else bind_p
    lr = _continuity(lr)
    return PoseSeq(name=dump['Name'], fps=FPS, frames=frames, local_rot=lr, local_pos=lp,
                   bones=bones, step=step,
                   hash_index={b['hash']: i for i, b in enumerate(bones)})


def entity_transforms(seq):
    """World transforms in entity space: (rot [F,B,4], pos [F,B,3]).

    Reference's track transform is used once, as the root transform; its local is set to identity.
    """
    lr = seq.local_rot.copy()
    lp = seq.local_pos.copy()
    root_r = lr[:, 0].copy()
    root_p = lp[:, 0].copy()
    lr[:, 0] = (0.0, 0.0, 0.0, 1.0)
    lp[:, 0] = 0.0
    return fkmod.fk(seq.bones, lr, lp, root_rot=root_r, root_pos=root_p)


def entity_positions(seq):
    """Bone positions in entity space [F, B, 3]."""
    return entity_transforms(seq)[1]


def markers(dump, frames):
    """dict name -> [F, 3] raw marker positions (entity space) at the given frame numbers.

    A marker whose track is missing in the dump maps to None.
    """
    out = {}
    frames = np.asarray(frames)
    for name, tid in MARKER_TRACKS.items():
        keys = dumps.track(dump, tid, 3)
        if not keys:
            out[name] = None
            continue
        out[name] = np.array([dumps.sample_position(keys, int(f)) for f in frames], float)
    return out


def world_positions(seq, dump):
    """Bone positions in world space [F, B, 3]: entity positions + root displacement (TrackID 0)."""
    disp = markers(dump, seq.frames)['disp']
    return entity_positions(seq) + disp[:, None, :]


def obstacle(dump):
    """Obstacle box in world space from the edge markers (median over the whole clip).

    Returns dict(height, depth, near_y, far_y, near_x, far_z, landing_y, landing_z, spread) where
    spread holds the largest deviation (m) of near_y, near_z, far_y, far_z from their medians.
    """
    last = int(round(float(dump['Length']) * FPS))
    frames = np.arange(0, last + 1)
    m = markers(dump, frames)
    if m['near_edge'] is None or m['far_edge'] is None:
        raise ValueError('%s has no obstacle edge markers' % dump['Name'])
    disp = m['disp']
    near = m['near_edge'] + disp
    far = m['far_edge'] + disp
    landing = m['landing'] + disp if m['landing'] is not None else None

    def spread(v):
        return float(np.max(np.abs(v - np.median(v))))

    res = dict(
        height=float(np.median(near[:, 2])),
        depth=float(np.median(far[:, 1] - near[:, 1])),
        near_y=float(np.median(near[:, 1])),
        far_y=float(np.median(far[:, 1])),
        near_x=float(np.median(near[:, 0])),
        far_z=float(np.median(far[:, 2])),
        spread=dict(near_y=spread(near[:, 1]), near_z=spread(near[:, 2]),
                    far_y=spread(far[:, 1]), far_z=spread(far[:, 2])),
    )
    if landing is not None:
        res['landing_y'] = float(np.median(landing[:, 1]))
        res['landing_z'] = float(np.median(landing[:, 2]))
    return res
