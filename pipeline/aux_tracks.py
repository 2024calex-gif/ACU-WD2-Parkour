"""Auxiliary tracks for the dive patches: foot-plant IK targets and the inverse of the Reference transform.

The bone patches (bake_dive / dive6) do not carry two groups of tracks that the original ACU clips have:

(a) Foot-plant IK targets, TrackID 4294967269 (left) and 4294967270 (right), infotype 3, LocalPosition.
    [10, 10, 10] is the sentinel "not planted"; any other value is a ground position in entity space.
    add_aux_tracks() sets the window [foot_free_from_frame, foot_free_to_frame) to the sentinel and keeps the
    original values elsewhere. Values are sampled per frame with the original's semantics: a segment that
    touches a sentinel key holds the previous key (step, no interpolation); two real keys are interpolated
    linearly. Keys are written at frame 0, at the last frame, at every frame whose value changes and at the
    frame before each change (holding the old value), so the game's linear interpolation between keys never
    produces an in-between value at an integer frame.

(b) TrackIDs 2246150967 and 3215835963 (each as rotation infotype 0 and position infotype 3). Both equal the
    INVERSE of the Reference transform at every frame. With the Reference transform X(x) = R x + P (rotate,
    then translate, entity space):

        R_inv = conj(R)                  (rotation; right-hand quaternion, (x, y, z, w))
        P_inv = -conj(R) * P             (position: minus the inverse rotation applied to P, NOT -P)

    so that X_inv(X(x)) = x. Verified numerically on the 20 vault dumps (tests/test_aux_tracks.py). The
    inverse is computed per frame 0..round(Length * 60) from the patch's own Reference tracks, sampled the
    way the game samples them (slerp for rotation, linear for position), with quaternion sign continuity.

CLI (writes a new patch; the input patch is not changed):
    python aux_tracks.py PATCH_IN DUMP PATCH_OUT --foot-free-from N [--foot-free-to M]
    python aux_tracks.py --self-check DUMP
"""
import argparse
import bisect
import copy
import json
import os

import numpy as np

import dumps
from retarget import qconj, qn, qrot

FPS = 60
REF_ID = 743623600                       # bone hash of 'Reference'
LEFT_FOOT_ID = 4294967269
RIGHT_FOOT_ID = 4294967270
FOOT_IDS = (LEFT_FOOT_ID, RIGHT_FOOT_ID)
INV_IDS = (2246150967, 3215835963)       # both equal the inverse of Reference
ROT_INFOTYPE = 0
POS_INFOTYPE = 3
ROT_CODEC = 1
POS_CODEC = 27
SENTINEL = (10.0, 10.0, 10.0)


def last_frame(length):
    return int(round(float(length) * FPS))


def _find_track(tracks, track_id, infotype):
    """Keyframes of the first track with this TrackID and infotype, or None."""
    for t in tracks:
        if int(t['TrackID']) == int(track_id) and t.get('infotype') == infotype:
            return t['Keyframes']
    return None


# ---- foot-plant targets ----------------------------------------------------------------------------

def _is_sentinel(p):
    return bool(np.allclose(p, SENTINEL, rtol=0.0, atol=1e-6))


def _pos(key):
    return np.array(key['LocalPosition'], float)


def sample_foot(keys, frame):
    """Foot target at an integer frame, with the original's sentinel semantics (see the module docstring).

    Before the first and after the last key the end value is held. Between keys a and b: frame == a.Frame gives a;
    if either key is the sentinel the previous key a is held; otherwise the position is interpolated linearly.
    """
    frames = [k['Frame'] for k in keys]
    if frame <= frames[0]:
        return _pos(keys[0])
    if frame >= frames[-1]:
        return _pos(keys[-1])
    i = bisect.bisect_right(frames, frame) - 1            # frames[i] <= frame < frames[i + 1]
    if frame == frames[i]:
        return _pos(keys[i])
    a, b = _pos(keys[i]), _pos(keys[i + 1])
    if _is_sentinel(a) or _is_sentinel(b):
        return a                                           # step: hold the previous key
    u = (frame - frames[i]) / (frames[i + 1] - frames[i])
    return a + (b - a) * u


def foot_values(keys, last, free_from, free_to=None):
    """Per-frame foot target for frames 0..last.

    The original's values (keys None = track missing = never planted), with [free_from, free_to) forced to the
    sentinel; free_to None means to the end of the clip.
    """
    sentinel = np.array(SENTINEL, float)
    out = []
    for f in range(last + 1):
        in_window = f >= free_from and (free_to is None or f < free_to)
        if in_window or keys is None:
            out.append(sentinel)
        else:
            out.append(sample_foot(keys, f))
    return out


def step_keys(values):
    """LocalPosition keyframes for per-frame values (index = frame, from 0).

    Keys at frame 0 and at the last frame, at every frame c whose value differs from frame c-1, and at c-1 (which
    holds the old value). Consecutive keys therefore differ by one frame only where the value changes.
    """
    last = len(values) - 1
    frames = {0, last}
    for c in range(1, last + 1):
        if not np.array_equal(values[c], values[c - 1]):
            frames.add(c - 1)
            frames.add(c)
    return [{'Frame': f, 'LocalPosition': [float(x) for x in values[f]]} for f in sorted(frames)]


# ---- inverse of Reference ----------------------------------------------------------------------------

def reference_inverse(patch):
    """Per-frame inverse of the patch's Reference transform, frames 0..round(Length * 60).

    Returns (rot [F, 4], pos [F, 3]): rot = conj(R), pos = -conj(R) * P, with R and P sampled from the patch's
    Reference rotation (infotype 0) and translation (infotype 3) tracks. R is sign-continuous across frames.
    """
    rk = _find_track(patch['Tracks'], REF_ID, ROT_INFOTYPE)
    pk = _find_track(patch['Tracks'], REF_ID, POS_INFOTYPE)
    if not rk or not pk:
        raise ValueError('patch has no Reference rotation/translation track (TrackID %d)' % REF_ID)
    last = last_frame(patch['Length'])
    rots = []
    for f in range(last + 1):
        r = qn(dumps.sample_rotation(rk, f))
        if rots and float(np.dot(r, rots[-1])) < 0.0:
            r = -r
        rots.append(r)
    pos = [dumps.sample_position(pk, f) for f in range(last + 1)]
    rot_inv = np.array([qconj(r) for r in rots], float)
    pos_inv = np.array([-qrot(qconj(r), p) for r, p in zip(rots, pos)], float)
    return rot_inv, pos_inv


def _inverse_keys(rot_inv, pos_inv):
    rot = [{'Frame': f, 'LocalRotation': [float(x) for x in rot_inv[f]]} for f in range(len(rot_inv))]
    pos = [{'Frame': f, 'LocalPosition': [float(x) for x in pos_inv[f]]} for f in range(len(pos_inv))]
    return rot, pos


def _track(track_id, infotype, keys):
    codec = ROT_CODEC if infotype == ROT_INFOTYPE else POS_CODEC
    return {'TrackID': str(int(track_id)), 'codec': codec, 'infotype': infotype, 'Keyframes': keys}


# ---- public API --------------------------------------------------------------------------------------

def aux_tracks(patch, dump, foot_free_from_frame, foot_free_to_frame=None):
    """The 6 new track dicts for this patch (2 foot targets, 2 inverse ids x rotation and position), not appended."""
    if foot_free_from_frame < 0:
        raise ValueError('foot_free_from_frame must be >= 0')
    if foot_free_to_frame is not None and foot_free_to_frame < foot_free_from_frame:
        raise ValueError('foot_free_to_frame must be None or >= foot_free_from_frame')
    last = last_frame(patch['Length'])
    out = []
    for tid in FOOT_IDS:
        keys = _find_track(dump['Tracks'], tid, POS_INFOTYPE)
        vals = foot_values(keys, last, foot_free_from_frame, foot_free_to_frame)
        out.append(_track(tid, POS_INFOTYPE, step_keys(vals)))
    rot_inv, pos_inv = reference_inverse(patch)
    rot_keys, pos_keys = _inverse_keys(rot_inv, pos_inv)
    for tid in INV_IDS:
        out.append(_track(tid, ROT_INFOTYPE, copy.deepcopy(rot_keys)))
        out.append(_track(tid, POS_INFOTYPE, copy.deepcopy(pos_keys)))
    return out


def add_aux_tracks(patch, dump, foot_free_from_frame, foot_free_to_frame=None):
    """A copy of the patch with the foot-plant and inverse-Reference tracks appended.

    Existing tracks are kept as they are. Raises ValueError if the patch already has one of the new
    (TrackID, infotype) pairs, so that no pair is ever duplicated.
    """
    new = aux_tracks(patch, dump, foot_free_from_frame, foot_free_to_frame)
    existing = {(int(t['TrackID']), t['infotype']) for t in patch['Tracks']}
    added = [(int(t['TrackID']), t['infotype']) for t in new]
    if len(set(added)) != len(added):
        raise ValueError('internal error: duplicate aux pairs')
    clash = sorted(set(added) & existing)
    if clash:
        raise ValueError('patch already has %s' % ', '.join('TrackID %d infotype %d' % p for p in clash))
    out = copy.deepcopy(patch)
    out['Tracks'] = list(out['Tracks']) + new
    return out


def self_check(dump):
    """Error of the inverse of the dump's own Reference against its original 2246150967 / 3215835963 tracks.

    Returns {TrackID: {'rot_at_keys', 'pos_at_keys', 'rot_all', 'pos_all'}} with the largest sign-aligned rotation
    error and largest position error (metres). '*_at_keys' compares at the original track's key frames, '*_all'
    at every frame (where the original's own linear interpolation of its keys can differ from the exact inverse).
    A TrackID whose original rotation or position track is missing in the dump is left out.
    """
    ref_only = {'Length': dump['Length'],
                'Tracks': [t for t in dump['Tracks'] if int(t['TrackID']) == REF_ID]}
    rot_inv, pos_inv = reference_inverse(ref_only)
    res = {}
    for tid in INV_IDS:
        rk = _find_track(dump['Tracks'], tid, ROT_INFOTYPE)
        pk = _find_track(dump['Tracks'], tid, POS_INFOTYPE)
        if not rk or not pk:
            continue
        kf_r = {k['Frame'] for k in rk}
        kf_p = {k['Frame'] for k in pk}
        e = dict(rot_at_keys=0.0, pos_at_keys=0.0, rot_all=0.0, pos_all=0.0)
        for f in range(len(rot_inv)):
            orr = dumps.sample_rotation(rk, f)
            er = float(min(np.abs(rot_inv[f] - orr).max(), np.abs(rot_inv[f] + orr).max()))
            ep = float(np.abs(pos_inv[f] - dumps.sample_position(pk, f)).max())
            e['rot_all'] = max(e['rot_all'], er)
            e['pos_all'] = max(e['pos_all'], ep)
            if f in kf_r:
                e['rot_at_keys'] = max(e['rot_at_keys'], er)
            if f in kf_p:
                e['pos_at_keys'] = max(e['pos_at_keys'], ep)
        res[tid] = e
    return res


def write_patch(path, patch):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(patch, f)


def _load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Add the foot-plant and inverse-Reference tracks to a dive patch.')
    ap.add_argument('patch_in', nargs='?', help='dive patch JSON (must hold the Reference tracks)')
    ap.add_argument('dump', nargs='?', help='original ACU clip dump JSON (source of the foot targets)')
    ap.add_argument('patch_out', nargs='?', help='new patch JSON to write')
    ap.add_argument('--foot-free-from', type=int, help='first frame with feet not planted (ACU takeoff)')
    ap.add_argument('--foot-free-to', type=int, default=None, help='first frame planted again (default: clip end)')
    ap.add_argument('--self-check', metavar='DUMP', help='print the inverse error against this original dump')
    args = ap.parse_args(argv)

    if args.self_check:
        for tid, e in self_check(_load_json(args.self_check)).items():
            print('TrackID %d: rot err at keys %.2e, pos err at keys %.2e (all frames: rot %.2e, pos %.2e) m' % (
                tid, e['rot_at_keys'], e['pos_at_keys'], e['rot_all'], e['pos_all']))
        return 0

    if not (args.patch_in and args.dump and args.patch_out) or args.foot_free_from is None:
        ap.error('patch_in, dump, patch_out and --foot-free-from are required')
    patch = _load_json(args.patch_in)
    out = add_aux_tracks(patch, _load_json(args.dump), args.foot_free_from, args.foot_free_to)
    write_patch(args.patch_out, out)
    print('wrote %s: %d tracks (%d added)' % (args.patch_out, len(out['Tracks']),
                                             len(out['Tracks']) - len(patch['Tracks'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
