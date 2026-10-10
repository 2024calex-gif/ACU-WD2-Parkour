"""Blending helpers for pose sequences: crossfade, fade weights and event-based time warp.

All sequences are acuclip.PoseSeq (local_rot [F, B, 4], local_pos [F, B, 3], quaternions (x, y, z, w)).
Frame arguments of crossfade() and fade_weights() are indices into the sequence arrays (0 .. F-1).
Times of time_warp() are seconds; dst_frames are 60 Hz frame numbers.
"""
from dataclasses import replace

import numpy as np

import acuclip
import fk

FPS = acuclip.FPS


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _curve(t, curve):
    if curve == 'smoothstep':
        return _smoothstep(t)
    if curve == 'linear':
        return np.clip(t, 0.0, 1.0)
    raise ValueError("curve must be 'smoothstep' or 'linear', got %r" % (curve,))


def _continuity(q):
    """Flip quaternion signs so consecutive frames of each bone lie in the same hemisphere (same rotation)."""
    F, B = q.shape[0], q.shape[1]
    if F < 2:
        return q
    d = np.sum(q[1:] * q[:-1], axis=-1)
    s = np.where(d < 0, -1.0, 1.0)
    s = np.cumprod(np.concatenate([np.ones((1, B)), s], axis=0), axis=0)
    return q * s[..., None]


def crossfade(seq_a, seq_b, f0, f1, curve='smoothstep'):
    """Blend from seq_a (frames <= f0) to seq_b (frames >= f1), with a smoothstep weight in between.

    Per bone and frame: rotation = slerp(a, b, w), position = lerp(a, b, w), with w = 0 at f0 and 1 at f1.
    Both sequences must have the same frames and bones. Output rotations are sign-continuous, so a
    sequence whose quaternions were negated (same rotations) gives the same motion as its positive form;
    endpoints equal the inputs up to quaternion sign.
    """
    assert np.array_equal(seq_a.frames, seq_b.frames), 'crossfade: sequences have different frames'
    assert [b['hash'] for b in seq_a.bones] == [b['hash'] for b in seq_b.bones], \
        'crossfade: sequences have different bones'
    assert seq_a.local_rot.shape == seq_b.local_rot.shape, 'crossfade: shape mismatch'
    F = len(seq_a.frames)
    f0 = int(f0)
    f1 = int(f1)
    if not 0 <= f0 < f1 <= F - 1:
        raise ValueError('need 0 <= f0 < f1 <= %d, got f0=%d f1=%d' % (F - 1, f0, f1))
    idx = np.arange(F, dtype=float)
    w = _curve((idx - f0) / float(f1 - f0), curve)           # [F], 0 before f0, 1 after f1
    rot = fk.slerp(seq_a.local_rot, seq_b.local_rot, w[:, None])
    pa, pb = seq_a.local_pos, seq_b.local_pos
    pos = pa + (pb - pa) * w[:, None, None]
    return replace(seq_a, local_rot=_continuity(rot), local_pos=pos)


def fade_weights(n_frames, intervals, ramp):
    """Per-frame weight [n_frames]: 1 inside each [start, end] interval (inclusive indices).

    Outside the intervals the weight falls linearly from 1 to 0 over `ramp` frames: a frame at distance d
    from the nearest interval gets max(0, 1 - d / ramp). ramp = 0 gives a hard 0/1 mask. Overlapping
    intervals take the maximum.
    """
    n = int(n_frames)
    idx = np.arange(n, dtype=float)
    w = np.zeros(n)
    for start, end in intervals:
        s = float(start)
        e = float(end)
        if e < s:
            raise ValueError('interval [%s, %s] ends before it starts' % (start, end))
        d = np.where(idx < s, s - idx, np.where(idx > e, idx - e, 0.0))
        if ramp > 0:
            wi = np.clip(1.0 - d / float(ramp), 0.0, 1.0)
        else:
            wi = (d == 0).astype(float)
        w = np.maximum(w, wi)
    return w


def time_warp(seq, src_events, dst_events, dst_frames):
    """Re-time a sequence with a piecewise-linear map that sends dst_events onto src_events.

    src_events, dst_events: matching lists of times in seconds (increasing, same length >= 2). Each
    output time t_dst maps to t_src = interp(t_dst, dst_events, src_events); times outside the event
    range are clamped to the nearest event. seq is sampled at t_src by slerp (rotations) and lerp
    (positions) between its frames. dst_frames: 60 Hz frame numbers of the result.

    Example: src_events=[0, 1, 2], dst_events=[0, 2, 3] stretches the first second of seq to two seconds.
    """
    src = np.asarray(src_events, float)
    dst = np.asarray(dst_events, float)
    if src.shape != dst.shape or src.ndim != 1 or len(src) < 2:
        raise ValueError('src_events and dst_events must be 1-D lists of equal length >= 2')
    if np.any(np.diff(src) <= 0) or np.any(np.diff(dst) <= 0):
        raise ValueError('events must be strictly increasing')
    frames_out = np.asarray(dst_frames)
    if len(seq.frames) < 2:
        raise ValueError('seq needs at least 2 frames')

    t_src = np.interp(frames_out / float(FPS), dst, src)
    x = t_src * float(seq.fps)                                  # fractional frame number in seq.frames
    fr = np.asarray(seq.frames, float)
    F = len(fr)
    lo = np.clip(np.searchsorted(fr, x, side='right') - 1, 0, F - 2)
    u = np.clip((x - fr[lo]) / (fr[lo + 1] - fr[lo]), 0.0, 1.0)

    rot = fk.slerp(seq.local_rot[lo], seq.local_rot[lo + 1], u[:, None])
    pa = seq.local_pos[lo]
    pb = seq.local_pos[lo + 1]
    pos = pa + (pb - pa) * u[:, None, None]
    step = int(frames_out[1] - frames_out[0]) if len(frames_out) > 1 else 1
    return acuclip.PoseSeq(name=seq.name, fps=FPS, frames=frames_out, local_rot=_continuity(rot),
                           local_pos=pos, bones=seq.bones, step=step, hash_index=dict(seq.hash_index))
