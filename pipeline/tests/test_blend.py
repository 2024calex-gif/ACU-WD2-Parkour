"""Crossfade, fade weights and time warp (blend.py) on the real ACU skeleton and a real vault dump."""
import dataclasses

import numpy as np
import pytest

import acuclip
import blend
import dumps
import fk

CLIP = 'xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm'


@pytest.fixture(scope='module')
def seq():
    return acuclip.pose_sequence(dumps.load_by_name(CLIP))


def _same_rotations(qa, qb, tol=1e-9):
    """Quaternion arrays describe the same rotations (sign-insensitive)."""
    return bool(np.all(1.0 - np.abs(np.sum(qa * qb, axis=-1)) < tol))


def _max_step_deg(q):
    d = np.abs(np.sum(q[1:] * q[:-1], axis=-1)).clip(0.0, 1.0)
    return float(np.degrees(2.0 * np.arccos(d)).max())


def test_crossfade_endpoints_equal_inputs(seq):
    rng = np.random.default_rng(1)
    noise = rng.normal(scale=0.05, size=seq.local_rot.shape)
    other = fk.qnormalize(seq.local_rot + noise)
    seq_b = dataclasses.replace(seq, local_rot=other, local_pos=seq.local_pos + 0.01)
    f0, f1 = 20, 40
    out = blend.crossfade(seq, seq_b, f0, f1)

    assert _same_rotations(out.local_rot[f0], seq.local_rot[f0])
    assert _same_rotations(out.local_rot[f1], seq_b.local_rot[f1])
    assert np.allclose(out.local_pos[f0], seq.local_pos[f0], atol=1e-12)
    assert np.allclose(out.local_pos[f1], seq_b.local_pos[f1], atol=1e-12)
    assert _same_rotations(out.local_rot[:f0 + 1], seq.local_rot[:f0 + 1])
    assert _same_rotations(out.local_rot[f1:], seq_b.local_rot[f1:])
    assert np.array_equal(out.frames, seq.frames)
    assert out.local_rot.shape == seq.local_rot.shape


def test_crossfade_midpoint_is_smoothstep_half(seq):
    ident = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), seq.local_rot.shape).copy()
    seq_b = dataclasses.replace(seq, local_rot=ident, local_pos=np.zeros_like(seq.local_pos))
    f0, f1 = 10, 30
    out = blend.crossfade(seq, seq_b, f0, f1, curve='smoothstep')
    fm = (f0 + f1) // 2
    assert np.allclose(out.local_pos[fm], 0.5 * seq.local_pos[fm], atol=1e-12)


def test_crossfade_negated_quaternions_do_not_add_flips(seq):
    F = seq.local_rot.shape[0]
    sign = np.where(np.arange(F) % 2 == 0, 1.0, -1.0)[:, None, None]
    seq_neg = dataclasses.replace(seq, local_rot=seq.local_rot * sign)
    f0, f1 = 20, 60
    plain = blend.crossfade(seq, seq, f0, f1)
    flipped = blend.crossfade(seq, seq_neg, f0, f1)

    assert _same_rotations(flipped.local_rot, plain.local_rot)
    # output must be sign-continuous: no step bigger than the original clip's own largest step
    assert _max_step_deg(flipped.local_rot) <= _max_step_deg(seq.local_rot) + 1e-6
    # NOTE: the raw dump already has steps up to ~26 deg per frame, so the literal "< 10 deg" check cannot
    # hold for the whole clip; the check above is the sign-flip test.


def test_crossfade_rejects_mismatched_frames(seq):
    seq2 = acuclip.pose_sequence(dumps.load_by_name(CLIP), step=2)
    with pytest.raises(AssertionError):
        blend.crossfade(seq, seq2, 10, 20)


def test_crossfade_rejects_bad_curve(seq):
    with pytest.raises(ValueError):
        blend.crossfade(seq, seq, 10, 20, curve='cubic')


def test_fade_weights_ramp():
    w = blend.fade_weights(20, [(5, 8)], ramp=4)
    assert np.allclose(w[5:9], 1.0)
    assert np.allclose(w[[4, 3, 2, 1, 0]], [0.75, 0.5, 0.25, 0.0, 0.0])
    assert np.allclose(w[9:14], [0.75, 0.5, 0.25, 0.0, 0.0])
    assert np.all(w[14:] == 0.0)


def test_fade_weights_union_and_hard_mask():
    w = blend.fade_weights(12, [(1, 2), (6, 7)], ramp=0)
    assert np.array_equal(w, [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    w2 = blend.fade_weights(12, [(1, 2), (4, 4)], ramp=2)
    assert np.allclose(w2, [0.5, 1.0, 1.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_time_warp_identity_reproduces_sequence(seq):
    last = float(seq.frames[-1]) / acuclip.FPS
    ev = [0.0, 0.5, 1.0, last]
    out = blend.time_warp(seq, ev, ev, seq.frames)
    assert np.array_equal(out.frames, seq.frames)
    assert np.allclose(out.local_rot, seq.local_rot, atol=1e-9)
    assert _same_rotations(out.local_rot, seq.local_rot)
    assert np.allclose(out.local_pos, seq.local_pos, atol=1e-9)


def test_time_warp_doubled_segment_doubles_duration(seq):
    dst_frames = np.arange(0, 4 * acuclip.FPS + 1)
    src_events = [0.0, 1.0, 2.0]
    dst_events = [0.0, 2.0, 4.0]          # the first source second is played over two seconds
    out = blend.time_warp(seq, src_events, dst_events, dst_frames)

    # source time 1.0 s (frame 60) must now appear at destination time 2.0 s (frame 120)
    assert _same_rotations(out.local_rot[120], seq.local_rot[60])
    assert np.allclose(out.local_pos[120], seq.local_pos[60], atol=1e-9)
    # source 0.5 s appears at 1.0 s
    assert _same_rotations(out.local_rot[60], seq.local_rot[30])
    # the second source segment keeps its speed: source 1.5 s -> destination 3.0 s
    assert _same_rotations(out.local_rot[180], seq.local_rot[90])
    assert out.fps == acuclip.FPS
    assert np.array_equal(out.frames, dst_frames)


def test_time_warp_rejects_non_increasing_events(seq):
    with pytest.raises(ValueError):
        blend.time_warp(seq, [0.0, 1.0, 0.5], [0.0, 1.0, 2.0], np.arange(10))
    with pytest.raises(ValueError):
        blend.time_warp(seq, [0.0, 1.0], [0.0], np.arange(10))
