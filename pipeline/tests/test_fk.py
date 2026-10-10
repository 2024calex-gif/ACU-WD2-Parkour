"""Forward kinematics and quaternion helpers (fk.py)."""
import zlib

import numpy as np
import pytest

import dumps
import fk
import retarget
import skel


def _bind_locals(bones):
    lr = np.array([b['lrot'] for b in bones], float)
    lp = np.array([b['lpos'] for b in bones], float)
    lr[0] = (0.0, 0.0, 0.0, 1.0)     # root record is not a usable pose; identity at the origin
    lp[0] = 0.0
    return lr, lp


def test_bind_pose_reproduces_stored_globals():
    bones = skel.load_acu()
    lr, lp = _bind_locals(bones)
    WR, WP = fk.fk(bones, lr, lp)
    worst_pos = 0.0
    worst_rot = 0.0
    for i in range(1, len(bones)):
        g = np.array(bones[i]['gpos'], float)
        gr = np.array(bones[i]['grot'], float)
        gr = gr / np.linalg.norm(gr)
        pos_err = float(np.linalg.norm(WP[i] - g))
        rot_err = float(min(np.linalg.norm(WR[i] - gr), np.linalg.norm(WR[i] + gr)))   # sign ambiguity
        worst_pos = max(worst_pos, pos_err)
        worst_rot = max(worst_rot, rot_err)
        assert pos_err < 1e-4, 'bone %d position error %.2e' % (i, pos_err)
        assert rot_err < 1e-3, 'bone %d rotation error %.2e' % (i, rot_err)
    print('\nbind pose vs stored globals: worst pos %.2e, worst rot %.2e' % (worst_pos, worst_rot))


def test_batched_frames_match_single_frame():
    bones = skel.load_acu()
    lr, lp = _bind_locals(bones)
    rng = np.random.default_rng(0)
    F = 4
    lrs = np.repeat(lr[None], F, axis=0)
    lps = np.repeat(lp[None], F, axis=0)
    noise = rng.normal(scale=0.05, size=(F, len(bones), 3))
    perturb = np.concatenate([noise, np.zeros((F, len(bones), 1))], axis=-1)
    lrs[:, 1:] = fk.qnormalize(lrs[:, 1:] + perturb[:, 1:])
    root_rot = np.array([fk.qnormalize([0.1 * f, 0.0, 0.2, 1.0]) for f in range(F)])
    root_pos = np.arange(F * 3, dtype=float).reshape(F, 3)
    WRb, WPb = fk.fk(bones, lrs, lps, root_rot=root_rot, root_pos=root_pos)
    for f in range(F):
        WR1, WP1 = fk.fk(bones, lrs[f], lps[f], root_rot=root_rot[f], root_pos=root_pos[f])
        assert np.allclose(WRb[f], WR1, atol=1e-12)
        assert np.allclose(WPb[f], WP1, atol=1e-12)


def test_root_transform_is_applied_once_and_composed_with_local():
    bones = [dict(parent=-1), dict(parent=0), dict(parent=1)]
    # 90 degrees about z: (x, y) -> (-y, x)
    rz = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    lr = np.array([(0, 0, 0, 1.0), (0, 0, 0, 1.0), (0, 0, 0, 1.0)])
    lp = np.array([(0, 0, 0), (1.0, 0, 0), (1.0, 0, 0)])
    WR, WP = fk.fk(bones, lr, lp, root_rot=rz, root_pos=(10.0, 0.0, 0.0))
    assert np.allclose(WP[0], (10.0, 0.0, 0.0))
    assert np.allclose(WP[1], (10.0, 1.0, 0.0), atol=1e-12)        # child at +x rotated to +y
    assert np.allclose(WP[2], (10.0, 2.0, 0.0), atol=1e-12)
    assert np.allclose(WR[2], rz, atol=1e-12)


def test_parent_after_child_is_rejected():
    bones = [dict(parent=1), dict(parent=-1)]
    with pytest.raises(ValueError):
        fk.fk(bones, np.zeros((2, 4)) + [0, 0, 0, 1], np.zeros((2, 3)))


def test_quaternion_helpers_match_retarget():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = fk.qnormalize(rng.normal(size=4))
        b = fk.qnormalize(rng.normal(size=4))
        v = rng.normal(size=3)
        assert np.allclose(fk.qmul(a, b), retarget.qmul(a, b), atol=1e-12)
        assert np.allclose(fk.qrot(a, v), retarget.qrot(a, v), atol=1e-12)
        assert np.allclose(fk.slerp(a, b, 0.0), retarget.qn(a), atol=1e-12)
        assert np.allclose(fk.slerp(a, b, 0.5), retarget.slerp(a, b, 0.5), atol=1e-9)
    assert np.allclose(fk.qmul([0, 0, 0, 1.0], [0.1, 0.2, 0.3, 0.9]), [0.1, 0.2, 0.3, 0.9])


def test_entity_reference_transform_matches_track_exactly():
    """Reference's track transform is the root transform: entity-space bone 0 equals the track sample."""
    import acuclip
    d = dumps.load_by_name('xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm')
    seq = acuclip.pose_sequence(d, step=6)
    R, P = acuclip.entity_transforms(seq)
    assert np.allclose(P[:, 0], seq.local_pos[:, 0], atol=1e-12)
    for f in range(len(seq.frames)):
        q = seq.local_rot[f, 0]
        assert min(np.linalg.norm(R[f, 0] - q), np.linalg.norm(R[f, 0] + q)) < 1e-9
