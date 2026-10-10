"""Two-bone IK (ik.py) on the real ACU skeleton and a real vault dump pose."""
import dataclasses

import numpy as np
import pytest

import acuclip
import dumps
import fk
import ik

CLIP = 'xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm'
EFFECTORS = ('LeftHand', 'RightHand', 'LeftFoot', 'RightFoot')
IDENT = (0.0, 0.0, 0.0, 1.0)


@pytest.fixture(scope='module')
def seq():
    return acuclip.pose_sequence(dumps.load_by_name(CLIP))


@pytest.fixture(scope='module')
def frame(seq):
    return len(seq.frames) // 2


def _world(bones, lr, lp, rr=IDENT, rp=(0.0, 0.0, 0.0)):
    return fk.fk(bones, lr, lp, root_rot=rr, root_pos=rp)


def _is_ancestor(bones, a, d):
    j = int(d)
    while j != -1:
        if j == a:
            return True
        j = int(bones[j]['parent'])
    return False


def _descendants(bones, i):
    return {j for j in range(len(bones)) if _is_ancestor(bones, i, j)}


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _perp(v, u):
    return v - np.dot(v, u) * u


def test_chains_are_parent_chains_with_expected_names(seq):
    bones = seq.bones
    names = dumps.bone_names()
    for eff, (up_n, lo_n, en_n) in ik.CHAINS.items():
        up, lo, en = ik.chain_indices(bones, eff)
        assert names[int(bones[up]['hash'])] == up_n
        assert names[int(bones[lo]['hash'])] == lo_n
        assert names[int(bones[en]['hash'])] == en_n
        assert _is_ancestor(bones, up, lo) and up != lo, eff
        assert _is_ancestor(bones, lo, en) and lo != en, eff
        print('\n%s: %s' % (eff, ' <- '.join(
            names.get(int(bones[i]['hash']), '?') for i in _path(bones, en))))


def _path(bones, i):
    out = []
    j = int(i)
    while j != -1:
        out.append(j)
        j = int(bones[j]['parent'])
    return out


def test_bone_index_unknown_name_raises(seq):
    with pytest.raises(KeyError):
        ik.bone_index(seq.bones, 'NoSuchBone')


@pytest.mark.parametrize('effector', EFFECTORS)
def test_reachable_target_lands_and_changes_only_chain(seq, frame, effector):
    bones = seq.bones
    lr = seq.local_rot[frame]
    lp = seq.local_pos[frame]
    up, lo, en = ik.chain_indices(bones, effector)
    WR, WP = _world(bones, lr, lp)
    tgt = WP[en] + np.array([0.0, 0.0, 0.10])          # 10 cm up

    new = ik.two_bone_ik(bones, lr, lp, (up, lo, en), tgt)
    WR2, WP2 = _world(bones, new, lp)

    assert np.all(np.isfinite(new))
    assert np.linalg.norm(WP2[en] - tgt) < 1e-3, 'end misses target by %.2e' % np.linalg.norm(WP2[en] - tgt)
    changed = set(np.where(np.any(new != lr, axis=1))[0].tolist())
    assert changed == {up, lo, en}, changed
    outside = [j for j in range(len(bones)) if j not in _descendants(bones, up)]
    assert np.allclose(WP2[outside], WP[outside], atol=1e-12)
    cos = min(1.0, abs(float(np.dot(WR2[en], WR[en]))))
    assert np.degrees(2.0 * np.arccos(cos)) < 0.01
    d_old = np.linalg.norm(WP[lo] - WP[up])
    d_new = np.linalg.norm(WP2[lo] - WP2[up])
    assert abs(d_old - d_new) < 1e-9, 'upper segment length changed'


def test_reachable_target_without_keeping_end_rotation(seq, frame):
    bones = seq.bones
    lr = seq.local_rot[frame]
    lp = seq.local_pos[frame]
    up, lo, en = ik.chain_indices(bones, 'LeftHand')
    WR, WP = _world(bones, lr, lp)
    tgt = WP[en] + np.array([0.0, 0.0, 0.10])
    new = ik.two_bone_ik(bones, lr, lp, (up, lo, en), tgt, keep_end_world_rot=False)
    WR2, WP2 = _world(bones, new, lp)
    assert np.linalg.norm(WP2[en] - tgt) < 1e-3
    changed = set(np.where(np.any(new != lr, axis=1))[0].tolist())
    assert changed == {up, lo}, changed
    assert np.array_equal(new[en], lr[en])


@pytest.mark.parametrize('effector', EFFECTORS)
def test_unreachable_target_extends_chain_towards_target(seq, frame, effector):
    bones = seq.bones
    lr = seq.local_rot[frame]
    lp = seq.local_pos[frame]
    up, lo, en = ik.chain_indices(bones, effector)
    WR, WP = _world(bones, lr, lp)
    tgt = WP[en] + np.array([0.0, 0.0, 5.0])                 # 5 m away, far beyond the limb

    new = ik.two_bone_ik(bones, lr, lp, (up, lo, en), tgt)
    assert np.all(np.isfinite(new))
    WR2, WP2 = _world(bones, new, lp)
    A = WP2[up]
    u = _unit(tgt - A)
    assert np.linalg.norm(np.cross(WP2[en] - A, u)) < 1e-3, 'end is off the shoulder-target line'
    assert np.dot(WP2[en] - A, u) > 0
    total = np.linalg.norm(WP[lo] - WP[up]) + np.linalg.norm(WP[en] - WP[lo])
    assert abs(np.linalg.norm(WP2[en] - A) - 0.999 * total) < 1e-6
    assert np.all(np.isfinite(WR2))


@pytest.mark.parametrize('side', [1.0, -1.0])
def test_pole_puts_middle_joint_on_pole_side(seq, frame, side):
    bones = seq.bones
    lr = seq.local_rot[frame]
    lp = seq.local_pos[frame]
    up, lo, en = ik.chain_indices(bones, 'LeftHand')
    WR, WP = _world(bones, lr, lp)
    tgt = WP[en] + np.array([0.0, 0.0, 0.10])
    A = WP[up]
    u = _unit(tgt - A)
    n = _unit(_perp(np.array([0.0, 1.0, 0.0]), u))
    pole = A + 0.5 * u + side * 0.5 * n

    new = ik.two_bone_ik(bones, lr, lp, (up, lo, en), tgt, pole=pole)
    WR2, WP2 = _world(bones, new, lp)
    assert np.linalg.norm(WP2[en] - tgt) < 1e-3
    elbow_side = np.dot(_perp(WP2[lo] - A, u), n)
    assert np.sign(elbow_side) == side, 'elbow on the wrong side of the shoulder-target line'


def test_without_pole_bend_plane_is_kept(seq, frame):
    bones = seq.bones
    lr = seq.local_rot[frame]
    lp = seq.local_pos[frame]
    up, lo, en = ik.chain_indices(bones, 'LeftHand')
    WR, WP = _world(bones, lr, lp)
    tgt = WP[en] + np.array([0.0, 0.0, 0.10])
    new = ik.two_bone_ik(bones, lr, lp, (up, lo, en), tgt)
    WR2, WP2 = _world(bones, new, lp)
    u = _unit(tgt - WP[up])
    before = _perp(WP[lo] - WP[up], u)
    after = _perp(WP2[lo] - WP2[up], u)
    assert np.dot(before, after) > 0


def test_solve_sequence_weight_zero_unchanged_weight_one_reaches(seq):
    F = len(seq.frames)
    rr = np.tile(IDENT, (F, 1))
    rp = np.zeros((F, 3))
    f_zero, f_one = 40, 50
    WR, WP = _world(seq.bones, seq.local_rot[f_zero], seq.local_pos[f_zero])
    _, WP_one = _world(seq.bones, seq.local_rot[f_one], seq.local_pos[f_one])
    up, lo, en = ik.chain_indices(seq.bones, 'LeftHand')
    tgt0 = WP[en] + np.array([0.0, 0.0, 0.10])
    tgt1 = WP_one[en] + np.array([0.0, 0.0, 0.10])
    out = ik.solve_sequence(seq, rr, rp, {'LeftHand': [(f_zero, tgt0, 0.0), (f_one, tgt1, 1.0)]})

    assert out.local_rot.shape == seq.local_rot.shape
    assert np.array_equal(out.frames, seq.frames)
    assert np.array_equal(out.local_rot[f_zero], seq.local_rot[f_zero])
    others = [k for k in range(F) if k not in (f_zero, f_one)]
    assert np.array_equal(out.local_rot[others], seq.local_rot[others])
    _, WP_out = _world(seq.bones, out.local_rot[f_one], out.local_pos[f_one])
    assert np.linalg.norm(WP_out[en] - tgt1) < 1e-3


def test_solve_sequence_half_weight_is_slerp_of_full_ik(seq):
    F = len(seq.frames)
    rr = np.tile(IDENT, (F, 1))
    rp = np.zeros((F, 3))
    f = 45
    up, lo, en = ik.chain_indices(seq.bones, 'RightFoot')
    _, WP = _world(seq.bones, seq.local_rot[f], seq.local_pos[f])
    tgt = WP[en] + np.array([0.0, 0.0, 0.10])
    full = ik.solve_sequence(seq, rr, rp, {'RightFoot': [(f, tgt, 1.0)]})
    half = ik.solve_sequence(seq, rr, rp, {'RightFoot': [(f, tgt, 0.5)]})
    ch = [up, lo, en]
    expected = fk.slerp(seq.local_rot[f][ch], full.local_rot[f][ch], 0.5)
    assert np.allclose(half.local_rot[f][ch], expected, atol=1e-12)


def test_solve_sequence_with_root_transform(seq):
    F = len(seq.frames)
    ang = 0.4
    rr = np.tile([0.0, 0.0, np.sin(ang / 2), np.cos(ang / 2)], (F, 1))
    rp = np.tile([1.0, 2.0, 0.5], (F, 1))
    f = 30
    up, lo, en = ik.chain_indices(seq.bones, 'LeftHand')
    _, WPw = _world(seq.bones, seq.local_rot[f], seq.local_pos[f], rr[f], rp[f])
    tgt = WPw[en] + np.array([0.0, 0.0, 0.10])
    out = ik.solve_sequence(seq, rr, rp, {'LeftHand': [(f, tgt, 1.0)]})
    _, WPo = _world(seq.bones, out.local_rot[f], out.local_pos[f], rr[f], rp[f])
    assert np.linalg.norm(WPo[en] - tgt) < 1e-3


def test_solve_sequence_rejects_bad_input(seq):
    F = len(seq.frames)
    rr = np.tile(IDENT, (F, 1))
    rp = np.zeros((F, 3))
    with pytest.raises(KeyError):
        ik.solve_sequence(seq, rr, rp, {'LeftElbow': [(0, (0, 0, 0), 1.0)]})
    with pytest.raises(ValueError):
        ik.solve_sequence(seq, rr, rp, {'LeftHand': [(0, (0, 0, 0), 1.5)]})
    with pytest.raises(IndexError):
        ik.solve_sequence(seq, rr, rp, {'LeftHand': [(F, (0, 0, 0), 1.0)]})
