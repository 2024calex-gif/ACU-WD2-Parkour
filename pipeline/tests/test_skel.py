"""WD2 and ACU bind-pose skeleton loaders (skel.py)."""
import math

import numpy as np

import retarget
import skel

WD2_EXPECTED_BONES = 75
ACU_EXPECTED_BONES = 109


def test_wd2_skeleton_loads():
    bones = skel.load_wd2()
    print('\nWD2 bones: %d' % len(bones))
    assert len(bones) == WD2_EXPECTED_BONES
    assert bones[0]['name'] == 'Pelvis' and bones[0]['parent'] == -1
    for b in bones:
        assert b['parent'] == -1 or 0 <= b['parent'] < len(bones)


def test_acu_skeleton_loads():
    bones = skel.load_acu()
    print('\nACU bones: %d' % len(bones))
    assert len(bones) == ACU_EXPECTED_BONES
    assert bones[0]['parent'] == -1
    for b in bones:
        assert b['parent'] == -1 or 0 <= b['parent'] < len(bones)


def test_acu_forward_kinematics_matches_stored_globals():
    """Local pose -> world with FK reproduces the stored global pose for every bone except the root.

    The root record (index 0, 'Reference') has a different layout and its stored global is not a
    usable pose, so FK starts from identity at the origin for it. That makes all other bones match.
    """
    bones = skel.load_acu()
    n = len(bones)
    W = [None] * n
    P = [None] * n
    W[0] = np.array([0.0, 0.0, 0.0, 1.0])
    P[0] = np.zeros(3)
    worst_rot = 0.0
    worst_pos = 0.0
    for i in range(1, n):
        par = bones[i]['parent']
        assert par >= 0, 'bone %d has no parent' % i
        lrot = np.array(bones[i]['lrot'], float)
        lpos = np.array(bones[i]['lpos'], float)
        W[i] = retarget.qn(retarget.qmul(W[par], lrot))
        P[i] = P[par] + retarget.qrot(W[par], lpos)

        g = np.array(bones[i]['grot'], float)
        g = g / np.linalg.norm(g)
        rot_err = min(np.linalg.norm(W[i] - g), np.linalg.norm(W[i] + g))   # quaternion sign ambiguity
        pos_err = np.linalg.norm(P[i] - np.array(bones[i]['gpos'], float))
        worst_rot = max(worst_rot, rot_err)
        worst_pos = max(worst_pos, pos_err)
        assert rot_err < 1e-3, 'bone %d rotation error %.2e' % (i, rot_err)
        assert pos_err < 1e-3, 'bone %d position error %.2e' % (i, pos_err)
    print('\nACU FK worst error: rot %.2e, pos %.2e' % (worst_rot, worst_pos))


def test_bind_quaternions_are_unit_length():
    for label, bones in (('WD2', skel.load_wd2()), ('ACU', skel.load_acu())):
        for i, b in enumerate(bones):
            norm = float(np.linalg.norm(b['lrot']))
            assert math.isclose(norm, 1.0, abs_tol=1e-3), '%s bone %d lrot norm %.6f' % (label, i, norm)
