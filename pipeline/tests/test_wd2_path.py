"""Tests for wd2_path.py (the reconstructed WD2 frontdive pelvis path) and the dive3 output format.

Synthetic cases build a known pelvis path (stance at constant velocity, a ballistic flight with g = 9.81, planted hands)
and the pelvis-relative joint offsets that make every contact true, so the least-squares solution must recover it.
"""
import json
import os

import numpy as np
import pytest

import bake2
import config
import wd2_path as WP

G = WP.G
OUT = os.path.join(config.OUT, 'dive3')
MAPPING = os.path.join(OUT, 'patches', 'mapping_dive3.json')


class _Rig:
    """Only the joint names are needed by solve_path when the relative offsets are given."""

    def __init__(self, names):
        self.wd = [dict(name=n) for n in names]


NAMES = ['Pelvis', 'L Foot', 'R Foot', 'L Hand', 'R Hand', 'Spine', 'Head', 'L Index01']
EV = dict(takeoff=0.45, hand_plant_start=0.75, hand_plant_end=0.92, roll=[0.92, 1.18], landing=1.2)


def _truth(t):
    """Known pelvis path: stance at (vx, vy) at z = 0.8 before 0.45 s, a parabola with g on 0.45 .. 0.75 with the same
    horizontal velocity and zero vertical velocity at takeoff, then a continued parabola on the plant."""
    vx, vy = 0.2, 2.0
    x0, y0 = 0.0, 0.0
    P = np.zeros((len(t), 3))
    P_to = np.array([x0 + vx * 0.25, y0 + vy * 0.25, 0.8])
    for k, tt in enumerate(t):
        if tt < 0.45:
            P[k] = [x0 + vx * (tt - 0.2), y0 + vy * (tt - 0.2), 0.8]
        else:
            tau = tt - 0.45
            P[k] = P_to + np.array([vx, vy, 0.0]) * tau + 0.5 * np.array([0.0, 0.0, -G]) * tau * tau
    return P


def _synthetic(t):
    """(rel [F, B, 3], truth P) with every contact true for the known path (joint offsets relative to the pelvis)."""
    P = _truth(t)
    F = len(t)
    B = len(NAMES)
    rel = np.zeros((F, B, 3))
    idx = {n: i for i, n in enumerate(NAMES)}
    anchors = {'L Foot': np.array([0.1, 0.5, WP.FOOT_Z]), 'L Hand': np.array([-0.4, 4.0, WP.HAND_Z]),
               'R Hand': np.array([0.3, 4.1, WP.HAND_Z])}
    for k, tt in enumerate(t):
        rel[k, idx['Spine']] = [0.0, -0.1, 0.5]
        rel[k, idx['Head']] = [0.0, 0.2, 0.9]
        rel[k, idx['L Index01']] = [0.0, 0.1, 0.1]
        rel[k, idx['R Foot']] = [0.3, -0.6, 0.2]
        if tt < 0.45:
            rel[k, idx['L Foot']] = np.array([anchors['L Foot'][0], anchors['L Foot'][1], WP.FOOT_Z]) - P[k]
        if 0.75 - 1e-9 <= tt <= 0.92 + 1e-9:
            for h in ('L Hand', 'R Hand'):
                rel[k, idx[h]] = anchors[h] - P[k]
        else:
            rel[k, idx['L Hand']] = [-0.5, 0.3, 0.2]
            rel[k, idx['R Hand']] = [0.4, 0.3, 0.1]
    return rel, P


@pytest.fixture(scope='module')
def synth():
    t = WP.time_grid(0.2, 0.9)
    rel, P = _synthetic(t)
    sol = WP.solve_path(_Rig(NAMES), None, EV, t=t, rel=rel)
    return dict(t=t, rel=rel, P=P, sol=sol)


# ------------------------------------------------------------------------------------------------ ballistic
def _from_takeoff(t, P):
    """The path relative to its takeoff sample (x and y), the form the solver is pinned to."""
    k = int(np.argmin(np.abs(t - 0.45)))
    Q = P.copy()
    Q[:, :2] -= P[k, :2]
    return Q


def test_synthetic_ballistic_flight_is_recovered(synth):
    """The flight is a parabola with g = 9.81 between takeoff and the plant start: the recovered path matches the known one."""
    t, P_true, sol = synth['t'], synth['P'], synth['sol']
    fl = (t >= 0.45 - 1e-9) & (t <= 0.75 + 1e-9)
    err = np.abs(sol['P'][fl] - _from_takeoff(t, P_true)[fl]).max()
    assert err < 3e-3, 'flight error %.4f m' % err
    # the second difference inside the flight equals (0, 0, -g) dt^2
    d2 = sol['P'][2:] - 2 * sol['P'][1:-1] + sol['P'][:-2]
    mid = (t[1:-1] >= 0.45 + 1e-9) & (t[1:-1] <= 0.75 - 1e-9)
    assert np.allclose(d2[mid], np.array([0.0, 0.0, -G]) * WP.DT ** 2, atol=2e-5)


def test_synthetic_takeoff_velocity_and_apex_are_recovered(synth):
    t, P_true, sol = synth['t'], synth['P'], synth['sol']
    Q = _from_takeoff(t, P_true)
    v_true = (Q[2:] - Q[:-2]) / (2 * WP.DT)
    v_fit = (sol['P'][2:] - sol['P'][:-2]) / (2 * WP.DT)
    k = int(round((0.60 - t[0]) * 60))           # mid-flight sample
    assert np.allclose(v_fit[k - 1], v_true[k - 1], atol=0.05)
    apex = P_true[(t >= 0.45 - 1e-9) & (t <= 0.75 + 1e-9), 2].max()
    assert abs(sol['P'][(t >= 0.45 - 1e-9) & (t <= 0.75 + 1e-9), 2].max() - apex) < 3e-3


# ------------------------------------------------------------------------------------------------ contact pinning
def test_synthetic_stance_foot_pinned_and_hands_planted(synth):
    t, rel, sol = synth['t'], synth['rel'], synth['sol']
    idx = {n: i for i, n in enumerate(NAMES)}
    pre = t < 0.45 - 1e-9
    foot = sol['P'] + rel[:, idx['L Foot']]
    assert np.allclose(foot[pre, 2], WP.FOOT_Z, atol=1e-3)
    assert np.abs(np.diff(foot[pre, :2], axis=0)).max() < 1e-3
    plant = (t >= 0.75 - 1e-9) & (t <= 0.92 + 1e-9)
    for h in ('L Hand', 'R Hand'):
        hand = sol['P'] + rel[:, idx[h]]
        assert np.allclose(hand[plant, 2], WP.HAND_Z, atol=2e-3)
        assert np.abs(np.diff(hand[plant, :2], axis=0)).max() < 2e-3


def test_real_stance_foot_and_hands_pinned():
    """The real WD2 frontdive: the stance foot sits at the foot height before takeoff and the hands on the ground in the
    plant (the rows are heavy; residual RMS is reported in the solver residuals, a few mm at most)."""
    import wd2
    import retarget as R
    ev = WP.load_events()
    rig = R.Rig()
    m = wd2.load(ev['stem'])
    sol = WP.solve_path(rig, m, ev)
    idx = sol['idx']
    t = sol['t']
    pre = t < 0.45 - 1e-9
    foot = sol['P'] + sol['rel'][:, idx[sol['stance']]]
    assert np.allclose(foot[pre, 2], WP.FOOT_Z, atol=3e-3)
    assert sol['residuals']['plant_z']['rms'] < 0.1           # hands: 6 cm rms is the conflict of the two hand heights
    assert sol['residuals']['stance_xy']['rms'] < 1e-6
    assert sol['residuals']['flight']['rms'] < 0.01


# ------------------------------------------------------------------------------------------------ placement and bump
def test_placement_maps_takeoff_and_scales_forward_travel():
    sol = dict(t=np.array([0.45, 0.6, 0.75]), P=np.array([[0.0, 1.0, 0.8], [0.0, 1.5, 1.0], [0.0, 2.0, 0.8]]),
               anchors={'hand_L_xy': np.array([0.1, 2.5]), 'hand_R_xy': np.array([-0.1, 2.4])},
               residuals={}, body=[])
    pl = WP.Placement(sol, s=0.5, y_take=3.0)
    p = pl.pelvis(np.array([0.45, 0.75]))
    assert p[0, 1] == pytest.approx(3.0)
    assert p[1, 1] == pytest.approx(3.0 + 0.5 * 1.0)          # 0.5 of the WD2 forward travel of 1.0 m
    hL = pl.hand_anchor('L')
    assert hL[1] == pytest.approx(3.0 + 0.5 * (2.5 - 1.0))
    assert hL[2] == pytest.approx(WP.ACU_HAND_Z)


def test_bump_is_a_raised_cosine_zero_at_the_ends_with_its_amplitude_at_the_middle():
    sol = dict(t=np.array([0.45]), P=np.zeros((1, 3)), anchors={'hand_L_xy': np.zeros(2), 'hand_R_xy': np.zeros(2)},
               residuals={}, body=[])
    pl = WP.Placement(sol, s=1.0, y_take=0.0, bump_amp=0.8)
    w = np.array([WP.BUMP_T0, 0.6, WP.BUMP_T1, 0.9])
    b = pl.bump(w)
    assert b[0] == pytest.approx(0.0) and b[2] == pytest.approx(0.0) and b[3] == pytest.approx(0.0)
    assert b[1] == pytest.approx(0.8)
    shape = WP.raised_cosine(np.linspace(WP.BUMP_T0, WP.BUMP_T1, 101))
    assert shape.min() >= -1e-12 and shape.max() == pytest.approx(1.0)


def test_minimum_amplitude_covers_the_lift_and_refuses_when_the_shape_is_zero():
    w = np.linspace(WP.BUMP_T0, WP.BUMP_T1, 31)
    lift = 0.5 * WP.raised_cosine(w)
    assert WP.min_amplitude_for_lift(w, lift) == pytest.approx(0.5)
    assert WP.min_amplitude_for_lift(w, np.where(w < 0.46, 0.2, 0.0)) is None


# ------------------------------------------------------------------------------------------------ output format
def test_dive3_outputs_have_the_plugin_format_and_no_root_track():
    if not os.path.exists(MAPPING):
        pytest.skip('dive3 outputs not written yet')
    for hexname in ('47BFB02DF', 'F7919BDF0'):
        with open(os.path.join(OUT, 'patches', hexname + '.baked.json'), encoding='utf-8') as f:
            d = json.load(f)
        assert set(d) >= {'Length', 'Tracks'}
        ids = [tr['TrackID'] for tr in d['Tracks']]
        assert '0' not in ids, 'dive3 never patches TrackID 0'
        assert str(bake2.REF_TID) in ids, 'Reference translation track present'
        for tr in d['Tracks']:
            if tr['codec'] == 27:
                assert tr['infotype'] == 3 and tr['TrackID'] == str(bake2.REF_TID)
                assert all(len(k['LocalPosition']) == 3 for k in tr['Keyframes'])
            else:
                assert tr['codec'] == 1 and tr['infotype'] == 0
            frames = [k['Frame'] for k in tr['Keyframes']]
            assert frames == sorted(frames)
    with open(MAPPING, encoding='utf-8') as f:
        mp = json.load(f)
    so, ex = mp['clips']
    assert so['prebaked'] is True and so['patch_translation'] is True
    assert ex['prebaked'] is True and ex['patch_translation'] is True
    assert ex['only_after'] == ['xx_Parkour_Sprint_SlideOver_Fd100cm-Up100cm']
    assert ex['arm_seconds'] == 2.0
    assert 'only_after' not in so
