"""wd2events: contact candidates and event suggestion on synthetic traces, plus one real-clip smoke test."""
import numpy as np
import pytest

import wd2events as we

KONG = 'b00-000-00_vaultover-kong_mplyr_000f_norm_nowep'


def synth_traces(vf_by_name, n=None):
    """Traces dict with given forward velocity arrays (m/s) at 30 Hz; z and pos are zeros."""
    n = n or len(next(iter(vf_by_name.values())))
    t = np.arange(n) / we.FPS
    eff = {}
    for name, vf in vf_by_name.items():
        vf = np.asarray(vf, float)
        assert len(vf) == n
        eff[name] = {'vf': vf, 'z': np.zeros(n), 'pos': np.zeros((n, 3))}
    return {'stem': 'synthetic', 'dur': n / we.FPS, 't': t, 'dt': 1 / we.FPS,
            'forward': np.array([1.0, 0.0, 0.0]), 'eff': eff, 'joints': {}}


def foot_with_steps():
    """Foot: 10 ticks swing forward (+6), 10 ticks planted at -5 (body speed 5), 10 ticks swing (+8),
    10 ticks planted at -5 again, then 10 ticks swing (+5)."""
    seg = [6.0] * 10 + [-5.0] * 10 + [8.0] * 10 + [-5.0] * 10 + [5.0] * 10
    return np.array(seg)


def test_body_speed_from_runup_foot():
    tr = synth_traces({'LeftFoot': foot_with_steps()})
    assert we.estimate_body_speed(tr) == pytest.approx(5.0, abs=1e-9)


def test_body_speed_zero_when_no_backward_foot():
    tr = synth_traces({'LeftFoot': np.full(60, 1.0), 'RightFoot': np.zeros(60)})
    assert we.estimate_body_speed(tr) == 0.0


def test_candidate_contacts_backward_foot_at_body_speed():
    """A foot moving backward at body speed for 10 ticks is one candidate interval, the swings are not."""
    tr = synth_traces({'LeftFoot': foot_with_steps()})
    c = we.candidate_contacts(tr)['LeftFoot']
    assert len(c) == 2
    (a0, a1), (b0, b1) = c
    # ticks 10..19 and 30..39 (inclusive), within one tick of the truth
    dt = 1 / we.FPS
    assert a0 == pytest.approx(10 * dt, abs=dt) and a1 == pytest.approx(19 * dt, abs=dt)
    assert b0 == pytest.approx(30 * dt, abs=dt) and b1 == pytest.approx(39 * dt, abs=dt)


def test_candidate_contacts_band_and_min_ticks():
    tr = synth_traces({'LeftFoot': foot_with_steps()})
    # within 35 percent of 5 m/s: -3.25 .. -6.75
    assert we.candidate_contacts(tr, speed=5.0)['LeftFoot'] != []
    # body speed 10 m/s: planted feet at -5 are far outside the band
    assert we.candidate_contacts(tr, speed=10.0)['LeftFoot'] == []
    # speed 0: nothing is a candidate
    assert we.candidate_contacts(tr, speed=0.0)['LeftFoot'] == []
    # a 2-tick backward run is too short
    short = np.array([6.0] * 10 + [-5.0] * 2 + [6.0] * 10)
    tr2 = synth_traces({'LeftFoot': short})
    assert we.candidate_contacts(tr2, speed=5.0)['LeftFoot'] == []
    # a 3-tick run is kept
    short3 = np.array([6.0] * 10 + [-5.0] * 3 + [6.0] * 10)
    assert len(we.candidate_contacts(synth_traces({'LeftFoot': short3}), speed=5.0)['LeftFoot']) == 1


def test_candidate_contacts_slow_backward_motion_is_not_planted():
    tr = synth_traces({'RightFoot': np.full(40, -1.0)})
    assert we.candidate_contacts(tr, speed=5.0)['RightFoot'] == []


def test_suggest_events_order():
    dt = 1 / we.FPS
    contacts = {
        'LeftFoot': [(0.5, 0.9), (1.6, 2.0)],
        'RightFoot': [],
        'LeftHand': [(1.0, 1.3)],
        'RightHand': [(1.05, 1.2)],
    }
    ev = we.suggest_events(contacts, dt)
    assert ev['hand_plant_start'] == pytest.approx(1.0)
    assert ev['hand_plant_end'] == pytest.approx(1.3)
    assert ev['takeoff'] == pytest.approx(0.9)
    assert ev['landing'] == pytest.approx(1.6)


def test_suggest_events_none_without_hands():
    ev = we.suggest_events({'LeftFoot': [(0.5, 0.9)]})
    assert ev == {'takeoff': None, 'hand_plant_start': None, 'hand_plant_end': None, 'landing': None}


def test_effector_traces_real_clip_shape_and_forward():
    tr = we.effector_traces(KONG)
    assert set(we.EFFECTORS) <= set(tr['eff'])
    n = len(tr['t'])
    assert n == 88
    for name in we.EFFECTORS:
        assert len(tr['eff'][name]['vf']) == n and len(tr['eff'][name]['z']) == n
    # the bind-pose toes and the run-up pelvis heading both point along +X in ACU world
    assert tr['forward'][0] > 0.9
    assert tr['speed'] >= 0.0
