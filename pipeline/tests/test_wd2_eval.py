"""wd2_eval: event detector (synthetic + the hand-annotated kong and frontdive), trunk and recovery measures, path
self-consistency and the reference verdicts (kong strong, frontdive poor). The real-clip tests take a few seconds."""
import json

import numpy as np
import pytest

import wd2_eval as E

DT = 1.0 / 30.0
ANN = json.load(open(E.wp.EVENTS_JSON, encoding='utf-8'))


def _ann(key):
    return ANN[key]


# ------------------------------------------------------------------------------------------------ synthetic events
def test_detect_events_hand_plant_rule():
    contacts = {
        'LeftFoot': [(0.0, 0.5), (1.9, 2.2)],
        'RightFoot': [(0.1, 0.45), (1.6, 1.9)],
        'LeftFootThumb1': [],
        'RightFootThumb1': [],
        'LeftHand': [(0.8, 1.0)],
        'RightHand': [(0.8, 1.05)],
    }
    ev = E.detect_events(contacts)
    assert ev['rule'] == 'hand plant'
    assert ev['takeoff'] == pytest.approx(0.5)             # last foot run that ends before the hands
    assert ev['hand_plant_start'] == pytest.approx(0.8)
    assert ev['hand_plant_end'] == pytest.approx(1.05)
    assert ev['landing'] == pytest.approx(1.6)            # first foot run after the hands
    assert ev['gap_s'] == pytest.approx(1.1)


def test_detect_events_foot_gap_rule_without_hands():
    contacts = {'LeftFoot': [(0.0, 0.4)], 'RightFoot': [(0.1, 0.45), (1.2, 1.5)],
                'LeftFootThumb1': [(1.15, 1.25)], 'RightFootThumb1': [], 'LeftHand': [], 'RightHand': []}
    ev = E.detect_events(contacts)
    assert ev['rule'] == 'foot gap'
    assert ev['takeoff'] == pytest.approx(0.45)
    assert ev['landing'] == pytest.approx(1.15)
    assert ev['hand_plant_start'] is None


def test_detect_events_short_gap_is_not_a_flight():
    contacts = {'LeftFoot': [(0.0, 0.4), (0.5, 0.9)], 'RightFoot': [], 'LeftFootThumb1': [], 'RightFootThumb1': [],
                'LeftHand': [], 'RightHand': []}
    ev = E.detect_events(contacts)
    assert ev['takeoff'] is None and ev['landing'] is None


def test_name_cm_hint():
    assert E.name_cm('a00-000-00_vaultover-sprint-100cm-long-frontdive_male') == 100
    assert E.name_cm('b00-000-00_vaultover-kong_mplyr') is None


# ------------------------------------------------------------------------------------------------ trunk and recovery
def _traces(head_vec, n=60):
    t = np.arange(n) * DT
    J = {'Hips': np.zeros((n, 3)), 'Head': np.tile(np.asarray(head_vec, float), (n, 1))}
    z = np.zeros(n)
    eff = {k: {'z': z.copy()} for k in ('LeftHand', 'RightHand', 'LeftFoot', 'RightFoot')}
    return dict(t=t, joints=J, eff=eff)


def test_trunk_upright_and_inverted_classes():
    ev = dict(takeoff=0.2, landing=1.0)
    up, _tilt, _ang = E.trunk_and_hands(_traces([0.0, 0.0, 0.5]), ev)
    assert up['tilt_max_flight'] == pytest.approx(0.0, abs=1e-6)
    assert up['inversion'] == 'none'
    inv, _tilt, _ang = E.trunk_and_hands(_traces([0.0, 0.0, -0.5]), ev)
    assert inv['tilt_max_flight'] == pytest.approx(180.0, abs=1e-6)
    assert inv['inversion'] == 'severe'
    assert inv['head_below_hips_s'] > 0.7          # the flight is 0.8 s long


def test_recovery_time_on_a_ramp():
    t = np.arange(0.0, 2.0, 1.0 / 60.0)
    z = np.where(t < 1.0, 0.5, 0.5 + (t - 1.0) * 2.0)     # rises from 0.5 at t = 1 s, reaches 1.0 at 1.25 s
    z = np.minimum(z, 1.0)
    rec = E.recovery_time(z, t, landing=1.0, standing=1.0)  # need 0.85: reached at 1.175 s, sustained 0.25 s
    assert rec == pytest.approx(0.175, abs=0.03)
    assert E.recovery_time(np.full_like(t, 0.5), t, 1.0, 1.0) is None


# ------------------------------------------------------------------------------------------------ real clips
KONG = E.KONG
FRONT = E.FRONTDIVE


@pytest.fixture(scope='module')
def rig():
    return E.retarget.Rig()


@pytest.fixture(scope='module')
def kong(rig):
    return E.analyse_wd2(KONG, rig)


@pytest.fixture(scope='module')
def frontdive(rig):
    return E.analyse_roll_reference(FRONT, rig)


def test_kong_takeoff_and_landing_match_annotation(kong):
    ev = kong[0]['events']
    a = _ann('catpass_kong')
    assert ev['takeoff'] == pytest.approx(a['takeoff'], abs=0.05)
    assert ev['landing'] == pytest.approx(a['landing'], abs=0.05)


def test_kong_hand_plant_start_matches_annotation(kong):
    ev = kong[0]['events']
    assert ev['hand_plant_start'] == pytest.approx(_ann('catpass_kong')['hand_plant_start'], abs=0.05)


@pytest.mark.xfail(strict=True, reason='kong hand plant end: detector 1.37 s, annotation 1.52 s (the band loses the '
                                       'hands as the body decelerates); known deviation, see METHOD_NOTES')
def test_kong_hand_plant_end_matches_annotation(kong):
    ev = kong[0]['events']
    assert ev['hand_plant_end'] == pytest.approx(_ann('catpass_kong')['hand_plant_end'], abs=0.05)


def test_frontdive_takeoff_and_landing_match_annotation(frontdive):
    ev = frontdive[0]['events']
    a = _ann('diveroll_frontdive')
    assert ev['takeoff'] == pytest.approx(a['takeoff'], abs=0.05)
    assert ev['landing'] == pytest.approx(a['landing'], abs=0.05)


@pytest.mark.xfail(strict=True, reason='frontdive hand plant start: detector 0.50 s (an arm sweep at the speed band), '
                                       'annotation 0.75 s; known deviation')
def test_frontdive_hand_plant_start_matches_annotation(frontdive):
    ev = frontdive[0]['events']
    assert ev['hand_plant_start'] == pytest.approx(_ann('diveroll_frontdive')['hand_plant_start'], abs=0.05)


def test_frontdive_is_inverted_and_kong_is_not(kong, frontdive):
    assert frontdive[0]['inversion'] == 'severe'
    assert kong[0]['inversion'] in ('none', 'mild')


def test_path_is_self_consistent_and_travel_is_stable(frontdive):
    res, arr = frontdive
    rr = res['path']['residuals']
    assert rr['contact_xy']['rms'] < 0.05
    assert rr['ground_z']['rms'] < 0.05
    # the annotated roll window changes the travel only a little (the uncertainty of the path travel)
    diff = abs(res['path']['travel_m'] - res['path_with_roll_window']['travel_m'])
    assert diff < 0.15
    assert 1.5 < res['path']['travel_m'] < 2.3


def test_kong_recovers_upright_at_landing(kong):
    res, _ = kong
    assert res['recovery_s'] is not None and res['recovery_s'] <= 0.1


def test_calibration_verdicts(kong, frontdive):
    """The references set the scale: kong strong, frontdive poor, against the same free ACU target."""
    target = 'xx_Parkour_Sprint_SlideOver_Fd100cm-Up100cm'
    acu = E.analyse_acu(target, 'free')
    k = E.score_pair(kong[0], acu)
    f = E.score_pair(frontdive[0], acu)
    assert k['verdict'] == 'strong fit' and k['score'] >= 70
    assert f['verdict'] == 'poor fit' and f['score'] < 55
    assert f['parts']['upright'] == 0.0 and f['parts']['recovery'] < 1.0
