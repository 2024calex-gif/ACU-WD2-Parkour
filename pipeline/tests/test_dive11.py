"""dive11 (T23): the delay15 floor fix. Checks: the Exit's event tracks are identical to out\\dive10\\delay15, the floor rule
(>= 0 cm on every non-finger bone, outside the plant's hands), the JSON and mapping format, the foot IK's toe angle, the
contact residual target, and that dive9 and dive10 keep their defaults. Output checks read out\\dive11 and out\\dive10 and
skip when they are not written yet; the bake itself is not run here (it takes minutes)."""
import json
import os

import numpy as np
import pytest

import config
import dive8 as D8
import dive9 as D9
import dive10 as D10
import dive11 as D11
import retarget as R

OUT = os.path.join(config.OUT, 'dive11')
OUT10 = os.path.join(config.OUT, 'dive10', 'delay15')
EXIT_HEX = D9.EXIT_HEX
SO_HEX = D9.SO_HEX
SUMMARY = os.path.join(OUT, 'summary.json')


def _load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _need(path):
    if not os.path.exists(path):
        pytest.skip('%s is not written yet' % path)
    return _load(path)


# ------------------------------------------------------------------------------------------------ events and format
def test_events_identical_to_delay15():
    a = _need(os.path.join(OUT, EXIT_HEX + '.baked.json'))
    b = _need(os.path.join(OUT10, EXIT_HEX + '.baked.json'))
    ea = [t for t in a['Tracks'] if t.get('infotype') == 5]
    eb = [t for t in b['Tracks'] if t.get('infotype') == 5]
    assert len(ea) == len(eb) == 23
    assert json.dumps(ea) == json.dumps(eb)            # byte-identical as written


def test_exit_track_list_matches_delay15_shape():
    a = _need(os.path.join(OUT, EXIT_HEX + '.baked.json'))
    b = _need(os.path.join(OUT10, EXIT_HEX + '.baked.json'))
    assert a['Length'] == b['Length']
    assert len(a['Tracks']) == len(b['Tracks'])
    assert [sorted(t.keys()) for t in a['Tracks']] == [sorted(t.keys()) for t in b['Tracks']]


def test_patch_json_format_matches_delay15():
    for hx in (SO_HEX, EXIT_HEX):
        a = _need(os.path.join(OUT, hx + '.baked.json'))
        b = _need(os.path.join(OUT10, hx + '.baked.json'))
        assert sorted(a.keys()) == sorted(b.keys()) == ['Length', 'Tracks']
        assert set(t.get('infotype') for t in a['Tracks']) == set(t.get('infotype') for t in b['Tracks'])
        for t in a['Tracks']:
            for k in t.get('Keyframes', []):
                assert 'Frame' in k


def test_mapping_schema_and_override():
    m = _need(os.path.join(OUT, 'mapping_dive11.json'))
    m10 = _need(os.path.join(OUT10, 'mapping_dive10.json'))
    assert list(m.keys()) == list(m10.keys()) == ['clips']
    assert [c['name'] for c in m['clips']] == [c['name'] for c in m10['clips']]
    assert [sorted(c.keys()) for c in m['clips']] == [sorted(c.keys()) for c in m10['clips']]
    for c in m['clips']:
        assert c['override'] == 'preview: dive11 delay15 floor fix'
        assert c['enabled'] is True
    assert m['clips'][1]['only_after'] == [m['clips'][0]['name']]


# ------------------------------------------------------------------------------------------------ the floor rule
def test_floor_rule_non_finger_bones_at_or_above_ground():
    s = _need(SUMMARY)
    fl = s['after']['metrics']['floor']
    assert set(fl.keys()) == set(D11.PHASES)
    for name, v in fl.items():
        if v['min_cm'] is None:
            continue
        assert v['min_cm'] >= 0.0, '%s: %.3f cm (%s)' % (name, v['min_cm'], v['bone'])
    assert s['after']['metrics']['floor_pass'] is True


def test_exit_aim_two_cm_in_window_and_tail():
    s = _need(SUMMARY)
    fl = s['after']['metrics']['floor']
    assert fl['Exit window']['min_cm'] >= 1.9
    assert fl['Exit tail']['min_cm'] >= 1.9


def test_exit_profile_and_speed_rules():
    s = _need(SUMMARY)
    m = s['after']['metrics']
    assert m['z_at_p_up'] >= 0.78
    assert m['rise_speed_joined_post_W_mps'] <= 4.0
    assert m['gap_at_window_end_max_patched'] <= 25.0
    assert m['inv_exit'] is False and m['inv_W'] is False


# ------------------------------------------------------------------------------------------------ the foot IK
def test_toe_angle_lifts_the_toe_to_target():
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(400):
        d = rng.normal(size=3)
        d = d / np.linalg.norm(d) * 0.15
        Az = rng.uniform(0.0, 0.14)
        if Az + d[2] >= D11.TOE_T:
            continue
        th, reach = D11.toe_angle(Az, d)
        if not reach or th >= D11.IK_THETA_MAX - 1e-9:      # the cap (50 deg) leaves the rest to the body lift
            continue
        up = np.array([0.0, 0.0, 1.0])
        a = np.cross(d, up)
        a = a / np.linalg.norm(a)
        q = np.concatenate([a * np.sin(th / 2.0), [np.cos(th / 2.0)]])
        d2 = R.qrot(q, d)
        assert Az + d2[2] >= D11.TOE_T - 1e-9
        checked += 1
    assert checked > 50


def test_vectorised_quaternion_helpers_match_retarget():
    rng = np.random.default_rng(5)
    q = rng.normal(size=(6, 4))
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    v = rng.normal(size=(6, 3))
    got = D11._qrot_v(q, v)
    want = np.array([R.qrot(q[i], v[i]) for i in range(6)])
    assert np.allclose(got, want, atol=1e-9)
    p = D11._qmul_v(q, q)
    want_p = np.array([R.qmul(q[i], q[i]) for i in range(6)])
    assert np.allclose(p, want_p, atol=1e-9)


# ------------------------------------------------------------------------------------------------ contact residual
def test_residual_targets_two_cm_above_the_surface_one_sided():
    # a non-contact bone on the ground is raised by the full target
    ground = [dict(tau=np.array([0.5]), lowest=np.array([0.0]), surf=np.array([0.0]), box_need=np.array([0.0]),
                   contact=np.array([False]), in_plant=np.array([False]))]
    r1, _ = D11.residual11(ground, 100)
    assert abs(r1[30] - D11.TARGET) < 1e-12
    # a non-contact bone 5 cm up is never lowered (one-sided)
    high = [dict(tau=np.array([0.5]), lowest=np.array([0.05]), surf=np.array([0.0]), box_need=np.array([0.0]),
                 contact=np.array([False]), in_plant=np.array([False]))]
    r2, _ = D11.residual11(high, 100)
    assert r2[30] == 0.0
    # the box clearance raises a bone that is inside the box band
    boxed = [dict(tau=np.array([0.5]), lowest=np.array([0.5]), surf=np.array([1.0]), box_need=np.array([0.03]),
                  contact=np.array([False]), in_plant=np.array([False]))]
    r3, _ = D11.residual11(boxed, 100)
    assert abs(r3[30] - 0.03) < 1e-12 or r3[30] >= 0.03 - 1e-12
    assert D11.TARGET == 0.02 and D11.TOE_T == D11.TARGET and D11.FOOT_NEAR == 0.15


def test_plant_target_is_floor_only_and_cap_is_bounded():
    assert D11.PLANT_TARGET == 0.005
    assert 0.0 < D11.S_STEP_CAP <= 0.1
    assert D11.S_PASSES >= 1


# ------------------------------------------------------------------------------------------------ defaults of dive9 and dive10
def test_dive9_and_dive10_defaults_unchanged():
    assert D9.P_UP == 0.16 and D9.Z_UP == 0.80 and D9.Z_UP_MIN == 0.78
    assert D9.RISE_V_MAX == 4.0 and D9.PELVIS_FLOOR_V == 5.9
    assert D9.OVERRIDE == 'preview: dive9 fast roll, standing by Nav'
    assert D10.DELAYS == (8, 15)
    assert D10.override_for(15) == 'preview: dive10 exit events +15'
    assert D10.EVENT_FROM == 13 and D10.K_LIFT_E == 140
    assert tuple(D8.PLANT_W) == (0.75, 0.92)


def test_dive11_writes_only_under_out_dive11():
    assert os.path.normpath(D11.OUT_DIR) == os.path.normpath(os.path.join(config.OUT, 'dive11'))
    assert os.path.normpath(D11.D10_EXIT).startswith(os.path.normpath(os.path.join(config.OUT, 'dive10')))
