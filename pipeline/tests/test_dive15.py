"""dive15 (T27): WD2 clearance in the contact solve, foot locks, and the Exit slowdown in the locks. Checks: the native
clearance target (ground where WD2 is in contact, else the WD2 lowest height, never below 0), the residual of the three modes
(one-sided ground before the plant, two-sided ground in the plant, two-sided clearance with a 2 cm deadband), the lock ease ramp,
the slowdown factor and its integral, the native foot intervals (from 1.20 s, at least 3 frames), dive9's constants unchanged,
and (when written) the output patches: the SlideOver root before 1.20 s, the events byte-identical to dive12 S18 and dive11, the
Exit foot-target tracks free (sentinel) in every lock frame, the mapping. Output checks skip when out\\dive15 is not written; the
bake itself is not run here."""
import json
import os

import numpy as np
import pytest

import acuclip
import config
import dive4 as D4
import dive6 as S6
import dive9 as D9
import dive10 as D10
import dive13 as D13
import dive15 as D15
import dive5 as F5
import dumps
import aux_tracks as AT

OUT = os.path.join(config.OUT, 'dive15')
OUT12 = os.path.join(config.OUT, 'dive12', 'S18')
OUT11 = os.path.join(config.OUT, 'dive11')
SO_HEX, EXIT_HEX = D9.SO_HEX, D9.EXIT_HEX
J = 72
LAST_SO = 98


@pytest.fixture(scope='module')
def ctx():
    return F5.make_ctx()


@pytest.fixture(scope='module')
def nat(ctx):
    return D15.Native(ctx.sol)


def _load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _need(path):
    if not os.path.exists(path):
        pytest.skip('%s is not written yet' % path)
    return _load(path)


def _root_of(patch):
    tr = [t for t in patch['Tracks'] if str(t['TrackID']) == '0']
    assert len(tr) == 1
    fr = np.array([k['Frame'] for k in tr[0]['Keyframes']], int)
    pos = np.array([k['LocalPosition'] for k in tr[0]['Keyframes']], float)
    return fr, pos


def _ev(patch):
    return sorted([t for t in patch['Tracks'] if t.get('infotype') == 5], key=lambda t: (str(t['TrackID']), t['Slot']))


# ------------------------------------------------------------------------------------------------ the clearance target
def test_native_target_is_ground_in_wd2_contact_and_the_wd2_height_otherwise(nat):
    w = np.linspace(0.9, 1.6, 141)
    tgt, contact = nat.target(w)
    h = nat.h(w)
    assert np.all(tgt >= 0.0)
    assert np.allclose(tgt[contact], D15.CONTACT_TARGET)
    assert np.allclose(tgt[~contact], np.maximum(h[~contact], 0.0))
    assert np.all(h[contact] <= D15.GROUND_CAP + 1e-12)


def test_native_foot_intervals_start_at_feet_down_and_last_at_least_three_frames(nat):
    for side in ('L', 'R'):
        for w0, w1 in nat.intervals(side):
            assert w0 >= D15.LOCK_W0 - 1e-9
            assert w1 - w0 >= (D15.LOCK_MIN - 1) / 60.0 - 1e-9


# ------------------------------------------------------------------------------------------------ the residual (three modes)
def _sample(lowest, target, mode, tau=None):
    n = len(lowest)
    return dict(lowest=np.asarray(lowest, float), target=np.asarray(target, float), mode=np.full(n, mode),
                box_need=np.zeros(n), tau=np.asarray(tau if tau is not None else np.arange(n) / 60.0 + 1.0, float))


def test_residual_mode_two_is_a_deadband_around_the_clearance_target():
    K = 200
    r_in, _ = D15.residual15([_sample([0.10 + 0.01], [0.10], 2)], K)
    assert np.allclose(r_in, 0.0)
    r_low, _ = D15.residual15([_sample([0.10 - 0.05], [0.10], 2)], K)
    assert np.max(r_low) == pytest.approx(0.03, abs=1e-9)           # raise by the excess over the deadband
    r_high, _ = D15.residual15([_sample([0.10 + 0.05], [0.10], 2)], K)
    assert np.min(r_high) == pytest.approx(-0.03, abs=1e-9)         # lower by the excess over the deadband


def test_residual_mode_one_is_two_sided_to_the_ground_and_mode_zero_is_one_sided():
    K = 200
    r1, _ = D15.residual15([_sample([0.05], [0.01], 1)], K)
    assert np.min(r1) == pytest.approx(-0.04, abs=1e-9)             # two-sided: pull down to the ground
    r0, _ = D15.residual15([_sample([0.05], [0.01], 0)], K)
    assert np.max(np.abs(r0)) == pytest.approx(0.0, abs=1e-12)      # one-sided: above the ground nothing happens
    r0b, _ = D15.residual15([_sample([-0.02], [0.01], 0)], K)
    assert np.max(r0b) == pytest.approx(0.03, abs=1e-9)             # below the ground it raises


# ------------------------------------------------------------------------------------------------ the lock ease and the slowdown
def test_lock_ease_ramps_in_and_out_over_three_frames():
    assert D15._ramp(10, 10, 20) == pytest.approx(1.0 / 3.0)
    assert D15._ramp(11, 10, 20) == pytest.approx(2.0 / 3.0)
    assert D15._ramp(12, 10, 20) == pytest.approx(1.0)
    assert D15._ramp(20, 10, 20) == pytest.approx(1.0 / 3.0)
    assert D15._ramp(15, 10, 20) == pytest.approx(1.0)


def test_slow_factor_is_one_outside_and_one_minus_alpha_inside_a_span():
    q = np.linspace(0.0, 1.0, 1001)
    f = D15.slow_factor(q, [(0.3, 0.6)], [0.4])
    inside = (q > 0.35) & (q < 0.55)
    assert np.allclose(f[inside], 0.6, atol=1e-9)
    assert np.allclose(f[(q < 0.2) | (q > 0.7)], 1.0)
    assert np.all(f >= 0.05)


def test_exit_delta_without_slowdown_is_dive13s_and_the_slowdown_reduces_the_displacement():
    p = np.linspace(0.0, 0.6, 121)
    base = D15.exit_delta15(p, 2.0, 4.0, 0.4867, [], [])
    ref = D13.exit_delta(p, 2.0, 4.0, 0.4867)
    assert np.allclose(base, ref, atol=1e-6)
    slow = D15.exit_delta15(p, 2.0, 4.0, 0.4867, [(0.1, 0.3)], [0.5])
    assert slow[-1] < base[-1]
    assert np.all(np.diff(slow) >= -1e-12)                          # never reverses


# ------------------------------------------------------------------------------------------------ the constants of dive9 and dive13
def test_dive9_defaults_are_unchanged_by_dive15():
    assert D9.PELVIS_FLOOR_V == 5.9 and D9.P_UP == 0.16 and D9.Z_UP == 0.80
    assert S6.CONTACT_TARGET == 0.01 and S6.CONTACT_TOL == 0.03 and S6.CONTACT_ITER == 30
    assert D13.D_S == 18 and D13.K_PLANT == 1.0 and D13.K_ROLL == 1.0
    assert D15.CLEAR_DB == 0.02 and D15.GROUND_CAP == 0.03 and D15.MAX_SLOW_ITERS == 0


# ------------------------------------------------------------------------------------------------ the output patches
def test_slideover_root_before_1p20_is_the_original():
    patch = _need(os.path.join(OUT, SO_HEX + '.baked.json'))
    so = dumps.load_by_name(D9.SO_NAME)
    orig = acuclip.markers(so, np.arange(LAST_SO + 1))['disp']
    fr, pos = _root_of(patch)
    new = np.stack([np.interp(np.arange(LAST_SO + 1), fr, pos[:, c]) for c in range(3)], axis=1)
    assert np.max(np.abs(new[:J + 1] - orig[:J + 1])) < 1e-8


def test_events_are_byte_identical_to_dive12_s18_and_dive11():
    so = _need(os.path.join(OUT, SO_HEX + '.baked.json'))
    so12 = _need(os.path.join(OUT12, SO_HEX + '.baked.json'))
    assert json.dumps(_ev(so), sort_keys=True) == json.dumps(_ev(so12), sort_keys=True)
    ex = _need(os.path.join(OUT, EXIT_HEX + '.baked.json'))
    ex11 = _need(os.path.join(OUT11, EXIT_HEX + '.baked.json'))
    assert len(_ev(ex)) == 23
    assert json.dumps(_ev(ex), sort_keys=True) == json.dumps(_ev(ex11), sort_keys=True)


def test_exit_foot_targets_are_free_in_every_lock_frame():
    ex = _need(os.path.join(OUT, EXIT_HEX + '.baked.json'))
    summ = _need(os.path.join(OUT, 'summary.json'))
    sched = summ['lock_sched']['E']
    n = int(round(float(ex['Length']) * 60)) + 1
    dump = dumps.load_by_name(D9.EXIT_NAME)
    for foot, tid in (('LeftFoot', AT.LEFT_FOOT_ID), ('RightFoot', AT.RIGHT_FOOT_ID)):
        keys = [t for t in ex['Tracks'] if str(t['TrackID']) == str(tid) and t['infotype'] == 3]
        assert len(keys) == 1
        kf = keys[0]['Keyframes']
        fr = np.array([k['Frame'] for k in kf], int)
        val = np.array([k['LocalPosition'] for k in kf], float)
        for f0, f1 in sched.get(foot, []):
            for f in range(f0, f1 + 1):
                v = np.array([np.interp(f, fr, val[:, c]) for c in range(3)])
                assert np.allclose(v, AT.SENTINEL, atol=1e-6), (foot, f, v)


def test_patches_and_mapping_have_the_plugin_json_format():
    so = _need(os.path.join(OUT, SO_HEX + '.baked.json'))
    ex = _need(os.path.join(OUT, EXIT_HEX + '.baked.json'))
    for patch in (so, ex):
        assert set(patch.keys()) >= {'Length', 'Tracks'}
        for t in patch['Tracks']:
            assert isinstance(str(t['TrackID']), str) and isinstance(t['codec'], int) and isinstance(t['infotype'], int)
            for k in t['Keyframes']:
                assert isinstance(k['Frame'], int)
    assert len(_root_of(so)[0]) == LAST_SO + 1
    for t in [t for t in so['Tracks'] + ex['Tracks'] if str(t['TrackID']) == '0']:
        assert t['codec'] == 27 and t['infotype'] == 3
    m = _need(os.path.join(OUT, D15.MAPPING_NAME))
    assert len(m['clips']) == 2
    for c in m['clips']:
        for key in ('handle', 'name', 'source', 'enabled', 'prebaked', 'patch_translation', 'override'):
            assert key in c
        assert c['override'] == 'preview: dive15 grounded roll'
    assert m['clips'][1]['only_after'] == [D9.SO_NAME] and m['clips'][1]['arm_seconds'] == 2.0
