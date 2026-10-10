"""dive14 (T26): the body's forward path in the velocity domain (VelPlace). Checks: VelPlace equals dive4.Place when the flight
scale is 1, the scale blend (s_f before the blend, 1 after it, C1 ends), the velocity-domain increments (s * WD2 increments),
the placement map is monotone, the hand rule at (s_f 1.40, shift 0.22), the native windows, dive4.Place is restored after the
context manager, the defaults of dive4 to dive13 and wd2_path are unchanged, and (when written) the output patches: the
SlideOver root identical before 1.20 s, the events byte-identical to dive12 S18 and dive11, the JSON format and the mapping.
The output checks skip when out\\dive14 is not written; the bake itself is not run here (it takes minutes)."""
import json
import os

import numpy as np
import pytest

import acuclip
import config
import dive4 as D4
import dive9 as D9
import dive10 as D10
import dive12 as D12
import dive13 as D13
import dive14 as D14
import dive5 as F5
import wd2_path as WP
import dumps

OUT = os.path.join(config.OUT, 'dive14')
OUT12 = os.path.join(config.OUT, 'dive12', 'S18')
OUT11 = os.path.join(config.OUT, 'dive11')
SO_HEX, EXIT_HEX = D9.SO_HEX, D9.EXIT_HEX
J = 72
LAST_SO = 98


@pytest.fixture(scope='module')
def ctx():
    return F5.make_ctx()


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


# ------------------------------------------------------------------------------------------------ the placement
def test_velplace_equals_dive4_place_when_the_flight_scale_is_one(ctx):
    y_take = 3.7
    u = np.arange(0.45, 1.30, 0.01)
    pts = WP.path_at(ctx.sol, u)
    a = D14.VelPlace(ctx.sol, 1.0, y_take).map_points(pts)
    b = D4.Place(ctx.sol, 1.0, y_take).map_points(pts)
    assert np.allclose(a, b, atol=2e-4)


def test_scale_is_s_f_before_the_blend_and_one_after_it(ctx):
    s_f = 1.40
    tab = D14._tables(ctx.sol, s_f)
    u, s = tab['u'], tab['s']
    assert np.allclose(s[u <= D14.BLEND_WD2[0] + 1e-9], s_f)
    assert np.allclose(s[u >= D14.BLEND_WD2[1] - 1e-9], 1.0)
    blend = (u > D14.BLEND_WD2[0]) & (u < D14.BLEND_WD2[1])
    assert np.all(np.diff(s[blend]) <= 1e-12)                                       # monotone down, no overshoot
    # C1 at the ends: the derivative of the smoothstep is 0 there (the step between samples is small at the join)
    assert abs(s[np.searchsorted(u, D14.BLEND_WD2[0]) + 1] - s_f) < 1e-3


def test_velocity_increments_are_the_scale_times_the_wd2_increments(ctx):
    s_f = 1.40
    tab = D14._tables(ctx.sol, s_f)
    u, Py, cum = tab['u'], tab['Py'], tab['cum']
    def ratio(a, b):
        ia, ib = np.argmin(np.abs(u - a)), np.argmin(np.abs(u - b))
        return (cum[ib] - cum[ia]) / (Py[ib] - Py[ia])
    # windowed displacements: the mapped displacement over a window is the scale times the WD2 displacement there
    assert ratio(0.50, 0.60) == pytest.approx(s_f, rel=1e-3)
    assert ratio(0.80, 1.20) == pytest.approx(1.0, rel=1e-3)


def test_placement_is_monotone_in_the_forward_direction(ctx):
    pl = D14.VelPlace(ctx.sol, 1.40, 3.7)
    u = np.arange(0.40, 1.70, 0.005)
    y = pl.map_points(WP.path_at(ctx.sol, u))[:, 1]
    assert np.all(np.diff(y) > 0)


def test_hand_rule_holds_at_the_chosen_placement(ctx):
    clear, hy = D14.hand_clearance(ctx, D14.S_FLIGHT, D14.SHIFT)
    assert clear >= D14.FAR_CLEAR - 1e-9
    assert min(hy.values()) >= float(ctx.ob['far_y']) + 0.2 - 1e-9
    assert D14.S_FLIGHT == pytest.approx(1.40) and D14.SHIFT == pytest.approx(0.22)


def test_dive4_place_is_restored_after_the_context():
    before = D4.Place
    with D14.velocity_place():
        assert D4.Place is D14.VelPlace
    assert D4.Place is before


def test_native_windows_follow_the_chain_timing(ctx):
    t = np.round(np.arange(0.2, 1.8 + 1e-9, 1.0 / 60.0), 6)
    w_W = 1.3311
    ph = D14.native_phases({'t': t}, w_W)
    assert set(ph) == {'plant', 'roll', 'get-up', 'Exit window'}
    assert t[ph['plant']][0] == pytest.approx(0.75, abs=1e-6)
    assert t[ph['plant']][-1] == pytest.approx(0.92, abs=1.0 / 60.0)     # the last 60 Hz sample at or before 0.92
    assert t[ph['roll']][-1] <= w_W + 1e-9
    assert t[ph['get-up']][0] > w_W


# ------------------------------------------------------------------------------------------------ defaults unchanged
def test_defaults_of_dive4_to_dive13_are_unchanged():
    assert D12.S_SCALE == 1.35 and D12.SHIFT == 0.20 and D12.T_TO == 0.62
    assert D13.D_S == 18 and D13.K_PLANT == 1.0 and D13.K_ROLL == 1.0
    assert D13.OVERRIDE == 'preview: dive13 slow root during roll'
    assert D9.P_UP == 0.16 and D10.EXIT_WIN == pytest.approx(0.228)


# ------------------------------------------------------------------------------------------------ the output patches
def test_slideover_root_before_1p20_is_the_original():
    patch = _need(os.path.join(OUT, SO_HEX + '.baked.json'))
    so = dumps.load_by_name(D9.SO_NAME)
    orig = acuclip.markers(so, np.arange(LAST_SO + 1))['disp']
    fr, pos = _root_of(patch)
    new = np.stack([np.interp(np.arange(LAST_SO + 1), fr, pos[:, c]) for c in range(3)], axis=1)
    assert np.max(np.abs(new[:J + 1] - orig[:J + 1])) < 1e-8


def test_slideover_root_is_c1_at_1p20_and_nonnegative():
    patch = _need(os.path.join(OUT, SO_HEX + '.baked.json'))
    fr, pos = _root_of(patch)
    y = np.stack([np.interp(np.arange(LAST_SO + 1), fr, pos[:, c]) for c in range(3)], axis=1)[:, 1]
    v = np.diff(y) * 60.0
    assert abs(v[J] - v[J - 1]) < 0.1
    assert np.all(v >= -1e-9)


def test_events_are_byte_identical_to_dive12_s18_and_dive11():
    so = _need(os.path.join(OUT, SO_HEX + '.baked.json'))
    so12 = _need(os.path.join(OUT12, SO_HEX + '.baked.json'))
    assert json.dumps(_ev(so), sort_keys=True) == json.dumps(_ev(so12), sort_keys=True)
    ex = _need(os.path.join(OUT, EXIT_HEX + '.baked.json'))
    ex11 = _need(os.path.join(OUT11, EXIT_HEX + '.baked.json'))
    assert len(_ev(ex)) == 23
    assert json.dumps(_ev(ex), sort_keys=True) == json.dumps(_ev(ex11), sort_keys=True)


def test_patches_and_mapping_have_the_plugin_json_format():
    so = _need(os.path.join(OUT, SO_HEX + '.baked.json'))
    ex = _need(os.path.join(OUT, EXIT_HEX + '.baked.json'))
    for patch in (so, ex):
        assert set(patch.keys()) >= {'Length', 'Tracks'}
        for t in patch['Tracks']:
            assert isinstance(str(t['TrackID']), str) and isinstance(t['codec'], int) and isinstance(t['infotype'], int)
            for k in t['Keyframes']:
                assert isinstance(k['Frame'], int)
    so_root = _root_of(so)
    ex_root = _root_of(ex)
    assert len(so_root[0]) == LAST_SO + 1 and len(ex_root[0]) > 0
    for t in [t for t in so['Tracks'] + ex['Tracks'] if str(t['TrackID']) == '0']:
        assert t['codec'] == 27 and t['infotype'] == 3
    m = _need(os.path.join(OUT, D14.MAPPING_NAME))
    assert len(m['clips']) == 2
    for c in m['clips']:
        for key in ('handle', 'name', 'source', 'enabled', 'prebaked', 'patch_translation', 'override'):
            assert key in c
        assert c['override'] == 'preview: dive14 roll unstretched'
    assert m['clips'][1]['only_after'] == [D9.SO_NAME] and m['clips'][1]['arm_seconds'] == 2.0
