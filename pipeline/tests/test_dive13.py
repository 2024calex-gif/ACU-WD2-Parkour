"""dive13 (T25): the slow root during the roll and the get-up. Checks: the root before 1.20 s is the original (SlideOver, frames
0..72), C1 at 1.20 s, speed >= 0, z = 0 after 1.20 s, the Exit root (before 0.208 s the original, from 0.208 the played formula,
the window end at dive10.p_end_for), the events byte-identical to dive12 S18 (SlideOver) and dive11 (Exit), the JSON format of
both patches and the mapping, the skate sampler on synthetic data, and the Exit delta on synthetic data. The output checks read
out\\dive13 and skip when it is not written yet; the bake itself is not run here (it takes minutes)."""
import json
import os

import numpy as np
import pytest

import acuclip
import config
import dive9 as D9
import dive10 as D10
import dive13 as D13
import dumps

OUT = os.path.join(config.OUT, 'dive13')
OUT12 = os.path.join(config.OUT, 'dive12', 'S18')
OUT11 = os.path.join(config.OUT, 'dive11')
SO_HEX, EXIT_HEX = D9.SO_HEX, D9.EXIT_HEX
J = 72                                  # clip frame of 1.20 s
W_S18 = 1.5011                          # S18 window end (clip s)
LAST_SO = 98


def _load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _need(path):
    if not os.path.exists(path):
        pytest.skip('%s is not written yet' % path)
    return _load(path)


def _root_of(patch):
    tr = [t for t in patch['Tracks'] if str(t['TrackID']) == '0']
    assert len(tr) == 1, 'one TrackID 0 root track expected'
    kf = tr[0]['Keyframes']
    fr = np.array([k['Frame'] for k in kf], int)
    pos = np.array([k['LocalPosition'] for k in kf], float)
    return fr, pos


def _dense(fr, pos, n):
    F = np.arange(n)
    return np.stack([np.interp(F, fr, pos[:, c]) for c in range(3)], axis=1)


def _ev(patch):
    return sorted([t for t in patch['Tracks'] if t.get('infotype') == 5], key=lambda t: (str(t['TrackID']), t['Slot']))


# ------------------------------------------------------------------------------------------------ synthetic: the Exit delta
def test_exit_delta_is_linear_for_equal_speeds_and_starts_at_zero():
    p = np.linspace(0.0, 0.6, 61)
    D = D13.exit_delta(p, 2.0, 2.0, 0.4867)
    assert D[0] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(D, 2.0 * p, atol=1e-6)


def test_exit_delta_is_monotone_for_nonnegative_speeds_and_ends_at_the_nav_speed():
    p = np.linspace(0.0, 1.0, 201)
    D = D13.exit_delta(p, 2.9, 4.5, 0.4867)
    assert np.all(np.diff(D) >= -1e-12)
    slope_end = (D[-1] - D[-2]) / (p[1] - p[0])
    assert slope_end == pytest.approx(4.5, abs=1e-3)


def test_exit_delta_speed_profile_moves_from_v0_to_v1_across_the_window():
    p = np.linspace(0.0, 0.4867, 4001)
    D = D13.exit_delta(p, 2.9, 4.5, 0.4867)
    v = np.gradient(D, p)
    assert v[1] == pytest.approx(2.9, abs=0.05)                 # starts at v0
    assert v[-5] == pytest.approx(4.5, abs=0.05)                # ends at v1
    assert np.all(v >= 2.9 - 1e-6)                              # v1 > v0: no reversal


# ------------------------------------------------------------------------------------------------ synthetic: the skate sampler
def _synthetic(nf=12):
    """Two bones: bone 0 slides at 1 m/s along x on the ground; bone 1 stays 1 m up."""
    P = np.zeros((nf, 2, 3))
    P[:, 0, 0] = np.arange(nf) / 60.0
    P[:, 1, 2] = 1.0
    return P


def test_skate_sampler_reports_a_sliding_bone_on_the_ground_at_its_speed():
    P = _synthetic()
    h = P[:, :, 2].copy()
    sp, lab, runs = D13.skate_samples(P, np.arange(12), [0, 1], {0: 'A', 1: 'B'}, 1.0 / 60.0, h)
    assert runs == 1
    assert sp.size == 12
    assert np.allclose(sp, 1.0, atol=1e-6)
    assert set(lab) == {'A'}


def test_skate_sampler_drops_a_run_shorter_than_three_frames():
    P = _synthetic()
    h = P[:, :, 2].copy()
    h[2:, 0] = 0.5                                                 # contact only on frames 0 and 1
    sp, lab, runs = D13.skate_samples(P, np.arange(12), [0, 1], {0: 'A', 1: 'B'}, 1.0 / 60.0, h)
    assert sp.size == 0 and runs == 0


def test_skate_sampler_drops_a_bone_that_is_not_in_contact():
    P = _synthetic()
    h = P[:, :, 2].copy()
    h[:, 0] = 0.20                                                 # 20 cm above the surface: not in contact (12 cm)
    sp, lab, runs = D13.skate_samples(P, np.arange(12), [0, 1], {0: 'A', 1: 'B'}, 1.0 / 60.0, h)
    assert sp.size == 0


def test_skate_sampler_drops_a_contact_that_changes_bone_every_frame():
    nf = 12
    P = np.zeros((nf, 2, 3))
    P[:, :, 0] = np.arange(nf)[:, None] / 60.0
    h = np.zeros((nf, 2))
    h[::2, 1] = 1.0                                                # bone 0 low on even frames, bone 1 low on odd frames
    h[1::2, 0] = 1.0
    sp, lab, runs = D13.skate_samples(P, np.arange(nf), [0, 1], {0: 'A', 1: 'B'}, 1.0 / 60.0, h)
    assert sp.size == 0 and runs == 0


# ------------------------------------------------------------------------------------------------ the written SlideOver root
def test_root_before_1p20_is_the_original_clip_root():
    patch = _need(os.path.join(OUT, SO_HEX + '.baked.json'))
    so = dumps.load_by_name(D9.SO_NAME)
    F = np.arange(LAST_SO + 1)
    orig = acuclip.markers(so, F)['disp']
    fr, pos = _root_of(patch)
    new = _dense(fr, pos, LAST_SO + 1)
    assert np.array_equal(fr, np.arange(LAST_SO + 1))             # a key on every frame of the clip
    assert np.max(np.abs(new[:J + 1] - orig[:J + 1])) < 1e-8


def test_root_is_c1_at_1p20_s():
    patch = _need(os.path.join(OUT, SO_HEX + '.baked.json'))
    fr, pos = _root_of(patch)
    y = _dense(fr, pos, LAST_SO + 1)[:, 1]
    v = np.diff(y) * 60.0                                          # v[f-1] = speed between frames f-1 and f
    assert abs(y[J + 1] - y[J]) * 60.0 == pytest.approx(v[J], abs=1e-12)
    assert abs(v[J] - v[J - 1]) < 0.1                              # no velocity step at the join (the blend starts flat)
    assert abs(v[J + 1] - v[J]) < 0.5                              # the blend itself moves the speed smoothly after it


def test_root_speed_is_nonnegative_and_z_is_zero_after_1p20_s():
    patch = _need(os.path.join(OUT, SO_HEX + '.baked.json'))
    fr, pos = _root_of(patch)
    new = _dense(fr, pos, LAST_SO + 1)
    v = np.diff(new[:, 1]) * 60.0
    assert np.all(v >= -1e-9)
    assert np.max(np.abs(new[J + 1:, 2])) < 1e-9
    assert np.max(np.abs(new[:, 0])) < 1e-4


def test_slideover_events_are_the_dive12_s18_events():
    patch = _need(os.path.join(OUT, SO_HEX + '.baked.json'))
    ref = _need(os.path.join(OUT12, SO_HEX + '.baked.json'))
    a, b = _ev(patch), _ev(ref)
    assert len(a) == len(b) == 11
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ------------------------------------------------------------------------------------------------ the written Exit root
def test_exit_root_before_0p208_is_the_original_and_played_part_follows_the_formula():
    patch = _need(os.path.join(OUT, EXIT_HEX + '.baked.json'))
    ex = dumps.load_by_name(D9.EXIT_NAME)
    so_patch = _need(os.path.join(OUT, SO_HEX + '.baked.json'))
    F_E = np.arange(int(round(float(ex['Length']) * 60)) + 1)
    orig_E = acuclip.markers(ex, F_E)['disp']
    fr, pos = _root_of(patch)
    track = _dense(fr, pos, len(F_E))
    played = F_E / 60.0 >= D13.EXIT_O + 1e-9
    before = F_E / 60.0 < D13.EXIT_O - 1e-9
    assert np.max(np.abs(track[before, 1] - orig_E[before, 1])) < 1e-6
    # the joined root: root_S(W) + (track - track(0.208)) must equal root_S(W) + D(p) on the played frames
    frS, posS = _root_of(so_patch)
    so_track = _dense(frS, posS, LAST_SO + 1)
    root_W = float(np.interp(W_S18 * 60.0, np.arange(LAST_SO + 1), so_track[:, 1]))
    cut = float(np.interp(D13.EXIT_O * 60.0, F_E, orig_E[:, 1]))
    p = (F_E[played] / 60.0 - D13.EXIT_O) / D13.EXIT_R
    joined = root_W + (track[played, 1] - cut)
    v0 = float(np.interp(W_S18 * 60.0, np.arange(LAST_SO + 1), D13.root_velocity(so_track)))
    v1, _ = D13.nav_run_speed()
    D = D13.exit_delta(p, v0, v1, float(D10.p_end_for(D13.DELAY)))
    assert np.max(np.abs(joined - root_W - D)) < 1e-4


def test_exit_window_end_is_the_delay15_window():
    assert float(D10.p_end_for(D13.DELAY)) == pytest.approx(D10.EXIT_WIN + 15 / (60.0 * D10.EXIT_R), abs=1e-12)
    assert D13.DELAY == 15


def test_exit_events_are_the_dive11_delay15_events():
    patch = _need(os.path.join(OUT, EXIT_HEX + '.baked.json'))
    ref = _need(os.path.join(OUT11, EXIT_HEX + '.baked.json'))
    a, b = _ev(patch), _ev(ref)
    assert len(a) == len(b) == 23
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ------------------------------------------------------------------------------------------------ JSON format
def _check_patch_format(patch):
    assert set(patch.keys()) >= {'Length', 'Tracks'}
    assert isinstance(patch['Length'], (int, float))
    for t in patch['Tracks']:
        assert isinstance(str(t['TrackID']), str)
        assert isinstance(t['codec'], int) and isinstance(t['infotype'], int)
        for k in t['Keyframes']:
            assert isinstance(k['Frame'], int)


def test_patches_and_mapping_have_the_plugin_json_format():
    so = _need(os.path.join(OUT, SO_HEX + '.baked.json'))
    ex = _need(os.path.join(OUT, EXIT_HEX + '.baked.json'))
    _check_patch_format(so)
    _check_patch_format(ex)
    for patch, n_frames in ((so, LAST_SO + 1), (ex, None)):
        fr, pos = _root_of(patch)
        assert np.all(np.diff(fr) > 0)
        assert pos.shape[1] == 3
        if n_frames is not None:
            assert len(fr) == n_frames
    so_root = [t for t in so['Tracks'] if str(t['TrackID']) == '0'][0]
    ex_root = [t for t in ex['Tracks'] if str(t['TrackID']) == '0'][0]
    for t in (so_root, ex_root):
        assert t['codec'] == 27 and t['infotype'] == 3
        assert all(len(k['LocalPosition']) == 3 for k in t['Keyframes'])
    m = _need(os.path.join(OUT, 'mapping_dive13.json'))
    assert set(m.keys()) == {'clips'} and len(m['clips']) == 2
    for c in m['clips']:
        for key in ('handle', 'name', 'source', 'enabled', 'prebaked', 'patch_translation', 'override'):
            assert key in c
        assert c['override'] == 'preview: dive13 slow root during roll'
    exit_entry = m['clips'][1]
    assert exit_entry['only_after'] == [D9.SO_NAME] and exit_entry['arm_seconds'] == 2.0
    assert m['clips'][0]['source'] == SO_HEX + '.baked.json' and m['clips'][1]['source'] == EXIT_HEX + '.baked.json'


def test_dive9_to_dive12_defaults_are_unchanged():
    # dive13 only imports these modules; their defaults that it relies on must stay as dive12 left them
    assert D9.P_UP == 0.16 and D9.Z_UP == 0.80
    assert D10.EXIT_WIN == pytest.approx(0.228)
    assert D13.D_S == 18 and D13.K_PLANT == 1.0 and D13.K_ROLL == 1.0
