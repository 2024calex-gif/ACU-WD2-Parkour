"""Tests for dive_root.py (replacement TrackID 0 root for the Sprint SlideOver) and dive_chain2.py (the T13 bake):
root identity outside the dive window, far-edge crossing, clearance, monotone speed, C1 joins, the Exit root-delta
convention, the written patch format, the gate, and the enabled flags against the recorded validation."""
import json
import os
import zlib

import numpy as np
import pytest

import acuclip
import bake2
import bake_chain as BC
import bake_dive as BD
import config
import dive_root as DR
import dumps
import retarget as R

OUT = os.path.join(config.OUT, 'dive2')
MAPPING = os.path.join(OUT, 'patches', 'mapping_dive2.json')
SUMMARY = os.path.join(OUT, 'chain_summary.json')
SO_HEX = '47BFB02DF'
EXIT_HEX = 'F7919BDF0'


@pytest.fixture(scope='module')
def so_dump():
    return dumps.load_by_name(DR.SO_NAME)


@pytest.fixture(scope='module')
def root(so_dump):
    return DR.build_root(so_dump)


# ------------------------------------------------------------------------------------------------ the root curve
def test_identical_to_the_original_outside_the_dive_window(root):
    F, orig, new = root['F'], root['orig'], root['new']
    t = F / acuclip.FPS
    outside = (t <= DR.T_TAKEOFF + 1e-9) | (t >= DR.T_JOIN - 1e-9)
    assert np.array_equal(new[outside], orig[outside])


def test_identical_to_the_original_at_the_join_and_takeoff_frames(root):
    i_take = int(round(DR.T_TAKEOFF * acuclip.FPS))
    i_join = int(round(DR.T_JOIN * acuclip.FPS))
    assert root['new'][i_take, 1] == pytest.approx(root['orig'][i_take, 1], abs=1e-12)
    assert root['new'][i_join, 1] == pytest.approx(root['orig'][i_join, 1], abs=0.05)     # within 5 cm (exact here)
    assert root['new'][i_join, 2] == pytest.approx(root['orig'][i_join, 2], abs=1e-5)


def test_far_edge_is_crossed_early_and_at_the_target(root):
    m = root['metrics']
    assert 0.75 - 1e-9 <= m['t_cross'] <= 0.80 + 1.0 / acuclip.FPS
    assert abs(m['t_cross'] - m['t_far']) <= 1.0 / acuclip.FPS + 1e-9
    assert 0.5 <= m['y_beyond_far_at_z0'] <= 0.8                    # z back to 0 about 0.5 to 0.8 m beyond the edge


def test_clearance_over_the_box_and_peak_height(root):
    m = root['metrics']
    assert 1.10 <= m['z_peak'] <= 1.20
    assert m['clearance_over_box_cm'] >= 10.0
    assert m['clearance_over_box_cm'] <= 16.0                        # about 10 to 15 cm, one frame of slack
    assert m['z_join'] == pytest.approx(0.0, abs=1e-5)


def test_root_is_monotone_and_speed_stays_in_bounds(root):
    m = root['metrics']
    y = root['new'][:, 1]
    assert m['monotone'] and np.all(np.diff(y) >= 0.0)
    assert m['v_min'] >= DR.MIN_SPEED - 1e-9
    assert m['v_max'] <= DR.MAX_SPEED + 1e-9


def test_c1_joins_by_construction(root):
    g = root['g']
    knots = root['knots']
    assert knots[0] == pytest.approx(DR.T_TAKEOFF) and knots[-1] == pytest.approx(DR.T_JOIN)
    assert g[0] == pytest.approx(0.0, abs=1e-12) and g[-1] == pytest.approx(0.0, abs=1e-12)
    # the speed change is g: zero at both joins, so the root velocity matches the original there
    assert np.interp(DR.T_TAKEOFF, knots, g) == pytest.approx(0.0, abs=1e-12)
    assert np.interp(DR.T_JOIN, knots, g) == pytest.approx(0.0, abs=1e-12)


def test_displacement_integrates_to_the_constraints(root, so_dump):
    grid, D = root['grid'], root['D_grid']
    assert D[-1] == pytest.approx(0.0, abs=1e-6)                     # D(1.2) = 0
    yo = DR.rows_at(root['orig'], grid)[:, 1]
    i = int(np.argmin(np.abs(grid - root['t_far'])))
    assert D[i] == pytest.approx(root['geometry']['far_y'] - yo[i], abs=1e-6)


def test_z_arc_is_zero_at_the_joins_and_smooth(root):
    z = root['new'][:, 2]
    t = root['F'] / acuclip.FPS
    assert np.all(z[t <= DR.T_TAKEOFF + 1e-9] == 0.0)
    assert np.array_equal(z[t >= DR.T_JOIN - 1e-9], root['orig'][t >= DR.T_JOIN - 1e-9, 2])
    dz = np.diff(z) * acuclip.FPS
    assert np.max(np.abs(dz)) < 12.0                                 # m/s, the rise is the fastest part


def test_root_keys_cover_every_frame_in_the_plugin_format(root, so_dump):
    keys = root['keys']
    frames = [k['Frame'] for k in keys]
    assert frames == list(range(len(keys)))
    assert len(keys) == int(round(float(so_dump['Length']) * acuclip.FPS)) + 1
    assert all(len(k['LocalPosition']) == 3 for k in keys)
    track = DR.root_track(keys, so_dump['Length'])
    assert track['TrackID'] == '0' and track['codec'] == 27 and track['infotype'] == 3


# ------------------------------------------------------------------------------------------------ bake helpers
def test_world_with_the_dumps_own_root_is_unchanged():
    """bake_dive.world(disp=None) is the old behaviour: the dump's own root displacement."""
    rig = R.Rig()
    dump = dumps.load_by_name(BC.MAIN_SO)
    seq = acuclip.pose_sequence(dump, step=1)
    z = np.zeros(len(seq.frames))
    P_old = acuclip.world_positions(bake2.composed(seq, seq.local_rot, seq.local_pos[:, 0]), dump)
    P_new = BD.world(seq, dump, seq.local_rot, seq.local_pos[:, 0], z, z)
    assert np.allclose(P_old, P_new, atol=1e-12)
    disp = acuclip.markers(dump, seq.frames)['disp']
    P_disp = BD.world(seq, dump, seq.local_rot, seq.local_pos[:, 0], z, z, disp=disp)
    assert np.allclose(P_old, P_disp, atol=1e-12)


def test_exit_root_delta_keeps_the_exit_on_the_ground():
    """The Exit plays from its root DELTA: z = 0 over its played window, and y continues from the SlideOver root at W."""
    import dive_chain2 as D
    so_dump = dumps.load_by_name(BC.MAIN_SO)
    mp = BD.clip_model(BC.MAIN_SO, BD.load_play_model_so(), float(so_dump['Length']))
    root = DR.build_root(so_dump)
    ex = dumps.load_by_name(BC.EXIT_NAME)
    ex_seq = acuclip.pose_sequence(ex, step=1)
    d, root_w, tW, cut = D.exit_root_delta(root, mp, ex, ex_seq)
    win = bake2.window_frames(BC.EXIT_O, BC.EXIT_R * BC.EXIT_WIN, int(ex_seq.frames[-1]))
    assert np.max(np.abs(d[win, 2])) < 0.01                          # the drop happened before the cut
    assert d[int(round(BC.EXIT_O * 60)), 1] == pytest.approx(root_w[1], abs=1e-6)
    # the absolute displacement is -1 m at the cut (the old convention): the joined delta removes it
    assert float(acuclip.markers(ex, ex_seq.frames)['disp'][int(round(BC.EXIT_O * 60)), 2]) < -0.9


# ------------------------------------------------------------------------------------------------ gate and status
def test_status_and_narrow_rules():
    import dive_chain2 as D
    assert D.status_of([]) == 'PASS'
    assert D.status_of(['non-finger penetration 7.0 cm']) == 'OVERRIDE'
    assert D.status_of(['hand touch 2 frames (need 4)']) == 'OVERRIDE'
    assert D.status_of(['hand touch 0 frames (need 4)']) == 'FAIL'
    assert D.status_of(['continuity at W 52 deg (> 45)']) == 'OVERRIDE'
    assert D.status_of(['continuity at W 177 deg (> 45)']) == 'FAIL'
    assert D.status_of(['Delta step 59.9 cm/frame']) == 'FAIL'
    assert D.status_of(['non-finger penetration 7.0 cm', 'Delta step 59.9 cm/frame']) == 'FAIL'


# ------------------------------------------------------------------------------------------------ outputs
def _mapping():
    if not os.path.exists(MAPPING):
        pytest.skip('dive_chain2 has not written its mapping yet')
    with open(MAPPING, encoding='utf-8') as f:
        return json.load(f)['clips']


def _summary():
    if not os.path.exists(SUMMARY):
        pytest.skip('no chain_summary.json yet')
    with open(SUMMARY, encoding='utf-8') as f:
        return json.load(f)


def test_mapping_entries_follow_the_spec():
    clips = _mapping()
    so = next(c for c in clips if c['name'] == BC.MAIN_SO)
    ex = next(c for c in clips if c['name'] == BC.EXIT_NAME)
    assert so['handle'] == '0x' + SO_HEX and so['source'] == SO_HEX + '.baked.json'
    assert so['prebaked'] is True and so['patch_translation'] is True and 'only_after' not in so
    assert ex['handle'] == '0x' + EXIT_HEX and ex['source'] == EXIT_HEX + '.baked.json'
    assert ex['prebaked'] is True and ex['patch_translation'] is True
    assert ex['only_after'] == [BC.MAIN_SO] and ex['arm_seconds'] == 2.0


def test_mapping_enabled_flags_match_the_recorded_status():
    s = _summary()
    clips = {c['name']: c for c in _mapping()}
    so_ok = s['slideover']['status'] in ('PASS', 'OVERRIDE')
    assert clips[BC.MAIN_SO]['enabled'] == so_ok
    assert clips[BC.EXIT_NAME]['enabled'] == (s['exit']['status'] in ('PASS', 'OVERRIDE') and so_ok)
    if not so_ok:
        assert not clips[BC.MAIN_SO]['enabled']
    for name, status in ((BC.MAIN_SO, s['slideover']['status']), (BC.EXIT_NAME, s['exit']['status'])):
        if status == 'OVERRIDE':
            assert clips[name].get('override', '').startswith('preview')


def test_written_patch_format_and_root_track():
    path = os.path.join(OUT, 'patches', SO_HEX + '.baked.json')
    if not os.path.exists(path):
        pytest.skip('no SlideOver patch yet')
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    assert set(d.keys()) == {'Length', 'Tracks'}
    root_tracks = [t for t in d['Tracks'] if t['TrackID'] == '0']
    assert len(root_tracks) == 1
    rt = root_tracks[0]
    assert rt['codec'] == 27 and rt['infotype'] == 3
    frames = [k['Frame'] for k in rt['Keyframes']]
    assert frames == sorted(frames) and len(set(frames)) == len(frames)
    trans = [t for t in d['Tracks'] if t['infotype'] == 3 and t['TrackID'] == str(bake2.REF_TID)]
    assert len(trans) == 1
    rot = [t for t in d['Tracks'] if t['infotype'] == 0]
    assert rot and all(t['codec'] == 1 for t in rot)
    for t in rot:
        for k in t['Keyframes']:
            assert abs(np.linalg.norm(k['LocalRotation']) - 1.0) < 5e-4


def test_root_track_in_the_patch_is_the_replacement_curve(root):
    path = os.path.join(OUT, 'patches', SO_HEX + '.baked.json')
    if not os.path.exists(path):
        pytest.skip('no SlideOver patch yet')
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    rt = next(t for t in d['Tracks'] if t['TrackID'] == '0')
    got = np.array([k['LocalPosition'] for k in rt['Keyframes']])
    assert np.allclose(got[:, 1:], root['new'][:, 1:], atol=1e-6)


def test_noroot_package_has_no_root_track():
    path = os.path.join(OUT, 'patches_noroot', SO_HEX + '.baked.json')
    if not os.path.exists(path):
        pytest.skip('no noroot package yet')
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    assert not any(t['TrackID'] == '0' for t in d['Tracks'])


def test_exit_patch_has_no_root_track_and_keeps_the_reference_track():
    path = os.path.join(OUT, 'patches', EXIT_HEX + '.baked.json')
    if not os.path.exists(path):
        pytest.skip('no Exit patch yet')
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    assert not any(t['TrackID'] == '0' for t in d['Tracks'])
    assert any(t['TrackID'] == str(bake2.REF_TID) and t['infotype'] == 3 for t in d['Tracks'])
