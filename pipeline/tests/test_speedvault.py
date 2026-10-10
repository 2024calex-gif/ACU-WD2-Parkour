"""Tests for speedvault.py: event file format, patch and mapping format, time map, bake2 defaults untouched."""
import json
import os

import numpy as np
import pytest

import bake2
import config
import speedvault as sv

EV_PATH = sv.EVENTS_JSON
EVENT_KEYS = ('stem', 'target', 'static_until', 'takeoff', 'hand_plant_start', 'hand_plant_end', 'hands_on',
              'planting_hand', 'landing', 'landing_foot', 'next_step', 'comment')


@pytest.fixture(scope='module')
def events():
    with open(EV_PATH, encoding='utf-8') as f:
        return json.load(f)


def _clip_events(events):
    return {k: v for k, v in events.items() if not k.startswith('_')}


# ------------------------------------------------------------------------------------------------ event file
def test_events_file_has_both_clips_with_required_fields(events):
    ce = _clip_events(events)
    assert set(ce) == {'speed_1m_100cm', 'speed_2m_100cm'}
    for key, ev in ce.items():
        for k in EVENT_KEYS:
            assert k in ev, '%s misses %s' % (key, k)
        assert ev['hands_on'] == 'top'
        assert ev['planting_hand'] in ('LeftHand', 'RightHand')
        assert ev['landing_foot'] in ('LeftFoot', 'RightFoot')


def test_events_are_ordered_in_time(events):
    for key, ev in _clip_events(events).items():
        assert 0.0 <= ev['static_until'] <= ev['takeoff'], key
        assert ev['takeoff'] < ev['hand_plant_start'] < ev['hand_plant_end'], key
        assert ev['hand_plant_start'] < ev['landing'] < ev['next_step'], key


def test_event_stems_exist_as_wd2_mab(events):
    index = sv_index()
    for key, ev in _clip_events(events).items():
        assert ev['stem'] in index, key


def sv_index():
    import wd2
    return wd2.mab_index()


def test_event_targets_match_clip_list(events):
    names = {c['name'] for c in sv.CLIPS}
    assert {ev['target'] for ev in _clip_events(events).values()} == names


# ------------------------------------------------------------------------------------------------ time map
def test_time_map_is_monotonic_for_the_speedvault_anchors(events):
    """ACU anchors of the two clips (takeoff, hand plant on the top, feet down) against the WD2 events."""
    acu = {'speed_1m_100cm': dict(takeoff=0.45, hand_plant=0.9167, landing=1.15),
           'speed_2m_100cm': dict(takeoff=0.45, hand_plant=0.6167, landing=1.4333)}
    for key, ev in _clip_events(events).items():
        wd2_ev = dict(takeoff=ev['takeoff'], hand_plant_start=ev['hand_plant_start'],
                      hand_plant_end=ev['hand_plant_end'], landing=ev['landing'],
                      static_until=ev['static_until'], duration=3.0)
        tm = bake2.time_map(acu[key], wd2_ev, 2.4)
        t = np.linspace(0.0, 2.4, 2401)
        w = tm(t)
        assert np.all(np.diff(w) >= -1e-12), key
        np.testing.assert_allclose(tm(np.array([0.45, 0.6167 if key.endswith('2m_100cm') else 0.9167, 1.15 if key.endswith('1m_100cm') else 1.4333])),
                                   [ev['takeoff'], ev['hand_plant_start'], ev['landing']], atol=1e-9)


# ------------------------------------------------------------------------------------------------ patch and mapping format
def _mapping():
    p = os.path.join(sv.OUT_DIR, 'mapping_speedvault.json')
    if not os.path.exists(p):
        pytest.skip('speedvault has not written its outputs yet')
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def test_mapping_entries_have_the_patch_schema():
    m = _mapping()
    assert len(m['clips']) == len(sv.CLIPS)
    for e in m['clips']:
        assert set(('handle', 'name', 'source', 'enabled', 'prebaked', 'patch_translation', 'pool_weight')) <= set(e)
        assert e['handle'].startswith('0x')
        assert e['prebaked'] is True and e['patch_translation'] is True
        assert isinstance(e['enabled'], bool)
        assert e['pool_weight'] == 0.5
        if e['enabled']:
            assert e.get('override', '').startswith('preview:'), e['name']
        assert e['name'].startswith('xx_Parkour_Sprint_SlideOver_')


def test_patch_files_follow_the_patch_format():
    m = _mapping()
    for e in m['clips']:
        path = os.path.join(sv.OUT_DIR, e['source'])
        assert os.path.exists(path), e['source']
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        assert isinstance(d['Length'], float) and d['Length'] > 0
        assert d['Tracks'], e['source']
        n_frames = int(round(d['Length'] * 60))
        has_rot = has_pos = False
        for tr in d['Tracks']:
            assert isinstance(tr['TrackID'], str) and tr['TrackID'].isdigit()
            assert tr['codec'] in (1, 27)
            has_rot |= tr['codec'] == 1
            has_pos |= tr['codec'] == 27
            frames = [k['Frame'] for k in tr['Keyframes']]
            assert frames == sorted(frames) and len(set(frames)) == len(frames)
            assert all(0 <= f <= n_frames for f in frames)
            key = 'LocalRotation' if tr['codec'] == 1 else 'LocalPosition'
            assert all(len(k[key]) in (3, 4) for k in tr['Keyframes'])
        assert has_rot and has_pos, 'a speed-vault patch has rotation and Reference position tracks'


# ------------------------------------------------------------------------------------------------ bake2 defaults untouched
def test_bake2_defaults_are_unchanged():
    assert bake2.FADE_IN == 0.10
    assert bake2.FADE_OUT == 0.15
    assert bake2.STEP_FRAMES == 2
    assert bake2.FPS == 60
    assert bake2.T9E_FADE_IN_MAX == 0.20 and bake2.T9E_FADE_IN_FRAC == 0.30
    assert bake2.T9E_FADE_OUT_MAX == 0.25 and bake2.T9E_FADE_OUT_FRAC == 0.35
    assert bake2.T9E_SIDES == (('Left', 'LeftHand'), ('Right', 'RightHand'))
    assert bake2.T9C_HANDS == ('LeftHand', 'RightHand')
    assert bake2.TARGET_RX == r'^xx_Parkour_Sprint_SkipOver_'
    assert bake2.SOURCE_EVENT == 'catpass_kong'
    assert bake2.TAIL_ORIG == 'orig' and bake2.TAIL_WD2 == 'wd2' and bake2.TAIL_HOLD == 'hold'


def test_planting_sides_restores_bake2_sides():
    before = bake2.T9E_SIDES
    with sv.planting_sides(sv.PLANTING_SIDES):
        assert bake2.T9E_SIDES == (('Left', 'LeftHand'),)
    assert bake2.T9E_SIDES == before


def test_speedvault_does_not_write_into_the_installed_patches():
    assert os.path.normcase(os.path.abspath(sv.OUT_DIR)) != os.path.normcase(os.path.abspath(config.PATCHES))
    assert os.path.normcase(os.path.abspath(sv.OUT_DIR)).startswith(os.path.normcase(os.path.abspath(config.OUT)))


def test_play_cut_gives_positive_windows():
    model = sv.load_model()
    for c in sv.CLIPS:
        o, w = sv.play_cut(model, c['name'])
        assert o > 0 and w > 0.5, c['name']


def test_elbow_continuity_solver_matches_ik_without_a_pole():
    """solve_sequence_continuous with no hand pole reproduces ik.solve_sequence for an unchanged pose (weight 0)."""
    import retarget as R
    import acuclip as ac
    import dumps as dp
    rig = R.Rig()
    dump = dp.load_by_handle(sv.CLIPS[0]['handle'])
    seq = ac.pose_sequence(dump, step=1)
    import ik
    F = len(seq.frames)
    rr = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (F, 1))
    rp = np.zeros((F, 3))
    tg = {'LeftHand': [(40, np.array([0.0, 5.2, 1.0]), 0.0)]}
    a = sv.solve_sequence_continuous(seq, rr, rp, tg)
    b = ik.solve_sequence(seq, rr, rp, tg)
    np.testing.assert_allclose(a.local_rot, b.local_rot, atol=1e-12)
    assert np.array_equal(np.asarray(a.local_rot), np.asarray(seq.local_rot))
