"""dive10 (T22): the Exit event shift rules, the timing of the delay, the JSON format and mapping of the written variants, the
hold after the window end, the SlideOver copy (identical to dive9), and the summary rules. Output checks read out\\dive10
and skip when it is not written yet; the bake itself is not run here (it takes minutes)."""
import filecmp
import json
import os

import numpy as np
import pytest

import config
import dive4 as D4
import dive10 as D10
import dive9 as D9
import dumps

OUT = os.path.join(config.OUT, 'dive10')
DUMP = os.path.join(config.ACU_DUMPS, 'F7919BDF0_xx_Parkour_Muraille_Exit_Fd100cm-Dn100cm.json')
DIRS = {8: os.path.join(OUT, 'delay08'), 15: os.path.join(OUT, 'delay15')}
SO9 = os.path.join(config.OUT, 'dive9', 'patches', D9.SO_HEX + '.baked.json')
LAST = 52                                    # round(0.866667 * 60): the Exit's last frame


def _load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _dump():
    if not os.path.exists(DUMP):
        pytest.skip('the Exit dump is not in data\\acu_dumps')
    return _load(DUMP)


def _keys(pairs):
    return [{'Frame': f, 'Int': v} for f, v in pairs]


# ------------------------------------------------------------------------------------------------ event shift rules
def test_keys_before_frame_13_stay_and_keys_from_13_move_by_the_delay():
    src = _keys([(0, 5), (12, 6), (13, 7), (18, 8), (26, 9)])
    out = D10.shift_event_keys(src, 8, LAST)
    assert out == _keys([(0, 5), (12, 6), (21, 7), (26, 8), (34, 9)])


def test_frame_12_is_not_moved_but_a_key_at_the_played_start_is():
    assert D10.shift_event_keys(_keys([(12, 1)]), 15, LAST) == _keys([(12, 1)])
    assert D10.shift_event_keys(_keys([(13, 1)]), 15, LAST) == _keys([(28, 1)])


def test_frames_past_the_last_frame_clamp_to_it_and_the_last_value_per_frame_is_kept():
    src = _keys([(0, 0), (38, 11), (50, 22), (52, 33)])
    out = D10.shift_event_keys(src, 15, LAST)
    assert out == _keys([(0, 0), (52, 33)])                   # 38 -> 53 and 50 -> 65 and 52 -> 67 all land on 52: the last wins


def test_clamped_keys_keep_the_order_and_a_partial_clamp_is_not_merged():
    src = _keys([(18, 1), (34, 2), (40, 3), (50, 4)])
    out = D10.shift_event_keys(src, 8, LAST)
    assert out == _keys([(26, 1), (42, 2), (48, 3), (52, 4)])


@pytest.mark.parametrize('delay', [0, 1, 8, 15, 30])
def test_shifted_keys_are_strictly_increasing_and_within_the_clip(delay):
    src = _keys([(0, 0), (2, 1), (6, 2), (12, 3), (13, 4), (18, 5), (26, 6), (34, 7), (38, 8), (40, 9), (50, 10), (52, 11)])
    out = D10.shift_event_keys(src, delay, LAST)
    frames = [k['Frame'] for k in out]
    assert all(b > a for a, b in zip(frames, frames[1:]))
    assert frames[0] == 0 and max(frames) <= LAST
    assert all(isinstance(k['Int'], int) for k in out)


def test_zero_delay_is_the_identity_on_a_sorted_track():
    src = _keys([(0, 0), (18, 4194352), (26, 4194353), (40, 4194352), (50, 4194304)])
    assert D10.shift_event_keys(src, 0, LAST) == src


def test_unsorted_source_keys_are_rejected():
    with pytest.raises(ValueError):
        D10.shift_event_keys(_keys([(10, 1), (5, 2)]), 8, LAST)


def test_event_tracks_keep_the_dump_slot_trackid_and_codec_on_a_synthetic_dump():
    dump = {'Length': 0.866667, 'Tracks': [
        {'TrackID': '743623600', 'codec': 1, 'infotype': 0, 'Keyframes': []},
        {'TrackID': '4294967261', 'codec': 57, 'infotype': 5, 'Keyframes': _keys([(0, 0), (26, 4194305), (50, 4194304)])},
        {'TrackID': '4294967264', 'codec': 59, 'infotype': 5, 'Keyframes': _keys([(0, 0), (10, 4194305)])},
        {'TrackID': '1', 'codec': 48, 'infotype': 4, 'Keyframes': []},
        {'TrackID': '4294967261', 'codec': 57, 'infotype': 5, 'Keyframes': _keys([(0, 0), (18, 4194517)])},
    ]}
    tracks = D10.event_patch_tracks(dump, 8)
    assert [t['Slot'] for t in tracks] == [1, 2, 4]
    assert [t['TrackID'] for t in tracks] == ['4294967261', '4294967264', '4294967261']
    assert [t['codec'] for t in tracks] == [57, 59, 57]
    assert all(t['infotype'] == 5 for t in tracks)
    assert tracks[0]['Keyframes'] == _keys([(0, 0), (34, 4194305), (52, 4194304)])
    assert tracks[1]['Keyframes'] == _keys([(0, 0), (10, 4194305)])          # 10 < 13: not moved
    assert tracks[2]['Keyframes'] == _keys([(0, 0), (26, 4194517)])


def test_real_exit_dump_has_23_event_tracks_and_the_patch_slots_match_it():
    dump = _dump()
    slots = [i for i, t in enumerate(dump['Tracks']) if t.get('infotype') == 5]
    assert len(slots) == 23
    tracks = D10.event_patch_tracks(dump, 8)
    assert [t['Slot'] for t in tracks] == slots
    for t in tracks:
        src = dump['Tracks'][t['Slot']]
        assert src['infotype'] == 5
        assert t['TrackID'] == str(int(src['TrackID']))
        assert t['codec'] == src['codec'] and t['codec'] in (57, 59)


def test_real_exit_dump_slot_73_moves_the_cut_event_from_26_to_34_for_delay_8():
    dump = _dump()
    src = dump['Tracks'][75]
    assert src['TrackID'] == '4294967261' and src['codec'] == 57
    assert {'Frame': 26, 'Int': 4194305} in src['Keyframes']                 # the 0x400001 cut event
    tracks = {t['Slot']: t for t in D10.event_patch_tracks(dump, 8)}
    assert {'Frame': 34, 'Int': 4194305} in tracks[75]['Keyframes']
    tracks15 = {t['Slot']: t for t in D10.event_patch_tracks(dump, 15)}
    assert {'Frame': 41, 'Int': 4194305} in tracks15[75]['Keyframes']


# ------------------------------------------------------------------------------------------------ timing of the delay
@pytest.mark.parametrize('delay', [8, 15])
def test_stand_up_and_window_timing_follow_the_delay(delay):
    assert D10.p_up_for(delay) == pytest.approx(0.16 + delay / (60 * D4.EXIT_R))
    assert D10.p_end_for(delay) == pytest.approx(D4.EXIT_WIN + delay / (60 * D4.EXIT_R))
    assert D10.p_up_for(delay) < D10.p_end_for(delay)


@pytest.mark.parametrize('delay', [8, 15])
def test_ease_completes_by_the_window_end_minus_50_ms(delay):
    clip_ease = D4.EXIT_O + D4.EXIT_R * D10.p_ease_for(delay)
    clip_end = D4.EXIT_O + D4.EXIT_R * D10.p_end_for(delay)
    assert clip_ease == pytest.approx(clip_end - D10.EASE_LEAD, abs=1e-9)
    assert clip_ease <= clip_end - 0.05 + 1e-9


def test_window_end_frames_are_dive9_plus_the_delay():
    assert D10.frame_of_played(D4.EXIT_WIN) == 26                              # 25.7 rounds to 26 (dive9's cut frame)
    assert D10.frame_of_played(D10.p_end_for(8)) == 34
    assert D10.frame_of_played(D10.p_end_for(15)) == 41


# ------------------------------------------------------------------------------------------------ written variants
@pytest.mark.parametrize('delay', [8, 15])
def test_exit_patch_json_format(delay):
    path = os.path.join(DIRS[delay], D9.EXIT_HEX + '.baked.json')
    if not os.path.exists(path):
        pytest.skip('dive10 outputs not written yet')
    p = _load(path)
    assert set(p.keys()) == {'Length', 'Tracks'}
    assert isinstance(p['Length'], float) and p['Length'] == pytest.approx(0.866667)
    body = [t for t in p['Tracks'] if t['infotype'] != 5]
    events = [t for t in p['Tracks'] if t['infotype'] == 5]
    assert len(body) == 59 and len(events) == 23
    for t in body:
        assert set(t.keys()) == {'TrackID', 'codec', 'infotype', 'Keyframes'}
        assert t['infotype'] in (0, 3) and isinstance(t['codec'], int)
        frames = [k['Frame'] for k in t['Keyframes']]
        assert all(b > a for a, b in zip(frames, frames[1:]))
    for t in events:
        assert set(t.keys()) == {'TrackID', 'codec', 'infotype', 'Slot', 'Keyframes'}
        assert isinstance(t['Slot'], int) and 0 <= t['Slot'] < 99
        assert t['codec'] in (57, 59)
        frames = [k['Frame'] for k in t['Keyframes']]
        assert all(isinstance(f, int) for f in frames)
        assert all(b > a for a, b in zip(frames, frames[1:]))
        assert frames[0] == 0 and max(frames) <= LAST
        assert all(set(k.keys()) == {'Frame', 'Int'} for k in t['Keyframes'])
    assert len({t['Slot'] for t in events}) == 23


@pytest.mark.parametrize('delay', [8, 15])
def test_exit_events_match_the_dump_and_the_shift_rule(delay):
    path = os.path.join(DIRS[delay], D9.EXIT_HEX + '.baked.json')
    if not os.path.exists(path):
        pytest.skip('dive10 outputs not written yet')
    dump = _dump()
    written = {t['Slot']: t for t in _load(path)['Tracks'] if t['infotype'] == 5}
    expect = {t['Slot']: t for t in D10.event_patch_tracks(dump, delay)}
    assert written == expect


@pytest.mark.parametrize('delay', [8, 15])
def test_mapping_is_the_dive9_schema_with_the_dive10_override(delay):
    path = os.path.join(DIRS[delay], D10.MAPPING_NAME)
    if not os.path.exists(path):
        pytest.skip('dive10 mapping not written yet')
    m = _load(path)
    assert set(m.keys()) == {'clips'} and len(m['clips']) == 2
    so, ex = m['clips']
    ov = 'preview: dive10 exit events +%d' % delay
    for c in (so, ex):
        assert set(c.keys()) >= {'handle', 'name', 'source', 'enabled', 'prebaked', 'patch_translation', 'override'}
        assert c['enabled'] is True and c['prebaked'] is True and c['patch_translation'] is True
        assert c['override'] == ov
    assert so['source'] == D9.SO_HEX + '.baked.json' and ex['source'] == D9.EXIT_HEX + '.baked.json'
    assert 'only_after' not in so
    assert ex['only_after'] == [D9.SO_NAME] and ex['arm_seconds'] == 2.0
    d9 = os.path.join(config.OUT, 'dive9', 'patches', 'mapping_dive9.json')
    if os.path.exists(d9):
        ref = _load(d9)
        for a, b in zip(m['clips'], ref['clips']):
            assert set(a.keys()) == set(b.keys())


@pytest.mark.parametrize('delay', [8, 15])
def test_slideover_patch_is_identical_to_dive9(delay):
    path = os.path.join(DIRS[delay], D9.SO_HEX + '.baked.json')
    if not (os.path.exists(path) and os.path.exists(SO9)):
        pytest.skip('dive10 or dive9 SlideOver patch not written yet')
    assert filecmp.cmp(path, SO9, shallow=False)


@pytest.mark.parametrize('delay', [8, 15])
def test_the_exit_clip_holds_its_pose_and_reference_after_the_window_end(delay):
    path = os.path.join(DIRS[delay], D9.EXIT_HEX + '.baked.json')
    if not os.path.exists(path):
        pytest.skip('dive10 outputs not written yet')
    p = _load(path)
    hold = D10.frame_of_played(D10.p_end_for(delay))
    ref = [t for t in p['Tracks'] if t['infotype'] == 3 and t['TrackID'] == str(D9.D8.REF_ID)][0]['Keyframes']
    base = dumps.sample_position(ref, hold)
    for f in range(hold, LAST + 1):
        assert np.allclose(dumps.sample_position(ref, f), base, atol=1e-6), f
    for t in p['Tracks']:
        if t['infotype'] != 0:
            continue
        a = dumps.sample_rotation(t['Keyframes'], hold)
        for f in range(hold, LAST + 1):
            b = dumps.sample_rotation(t['Keyframes'], f)
            ang = np.degrees(2.0 * np.arccos(min(1.0, abs(float(np.dot(a, b))))))
            assert ang <= 0.5, (t['TrackID'], f, ang)


# ------------------------------------------------------------------------------------------------ summary and layout
def test_summary_meets_the_checks_it_reports():
    path = D10.SUMMARY
    if not os.path.exists(path):
        pytest.skip('dive10 summary not written yet')
    s = _load(path)
    for d in (8, 15):
        m = s['variants'][str(d)]
        assert m['rise_speed_joined_post_W_mps'] <= 4.0 + 1e-6
        assert m['rise_speed_exit_clip_mps'] <= 4.0 + 1e-6
        assert m['z_at_p_up'] >= D9.Z_UP_MIN
        assert m['gap_at_window_end_max_patched'] <= D9.GAP_GOAL
        assert not m['inv_exit'] and m['tilt_exit_max'] <= D9.TILT_MAX
        assert m['pen_patch_cm'] <= D9.PEN_MAX + 1e-6
        assert m['touch_min'] >= D9.TOUCH_MIN
        assert m['lift_S_frozen'] is True
        assert m['feet_free_to_frame'] == 13


def test_out_dive10_holds_only_the_expected_files():
    if not os.path.isdir(OUT):
        pytest.skip('out\\dive10 not written yet')
    assert set(os.listdir(OUT)) <= {'delay08', 'delay15', 'report.md', 'summary.json'}
    for d in (8, 15):
        names = set(os.listdir(DIRS[d])) if os.path.isdir(DIRS[d]) else set()
        assert names <= {D9.SO_HEX + '.baked.json', D9.EXIT_HEX + '.baked.json', D10.MAPPING_NAME}
