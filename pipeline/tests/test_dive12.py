"""dive12 (T24): the SlideOver's events moved by D_S (12, 18) and the re-timed dive. Checks: the SlideOver shift rules (keys
before F0 stay, keys from F0 move, clamp at the last frame, last value per frame, strictly increasing), the real SlideOver dump
(the cut at frame 72 moves to 84 and 90; the moved slots are 72, 75 and 78), the JSON format, the mapping override, the Exit
events identical to dive11, the roll margin formula, and that dive9, dive10 and dive11 keep their defaults. Output checks read
out\\dive12 and out\\dive11 and skip when they are not written yet; the bake itself is not run here (it takes minutes)."""
import json
import os

import pytest

import config
import dive8 as D8
import dive9 as D9
import dive10 as D10
import dive11 as D11
import dive12 as D12
import dumps

OUT = os.path.join(config.OUT, 'dive12')
OUT11 = os.path.join(config.OUT, 'dive11')
SO_HEX, EXIT_HEX = D9.SO_HEX, D9.EXIT_HEX
LAST_SO = 98                                    # the SlideOver clip: round(1.63333 * 60)
CUT = 4194305                                   # 0x400001: the cut event
CUT_OFF = 4194304                               # 0x400000


def _keys(pairs):
    return [{'Frame': f, 'Int': v} for f, v in pairs]


def _load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _need(path):
    if not os.path.exists(path):
        pytest.skip('%s is not written yet' % path)
    return _load(path)


def _so_dump():
    return dumps.load_by_name(D9.SO_NAME)


# ------------------------------------------------------------------------------------------------ shift rules
def test_keys_before_f0_stay_and_keys_from_f0_move_by_d():
    keys = _keys([(0, 0), (28, 8388612), (40, 8388635), (60, 8388636), (72, CUT)])
    out = D12.shift_slideover_keys(keys, 12, LAST_SO, from_frame=50)
    assert out == _keys([(0, 0), (28, 8388612), (40, 8388635), (72, 8388636), (84, CUT)])


def test_default_from_frame_is_50_and_the_cut_at_72_moves():
    assert D12.F0 == 50
    keys = _keys([(72, CUT)])
    assert D12.shift_slideover_keys(keys, 18, LAST_SO) == _keys([(90, CUT)])


def test_frames_past_the_last_frame_clamp_to_98_and_the_last_value_per_frame_is_kept():
    keys = _keys([(84, 4160749570), (92, 4026531842), (98, 2)])
    out = D12.shift_slideover_keys(keys, 12, LAST_SO)
    # 84 -> 96, 92 -> 104 -> 98 and 98 -> 110 -> 98: the later source key (98, 2) is the value kept at 98
    assert out == _keys([(96, 4160749570), (98, 2)])


def test_the_moved_keys_are_strictly_increasing_and_within_the_clip():
    keys = _keys([(0, 2), (60, 2), (72, 2), (78, 134217730), (84, 4160749570), (92, 4026531842), (98, 2)])
    for d in (12, 18):
        out = D12.shift_slideover_keys(keys, d, LAST_SO)
        frames = [k['Frame'] for k in out]
        assert frames == sorted(set(frames))
        assert max(frames) <= LAST_SO and min(frames) >= 0


def test_zero_shift_is_the_identity_on_a_sorted_track():
    keys = _keys([(0, 0), (28, 8388612), (60, 8388636), (72, CUT)])
    assert D12.shift_slideover_keys(keys, 0, LAST_SO) == keys


def test_unsorted_source_keys_are_rejected():
    with pytest.raises(ValueError):
        D12.shift_slideover_keys(_keys([(72, CUT), (60, 1)]), 12, LAST_SO)


# ------------------------------------------------------------------------------------------------ real SlideOver dump
def test_real_slideover_cut_moves_from_72_to_84_and_90():
    dump = _so_dump()
    tracks = D12.slideover_event_tracks(dump, 12)
    by_slot = {t['Slot']: t for t in tracks}
    assert by_slot[72]['Keyframes'][-1] == {'Frame': 84, 'Int': CUT}
    assert by_slot[78]['Keyframes'][-1] == {'Frame': 84, 'Int': CUT}
    tracks18 = {t['Slot']: t for t in D12.slideover_event_tracks(dump, 18)}
    assert tracks18[72]['Keyframes'][-1] == {'Frame': 90, 'Int': CUT}
    assert tracks18[78]['Keyframes'][-1] == {'Frame': 90, 'Int': CUT}


def test_only_slots_72_75_and_78_move_at_f0_50():
    rows = D12.slideover_event_table(_so_dump(), 12)
    moved = sorted(r['slot'] for r in rows if r['moved'])
    assert moved == [72, 75, 78]
    unmoved = [r for r in rows if not r['moved']]
    assert all(max(f for f, _ in r['before']) < 50 for r in unmoved)


def test_slot_75_clamps_and_dedupes_at_the_clip_end():
    rows = {r['slot']: r for r in D12.slideover_event_table(_so_dump(), 18)}
    after = rows[75]['after']
    assert after[-1] == (98, 2)                                  # 98 -> 116 -> 98 keeps the later value 2
    assert [f for f, _ in after] == sorted(set(f for f, _ in after))
    assert (98, 4026531842) not in after                         # the 92 key clamps to 98 and is replaced by the later 2


def test_slideover_event_tracks_keep_slot_trackid_codec_and_infotype():
    dump = _so_dump()
    tracks = D12.slideover_event_tracks(dump, 12)
    slots = [i for i, t in enumerate(dump['Tracks']) if t.get('infotype') == 5]
    assert [t['Slot'] for t in tracks] == slots
    for t in tracks:
        src = dump['Tracks'][t['Slot']]
        assert src['infotype'] == 5 and t['infotype'] == 5
        assert t['TrackID'] == str(int(src['TrackID'])) and t['codec'] == src['codec']


# ------------------------------------------------------------------------------------------------ the window and the roll
def test_roll_margin_formula_matches_the_time_map():
    W = 1.40109
    # feet down at k = 1: the plant 0.17 s and the roll 0.28 s after the takeoff pose at 0.92 (clip) = 1.37 s
    assert abs(D12.margin_of(W, 1.0, 1.0) - (W - 1.37)) < 1e-9
    assert abs(D12.margin_of(W, 1.5, 2.0) - (W - D8.Timing8(D12.T_TO, 1.5, 2.0).t_feet)) < 1e-12
    assert D12.margin_of(W, 1.0, 1.0) < D12.W_MARGIN             # S12 cannot be (1, 1)
    assert D12.margin_of(1.50109, 1.0, 1.0) >= D12.W_MARGIN      # S18 can


def test_k_search_grid_is_closest_to_one_and_respects_the_margin():
    import numpy as np
    W = 1.40109
    grid = np.round(np.arange(1.0, 1.5 + 1e-9, 0.01), 4)
    feas = [(kp, kr) for kp in grid for kr in grid if D12.margin_of(W, kp, kr) >= D12.W_MARGIN - 1e-9]
    best = min(feas, key=lambda p: ((p[0] - 1) ** 2 + (p[1] - 1) ** 2, p[1]))
    assert D12.margin_of(W, *best) >= D12.W_MARGIN - 1e-9
    assert best[0] <= 1.1 and best[1] <= 1.1                    # the S12 window forces both k above 1.0 only slightly


def test_upright_rule_uses_the_spec_limits():
    row = dict(margin_s=0.06, inv_W=False, up_W=True, tilt_W=40.0, tilt_last_max=50.0, inv_last=False, up_last=True)
    assert D12.upright_ok(row)
    assert not D12.upright_ok(dict(row, margin_s=0.04))
    assert not D12.upright_ok(dict(row, tilt_W=61.0))
    assert not D12.upright_ok(dict(row, inv_last=True))


def test_set_window_helper_is_restorable():
    class C:
        pass
    c = C()
    c.t_W, c.span_so, c.W_played, c.w_W, c.orig_W, c.root_W, c.disp_ex, c.P_W_rel_y = 1.0, 0.8, 0.9, 1.0, 0, 0, 0, 0
    saved = {k: getattr(c, k) for k in D12.W_ATTRS}
    restored = dict(saved)
    D12.restore_window(c, restored)
    assert c.t_W == 1.0 and c.span_so == 0.8


# ------------------------------------------------------------------------------------------------ outputs
def test_json_format_of_the_variant_patches():
    for name, d in (('S12', 12), ('S18', 18)):
        so = _need(os.path.join(OUT, name, SO_HEX + '.baked.json'))
        ex = _need(os.path.join(OUT, name, EXIT_HEX + '.baked.json'))
        for p in (so, ex):
            assert sorted(p.keys()) == ['Length', 'Tracks']
            for t in p['Tracks']:
                if t.get('infotype') == 5:
                    assert set(t.keys()) >= {'TrackID', 'codec', 'infotype', 'Slot', 'Keyframes'}
                for k in t.get('Keyframes', []):
                    assert 'Frame' in k
        so_ev = [t for t in so['Tracks'] if t.get('infotype') == 5]
        assert [t['Slot'] for t in so_ev] == [i for i, t in enumerate(_so_dump()['Tracks']) if t.get('infotype') == 5]
        assert [t for t in so_ev] == D12.slideover_event_tracks(_so_dump(), d)


def test_mapping_schema_and_override():
    for name, d in (('S12', 12), ('S18', 18)):
        m = _need(os.path.join(OUT, name, D12.MAPPING_NAME))
        assert len(m['clips']) == 2
        overrides = [c.get('override') for c in m['clips']]
        assert overrides == ['preview: dive12 slideover events +%d' % d] * 2


def test_exit_events_identical_to_dive11():
    dump = dumps.load_by_name(D9.EXIT_NAME)
    calc = D10.event_patch_tracks(dump, 15)
    for name in ('S12', 'S18'):
        ex = _need(os.path.join(OUT, name, EXIT_HEX + '.baked.json'))
        ev = [t for t in ex['Tracks'] if t.get('infotype') == 5]
        assert json.dumps(ev, sort_keys=True) == json.dumps(calc, sort_keys=True)
        ref = _need(os.path.join(OUT11, EXIT_HEX + '.baked.json'))
        ev11 = [t for t in ref['Tracks'] if t.get('infotype') == 5]
        assert json.dumps(ev, sort_keys=True) == json.dumps(ev11, sort_keys=True)
    assert len(calc) == 23


def test_summary_checks_when_written():
    s = _need(os.path.join(OUT, 'summary.json'))
    for name, d in (('S12', 12), ('S18', 18)):
        v = s['variants'][name]
        assert v['D_S'] == d
        assert v['margin_s'] >= 0.05 - 1e-9
        assert v['k_plant'] <= 1.5 and v['k_roll'] <= 1.5


def test_dive9_dive10_and_dive11_defaults_unchanged():
    assert D9.P_UP == 0.16 and D9.Z_UP == 0.80 and D9.RISE_V_MAX == 4.0
    assert D9.K_PLANT == 2.0 and D9.K_ROLL == 3.0 and D9.T_TO == 0.62
    assert D10.DELAYS == (8, 15) and D10.EVENT_FROM == 13
    assert D11.DELAY == 15 and D11.OVERRIDE == 'preview: dive11 delay15 floor fix'
    assert tuple(D8.PLANT_W) == (0.75, 0.92) and D8.FEET_W == 1.2


def test_dive12_writes_only_under_out_dive12():
    assert os.path.normpath(D12.OUT_DIR) == os.path.normpath(os.path.join(config.OUT, 'dive12'))
    assert os.path.normpath(D12.SUMMARY).startswith(os.path.normpath(D12.OUT_DIR))
    assert os.path.normpath(D12.REPORT).startswith(os.path.normpath(D12.OUT_DIR))
