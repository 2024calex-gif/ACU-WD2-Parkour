"""Tests for bake_chain.py (T12b): the joined time map, the Exit clock, the gated mapping and the written patches."""
import json
import os

import numpy as np
import pytest

import bake2
import bake_chain as BC
import bake_dive as BD
import config

MAP = os.path.join(BC.PATCH_DIR, 'mapping_dive.json')
SUMMARY = os.path.join(BC.OUT_DIR, 'chain_summary.json')


# ------------------------------------------------------------------------------------------------ time map
def test_wd2_map_hits_its_knots_and_has_slope_one_at_both_joins():
    p_to, T = 0.03, 0.55
    assert BC.wd2_of_J(p_to, p_to, T) == pytest.approx(BD.WD2_TAKEOFF, abs=1e-12)
    assert BC.wd2_of_J(T, p_to, T) == pytest.approx(BD.WD2_PLANT[0], abs=1e-12)
    eps = 1e-6
    for J in (p_to, T):
        d = (BC.wd2_of_J(J + eps, p_to, T) - BC.wd2_of_J(J - eps, p_to, T)) / (2 * eps)
        assert d == pytest.approx(1.0, abs=1e-3), J


def test_wd2_map_is_monotone_and_k_flight_stretches_the_flight():
    p_to, T = 0.03, 0.55
    J = np.linspace(-0.5, 1.5, 4001)
    w = BC.wd2_of_J(J, p_to, T)
    assert np.all(np.diff(w) > 0)
    # the flight between takeoff and plant is stretched: more played seconds than WD2 seconds (0.30 s of WD2)
    assert (T - p_to) / (BD.WD2_PLANT[0] - BD.WD2_TAKEOFF) > 1.0
    after = BC.wd2_of_J(np.array([T + 0.3]), p_to, T)[0]
    assert after == pytest.approx(BD.WD2_PLANT[0] + 0.3, abs=1e-12)     # k = 1 after the plant


def test_exit_clock_and_joined_window():
    tl = BC.Timeline(dict(o=0.423, r=1.046, W=0.744), p_to=0.0255, W=0.744, T=0.5)
    assert tl.J_exit(BC.EXIT_O) == pytest.approx(0.744, abs=1e-12)
    assert tl.J_exit(BC.EXIT_O + BC.EXIT_R * BC.EXIT_WIN) == pytest.approx(0.744 + BC.EXIT_WIN, abs=1e-9)
    assert tl.J_end == pytest.approx(0.744 + BC.EXIT_WIN)


def test_rotation_identity_at_the_patch_start():
    """Additive rotations of the SlideOver equal the clip's own local rotations at the patch start."""
    import acuclip, contacts, dumps, retarget as R, wd2
    rig = R.Rig()
    m = wd2.load(BD.load_dive_events()['stem'])
    dump = dumps.load_by_name(BC.MAIN_SO)
    seq = acuclip.pose_sequence(dump, step=1)
    ev = contacts.events(seq, dump)
    mp = BD.clip_model(BC.MAIN_SO, BD.load_play_model_so(), float(dump['Length']))
    tl = BC.Timeline(mp, p_to=(float(ev['takeoff']['seconds']) - mp['o']) / mp['r'], W=mp['W'], T=0.5)
    patset, rot = BD.patch_set(rig, dump)
    f0 = int(round(mp['o'] * 60))
    out = BC.chain_rotations(rig, m, tl.w_so, tl.w_so(f0 / 60.0), [f0], patset, rot)
    for ai in patset:
        assert bake2.angle_deg(out[ai][0], bake2.orig_local(rig, rot, ai, f0)) < 1e-3, rig.ac[ai]['name']


# ------------------------------------------------------------------------------------------------ outputs
def _entries():
    if not os.path.exists(MAP):
        pytest.skip('bake_chain has not written its mapping yet')
    with open(MAP, encoding='utf-8') as f:
        return json.load(f)['clips']


def test_mapping_has_the_scoped_slideovers_and_one_gated_exit():
    clips = _entries()
    names = [c['name'] for c in clips]
    assert BC.MAIN_SO in names and BC.EXIT_NAME in names
    # a scoped SlideOver without a feasible plant has no patch file and therefore no entry (reported in the report)
    assert set(names) <= set(BC.SO_SCOPE) | {BC.EXIT_NAME}
    assert len(clips) == len([n for n in names if n != BC.EXIT_NAME]) + 1
    exits = [c for c in clips if c['name'] == BC.EXIT_NAME]
    assert len(exits) == 1
    ex = exits[0]
    assert ex['handle'] == '0x' + BC.EXIT_HEX and ex['source'] == BC.EXIT_HEX + '.baked.json'
    assert ex['arm_seconds'] == 2.0
    assert ex['prebaked'] is True and ex['patch_translation'] is True
    so = [c for c in clips if c['name'] != BC.EXIT_NAME]
    for c in so:
        assert c['name'] in BC.SO_SCOPE and c['prebaked'] is True and c['patch_translation'] is True
        assert isinstance(c['enabled'], bool)
        assert 'only_after' not in c


def test_exit_gate_lists_exactly_the_enabled_slideovers():
    clips = _entries()
    ex = next(c for c in clips if c['name'] == BC.EXIT_NAME)
    enabled = sorted(c['name'] for c in clips if c['name'] != BC.EXIT_NAME and c['enabled'])
    assert sorted(ex['only_after']) == enabled
    if ex['enabled']:
        assert enabled, 'Exit enabled with no SlideOver enabled'


def test_written_patch_format_for_every_entry():
    for c in _entries():
        path = os.path.join(BC.PATCH_DIR, c['source'])
        assert os.path.exists(path), path
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        assert set(d.keys()) == {'Length', 'Tracks'}
        codecs = {(t['infotype'], t['codec']) for t in d['Tracks']}
        assert (0, 1) in codecs and (3, 27) in codecs
        trans = [t for t in d['Tracks'] if t['infotype'] == 3]
        assert len(trans) == 1 and trans[0]['TrackID'] == str(bake2.REF_TID)
        for t in d['Tracks']:
            frames = [k['Frame'] for k in t['Keyframes']]
            assert frames == sorted(frames) and len(set(frames)) == len(frames)
            if t['infotype'] == 0:
                for k in t['Keyframes']:
                    assert abs(np.linalg.norm(k['LocalRotation']) - 1.0) < 5e-4


def test_no_enabled_entry_fails_its_recorded_validation():
    if not os.path.exists(SUMMARY):
        pytest.skip('no chain_summary.json yet')
    with open(SUMMARY, encoding='utf-8') as f:
        s = json.load(f)
    for r in s['slideovers']:
        if r['enabled']:
            assert not r['fails'], r['name']
            assert r['continuity_deg'] <= BC.CONT_DEG
    if s['exit']['enabled']:
        assert not s['exit']['fails']
        assert any(r['enabled'] for r in s['slideovers'])
