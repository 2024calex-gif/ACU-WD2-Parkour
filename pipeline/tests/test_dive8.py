"""dive8 (T20): the B_fastroll timing, the Exit contact rule, the aux tracks of both variants, the C no-op Exit (equal to
the original), and the mapping entries."""
import json
import os
import zlib

import numpy as np
import pytest

import aux_tracks as AT
import config
import dive8 as D8
import dumps

OUT = os.path.join(config.OUT, 'dive8')
ORIG_EXIT = os.path.join(config.V2, 'data', 'acu_dumps', 'F7919BDF0_xx_Parkour_Muraille_Exit_Fd100cm-Dn100cm.json')


def test_timing8_maps_the_plant_and_the_roll_and_meets_the_feet_down():
    tm = D8.Timing8(0.74, 1.5, 2.5)
    assert float(tm.w(0.74)) == pytest.approx(0.45)
    assert float(tm.w(tm.t_pl0)) == pytest.approx(0.75)
    assert float(tm.w(tm.t_pe)) == pytest.approx(0.92)
    assert float(tm.w(tm.t_feet)) == pytest.approx(1.2)
    assert tm.t_pe == pytest.approx(0.74 + 0.30 + 0.17 / 1.5)
    assert tm.k == pytest.approx(2.5)


def test_timing8_is_continuous_and_monotone():
    tm = D8.Timing8(0.74, 1.5, 2.5)
    t = np.linspace(0.3, 1.6, 2000)
    w = tm.w(t)
    assert np.all(np.diff(w) >= -1e-12)
    assert np.max(np.abs(np.diff(w))) < 0.05


def test_roll_speed_moves_the_feet_down_before_the_slow_roll():
    slow = D8.Timing8(0.74, 1.0, 1.0)
    fast = D8.Timing8(0.74, 1.5, 2.5)
    assert fast.t_feet < slow.t_feet


def test_exit_lowest_is_the_body_when_the_body_would_be_underground():
    lb = np.array([-0.2, 0.05, 0.3])
    lf = np.array([0.0, 0.0, 0.4])
    out = D8.exit_lowest(lb, lf)
    assert out[0] == pytest.approx(-0.2)          # body below the floor: the body is the contact point
    assert out[1] == pytest.approx(0.0)           # body above the floor: the feet are the contact point
    assert out[2] == pytest.approx(0.4)


def test_noop_exit_patch_samples_every_frame_and_keeps_the_reference_track():
    dump = json.load(open(ORIG_EXIT, encoding='utf-8'))
    ref_patch = {'Tracks': [{'TrackID': str(D8.REF_ID), 'codec': 1, 'infotype': 0, 'Keyframes': []},
                            {'TrackID': '1234', 'codec': 1, 'infotype': 0, 'Keyframes': []}]}
    patch, skipped = D8.noop_exit_patch(dump, ref_patch)
    last = int(round(dump['Length'] * 60))
    rot = [t for t in patch['Tracks'] if t['infotype'] == 0]
    assert all(len(t['Keyframes']) == last + 1 for t in rot)
    assert all(t['Keyframes'][f]['Frame'] == f for t in rot for f in (0, 1, last))
    pos = [t for t in patch['Tracks'] if t['infotype'] == 3]
    assert len(pos) == 1 and pos[0]['codec'] == 27 and len(pos[0]['Keyframes']) == last + 1
    assert 1234 in skipped                        # a track the dump does not have is reported, not invented


def test_noop_exit_equals_the_original_at_every_frame():
    if not os.path.exists(ORIG_EXIT):
        pytest.skip('original dump not present')
    dump = json.load(open(ORIG_EXIT, encoding='utf-8'))
    ref_patch = {'Tracks': [dict(TrackID=str(t['TrackID']), codec=1, infotype=0, Keyframes=[]) for t in dump['Tracks']
                            if t['infotype'] == 0][:20]}
    patch, _sk = D8.noop_exit_patch(dump, ref_patch)
    s = D8.sanity_noop(dump, patch)
    assert s['rot_frame_deg'] < 1e-4 and s['pos_frame_m'] < 1e-9


def _load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def test_b_patches_carry_the_aux_tracks_with_the_brief_windows():
    so_path = os.path.join(OUT, 'B_fastroll', D8.SO_HEX + '.baked.json')
    ex_path = os.path.join(OUT, 'B_fastroll', D8.EXIT_HEX + '.baked.json')
    if not (os.path.exists(so_path) and os.path.exists(ex_path)):
        pytest.skip('dive8 B outputs not written yet')
    so, ex = _load(so_path), _load(ex_path)
    for p in (so, ex):
        ids = {(int(t['TrackID']), t['infotype']) for t in p['Tracks']}
        for tid in AT.FOOT_IDS:
            assert (tid, 3) in ids
        for tid in AT.INV_IDS:
            assert (tid, 0) in ids and (tid, 3) in ids
    foot = [t for t in so['Tracks'] if int(t['TrackID']) == AT.LEFT_FOOT_ID][0]['Keyframes']
    frames = {k['Frame']: k['LocalPosition'] for k in foot}
    sample_in_window = AT.sample_foot(foot, D8.TAKEOFF_FRAME + 5)
    assert np.allclose(sample_in_window, AT.SENTINEL)          # feet free after the takeoff
    exit_foot = [t for t in ex['Tracks'] if int(t['TrackID']) == AT.LEFT_FOOT_ID][0]['Keyframes']
    assert np.allclose(AT.sample_foot(exit_foot, 10), AT.SENTINEL)   # feet free for the whole Exit


def test_c_exit_has_no_aux_and_the_slideover_is_dive7s():
    c_so = os.path.join(OUT, 'C_noexit', D8.SO_HEX + '.baked.json')
    c_ex = os.path.join(OUT, 'C_noexit', D8.EXIT_HEX + '.baked.json')
    if not (os.path.exists(c_so) and os.path.exists(c_ex)):
        pytest.skip('dive8 C outputs not written yet')
    ex = _load(c_ex)
    ids = {(int(t['TrackID']), t['infotype']) for t in ex['Tracks']}
    for tid in AT.FOOT_IDS:
        assert (tid, 3) not in ids                # no foot-plant aux track in the no-op Exit
    for tid in AT.INV_IDS:
        assert (tid, 3) not in ids                # no inverse-Reference aux track either
    d7 = os.path.join(config.OUT, 'dive7', 'patches', D8.SO_HEX + '.baked.json')
    with open(c_so, 'rb') as a, open(d7, 'rb') as b:
        assert a.read() == b.read()


def test_mappings_enable_both_with_the_override_and_the_gate():
    for name, label in (('B_fastroll', 'preview: dive8 B fastroll'), ('C_noexit', 'preview: dive8 C noexit')):
        path = os.path.join(OUT, name, 'mapping_dive8_%s.json' % name[0])
        if not os.path.exists(path):
            pytest.skip('dive8 mappings not written yet')
        m = _load(path)
        by = {c['handle'].replace('0x', ''): c for c in m['clips']}
        so, ex = by[D8.SO_HEX], by[D8.EXIT_HEX]
        for c in (so, ex):
            assert c['enabled'] is True and c['override'] == label
        assert ex['only_after'] == [D8.SO_NAME] and ex['arm_seconds'] == 2.0
