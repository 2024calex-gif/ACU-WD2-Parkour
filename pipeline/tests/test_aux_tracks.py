"""aux_tracks: the inverse of Reference (checked against the original vault dumps), the foot-plant step semantics,
no duplicate (TrackID, infotype) pairs, the JSON format, and the dive7 patches written from dive6."""
import collections
import json
import math
import os

import numpy as np
import pytest

import acuclip
import aux_tracks as AT
import config
import dumps
from retarget import qconj, qn, qrot

SO_NAME = 'xx_Parkour_Sprint_SlideOver_Fd100cm-Up100cm'
EXIT_NAME = 'xx_Parkour_Muraille_Exit_Fd100cm-Dn100cm'
DIVE6 = os.path.join(config.OUT, 'dive6', 'patches')
DIVE7 = os.path.join(config.OUT, 'dive7', 'patches')
NEW_PAIRS = {(4294967269, 3), (4294967270, 3), (2246150967, 0), (2246150967, 3), (3215835963, 0), (3215835963, 3)}
MAPPING_OVERRIDE = 'preview: roll on ground + foot IK off'


def _vault_names():
    return [name for _, name, _ in dumps.list_dumps() if acuclip.is_vault(name)]


def _ref_patch(dump):
    """A patch made only of the original dump's Reference tracks (what the inverse reads)."""
    return {'Length': dump['Length'],
            'Tracks': [t for t in dump['Tracks'] if int(t['TrackID']) == AT.REF_ID]}


def _track(tracks, tid, infotype):
    return next(t for t in tracks if int(t['TrackID']) == tid and t['infotype'] == infotype)


def _sentinel():
    return np.array(AT.SENTINEL, float)


# ---- inverse of Reference -------------------------------------------------------------------------

@pytest.mark.parametrize('name', _vault_names())
def test_inverse_reproduces_original_inverse_tracks(name):
    dump = dumps.load_by_name(name)
    err = AT.self_check(dump)
    assert err, 'no inverse track in %s' % name
    for tid, e in err.items():
        # dump values are stored with ~6 decimals; the inverse is exact per frame, so the errors stay small
        assert e['rot_at_keys'] < 1e-3, (name, tid, e)
        assert e['pos_at_keys'] < 1e-3, (name, tid, e)
        assert e['rot_all'] < 1e-3, (name, tid, e)       # slerp commutes with conjugation
        assert e['pos_all'] < 0.05, (name, tid, e)       # the original interpolates the inverse's position linearly


def test_inverse_position_is_minus_inverse_rotation_applied_to_p_not_minus_p():
    dump = dumps.load_by_name(SO_NAME)
    rot, pos = AT.reference_inverse(_ref_patch(dump))
    rk = dumps.track(dump, AT.REF_ID, 0)
    pk = dumps.track(dump, AT.REF_ID, 3)
    err_minus_p = max(np.abs(pos[f] + dumps.sample_position(pk, f)).max() for f in range(len(pos)))
    assert err_minus_p > 0.05        # -P is clearly not the inverse position here
    for f in range(len(pos)):        # position = -R^-1 P exactly
        R = qn(dumps.sample_rotation(rk, f))
        assert np.allclose(pos[f], -qrot(qconj(R), dumps.sample_position(pk, f)), atol=1e-12)
        assert np.allclose(rot[f], qconj(R), atol=1e-12)


def test_inverse_round_trip_is_identity():
    """Applying the emitted inverse after the Reference transform returns the point unchanged."""
    dump = dumps.load_by_name(SO_NAME)
    patch = _ref_patch(dump)
    out = AT.add_aux_tracks(patch, dump, foot_free_from_frame=27)
    rot_keys = _track(out['Tracks'], 2246150967, 0)['Keyframes']
    pos_keys = _track(out['Tracks'], 2246150967, 3)['Keyframes']
    pk = dumps.track(dump, AT.REF_ID, 3)
    rk = dumps.track(dump, AT.REF_ID, 0)
    rng = np.random.default_rng(0)
    for f in range(0, int(round(dump['Length'] * 60)) + 1, 7):
        R = qn(dumps.sample_rotation(rk, f))
        P = dumps.sample_position(pk, f)
        Ri = np.array(rot_keys[f]['LocalRotation'], float)
        Pi = np.array(pos_keys[f]['LocalPosition'], float)
        for x in rng.normal(size=(3, 3)):
            y = qrot(R, x) + P                 # forward: rotate, then translate
            back = qrot(Ri, y) + Pi            # inverse
            assert np.allclose(back, x, atol=1e-9)


def test_inverse_one_key_per_frame_and_sign_continuous():
    dump = dumps.load_by_name(SO_NAME)
    out = AT.add_aux_tracks(_ref_patch(dump), dump, foot_free_from_frame=27)
    last = int(round(dump['Length'] * 60))
    for tid in (2246150967, 3215835963):
        rot = _track(out['Tracks'], tid, 0)['Keyframes']
        pos = _track(out['Tracks'], tid, 3)['Keyframes']
        assert [k['Frame'] for k in rot] == list(range(last + 1))
        assert [k['Frame'] for k in pos] == list(range(last + 1))
        q = np.array([k['LocalRotation'] for k in rot])
        assert np.all(np.sum(q[1:] * q[:-1], axis=1) > 0.0)


def test_inverse_needs_the_reference_tracks():
    dump = dumps.load_by_name(SO_NAME)
    with pytest.raises(ValueError):
        AT.add_aux_tracks({'Length': dump['Length'], 'Tracks': []}, dump, foot_free_from_frame=0)


# ---- foot-plant targets ---------------------------------------------------------------------------

def _keys(*pairs):
    return [{'Frame': f, 'LocalPosition': list(p)} for f, p in pairs]


def test_sample_foot_holds_the_sentinel_and_never_interpolates_into_it():
    keys = _keys((0, (10, 10, 10)), (10, (0.1, 0.2, 0.0)), (20, (0.3, 0.4, 0.0)))
    for f in range(0, 10):
        assert np.allclose(AT.sample_foot(keys, f), _sentinel())       # sentinel holds until the real key
    assert np.allclose(AT.sample_foot(keys, 10), (0.1, 0.2, 0.0))
    assert np.allclose(AT.sample_foot(keys, 15), (0.2, 0.3, 0.0))       # real-to-real stays linear
    assert np.allclose(AT.sample_foot(keys, 25), (0.3, 0.4, 0.0))       # clamped after the last key


def test_sample_foot_holds_the_real_key_before_a_sentinel():
    keys = _keys((0, (0.5, 0.0, 0.0)), (10, (10, 10, 10)))
    for f in range(0, 10):
        assert np.allclose(AT.sample_foot(keys, f), (0.5, 0.0, 0.0))    # no ramp towards the sentinel
    assert np.allclose(AT.sample_foot(keys, 10), _sentinel())


def test_step_keys_reproduce_every_frame_under_linear_interpolation():
    a, b, c = (0.0, 0.0, 0.0), (1.0, 2.0, 0.0), (5.0, 5.0, 5.0)
    values = [np.array(v, float) for v in (a, a, a, b, b, c)]
    keys = AT.step_keys(values)
    assert [k['Frame'] for k in keys] == [0, 2, 3, 4, 5]    # old-value key before each change, plus the ends
    for f, v in enumerate(values):
        assert np.allclose(dumps.sample_position(keys, f), v, atol=1e-12)


def test_foot_targets_follow_the_original_then_the_window_then_the_end_value():
    dump = dumps.load_by_name(SO_NAME)
    last = int(round(dump['Length'] * 60))
    out = AT.add_aux_tracks(_ref_patch(dump), dump, foot_free_from_frame=27)
    for tid in AT.FOOT_IDS:
        keys = dumps.track(dump, tid, 3)
        got = _track(out['Tracks'], tid, 3)
        assert got['codec'] == 27 and got['infotype'] == 3
        for f in range(last + 1):
            want = _sentinel() if f >= 27 else AT.sample_foot(keys, f)
            assert np.allclose(dumps.sample_position(got['Keyframes'], f), want, atol=1e-12), (tid, f)
    # the right foot is the sentinel until its first real key (frame 10), then planted until the takeoff (27)
    right = _track(out['Tracks'], AT.RIGHT_FOOT_ID, 3)['Keyframes']
    assert np.allclose(dumps.sample_position(right, 9), _sentinel())
    assert not np.allclose(dumps.sample_position(right, 10), _sentinel())
    assert np.allclose(dumps.sample_position(right, 27), _sentinel())


def test_foot_window_with_an_end_restores_the_original_values():
    dump = dumps.load_by_name(SO_NAME)
    last = int(round(dump['Length'] * 60))
    out = AT.add_aux_tracks(_ref_patch(dump), dump, foot_free_from_frame=10, foot_free_to_frame=20)
    keys = dumps.track(dump, AT.LEFT_FOOT_ID, 3)
    got = _track(out['Tracks'], AT.LEFT_FOOT_ID, 3)['Keyframes']
    for f in range(last + 1):
        want = _sentinel() if 10 <= f < 20 else AT.sample_foot(keys, f)
        assert np.allclose(dumps.sample_position(got, f), want, atol=1e-12), f


def test_exit_feet_are_never_planted():
    dump = dumps.load_by_name(EXIT_NAME)
    out = AT.add_aux_tracks(_ref_patch(dump), dump, foot_free_from_frame=0)
    for tid in AT.FOOT_IDS:
        keys = _track(out['Tracks'], tid, 3)['Keyframes']
        assert len(keys) == 2 and keys[0]['Frame'] == 0
        assert keys[-1]['Frame'] == int(round(dump['Length'] * 60))
        assert all(np.allclose(k['LocalPosition'], AT.SENTINEL) for k in keys)


def test_bad_window_is_rejected():
    dump = dumps.load_by_name(SO_NAME)
    with pytest.raises(ValueError):
        AT.add_aux_tracks(_ref_patch(dump), dump, foot_free_from_frame=-1)
    with pytest.raises(ValueError):
        AT.add_aux_tracks(_ref_patch(dump), dump, foot_free_from_frame=30, foot_free_to_frame=20)


# ---- duplicates and format -------------------------------------------------------------------------

def test_no_duplicate_track_pairs_and_no_second_add():
    dump = dumps.load_by_name(SO_NAME)
    patch = json.load(open(os.path.join(DIVE6, '47BFB02DF.baked.json'), encoding='utf-8'))
    out = AT.add_aux_tracks(patch, dump, foot_free_from_frame=27)
    counts = collections.Counter((int(t['TrackID']), t['infotype']) for t in out['Tracks'])
    assert all(v == 1 for v in counts.values())
    assert len(out['Tracks']) == len(patch['Tracks']) + 6
    with pytest.raises(ValueError):
        AT.add_aux_tracks(out, dump, foot_free_from_frame=27)     # the pairs are already there


def test_json_format_and_existing_tracks_unchanged():
    dump = dumps.load_by_name(SO_NAME)
    patch = _ref_patch(dump)
    out = AT.add_aux_tracks(patch, dump, foot_free_from_frame=27)
    assert json.loads(json.dumps(out)) == out
    assert out['Length'] == patch['Length']
    assert out['Tracks'][:len(patch['Tracks'])] == patch['Tracks']
    last = int(round(float(patch['Length']) * 60))
    for t in out['Tracks'][len(patch['Tracks']):]:
        assert set(t) == {'TrackID', 'codec', 'infotype', 'Keyframes'}
        assert isinstance(t['TrackID'], str) and t['TrackID'].isdigit()
        assert t['codec'] == (1 if t['infotype'] == 0 else 27) and t['infotype'] in (0, 3)
        frames = [k['Frame'] for k in t['Keyframes']]
        assert all(isinstance(f, int) for f in frames)
        assert frames == sorted(set(frames)) and frames[0] == 0 and frames[-1] == last
        field = 'LocalRotation' if t['infotype'] == 0 else 'LocalPosition'
        width = 4 if t['infotype'] == 0 else 3
        for k in t['Keyframes']:
            assert set(k) == {'Frame', field}
            assert len(k[field]) == width
            assert all(math.isfinite(x) for x in k[field])


def test_dive7_patches_are_dive6_plus_aux_tracks():
    """The dive7 files on disk: the dive6 tracks unchanged, the six aux tracks added, and rebuilt from the dumps."""
    cases = [('47BFB02DF', SO_NAME, 27), ('F7919BDF0', EXIT_NAME, 0)]
    for stem, name, free in cases:
        old = json.load(open(os.path.join(DIVE6, stem + '.baked.json'), encoding='utf-8'))
        new = json.load(open(os.path.join(DIVE7, stem + '.baked.json'), encoding='utf-8'))
        assert new['Length'] == old['Length']
        assert new['Tracks'][:len(old['Tracks'])] == old['Tracks']
        added = {(int(t['TrackID']), t['infotype']) for t in new['Tracks'][len(old['Tracks']):]}
        assert added == NEW_PAIRS
        counts = collections.Counter((int(t['TrackID']), t['infotype']) for t in new['Tracks'])
        assert all(v == 1 for v in counts.values())
        rebuilt = json.loads(json.dumps(AT.add_aux_tracks(old, dumps.load_by_name(name), free)))
        assert rebuilt == new


def test_dive7_mapping_is_dive6_with_the_new_override():
    old = json.load(open(os.path.join(DIVE6, 'mapping_dive6.json'), encoding='utf-8'))
    new = json.load(open(os.path.join(DIVE7, 'mapping_dive7.json'), encoding='utf-8'))
    assert [c['name'] for c in new['clips']] == [c['name'] for c in old['clips']]
    assert [c['source'] for c in new['clips']] == [c['source'] for c in old['clips']]
    assert all(c['override'] == MAPPING_OVERRIDE for c in new['clips'])
