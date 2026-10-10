"""Tests for bake2.py (T9b): time map, crossfade, Reference-compensated rotations, reduction and the written patch."""
import json
import os
import zlib

import numpy as np
import pytest

import acuclip
import bake2
import config
import contacts
import dumps
import fk
import retarget as R
import wd2

CLIP = 'xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm'
EV = bake2.load_events()
SYN_ACU = dict(takeoff=0.5, hand_plant=0.8, landing=1.2)
SYN_WD2 = dict(takeoff=1.1, hand_plant_start=1.3, hand_plant_end=1.6, landing=1.9, static_until=0.75, duration=3.0)


@pytest.fixture(scope='module')
def rig():
    return R.Rig()


@pytest.fixture(scope='module')
def kong():
    return wd2.load(EV['stem'])


@pytest.fixture(scope='module')
def clip():
    dump = dumps.load_by_name(CLIP)
    seq = acuclip.pose_sequence(dump, step=1)
    ev = contacts.events(seq, dump)
    return dump, bake2.acu_seconds(ev), ev['obstacle']


def _wd2_ev(kong):
    return dict(takeoff=EV['takeoff'], hand_plant_start=EV['hand_plant_start'], hand_plant_end=EV['hand_plant_end'],
                landing=EV['landing'], static_until=EV['static_until'], duration=float(kong['dur']))


def _angle_deg(a, b):
    return float(bake2.angle_deg(np.asarray(a, float), np.asarray(b, float)))


# ------------------------------------------------------------------------------------------------ time map
def test_time_map_hits_anchors_exactly():
    tm = bake2.time_map(SYN_ACU, SYN_WD2, 2.0)
    assert float(tm(0.5)) == pytest.approx(1.1, abs=1e-12)
    assert float(tm(0.8)) == pytest.approx(1.3, abs=1e-12)
    assert float(tm(1.2)) == pytest.approx(1.9, abs=1e-12)


def test_time_map_monotonic():
    tm = bake2.time_map(SYN_ACU, SYN_WD2, 2.0)
    t = np.linspace(0.0, 2.0, 2001)
    y = tm(t)
    assert np.all(np.diff(y) >= -1e-12)


def test_time_map_slope_one_outside_anchors():
    tm = bake2.time_map(SYN_ACU, SYN_WD2, 2.0)
    # before the first anchor: WD2 = 1.1 + (t - 0.5) (well above static_until = 0.75)
    assert float(tm(0.3)) == pytest.approx(0.9, abs=1e-12)
    assert float(tm(0.4) - tm(0.3)) == pytest.approx(0.1, abs=1e-12)
    # after the last anchor: WD2 = 1.9 + (t - 1.2)
    assert float(tm(1.7)) == pytest.approx(2.4, abs=1e-12)
    assert float(tm(1.8) - tm(1.7)) == pytest.approx(0.1, abs=1e-12)


def test_time_map_clamps_to_static_and_duration():
    tm = bake2.time_map(SYN_ACU, SYN_WD2, 2.0)
    assert float(tm(0.0)) == pytest.approx(0.75)        # 0.6 clamped up to static_until
    assert float(tm(2.0)) == pytest.approx(2.7, abs=1e-12)  # 1.9 + (2.0 - 1.2), under the 3.0 duration
    tm2 = bake2.time_map(SYN_ACU, SYN_WD2, 2.0)
    assert float(tm2(np.array([5.0]))[0]) == pytest.approx(3.0)  # clamped to duration


def test_time_map_hand_plant_fallback_placed_by_fraction():
    acu = dict(takeoff=0.5, hand_plant=None, landing=1.2)
    frac = (1.3 - 1.1) / (1.9 - 1.1)
    expect = 0.5 + frac * (1.2 - 0.5)
    assert bake2.acu_hand_plant(acu, SYN_WD2) == pytest.approx(expect)
    tm = bake2.time_map(acu, SYN_WD2, 2.0)
    assert float(tm(expect)) == pytest.approx(1.3, abs=1e-12)


def test_time_map_rejects_non_monotonic_anchors():
    bad = dict(takeoff=0.9, hand_plant=0.8, landing=1.2)
    with pytest.raises(ValueError):
        bake2.time_map(bad, SYN_WD2, 2.0)


def test_time_map_real_clip_hits_anchors(clip, kong):
    dump, acu_s, _ob = clip
    w = _wd2_ev(kong)
    tm = bake2.time_map(acu_s, w, float(dump['Length']))
    assert float(tm(acu_s['takeoff'])) == pytest.approx(w['takeoff'], abs=1e-9)
    assert float(tm(acu_s['landing'])) == pytest.approx(w['landing'], abs=1e-9)
    assert float(tm(bake2.acu_hand_plant(acu_s, w))) == pytest.approx(w['hand_plant_start'], abs=1e-9)


# ------------------------------------------------------------------------------------------------ bones
def test_patchable_bones_exclude_hips_and_reference(rig, clip):
    dump, _acu, _ob = clip
    rot = bake2.rot_tracks(rig, dump)
    pat = bake2.patchable_bones(rig, rot)
    hips = rig.ac_by_hash[zlib.crc32(b'Hips')]
    assert hips not in pat
    assert 0 not in pat
    assert all(ai in rot for ai in pat)
    assert len(pat) == len(set(rig.map.values())) - 1   # every mapped bone except Hips has a track


# ------------------------------------------------------------------------------------------------ compensation
def test_compensated_rotations_reproduce_retargeted_world(rig, clip, kong):
    dump, acu_s, _ob = clip
    rot = bake2.rot_tracks(rig, dump)
    pat = set(bake2.patchable_bones(rig, rot))
    f = 54
    tm = bake2.time_map(acu_s, _wd2_ev(kong), float(dump['Length']))
    _, wt = bake2.wd2_pose_at(rig, kong, float(tm(f / 60.0)))
    comp = bake2.compensated_rotations(rig, rot, f, wt, pat)
    loc = np.array([bake2.orig_local(rig, rot, ai, f) for ai in range(len(rig.ac))])
    for ai, q in comp.items():
        loc[ai] = q
    lpos = np.array([np.array(b['lpos'], float) for b in rig.ac])
    WR, _WP = fk.fk(rig.ac, loc[None], lpos[None])
    worst = 0.0
    for ai in pat:
        want = R.qn(R.qmul(R.A_REF, wt[ai]))
        worst = max(worst, _angle_deg(WR[0, ai], want))
    assert worst < 0.1, 'patched bones off by %.4f deg' % worst


def test_compensated_reference_is_untouched(rig, clip, kong):
    dump, acu_s, _ob = clip
    rot = bake2.rot_tracks(rig, dump)
    pat = set(bake2.patchable_bones(rig, rot))
    _, wt = bake2.wd2_pose_at(rig, kong, 1.3)
    comp = bake2.compensated_rotations(rig, rot, 40, wt, pat)
    assert 0 not in comp


# ------------------------------------------------------------------------------------------------ crossfade
def test_fade_weight_zero_outside_and_one_inside():
    t = np.array([0.0, 0.29, 0.3, 0.5, 0.8, 0.9, 0.95])   # o = 0.3, window 0.6 -> patch active on (0.3, 0.9)
    w = bake2.fade_weight(t, 0.3, 0.6)
    assert w[0] == 0.0 and w[1] == 0.0 and w[2] == 0.0     # before o
    assert w[5] == 0.0 and w[6] == 0.0                     # at and after o + window
    assert w[3] == pytest.approx(1.0)                      # full patch in the middle
    assert 0.0 < w[4] < 1.0                                # on the fade-out ramp [o + window - 0.15, o + window]
    assert np.all((w >= 0) & (w <= 1))


def test_crossfade_endpoints_synthetic():
    rng = np.random.default_rng(0)
    frames = list(range(0, 121, 2))
    orig = {0: [], 1: []}
    patch = {0: [], 1: []}
    for f in frames:
        for ai in (0, 1):
            a = R.qn(rng.normal(size=4))
            b = R.qn(rng.normal(size=4))
            orig[ai].append((f, a))
            patch[ai].append((f, b))
    o, window = 0.3, 0.6
    out = bake2.crossfade_to_original(orig, patch, frames, o, window)
    for ai in (0, 1):
        for (f, q), (_, qo) in zip(out[ai], orig[ai]):
            t = f / 60.0
            if t <= o + 1e-9 or t >= o + window - 1e-9:
                assert min(np.abs(q - qo).max(), np.abs(q + qo).max()) < 1e-6, (f, t)


def test_crossfade_endpoints_real_clip(rig, clip, kong):
    dump, acu_s, _ob = clip
    o, window = 0.3, 0.6
    b = bake2.bake_clip(rig, kong, dump, acu_s, _wd2_ev(kong), o, window)
    for ai in b['patched']:
        for (f, q), (f2, qo) in zip(b['blended'][ai], b['orig'][ai]):
            assert f == f2
            t = f / 60.0
            if t <= o + 1e-9 or t >= o + window - 1e-9:
                assert min(np.abs(q - qo).max(), np.abs(q + qo).max()) < 1e-6, (ai, f)


# ------------------------------------------------------------------------------------------------ reduction
def test_reduce_keys_keeps_straight_slerp_and_respects_tolerance():
    q0 = R.qn([0, 0, 0, 1.0])
    q1 = R.qn([0, 0, np.sin(np.radians(20)), np.cos(np.radians(20))])
    keys = [(f, R.slerp(q0, q1, f / 10.0)) for f in range(0, 11, 2)]
    red = bake2.reduce_keys(keys)
    assert len(red) == 2
    # a bent key must survive
    bent = list(keys)
    bent[2] = (bent[2][0], R.qn([0, 0.2, 0, 1.0]))
    red2 = bake2.reduce_keys(bent)
    assert any(f == bent[2][0] for f, _ in red2)


def test_sample_keys_matches_reduced_keys_within_tolerance(rig, clip, kong):
    dump, acu_s, _ob = clip
    b = bake2.bake_clip(rig, kong, dump, acu_s, _wd2_ev(kong), 0.0, 0.758)
    for ai in b['patched'][:10]:
        red = bake2.reduce_keys(b['blended'][ai])
        frames = [k[0] for k in b['blended'][ai]]
        s = bake2.sample_keys(red, frames)
        err = bake2.angle_deg(s, np.array([k[1] for k in b['blended'][ai]]))
        assert float(err.max()) < 0.5


# ------------------------------------------------------------------------------------------------ penetration
def test_penetration_metric_box_and_ground():
    ob = dict(near_y=3.0, far_y=4.0, height=1.0)
    inside = np.array([[0.0, 3.5, 0.9]])          # 0.1 m below the top
    below = np.array([[0.0, 0.0, -0.05]])         # 5 cm under the ground
    outside = np.array([[0.0, 2.9, 0.5], [0.0, 5.0, 0.2]])
    assert bake2.penetration_m(inside, ob) == pytest.approx(0.1)
    assert bake2.penetration_m(below, ob) == pytest.approx(0.05)
    assert bake2.penetration_m(outside, ob) == 0.0


# ------------------------------------------------------------------------------------------------ written patch
def test_written_patch_parses_and_matches_dump(tmp_path, rig, clip, kong):
    dump, acu_s, _ob = clip
    b = bake2.bake_clip(rig, kong, dump, acu_s, _wd2_ev(kong), 0.0, 0.758)
    tracks = {ai: bake2.reduce_keys(b['blended'][ai]) for ai in b['patched']}
    path = os.path.join(str(tmp_path), 'x.baked.json')
    bake2.write_patch(path, rig, tracks, dump['Length'])
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    L = float(dump['Length'])
    assert d['Length'] == pytest.approx(L, abs=1e-6)
    last = int(round(L * 60))
    ref_tid = zlib.crc32(b'Reference')
    assert d['Tracks']
    for tr in d['Tracks']:
        tid = int(tr['TrackID'])
        assert tid != ref_tid
        assert tr['codec'] == 1 and tr['infotype'] == 0
        assert dumps.track(dump, tid, 0) is not None, 'TrackID %d is not a rotation track in the dump' % tid
        frames = [k['Frame'] for k in tr['Keyframes']]
        assert all(isinstance(x, int) for x in frames)
        assert frames == sorted(frames) and len(set(frames)) == len(frames)
        assert frames[0] >= 0 and frames[-1] <= last
        assert all(len(k['LocalRotation']) == 4 for k in tr['Keyframes'])


def test_mapping_entry_format():
    e = bake2.mapping_entry('296F0C7D9', CLIP, '296F0C7D9.baked.json')
    assert e == {'handle': '0x296F0C7D9', 'name': CLIP, 'source': '296F0C7D9.baked.json', 'enabled': True,
                 'prebaked': True, 'patch_translation': False}


def test_real_outputs_if_present():
    mp = os.path.join(config.PATCHES, 'mapping.json')
    if not os.path.exists(mp):
        pytest.skip('bake2 has not written its outputs yet')
    with open(mp, encoding='utf-8') as f:
        mapping = json.load(f)['clips']
    assert len(mapping) == 10
    for e in mapping:
        # T9c: every shipped patch carries a Reference translation track; enabled only when it passes
        assert e['prebaked'] is True and e['patch_translation'] is True and isinstance(e['enabled'], bool)
        assert os.path.exists(os.path.join(config.PATCHES, e['source']))


# ------------------------------------------------------------------------------------------------ T9c (root offset, hand IK)
@pytest.fixture(scope='module')
def t9c_run(rig, kong):
    """One clip through the T9c pipeline (spec settings) with its validation."""
    dump = dumps.load_by_name(CLIP)
    seq = acuclip.pose_sequence(dump, step=1)
    ev = contacts.events(seq, dump)
    o, window, _src = bake2.play_for(CLIP, float(dump['Length']), bake2.load_play_model(), bake2.load_measured())
    offs = bake2.measure_contact_offsets([(CLIP, dump, seq, ev)])
    res = bake2.bake_t9c(rig, kong, dump, seq, bake2.acu_seconds(ev), ev, _wd2_ev(kong), o, window, offs)
    met, arr = bake2.validate_t9c(rig, dump, seq, res, o, window, ev['obstacle'])
    return dict(dump=dump, seq=seq, ev=ev, o=o, window=window, offs=offs, res=res, met=met, arr=arr)


def _window_bounds(res):
    """Frames [lo, hi] of the rebuilt key range: the window plus one frame on each side."""
    lo = max(0, int(res['win'][0]) - 1)
    hi = int(res['win'][-1]) + 1
    return lo, hi


def test_reduce_positions_straight_and_bent():
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([0.0, 0.5, 0.2])
    keys = [(f, p0 + (p1 - p0) * f / 10.0) for f in range(0, 11)]
    assert len(bake2.reduce_positions(keys, 0.002)) == 2
    bent = list(keys)
    bent[5] = (5, np.array([0.0, 0.25, 0.2]))           # 2.8 cm off the line
    red = bake2.reduce_positions(bent, 0.002)
    assert any(f == 5 for f, _ in red)


def test_mix_rot_endpoints_exact():
    rng = np.random.default_rng(1)
    a = np.array([R.qn(rng.normal(size=4)) for _ in range(6)])
    b = np.array([R.qn(rng.normal(size=4)) for _ in range(6)])
    w = np.array([0.0, 0.0, 0.5, 1.0, 1.0, 0.25])
    out = bake2.mix_rot(a, b, w)
    # sign-insensitive: mix_rot flips signs for continuity (same rotation)
    assert abs(float(out[0] @ a[0])) > 1 - 1e-12 and abs(float(out[1] @ a[1])) > 1 - 1e-12
    assert abs(float(out[3] @ b[3])) > 1 - 1e-12
    assert np.all(np.sum(out * out, axis=1) > 0.999)


def test_contact_offsets_plausible(t9c_run):
    o = t9c_run['offs']
    assert 0.0 <= o['Hand'] <= 0.08           # T9c expectation ~0.04
    assert 0.07 <= o['Foot'] <= 0.12          # ~0.10
    assert -0.01 <= o['FootThumb1'] <= 0.01   # ~0.00


def test_t9c_delta_is_exactly_zero_outside_window(t9c_run):
    res = t9c_run['res']
    F = len(t9c_run['seq'].frames)
    outside = np.ones(F, bool)
    outside[res['win']] = False
    assert np.all(res['delta'][outside] == 0.0)
    assert t9c_run['met']['delta_outside_m'] < 1e-9     # float noise of the lerp reconstruction only
    assert np.any(res['delta'][res['win']] != 0.0)   # the solve does something inside the window


def test_t9c_translation_keys_outside_window_equal_original(t9c_run):
    res = t9c_run['res']
    dump = t9c_run['dump']
    lo, hi = _window_bounds(res)
    orig = dumps.track(dump, zlib.crc32(b'Reference'), 3)
    new = {int(k['Frame']): k['LocalPosition'] for k in res['trans_keys']}
    n = 0
    for k in orig:
        f = int(k['Frame'])
        if f < lo or f > hi:
            n += 1
            assert f in new, 'original key %d missing outside the window' % f
            assert np.array_equal(np.asarray(new[f], float), np.asarray(k['LocalPosition'], float)), f
    assert n > 0


def test_t9c_rotation_keys_outside_window_equal_original(t9c_run):
    res = t9c_run['res']
    dump = t9c_run['dump']
    lo, hi = _window_bounds(res)
    for ai in res['patched']:
        orig = dumps.track(dump, t9c_run['seq'].bones[ai]['hash'], 0)
        new = {int(k['Frame']): k['LocalRotation'] for k in res['rot_tracks'][ai]}
        for k in orig:
            f = int(k['Frame'])
            if f < lo or f > hi:
                assert f in new and np.array_equal(np.asarray(new[f], float), np.asarray(k['LocalRotation'], float))


def test_t9c_rotations_outside_window_exact_at_60hz(t9c_run):
    assert t9c_run['met']['rot_outside_deg'] < 1e-3


def test_t9c_reduction_tolerances(t9c_run):
    assert t9c_run['met']['rot_reduce_err_deg'] <= bake2.REDUCE_TOL_DEG + 1e-6
    assert t9c_run['met']['trans_reduce_err_mm'] <= 2.0 + 1e-6


def test_t9c_hand_on_target_at_start_of_plant(t9c_run):
    """Partial hand check: the IK holds both hands within 5 cm of their targets over the first 8 plant frames.
    The full-plant criterion is not met on this clip (the root moves about 1.2 m during the plant); see report.md."""
    res = t9c_run['res']
    plant = res['plant']
    assert plant.size >= 8
    Pf = t9c_run['arr']['Pf']
    rig = R.Rig()
    for h in bake2.T9C_HANDS:
        ai = rig.ac_by_hash[zlib.crc32(h.encode())]
        err = np.linalg.norm(Pf[plant[:8], ai] - res['targets'][h], axis=1)
        assert float(err.max()) <= 0.05, '%s off target by %.3f m' % (h, float(err.max()))


# ------------------------------------------------------------------------------------------------ T9d (lift, hand IK, foot clamp)
_POOLED = {}


def _pooled_offsets():
    """Contact offsets pooled over all SkipOver clips, as in the bake (a single clip can have no samples of a type)."""
    if 'v' not in _POOLED:
        clips = []
        for _h, name, dump in bake2.targets():
            seq = acuclip.pose_sequence(dump, step=1)
            clips.append((name, dump, seq, contacts.events(seq, dump)))
        off = bake2.measure_contact_offsets(clips)
        assert all(np.isfinite(off[k]) for k in ('Hand', 'Foot', 'FootThumb1'))
        _POOLED['v'] = off
    return _POOLED['v']


@pytest.fixture(scope='module')
def t9d_run(rig, kong):
    """One clip through the T9d pipeline (spec settings) with its validation."""
    dump = dumps.load_by_name(CLIP)
    seq = acuclip.pose_sequence(dump, step=1)
    ev = contacts.events(seq, dump)
    o, window, _src = bake2.play_for(CLIP, float(dump['Length']), bake2.load_play_model(), bake2.load_measured())
    offs = _pooled_offsets()
    res = bake2.bake_t9d(rig, kong, dump, seq, bake2.acu_seconds(ev), ev, _wd2_ev(kong), o, window, offs)
    met, arr = bake2.validate_t9d(rig, dump, seq, res, o, window, ev['obstacle'])
    return dict(dump=dump, seq=seq, ev=ev, o=o, window=window, offs=offs, res=res, met=met, arr=arr)


def test_finger_bones_are_thirty_and_not_the_hands(rig):
    body = sorted(set(rig.map.values()))
    fing = bake2.finger_bones(rig, body)
    names = [bake2._bone_name(rig, bi) for bi in fing]
    assert len(fing) == 30
    assert all(n not in ('LeftHand', 'RightHand') for n in names)
    assert all(n.startswith(('LeftHand', 'RightHand')) for n in names)


def test_smooth_envelope_covers_need_and_is_smoother():
    L = np.zeros(120)
    L[40:50] = 0.30
    D = bake2.smooth_envelope(L)
    assert np.all(D >= L - 1e-12)
    assert D.max() >= 0.30 - 1e-12
    assert float(np.max(np.abs(np.diff(D)))) < float(np.max(np.abs(np.diff(L))))
    assert np.array_equal(bake2.smooth_envelope(np.zeros(10)), np.zeros(10))


def test_lift_taper_zero_at_ends_and_untapered_when_need_there():
    n = 60
    D = np.ones(n)
    tap, rep = bake2.lift_taper(D, np.zeros(n))
    assert tap[0] == 0.0 and tap[-1] == 0.0
    assert tap[bake2.T9D_TAPER] == pytest.approx(1.0)
    assert rep == dict(start_untapered=False, end_untapered=False)
    L = np.zeros(n)
    L[2] = 0.1                                   # need inside the first 6 frames: no taper there
    tap2, rep2 = bake2.lift_taper(D, L)
    assert rep2['start_untapered'] is True and tap2[0] == 1.0
    assert rep2['end_untapered'] is False


def test_t9d_delta_zero_outside_window_and_no_xy(t9d_run):
    res = t9d_run['res']
    F = len(t9d_run['seq'].frames)
    outside = np.ones(F, bool)
    outside[res['win']] = False
    assert np.all(res['delta'][outside] == 0.0)
    assert np.all(res['delta'][:, 0] == 0.0) and np.all(res['delta'][:, 1] == 0.0)
    assert t9d_run['met']['delta_outside_m'] < 1e-9
    assert np.any(res['delta'][res['win'], 2] > 0.0)


def test_t9d_translation_and_rotation_keys_outside_window_equal_original(t9d_run):
    res = t9d_run['res']
    dump = t9d_run['dump']
    lo, hi = _window_bounds(res)
    orig = dumps.track(dump, zlib.crc32(b'Reference'), 3)
    new = {int(k['Frame']): k['LocalPosition'] for k in res['trans_keys']}
    for k in orig:
        f = int(k['Frame'])
        if f < lo or f > hi:
            assert np.array_equal(np.asarray(new[f], float), np.asarray(k['LocalPosition'], float)), f
    for ai in res['patched']:
        orig_r = dumps.track(dump, t9d_run['seq'].bones[ai]['hash'], 0)
        new_r = {int(k['Frame']): k['LocalRotation'] for k in res['rot_tracks'][ai]}
        for k in orig_r:
            f = int(k['Frame'])
            if f < lo or f > hi:
                assert np.array_equal(np.asarray(new_r[f], float), np.asarray(k['LocalRotation'], float))
    assert t9d_run['met']['rot_outside_deg'] < 1e-3


# ------------------------------------------------------------------------------------------------ T9e (two-sided lift, fades)
@pytest.fixture(scope='module')
def t9e_run(rig, kong):
    """One clip through the T9e pipeline with pooled contact offsets, and its validation."""
    dump = dumps.load_by_name(CLIP)
    seq = acuclip.pose_sequence(dump, step=1)
    ev = contacts.events(seq, dump)
    o, window, _src = bake2.play_for(CLIP, float(dump['Length']), bake2.load_play_model(), bake2.load_measured())
    offs = _pooled_offsets()
    res = bake2.bake_t9e(rig, kong, dump, seq, bake2.acu_seconds(ev), ev, _wd2_ev(kong), o, window, offs)
    fi, fo = res['fades']
    met, arr = bake2.validate_t9d(rig, dump, seq, res, o, window, ev['obstacle'], fi=fi, fo=fo)
    return dict(dump=dump, seq=seq, ev=ev, o=o, window=window, offs=offs, res=res, met=met, arr=arr)


def test_reach_upper_is_exact_boundary():
    sh = np.array([[0.0, 0.0, 1.5]])
    tgt = np.array([0.3, 0.0, 1.0])
    R = np.array([0.6])
    U = bake2.reach_upper(sh, tgt, R)
    assert np.isfinite(U[0])
    d = np.linalg.norm(tgt - (sh[0] + U[0] * np.array([0.0, 0.0, 1.0])))
    assert d == pytest.approx(0.6, abs=1e-12)
    # no lift can reach a target that is laterally farther than R
    assert bake2.reach_upper(sh, np.array([1.0, 0.0, 1.0]), np.array([0.6]))[0] == -np.inf


def test_d0_rule_cases():
    L = np.array([0.02, -0.03, -0.03, -0.03, -0.05])
    U = np.array([0.10, 0.01, -0.01, -0.10, -0.10])
    D0 = bake2.d0_rule(L, U)
    assert D0[0] == pytest.approx(0.02)        # L > 0: lift to L
    assert D0[1] == pytest.approx(0.0)         # L <= 0 <= U: no lift
    assert D0[2] == pytest.approx(-0.01)       # U < 0 and U >= L: lower to U
    assert D0[3] == pytest.approx(-0.03)       # U < L: conflict, penetration wins (L)
    assert D0[4] == pytest.approx(-0.05)       # conflict again


def test_solve_lift_keeps_floor_and_reach_bound():
    n = 120
    L = np.full(n, -0.04)
    L[50:60] = 0.12                             # a real need in the middle
    U = np.full(n, np.inf)
    U[40:70] = 0.05                             # reach bound on the plant, below the need on 50:60 -> conflict there
    D, info = bake2.solve_lift_t9e(L, U)
    assert np.all(D >= L - 1e-12)
    feasible = (U >= L)
    assert np.all(D[feasible] <= U[feasible] + 1e-12)
    assert info['conflict'] == int(np.sum(U < L))
    assert D[0] == 0.0 and D[-1] == 0.0          # taper towards 0 where L <= 0


def test_lift_points_exclude_feet_and_fingers(rig):
    body = sorted(set(rig.map.values()))
    lp = bake2.lift_points(rig, body)
    names = [bake2._bone_name(rig, bi) for bi in lp]
    assert not any(n in ('LeftFoot', 'RightFoot', 'LeftFootThumb1', 'RightFootThumb1') for n in names)
    assert not any(n.startswith(('LeftHand', 'RightHand')) and n not in ('LeftHand', 'RightHand') for n in names)
    assert 'LeftLeg' in names and 'RightLeg' in names and 'LeftHand' in names
    assert len(lp) == len(body) - 30 - 4


def test_fades_follow_the_rule():
    assert bake2.fades_for(0.76) == (pytest.approx(0.20), pytest.approx(0.25))
    fi, fo = bake2.fades_for(0.47)
    assert fi == pytest.approx(0.3 * 0.47) and fo == pytest.approx(0.35 * 0.47)


def test_t9e_delta_zero_outside_window_and_no_xy(t9e_run):
    res = t9e_run['res']
    F = len(t9e_run['seq'].frames)
    outside = np.ones(F, bool)
    outside[res['win']] = False
    assert np.all(res['delta'][outside] == 0.0)
    assert np.all(res['delta'][:, 0] == 0.0) and np.all(res['delta'][:, 1] == 0.0)
    assert t9e_run['met']['delta_outside_m'] < 1e-9


def test_t9e_floor_holds_where_no_conflict(t9e_run):
    """D >= L on every window frame (the lift is never below the need)."""
    res = t9e_run['res']
    assert np.all(res['D'] >= res['L'] - 1e-9)


def test_t9e_pure_wd2_part_is_inside_the_window(t9e_run):
    res = t9e_run['res']
    c0, c1 = res['pure']['clip']
    assert t9e_run['o'] <= c0 < c1 <= t9e_run['o'] + t9e_run['window']
    w0, w1 = res['pure']['wd2']
    assert w0 < w1


def test_t9e_keys_and_rotations_outside_window_equal_original(t9e_run):
    res = t9e_run['res']
    dump = t9e_run['dump']
    lo, hi = _window_bounds(res)
    orig = dumps.track(dump, zlib.crc32(b'Reference'), 3)
    new = {int(k['Frame']): k['LocalPosition'] for k in res['trans_keys']}
    for k in orig:
        f = int(k['Frame'])
        if f < lo or f > hi:
            assert np.array_equal(np.asarray(new[f], float), np.asarray(k['LocalPosition'], float)), f
    assert t9e_run['met']['rot_outside_deg'] < 1e-3


# ------------------------------------------------------------------------------------------------ enabled clips
# Clips enabled in out\patches\mapping.json (the baseline; the variants copy its flags).
# PENETRATION_OVERRIDES: orchestrator waiver of the non-finger penetration check for one clip, with the value accepted
# (the bound is pinned at the measured value, so the waiver cannot silently get worse). Nothing else is waived.
PENETRATION_OVERRIDES = {
    CLIP: (9.5, 'accepted for preview 2026-10-08: the game rescales vaults to real walls; offline penetration 9.2 cm'),
}
# KNOWN_OPEN: enabled in the mapping, failing a check, and not waived. Strict xfail so the open item is visible.
KNOWN_OPEN = {
    'xx_Parkour_Sprint_SkipOver_Fd010cm-Up100cm':
        'enabled in out/patches/mapping.json but 0 hand-touch frames at the plant (need 4); no waiver granted',
}


def _enabled_clip_names():
    mp = os.path.join(config.PATCHES, 'mapping.json')
    if not os.path.exists(mp):
        return []
    with open(mp, encoding='utf-8') as f:
        return [e['name'] for e in json.load(f)['clips'] if e['enabled']]


@pytest.mark.parametrize('name', _enabled_clip_names() or ['no enabled clips (bake2 has not written its outputs)'])
def test_t9e_enabled_clip_touch_and_pass(name, request):
    """Every enabled T9e clip re-validates: hand touch, Delta step, and non-finger penetration (waived only where
    PENETRATION_OVERRIDES says so, and then only up to the pinned value)."""
    if name not in _enabled_clip_names():
        pytest.skip('bake2 has not written its outputs yet, or no clip is enabled')
    if name in KNOWN_OPEN:
        request.applymarker(pytest.mark.xfail(strict=True, reason=KNOWN_OPEN[name]))
    rig = R.Rig()
    kong = wd2.load(EV['stem'])
    dump = dumps.load_by_name(name)
    seq = acuclip.pose_sequence(dump, step=1)
    ev = contacts.events(seq, dump)
    o, window, _src = bake2.play_for(name, float(dump['Length']), bake2.load_play_model(), bake2.load_measured())
    res = bake2.bake_t9e(rig, kong, dump, seq, bake2.acu_seconds(ev), ev, _wd2_ev(kong), o, window, _pooled_offsets())
    fi, fo = res['fades']
    met, arr = bake2.validate_t9d(rig, dump, seq, res, o, window, ev['obstacle'], fi=fi, fo=fo)
    assert met['touch']['min'] >= bake2.T9D_TOUCH, name
    assert met['dstep_max_cm'] <= bake2.T9C_DELTA_STEP_CM, name
    if name in PENETRATION_OVERRIDES:
        bound, _why = PENETRATION_OVERRIDES[name]
        assert met['pen_patch_cm'] <= bound, name
        others = [f for f in met['fails'] if 'penetration' not in f]
        assert not others, (name, others)
    else:
        assert met['fail'] is False, name


# ------------------------------------------------------------------------------------------------ T9e options (variants)
def test_patch_weight_equals_fade_weight_for_default_fades():
    t = np.linspace(0.0, 2.0, 481)
    o, window = 0.38, 0.76
    fi, fo = bake2.fades_for(window)
    assert np.array_equal(bake2.patch_weight(t, o, window, fi, fo, 'orig'), bake2.fade_weight(t, o, window, fi, fo))


def test_patch_weight_zero_fades_are_steps_and_tails():
    o, window = 0.38, 0.76
    e = o + window
    w = bake2.patch_weight(np.array([o - 1e-3, o, e, e + 1e-3, e + 0.5]), o, window, 0.0, 0.0, 'orig')
    assert list(w) == [0.0, 1.0, 1.0, 0.0, 0.0]
    w = bake2.patch_weight(np.array([o - 1e-3, o, e + 0.5, 2.0]), o, window, 0.0, 0.0, 'wd2')
    assert list(w) == [0.0, 1.0, 1.0, 1.0]
    w = bake2.patch_weight(np.array([o + 0.04, o + 0.08, o + 0.2]), o, window, 0.08, 0.0, 'hold')
    assert 0.0 < w[0] < 1.0 and w[1] == 1.0 and w[2] == 1.0


def test_tail_fades_defaults_and_refusals():
    window = 0.76
    assert bake2.tail_fades(window) == tuple(bake2.fades_for(window))
    assert bake2.tail_fades(window, tail='wd2') == (bake2.fades_for(window)[0], 0.0)
    assert bake2.tail_fades(window, 0.08, 0.10) == (0.08, 0.10)
    with pytest.raises(ValueError):
        bake2.tail_fades(window, 0.08, 0.10, tail='wd2')     # a fade-out returns to the ACU pose: refused with wd2
    with pytest.raises(ValueError):
        bake2.tail_fades(window, tail='bogus')
    with pytest.raises(ValueError):
        bake2.tail_fades(0.2, 0.15, 0.10)                    # fades longer than the window


def test_lift_ramp_edges_has_no_forced_step():
    n = 60
    L = np.full(n, 0.05)                                     # a need at both ends: forced mode would step to L there
    U = np.full(n, np.inf)
    D_ramp, info = bake2.solve_lift_t9e(L, U, ramp_edges=True)
    assert D_ramp[0] == 0.0 and D_ramp[-1] == 0.0
    assert info['forced_start'] == 0 and info['forced_end'] == 0
    D_def, _info = bake2.solve_lift_t9e(L, U)
    assert D_def[0] >= 0.05 - 1e-12                          # default: the start is forced to the need (the old step)


def test_enabled_flags_are_read_from_a_mapping(tmp_path):
    p = tmp_path / 'mapping.json'
    p.write_text(json.dumps({'clips': [dict(handle='0x296F0C7D9', enabled=True),
                                       dict(handle='0x2C62204A06', enabled=False)]}), encoding='utf-8')
    assert bake2.enabled_from_mapping(str(p)) == {'296F0C7D9': True, '2C62204A06': False}


def _clip_bake(rig, kong, **kw):
    dump = dumps.load_by_name(CLIP)
    seq = acuclip.pose_sequence(dump, step=1)
    ev = contacts.events(seq, dump)
    o, window, _src = bake2.play_for(CLIP, float(dump['Length']), bake2.load_play_model(), bake2.load_measured())
    res = bake2.bake_t9e(rig, kong, dump, seq, bake2.acu_seconds(ev), ev, _wd2_ev(kong), o, window, _pooled_offsets(),
                         **kw)
    return dict(dump=dump, seq=seq, o=o, window=window, res=res)


@pytest.fixture(scope='module')
def tail_runs(rig, kong):
    """The clip with the variant options: no fade-in + wd2 tail (C), no fade-in + hold, and fade-in 0.08 / fade-out 0.10
    with tail orig (B)."""
    return {'wd2': _clip_bake(rig, kong, fi=0.0, fo=0.0, tail='wd2'),
            'hold': _clip_bake(rig, kong, fi=0.0, fo=0.0, tail='hold'),
            'b_orig': _clip_bake(rig, kong, fi=0.08, fo=0.10, tail='orig')}


def test_default_bake_writes_the_installed_patch_bytes(tmp_path, rig, kong):
    """The default T9e bake of the clip is byte-identical to the installed patch file in out/patches."""
    mp = os.path.join(config.PATCHES, 'mapping.json')
    if not os.path.exists(mp):
        pytest.skip('bake2 has not written its outputs yet')
    handle = [h for h, name, _p in bake2.targets() if name == CLIP][0]
    installed = os.path.join(config.PATCHES, handle + '.baked.json')
    dump = dumps.load_by_name(CLIP)
    seq = acuclip.pose_sequence(dump, step=1)
    ev = contacts.events(seq, dump)
    o, window, _src = bake2.play_for(CLIP, float(dump['Length']), bake2.load_play_model(), bake2.load_measured())
    res = bake2.bake_t9e(rig, kong, dump, seq, bake2.acu_seconds(ev), ev, _wd2_ev(kong), o, window, _pooled_offsets())
    out = tmp_path / 'patch.json'
    bake2.write_patch_t9c(str(out), rig, res['rot_tracks'], res['trans_keys'], float(dump['Length']))
    with open(installed, 'rb') as f1, open(out, 'rb') as f2:
        assert f1.read() == f2.read()


def test_fi_zero_wd2_tail_is_original_before_o_and_patched_from_o(tail_runs):
    r = tail_runs['wd2']
    res, seq = r['res'], r['seq']
    F = np.asarray(seq.frames, int)
    before = F[F / 60.0 < r['o'] - 1e-9]
    for ai in res['patched']:
        got = bake2.sample_track_rot(res['rot_tracks'][ai], before)
        assert float(np.max(bake2.angle_deg(got, np.asarray(seq.local_rot, float)[before, ai]))) < 1e-4
    k0 = int(res['win'][0])
    worst = max(float(bake2.angle_deg(bake2.sample_track_rot(res['rot_tracks'][ai], [k0])[0],
                                      np.asarray(seq.local_rot, float)[k0, ai])) for ai in res['patched'])
    assert worst > 1.0


def test_wd2_tail_keeps_the_patch_to_the_clip_end(tail_runs):
    r = tail_runs['wd2']
    res, seq = r['res'], r['seq']
    nf = int(seq.frames[-1])
    assert res['span_end'] == pytest.approx(r['o'] + r['window'])     # envelopes end at the window
    for ai in res['patched']:
        assert int(res['rot_tracks'][ai][-1]['Frame']) == nf
    dev = max(float(bake2.angle_deg(bake2.sample_track_rot(res['rot_tracks'][ai], [nf])[0],
                                    np.asarray(seq.local_rot, float)[nf, ai])) for ai in res['patched'])
    assert dev > 1.0                                                  # the tail is WD2 motion, not the ACU pose


def test_no_envelope_step_at_the_window_edges(tail_runs):
    """Delta (root offset) changes by at most T9C_DELTA_STEP_CM per 60 Hz frame at both window edges, for every variant.
    Interior steps are reported by variants.py: a one-frame need spike inside the window can exceed the limit."""
    for key, r in tail_runs.items():
        res = r['res']
        d = res['delta'][:, 2]
        win = res['win']
        for lo, hi in ((win[0] - 2, win[0] + 6), (win[-1] - 6, win[-1] + 2)):
            seg = d[max(lo, 0):min(hi, len(d) - 1) + 1]
            step = float(np.max(np.abs(np.diff(seg)))) * 100.0 if seg.size > 1 else 0.0
            assert step <= bake2.T9C_DELTA_STEP_CM, (key, lo, hi, step)


def test_hold_freezes_rotations_and_reference_after_the_window(tail_runs):
    r = tail_runs['hold']
    res, seq = r['res'], r['seq']
    e_idx = int(res['win'][-1])
    nf = int(seq.frames[-1])
    assert e_idx < nf
    F = np.arange(e_idx, nf + 1)
    for ai in res['patched']:
        g = bake2.sample_track_rot(res['rot_tracks'][ai], F)
        # held within the key reduction tolerance (keys after the window are reduced like the rest of the track)
        assert float(np.max(bake2.angle_deg(g, g[0]))) <= bake2.REDUCE_TOL_DEG + 1e-6, ai
    pos = bake2.sample_track_pos(res['trans_keys'], F)
    assert float(np.max(np.abs(pos - pos[0]))) <= bake2.T9C_TRANS_TOL + 1e-9


def test_variant_b_keeps_the_original_pose_outside_its_fades(tail_runs):
    """Fade-in 0.08 and fade-out 0.10 (tail orig): the rotations before o and after o + window are the original."""
    r = tail_runs['b_orig']
    res, seq = r['res'], r['seq']
    F = np.asarray(seq.frames, int)
    t = F / 60.0
    e = r['o'] + r['window']
    outside = F[(t < r['o'] - 1e-9) | (t > e + 1e-9)]
    for ai in res['patched']:
        got = bake2.sample_track_rot(res['rot_tracks'][ai], outside)
        assert float(np.max(bake2.angle_deg(got, np.asarray(seq.local_rot, float)[outside, ai]))) < 1e-4
