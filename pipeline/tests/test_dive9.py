"""dive9 (T21): the Exit stand-up profile (C1), the pelvis rate floor, the timing map, the upright metric, the mapping and aux
windows of the written patches, and the W seam. Output checks read out\\dive9 and skip when it is not written yet."""
import json
import os

import numpy as np
import pytest

import aux_tracks as AT
import config
import dive8 as D8
import dive9 as D9
import dumps

OUT = os.path.join(config.OUT, 'dive9')
PATCHES = os.path.join(OUT, 'patches')
MAPPING = os.path.join(PATCHES, 'mapping_dive9.json')
SUMMARY = os.path.join(OUT, 'summary.json')
W_CLIP = 1.201                   # SlideOver W (clip s), the dive4 constant (D4.EXIT_O is the Exit played start)


def _load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------------ the stand-up profile
def test_exit_profile_starts_at_the_takeoff_height_with_the_entry_slope():
    z0, v0 = 0.404, 2.5
    p = np.array([0.0])
    assert D9.exit_z_profile(p, z0, v0)[0] == pytest.approx(z0)
    h = 1e-5
    slope = (D9.exit_z_profile(np.array([h]), z0, v0)[0] - z0) / h
    assert slope == pytest.approx(v0, rel=1e-3)                 # C1 at W: the SlideOver's vertical speed carries over


def test_exit_profile_reaches_the_stand_height_flat_and_holds_it():
    z0, v0 = 0.404, 2.5
    at = D9.exit_z_profile(np.array([D9.P_UP]), z0, v0)[0]
    assert at == pytest.approx(D9.Z_UP)
    h = 1e-5
    slope_end = (D9.exit_z_profile(np.array([D9.P_UP + h]), z0, v0)[0] - at) / h
    assert abs(slope_end) < 1e-3                                 # zero slope at the rise end (C1)
    held = D9.exit_z_profile(np.array([0.30, 0.5]), z0, v0)
    assert np.allclose(held, D9.Z_UP)


def test_exit_profile_peak_speed_is_the_hermite_formula_when_the_slope_is_zero():
    z0 = 0.404
    p = np.linspace(0.0, D9.P_UP, 4001)
    z = D9.exit_z_profile(p, z0, 0.0)
    peak = np.max(np.abs(np.diff(z))) * (len(p) - 1) / D9.P_UP
    assert peak == pytest.approx(1.5 * (D9.Z_UP - z0) / D9.P_UP, rel=2e-3)


def test_exit_profile_is_monotone_when_it_rises():
    p = np.linspace(0.0, 0.4, 500)
    z = D9.exit_z_profile(p, 0.404, 2.5)
    assert np.all(np.diff(z) >= -1e-12)


# ------------------------------------------------------------------------------------------------ the rate floor
def test_rate_floor_only_raises_and_limits_the_fall_per_frame():
    f = D9.FPS
    base = np.concatenate([np.full(20, 1.0), np.full(20, 0.1)])        # a 0.9 m drop in one frame
    lift = np.zeros_like(base)
    out = D9.rate_floor(lift, base, 6.0)
    assert np.all(out >= lift - 1e-12)
    z = base + out
    assert np.max(-np.diff(z)) * f <= 6.0 + 1e-9


def test_rate_floor_leaves_a_slow_fall_alone_and_is_idempotent():
    base = np.linspace(1.0, 0.5, 60)                            # 0.5 m over 1 s: 0.5 m/s, well under 6 m/s
    lift = np.zeros_like(base)
    once = D9.rate_floor(lift, base, 6.0)
    assert np.allclose(once, lift)
    fast = np.concatenate([np.full(10, 1.0), np.full(10, 0.0)])
    a = D9.rate_floor(np.zeros_like(fast), fast, 6.0)
    b = D9.rate_floor(a, fast, 6.0)
    assert np.allclose(a, b)


# ------------------------------------------------------------------------------------------------ the ease
def test_ease_weight_is_zero_before_the_exit_and_flat_at_the_end():
    w = 0.7
    assert np.allclose(D9.ease_weight(np.array([-0.2, -0.01, 0.0]), w), 0.0)
    assert D9.ease_weight(np.array([D9.EASE_T]), w)[0] == pytest.approx(w)
    assert np.allclose(D9.ease_weight(np.array([0.3, 0.5]), w), w)
    h = 1e-6
    assert D9.ease_weight(np.array([h]), w)[0] < 1e-8           # smoothstep: zero slope at the start


# ------------------------------------------------------------------------------------------------ the timing
def test_the_chosen_timing_is_the_dive8_map_at_the_chosen_speeds():
    case = D9.make_case9()
    tm = case.tm
    assert isinstance(tm, D8.Timing8)
    assert tm.k_plant == pytest.approx(2.0) and tm.k == pytest.approx(3.0)
    assert float(tm.w(tm.t_pl0)) == pytest.approx(0.75)
    assert float(tm.w(tm.t_pe)) == pytest.approx(0.92)
    assert float(tm.w(tm.t_feet)) == pytest.approx(1.2)
    assert float(tm.w(W_CLIP)) == pytest.approx(1.508, abs=2e-3)
    assert tm.t_feet - W_CLIP == pytest.approx(-0.103, abs=1e-3)   # feet down 0.10 s before W


def test_screen_rows_show_the_chosen_timing_upright_at_w_with_no_inverted_trunk():
    path = os.path.join(OUT, 'screen.json')
    if not os.path.exists(path):
        pytest.skip('screen not written yet')
    rows = _load(path)
    row = [r for r in rows if abs(r['T_to'] - 0.62) < 1e-9 and r['k_plant'] == 2.0 and r['k_roll'] == 3.0]
    assert len(row) == 1
    r = row[0]
    assert r['tilt_W'] <= 60.0 and not r['inv_W']
    assert r['tilt_before_W_max'] <= 60.0 and not r['inv_before_W']


# ------------------------------------------------------------------------------------------------ the upright metric
def _poses(hips_z, head_z, n=1):
    chain = [0, 1, 2, 3, 4, 5]
    P = np.zeros((n, 6, 3))
    zs = np.linspace(hips_z, head_z, 6)
    P[:, :, 2] = zs
    return P, chain


def test_upright_metric_vertical_trunk_is_zero_tilt_and_not_inverted():
    P, chain = _poses(1.0, 1.8)
    tilt, inv, up = D9.upright(P, chain)
    assert tilt[0] == pytest.approx(0.0, abs=1e-9)
    assert not inv[0] and up[0]


def test_upright_metric_head_below_hips_is_inverted():
    P, chain = _poses(1.0, 0.2)
    tilt, inv, up = D9.upright(P, chain)
    assert tilt[0] == pytest.approx(180.0, abs=1e-6)
    assert inv[0] and not up[0]


def test_upright_metric_flat_trunk_is_ninety_degrees():
    P = np.zeros((1, 6, 3))
    P[0, :, 0] = np.linspace(0.0, 0.8, 6)                      # horizontal trunk
    P[0, :, 2] = 0.3
    tilt, inv, up = D9.upright(P, [0, 1, 2, 3, 4, 5])
    assert tilt[0] == pytest.approx(90.0, abs=1e-6)


# ------------------------------------------------------------------------------------------------ the written outputs
@pytest.mark.parametrize('name', ['47BFB02DF.baked.json', 'F7919BDF0.baked.json'])
def test_patches_carry_the_inverse_reference_and_foot_tracks(name):
    path = os.path.join(PATCHES, name)
    if not os.path.exists(path):
        pytest.skip('dive9 patches not written yet')
    ids = {(int(t['TrackID']), t['infotype']) for t in _load(path)['Tracks']}
    for tid in AT.FOOT_IDS:
        assert (tid, 3) in ids
    for tid in AT.INV_IDS:
        assert (tid, 0) in ids and (tid, 3) in ids


def test_mapping_enables_both_clips_with_the_override_and_the_gate():
    if not os.path.exists(MAPPING):
        pytest.skip('dive9 mapping not written yet')
    m = _load(MAPPING)
    by = {c['handle'].replace('0x', ''): c for c in m['clips']}
    so, ex = by[D9.SO_HEX], by[D9.EXIT_HEX]
    for c in (so, ex):
        assert c['enabled'] is True and c['override'] == D9.OVERRIDE
        assert c['prebaked'] is True and c['patch_translation'] is True
    assert ex['only_after'] == [D9.SO_NAME] and ex['arm_seconds'] == 2.0
    assert 'only_after' not in so
    assert so['source'] == D9.SO_HEX + '.baked.json' and ex['source'] == D9.EXIT_HEX + '.baked.json'


def test_mapping_has_the_same_keys_as_the_dive8_b_mapping():
    d8 = os.path.join(config.OUT, 'dive8', 'B_fastroll', 'mapping_dive8_B.json')
    if not (os.path.exists(MAPPING) and os.path.exists(d8)):
        pytest.skip('a mapping is missing')
    a, b = _load(MAPPING), _load(d8)
    assert set(a) == set(b)
    for ca, cb in zip(a['clips'], b['clips']):
        assert set(ca) == set(cb)


def test_exit_feet_are_free_up_to_the_reported_frame_and_the_so_feet_from_takeoff():
    if not os.path.exists(SUMMARY):
        pytest.skip('summary not written yet')
    feet_to = _load(SUMMARY)['feet_free_to_frame']
    ex = _load(os.path.join(PATCHES, D9.EXIT_HEX + '.baked.json'))
    so = _load(os.path.join(PATCHES, D9.SO_HEX + '.baked.json'))
    lk = [t['Keyframes'] for t in ex['Tracks'] if int(t['TrackID']) == AT.LEFT_FOOT_ID][0]
    for f in range(0, feet_to):
        assert np.allclose(AT.sample_foot(lk, f), AT.SENTINEL), f
    sk = [t['Keyframes'] for t in so['Tracks'] if int(t['TrackID']) == AT.LEFT_FOOT_ID][0]
    assert np.allclose(AT.sample_foot(sk, D9.TAKEOFF_FRAME + 2), AT.SENTINEL)


def test_the_reference_is_continuous_across_w_in_the_patches():
    so_p = os.path.join(PATCHES, D9.SO_HEX + '.baked.json')
    ex_p = os.path.join(PATCHES, D9.EXIT_HEX + '.baked.json')
    if not (os.path.exists(so_p) and os.path.exists(ex_p)):
        pytest.skip('dive9 patches not written yet')
    rk_s = [t['Keyframes'] for t in _load(so_p)['Tracks'] if int(t['TrackID']) == D9.D8.REF_ID and t['infotype'] == 3][0]
    rk_e = [t['Keyframes'] for t in _load(ex_p)['Tracks'] if int(t['TrackID']) == D9.D8.REF_ID and t['infotype'] == 3][0]
    a = dumps.sample_position(rk_s, 72)
    b = dumps.sample_position(rk_e, 13)
    assert np.linalg.norm(a[:2] - b[:2]) < 0.05                 # xy: within 5 cm across the seam
    assert abs(a[2] - b[2]) < 0.06                              # z: within 6 cm (the lift and one frame of the rise)


def test_summary_meets_the_rules_it_claims_to_meet():
    if not os.path.exists(SUMMARY):
        pytest.skip('summary not written yet')
    s = _load(SUMMARY)
    m = s['metrics']
    assert m['tilt_W'] <= 60.0 and not m['inv_W']
    assert m['tilt_exit_max'] <= 60.0 and not m['inv_exit']
    assert m['z_at_0.16'] >= D9.Z_UP_MIN
    assert m['gap_post_max'] <= D9.GAP_GOAL
    assert m['pelvis_drop_max_mps'] <= D9.PELVIS_V_MAX + 1e-6
    assert s['touch']['min'] >= D9.TOUCH_MIN
    assert s['cont'] <= D9.CONT_MAX


def test_out_dive9_holds_only_the_expected_files():
    if not os.path.isdir(OUT):
        pytest.skip('out\\dive9 not written yet')
    names = set(os.listdir(OUT))
    assert names <= {'patches', 'nofloor', 'report.md', 'summary.json', 'screen.json', 'joined_sheet.png', 'joined_sheet.gif'}
    assert set(os.listdir(PATCHES)) == {'47BFB02DF.baked.json', 'F7919BDF0.baked.json', 'mapping_dive9.json'}


# ------------------------------------------------------------------------------------------------ the no-floor A/B set
NOFLOOR = os.path.join(OUT, 'nofloor')


def test_nofloor_differs_from_the_hover_patches_only_in_the_fall_window():
    """out/dive9/nofloor is the same setup without the 6 m/s rate floor. The change must stay inside the pelvis fall: the
    Reference translation, the rotations the IK re-solves from it, and the inverse-Reference position tracks derived from the
    Reference. The Exit patch must be identical (the floor only acts before W)."""
    if not os.path.isdir(NOFLOOR):
        pytest.skip('nofloor set not written yet')
    for name in (D9.SO_HEX, D9.EXIT_HEX):
        a = _load(os.path.join(PATCHES, name + '.baked.json'))
        b = _load(os.path.join(NOFLOOR, name + '.baked.json'))
        ta = {(int(t['TrackID']), t['infotype']): t for t in a['Tracks']}
        tb = {(int(t['TrackID']), t['infotype']): t for t in b['Tracks']}
        assert set(ta) == set(tb)
        last = int(round(a['Length'] * 60))
        changed_tracks, changed_frames = set(), set()
        for k in ta:
            ka, kb = ta[k]['Keyframes'], tb[k]['Keyframes']
            if ka == kb:
                continue
            changed_tracks.add(k)
            for f in range(last + 1):
                if k[1] == 0:
                    d = np.abs(np.array(dumps.sample_rotation(ka, f)) - np.array(dumps.sample_rotation(kb, f))).max()
                else:
                    d = np.abs(np.array(dumps.sample_position(ka, f)) - np.array(dumps.sample_position(kb, f))).max()
                if d > 1e-9:
                    changed_frames.add(f)
        if name == D9.EXIT_HEX:
            assert not changed_tracks and not changed_frames
            continue
        ref_pos = (D9.D8.REF_ID, 3)
        derived_pos = {(tid, 3) for tid in AT.INV_IDS}
        changed_pos = {k for k in changed_tracks if k[1] == 3}
        assert changed_pos <= {ref_pos} | derived_pos
        assert not any(k in changed_tracks for k in ((tid, 0) for tid in AT.INV_IDS))   # the inverse rotation is unchanged
        assert changed_frames and min(changed_frames) >= 60 and max(changed_frames) <= 66


def test_nofloor_summary_reports_the_unfloored_fall():
    path = os.path.join(NOFLOOR, 'summary.json')
    if not os.path.exists(path):
        pytest.skip('nofloor summary not written yet')
    s = _load(path)
    assert s['variant'] == 'nofloor'
    assert s['metrics']['pelvis_drop_max_mps'] > D9.PELVIS_V_MAX          # the floor is off: the fall is unlimited
    assert s['contact_dev_m'] < 0.2                                       # the contact envelope is still on
    m = _load(os.path.join(NOFLOOR, 'mapping_dive9.json'))
    assert all(c['override'] == D9.NOFLOOR_OVERRIDE for c in m['clips'])
