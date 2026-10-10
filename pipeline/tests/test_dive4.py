"""dive4 (T15): the pure parts (time map, bump amplitude, placement scan and pick, the Place mapping, the played-window step,
the Exit convergence, the mapping schema, the written patch contents). No game files are read except the written outputs."""
import json
import os
import types
import zlib

import numpy as np
import pytest

import bake2
import config
import dive4 as D4
import dive_chain2 as DC2
import wd2_path as WP

OUT = os.path.join(config.OUT, 'dive4')


# ------------------------------------------------------------------------------------------------ time map (k = 1)
def test_w_of_t_maps_takeoff_to_wd2_takeoff_at_unit_speed():
    t_acu = 0.4667
    assert float(D4.w_of_t(t_acu, t_acu)) == pytest.approx(WP.TAKEOFF)
    t = np.array([t_acu, t_acu + 0.3, t_acu + 0.75])
    assert np.allclose(D4.w_of_t(t, t_acu), [0.45, 0.75, 1.2])        # slope 1: WD2 seconds = clip seconds


# ------------------------------------------------------------------------------------------------ bump amplitude
def _grid():
    return np.round(np.arange(D4.BUMP_T0, D4.BUMP_T1 + 1e-9, 1.0 / 60), 6)


def test_amplitude_from_lift_is_the_minimum_that_covers_the_lift():
    ws = _grid()
    shape = WP.raised_cosine(ws, D4.BUMP_T0, D4.BUMP_T1)
    amp, info = D4.amplitude_from_lift(ws, 0.1 * shape)
    assert amp == pytest.approx(0.1)
    assert info['L_max'] == pytest.approx(0.1)


def test_amplitude_from_lift_refuses_a_lift_where_the_shape_is_zero():
    ws = _grid()
    L = np.zeros_like(ws)
    L[0] = 0.02                                   # at w = 0.35, where the raised cosine is zero
    amp, info = D4.amplitude_from_lift(ws, L)
    assert amp is None and info['bad_w']


def test_amplitude_from_lift_zero_lift_gives_zero():
    ws = _grid()
    amp, _info = D4.amplitude_from_lift(ws, np.zeros_like(ws))
    assert amp == 0.0


# ------------------------------------------------------------------------------------------------ placement scan
def _fake_ctx():
    return types.SimpleNamespace(
        ob=dict(far_y=4.6043, near_y=3.6047, height=0.4998),
        foot_take_y=2.4413, stance_rel_y=-0.652, hand_rel_y={'L': 0.957, 'R': 0.889}, P_W_rel_y=1.70,
        orig_W=np.array([0.0, 5.458, 0.714]), orig_takeoff_y=2.5756)


def test_scan_rows_follow_the_foot_anchor_and_the_cost():
    ctx = _fake_ctx()
    rows = D4.scan_rows(ctx, s_grid=np.array([1.0, 1.5]), shift_grid=np.array([0.0, 0.4]))
    r = {(x['s'], x['shift']): x for x in rows}
    r10 = r[(1.0, 0.0)]
    assert r10['y_take'] == pytest.approx(2.4413 + 0.652 * 1.0)          # the stance foot lands on the measured foot
    assert r10['hand_R'] == pytest.approx(r10['y_take'] + 0.889)
    assert r10['hand_clear'] == pytest.approx(min(r10['hand_L'], r10['hand_R']) - 4.6043)
    assert r10['dW'] == pytest.approx(r10['y_take'] + 1.70 - 5.458)
    assert r10['cost'] == pytest.approx(0.0 + 0.5 * 0.0 + 0.5 * r10['dW'] ** 2)
    r15 = r[(1.5, 0.4)]
    assert r15['cost'] == pytest.approx(0.4 ** 2 + 0.5 * 0.25 + 0.5 * r15['dW'] ** 2)
    assert r15['feasible'] == (r15['hand_clear'] >= D4.FAR_CLEAR - 1e-9)


def test_pick_is_the_lowest_cost_feasible_row():
    rows = [dict(feasible=False, cost=0.01, s=1.0, shift=0.0),
            dict(feasible=True, cost=0.5, s=1.2, shift=0.1),
            dict(feasible=True, cost=0.3, s=1.4, shift=0.2),
            dict(feasible=True, cost=0.3, s=1.3, shift=0.2)]
    pick = D4.pick_placement(rows)
    assert pick['s'] == pytest.approx(1.3)           # tie on cost: the smaller scale wins
    assert D4.pick_placement([dict(feasible=False, cost=0.0, s=1.0, shift=0.0)]) is None


def test_pick_on_the_real_scan_grid_meets_the_hand_rule():
    ctx = _fake_ctx()
    rows = D4.scan_rows(ctx)
    pick = D4.pick_placement(rows)
    assert pick is not None
    assert min(pick['hand_L'], pick['hand_R']) >= ctx.ob['far_y'] + D4.FAR_CLEAR - 1e-6
    assert D4.S_GRID.min() >= 0.9 - 1e-9 and D4.S_GRID.max() <= 1.5 + 1e-9
    assert D4.SHIFT_GRID.min() >= 0.0 and D4.SHIFT_GRID.max() <= 0.4 + 1e-9


# ------------------------------------------------------------------------------------------------ the Place mapping
def _fake_sol():
    t = np.round(np.arange(0.2, 1.8 + 1e-9, 1.0 / 60), 6)
    P = np.stack([np.zeros_like(t), 0.9 * (t - 0.45), 0.8 + 0.0 * t], axis=1)
    anchors = dict(stance_xy=np.array([0.0, -0.65]), hand_L_xy=np.array([0.2, 0.96]), hand_R_xy=np.array([-0.4, 0.89]))
    return dict(t=t, P=P, anchors=anchors)


def test_place_maps_forward_travel_with_the_scale_and_the_takeoff_offset():
    sol = _fake_sol()
    pl = D4.Place(sol, s=1.25, y_take=3.0)
    # the WD2 takeoff pelvis is at y = 0 in this fake path, so a point 0.4 m ahead of it maps to y_take + s * 0.4
    p = pl.map_points(np.array([0.1, 0.4, 0.8]))
    assert p[1] == pytest.approx(3.0 + 1.25 * 0.4)
    assert p[0] == pytest.approx(0.1)                             # lateral travel is not scaled
    assert p[2] == pytest.approx(0.8)
    # the hand anchor sits at s times its WD2 offset from the takeoff pelvis, on the ACU hand height
    hl = pl.hand_anchor('L')
    assert hl[1] == pytest.approx(3.0 + 1.25 * 0.96)
    assert hl[2] == pytest.approx(WP.ACU_HAND_Z)
    assert hl[0] == pytest.approx(0.2)                            # lateral travel is not scaled


def test_place_bump_is_a_raised_cosine_on_the_reference_height():
    sol = _fake_sol()
    pl = D4.Place(sol, s=1.0, y_take=3.0, bump_amp=0.07)
    w = np.array([0.35, 0.55, 0.75, 0.9])
    b = pl.bump(w)
    assert b[0] == pytest.approx(0.0) and b[2] == pytest.approx(0.0) and b[3] == pytest.approx(0.0)
    assert b[1] == pytest.approx(0.07)                           # the peak at the middle of the span
    z = pl.pelvis(np.array([0.55]))[0, 2]
    assert z == pytest.approx(0.8 + 0.07)


# ------------------------------------------------------------------------------------------------ played-window step
def test_window_step_counts_only_the_window():
    d = np.zeros((20, 3))
    d[10:] = [0.0, 0.02, 0.0]                  # a 2 cm jump between frames 9 and 10
    assert D4.window_step_cm(d, np.arange(0, 10)) == pytest.approx(0.0)     # the jump is outside the window
    assert D4.window_step_cm(d, np.arange(5, 15)) == pytest.approx(2.0)
    assert D4.window_step_cm(d, np.array([3])) == 0.0


# ------------------------------------------------------------------------------------------------ Exit convergence
def test_exit_offset_converges_within_the_step_limit():
    L = 0.8666667                                # the Dn100cm Exit clip length (capture: 0.866667)
    t = np.arange(0, int(round(L * 60)) + 1) / 60.0
    c = np.where(t >= D4.EXIT_O - 1e-9, 1.0, 0.0) * (1.0 - D4._smoothstep((t - D4.EXIT_O) / (L - D4.EXIT_O)))
    assert c[t >= D4.EXIT_O][0] == pytest.approx(1.0, abs=1e-3)
    assert c[-1] == pytest.approx(0.0)
    assert np.all(np.diff(c[t >= D4.EXIT_O]) <= 1e-12)           # monotone
    off = 0.6645                                 # the SlideOver offset at W in the baked run (z -0.66, y +0.66)
    step_cm = np.max(np.abs(np.diff(c[t >= D4.EXIT_O]))) * off * 100.0     # per 60 Hz frame
    assert step_cm <= 4.0 + 1e-9


def _ramp_step_cm(D, T):
    """Largest change per 60 Hz frame (cm) of a smoothstep ramp of length T (s) and size D (m)."""
    t = np.arange(0.0, T + 1e-9, 1.0 / 60)
    s = D4._smoothstep(np.clip(t / T, 0.0, 1.0))
    return float(np.max(np.diff(s)) * D * 100.0)


def test_ramp_length_needed_for_the_step_limit():
    # a smoothstep of length T moves at most 1.5 D / T per second: 4 cm/frame needs T >= 1.5 D / 2.4 seconds
    D = 0.9
    T_need = 1.5 * D / (0.04 * 60)
    assert _ramp_step_cm(D, 0.15) > 4.0                   # the specified 0.15 s fade breaks the rule
    assert _ramp_step_cm(D, 0.25) > 4.0
    assert _ramp_step_cm(D, T_need * 1.02) <= 4.0 + 0.2   # about 0.56 s is enough (discrete frames)


# ------------------------------------------------------------------------------------------------ mapping and patches
def test_mapping_entry_for_the_gated_exit_has_the_gate_fields():
    e = DC2.mapping_entry('F7919BDF0', D4.EXIT_NAME, 'F7919BDF0.baked.json', False,
                          only_after=[D4.SO_NAME], override=None)
    assert e['prebaked'] is True and e['patch_translation'] is True
    assert e['only_after'] == [D4.SO_NAME] and e['arm_seconds'] == 2.0
    assert 'override' not in e


def test_written_mapping_matches_the_status():
    path = os.path.join(OUT, 'patches', 'mapping_dive4.json')
    if not os.path.exists(path):
        pytest.skip('dive4 outputs not written yet')
    with open(path, encoding='utf-8') as f:
        m = json.load(f)
    by = {c['name']: c for c in m['clips']}
    so, ex = by[D4.SO_NAME], by[D4.EXIT_NAME]
    for c in (so, ex):
        assert c['prebaked'] is True and c['patch_translation'] is True
    assert ex['only_after'] == [D4.SO_NAME] and ex['arm_seconds'] == 2.0
    assert so['handle'] == '0x' + D4.SO_HEX and ex['handle'] == '0x' + D4.EXIT_HEX
    if 'override' not in so:
        assert so['enabled'] is False                   # FAIL: the enable rule does not pass
    if ex['enabled']:
        assert so['enabled'] is True                    # the Exit is enabled only with the SlideOver


def _track_ids(path):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return [str(t['TrackID']) for t in d['Tracks']], d


def test_written_patches_have_no_root_track_and_a_reference_translation():
    for hx in (D4.SO_HEX, D4.EXIT_HEX):
        path = os.path.join(OUT, 'patches', hx + '.baked.json')
        if not os.path.exists(path):
            pytest.skip('dive4 outputs not written yet')
        ids, d = _track_ids(path)
        assert '0' not in ids                        # no TrackID 0 (the ACU root stays)
        assert str(bake2.REF_TID) in ids             # the Reference translation track
        # the Reference carries its rotation (codec 1, infotype 0) and its translation (codec 27, infotype 3) under the
        # same TrackID, as in dive3's patches
        kinds = {(t['codec'], t['infotype']) for t in d['Tracks'] if str(t['TrackID']) == str(bake2.REF_TID)}
        assert (27, 3) in kinds and (1, 0) in kinds
        assert str(zlib.crc32(b'Reference')) == str(bake2.REF_TID)
