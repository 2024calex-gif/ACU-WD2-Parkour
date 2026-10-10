"""dive4b (T16, round 2): the pure parts (delayed-takeoff time map, joined played sampling, world smoothness metric, the
ground-floor envelope, the spec judgement with its narrow fail, the scan and the choices) and the written outputs."""
import json
import os
import types

import numpy as np
import pytest

import bake2
import config
import dive4b as B
import dive4 as D4
import wd2_path as WP

OUT = os.path.join(config.OUT, 'dive4b')


# ------------------------------------------------------------------------------------------------ time map
def test_timing_maps_the_delayed_takeoff_to_wd2_0p45():
    tm = B.Timing(0.65, 1.0)
    assert float(tm.w(0.65)) == pytest.approx(0.45)
    assert float(tm.w(0.65 + 0.30)) == pytest.approx(0.75)        # plant start, unit speed
    assert float(tm.w(tm.t_plant_end)) == pytest.approx(0.92)     # plant end
    assert float(tm.t_feet) == pytest.approx(0.65 + 0.75)         # feet-down at k = 1


def test_timing_run_up_is_clamped_and_continuous_at_the_takeoff():
    tm = B.Timing(0.70, 1.0)
    t = np.array([0.42, 0.50, 0.60, 0.699, 0.7001, 0.701])
    w = tm.w(t)
    assert w[0] == pytest.approx(0.2)                        # clamped at WD2 0.2 s
    assert np.all(w[:4] >= 0.2 - 1e-12)
    assert np.allclose(np.diff(w)[3:], np.diff(t)[3:], atol=1e-9)   # unit speed across the takeoff (no jump)
    assert np.all(np.diff(tm.w(np.linspace(0.3, 1.5, 500))) >= -1e-12)  # monotone


def test_roll_speed_k_moves_feet_down_earlier():
    slow = B.Timing(0.65, 1.0)
    fast = B.Timing(0.65, 1.3)
    assert fast.t_feet < slow.t_feet
    assert float(fast.w(fast.t_feet)) == pytest.approx(1.2)
    assert float(fast.w(fast.t_plant_end)) == pytest.approx(0.92)


# ------------------------------------------------------------------------------------------------ joined sampling
def test_joined_played_is_uniform_and_crosses_the_seam_without_duplicates():
    class C:
        pass
    ctx = C()
    ctx.t_W = 1.0827866
    ctx.span_e = 0.220362
    n = 200
    PS = np.zeros((n, 1, 3))
    PE = np.zeros((n, 1, 3))
    PS[:, 0, 0] = np.arange(n) / 60.0                       # pelvis-like: x = clip time
    # the Exit's x continues from the SlideOver at W: x = W + (t - 0.208) / 0.9665 (played time)
    PE[:, 0, 0] = ctx.t_W + (np.arange(n) / 60.0 - 0.208) / 0.9665
    win_S = np.arange(26, 65)
    win_E = np.arange(13, 26)
    J = B.joined_played(ctx, PS, PE, 0, win_S, win_E)
    steps = np.diff(J[:, 0])
    assert np.all(steps > 0)                                 # no duplicated samples
    # the played time advances by 1/60 s per sample on the S side; the Exit side advances at 1/(60 * 0.9665)
    assert np.allclose(steps[:5], 1.0 / 60.0, atol=1e-9)


def test_joined_played_exit_interpolates_patched_frames_only():
    class C:
        pass
    ctx = C()
    ctx.t_W = 1.0
    ctx.span_e = 0.2
    PS = np.zeros((90, 1, 3))
    PE = np.zeros((90, 1, 3))
    PE[:, 0, 0] = np.arange(90, dtype=float)                # Exit frame index as the value
    win_S = np.arange(40, 61)
    win_E = np.arange(13, 30)
    J = B.joined_played(ctx, PS, PE, 0, win_S, win_E)
    exit_vals = J[len(win_S):, 0]
    assert np.all(exit_vals >= 13.0 - 1e-9)                  # never samples the un-patched frames 0..12


# ------------------------------------------------------------------------------------------------ world smoothness
def _metric_from_path(P):
    d1 = np.diff(P, axis=0)
    d2 = np.diff(P, 2, axis=0)
    return dict(step_cm=float(np.linalg.norm(d1, axis=1).max() * 100.0),
                acc_cm=float(np.linalg.norm(d2, axis=1).max() * 100.0),
                dv_ms=float(np.linalg.norm(d2, axis=1).max() * 60.0))


def test_world_dv_is_acceleration_times_frame_rate():
    # a constant acceleration of a per frame^2 gives a second difference of a, so dv = a * 60 m/s per frame
    a = 0.025                                                # 2.5 cm/frame^2 = 1.5 m/s per frame
    f = np.arange(30, dtype=float)
    P = np.stack([0.5 * a * f ** 2, np.zeros_like(f), np.zeros_like(f)], axis=1)
    m = _metric_from_path(P)
    assert m['acc_cm'] == pytest.approx(2.5, abs=1e-6)
    assert m['dv_ms'] == pytest.approx(1.5, abs=1e-6)


def test_spec_dv_limit_matches_the_brief():
    assert B.DV_MAX == pytest.approx(1.5)                    # m/s per frame
    assert B.DV_NARROW == pytest.approx(2.5)
    assert B.STEP_MAX == pytest.approx(12.0)                 # cm per frame (about 7 m/s)
    assert B.OFF_MAX == pytest.approx(1.0)


# ------------------------------------------------------------------------------------------------ ground floor
def test_floor_envelope_covers_the_need_and_only_rises():
    need_idx = np.array([10, 11, 30])
    need = np.array([0.02, 0.01, 0.05])
    lift = B.envelope_update(np.zeros(B.LIFT_K), need_idx / 60.0, need)
    assert np.all(lift >= 0.0)
    for k, n in zip(need_idx, need):
        assert lift[k] >= n - 1e-12
    assert np.max(np.abs(np.diff(lift))) < 0.05              # smooth (rise and fall limited)


def test_floor_envelope_with_no_need_is_zero():
    lift = B.envelope_update(np.zeros(B.LIFT_K), np.array([0.5]), np.array([0.0]))
    assert np.allclose(lift, 0.0)


def test_lift_eval_is_zero_outside_the_grid():
    lk = np.zeros(B.LIFT_K)
    lk[30] = 0.1
    assert float(B.lift_eval(lk, 0.5)) == pytest.approx(0.1)
    assert float(B.lift_eval(lk, 5.0)) == 0.0


# ------------------------------------------------------------------------------------------------ judgement
def _res(pen_box=0.0, pen_ground=0.0, touch=10, dv=1.0, step=5.0, off=0.2, cont=10.0):
    met = dict(pen_patch_cm=max(pen_box, pen_ground), pen_box_cm=pen_box, pen_ground_cm=pen_ground,
               touch=dict(min=touch))
    world = dict(dv_ms=dv, step_cm=step, offset_max_m=off)
    return dict(met_S=met, met_E=dict(met), world=world, cont=dict(max_deg=cont, worst='x'))


def test_judge_passes_inside_all_limits():
    st = B.judge(_res())
    assert st['S']['status'] == 'PASS' and st['E']['status'] == 'PASS'
    assert st['S']['enabled'] and st['E']['enabled']


def test_judge_narrow_fail_is_an_override_with_the_reason():
    st = B.judge(_res(pen_ground=8.0))
    assert st['S']['status'] == 'OVERRIDE'
    assert st['S']['override'].startswith('preview: ')
    assert st['S']['enabled']


def test_judge_world_step_is_not_overridable():
    st = B.judge(_res(step=13.0))
    assert st['S']['status'] == 'FAIL' and not st['S']['enabled']
    assert not st['E']['enabled']                             # the Exit needs the SlideOver


def test_judge_dv_narrow_limit_is_2p5():
    assert B.judge(_res(dv=2.4))['S']['status'] == 'OVERRIDE'
    assert B.judge(_res(dv=2.6))['S']['status'] == 'FAIL'


def test_judge_touch_narrow_limit_and_exit_exemption():
    assert B.judge(_res(touch=2))['S']['status'] == 'OVERRIDE'   # 2 frames: the narrow limit
    assert B.judge(_res(touch=1))['S']['status'] == 'FAIL'       # 1 frame: beyond the narrow limit
    st = B.judge(_res(touch=1))
    assert st['E']['status'] == 'PASS'                           # the Exit has no hands to judge


# ------------------------------------------------------------------------------------------------ scan and picks
def _ctx_for_scan():
    return types.SimpleNamespace(hand_rel_y={'L': 0.957, 'R': 0.889}, orig_W=np.array([0.0, 5.456, 0.714]),
                                 t_W=1.0828, span_e=0.2204, hips_orig_so=np.column_stack([
                                     np.zeros(200), np.linspace(0.0, 10.0, 200), np.zeros(200)]),
                                 sol=None, p0=None)


def test_scan_feasibility_is_the_closer_hand_rule(monkeypatch):
    ctx = _ctx_for_scan()
    monkeypatch.setattr(B, 'path_y_rel', lambda c, w: (w - 0.45) * 2.0)
    rows = B.scan_rows(ctx, T_grid=np.array([0.6, 0.7]), s_grid=np.array([0.9, 1.4]), shift_grid=np.array([-0.2, 0.3]))
    for r in rows:
        ok = min(r['hand_L'], r['hand_R']) >= B.HAND_MIN - 1e-9
        assert r['feasible'] == ok or abs(min(r['hand_L'], r['hand_R']) - B.HAND_MIN) < 1e-9


def test_top_choices_are_unique_and_sorted_by_cost():
    rows = [dict(T_to=0.6, s=1.0, shift=0.0, cost=0.3, feasible=True),
            dict(T_to=0.6, s=1.0, shift=0.0, cost=0.3, feasible=True),
            dict(T_to=0.7, s=1.1, shift=0.0, cost=0.1, feasible=True),
            dict(T_to=0.8, s=1.2, shift=0.0, cost=0.05, feasible=False),
            dict(T_to=0.75, s=1.0, shift=0.1, cost=0.2, feasible=True)]
    top = B.top_choices(rows, 5)
    costs = [r['cost'] for r in top]
    assert costs == sorted(costs)
    assert len(top) == 3                                     # the duplicate and the infeasible row are dropped


# ------------------------------------------------------------------------------------------------ outputs
def test_written_mapping_matches_the_status():
    path = os.path.join(OUT, 'patches', 'mapping_dive4b.json')
    if not os.path.exists(path):
        pytest.skip('dive4b outputs not written yet')
    with open(path, encoding='utf-8') as f:
        m = json.load(f)
    by = {c['name']: c for c in m['clips']}
    so, ex = by[D4.SO_NAME], by[D4.EXIT_NAME]
    for c in (so, ex):
        assert c['prebaked'] is True and c['patch_translation'] is True
    assert ex['only_after'] == [D4.SO_NAME] and ex['arm_seconds'] == 2.0
    assert so['handle'] == '0x' + D4.SO_HEX and ex['handle'] == '0x' + D4.EXIT_HEX
    if ex['enabled']:
        assert so['enabled'] is True


def test_written_patches_have_no_root_track():
    for hx in (D4.SO_HEX, D4.EXIT_HEX):
        path = os.path.join(OUT, 'patches', hx + '.baked.json')
        if not os.path.exists(path):
            pytest.skip('dive4b outputs not written yet')
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        ids = [str(t['TrackID']) for t in d['Tracks']]
        assert '0' not in ids
        kinds = {(t['codec'], t['infotype']) for t in d['Tracks'] if str(t['TrackID']) == str(bake2.REF_TID)}
        assert (27, 3) in kinds                              # the Reference translation
