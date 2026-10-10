"""transition_report.py: speed / jerk / angular-speed math with variable frame dt, and the synthetic-capture pipeline."""
import math

import numpy as np
import pytest

import capture
import synth_capture as sc
import transition_report as tr

FRAME = 1.0 / 60.0


def _irregular_t(n=120, seed=3):
    """Frame times with 40-70 ms steps, like the captures (fixed seed)."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.uniform(0.040, 0.070, size=n))


def test_velocity_exact_for_linear_motion_with_irregular_dt():
    t = _irregular_t()
    v = np.array([2.0, -1.5, 0.5])
    P = np.outer(t, v) + np.array([10.0, 3.0, 1.0])
    tm, V = tr.velocity(t, P)
    assert np.allclose(V, v, atol=1e-9)
    tm2, s = tr.speed(t, P)
    assert np.allclose(s, np.linalg.norm(v), atol=1e-9)
    assert np.allclose(tm, tm2)
    assert np.allclose(tm, 0.5 * (t[1:] + t[:-1]))


def test_jerk_is_zero_for_constant_velocity_and_exact_for_constant_acceleration():
    t = _irregular_t()
    P_const = np.outer(t, [1.0, 0.0, 0.0])
    _, j0 = tr.jerk(t, P_const)
    assert np.allclose(j0, 0.0, atol=1e-9)

    a = np.array([0.0, 120.0, 0.0])            # m/s^2
    P = 0.5 * np.outer(t * t, a)               # x = a t^2 / 2 -> velocity linear in time
    _, j = tr.jerk(t, P)
    # acceleration is exact on irregular frames: |a| / 60 is the per-frame velocity change
    assert np.allclose(j, np.linalg.norm(a) * FRAME, rtol=1e-9)


def test_jerk_scales_with_frame_rate_not_with_dt():
    """The same physical motion sampled at 60 Hz or at 30 Hz gives the same m/s-per-60-Hz-frame score."""
    a = np.array([50.0, 0.0, 0.0])
    t60 = np.arange(0, 1.0, FRAME)
    t30 = np.arange(0, 1.0, 2 * FRAME)
    for t in (t60, t30):
        P = 0.5 * np.outer(t * t, a)
        _, j = tr.jerk(t, P)
        assert np.allclose(j, 50.0 * FRAME, rtol=1e-9)


def test_angular_speed_of_constant_spin_with_irregular_dt():
    t = _irregular_t()
    rate = math.radians(90.0)
    ang = rate * t
    Q = np.column_stack([np.zeros_like(t), np.zeros_like(t), np.sin(ang / 2), np.cos(ang / 2)])  # xyzw about Z
    tm, w = tr.angular_speed(t, Q)
    assert np.allclose(w, 90.0, rtol=1e-9)
    assert np.allclose(tm, 0.5 * (t[1:] + t[:-1]))


def test_window_max_returns_value_time_and_count():
    tt = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    vals = np.array([1.0, 5.0, 2.0, 7.0, 3.0])
    mx, tmx, n = tr.window_max(tt, vals, 0.05, 0.35)
    assert mx == 7.0 and tmx == pytest.approx(0.3) and n == 3
    mx, tmx, n = tr.window_max(tt, vals, 1.0, 2.0)
    assert np.isnan(mx) and np.isnan(tmx) and n == 0


def test_boundary_step_separates_a_cut_from_ordinary_motion():
    t = np.arange(40) * FRAME
    P = np.outer(t, [4.0, 0.0, 0.0])                 # 4 m/s, continuous
    smooth = tr.boundary_step(t, P, t[20])
    assert smooth['ratio'] == pytest.approx(1.0, rel=1e-6)
    Pj = P.copy()
    Pj[20:] += np.array([0.30, 0.0, 0.0])            # jump of 30 cm at the cut
    jump = tr.boundary_step(t, Pj, t[20])
    assert jump['step'] == pytest.approx(0.30 + 4.0 * FRAME, rel=1e-6)
    assert jump['ratio'] > 5.0
    assert tr.boundary_step(t, P, t[0]) is None      # too close to the start


def test_control_median_skips_windows_that_touch_a_clip_start():
    t = np.arange(0, 3.0, FRAME)
    vals = np.ones_like(t)
    vals[(t > 1.0) & (t < 1.2)] = 100.0               # a spike that only sits inside the excluded area
    clip_starts = [1.1]
    med, n = tr.control_median_max(t, vals, clip_starts)
    assert med == pytest.approx(1.0)
    assert n > 0
    med_all, n_all = tr.control_median_max(t, vals, [])
    assert n_all > n
    assert med_all == pytest.approx(1.0)              # the spike is inside the window maxima of some starts
    med_none, n_none = tr.control_median_max(t, vals, [t[-1]])
    assert n_none > 0


def test_write_report_keeps_the_findings_block(tmp_path):
    path = str(tmp_path / 'report.md')
    tr.write_report(path, ['# head', '', '## Findings', ''])
    text = open(path, encoding='utf-8').read()
    assert tr.FINDINGS_BEGIN in text and 'findings not written yet' in text
    kept = tr.FINDINGS_BEGIN + '\nhand written\n' + tr.FINDINGS_END
    text = text.replace(tr.FINDINGS_BEGIN + '\n(findings not written yet)\n' + tr.FINDINGS_END, kept)
    open(path, 'w', encoding='utf-8').write(text)
    tr.write_report(path, ['# head v2', '', '## Findings', ''])
    again = open(path, encoding='utf-8').read()
    assert '# head v2' in again and 'hand written' in again and 'not written yet' not in again


@pytest.fixture
def synth(tmp_path):
    p = str(tmp_path / 'vault_synth.json')
    truth = sc.write(p, convention='rows', heading_deg=30.0)
    cap = capture.load(p)
    occs = tr.occurrences_of(cap, p, 'session_test', {})
    assert len(occs) == 1
    occ = occs[0]
    tr.analyse(cap, occ)
    return cap, occ, truth


def test_synthetic_capture_occurrence_and_windows(synth):
    cap, occ, truth = synth
    assert occ.name == sc.CLIP_SKIP
    assert occ.t_in == pytest.approx(sc.T_SKIP)
    assert occ.t_out == pytest.approx(sc.T_MURAILLE)
    assert occ.pose_ok is True and occ.patched is False
    assert occ.truncated == {'in': False, 'out': False}


def test_synthetic_rigid_bones_have_no_speed_or_jerk_in_the_in_window(synth):
    _, occ, _ = synth
    w = occ.win['in']
    for b in ('Hips', 'Head'):
        assert w['jerk'][b][0] < 1e-6, b
    assert w['speed']['Hips'][0] < 1e-6
    assert w['ang']['Spine2'][0] < 1e-6 and w['ang']['Head'][0] < 1e-6


def test_synthetic_hands_move_only_in_the_out_window(synth):
    _, occ, _ = synth
    assert occ.win['in']['jerk']['LeftHand'][0] < 1e-6         # hands are raised and still in the in-window
    mx, rel, n = occ.win['out']['jerk']['LeftHand']
    assert mx > 0.1
    assert -0.2 <= rel <= 0.05                                  # ramp edges at 1.25 s and 1.35 s, t_out = 1.4 s


def test_synthetic_root_speed_is_the_entity_speed(synth):
    _, occ, _ = synth
    mx, _rel, _n = occ.win['in']['root_speed']
    assert mx == pytest.approx(sc.SPEED, rel=1e-4)


def test_synthetic_boundary_at_t_in_has_no_jump_for_hips(synth):
    _, occ, _ = synth
    hips = occ.bnd['in']['Hips']
    assert hips is not None and hips['step'] < 1e-9     # rigid bone: no step (the ratio is undefined at zero rate)
    hands = occ.bnd['out']['LeftHand']
    assert hands is not None and hands['step'] > 0.0    # the hands move at the out cut


def test_report_renders_without_plots(synth, tmp_path):
    cap, occ, _ = synth
    summary = {'vault_synth.json': dict(session='session_test', pose_ok=True, n_frames=len(cap.t),
                                        n_skip=1, t_span=float(cap.t[-1] - cap.t[0]), timing_r=float('nan'))}
    lines = tr.report_md(summary, [occ], [])
    text = '\n'.join(lines)
    assert '## 2. Window maxima' in text and 'Fd100cm-Up100cm' in text
    assert '## 6. Patched clips' in text and '(no patched SkipOver)' in text
