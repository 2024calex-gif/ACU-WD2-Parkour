"""dive6 (T19): the ground-contact residual (two-sided on the contact window, one-sided box clearance), the light lift
smoothing, the calibration helpers, and the written mapping and patches."""
import json
import os

import numpy as np
import pytest

import bake2
import config
import dive4 as D4
import dive6 as S6

OUT = os.path.join(config.OUT, 'dive6')


def _sample(lowest, contact=True, box_need=0.0, surf=0.0, tau=(0.5,)):
    n = len(tau)
    return dict(tag='E', tau=np.array(tau, float), lowest=np.array(lowest, float), surf=np.full(n, surf),
                box_need=np.full(n, box_need), contact=np.full(n, contact))


def test_contact_residual_keeps_the_lowering():
    avg, worst = S6.residual_from_samples([_sample([0.20])], 100)
    k = int(round(0.5 * 60))
    assert avg[k] == pytest.approx(0.01 - 0.20)               # negative: the body must come down
    assert worst == pytest.approx(0.19)


def test_contact_residual_lifts_a_point_below_the_ground():
    avg, _ = S6.residual_from_samples([_sample([-0.05])], 100)
    assert avg[int(round(0.5 * 60))] == pytest.approx(0.06)


def test_box_clearance_is_one_sided_and_wins_where_needed():
    avg, _ = S6.residual_from_samples([_sample([0.20], box_need=0.03)], 100)
    assert avg[int(round(0.5 * 60))] == pytest.approx(0.03)   # the clearance requirement overrides the lowering


def test_outside_the_contact_window_only_lifts():
    avg, _ = S6.residual_from_samples([_sample([0.20], contact=False)], 100)
    assert avg[int(round(0.5 * 60))] == 0.0                   # no lowering before the plant


def test_surface_under_the_lowest_point_is_the_box_top_inside_the_band():
    avg, _ = S6.residual_from_samples([_sample([1.05], surf=1.0)], 100)
    assert avg[int(round(0.5 * 60))] == pytest.approx(0.01 + 1.0 - 1.05)


def test_two_sided_smoothing_keeps_the_sign():
    x = np.zeros(100)
    x[40:50] = -0.2
    y = bake2._gauss_smooth(x, S6.CONTACT_SIGMA_LIFT)
    assert y[45] < 0.0 and np.all(y[:30] == pytest.approx(0.0, abs=1e-12))


def test_contact_target_and_tolerance_match_the_brief():
    assert S6.CONTACT_TARGET == pytest.approx(0.01)
    assert S6.CONTACT_TOL == pytest.approx(0.03)
    assert S6.WD2_UPRIGHT == pytest.approx(1.5)


def test_written_mapping_enables_both_with_the_roll_on_ground_override():
    path = os.path.join(OUT, 'patches', 'mapping_dive6.json')
    if not os.path.exists(path):
        pytest.skip('dive6 outputs not written yet')
    with open(path, encoding='utf-8') as f:
        m = json.load(f)
    by = {c['name']: c for c in m['clips']}
    so, ex = by[S6.F5.SO_NAME], by[S6.F5.EXIT_NAME]
    for c in (so, ex):
        assert c['enabled'] is True
        assert c['override'] == 'preview: roll on ground'
    assert ex['only_after'] == [S6.F5.SO_NAME] and ex['arm_seconds'] == 2.0
    assert 'only_after' not in so


def test_written_patches_have_no_root_track():
    import zlib
    for hx in (S6.F5.SO_HEX, S6.F5.EXIT_HEX):
        path = os.path.join(OUT, 'patches', hx + '.baked.json')
        if not os.path.exists(path):
            pytest.skip('dive6 outputs not written yet')
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        ids = [str(t['TrackID']) for t in d['Tracks']]
        assert '0' not in ids
        assert str(zlib.crc32(b'Reference')) in ids
