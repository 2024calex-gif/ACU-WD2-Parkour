"""Tests for slideover_model.py: clip-time mapping, crossings, WD2 dive placement, occurrence wiring, clip summary, the
partial-pose fit (matchclip.partial), and one real SlideOver dump."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

import capture
import dumps
import matchclip as mc
import slideover_model as sm

WD2 = dict(takeoff=0.45, hands_start=0.75, hands_end=0.92, roll_start=0.92, roll_end=1.18, feet_down=1.2,
           run_out=1.6)


# ---------------------------------------------------------------------------------------------- names and mapping

def test_family_picks_sprint_slideover_and_landroll_only():
    assert sm.family('xx_Parkour_Sprint_SlideOver_Fd100cm-Up100cm') == 'slide'
    assert sm.family('xx_Parkour_LandingFwd_LandRoll_tr_Run_Impultion') == 'roll'
    assert sm.family('xx_Parkour_Idle_SlideOver_Fd100cm-Up100cm') is None
    assert sm.family('xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm') is None
    assert sm.family(None) is None


def test_played_time_and_clip_time_are_inverse():
    for o, r in [(0.0, 1.0), (0.38, 1.0), (1.38, 1.55), (0.2, 0.8)]:
        for x in [0.0, 0.5, 1.2, 3.0]:
            assert sm.clip_time(sm.played_time(x, o, r), o, r) == pytest.approx(x)
    assert sm.played_time(0.7, 0.7, 1.3) == pytest.approx(0.0)
    assert sm.played_time(1.7, 0.7, 2.0) == pytest.approx(0.5)


def test_first_crossing_interpolates_and_handles_edges():
    t = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    v = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert sm.first_crossing(t, v, 2.5) == pytest.approx(0.25)
    assert sm.first_crossing(t, v, -1.0) == pytest.approx(0.0)      # already above at the first sample
    assert sm.first_crossing(t, v, 9.0) is None                     # never reached
    assert sm.first_crossing(t, v, 4.0) == pytest.approx(0.4)


def test_next_clip_any_returns_the_first_later_clip_of_any_name():
    clips = [dict(t=1.0, name='a'), dict(t=1.5, name=None), dict(t=2.2, name='c')]
    cap = SimpleNamespace(clips=clips)
    assert sm.next_clip_any(cap, 0) == (None, 1.5)
    assert sm.next_clip_any(cap, 1) == ('c', 2.2)
    assert sm.next_clip_any(cap, 2) == (None, None)


# ---------------------------------------------------------------------------------------------- WD2 dive

def test_dive_on_played_aligns_the_takeoff():
    ev = sm.dive_on_played(0.3, WD2)
    assert ev['takeoff'] == pytest.approx(0.3)
    assert ev['feet_down'] == pytest.approx(0.3 + 0.75)
    assert ev['run_out'] == pytest.approx(0.3 + 1.15)
    assert ev['roll_end'] == pytest.approx(0.3 + 0.73)


def test_dive_check_spill_and_blend():
    chk = sm.dive_check(0.3, 1.0, WD2)                 # feet down at 1.05, roll end 1.03, run-out 1.45
    assert chk['spill_feet_down'] == pytest.approx(0.05)
    assert chk['spill_roll_end'] == pytest.approx(0.03)
    assert chk['spill_run_out'] == pytest.approx(0.45)
    assert chk['roll_inside'] is False
    assert chk['fits_window'] is False
    assert chk['fits_with_blend'] is True              # 1.05 <= 1.0 + 0.25
    assert chk['takeoff_played_ok'] is True


def test_dive_check_inside_window_and_takeoff_before_trigger():
    inside = sm.dive_check(0.0, 1.5, WD2)
    assert inside['fits_window'] and inside['roll_inside']
    assert inside['spill_feet_down'] == 0.0 and inside['spill_run_out'] == 0.0
    early = sm.dive_check(-0.1, 1.5, WD2)              # ACU takeoff happens before the played window starts
    assert early['takeoff_played_ok'] is False


def test_scale_to_wd2_is_the_span_ratio():
    assert sm.scale_to_wd2(1.5, WD2) == pytest.approx(2.0)     # WD2 span is 0.75 s
    assert sm.scale_to_wd2(0.75, WD2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------- occurrences

HANDLE_SKIP = '0x296F0F8B8'   # xx_Parkour_Sprint_SkipOver_Fd200cm-Up100cm (used by the synthetic capture)


def _synthetic_pair():
    import test_matchclip as tm
    cap, dump, truth = tm._synthetic(o=0.3, r=1.1, yaw=0.0)
    cap.clips[0]['name'] = 'xx_Parkour_Sprint_SlideOver_Fd200cm-Up100cm'
    cap.clips[0]['patched'] = False
    return cap, dump, truth


def test_partial_pose_fit_recovers_time_map_without_one_bone():
    cap, dump, truth = _synthetic_pair()
    cap.pose_ok = False
    cap.bones['RightHand'] = None
    w = mc.match_window(cap, 0)
    with pytest.raises(ValueError):
        mc.fit_time_map(cap, dump, w)                  # the matchclip rule: no complete pose, no fit
    fit = mc.fit_time_map(cap, dump, w, partial=True)
    assert 'RightHand' not in fit['bones'] and 'LeftHand' in fit['bones']
    assert fit['o'] == pytest.approx(truth['o'], abs=0.02)
    assert fit['r'] == pytest.approx(truth['r'], abs=0.02)
    assert fit['rms_cm'] < 0.5


def test_analyse_occurrence_wires_window_next_and_fit():
    cap, dump, truth = _synthetic_pair()
    row = sm.analyse_occurrence(cap, 0, dump, path='cap_x_1.json')
    assert row['family'] == 'slide'
    assert row['next_name'] is not None and row['window_s'] == pytest.approx(1.2 - cap.clips[0]['t'])
    assert row['usable'] is True
    assert row['o'] == pytest.approx(truth['o'], abs=0.02)
    assert row['r'] == pytest.approx(truth['r'], abs=0.02)


def test_analyse_occurrence_without_dump_or_pose():
    cap, dump, _ = _synthetic_pair()
    row = sm.analyse_occurrence(cap, 0, None, path='cap_x_1.json')
    assert row['skip'] == 'no original dump' and row['o'] is None and row['usable'] is False
    cap.bones['Hips'] = None
    row = sm.analyse_occurrence(cap, 0, dump, path='cap_x_1.json')
    assert row['skip'] == 'no entity-space Hips (no pose)'


# ---------------------------------------------------------------------------------------------- clip summary

def _row(clip, o, r, rms, usable, window=1.0, next_name='xx_Parkour_Muraille_Exit_Fd100cm-Dn100cm'):
    return dict(clip=clip, o=o, r=r, rms=rms, usable=usable, window_s=window, next_name=next_name,
                next_any=next_name, end_reason='next clip', skip=None)


def _geo(height=1.0, depth=1.0):
    return dict(height=height, depth=depth, near_y=4.0, far_y=4.0 + depth, landing_y=5.0, landing_z=0.0,
                length_s=2.0, root_end_y=8.0, root_z_max=height, root_cross_near_s=0.6,
                root_cross_far_s=1.0, root_cross_landing_s=1.1, takeoff_s=0.45, hand_plant_s=0.9,
                foot_on_top_s=0.95, feet_down_s=1.15, near_foot_s=0.55, far_foot_s=0.8, takeoff_foot='LeftFoot',
                feet_down_foot='LeftFoot', hand_plant_hand='LeftHand', hand_top_contacts={},
                root_t=[], root_y=[], root_z=[])


def test_clip_summary_uses_usable_fits_and_reports_the_dive():
    name = 'xx_Parkour_Sprint_SlideOver_Fd100cm-Up100cm'
    rows = [_row(name, 0.4, 1.0, 3.0, True), _row(name, 0.6, 1.2, 4.0, True),
            _row(name, 1.5, 0.5, 60.0, False, window=0.8), _row(name, None, None, None, False, window=0.7)]
    e = sm.clip_summary(name, rows, _geo(), WD2, handle='ABC')
    assert e['n'] == 4 and e['n_fit'] == 3 and e['n_usable'] == 2
    assert e['o'] == pytest.approx(0.5) and e['r'] == pytest.approx(1.1)
    assert e['map_used'] == dict(o=pytest.approx(0.5), r=pytest.approx(1.1), source='own')
    assert e['window'] == pytest.approx(0.9)      # median of 1.0, 1.0, 0.8, 0.7
    assert e['acu_played_s']['takeoff'] == pytest.approx((0.45 - 0.5) / 1.1, abs=1e-3)
    assert e['acu_takeoff_to_feet_clip_s'] == pytest.approx(0.7)
    assert e['wd2_scale_to_acu_span'] == pytest.approx((0.7 / 1.1) / 0.75, abs=1e-3)
    d = e['wd2_dive']
    assert d['takeoff_played'] == pytest.approx(e['acu_played_s']['takeoff'])
    assert d['events_played']['feet_down'] == pytest.approx(d['takeoff_played'] + 0.75, abs=1e-3)
    assert d['hands_on_ground_vs_acu_hand_top_s'] == pytest.approx((0.9 - 0.45) / 1.1, abs=1e-3)


def test_clip_summary_without_usable_fit_uses_the_flagged_reference_map():
    name = 'xx_Parkour_Sprint_SlideOver_Fd200cm-Up200cm'
    rows = [_row(name, 1.6, 0.9, 40.0, False, window=0.8)]
    e = sm.clip_summary(name, rows, _geo(), WD2, handle=None, ref=(0.42, 1.05))
    assert e['o'] is None and e['n_fit'] == 1 and e['n_usable'] == 0
    assert e['map_used']['o'] == pytest.approx(0.42) and e['map_used']['source'] == 'assumed (reference clip map)'
    assert e['acu_played_s']['takeoff'] == pytest.approx((0.45 - 0.42) / 1.05, abs=1e-3)
    e = sm.clip_summary(name, rows, None, WD2, handle=None, ref=None)
    assert e['map_used'] == dict(o=0.0, r=1.0, source='assumed (o = 0, r = 1)')
    assert 'acu_played_s' not in e


def test_clip_summary_windows_count_only_measured_ends():
    name = 'xx_Parkour_Sprint_SlideOver_Fd100cm-Up100cm'
    rows = [_row(name, 0.4, 1.0, 3.0, True, window=0.75),
            dict(_row(name, 0.4, 1.0, 3.0, True, window=1.6), end_reason='clip length')]
    e = sm.clip_summary(name, rows, None, WD2, handle=None, ref=None)
    assert e['n_window_measured'] == 1 and e['window'] == pytest.approx(0.75)
    assert e['window_censored'] == [1.6]


def test_dive_factor_matches_feet_and_keeps_the_roll_inside_the_window():
    # ACU span 0.67 s (feet match needs k = 0.89); window 0.744 with takeoff at 0.025: the roll (0.73 s WD2) must end
    # by 0.744, which allows k up to (0.744 - 0.025) / 0.73 = 0.985. So k = 0.89.
    fac = sm.dive_factor(0.025, 0.744, 0.669, WD2)
    assert fac['k_acu'] == pytest.approx(0.892, abs=1e-3)
    assert fac['k_roll'] == pytest.approx(0.985, abs=1e-3)
    assert fac['k'] == pytest.approx(fac['k_acu'])
    # a long ACU span (1.5 s) is capped by the roll room
    fac = sm.dive_factor(0.065, 1.294, 1.151, WD2)
    assert fac['k'] == pytest.approx(min(fac['k_acu'], fac['k_roll']))
    assert fac['k_roll'] > fac['k_acu']
    # takeoff after the window: no room
    assert sm.dive_factor(0.9, 0.5, 0.7, WD2)['k'] is None or sm.dive_factor(0.9, 0.5, 0.7, WD2)['k_roll'] is None


def test_dive_check_k_scales_the_events_about_the_takeoff():
    chk = sm.dive_check_k(0.025, 0.744, WD2, 0.892)
    ev = chk['events']
    assert ev['takeoff'] == pytest.approx(0.025)
    assert ev['feet_down'] == pytest.approx(0.025 + 0.75 * 0.892, abs=1e-3)     # ~ the ACU feet down (0.695)
    assert ev['roll_end'] == pytest.approx(0.025 + 0.73 * 0.892, abs=1e-3)
    assert chk['spill_roll_end'] == 0.0 and chk['roll_inside'] is True
    assert chk['fits_window'] is True
    # k = 1 is the plain check
    assert sm.dive_check_k(0.3, 1.0, WD2, 1.0)['events']['feet_down'] == pytest.approx(
        sm.dive_check(0.3, 1.0, WD2)['events']['feet_down'])


# ---------------------------------------------------------------------------------------------- real dump

@pytest.mark.skipif(dumps.load_by_name('xx_Parkour_Sprint_SlideOver_Fd100cm-Up100cm') is None,
                    reason='SlideOver dump not present')
def test_dump_geometry_of_a_real_slideover_clip():
    dump = dumps.load_by_name('xx_Parkour_Sprint_SlideOver_Fd100cm-Up100cm')
    geo = sm.dump_geometry(dump)
    assert geo['height'] == pytest.approx(1.0, abs=0.01)
    assert geo['depth'] == pytest.approx(1.0, abs=0.01)
    assert 0.4 < geo['root_cross_near_s'] < 0.9
    assert geo['root_cross_near_s'] < geo['root_cross_far_s'] < geo['length_s']
    assert 0.3 < geo['takeoff_s'] < geo['feet_down_s'] < 1.6
    assert len(geo['root_t']) == len(geo['root_y']) == len(geo['root_z'])
    assert max(geo['root_z']) == pytest.approx(geo['root_z_max'], abs=0.05)
