"""Tests for exit_model.py: Exit chains, predecessor families, WD2 placement, ground touch, the entity pair and the time
budget on synthetic captures, and the Exit dump geometry on one real dump."""
from types import SimpleNamespace

import numpy as np
import pytest

import dumps
import exit_model as em
import slideover_model as sm

SLIDE = 'xx_Parkour_Sprint_SlideOver_Fd100cm-Up100cm'
DN500 = 'xx_Parkour_Muraille_Exit_Fd100cm-Dn500cm'
DN100 = 'xx_Parkour_Muraille_Exit_Fd100cm-Dn100cm'
NAV = 'xx_Nav_Run_3-Loop'
WD2 = dict(takeoff=0.45, hands_start=0.75, hands_end=0.92, roll_start=0.92, roll_end=1.18, feet_down=1.2,
           run_out=1.6)


def _cap(clips, t_end=5.0, entity_z=None, entity_y=None):
    t = np.arange(0.0, t_end + 1e-9, 0.05)
    z = np.zeros_like(t) if entity_z is None else entity_z(t)
    y = np.zeros_like(t) if entity_y is None else entity_y(t)
    pos = np.stack([np.zeros_like(t), y, z], axis=1)
    return SimpleNamespace(clips=[dict(t=c[0], name=c[1]) for c in clips], t=t, entity_pos=pos)


# ---------------------------------------------------------------------------------------------- chains

def test_chain_groups_the_1ms_pair_and_takes_the_last_member_as_played():
    cap = _cap([(1.0, SLIDE), (2.0, DN500), (2.001, DN100), (2.25, NAV)])
    chains = em.exit_chains(cap)
    assert len(chains) == 1
    ch = chains[0]
    assert ch['eff'] == DN100 and ch['names'] == [DN500, DN100]
    assert ch['prev']['name'] == SLIDE
    assert ch['nxt']['name'] == NAV
    assert ch['window'] == pytest.approx(0.249, abs=1e-6)     # from the last start to the next clip


def test_chain_without_a_next_clip_has_no_window():
    cap = _cap([(1.0, SLIDE), (2.0, DN100)], t_end=2.5)
    ch = em.exit_chains(cap)[0]
    assert ch['window'] is None and ch['window_end'] == 'capture end' and ch['nxt'] is None


def test_separate_exit_starts_are_separate_chains():
    cap = _cap([(1.0, 'xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm'), (1.8, DN100), (1.9, NAV), (3.0, DN100),
                (3.6, NAV)])
    chains = em.exit_chains(cap)
    assert [c['eff'] for c in chains] == [DN100, DN100]
    assert chains[0]['prev']['name'].startswith('xx_Parkour_Sprint_SkipOver')
    assert chains[1]['prev']['name'] == NAV


@pytest.mark.parametrize('name, fam', [
    (None, 'capture start'), (SLIDE, 'SlideOver'), ('xx_Parkour_LandingFwd_LandRoll_tr_Walk', 'LandRoll'),
    ('xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm', 'SkipOver'), ('xx_Parkour_Leaping_Fd100cm-Dn100cm', 'Leaping'),
    ('xx_NavSneak_LowProfile_DropDownCmn_tr_WalkCr', 'drop / sneak'), (DN500, 'Muraille_Exit'),
    ('xx_Parkour_Idle_Vault_Fd200cm-Dn300cm_06', 'other parkour'), ('xx_Nav_Run_3-Loop', 'nav'),
    ('Facial_FightAction_Effort', 'other')])
def test_predecessor_families(name, fam):
    assert em.pred_family(name) == fam


# ---------------------------------------------------------------------------------------------- timing

def test_place_events_split_at_the_boundary_and_map_the_exit_clock():
    ev = dict(takeoff=0.45, hands_start=0.75, roll_end=1.18, feet_down=1.2, run_out=1.6)
    p = em.place_events(ev, 0.025, 1.0, 0.744, (0.42, 1.05), (0.21, 0.97), WD2)
    assert p['takeoff']['clip'] == 'SlideOver' and p['takeoff']['clip_s'] == pytest.approx(0.42 + 1.05 * 0.025)
    assert p['hands_start']['clip'] == 'SlideOver'
    assert p['roll_end']['played'] == pytest.approx(0.755)
    assert p['roll_end']['clip'] == 'Exit'
    assert p['roll_end']['clip_s'] == pytest.approx(0.21 + 0.97 * (0.755 - 0.744))
    assert p['run_out']['played'] == pytest.approx(0.025 + 1.15)


def test_budget_totals_and_run_out_margin():
    b = em.budget(0.025, 0.744, 0.228, WD2, 1.0)
    assert b['exit_end_played'] == pytest.approx(0.972)
    assert b['total_from_takeoff'] == pytest.approx(0.947)
    assert b['wd2_takeoff_to_run_out'] == pytest.approx(1.15)
    assert b['run_out_margin'] == pytest.approx(0.972 - (0.025 + 1.15))


def test_ground_touch_finds_the_landing_and_none_when_airborne():
    fps = 60
    t = np.arange(0, 1.0, 1 / fps)
    z = np.where(t < 0.3, 1.0 - t / 0.3, 0.0)          # falls to the ground at 0.3 s, then stays
    speed = np.where(t < 0.3, 3.0, 0.0)
    tt, sup = em.ground_touch(z, speed, fps)
    assert sup == pytest.approx(0.0)
    assert tt is not None and 0.28 <= tt <= 0.36
    tt, _ = em.ground_touch(np.full_like(t, 0.8), np.zeros_like(t), fps, support=0.0)
    assert tt is None


def test_first_run_needs_min_run_samples():
    assert em.first_run([False, True, False, True, True, False], np.arange(6)) == pytest.approx(3.0)
    assert em.first_run([True, False, True, False], np.arange(4)) is None


# ---------------------------------------------------------------------------------------------- entity pair

def test_entity_pair_reports_z_and_y_at_the_exit_start_and_end():
    # SlideOver at 1.0 s; Exit starts at 1.75 (W = 0.75) and ends at 2.0; entity rises 1 m then is back on the ground
    def zf(t):
        return np.where((t >= 1.0) & (t < 1.75), np.clip(np.sin((t - 1.0) * np.pi / 0.75), 0, 1), 0.0)
    cap = _cap([(1.0, SLIDE), (1.75, DN100), (2.0, NAV)], t_end=2.5, entity_z=zf, entity_y=lambda t: 3.0 * t)
    ch = em.exit_chains(cap)[0]
    prev = ch['prev']
    p = em.entity_pair(cap, ch, prev)
    assert p['W_from_slide'] == pytest.approx(0.75)
    assert p['exit_window'] == pytest.approx(0.25)
    assert p['h'] == pytest.approx(1.0)
    assert p['z_W_rise'] == pytest.approx(0.0, abs=1e-6)                 # back on the ground at W
    assert p['z_max_rise'] == pytest.approx(1.0, abs=0.02)      # samples every 0.05 s
    assert p['y_W'] == pytest.approx(3.0 * 0.75, abs=1e-6)
    assert p['y_end'] == pytest.approx(3.0 * 1.0, abs=1e-6)


# ---------------------------------------------------------------------------------------------- real dump

@pytest.mark.skipif(dumps.load_by_name(DN100) is None, reason='Exit dump not present')
def test_exit_dn100_dump_drops_one_metre_and_lands():
    g = em.exit_geometry(dumps.load_by_name(DN100))
    assert g['length_s'] == pytest.approx(0.867, abs=1e-3)
    assert g['disp_z_end'] == pytest.approx(-1.0, abs=0.01)
    assert g['root_settle_s'] is not None and g['root_settle_s'] < 0.3
    assert g['feet_touch_s'] is not None and 0.2 < g['feet_touch_s'] < 0.6
    # the Exit dump's edge markers do not follow the clip name (height 0, landing 1 m down): kept, not used
    assert g['markers'] is None or 'landing_z' in g['markers']
    assert len(g['disp_t']) == len(g['disp_z'])


def test_model_helpers_are_reused_from_slideover():
    # the chain window and the SlideOver family share the same prefix tests
    assert em.is_exit(DN100) and not em.is_exit(SLIDE)
    assert sm.family(SLIDE) == 'slide'
