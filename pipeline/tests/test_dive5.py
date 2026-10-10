"""dive5 (T18): the 1 m SlideOver preview: the bump shape and amplitude rule, the hand rule, the mapping (both entries
enabled with the preview override, the gate on the Exit) and the written patches."""
import json
import os

import numpy as np
import pytest

import bake2
import config
import dive4 as D4
import dive5 as F

OUT = os.path.join(config.OUT, 'dive5')


def test_hand_rule_is_far_edge_plus_20cm_for_the_1m_wall():
    assert F.HAND_MIN5 == pytest.approx(5.30 + 0.20)


def test_raised_cosine_span_is_zero_outside_and_one_at_the_middle():
    t = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    s = F.raised_cosine_span(t, 0.5, 0.9)
    assert s[0] == 0.0 and s[-1] == 0.0
    assert s[2] == pytest.approx(1.0)                      # the middle of the span


def test_bump_amplitude_is_the_minimum_over_the_middle_of_the_span(monkeypatch):
    """A need profile that is exactly the bump shape times 0.6 m gives an amplitude of 0.6 m."""
    t = np.round(np.arange(0.4, 1.3, 1.0 / 60), 6)
    shape = F.raised_cosine_span(t, 0.54, 1.04)
    need = 0.6 * shape

    class Ctx:
        pass

    monkeypatch.setattr(F, 'box_need', lambda ctx, res: (t, need))
    monkeypatch.setattr(F.B, 'bake_case5', lambda ctx, case, lift, ease=False: {})
    case = F.B.Case(0.74, 1.03, 0.02, 1.0)
    A, t0, t1, info = F.bump_span_and_amplitude(Ctx(), case)
    assert A == pytest.approx(0.6, abs=1e-9)
    assert t0 == pytest.approx(0.54) and t1 == pytest.approx(1.04)


def test_tail_need_is_left_to_the_floor_and_reported(monkeypatch):
    t = np.round(np.arange(0.4, 1.3, 1.0 / 60), 6)
    need = np.where(t > 1.0, 0.1, 0.0)                      # need only after the span
    monkeypatch.setattr(F, 'box_need', lambda ctx, res: (t, need))
    monkeypatch.setattr(F.B, 'bake_case5', lambda ctx, case, lift, ease=False: {})

    class Ctx:
        pass

    A, t0, t1, info = F.bump_span_and_amplitude(Ctx(), F.B.Case(0.74, 1.03, 0.02, 1.0))
    assert A == 0.0
    assert info['tail_need_m'] == pytest.approx(0.1)


def test_written_mapping_enables_both_with_the_override_and_the_gate():
    path = os.path.join(OUT, 'patches', 'mapping_dive5.json')
    if not os.path.exists(path):
        pytest.skip('dive5 outputs not written yet')
    with open(path, encoding='utf-8') as f:
        m = json.load(f)
    by = {c['name']: c for c in m['clips']}
    so = by[F.SO_NAME]
    ex = by[F.EXIT_NAME]
    assert so['handle'] == '0x' + F.SO_HEX and ex['handle'] == '0x' + F.EXIT_HEX
    for c in (so, ex):
        assert c['enabled'] is True
        assert c['override'] == F.OVERRIDE == 'preview: dive roll on 1 m SlideOver'
        assert c['prebaked'] is True and c['patch_translation'] is True
    assert ex['only_after'] == [F.SO_NAME] and ex['arm_seconds'] == 2.0
    assert 'only_after' not in so


def test_written_patches_have_no_root_and_a_reference_translation():
    for hx in (F.SO_HEX, F.EXIT_HEX):
        path = os.path.join(OUT, 'patches', hx + '.baked.json')
        if not os.path.exists(path):
            pytest.skip('dive5 outputs not written yet')
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        ids = [str(t['TrackID']) for t in d['Tracks']]
        assert '0' not in ids
        kinds = {(t['codec'], t['infotype']) for t in d['Tracks'] if str(t['TrackID']) == str(bake2.REF_TID)}
        assert (27, 3) in kinds
