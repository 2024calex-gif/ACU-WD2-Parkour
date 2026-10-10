"""dive4c (T17): the preview mapping (both entries enabled with the preview override, gate kept on the Exit) and the
hand-plant window used by the floor exclusion."""
import json
import os

import pytest

import config
import dive4 as D4
import dive4c as C

OUT = os.path.join(config.OUT, 'dive4c')


def test_plant_window_is_the_wd2_plant_mapped_at_k1_padded_by_two_frames():
    lo, hi = C.plant_window_joined(0.65)
    assert lo == pytest.approx(0.65 + 0.30 - 2 / 60.0)
    assert hi == pytest.approx(0.65 + 0.47 + 2 / 60.0)


def test_override_text_is_the_requested_preview_label():
    assert C.OVERRIDE == 'preview: in-game test of dive roll on low wall'


def test_written_mapping_enables_both_with_the_preview_override():
    path = os.path.join(OUT, 'patches', 'mapping_dive4c.json')
    if not os.path.exists(path):
        pytest.skip('dive4c outputs not written yet')
    with open(path, encoding='utf-8') as f:
        m = json.load(f)
    by = {c['name']: c for c in m['clips']}
    so, ex = by[D4.SO_NAME], by[D4.EXIT_NAME]
    for c in (so, ex):
        assert c['enabled'] is True
        assert c['override'] == C.OVERRIDE
        assert c['prebaked'] is True and c['patch_translation'] is True
    assert ex['only_after'] == [D4.SO_NAME] and ex['arm_seconds'] == 2.0
    assert 'only_after' not in so
