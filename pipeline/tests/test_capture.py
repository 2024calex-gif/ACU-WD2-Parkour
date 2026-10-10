"""Flight-recorder capture reader and analysis (capture.py) on synthetic captures (synth_capture.py)."""
import json
import zlib

import numpy as np
import pytest

import capture
import dumps
import synth_capture as sc

FRAME = 1.0 / 60.0


def _write_doc(tmp_path, doc, name='cap.json'):
    p = tmp_path / name
    p.write_text(json.dumps(doc), encoding='utf-8')
    return str(p)


@pytest.fixture
def rows_capture(tmp_path):
    p = str(tmp_path / 'rows.json')
    truth = sc.write(p, convention='rows', heading_deg=30.0)
    return capture.load(p), truth, p


def test_load_round_trip(rows_capture):
    cap, truth, _ = rows_capture
    assert cap.trigger == sc.CLIP_SKIP
    assert cap.pose_ok is True
    assert cap.bones_space == 'entity'
    assert cap.patching_enabled is False
    assert cap.t.shape == (sc.N_FRAMES,)
    assert np.allclose(cap.t, truth['times'], atol=1e-5)     # t is written with 6 decimals
    assert cap.M.shape == (sc.N_FRAMES, 4, 4)
    assert np.allclose(cap.entity_pos, truth['entity_pos'], atol=1e-9)
    assert set(cap.bones) == set(capture.BONE_NAMES)
    assert cap.bones['LeftHand'].shape == (sc.N_FRAMES, 7)
    assert np.all(np.isfinite(cap.bones['LeftHand']))
    assert [c['name'] for c in cap.clips] == [sc.CLIP_SKIP, sc.CLIP_MURAILLE]
    assert cap.clips[0]['handle'] == '0x296F0C7D9'
    assert cap.clips[0]['t'] == pytest.approx(0.5)
    assert cap.clips[1]['t'] == pytest.approx(1.4)


def test_clip_name_filled_from_handle(tmp_path):
    p = str(tmp_path / 'nameless.json')
    sc.write(p)
    doc = json.load(open(p, encoding='utf-8'))
    doc['clips'][0]['name'] = None
    cap = capture.load(_write_doc(tmp_path, doc, 'nameless2.json'))
    assert cap.clips[0]['name'] == sc.CLIP_SKIP


def test_load_rejects_other_format(tmp_path):
    p = _write_doc(tmp_path, {'format': 'something-else', 'version': 1, 'frames': []})
    with pytest.raises(ValueError):
        capture.load(p)


def test_pose_missing_gives_none_bones(tmp_path):
    p = str(tmp_path / 'nopose.json')
    sc.write(p, pose=False)
    cap = capture.load(p)
    assert cap.pose_ok is False
    assert all(v is None for v in cap.bones.values())
    assert capture.bone_world(cap, 'LeftHand') is None


def test_entity_axes_both_readings():
    X, Y, Z = sc._basis(30.0)
    T = np.array([1.0, 2.0, 3.0])
    for conv in ('rows', 'columns'):
        M = sc._pack(X, Y, Z, T, conv).reshape(4, 4)[None]
        Xa, Ya, Za, Ta = capture.entity_axes(M, row_major=(conv == 'rows'))
        assert np.allclose(Xa[0], X) and np.allclose(Ya[0], Y) and np.allclose(Za[0], Z), conv
        assert np.allclose(Ta[0], T), conv


def test_convention_rows_detected(tmp_path):
    p = str(tmp_path / 'rows.json')
    sc.write(p, convention='rows', heading_deg=30.0)
    conv = capture.check_convention(capture.load(p))
    assert conv['verdict'] == 'rows'
    assert conv['rows_deg'] < 1.0
    assert conv['cols_deg'] > 30.0
    assert conv['matches_constant'] == capture.ROW_MAJOR_BASIS


def test_convention_columns_detected(tmp_path):
    p = str(tmp_path / 'cols.json')
    sc.write(p, convention='columns', heading_deg=30.0)
    conv = capture.check_convention(capture.load(p))
    assert conv['verdict'] == 'columns'
    assert conv['cols_deg'] < 1.0
    assert conv['rows_deg'] > 30.0
    assert conv['row_major_fits'] is False


def test_convention_ambiguous_at_zero_heading(tmp_path):
    # With heading 0 the two readings of +Y coincide, so the data cannot tell them apart.
    p = str(tmp_path / 'zero.json')
    sc.write(p, convention='rows', heading_deg=0.0)
    assert capture.check_convention(capture.load(p))['verdict'] == 'ambiguous'


def test_convention_insufficient_motion(rows_capture):
    cap, _, _ = rows_capture
    cap.entity_pos = np.zeros_like(cap.entity_pos)
    res = capture.check_convention(cap)
    assert res['verdict'] == 'insufficient motion'


@pytest.mark.parametrize('conv', ['rows', 'columns'])
def test_to_world_matches_truth_for_each_reading(tmp_path, conv):
    p = str(tmp_path / ('%s.json' % conv))
    truth = sc.write(p, convention=conv, heading_deg=30.0)
    cap = capture.load(p)
    local = cap.bones['LeftHand'][:, 0:3]
    world = capture.to_world(cap, local, row_major=(conv == 'rows'))
    assert np.allclose(world, truth['bones_world']['LeftHand'], atol=1e-9)


def test_bone_world_uses_module_convention(rows_capture):
    cap, truth, _ = rows_capture
    assert capture.ROW_MAJOR_BASIS is True
    assert np.allclose(capture.bone_world(cap, 'RightFoot'), truth['bones_world']['RightFoot'], atol=1e-9)


def test_bone_world_requires_entity_space(tmp_path):
    p = str(tmp_path / 'world.json')
    sc.write(p)
    doc = json.load(open(p, encoding='utf-8'))
    doc['bones_space'] = 'world'
    cap = capture.load(_write_doc(tmp_path, doc, 'world2.json'))
    with pytest.raises(ValueError):
        capture.bone_world(cap, 'LeftHand')


def test_clip_segments(rows_capture):
    cap, _, _ = rows_capture
    segs = capture.clip_segments(cap)
    assert [s[0] for s in segs] == [sc.CLIP_SKIP, sc.CLIP_MURAILLE]
    assert segs[0][1] == pytest.approx(0.5) and segs[0][2] == pytest.approx(1.4)     # ends at next start (blend)
    assert segs[1][1] == pytest.approx(1.4) and segs[1][2] == pytest.approx(cap.t[-1])  # last ends at capture end


def test_obstacle_placed_within_5cm(rows_capture):
    cap, truth, _ = rows_capture
    obs = capture.obstacle_from_clip(cap, dumps)
    assert obs is not None
    assert np.allclose(obs.near, truth['near'], atol=0.05)
    assert np.allclose(obs.far, truth['far'], atol=0.05)
    assert obs.height == pytest.approx(1.0, abs=0.05)
    assert obs.depth == pytest.approx(1.0, abs=0.05)
    assert obs.t0 == pytest.approx(0.5)
    # the near edge is 3.59 m ahead of the entity at the clip start, from the dump marker (not 4 m)
    horiz = np.linalg.norm(obs.near[0:2] - cap.entity_pos[30, 0:2])
    assert horiz == pytest.approx(3.589, abs=0.02)


def test_obstacle_none_without_dump(tmp_path):
    p = str(tmp_path / 'nodump.json')
    sc.write(p)
    doc = json.load(open(p, encoding='utf-8'))
    # A clip that has no dump under either lookup key: handle 0x1 and a name that is not in the dumps folder.
    # (Muraille_Exit_Fd250cm-Dn500cm has a dump now, so it can no longer stand for "no dump".)
    name = 'xx_Parkour_NoSuchClip_Fd250cm-Dn500cm'
    assert dumps.load_by_handle('0x1') is None and dumps.load_by_name(name) is None
    doc['trigger'] = name
    doc['clips'][0]['name'] = name
    doc['clips'][0]['handle'] = '0x1'
    cap = capture.load(_write_doc(tmp_path, doc, 'nodump2.json'))
    assert capture.obstacle_from_clip(cap, dumps) is None
    ev = capture.events(cap, None)
    assert ev['pose'] is True and ev['hand_plant'] is None


def test_events_hand_plant_within_two_frames(rows_capture):
    cap, truth, _ = rows_capture
    obs = capture.obstacle_from_clip(cap, dumps)
    ev = capture.events(cap, obs)
    assert ev['pose'] is True
    assert ev['hand_plant'] is not None
    assert abs(ev['hand_plant'] - truth['hand_plant']) <= 2 * FRAME + 1e-9


def test_events_takeoff_and_landing(rows_capture):
    cap, truth, _ = rows_capture
    obs = capture.obstacle_from_clip(cap, dumps)
    ev = capture.events(cap, obs)
    assert ev['takeoff'] is not None and ev['landing'] is not None
    assert ev['takeoff'] < ev['hand_plant'] < ev['landing']
    assert ev['takeoff'] == pytest.approx(truth['takeoff'], abs=0.05)
    assert ev['landing'] == pytest.approx(truth['landing'], abs=0.05)


def test_events_without_pose_are_clip_based(tmp_path):
    p = str(tmp_path / 'nopose.json')
    sc.write(p, pose=False)
    cap = capture.load(p)
    ev = capture.events(cap, capture.obstacle_from_clip(cap, dumps))
    assert ev['pose'] is False
    assert ev['hand_plant'] is None and ev['takeoff'] is None and ev['landing'] is None
    assert ev['trigger_start'] == pytest.approx(0.5)
    assert [n for n, _ in ev['clip_starts']] == [sc.CLIP_SKIP, sc.CLIP_MURAILLE]


def test_events_without_obstacle_have_no_plant(rows_capture):
    cap, _, _ = rows_capture
    ev = capture.events(cap, None)
    assert ev['pose'] is True
    assert ev['hand_plant'] is None


def test_translation_and_bone_ids(rows_capture):
    cap, _, p = rows_capture
    doc = json.load(open(p, encoding='utf-8'))
    m = np.array(doc['frames'][0]['entity_matrix'])
    assert np.allclose(cap.entity_pos[0], m[12:15])
    key = str(zlib.crc32(b'LeftHand'))
    assert key in doc['frames'][0]['bones']
    assert np.allclose(cap.bones['LeftHand'][0], doc['frames'][0]['bones'][key])


def test_partial_pose_loads_present_bones(tmp_path):
    # Head and LeftFootThumb1 are missing in every frame: the header says so, the other bones load as before.
    doc, truth = sc.build(convention='rows', heading_deg=30.0)
    for fr in doc['frames']:
        fr['bones'].pop(str(zlib.crc32(b'Head')))
        fr['bones'].pop(str(zlib.crc32(b'LeftFootThumb1')))
    doc['pose_ok'] = False
    doc['pose_bones_missing'] = ['Head', 'LeftFootThumb1']
    cap = capture.load(_write_doc(tmp_path, doc, 'partial.json'))
    assert cap.pose_ok is False
    assert cap.pose_bones_missing == ['Head', 'LeftFootThumb1']
    assert cap.bones['Head'] is None and cap.bones['LeftFootThumb1'] is None
    assert np.allclose(capture.bone_world(cap, 'RightFoot'), truth['bones_world']['RightFoot'], atol=1e-9)
    obs = capture.obstacle_from_clip(cap, dumps)
    ev = capture.events(cap, obs)
    assert ev['pose'] is True
    assert ev['hand_plant'] is not None
    assert abs(ev['hand_plant'] - truth['hand_plant']) <= 2 * FRAME + 1e-9


def test_partial_frame_gap_is_nan_in_that_frame_only(tmp_path):
    # LeftHand is left out of one frame only: that row is NaN, every other row is the recorded value.
    doc, _ = sc.build(convention='rows', heading_deg=30.0)
    key = str(zlib.crc32(b'LeftHand'))
    gap = 10
    kept = doc['frames'][gap]['bones'].pop(key)
    doc['pose_ok'] = False
    doc['pose_bones_missing'] = ['LeftHand']
    cap = capture.load(_write_doc(tmp_path, doc, 'gap.json'))
    lh = cap.bones['LeftHand']
    assert np.all(np.isnan(lh[gap]))
    assert np.all(np.isfinite(np.delete(lh, gap, axis=0)))
    assert np.allclose(lh[gap - 1], doc['frames'][gap - 1]['bones'][key])
    assert np.allclose(lh[gap + 1], doc['frames'][gap + 1]['bones'][key])
    assert kept is not None


def test_missing_list_derived_for_old_files(tmp_path):
    # v0.4 files have no pose_bones_missing: a pose-less capture misses every bone, a full one misses none.
    p = str(tmp_path / 'old_nopose.json')
    sc.write(p, pose=False)
    doc = json.load(open(p, encoding='utf-8'))
    doc.pop('pose_bones_missing', None)
    doc.pop('variant', None)
    cap = capture.load(_write_doc(tmp_path, doc, 'old_nopose2.json'))
    assert cap.pose_bones_missing == list(capture.BONE_NAMES)
    assert cap.variant == ''
    p2 = str(tmp_path / 'old_pose.json')
    sc.write(p2, pose=True)
    doc2 = json.load(open(p2, encoding='utf-8'))
    doc2.pop('pose_bones_missing', None)
    assert capture.load(_write_doc(tmp_path, doc2, 'old_pose2.json')).pose_bones_missing == []


def test_variant_read_from_header(tmp_path):
    p = str(tmp_path / 'variant.json')
    sc.write(p)
    doc = json.load(open(p, encoding='utf-8'))
    doc['variant'] = 'B_nofade'
    assert capture.load(_write_doc(tmp_path, doc, 'variant2.json')).variant == 'B_nofade'
