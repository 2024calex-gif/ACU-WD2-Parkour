"""Original ACU vault clips: markers, pose sequences, plausibility and contact events (acuclip.py, contacts.py)."""
import functools
import zlib

import numpy as np
import pytest

import acuclip
import contacts
import dumps
import fk
import skel

FPS = acuclip.FPS
FEET = ('LeftFoot', 'RightFoot', 'LeftFootThumb1', 'RightFootThumb1')


@functools.lru_cache(maxsize=None)
def _vaults():
    return acuclip.vault_dumps()


@functools.lru_cache(maxsize=None)
def _seq(name):
    d = dumps.load_by_name(name)
    return acuclip.pose_sequence(d)


@functools.lru_cache(maxsize=None)
def _world(name):
    d = dumps.load_by_name(name)
    seq = _seq(name)
    return seq, acuclip.world_positions(seq, d), acuclip.obstacle(d)


def _names():
    return [name for _h, name, _d in _vaults()]


def _bone_names():
    nm = dumps.bone_names()
    return [nm.get(b['hash'], hex(b['hash'])) for b in skel.load_acu()]


def _is_helper(n):
    return (n.startswith('ChildOf') or n in ('Reference', 'A_BloodFX', 'PistolInHandLeft', 'RightHand_Weapon_Ref')
            or n.endswith('Pos') or 'Deform' in n)


def _is_finger(n):
    return n.startswith('Smth') or ('Hand' in n and any(k in n for k in ('Index', 'Middle', 'Ring', 'Pinky', 'Thumb')))


# The 20 original Sprint SkipOver/SlideOver vaults. Other vault dumps (Idle_SlideOver, Run, ...) may be added.
SPRINT_VAULTS = [
    'xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm',
    'xx_Parkour_Sprint_SkipOver_Fd100cm-Up150cm',
    'xx_Parkour_Sprint_SkipOver_Fd200cm-Up100cm',
    'xx_Parkour_Sprint_SkipOver_Fd010cm-Up100cm',
    'xx_Parkour_Sprint_SkipOver_Fd100cm-Up050cm',
    'xx_Parkour_Sprint_SkipOver_Fd200cm-Up050cm',
    'xx_Parkour_Sprint_SkipOver_Fd200cm-Up150cm',
    'xx_Parkour_Sprint_SkipOver_Fd300cm-Up150cm',
    'xx_Parkour_Sprint_SkipOver_Fd100cm-Up150cm_02',
    'xx_Parkour_Sprint_SkipOver_Fd010cm-Up200cm',
    'xx_Parkour_Sprint_SlideOver_Fd200cm-Up100cm',
    'xx_Parkour_Sprint_SlideOver_Fd100cm-Up100cm',
    'xx_Parkour_Sprint_SlideOver_Fd100cm-Up150cm',
    'xx_Parkour_Sprint_SlideOver_Fd300cm-Up100cm',
    'xx_Parkour_Sprint_SlideOver_Fd200cm-Up150cm',
    'xx_Parkour_Sprint_SlideOver_Fd300cm-Up150cm',
    'xx_Parkour_Sprint_SlideOver_Fd300cm-Up050cm',
    'xx_Parkour_Sprint_SlideOver_Fd200cm-Up050cm',
    'xx_Parkour_Sprint_SlideOver_Fd100cm-Up050cm',
    'xx_Parkour_Sprint_SlideOver_Fd200cm-Up200cm',
]


def _sprint_names():
    """Vault clips that are Sprint SkipOver/SlideOver (the set the marker facts were verified on)."""
    return [n for n in _names() if n.startswith('xx_Parkour_Sprint_')]


def test_twenty_vault_dumps_found():
    names = set(_names())
    print('\n%d vault clips, %d sprint' % (len(names), len(_sprint_names())))
    missing = [n for n in SPRINT_VAULTS if n not in names]
    assert not missing, 'sprint vault dumps missing: %s' % missing
    assert len(SPRINT_VAULTS) == 20
    assert all(acuclip.is_vault(n) for n in SPRINT_VAULTS)


def test_edge_markers_constant_in_world_within_3cm():
    for name in _names():
        _seq_, _W, ob = _world(name)
        for key, v in ob['spread'].items():
            assert v <= 0.03, '%s %s varies by %.4f m' % (name, key, v)


def test_height_and_depth_match_clip_name_within_2cm():
    # Sprint SkipOver/SlideOver only: the marker facts were verified on these. Idle_SlideOver is excluded
    # because Idle_SlideOver_Fd100cm-Up150cm measures 1.25 m from its edge markers, not the 1.50 m in its name.
    bad = []
    for name in _sprint_names():
        _seq_, _W, ob = _world(name)
        depth, height = acuclip.parse_name(name)
        if abs(ob['height'] - height) > 0.02 or abs(ob['depth'] - depth) > 0.02:
            bad.append('%s: height %.3f vs name %.2f, depth %.3f vs name %.2f'
                       % (name, ob['height'], height, ob['depth'], depth))
    assert not bad, '\n'.join(bad)


def test_displacement_track_moves_forward_and_obstacle_is_static_in_world():
    for name in _names():
        d = dumps.load_by_name(name)
        seq = _seq(name)
        disp = acuclip.markers(d, seq.frames)['disp']
        assert disp[-1, 1] > 5.0, '%s root displacement only %.2f m' % (name, disp[-1, 1])
        ob = acuclip.obstacle(d)
        near = acuclip.markers(d, seq.frames)['near_edge'] + disp
        assert np.max(np.abs(near[:, 1] - ob['near_y'])) <= 0.03


def test_feet_on_ground_during_first_0p3_seconds():
    for name in _names():
        seq, W, _ob = _world(name)
        nm = _bone_names()
        idx = [nm.index(f) for f in FEET]
        sel = seq.frames <= int(round(0.3 * FPS))
        lowest = float(W[sel][:, idx, 2].min())
        assert -0.06 <= lowest <= 0.15, '%s: lowest foot point %.3f m in first 0.3 s' % (name, lowest)


def test_hips_stay_well_above_obstacle_top_while_over_it():
    for name in _names():
        seq, W, ob = _world(name)
        hi = _bone_names().index('Hips')
        over = (W[:, hi, 1] >= ob['near_y']) & (W[:, hi, 1] <= ob['far_y'])
        if not over.any():
            continue
        zmin = float(W[over, hi, 2].min())
        assert zmin >= ob['height'] - 0.6, '%s: Hips z %.3f while over a %.2f m wall' % (name, zmin, ob['height'])


def _max_penetration(W, ob, mask, names_):
    """Largest depth (m) of any masked bone point inside the box y in (near, far), z in (0, height)."""
    y = W[:, :, 1]
    z = W[:, :, 2]
    inside = (y > ob['near_y']) & (y < ob['far_y']) & (z > 0) & (z < ob['height'])
    depth = np.minimum.reduce([y - ob['near_y'], ob['far_y'] - y, ob['height'] - z])
    pen = np.where(inside, depth, 0.0)[:, mask]
    if pen.max() <= 0:
        return 0.0, None, None
    f, bi = np.unravel_index(pen.argmax(), pen.shape)
    bidx = np.nonzero(mask)[0][bi]
    return float(pen.max()), names_[bidx], int(f)


def test_body_points_never_more_than_10cm_inside_box():
    """Trunk, arms, legs, feet and toes (no fingers, no helper bones) stay within 0.1 m of the surface."""
    names_ = _bone_names()
    mask = np.array([not _is_helper(n) and not _is_finger(n) for n in names_])
    worst = (0.0, None, None, None)
    for name in _names():
        _seq_, W, ob = _world(name)
        pen, bone, f = _max_penetration(W, ob, mask, names_)
        if pen > worst[0]:
            worst = (pen, name, bone, f)
    print('\nworst body penetration %.3f m: %s, %s, frame %s' % (worst[0], worst[1], worst[2], worst[3]))
    assert worst[0] <= 0.10, worst


@pytest.mark.xfail(reason='known: finger joints (not skin) reach 0.112 m inside the wall box, '
                          'LeftHandPinky3Pos in SkipOver_Fd200cm-Up150cm at frame 89; '
                          'thumb tip LeftHandThumb3 0.107 m at frame 88. Joint centres, not mesh.',
                   strict=False)
def test_finger_joints_never_more_than_10cm_inside_box():
    names_ = _bone_names()
    mask = np.array([_is_finger(n) or n.endswith('Pos') for n in names_])
    worst = (0.0, None, None, None)
    for name in _names():
        _seq_, W, ob = _world(name)
        pen, bone, f = _max_penetration(W, ob, mask, names_)
        if pen > worst[0]:
            worst = (pen, name, bone, f)
    print('\nworst finger/helper penetration %.3f m: %s, %s, frame %s' % worst)
    assert worst[0] <= 0.10, worst


def test_reference_bone_is_applied_once_in_world_space():
    d = dumps.load_by_name('xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm')
    seq = _seq('xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm')
    # Hips is a child of Reference: its entity position must equal Reference track + qrot(R_ref, bind lpos)
    hi = _bone_names().index('Hips')
    R, P = acuclip.entity_transforms(seq)
    bind = np.array(seq.bones[hi]['lpos'], float)
    expect = seq.local_pos[:, 0] + np.array([fk.qrot(seq.local_rot[f, 0], bind) for f in range(len(seq.frames))])
    assert np.allclose(P[:, hi], expect, atol=1e-9)


def test_inverse_markers_are_inverse_of_reference_transform():
    """Tracks 2246150967 and 3215835963 equal the inverse of the Reference transform (see acuclip docstring)."""
    d = dumps.load_by_name('xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm')
    frames = np.arange(0, 127)
    m = acuclip.markers(d, frames)
    ref_keys_r = dumps.track(d, acuclip.REF_HASH, 0)
    ref_keys_p = dumps.track(d, acuclip.REF_HASH, 3)
    for f in frames[::5]:
        r = dumps.sample_rotation(ref_keys_r, int(f))
        p = dumps.sample_position(ref_keys_p, int(f))
        inv_p = -fk.qrot(fk.qconj(r), p)
        for k in ('m2246150967', 'm3215835963'):
            assert np.allclose(m[k][f], inv_p, atol=2e-3), (k, f, m[k][f], inv_p)


def test_point_masks_on_top_and_ground_synthetic():
    """A point moving at 2 m/s over a box top, stopping for 10 frames inside the band, then moving on."""
    fps = 60.0
    near, far, h = 1.0, 2.0, 0.5
    f = np.arange(60)
    y = np.empty(60)
    y[:20] = 0.5 + f[:20] * (2.0 / fps)            # moving
    y[20:30] = y[19]                               # stopped on the top for 10 frames
    y[30:] = y[19] + (f[30:] - 29) * (2.0 / fps)   # moving again
    pts = np.stack([np.zeros(60), y, np.full(60, h + 0.02)], axis=1)
    top, ground = contacts.point_masks(pts, near, far, h, fps)
    assert not ground.any()
    assert top[22:28].all(), 'stopped interval must be a top contact'
    assert not top[5:12].any(), 'moving fast over the top must not be a contact'
    runs = contacts.runs(top, f)
    assert len(runs) == 1
    s, e = runs[0]
    assert 18 <= s <= 22 and 28 <= e <= 31, runs


def test_point_on_ground_before_wall_is_ground_contact():
    fps = 60.0
    f = np.arange(30)
    pts = np.stack([np.zeros(30), np.full(30, 0.0), np.full(30, 0.03)], axis=1)
    top, ground = contacts.point_masks(pts, 1.0, 2.0, 1.0, fps)
    assert ground.all() and not top.any()
    assert contacts.runs(ground, f) == [[0, 29]]


def test_events_are_consistent_on_one_clip():
    name = 'xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm'
    d = dumps.load_by_name(name)
    seq = _seq(name)
    ev = contacts.events(seq, d)
    assert ev['hand_plant'] is not None and ev['hand_plant']['hand'] in ('LeftHand', 'RightHand')
    assert ev['takeoff']['frame'] < ev['hand_plant']['frame'] < ev['landing']['frame']
    assert ev['landing']['frame'] >= ev['far_frame']
    assert ev['takeoff']['frame'] <= ev['near_frame']
    assert set(ev['contacts']) == set(contacts.POINTS)


def test_pose_sequence_step_sampling():
    d = dumps.load_by_name('xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm')
    seq = acuclip.pose_sequence(d, step=3)
    assert seq.frames[0] == 0 and seq.frames[1] - seq.frames[0] == 3
    assert seq.local_rot.shape == (len(seq.frames), len(seq.bones), 4)
    assert int(seq.frames[-1]) <= 126
