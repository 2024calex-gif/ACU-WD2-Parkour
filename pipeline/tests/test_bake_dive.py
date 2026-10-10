"""Tests for bake_dive.py (WD2 frontdive onto the Sprint SlideOver clips): time map, bump and envelope, the rotation
identity at o, the written patch and mapping format, and the enabled clips (none may be enabled unless they pass)."""
import json
import os
import zlib

import numpy as np
import pytest

import acuclip
import bake2
import bake_dive as BD
import config
import contacts
import dumps
import retarget as R
import wd2

REF = BD.REF_CLIP


@pytest.fixture(scope='module')
def rig():
    return R.Rig()


@pytest.fixture(scope='module')
def dive():
    return BD.load_dive_events()


@pytest.fixture(scope='module')
def kong_like(dive):
    return wd2.load(dive['stem'])


@pytest.fixture(scope='module')
def model():
    return BD.load_play_model_so()


@pytest.fixture(scope='module')
def ref_clip(model):
    dump = dumps.load_by_name(REF)
    seq = acuclip.pose_sequence(dump, step=1)
    ev = contacts.events(seq, dump)
    mp = BD.clip_model(REF, model, float(dump['Length']))
    return dict(dump=dump, seq=seq, ev=ev, mp=mp)


# ------------------------------------------------------------------------------------------------ time map
def test_reference_clock_matches_the_brief(ref_clip):
    """Fd100-Up100: k_acu ~ 0.89 (the brief), the WD2 takeoff sits at the ACU takeoff and the WD2 feet-down at 1.2 s."""
    clock = BD.make_clock(ref_clip['ev'], ref_clip['mp'])
    assert clock.k == pytest.approx(min(clock.k_acu, clock.k_roll))
    assert clock.k == pytest.approx(0.892, abs=0.01)
    assert clock.feet_wd2() == pytest.approx(BD.WD2_FEET, abs=1e-6)
    assert clock.w(ref_clip['mp']['o'] + ref_clip['mp']['r'] * clock.p_takeoff) == pytest.approx(BD.WD2_TAKEOFF, abs=1e-9)


def test_clock_inverse_and_monotonic():
    clock = BD.Clock(o=0.42, r=1.05, p_takeoff=0.03, p_feet=0.70, W=0.74)
    for wd2_s in (0.45, 0.75, 0.92, 1.18, 1.2):
        assert clock.w(clock.t_of_w(wd2_s)) == pytest.approx(wd2_s, abs=1e-9)
    t = np.linspace(0.42, 1.6, 200)
    w = clock.w(t)
    assert np.all(np.diff(w) > 0)
    assert np.allclose(np.diff(w) / np.diff(t), 1.0 / (clock.r * clock.k))


def test_k_is_the_minimum_of_the_two_rules():
    # feet late in the ACU clip relative to the roll: the roll rule binds
    c = BD.Clock(o=0.42, r=1.0, p_takeoff=0.03, p_feet=0.95, W=0.74)
    assert c.k == pytest.approx(c.k_roll) and c.k < c.k_acu
    # ACU feet early: the ACU span binds
    c2 = BD.Clock(o=0.42, r=1.0, p_takeoff=0.03, p_feet=0.60, W=0.90)
    assert c2.k == pytest.approx(c2.k_acu) and c2.k < c2.k_roll


def test_bump_is_zero_outside_and_peaks_at_the_plant():
    t = np.linspace(0.0, 1.5, 3001)
    b = BD.bump(t, 0.4, 0.73, 1.2)
    assert np.all(b[t < 0.4] == 0) and np.all(b[t > 1.2] == 0)
    assert b.max() == pytest.approx(1.0, abs=1e-6)
    assert t[int(np.argmax(b))] == pytest.approx(0.73, abs=1e-3)
    assert np.max(np.abs(np.diff(b))) < 0.02          # smooth (no step)


def test_clip_model_marks_unmodelled_clips_as_no_capture(model):
    mp = BD.clip_model('xx_Parkour_Sprint_SlideOver_Fd999cm-Up999cm', model, 2.0)
    assert mp['capture'] is False and 'ASSUMED' in mp['source']
    mp2 = BD.clip_model(REF, model, 2.0)
    assert mp2['capture'] is True and mp2['source'] == 'own map'
    assert mp2['W'] == pytest.approx(0.7436, abs=1e-3)


# ------------------------------------------------------------------------------------------------ envelope
def test_lift_envelope_is_smooth_and_respects_the_bounds():
    """Synthetic need: a 0.20 m bump over 0.3 s with no upper bound. The solved Dz is >= the need, smooth (no step
    above the 3 cm per 60 Hz frame limit) and zero at the window ends (the taper)."""
    n = 80
    t = np.arange(n)
    L = 0.20 * np.exp(-0.5 * ((t - 40) / 8.0) ** 2)
    U = np.full(n, np.inf)
    D, info = bake2.solve_lift_t9e(L, U, ramp_edges=True)
    inner = slice(BD.bake2.T9D_TAPER, n - BD.bake2.T9D_TAPER)
    assert np.all(D[inner] >= L[inner] - 1e-9)           # the taper ramps the ends to zero (by design, see the docstring)
    assert np.max(np.abs(np.diff(D))) * 100.0 <= BD.DSTEP_CM         # the spec's per-frame Delta limit
    assert D[0] == pytest.approx(0.0, abs=1e-9) and D[-1] == pytest.approx(0.0, abs=1e-9)
    assert info['conflict'] == 0


def test_lift_envelope_obeys_an_upper_bound_when_feasible():
    n = 60
    L = np.full(n, -0.05)
    U = np.full(n, -0.02)                 # lowering allowed by 2 cm at most, and the need is -5 cm
    D, info = bake2.solve_lift_t9e(L, U, ramp_edges=False)   # the forced-taper path: every frame obeys both bounds
    assert np.all(D <= U + 1e-9) and np.all(D >= L - 1e-9)
    assert info['conflict'] == 0


# ------------------------------------------------------------------------------------------------ rotations at o
def test_additive_rotations_equal_the_original_at_the_patch_start(rig, kong_like, ref_clip):
    """Additive mode: at o the WD2 change is zero, so every patched local equals the clip's own local rotation."""
    dump, seq = ref_clip['dump'], ref_clip['seq']
    mp = ref_clip['mp']
    clock = BD.make_clock(ref_clip['ev'], mp)
    patset, rot = BD.patch_set(rig, dump)
    f0 = int(round(mp['o'] * BD.FPS))
    out = BD.dive_rotations(rig, kong_like, clock, [f0], patset, rot, mode=BD.ROT_ADDITIVE)
    for ai in patset:
        orig = bake2.orig_local(rig, rot, ai, f0)
        assert bake2.angle_deg(out[ai][0], orig) < 1e-3, rig.ac[ai]['name']


def test_patch_set_contains_the_reference_and_no_hips(rig, ref_clip):
    patset, rot = BD.patch_set(rig, ref_clip['dump'])
    hips = rig.ac_by_hash[zlib.crc32(b'Hips')]
    assert 0 in patset and hips not in patset
    assert len(patset) >= 40


def test_absolute_reference_is_a_rotated_pelvis(rig, kong_like, ref_clip):
    """Absolute mode: the Reference is A_REF * Rp (the WD2 pelvis), as the Reference compensation of this dive."""
    dump = ref_clip['dump']
    clock = BD.make_clock(ref_clip['ev'], ref_clip['mp'])
    patset, rot = BD.patch_set(rig, dump)
    f = int(round(1.0 * BD.FPS))
    out = BD.dive_rotations(rig, kong_like, clock, [f], patset, rot, mode=BD.ROT_ABSOLUTE)
    Ls, _Wt = bake2.wd2_pose_at(rig, kong_like, float(clock.w(f / BD.FPS)))
    assert bake2.angle_deg(out[0][0], R.qn(R.qmul(R.A_REF, Ls[0]))) < 1e-6


def test_rotation_mode_must_be_known(rig, kong_like, ref_clip):
    patset, rot = BD.patch_set(rig, ref_clip['dump'])
    clock = BD.make_clock(ref_clip['ev'], ref_clip['mp'])
    with pytest.raises(ValueError):
        BD.dive_rotations(rig, kong_like, clock, [60], patset, rot, mode='nope')


