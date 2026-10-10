"""Tests for matchclip.py: window, time-map fit, yaw and pelvis correction on a synthetic capture built from a real
dump, plus smoke checks on one real capture when the data folder is present."""
import glob
import math
import os

import numpy as np
import pytest

import acuclip
import capture
import config
import dumps
import ik
import matchclip as mc

HANDLE = '0x296F0F8B8'        # xx_Parkour_Sprint_SkipOver_Fd200cm-Up100cm (2.233 s)
NEXT_NAME = 'xx_Parkour_Muraille_Exit_Fd250cm-Dn300cm'
REAL_DIR = os.path.join(config.V2, 'data', 'captures', 'session1_20261008')


def _rz(yaw_deg, v):
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]], axis=-1)


T_START = 6.0 / 18.0   # on a capture sample (frame 6 at 18 Hz): the entity rise starts exactly here, and the capture
                       # can only give z at the clip start from its samples (see pelvis_correction)


def _synthetic(o=0.35, r=1.1, yaw=0.0, rise_max=0.8, t_start=T_START, next_at=1.2, t_end=1.6, rate=18.0):
    """Capture whose bones are the dump's own clip positions under the time map (o, r) and yaw, plus a z rise of the
    entity (and of Hips in the game only, which the pelvis correction must recover)."""
    dump = dumps.load_by_handle(HANDLE)
    seq = acuclip.pose_sequence(dump)
    P = acuclip.entity_positions(seq)
    fr = np.arange(P.shape[0]) / 60.0
    t = np.arange(int(round(t_end * rate))) / rate
    n = t.size
    ct = o + r * (t - t_start)

    def at(bi):
        return np.stack([np.interp(ct, fr, P[:, bi, k]) for k in range(3)], axis=-1)

    hidx = ik.bone_index(seq.bones, mc.ROOT_BONE)
    hips_clip = at(hidx)
    rise = rise_max * np.clip((t - t_start) / 0.6, 0.0, 1.0)
    hips_game = _rz(yaw, hips_clip) + np.stack([np.zeros(n), np.zeros(n), rise], axis=-1)

    bones = {}
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    for name in capture.BONE_NAMES:
        if name in mc.FIT_BONES:
            bi = ik.bone_index(seq.bones, name)
            pos = hips_game + _rz(yaw, at(bi) - hips_clip)
        else:
            pos = hips_game.copy()
        bones[name] = np.hstack([pos, quat])

    M = np.zeros((n, 4, 4))
    M[:, 0, 0] = M[:, 1, 1] = M[:, 2, 2] = M[:, 3, 3] = 1.0
    M[:, 3, 0:3] = np.stack([np.zeros(n), np.zeros(n), rise], axis=-1)
    clips = [
        dict(t=t_start, ptr='0x1', handle=HANDLE, name=dump['Name'], length=float(dump['Length']), patched=False),
        dict(t=next_at, ptr='0x2', handle=None, name=NEXT_NAME, length=1.2, patched=False),
    ]
    cap = capture.Capture(
        path='synthetic', trigger=dump['Name'], patching_enabled=False, pose_ok=True, bones_space='entity',
        plugin_version='0.3.0', t=t, M=M, entity_pos=M[:, 3, 0:3].copy(), bones=bones, clips=clips)
    return cap, dump, dict(o=o, r=r, yaw=yaw, rise=rise, t_start=t_start)


@pytest.fixture(scope='module')
def synth():
    return _synthetic()


# ---------------------------------------------------------------------------------------------- window

def test_window_ends_at_next_clip_start(synth):
    cap, dump, truth = synth
    w = mc.match_window(cap, 0)
    assert w['t_start'] == pytest.approx(T_START)
    assert w['t_end'] == pytest.approx(1.2)
    assert w['end_reason'] == 'next clip'
    assert w['next_name'] == NEXT_NAME


def test_window_capped_at_clip_length_when_next_clip_is_late():
    cap, dump, _ = _synthetic(next_at=5.0, t_end=3.0)
    w = mc.match_window(cap, 0)
    assert w['length_s'] == pytest.approx(float(dump['Length']), abs=1e-6)
    assert w['end_reason'] == 'clip length'


def test_window_ignores_clips_that_do_not_match_prefixes():
    cap, dump, _ = _synthetic(next_at=0.9, t_end=1.6)
    cap.clips[1]['name'] = 'xx_Parkour_Sprint_Idle_Whatever'
    w = mc.match_window(cap, 0)
    assert w['end_reason'] != 'next clip'
    assert w['next_name'] is None


def test_window_is_all_pose_frames_to_capture_end():
    cap, _, _ = _synthetic(next_at=5.0, t_end=1.0)
    w = mc.match_window(cap, 0)
    assert w['end_reason'] == 'capture end'
    assert w['t_end'] == pytest.approx(float(cap.t[-1]))


def test_trigger_dump_finds_the_handle(synth):
    cap, dump, _ = synth
    idx, d = mc.trigger_dump(cap)
    assert idx == 0 and d is not None and d['Name'] == dump['Name']


# ---------------------------------------------------------------------------------------------- fit

def test_fit_recovers_time_map_on_noise_free_data(synth):
    cap, dump, truth = synth
    w = mc.match_window(cap, 0)
    fit = mc.fit_time_map(cap, dump, w)
    assert fit['o'] == pytest.approx(truth['o'], abs=0.02)
    assert fit['r'] == pytest.approx(truth['r'], abs=0.02)
    assert abs(fit['yaw_deg']) < 1.0
    assert fit['rms_cm'] < 0.5
    assert fit['poor'] is False
    # the unfitted baseline (o = 0, r = 1) is clearly worse than the fit
    assert fit['rms_identity_cm'] > 5.0


def test_fit_recovers_yaw():
    cap, dump, truth = _synthetic(o=0.2, r=0.9, yaw=30.0)
    fit = mc.fit_time_map(cap, dump, mc.match_window(cap, 0))
    assert fit['yaw_deg'] == pytest.approx(30.0, abs=1.0)
    assert fit['o'] == pytest.approx(0.2, abs=0.02)
    assert fit['r'] == pytest.approx(0.9, abs=0.02)


def test_fit_r1_is_worse_than_free_r_when_r_is_not_one(synth):
    cap, dump, _ = synth
    fit = mc.fit_time_map(cap, dump, mc.match_window(cap, 0))
    assert fit['rms_r1']['rms_cm'] > fit['rms_cm'] + 1.0


def test_landscape_minimum_matches_fit(synth):
    cap, dump, _ = synth
    w = mc.match_window(cap, 0)
    fit = mc.fit_time_map(cap, dump, w)
    og, rg, L = mc.fit_landscape(cap, dump, w, fit['yaw_deg'])
    i, j = np.unravel_index(int(np.argmin(L)), L.shape)
    assert abs(og[i] - fit['o']) <= 0.011
    assert abs(rg[j] - fit['r']) <= 0.021


# ---------------------------------------------------------------------------------------------- pelvis

def test_pelvis_correction_recovers_entity_rise(synth):
    cap, dump, truth = synth
    w = mc.match_window(cap, 0)
    fit = mc.fit_time_map(cap, dump, w)
    pc = mc.pelvis_correction(cap, dump, fit)
    assert pc['fit_z']['a'] == pytest.approx(1.0, abs=0.02)
    assert pc['fit_z']['b'] == pytest.approx(0.0, abs=0.005)
    assert pc['fit_z']['r2'] > 0.999
    assert np.max(np.abs(pc['dhips'][:, 2] - pc['rise'])) < 0.005
    assert pc['rise_max'] == pytest.approx(0.8, abs=1e-3)
    assert pc['dxy_mean_cm'] < 0.5


# ---------------------------------------------------------------------------------------------- real data

@pytest.mark.skipif(not glob.glob(os.path.join(REAL_DIR, '*_6.json')), reason='capture 6 not present')
def test_real_capture6_window_and_fit_run():
    cap = capture.load(glob.glob(os.path.join(REAL_DIR, '*_6.json'))[0])
    idx, dump = mc.trigger_dump(cap)
    assert dump is not None
    w = mc.match_window(cap, idx)
    assert w['t_start'] == pytest.approx(1.7376, abs=1e-3)
    assert w['end_reason'] == 'next clip'
    assert w['length_s'] == pytest.approx(0.4703, abs=1e-3)
    fit = mc.fit_time_map(cap, dump, w)
    assert np.isfinite(fit['rms_cm']) and np.isfinite(fit['o']) and np.isfinite(fit['r'])
    assert fit['n_frames'] >= 3
    pc = mc.pelvis_correction(cap, dump, fit)
    assert np.isfinite(pc['rise_max'])


# ---------------------------------------------------------------------------------------------- report helpers

def _row(clip, o, r, rms=3.0, poor=False, skip=None, window_s=0.5, a=-0.1, b=0.0, reason=''):
    return dict(clip=clip, o=o, r=r, yaw=0.0, rms=rms, rms_naive=40.0, rms_r1=rms, rise_max=1.0, a=a, b=b, r2=0.5,
                window_s=window_s, poor=poor, skip=skip, reason=reason, index=1, file='f', trigger=clip)


def test_play_model_medians_per_clip_and_exclusion():
    import matchclip_report as rep
    rows = [_row('A', 0.1, 1.0, window_s=0.4), _row('A', 0.3, 1.2, window_s=0.6, a=-0.3, b=0.1),
            _row('A', 0.5, 0.9, poor=True, rms=30.0, window_s=0.5),
            _row('B', 0.0, 1.0, window_s=0.8, a=0.2, b=-0.05),
            _row('C', 9.0, 9.0, skip='no entity-space pose')]
    pm = rep.play_model(rows, exclude_poor=False)
    assert set(pm['per_clip']) == {'A', 'B'}
    assert pm['per_clip']['A'] == pytest.approx(dict(o=0.3, r=1.0, window=0.5, n=3))
    assert pm['per_clip']['B'] == pytest.approx(dict(o=0.0, r=1.0, window=0.8, n=1))
    # all four fitted rows (o = 0.1, 0.3, 0.5, 0.0; the poor one included) -> median 0.2
    assert pm['median']['o'] == pytest.approx(0.2)
    assert pm['pelvis']['a'] == pytest.approx(-0.1)
    pm2 = rep.play_model(rows, exclude_poor=True)
    assert pm2['per_clip']['A']['n'] == 2
    assert pm2['per_clip']['A']['o'] == pytest.approx(0.2)


def test_poor_reason_names_the_cause():
    import matchclip_report as rep
    good = _row('A', 0.1, 1.0)
    assert rep.poor_reason(good) == ''
    bad = dict(_row('A', 0.1, 0.6, rms=18.0, poor=True), at_edge=dict(o=False, r=True), n_frames=9, n_window=9,
               rms_r1=26.0)
    reason = rep.poor_reason(bad)
    assert 'search border' in reason
    assert 'r = 1' not in reason
