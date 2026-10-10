"""T13: the WD2 frontdive as a SlideOver with a REPLACED root (dive_root.py) + the gated Muraille_Exit, on one joined timeline.

What differs from bake_chain.py (T12b, out\\dive):
  * The SlideOver plays a replacement root (TrackID 0, the entity's displacement): a dive arc that passes the far top
    edge at about 0.8 clip s, identical to the original outside [0.45, 1.2] s (dive_root.py). The patch carries it as
    a TrackID "0" codec 27 infotype 3 track (the plugin's patch_translation replaces it by TrackID).
  * The Exit is played from its own root DELTA. The original Exit's root drops 1.0 m at clip 0.00 to 0.20 s, before its
    played window starts at 0.208, so its absolute displacement is -1.0 m at the cut. The game joins the clip at the
    SlideOver's root at W and moves it by the clip's change since the cut: joined_root_E(t) = root_S(W) + (disp_E(t) -
    disp_E(0.208)). The old absolute convention put the Exit's ground at -1.0 m, the cause of the 32 cm penetration in
    bake_chain (its EXIT_Z_SHIFT = +0.751 was a support height of the absolute frame). With the delta ground is z = 0.
  * Time map k = 1: WD2 = 0.45 + (played - played takeoff), so the WD2 hand plant starts 0.30 s after the takeoff.
  * Variant D fades (fade-in 0.08 s, no fade-out, WD2 tail), gated Exit with no fade-in, ease to Nav_Run only when the
    pre-ease gap exceeds 45 deg (over the last 0.10 s of the Exit window).

Outputs, all under out\\dive2 (nothing else is written):
  patches\\47BFB02DF.baked.json  rotations + Reference translation + TrackID 0 (the replacement root)
  patches\\F7919BDF0.baked.json  rotations + Reference translation (Exit)
  patches\\mapping_dive2.json    SlideOver entry + gated Exit entry (only_after the SlideOver, arm 2.0 s)
  patches_noroot\\...            the same body without the TrackID 0 track (comparison)
  root_curve.png / root_curve.json, joined_sheet.png, joined_sheet.gif, report.md, chain_summary.json
Run:  python dive_chain2.py
"""
import json
import os
import re
import zlib

import numpy as np

import acuclip
import bake2
import bake_chain as BC
import bake_dive as BD
import config
import contacts
import dive_root as DR
import dumps
import retarget as R
import wd2

OUT_DIR = os.path.join(config.OUT, 'dive2')
PATCH_DIR = os.path.join(OUT_DIR, 'patches')
NOROOT_DIR = os.path.join(OUT_DIR, 'patches_noroot')
MAPPING = os.path.join(PATCH_DIR, 'mapping_dive2.json')
REPORT = os.path.join(OUT_DIR, 'report.md')
SUMMARY = os.path.join(OUT_DIR, 'chain_summary.json')
SO_HEX = '47BFB02DF'
MAIN_SO = BC.MAIN_SO
EXIT_HEX = BC.EXIT_HEX
EXIT_NAME = BC.EXIT_NAME
ROOT_TID = 0
EASE_S = 0.10                      # s, the ease to Nav_Run (dive spec); bake_chain reads it as a module global
NARROW = dict(pen_cm=10.0, touch=2, step_cm=6.0, gap_deg=60.0, cont_deg=60.0, ground_cm=10.0)
K1_LEAD = BD.WD2_PLANT[0] - BD.WD2_TAKEOFF      # 0.30 WD2 s between the takeoff and the plant start: k = 1


# ------------------------------------------------------------------------------------------------ writers
def root_track(keys):
    """The replacement root track in the plugin format (codec 27, infotype 3, LocalPosition keys at 60 Hz)."""
    return {'TrackID': str(ROOT_TID), 'codec': 27, 'infotype': 3,
            'Keyframes': [{'Frame': int(k['Frame']), 'LocalPosition': [float(x) for x in k['LocalPosition']]}
                          for k in keys]}


def write_patch(path, rig, rot_tracks, trans_keys, root_keys, length):
    """Patch with rotation tracks, the Reference translation, and (when root_keys is given) the TrackID 0 root."""
    tr = []
    for ai in sorted(rot_tracks, key=lambda a: rig.ac[a]['hash']):
        kf = [{'Frame': int(k['Frame']), 'LocalRotation': [round(float(x), 9) for x in k['LocalRotation']]}
              for k in rot_tracks[ai]]
        tr.append({'TrackID': str(rig.ac[ai]['hash']), 'codec': 1, 'infotype': 0, 'Keyframes': kf})
    tr.append({'TrackID': str(bake2.REF_TID), 'codec': 27, 'infotype': 3,
               'Keyframes': [{'Frame': int(k['Frame']), 'LocalPosition': [round(float(x), 9) for x in k['LocalPosition']]}
                             for k in trans_keys]})
    if root_keys is not None:
        tr.append({'TrackID': str(ROOT_TID), 'codec': 27, 'infotype': 3,
                   'Keyframes': [{'Frame': int(k['Frame']),
                                  'LocalPosition': [round(float(x), 9) for x in k['LocalPosition']]} for k in root_keys]})
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'Length': round(float(length), 6), 'Tracks': tr}, f)


def mapping_entry(handle, name, source, enabled, only_after=None, override=None):
    e = {'handle': '0x' + handle, 'name': name, 'source': source, 'enabled': bool(enabled), 'prebaked': True,
         'patch_translation': True}
    if only_after is not None:
        e['only_after'] = list(only_after)
        e['arm_seconds'] = 2.0
    if override:
        e['override'] = override
    return e


# ------------------------------------------------------------------------------------------------ the joined root
def exit_root_delta(root, mp, ex_dump, ex_seq):
    """Root the Exit plays with: the SlideOver's replacement root at W plus the Exit's own displacement change since the cut."""
    tW = float(mp['o'] + mp['r'] * mp['W'])
    root_w = DR.rows_at(root['new'], np.array([tW]))[0]
    disp_e = acuclip.markers(ex_dump, ex_seq.frames)['disp']
    cut = DR.rows_at(disp_e, np.array([BC.EXIT_O]))[0]
    d = root_w[None, :] + (disp_e - cut[None, :])
    d[:, 0] = 0.0
    return d, root_w, tW, cut


# ------------------------------------------------------------------------------------------------ narrowness
def _num(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def narrow_fail(fail):
    """True when one failure string is within the override thresholds (the spec's narrow failures)."""
    if fail.startswith('non-finger penetration'):
        v = _num(r'([\d.]+) cm', fail)
        return v is not None and v <= NARROW['pen_cm']
    if fail.startswith('bone below ground'):
        v = _num(r'([\d.]+) cm', fail)
        return v is not None and v <= NARROW['ground_cm']
    if fail.startswith('hand touch'):
        v = _num(r'hand touch (\d+) frames', fail)
        return v is not None and v >= NARROW['touch']
    if fail.startswith('Delta step') or fail.startswith('joined root step'):
        v = _num(r'([\d.]+) cm/frame', fail)
        return v is not None and v <= NARROW['step_cm']
    if fail.startswith('continuity at W'):
        v = _num(r'continuity at W ([\d.]+) deg', fail)
        return v is not None and v <= NARROW['cont_deg']
    return False


def status_of(fails):
    if not fails:
        return 'PASS'
    if all(narrow_fail(f) for f in fails):
        return 'OVERRIDE'
    return 'FAIL'


# ------------------------------------------------------------------------------------------------ main
def build_all(write=True, plots=True):
    os.makedirs(PATCH_DIR, exist_ok=True)
    os.makedirs(NOROOT_DIR, exist_ok=True)
    rig = R.Rig()
    m = wd2.load(BD.load_dive_events()['stem'])
    model = BD.load_play_model_so()
    clips = {}
    for handle, name, dump in BD.targets_dive():
        seq = acuclip.pose_sequence(dump, step=1)
        clips[name] = dict(handle=handle, dump=dump, seq=seq, ev=contacts.events(seq, dump),
                           mp=BD.clip_model(name, model, float(dump['Length'])))
    offsets = bake2.measure_contact_offsets([(n, c['dump'], c['seq'], c['ev']) for n, c in clips.items()])
    so = clips[MAIN_SO]
    assert so['handle'] == SO_HEX, so['handle']
    ex_dump = dumps.load_by_name(EXIT_NAME)
    ex_seq = acuclip.pose_sequence(ex_dump, step=1)
    mp = so['mp']
    tk = float(so['ev']['takeoff']['seconds'])
    p_to = (tk - mp['o']) / mp['r']
    T = p_to + K1_LEAD                 # k = 1

    root = DR.build_root(so['dump'])
    assert np.array_equal(root['F'], so['seq'].frames)
    disp_so = root['new']
    disp_orig_so = root['orig']
    disp_ex, root_w, tW, cut = exit_root_delta(root, mp, ex_dump, ex_seq)

    BC.EASE_S = EASE_S
    data = BC.build_chain(rig, m, so['seq'], so['dump'], so['ev'], mp, ex_seq, ex_dump, offsets, p_to=p_to, T=T,
                          disp_so=disp_so, disp_ex=disp_ex, z_shift_ex=0.0)
    data['seg_S']['disp_orig'] = disp_orig_so
    data['seg_E']['disp_orig'] = disp_ex
    rm = BC.run_main(rig, data)
    tl = data['tl']

    # ---- the SlideOver row (fails: its validation, the continuity and the root step, as in bake_chain)
    met_S, cont, step = rm['met_S'], rm['cont'], rm['step_cm']
    fails_S = BC.so_fail_list(met_S, cont, step)
    st_S = status_of(fails_S)
    so_enabled = st_S in ('PASS', 'OVERRIDE')
    # ---- the Exit row (its own validation; the hand-touch rule is not applied to the Exit, as in bake_chain)
    fails_E = list(rm['met_E']['fails'])
    st_E = status_of(fails_E)
    gap_pre = rm['gap_noease']
    gap_fail = gap_pre['max_deg'] is not None and gap_pre['max_deg'] > 60.0
    if gap_fail:
        st_E = 'FAIL'
    exit_enabled = (st_E in ('PASS', 'OVERRIDE')) and so_enabled

    so_override = ('preview: ' + '; '.join(fails_S)) if st_S == 'OVERRIDE' else None
    ex_override = ('preview: ' + '; '.join(fails_E)) if st_E == 'OVERRIDE' else None

    if write:
        # root track in the SlideOver patch; rotation and Reference tracks from the chain
        fS = rm['fS']
        write_patch(os.path.join(PATCH_DIR, SO_HEX + '.baked.json'), rig, fS['rot_tracks'], fS['trans_keys'],
                    root['keys'], so['dump']['Length'])
        write_patch(os.path.join(NOROOT_DIR, SO_HEX + '.baked.json'), rig, fS['rot_tracks'], fS['trans_keys'], None,
                    so['dump']['Length'])
        fE = rm['fE']
        write_patch(os.path.join(PATCH_DIR, EXIT_HEX + '.baked.json'), rig, fE['rot_tracks'], fE['trans_keys'], None,
                    ex_dump['Length'])
        write_patch(os.path.join(NOROOT_DIR, EXIT_HEX + '.baked.json'), rig, fE['rot_tracks'], fE['trans_keys'], None,
                    ex_dump['Length'])
        mapping = dict(clips=[
            mapping_entry(SO_HEX, MAIN_SO, SO_HEX + '.baked.json', so_enabled, override=so_override),
            mapping_entry(EXIT_HEX, EXIT_NAME, EXIT_HEX + '.baked.json', exit_enabled,
                          only_after=[MAIN_SO] if so_enabled else [MAIN_SO], override=ex_override),
        ])
        with open(MAPPING, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=1)
        with open(os.path.join(NOROOT_DIR, 'mapping_dive2.json'), 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=1)
        DR.write_outputs(so['dump'], root, OUT_DIR)
        if plots:
            title = 'SlideOver (new root) T %.2f played, A %.2f m, continuity %.0f deg' % (T, data['A'], cont['max_deg'])
            BC.plot_joined(os.path.join(OUT_DIR, 'joined_sheet.png'), rig, data['seg_S'] | {'name': MAIN_SO},
                           rm['arr_S'], data['seg_E'], rm['arr_E'], data['ob'], tl, title, show_root=True)
            BC.write_gif(os.path.join(OUT_DIR, 'joined_sheet.gif'), rig, data['seg_S'], rm['arr_S'], data['seg_E'],
                         rm['arr_E'], data['ob'], tl)
        write_report(root, rm, data, so, ex_dump, fails_S, st_S, fails_E, st_E, gap_pre, so_enabled, exit_enabled,
                     so_override, ex_override, offsets, p_to, T, disp_ex, root_w, tW, cut)
        summary = dict(T=float(T), p_to=float(p_to), A=float(data['A']), root=root['metrics'],
                       slideover=dict(name=MAIN_SO, status=st_S, enabled=so_enabled, fails=fails_S,
                                      continuity_deg=float(cont['max_deg']), step_cm=float(step),
                                      dstep_max_cm=float(met_S['dstep_max_cm']), touch=met_S['touch'],
                                      pen_box_cm=float(met_S['pen_box_cm']), pen_ground_cm=float(met_S['pen_ground_cm']),
                                      ground_all_cm=float(met_S['ground_all_cm']), dz_max_cm=float(met_S['dz_max_cm']),
                                      override=so_override),
                       exit=dict(name=EXIT_NAME, status=st_E, enabled=exit_enabled, fails=fails_E,
                                 gap_pre_ease_max_deg=gap_pre['max_deg'], gap_pre_ease_median_deg=gap_pre['median_deg'],
                                 ease_used=bool(rm['ease_used']), pen_box_cm=float(rm['met_E']['pen_box_cm']),
                                 pen_ground_cm=float(rm['met_E']['pen_ground_cm']),
                                 ground_all_cm=float(rm['met_E']['ground_all_cm']),
                                 dstep_max_cm=float(rm['met_E']['dstep_max_cm']), override=ex_override))
        with open(SUMMARY, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=1, default=float)
    return dict(root=root, rm=rm, data=data, fails_S=fails_S, st_S=st_S, fails_E=fails_E, st_E=st_E,
                so_enabled=so_enabled, exit_enabled=exit_enabled)


# ------------------------------------------------------------------------------------------------ report
def _fmt_touch(t):
    return '%d / %d' % (t.get('LeftHand', 0), t.get('RightHand', 0))


def write_report(root, rm, data, so, ex_dump, fails_S, st_S, fails_E, st_E, gap_pre, so_enabled, exit_enabled,
                 so_override, ex_override, offsets, p_to, T, disp_ex, root_w, tW, cut):
    rmet = root['metrics']
    met_S, met_E = rm['met_S'], rm['met_E']
    cont = rm['cont']
    lines = ['# T13: WD2 frontdive with a replaced root (SlideOver + gated Muraille_Exit)', '',
             'Generated by `v2/pipeline/dive_chain2.py` (root from `dive_root.py`). Times: clip seconds for the root, played '
             'seconds for the chain; heights in m unless marked; angles in degrees.', '',
             '## Time map (k = 1)', '',
             '- WD2 = 0.45 + (played - %.4f) : the WD2 takeoff is at the ACU takeoff (played %.4f = clip 0.450 s). The WD2 '
             'hand plant starts at played %.4f (clip %.3f s), the roll ends at played %.4f (WD2 1.18), the feet are down at '
             'WD2 1.20 (played %.4f, inside the Exit). Exit: clip = 0.208 + 0.9665 * p_e, gated at W = %.4f played (clip %.4f s).'
             % (p_to, p_to, T, mp_clip(so, T), T + (BD.WD2_ROLL_END - BD.WD2_PLANT[0]),
                T + (BD.WD2_FEET - BD.WD2_PLANT[0]), data['tl'].W, tW), '',
             '## Root (TrackID 0) replacement', '',
             '| quantity | value |', '|---|---|',
             '| far top edge crossed at (clip s) | %.3f (target %.2f) |' % (rmet['t_cross'] or float('nan'), rmet['t_far']),
             '| far edge y / near edge y / top z (dump frame) | %.4f / %.4f / %.4f |'
             % (rmet['far_y'], rmet['near_y'], rmet['top']),
             '| peak z (origin) | %.3f m at %.3f s |' % (rmet['z_peak'], rmet['t_z_peak']),
             '| min z over the box (clearance above the top) | %.3f m (%+.1f cm) |'
             % (rmet['z_min_over_box'], rmet['clearance_over_box_cm']),
             '| y error at 1.20 s vs original | %.2f cm |' % rmet['y_err_join_cm'],
             '| z at 1.20 s | %.4f m |' % rmet['z_join'],
             '| identical to original outside [0.45, 1.20] s (max deviation) | %.2e m |' % rmet['identity_max_dev_m'],
             '| origin beyond the far edge when z returns to 0 | %s m |' % ('%.2f' % rmet['y_beyond_far_at_z0']
                                                                       if rmet['y_beyond_far_at_z0'] is not None else 'n/a'),
             '| forward speed min / max (m/s) | %.2f / %.2f (monotone: %s) |' % (rmet['v_min'], rmet['v_max'], rmet['monotone']),
             '| forward speed at 0.45 / 0.60 / 0.70 / 0.80 / 0.90 / 1.00 / 1.10 / 1.20 s (m/s) | %s |'
             % ' / '.join('%.2f' % rmet['speed_at'][k] for k in ('0.45', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1', '1.2')),
             '| original speed at 0.45 / 1.20 s (m/s) | %.2f / %.2f |' % (rmet['v_orig_takeoff'], rmet['v_orig_join']),
             '', 'The far-edge time candidates (peak deceleration of the speed profile picks the smallest):', '',
             '| t_far (s) | min speed | max speed | peak decel (m/s2) | ok |', '|---|---|---|---|---|']
    for t in root['tried']:
        lines.append('| %.2f | %.2f | %.2f | %.1f | %s |' % (t['t_far'], t['min_speed'], t['max_speed'], t['max_decel'],
                                                          'yes' if t['ok'] else 'no'))
    lines += ['', '## Validation', '',
              '| clip | status | enabled | non-finger pen box / ground (cm) | bone below ground (cm) | hand touch L / R (fr, need 4) | '
              'max Delta step (cm/fr) | root step (cm/fr) | continuity at W max / median (deg) | gap to Nav_Run pre-ease max / median (deg) | '
              'ease | dz max (cm) | fails |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    lines.append('| %s | %s | %s | %.1f / %.1f | %.1f | %s | %.2f | %.2f | %.1f / %.1f | n/a | n/a | %.1f | %s |'
                 % (MAIN_SO.replace('xx_Parkour_Sprint_', ''), st_S, 'yes' if so_enabled else 'no',
                    met_S['pen_box_cm'], met_S['pen_ground_cm'], met_S['ground_all_cm'], _fmt_touch(met_S['touch']),
                    met_S['dstep_max_cm'], rm['step_cm'], cont['max_deg'], cont['median_deg'], met_S['dz_max_cm'],
                    '; '.join(fails_S) or 'none'))
    g = gap_pre
    gap_txt = ('%.1f / %.1f' % (g['max_deg'], g['median_deg'])) if g['max_deg'] is not None else 'n/a'
    row = [EXIT_NAME.replace('xx_Parkour_Muraille_Exit_', ''), st_E, 'yes' if exit_enabled else 'no',
           '%.1f / %.1f' % (met_E['pen_box_cm'], met_E['pen_ground_cm']), '%.1f' % met_E['ground_all_cm'], 'n/a',
           '%.2f' % met_E['dstep_max_cm'], 'n/a', 'n/a', gap_txt, 'yes' if rm['ease_used'] else 'no',
           '%.1f' % met_E['dz_max_cm'], '; '.join(fails_E) or 'none']
    lines.append('| ' + ' | '.join(row) + ' |')
    lines += ['', 'Continuity at W is the SlideOver tail against the Exit start (both at their nearest 60 Hz frame; the '
              'Exit row repeats it). Penetration is measured against the ground z = 0 and the box, on the written tracks.',
              '', 'Enabled: SlideOver %s, Exit %s (Exit only after the SlideOver).'
              % ('yes' if so_enabled else 'no', 'yes' if exit_enabled else 'no'), '']
    if so_override or ex_override:
        lines += ['Override (narrow failures, previewed): %s' % '; '.join(x for x in (so_override, ex_override) if x), '']
    lines += ['## Exit root convention (fix of the 32 cm penetration)', '',
              'The original Exit root drops 1.00 m before its played window: at clip 0.208 s the displacement is %.3f m '
              '(y %.3f) and it stays there, so the absolute frame puts the Exit ground at -1.0 m. The game joins the clip at '
              'the SlideOver root at W (%.3f m, z %.3f) and moves it by the change since the cut, so the Exit ground is z = 0. '
              'The bake now uses that delta everywhere (joined_root_E = root_S(W) + disp_E(t) - disp_E(0.208)). The old '
              'EXIT_Z_SHIFT (+0.751 m) is not needed. Original Exit feet over its window: %s.'
              % (cut[2], cut[1], root_w[1], root_w[2], _exit_feet_text(ex_dump, disp_ex)), '']
    lines += ['## Gates and files', '',
              '- Gated Exit: only_after [%s], arm_seconds 2.0, no fade-in; WD2 time at the cut = WD2 time at W.' % MAIN_SO,
              '- Ease: applied %s (pre-ease gap max %s deg, ease zone %.2f s).'
              % ('yes' if rm['ease_used'] else 'no',
                 '%.1f' % gap_pre['max_deg'] if gap_pre['max_deg'] is not None else 'n/a', EASE_S),
              '- Files: patches\\%s.baked.json (rotations + Reference + TrackID 0), patches\\%s.baked.json, '
              'patches\\mapping_dive2.json, patches_noroot\\ (body only, comparison), root_curve.png/json, joined_sheet.png/gif.'
              % (SO_HEX, EXIT_HEX), '',
              'Offsets (median over the Sprint SlideOver dumps): Hand %.3f m, Foot %.3f m, FootThumb1 %.3f m.'
              % (offsets['Hand'], offsets['Foot'], offsets['FootThumb1']), '']
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def mp_clip(so, t_played):
    """Clip seconds of a played time on the SlideOver."""
    return so['mp']['o'] + so['mp']['r'] * t_played


def _exit_feet_text(ex_dump, disp_joined):
    seq = acuclip.pose_sequence(ex_dump, step=1)
    E = acuclip.entity_positions(seq)
    F = seq.frames
    win = bake2.window_frames(BC.EXIT_O, BC.EXIT_R * BC.EXIT_WIN, int(F[-1]))
    disp_abs = acuclip.markers(ex_dump, F)['disp']
    out = []
    for label, d in (('absolute', disp_abs), ('joined delta', disp_joined)):
        P = E + d[:, None, :]
        lf = P[win][:, _acu_index(seq, 'LeftFoot'), 2]
        rf = P[win][:, _acu_index(seq, 'RightFoot'), 2]
        out.append('%s min %.2f m (feet)' % (label, float(min(lf.min(), rf.min()))))
    return '; '.join(out)


def _acu_index(seq, name):
    return next(i for i, b in enumerate(seq.bones) if zlib.crc32(name.encode()) == b['hash'])


if __name__ == '__main__':
    out = build_all(write=True, plots=True)
    print('SlideOver %s (%s); Exit %s (%s)' % (MAIN_SO, out['st_S'], EXIT_NAME, out['st_E']))
    for f in out['fails_S'] + out['fails_E']:
        print('  fail:', f)
    print(json.dumps(out['root']['metrics'], indent=1, default=float)[:1200])
