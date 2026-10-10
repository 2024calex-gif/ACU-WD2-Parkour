"""T9e transition variants for in-game A/B testing (out\\variants).

Bakes four variants of the shipped T9e (bake2.bake_all_t9e with fade / tail options) into out\\variants\\<name>\\,
then reads every written patch file back (what the game samples) and measures the transition:

  * validation of the bake (window penetration, hand touch, Delta step): bake2's own metrics
  * what the game plays of our clip: from o (entry) to o + window + POST_WIN (exit, while the game blends into the
    next clip). Joint steps are per 60 Hz frame over pairs of frames the game plays:
      entry: [o, o + STEP_WIN]            exit: [o + window - STEP_WIN, o + window + POST_WIN]
  * the gap the game would blend across: our pose vs the original ACU pose at o (the game blends from the previous
    clip, which matched the original entry), and the largest gap over [o + window, o + window + POST_WIN] (the game
    blends from our clip into the next one there)
  * penetration after o + window (the tail)

Run:  python variants.py   (bakes all four variants, compares with out\\patches, writes out\\variants\\report.md)
The installed patch folder (out\\patches) is only read: it is the source of the enabled flags and of the byte check.
"""
import filecmp
import json
import os
import sys

import numpy as np

import acuclip
import bake2
import config
import contacts
import retarget as R

FPS = acuclip.FPS
BASE_DIR = config.PATCHES
BASE_MAP = os.path.join(BASE_DIR, 'mapping.json')
VAR_ROOT = os.path.join(config.OUT, 'variants')
REPORT = os.path.join(VAR_ROOT, 'report.md')
STEP_WIN = 0.10            # s, length of the entry and exit step windows
POST_WIN = 0.25            # s, after o + window: the game's blend into the next clip

VARIANTS = [
    dict(name='A_current', fi=None, fo=None, tail='orig',
         note='current T9e defaults: fade-in min(0.20, 0.30 w), fade-out min(0.25, 0.35 w), tail orig'),
    dict(name='B_short', fi=0.08, fo=0.10, tail='orig',
         note='fade-in 0.08 s, fade-out 0.10 s, tail orig'),
    dict(name='C_nofade_wd2tail', fi=0.0, fo=0.0, tail='wd2',
         note='no fades (WD2 pose from o); no fade-out; WD2 motion to the clip end'),
    dict(name='D_short_in_wd2tail', fi=0.08, fo=0.0, tail='wd2',
         note='fade-in 0.08 s; no fade-out; WD2 motion to the clip end'),
]


def read_patch(path, rig):
    """(rotation keys {ac bone: keys}, Reference position keys or None) as written in a patch file."""
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    rot = {}
    pos_keys = None
    for tr in d['Tracks']:
        if int(tr['codec']) == 27:
            pos_keys = tr['Keyframes']
        else:
            rot[rig.ac_by_hash[int(tr['TrackID'])]] = tr['Keyframes']
    return rot, pos_keys


def _pairs(sel):
    """Indices k with frames k and k + 1 both selected."""
    sel = np.asarray(sel, bool)
    return np.nonzero(sel[:-1] & sel[1:])[0]


def game_view(rig, dump, seq, ev, rot_keys, pos_keys, o, window):
    """Measurements of one clip as the game samples it (tracks at 60 Hz) against the original clip.
    rot_keys = {} and pos_keys = None give the original clip."""
    F = np.asarray(seq.frames, int)
    nf = int(F[-1])
    t = F / FPS
    e = o + window
    orig_rot = np.asarray(seq.local_rot, float)
    orig_ref = np.asarray(seq.local_pos[:, 0], float)
    rot = orig_rot.copy()
    for ai, keys in rot_keys.items():
        rot[:, ai] = bake2.sample_track_rot(keys, F)
    pat = sorted(rot_keys)
    pos = orig_ref.copy() if pos_keys is None else bake2.sample_track_pos(pos_keys, F)
    off = (pos - orig_ref) * 100.0                          # cm, root offset from the original
    body = sorted(set(rig.map.values()))
    fingers = set(bake2.finger_bones(rig, body))
    body_nf = [bi for bi in body if bi not in fingers]
    ob = ev['obstacle']
    P = acuclip.world_positions(bake2.composed(seq, rot, pos), dump)
    Po = acuclip.world_positions(seq, dump)
    win = (t >= o - 1e-9) & (t <= e + 1e-9)
    tail = t > e + 1e-9
    pb, pg = bake2.penetration_parts(P[win][:, body_nf], ob)
    ob_, og_ = bake2.penetration_parts(Po[win][:, body_nf], ob)
    if np.any(tail):
        tb, tg = bake2.penetration_parts(P[tail][:, body_nf], ob)
    else:
        tb = tg = 0.0

    def gap_bone(f):
        return float(np.max(bake2.angle_deg(rot[f, pat], orig_rot[f, pat]))) if pat else 0.0

    def gap_root(f):
        return float(np.linalg.norm(pos[f] - orig_ref[f]) * 100.0)

    def idx_at(sec):
        return int(min(max(int(round(sec * FPS)), 0), nf))

    f_o = idx_at(o)
    f_e = idx_at(e)
    post = np.arange(f_e, idx_at(e + POST_WIN) + 1)
    entry = (t >= o - 1e-9) & (t <= o + STEP_WIN + 1e-9)
    exit_ = (t >= e - STEP_WIN - 1e-9) & (t <= e + POST_WIN + 1e-9)
    pe = _pairs(entry)
    px = _pairs(exit_)

    def joint_steps(arr, pairs):
        if pairs.size == 0 or not pat:
            return 0.0
        return float(np.max(bake2.angle_deg(arr[pairs + 1][:, pat], arr[pairs][:, pat])))

    def root_steps(pairs):
        if pairs.size == 0:
            return 0.0
        return float(np.max(np.linalg.norm(off[pairs + 1] - off[pairs], axis=1)))

    return dict(
        pat=pat,
        gap_o_deg=gap_bone(f_o), gap_o_cm=gap_root(f_o),
        gap_e_deg=gap_bone(f_e), gap_e_cm=gap_root(f_e),
        post_gap_deg=max(gap_bone(f) for f in post) if pat else 0.0,
        post_gap_cm=max(gap_root(f) for f in post),
        step_entry_deg=joint_steps(rot, pe), step_exit_deg=joint_steps(rot, px),
        orig_step_entry_deg=joint_steps(orig_rot, pe), orig_step_exit_deg=joint_steps(orig_rot, px),
        root_step_entry_cm=root_steps(pe), root_step_exit_cm=root_steps(px),
        root_step_all_cm=root_steps(_pairs(np.ones(len(t), bool))),
        pen_win_cm=max(pb, pg) * 100.0,
        pen_orig_win_cm=max(ob_, og_) * 100.0,
        pen_tail_cm=max(tb, tg) * 100.0,
    )


def table_rows(results, base):
    """Per clip x variant table lines (plus one original row per clip)."""
    lines = ['| clip | variant | fi / fo / tail | enabled | window pen (cm) | touch L / R (frames) | bake PASS | '
             'entry step (deg/frame) | exit step (deg/frame) | root step entry / exit (cm/frame) | '
             'game gap at o (deg / cm) | game gap at o+window (deg / cm) | '
             'max gap o+window..+%.2f s (deg / cm) | tail pen (cm) |' % POST_WIN,
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for c in results['clips']:
        for vname in results['variants']:
            r = results['data'][vname][c['name']]
            v = r['view']
            mt = r['metrics']
            lines.append('| %s | %s | %s | %s | %.1f | %d / %d | %s | %.1f | %.1f | %.2f / %.2f | %.1f / %.2f | '
                         '%.1f / %.2f | %.1f / %.2f | %.1f |'
                         % (c['short'], vname, r['fitxt'], 'yes' if r['enabled'] else 'no', v['pen_win_cm'],
                            mt['touch']['LeftHand'], mt['touch']['RightHand'],
                            'PASS' if not mt['fail'] else 'FAIL',
                            v['step_entry_deg'], v['step_exit_deg'], v['root_step_entry_cm'], v['root_step_exit_cm'],
                            v['gap_o_deg'], v['gap_o_cm'], v['gap_e_deg'], v['gap_e_cm'],
                            v['post_gap_deg'], v['post_gap_cm'], v['pen_tail_cm']))
        b = base[c['name']]
        lines.append('| %s | original ACU (reference) | - | - | %.1f | - | - | %.1f | %.1f | - | 0 / 0 | 0 / 0 | 0 / 0 | %.1f |'
                     % (c['short'], b['pen_win_cm'], b['step_entry_deg'], b['step_exit_deg'], b['pen_tail_cm']))
    return lines


def summary_lines(results):
    lines = ['| variant | clips PASS | touch fails (< %d frames) | max window pen (cm, PASS metric) | '
             'max entry step (deg) | max exit step (deg) | max game gap at o (deg) | '
             'max game gap o+window..+%.2f s (deg) | max tail pen (cm) | max root step (cm/frame) |'
             % (bake2.T9D_TOUCH, POST_WIN),
             '|---|---|---|---|---|---|---|---|---|---|']
    for vname in results['variants']:
        rs = [results['data'][vname][c['name']] for c in results['clips']]
        lines.append('| %s | %d / %d | %d | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.2f |'
                     % (vname,
                        sum(1 for r in rs if not r['metrics']['fail']), len(rs),
                        sum(1 for r in rs if min(r['metrics']['touch']['LeftHand'],
                                                 r['metrics']['touch']['RightHand']) < bake2.T9D_TOUCH),
                        max(r['metrics']['pen_patch_cm'] for r in rs),
                        max(r['view']['step_entry_deg'] for r in rs), max(r['view']['step_exit_deg'] for r in rs),
                        max(r['view']['gap_o_deg'] for r in rs), max(r['view']['post_gap_deg'] for r in rs),
                        max(r['view']['pen_tail_cm'] for r in rs),
                        max(r['view']['root_step_all_cm'] for r in rs)))
    return lines


def compare_with_installed(vname):
    """Byte comparison of a variant folder with out\\patches (every .json there)."""
    vdir = os.path.join(VAR_ROOT, vname)
    names = sorted(f for f in os.listdir(BASE_DIR) if f.endswith('.json') and f != 'mapping_t9e_strict.json')
    diff = []
    for n in names:
        b = os.path.join(vdir, n)
        if not os.path.exists(b):
            diff.append('%s (missing)' % n)
        elif not filecmp.cmp(os.path.join(BASE_DIR, n), b, shallow=False):
            diff.append(n)
    return names, diff


def main(bake=True):
    rig = R.Rig()
    clips = []
    for handle, name, dump in bake2.targets():
        seq = acuclip.pose_sequence(dump, step=1)
        clips.append(dict(handle=handle, name=name, short=name.replace('xx_Parkour_Sprint_SkipOver_', ''),
                          dump=dump, seq=seq, ev=contacts.events(seq, dump)))
    base_enabled = bake2.enabled_from_mapping(BASE_MAP)
    model = bake2.load_play_model()
    measured = bake2.load_measured()
    results = dict(variants=[v['name'] for v in VARIANTS], clips=[], data={}, cmp={})
    base = {}
    for c in clips:
        o, window, _src = bake2.play_for(c['name'], float(c['dump']['Length']), model, measured)
        c['o'], c['window'] = o, window
        results['clips'].append(dict(name=c['name'], short=c['short'], handle=c['handle']))
        ov = game_view(rig, c['dump'], c['seq'], c['ev'], {}, None, o, window)
        base[c['name']] = dict(pen_win_cm=ov['pen_orig_win_cm'], step_entry_deg=ov['orig_step_entry_deg'],
                               step_exit_deg=ov['orig_step_exit_deg'], pen_tail_cm=ov['pen_tail_cm'])
    for v in VARIANTS:
        vdir = os.path.join(VAR_ROOT, v['name'])
        rows, _t9d, _t9c, _b, _off, _jag = bake2.bake_all_t9e(
            write=True, fi=v['fi'], fo=v['fo'], tail=v['tail'], out_dir=vdir, enabled_from=BASE_MAP, compare=False) \
            if bake else (None, None, None, None, None, None)
        if rows is None:
            raise SystemExit('--no-bake is not supported: the bake returns the validation metrics')
        by_name = {r['name']: r for r in rows}
        results['data'][v['name']] = {}
        for c in clips:
            rot_keys, pos_keys = read_patch(os.path.join(vdir, '%s.baked.json' % c['handle']), rig)
            view = game_view(rig, c['dump'], c['seq'], c['ev'], rot_keys, pos_keys, c['o'], c['window'])
            row = by_name[c['name']]
            fi_r, fo_r = row['fades']
            results['data'][v['name']][c['name']] = dict(
                view=view, metrics=row['metrics'], fades=(fi_r, fo_r), enabled=base_enabled[c['handle'].upper()],
                fitxt='%.2f / %.2f / %s' % (fi_r, fo_r, v['tail']))
        names, diff = compare_with_installed(v['name'])
        results['cmp'][v['name']] = dict(files=len(names), diff=diff)
    write_report(results, base)
    return results


def write_report(results, base):
    lines = ['# T9e transition variants for in-game A/B (out\\variants)', '',
             'Generated by `v2/pipeline/variants.py`. Each variant is baked with `bake2.bake_all_t9e(compare=False)` into '
             '`out\\variants\\<name>\\` (patch files and mapping.json only). Enabled flags are copied from '
             '`out\\patches\\mapping.json`, not recomputed. Times in seconds of the clip; heights in cm; angles in degrees.',
             '', '## Variants', '']
    for v in VARIANTS:
        lines.append('- **%s**: fade-in %s, fade-out %s, tail %s. %s' % (
            v['name'], 'default' if v['fi'] is None else '%.2f s' % v['fi'],
            'default' if v['fo'] is None else '%.2f s' % v['fo'], v['tail'], v['note']))
    lines += ['', '## What is measured', '',
              '- Window penetration and touch: bake2 validation, non-finger body incl. feet over [o, span end]. PASS as in '
              'bake2: penetration <= %.0f cm, touch >= %d frames per hand, Delta step <= %.0f cm/frame.'
              % (bake2.T9C_PEN_CM, bake2.T9D_TOUCH, bake2.T9C_DELTA_STEP_CM),
              '- Entry step: largest angle between consecutive 60 Hz samples of any patched bone over frame pairs in '
              '[o, o + %.2f s] (what the game plays from o). Exit step: the same over [o + window - %.2f s, o + window + %.2f s].'
              % (STEP_WIN, STEP_WIN, POST_WIN),
              '- Root step: change of (Reference position - original) per 60 Hz frame in the same windows.',
              '- Game gap at o: our pose minus the original pose at o (the game blends from the previous clip, which matches '
              'the original entry, into ours). Game gap at o + window: the same at o + window. Max gap over '
              '[o + window, o + window + %.2f s]: the difference from the original pose while the game blends into the next clip.'
              % POST_WIN,
              '- Tail pen: non-finger penetration after o + window (for tail orig it is the ACU original).',
              '- The original row is the unpatched clip: its own steps and penetration in the same windows.', '',
              '## Comparison table (per clip x variant)', ''] + table_rows(results, base) + [
              '', '## Summary over the ten clips', ''] + summary_lines(results) + ['', '## A_current against out\\patches', '']
    for vname, c in results['cmp'].items():
        lines.append('- %s: %d files compared with out\\patches (10 patch files and mapping.json); differences: %s.'
                     % (vname, c['files'], ', '.join(c['diff']) if c['diff'] else 'none (byte-identical)'))
    lines += ['', '## Failure reasons (bake2 PASS criteria)', '']
    for vname in results['variants']:
        lines.append('### %s' % vname)
        for c in results['clips']:
            r = results['data'][vname][c['name']]
            why = '; '.join(r['metrics']['fails']) if r['metrics']['fails'] else 'none (PASS)'
            lines.append('- %s: %s' % (c['short'], why))
        lines.append('')
    os.makedirs(VAR_ROOT, exist_ok=True)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main(bake='--no-bake' not in sys.argv)
