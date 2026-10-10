"""Match every session capture against its original ACU clip: table, PNG per capture, summary and play_model.json.

usage: python matchclip_report.py [--captures DIR] [--out-dir DIR] [--exclude-poor]

For each capture with an entity-space pose and an original dump of its trigger clip (matchclip.py):
  - window (match_window), time map o, r and yaw (fit_time_map), RMS of the fit, RMS with the naive map
    (o = 0, r = 1, same yaw), pelvis correction dz = a * rise + b (pelvis_correction);
  - out-dir\\<capture name>.png: hand/foot z relative to Hips (in-game vs predicted), Hips z offset vs rise, RMS
    landscape over (o, r) at the fitted yaw.
Writes out-dir\\summary.txt (the table and summary) and out-dir\\play_model.json:
  {"per_clip": {clip: {"o", "r", "window", "n"}}, "median": {"o", "r", "yaw"}, "pelvis": {"a", "b"}}
  per_clip values are medians over the captures of that clip; window is the median window length in seconds
  (matchclip window, s); median and pelvis are medians over all used captures. By default every fitted capture
  is used, poor ones included (the RMS threshold in matchclip.POOR_RMS_CM is far below what the game gives);
  --exclude-poor leaves them out.
"""
import argparse
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import capture  # noqa: E402
import config  # noqa: E402
import matchclip as mc  # noqa: E402

DEFAULT_CAPTURES = os.path.join(config.V2, 'data', 'captures', 'session1_20261008')
DEFAULT_OUT = os.path.join(config.OUT, 'match')
LIMBS = ('LeftHand', 'RightHand', 'LeftFoot', 'RightFoot')
COLORS = {'LeftHand': 'tab:blue', 'RightHand': 'tab:orange', 'LeftFoot': 'tab:green', 'RightFoot': 'tab:red'}


def capture_paths(folder):
    files = glob.glob(os.path.join(folder, 'cap_*.json'))

    def key(p):
        m = re.search(r'_(\d+)\.json$', p)
        return int(m.group(1)) if m else 0
    return sorted(files, key=key)


def _nan(x):
    return None if x is None or not np.isfinite(x) else float(x)


def analyse(path):
    """One row per capture. Rows that cannot be fitted carry 'skip' (reason) and no fit."""
    cap = capture.load(path)
    row = dict(file=os.path.basename(path), index=int(re.search(r'_(\d+)\.json$', path).group(1)),
               trigger=cap.trigger, skip=None)
    if not cap.pose_ok or cap.bones_space != 'entity':
        row['skip'] = 'no entity-space pose'
        return row, None
    idx, dump = mc.trigger_dump(cap)
    if dump is None:
        row['skip'] = 'no original dump for the trigger clip'
        return row, None
    w = mc.match_window(cap, idx)
    try:
        fit = mc.fit_time_map(cap, dump, w)
        pc = mc.pelvis_correction(cap, dump, fit)
    except ValueError as e:
        row['skip'] = str(e)
        return row, None
    row.update(
        clip=w['clip_name'], window_s=w['length_s'], t_start=w['t_start'], t_end=w['t_end'],
        end_reason=w['end_reason'], next_name=w['next_name'],
        o=fit['o'], r=fit['r'], yaw=fit['yaw_deg'], rms=fit['rms_cm'], rms_naive=fit['rms_naive_cm'],
        rms_r1=fit['rms_r1']['rms_cm'], rms_r1_o=fit['rms_r1']['o'], rms_r1_yaw=fit['rms_r1']['yaw_deg'],
        rise_max=pc['rise_max'], a=pc['fit_z']['a'], b=pc['fit_z']['b'], r2=pc['fit_z']['r2'],
        n_frames=fit['n_frames'], n_window=int(np.count_nonzero((cap.t >= w['t_start']) & (cap.t <= w['t_end']))),
        at_edge=fit['at_edge'], poor=fit['poor'], dxy_cm=pc['dxy_mean_cm'], trigger_name=dump['Name'])
    row['reason'] = poor_reason(row)
    return row, dict(cap=cap, dump=dump, window=w, fit=fit, pc=pc)


def poor_reason(row):
    """Why a fit is poor (empty string when it is not). Uses the fit numbers only."""
    if row.get('skip') or not row['poor']:
        return ''
    why = []
    if row['at_edge']['o'] or row['at_edge']['r']:
        why.append('best time map on the search border (o %s, r %s)' % (
            'edge' if row['at_edge']['o'] else 'inside', 'edge' if row['at_edge']['r'] else 'inside'))
    if row['n_frames'] < 0.8 * row['n_window']:
        why.append('only %d of %d pose frames fall inside the clip' % (row['n_frames'], row['n_window']))
    if row['rms_r1'] < 1.2 * row['rms']:
        why.append('r = 1 fits as well (%.1f cm), so r is not identified' % row['rms_r1'])
    if not why:
        why.append('the bones do not follow this clip in its time shape: a blend or a different clip variant may '
                   'play in the game')
    return '; '.join(why)


def plot_capture(row, d, out_dir):
    cap, dump, fit, pc, w = d['cap'], d['dump'], d['fit'], d['pc'], d['window']
    ctx = mc._prepare(cap, dump, w)
    rel, hips, valid = mc._interp_frames(ctx, fit['o'], fit['r'])
    a = np.radians(fit['yaw_deg'])
    c, s = np.cos(a), np.sin(a)
    x, y = rel[..., 0], rel[..., 1]
    pred = np.stack([c * x - s * y, s * x + c * y, rel[..., 2]], axis=-1)
    tr = ctx.t_rel
    v = valid

    fig = plt.figure(figsize=(12, 9.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.1])
    ax = fig.add_subplot(gs[0, :])
    for k, b in enumerate(mc.FIT_BONES):
        if b not in LIMBS:
            continue
        ax.plot(tr, ctx.game_rel[:, k, 2], '-', color=COLORS[b], label='%s in-game' % b)
        ax.plot(tr[v], pred[v, k, 2], '--', color=COLORS[b], label='%s predicted' % b)
    ax.set_xlabel('time since clip start (s)')
    ax.set_ylabel('z relative to Hips (m)')
    ax.set_title('Hand and foot height relative to Hips: in-game (solid) vs predicted (dashed)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=4, loc='lower right')

    ax2 = fig.add_subplot(gs[1, 0])
    rise = pc['rise']
    ax2.plot(rise, pc['dhips'][:, 2], 'o', ms=3, color='k', label='frames')
    if np.isfinite(pc['fit_z']['a']):
        xs = np.linspace(rise.min(), rise.max(), 10) if rise.size else np.zeros(0)
        ax2.plot(xs, pc['fit_z']['a'] * xs + pc['fit_z']['b'], '-', color='tab:red',
                 label='a = %.3f, b = %.4f m, R2 = %.3f' % (pc['fit_z']['a'], pc['fit_z']['b'], pc['fit_z']['r2']))
    lim = max(abs(rise).max() if rise.size else 0.1, 0.05)
    ax2.plot([0, lim], [0, lim], ':', color='grey', label='dz = rise')
    ax2.set_xlabel('entity z rise since clip start (m)')
    ax2.set_ylabel('Hips z offset, game minus clip (m)')
    ax2.set_title('Pelvis correction')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[1, 1])
    og, rg, L = mc.fit_landscape(cap, dump, w, fit['yaw_deg'])
    Lp = np.where(np.isfinite(L), L, np.nan)
    im = ax3.imshow(np.log10(np.clip(Lp, 0.1, None)).T, origin='lower', aspect='auto',
                    extent=[og[0], og[-1], rg[0], rg[-1]], cmap='viridis')
    ax3.plot(fit['o'], fit['r'], 'r*', ms=12, label='best (%.2f, %.2f)' % (fit['o'], fit['r']))
    ax3.set_xlabel('o (s)')
    ax3.set_ylabel('r')
    ax3.set_title('RMS landscape at yaw %.0f deg (log10 cm)' % fit['yaw_deg'])
    ax3.legend(fontsize=8, loc='upper right')
    fig.colorbar(im, ax=ax3, fraction=0.046)

    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    txt = ('%s\ntrigger %s\nwindow %.3f s (%s), clip %s\n'
           'best o %.3f s, r %.3f, yaw %.1f deg, RMS %.2f cm (naive %.2f cm, r = 1 fit %.2f cm)\n'
           'pelvis dz = %.3f * rise %+.4f m (R2 %.3f), max rise %.3f m%s') % (
        row['file'], row['trigger'], row['window_s'], row['end_reason'],
        row['next_name'] or '-', fit['o'], fit['r'], fit['yaw_deg'], fit['rms_cm'], fit['rms_naive_cm'],
        fit['rms_r1']['rms_cm'], pc['fit_z']['a'], pc['fit_z']['b'], pc['fit_z']['r2'], pc['rise_max'],
        ('\nPOOR: ' + row['reason']) if row['reason'] else '')
    ax4.text(0.0, 1.0, txt, va='top', ha='left', family='monospace', fontsize=9)
    fig.suptitle('Capture %d vs original clip' % row['index'])
    fig.tight_layout()
    path = os.path.join(out_dir, os.path.splitext(row['file'])[0] + '.png')
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _median(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return float(np.median(vals)) if vals else None


def play_model(rows, exclude_poor):
    used = [r for r in rows if not r.get('skip') and not (exclude_poor and r['poor'])]
    per_clip = {}
    for name in sorted({r['clip'] for r in used}):
        group = [r for r in used if r['clip'] == name]
        per_clip[name] = dict(o=_median(group, 'o'), r=_median(group, 'r'), window=_median(group, 'window_s'),
                              n=len(group))
    return dict(
        per_clip=per_clip,
        median=dict(o=_median(used, 'o'), r=_median(used, 'r'), yaw=_median(used, 'yaw')),
        pelvis=dict(a=_median(used, 'a'), b=_median(used, 'b')),
    )


def table_lines(rows):
    head = ('%3s %-42s %6s %6s %6s %6s %7s %7s %7s %7s %8s %6s' % (
        '#', 'trigger', 'win s', 'o s', 'r', 'yaw', 'RMS', 'naive', 'r=1', 'max rise', 'a / b m', 'R2'))
    lines = [head, '-' * len(head)]
    for r in rows:
        if r.get('skip'):
            lines.append('%3d %-42s skipped: %s' % (r['index'], r['trigger'][:42], r['skip']))
            continue
        lines.append('%3d %-42s %6.3f %6.2f %6.2f %6.1f %7.2f %7.2f %7.2f %7.3f %4.2f/%+6.4f %6.3f%s' % (
            r['index'], r['trigger'][:42], r['window_s'], r['o'], r['r'], r['yaw'], r['rms'], r['rms_naive'],
            r['rms_r1'], r['rise_max'], r['a'], r['b'], r['r2'], '  POOR' if r['poor'] else ''))
    return lines


def summary_lines(rows, pm):
    fitted = [r for r in rows if not r.get('skip')]
    poor = [r for r in fitted if r['poor']]
    lines = ['', 'Summary', '-------']
    lines.append('captures: %d total, %d skipped, %d fitted, %d poor (RMS > %.1f cm)' % (
        len(rows), len(rows) - len(fitted), len(fitted), len(poor), mc.POOR_RMS_CM))
    if fitted:
        lines.append('median o %.3f s, r %.3f, yaw %.1f deg; median RMS %.2f cm (naive %.2f cm, r = 1 %.2f cm)' % (
            _median(fitted, 'o'), _median(fitted, 'r'), _median(fitted, 'yaw'), _median(fitted, 'rms'),
            _median(fitted, 'rms_naive'), _median(fitted, 'rms_r1')))
        lines.append('pelvis (median over fits): a %.3f, b %+.4f m, R2 median %.3f' % (
            pm['pelvis']['a'], pm['pelvis']['b'], _median(fitted, 'r2')))
    for name, v in pm['per_clip'].items():
        lines.append('clip %s: o %.3f, r %.3f, window %.3f s, n %d' % (name, v['o'], v['r'], v['window'], v['n']))
    if poor:
        lines.append('poor fits:')
        for r in poor:
            lines.append('  #%d %s RMS %.2f cm: %s' % (r['index'], r['trigger'], r['rms'], r['reason']))
    skipped = [r for r in rows if r.get('skip')]
    for r in skipped:
        lines.append('skipped #%d %s: %s' % (r['index'], r['trigger'], r['skip']))
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--captures', default=DEFAULT_CAPTURES)
    ap.add_argument('--out-dir', default=DEFAULT_OUT)
    ap.add_argument('--exclude-poor', action='store_true', help='leave poor fits out of play_model.json')
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for p in capture_paths(args.captures):
        row, d = analyse(p)
        if d is not None:
            plot_capture(row, d, args.out_dir)
        rows.append(row)
        print('done %s' % row['file'], file=sys.stderr)

    pm = play_model(rows, args.exclude_poor)
    lines = table_lines(rows) + summary_lines(rows, pm)
    text = '\n'.join(lines) + '\n'
    print(text)
    with open(os.path.join(args.out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(text)
    with open(os.path.join(args.out_dir, 'play_model.json'), 'w', encoding='utf-8') as f:
        json.dump(pm, f, indent=2)
        f.write('\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
