"""Markdown report and plots for the SlideOver play model (called by slideover_model.main).

write_report(res, md_path, plot_dir) writes slideover_report.md and three PNG plots next to it:
  slideover_roots.png     root distance past the near edge (m) and root z (m) against played time, one panel per
                          clip; window shaded; ACU takeoff and feet down; WD2 frontdive aligned at the ACU takeoff
  slideover_timeline.png  one bar per clip: played window, ACU takeoff..feet down, WD2 dive on the same time line
  slideover_fits.png      fitted (o, r) of every occurrence, coloured by RMS
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import dumps  # noqa: E402
import slideover_model as sm  # noqa: E402

# Filled in after the run (see the report file).
FEASIBILITY = """**Time map.** The Sprint SlideOver clips are played from clip time o of about 0.42 s, at a rate of about 1.0. The
only clip with several usable fits is Fd100cm-Up100cm (7 of 20 occurrences; o 0.38-0.44 s, r 1.02-1.13). Its o agrees
with the SkipOver start offset (o about 0.38 s, `play_model.json`). Map the dive with o = 0.42 s and r = 1.0 to 1.05;
the choice moves events by under 35 ms over the window.

**Dominant clip, Fd100cm-Up100cm (20 occurrences, 11 into Muraille_Exit_Dn100, 9 into Dn500).** Played window W = 0.744 s.
The ACU takeoff falls at played 0.03 s (the trigger), the hand touch on the top at 0.47 s, feet down at 0.70 s. The window
ends 0.05 s after the feet touch: the next clip starts when the root reaches the landing marker (clip 1.20 s).
- Dive at rate 1 from the ACU takeoff: roll ends at 0.755 s (12 ms past W), feet down at 0.775 s (32 ms past W), run-out
  at 1.175 s (0.43 s past W). The roll and landing fit within the 0.25 s out-blend; the run-out does not.
- Recommended: time factor k = 0.89 on the WD2 durations (k ACU = 0.89 puts WD2 feet down on the ACU feet down; k roll =
  0.98 is the limit for the roll to end at W, so the roll is not the binding constraint). Result with takeoff at 0.03 s:
  hands on the ground 0.29-0.44 s, roll 0.44-0.68 s, feet down 0.70 s, run-out 1.05 s (after W + blend).
- No separate roll compression is needed; the roll shortens with the global k.

**Tail: can WD2 motion continue past the window?** Not with the game as it is. At W the game starts Muraille_Exit (every
SlideOver occurrence, 32 of 32, has a next clip at W) and blends SlideOver out over about 0.25 s. WD2 motion after
W + 0.25 s (about 1.0 s played) is never shown unless the next clip is also changed. Bake the dive so the feet are down
before W (0.70 s, k = 0.89), and make the Muraille_Exit landing pose (first frames) match the WD2 landing pose. The WD2
run-out (1.2-1.6 s WD2) belongs to Exit.

**Other clips (fewer fits; these need a check in game).**
| clip | fits (usable) | map | k used | takeoff | roll end | feet down | W | note |
|---|---|---|---|---|---|---|---|---|
| Fd100cm-Up050cm | 1 (0) | ASSUMED | 0.72 | 0.04 | 0.57 | 0.58 | 0.63 | ACU span 0.54 s: WD2 sped up 1.4x |
| Fd200cm-Up100cm | 3 (0) | ASSUMED | 1.25 | 0.03 | 0.94 | 0.97 | 0.95 | roll ends at W; feet +0.02 s past W |
| Fd200cm-Up200cm | 3 (1) | own (o 0.47, r 1.02) | 1.09 | 0.03 | 0.82 | 0.84 | 0.82 | the Fd250-Dn300 exit starts 0.27 s before the ACU lands |
| Fd300cm-Up100cm | 3 (1) | own (o 0.40, r 0.97) | 1.53 | 0.07 | 1.19 | 1.22 | 1.29 | ACU span 1.15 s: WD2 slowed 1.5x, or k = 1 and hold the landing |

- Hands: the ACU hand touches the obstacle top in four of the five clips, for 0.03-0.2 s (Fd100-Up100 at 0.47 s played,
  33 ms, the shortest). Pair the WD2 hand plant end (0.44 s with k = 0.89) with that touch. Fd100-Up050 has no hand touch.
- Root: the ACU root passes the near edge at played 0.23 s, the far edge at 0.55 s, the landing marker at 0.74 s
  (Fd100-Up100). Its vertical root motion reaches the obstacle height at about 0.2 s and is back at 0 by about 0.7 s.
  Scaling the WD2 dive by k = 0.89 makes its root about 12 % faster. WD2 root travel was not measured here.

**Recommendation.** Time-map by event pairs: WD2 takeoff to ACU takeoff (clip 0.45 s), WD2 hand plant end to the ACU
hand touch where it exists, WD2 feet down to the ACU feet down (clip 1.15 s for Fd100-Up100). Use k = min(k ACU, k roll)
per clip. Bake for Fd100-Up100 first (k = 0.89). Treat the other clips as provisional until their own fits are checked
in game: 25 of 37 fitted occurrences are unusable (RMS 16-56 cm, yaw -120 to +64 deg), and the cause is not identified
(a clip starting shortly before the window does not separate usable from unusable fits; usable fits occur in the
session 1 top-level captures and in variants A, C and D).

**Risks.** The WD2 event times are hand-annotated (about 0.05 s). The 0.25 s out-blend is the figure given for this
analysis, not measured here. The WD2 root motion is not in this model. Changing the Muraille_Exit choice (not in scope) would be needed for
any WD2 motion after W + 0.25 s.
"""


def _short(name):
    if not name:
        return '-'
    for p in (sm.SLIDE_PREFIX, sm.ROLL_PREFIX):
        if name.startswith(p):
            return name[len(p):]
    return name


def _f(v, fmt='%.2f', none='-'):
    if v is None:
        return none
    if isinstance(v, str):
        return v
    try:
        if not np.isfinite(v):
            return none
    except TypeError:
        return str(v)
    return fmt % v


def _table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return out


# ---------------------------------------------------------------------------------------------- plots

def plot_roots(res, path):
    names = [n for n in sorted(res['per_clip']) if res['per_clip'][n].get('obstacle')]
    if not names:
        return
    ncol = 2
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.2 * nrow), squeeze=False)
    wd2 = res['wd2']
    for ax, name in zip(axes.ravel(), names):
        e = res['per_clip'][name]
        geo = res['geos'][name]
        o, r = e['map_used']['o'], e['map_used']['r']
        t = geo['_t']
        disp = geo['_disp']
        tp = (t - o) / r
        ax.plot(tp, disp[:, 1] - geo['near_y'], '-', color='tab:blue', lw=1.6)
        ax.axhline(geo['depth'], color='tab:blue', ls=':', lw=0.8)
        ax.set_ylabel('m past near edge', color='tab:blue')
        ax.set_xlabel('played time since clip start (s)')
        ax2 = ax.twinx()
        ax2.plot(tp, disp[:, 2], '-', color='tab:purple', lw=1.2)
        ax2.axhline(geo['height'], color='tab:purple', ls=':', lw=0.8)
        ax2.set_ylabel('root z (m)', color='tab:purple')
        W = e['window'] if e['window'] is not None else 0.0
        ax.axvspan(0, W, color='0.88', zorder=0)
        ax.axvline(0, color='k', lw=0.6)
        takeoff = e['acu_played_s']['takeoff']
        feet = e['acu_played_s']['feet_down']
        if takeoff is not None:
            ax.axvline(takeoff, color='tab:blue', ls='--', lw=1)
            off = takeoff - wd2['takeoff']
            ax.axvspan(wd2['hands_start'] + off, wd2['hands_end'] + off, color='tab:orange', alpha=0.3, zorder=0)
            ax.axvspan(wd2['roll_start'] + off, wd2['roll_end'] + off, color='tab:orange', alpha=0.12, zorder=0)
            ax.axvline(wd2['feet_down'] + off, color='tab:orange', lw=1.4)
            ax.axvline(wd2['run_out'] + off, color='tab:orange', ls=':', lw=1.4)
        if feet is not None:
            ax.axvline(feet, color='tab:green', ls='--', lw=1)
        right = max(W, (wd2['run_out'] + (takeoff or 0.0) - wd2['takeoff']), feet or 0.0) + 0.4
        ax.set_xlim(-0.3, right)
        tag = '' if e['map_used']['source'] == 'own' else '  [assumed map]'
        ax.set_title('%s  (o %.2f, r %.2f, W %.2f s)%s' % (_short(name), o, r, W, tag), fontsize=8.5)
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(names):]:
        ax.axis('off')
    handles = [plt.Line2D([], [], color='tab:blue', label='root: m past near edge'),
               plt.Line2D([], [], color='tab:purple', label='root z'),
               plt.Rectangle((0, 0), 1, 1, color='0.88', label='played window'),
               plt.Line2D([], [], color='tab:blue', ls='--', label='ACU takeoff'),
               plt.Line2D([], [], color='tab:green', ls='--', label='ACU feet down'),
               plt.Line2D([], [], color='tab:orange', label='WD2 feet down (aligned)'),
               plt.Line2D([], [], color='tab:orange', ls=':', label='WD2 run-out')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8, frameon=False)
    fig.suptitle('ACU Sprint SlideOver clips: root motion in played time, WD2 frontdive aligned at the ACU takeoff',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_timeline(res, path):
    names = [n for n in sorted(res['per_clip']) if res['per_clip'][n].get('acu_played_s')]
    if not names:
        return
    wd2 = res['wd2']
    fig, ax = plt.subplots(figsize=(12, 0.6 * len(names) + 1.8))
    for i, name in enumerate(names):
        e = res['per_clip'][name]
        y = len(names) - 1 - i
        W = e['window'] if e['window'] is not None else 0.0
        ax.barh(y, W, left=0, height=0.6, color='0.88', edgecolor='0.5')
        to = e['acu_played_s']['takeoff']
        fd = e['acu_played_s']['feet_down']
        if to is not None and fd is not None:
            ax.barh(y + 0.16, fd - to, left=to, height=0.22, color='tab:blue')
            ax.barh(y - 0.18, wd2['feet_down'] - wd2['takeoff'], left=to, height=0.22, color='tab:orange')
            off = to - wd2['takeoff']
            ax.plot([wd2['roll_end'] + off], [y - 0.18], marker='|', ms=11, color='k')
            ax.plot([wd2['run_out'] + off], [y - 0.18], marker='D', ms=4, color='tab:red')
        tag = '' if e['map_used']['source'] == 'own' else ' (assumed map)'
        ax.text(2.02, y, '%s%s' % (_short(name), tag), va='center', fontsize=7.5)
    ax.set_yticks([])
    ax.set_xlabel('played time since the clip start (s), with the map used for each clip')
    ax.set_xlim(-0.05, 2.0)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_title('Grey: played window (to the next clip). Blue: ACU takeoff to feet down. Orange: WD2 takeoff to feet '
                 'down, black tick: roll end, red diamond: run-out', fontsize=8)
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_fits(res, path):
    rows = [r for r in res['rows'] if r['o'] is not None and r['r'] is not None]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if rows:
        sc = ax.scatter([r['o'] for r in rows], [r['r'] for r in rows],
                        c=[np.log10(max(r['rms'], 0.5)) for r in rows], cmap='viridis',
                        s=[70 if r['usable'] else 24 for r in rows],
                        edgecolors=['k' if r['family'] == 'slide' else 'tab:red' for r in rows], linewidths=0.9)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label('log10 RMS (cm)')
    ax.axhline(1.0, color='0.5', ls=':', lw=0.8)
    ax.set_xlabel('o (clip time at the trigger, s)')
    ax.set_ylabel('r (rate)')
    ax.set_title('Fitted time maps per occurrence. Large = usable. Black edge = SlideOver, red = LandRoll.', fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- tables

def occurrence_table(rows):
    head = ['session', 'capture', 'clip', 'variant', 't_start', 'window s', 'end', 'next clip', 'next t',
            'o s', 'r', 'yaw', 'RMS cm', 'r=1 cm', 'RMS at ref map cm', 'usable']
    out = []
    for r in rows:
        if r['o'] is None:
            fit = ['-'] * 6
        else:
            fit = [_f(r['o'], '%.3f'), _f(r['r'], '%.3f'), _f(r['yaw'], '%.1f'), _f(r['rms'], '%.1f'),
                   _f(r['rms_r1'], '%.1f')]
            fit = fit[:4] + [fit[4]]
        skip = '' if not r['skip'] else ' (%s)' % r['skip']
        out.append([r['session'].replace('_20261008', ''), r['file'][4:-5], _short(r['clip']),
                    r['variant'] or '-', _f(r['t_start'], '%.3f'), _f(r['window_s'], '%.3f'),
                    r['end_reason'], _short(r['next_name']) + skip if r['next_name'] else ('-' + skip),
                    _f(r['next_t'], '%.3f')] + fit[:5] + [_f(r.get('rms_ref'), '%.1f'),
                                                          'yes' if r['usable'] else 'no'])
    return _table(head, out)


def clip_table(per):
    head = ['clip', 'fits: n / fitted / usable', 'own o, r (usable)', 'map used (source)', 'window W s (measured n, range)',
            'censored windows', 'h x depth m', 'ACU takeoff clip s', 'ACU feet down clip s',
            'takeoff played', 'hand top played', 'feet down played', 'takeoff..feet ACU s', 'WD2 scale',
            'roll end', 'feet down', 'run-out', 'dive in window']
    out = []
    for name in sorted(per):
        e = per[name]
        if e['family'] != 'slide':
            continue
        ob = e.get('obstacle') or {}
        wd = e.get('wd2_dive') or {}
        ev = wd.get('events_played') or {}
        acu_c = e.get('acu_clip_s') or {}
        acu_p = e.get('acu_played_s') or {}
        mu = e['map_used']
        src = 'own' if mu['source'] == 'own' else 'ASSUMED'
        if e['window'] is None:
            wtxt = '-'
        else:
            wtxt = '%.3f (n %d, %.2f-%.2f)' % (e['window'], e['n_window_measured'], e['window_min'], e['window_max'])
        if not wd:
            fits = 'no window'
        else:
            fits = 'yes' if wd['fits_window'] else ('blend only' if wd['fits_with_blend'] else 'no')
            if src != 'own':
                fits += ' (assumed map)'
        out.append([
            _short(name) + ('' if e['n_usable'] else ' *'),
            '%d / %d / %d' % (e['n'], e['n_fit'], e['n_usable']),
            '%s, %s' % (_f(e['o'], '%.3f'), _f(e['r'], '%.3f')),
            '%.3f, %.3f (%s)' % (mu['o'], mu['r'], src),
            wtxt,
            len(e['window_censored']),
            '%s x %s' % (_f(ob.get('height'), '%.2f'), _f(ob.get('depth'), '%.2f')),
            _f(acu_c.get('takeoff'), '%.3f'), _f(acu_c.get('feet_down'), '%.3f'),
            _f(acu_p.get('takeoff'), '%.3f'), _f(acu_p.get('hand_plant'), '%.3f'),
            _f(acu_p.get('feet_down'), '%.3f'),
            _f(e.get('acu_takeoff_to_feet_played_s'), '%.3f'), _f(e.get('wd2_scale_to_acu_span'), '%.2f'),
            _f(ev.get('roll_end'), '%.3f'), _f(ev.get('feet_down'), '%.3f'), _f(ev.get('run_out'), '%.3f'),
            fits,
        ])
    return _table(head, out)


def spill_table(per):
    head = ['clip', 'map', 'window W', 'WD2 takeoff (played)', 'roll end', 'feet down', 'run-out',
            'spill roll end', 'spill feet down', 'spill run-out', 'ACU clip time at W', 'ACU feet down (clip)']
    out = []
    for name in sorted(per):
        e = per[name]
        wd = e.get('wd2_dive')
        if e['family'] != 'slide' or not wd:
            continue
        ev = wd['events_played']
        out.append([_short(name), 'own' if e['map_used']['source'] == 'own' else 'ASSUMED',
                    _f(e['window'], '%.3f'), _f(wd['takeoff_played'], '%.3f'),
                    _f(ev['roll_end'], '%.3f'), _f(ev['feet_down'], '%.3f'), _f(ev['run_out'], '%.3f'),
                    _f(wd['spill_roll_end'], '%.3f'), _f(wd['spill_feet_down'], '%.3f'),
                    _f(wd['spill_run_out'], '%.3f'), _f(e.get('window_clip_end_s'), '%.3f'),
                    _f((e.get('acu_clip_s') or {}).get('feet_down'), '%.3f')])
    return _table(head, out)


def retimed_table(per):
    head = ['clip', 'map', 'W', 'k ACU (feet match)', 'k roll (roll inside W)', 'k used', 'takeoff', 'roll end',
            'feet down', 'run-out', 'spill roll end', 'spill feet down', 'spill run-out', 'ACU feet down (played)',
            'feet down fits']
    out = []
    for name in sorted(per):
        e = per[name]
        wd = e.get('wd2_dive_k')
        if e['family'] != 'slide' or not wd:
            continue
        ev = wd['events_played']
        fits = 'yes' if wd['fits_window'] else ('blend only' if wd['fits_with_blend'] else 'no')
        out.append([_short(name), 'own' if e['map_used']['source'] == 'own' else 'ASSUMED',
                    _f(e['window'], '%.3f'), _f(wd['k_acu'], '%.3f'), _f(wd['k_roll'], '%.3f'),
                    _f(wd['k_used'], '%.3f'), _f(ev['takeoff'], '%.3f'), _f(ev['roll_end'], '%.3f'),
                    _f(ev['feet_down'], '%.3f'), _f(ev['run_out'], '%.3f'),
                    _f(wd['spill_roll_end'], '%.3f'), _f(wd['spill_feet_down'], '%.3f'),
                    _f(wd['spill_run_out'], '%.3f'), _f((e.get('acu_played_s') or {}).get('feet_down'), '%.3f'),
                    fits])
    return _table(head, out)


def next_table(per, dump_names):
    head = ['clip', 'next clip (at window end)', 'count', 'dump in acu_dumps']
    out = []
    for name in sorted(per):
        e = per[name]
        for nm, c in e['next_clip_counts'].items():
            out.append([_short(name), nm, c, 'yes' if nm in dump_names else 'no'])
    return _table(head, out)


def _range(vals, one=False):
    vals = [v for v in vals if v is not None]
    if not vals:
        return '-'
    if one:
        return '%.1f cm' % min(vals)
    return '%.1f-%.1f cm' % (min(vals), max(vals))


def _dump_names():
    return {nm for _h, nm, _p in dumps.list_dumps()}


# ---------------------------------------------------------------------------------------------- report

def write_report(res, md_path, plot_dir, feasibility=None):
    os.makedirs(plot_dir, exist_ok=True)
    roots_png = os.path.join(plot_dir, 'slideover_roots.png')
    timeline_png = os.path.join(plot_dir, 'slideover_timeline.png')
    fits_png = os.path.join(plot_dir, 'slideover_fits.png')
    plot_roots(res, roots_png)
    plot_timeline(res, timeline_png)
    plot_fits(res, fits_png)

    rows = res['rows']
    per = res['per_clip']
    wd2 = res['wd2']
    ref = res.get('ref')
    sess = {}
    for r in rows:
        sess[r['session']] = sess.get(r['session'], 0) + 1
    fitted = [r for r in rows if r['o'] is not None]
    usable = [r for r in rows if r['usable']]
    skipped = {}
    for r in rows:
        if r['skip']:
            skipped[r['skip']] = skipped.get(r['skip'], 0) + 1
    nslide = sum(1 for r in rows if r['family'] == 'slide')
    lines = [
        '# SlideOver play model: ACU Sprint SlideOver (and LandRoll) as the game plays them',
        '',
        'Generated by `pipeline/slideover_model.py` (time maps from `matchclip.fit_time_map`, partial poses allowed). '
        'Times are played seconds since the clip start unless marked "clip s". Clip time = o + r * played s. '
        'WD2 frontdive from `data/wd2_events.json`: takeoff 0.45 s, hands on the ground 0.75-0.92 s, roll 0.92-1.18 s, '
        'feet down 1.20 s, run-out 1.60 s (WD2 clip s), placed on the played line with the WD2 takeoff at the ACU takeoff '
        'and rate 1.',
        '',
        '## Data and method',
        '',
        '- Captures: %s. Unpatched occurrences of Sprint SlideOver and LandRoll clips: %d (%d SlideOver, %d LandRoll).'
        % (', '.join('%s %d' % (k.replace('_20261008', ''), v) for k, v in sorted(sess.items())), len(rows), nslide,
           len(rows) - nslide),
        '- Fitted (pose with Hips and a fit bone): %d. Usable: %d (RMS <= %.0f cm, inside the search grid, not on its '
        'border). Usable RMS %s; the smallest unusable RMS is %s (fits of 40-56 cm are typical of unusable ones, '
        'several with r < 0 or r > 1.5).'
        % (len(fitted), len(usable), sm.USABLE_RMS_CM, _range([r['rms'] for r in usable]),
           _range([r['rms'] for r in fitted if not r['usable']], one=True)),
        '- Yaw (game frame against clip frame): usable fits %s, unusable fits %s. A usable fit needs almost no yaw; an '
        'unusable one is matched only with a large turn, so its pose is not the clip in another frame.'
        % (_range([r['yaw'] for r in usable]), _range([r['yaw'] for r in fitted if not r['usable']])),
        '- Not fitted: %s.' % ('; '.join('%d x %s' % (v, k) for k, v in sorted(skipped.items())) or 'none'),
        '- Partial poses: session 4 captures never record RightHand, and their fits use the other five fit bones. '
        'Session 2 and most of session 3 have no bones at all (no pose), so they add no fits.',
        '- Reference map: median of the usable fits of `%s` (o %s, r %s). It is used only where a clip has no usable fit '
        'of its own, and every number that uses it is marked ASSUMED.'
        % (sm.REF_CLIP, _f(ref[0] if ref else None, '%.3f'), _f(ref[1] if ref else None, '%.3f')),
        '- Windows: measured only where the window ends at a next clip; captures that end first are listed as censored.',
        '- Rows marked * in the clip table have no usable fit.',
        '',
        '## Per clip: SlideOver',
        '',
    ]
    lines += clip_table(per)
    lines += [
        '',
        '"ACU ... clip s" are the dump times (contacts.py: takeoff = end of the last foot ground contact before the near '
        'edge, feet down = first foot ground contact past the far edge, hand top = hand on the obstacle top). '
        '"WD2 scale" = ACU takeoff-to-feet span / WD2 span (0.75 s): below 1 the WD2 dive has to be sped up by that factor.',
        '',
        '## WD2 dive against the played window',
        '',
    ]
    lines += spill_table(per)
    lines += [
        '',
        'Spill = time past the window end W in played seconds (0 = inside). After W the game blends out of this clip '
        'for about %.2f s (the blend samples it), so a feet-down spill up to %.2f s is still visible as the landing.'
        % (sm.BLEND_OUT_S, sm.BLEND_OUT_S),
        '',
        '### WD2 dive re-timed: k used = min(k ACU, k roll)',
        '',
        'k ACU scales the WD2 durations so that the WD2 feet down lands on the ACU feet down; k roll is the largest factor '
        'that still ends the roll at the window end. The dive is then re-timed with k used, takeoff fixed.',
        '',
    ]
    lines += retimed_table(per)
    lines += [
        '',
        '## Clips that follow SlideOver and LandRoll',
        '',
    ]
    lines += next_table(per, _dump_names())
    lines += [
        '',
        '## LandRoll clips (o, r, window, next)',
        '',
    ]
    roll = {k: v for k, v in per.items() if v['family'] == 'roll'}
    rhead = ['clip', 'n / fitted / usable', 'own o, r', 'window W (measured n, range)', 'censored', 'next clips']
    rrows = []
    for k in sorted(roll):
        e = roll[k]
        wtxt = '-' if e['window'] is None else '%.3f (n %d, %.2f-%.2f)' % (
            e['window'], e['n_window_measured'], e['window_min'], e['window_max'])
        rrows.append([_short(k), '%d / %d / %d' % (e['n'], e['n_fit'], e['n_usable']),
                      '%s, %s' % (_f(e['o'], '%.3f'), _f(e['r'], '%.3f')), wtxt, len(e['window_censored']),
                      '; '.join('%s (%d)' % (_short(n), c) for n, c in e['next_clip_counts'].items())])
    lines += _table(rhead, rrows)
    lines += [
        '',
        '## Per occurrence',
        '',
    ]
    lines += occurrence_table(rows)
    lines += [
        '',
        '## Plots',
        '',
        '![root motion against played time](%s)' % os.path.basename(roots_png),
        '',
        '![window against ACU and WD2 timing](%s)' % os.path.basename(timeline_png),
        '',
        '![fitted time maps](%s)' % os.path.basename(fits_png),
        '',
        '## Feasibility',
        '',
        feasibility if feasibility is not None else FEASIBILITY,
        '',
    ]
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return md_path
