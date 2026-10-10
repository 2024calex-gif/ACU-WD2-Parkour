"""Report on the original ACU vault clips: contact events table, JSON and contact sheets.

Usage:  python acuclip_report.py
Writes  v2\\out\\acu_events.json  and  v2\\out\\sheets\\acu_<clipname>.png  (one per vault clip).
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

import acuclip                    # noqa: E402
import config                     # noqa: E402
import contacts                   # noqa: E402
import dumps                      # noqa: E402

SHEETS = os.path.join(config.OUT, 'sheets')
EVENTS_JSON = os.path.join(config.OUT, 'acu_events.json')

BODY = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftClavicle', 'LeftArm', 'LeftForeArm',
        'LeftHand', 'RightClavicle', 'RightArm', 'RightForeArm', 'RightHand', 'LeftUpLeg', 'LeftLeg',
        'LeftFoot', 'LeftFootThumb1', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightFootThumb1']
POINT_COLORS = {'LeftHand': 'tab:red', 'RightHand': 'tab:orange', 'LeftFoot': 'tab:blue',
                'RightFoot': 'tab:cyan'}
EVENT_COLORS = {'hand_plant': 'tab:red', 'takeoff': 'tab:green', 'foot_on_top': 'tab:purple',
                'landing': 'tab:blue'}


def _fmt(ev):
    if not ev:
        return '-'
    who = ev.get('hand') or ev.get('foot') or ''
    who = who.replace('Left', 'L').replace('Right', 'R')
    return '%d (%.2fs %s)' % (ev['frame'], ev['seconds'], who)


def analyse(handle, name, dump):
    seq = acuclip.pose_sequence(dump)
    ev = contacts.events(seq, dump)
    ev['name'] = name
    ev['handle'] = handle
    ev['length'] = float(dump['Length'])
    return seq, ev


def draw_sheet(seq, dump, ev, path):
    world = acuclip.world_positions(seq, dump)
    nm = dumps.bone_names()
    idx = {nm[b['hash']]: i for i, b in enumerate(seq.bones) if b['hash'] in nm}
    ob = ev['obstacle']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10), gridspec_kw=dict(height_ratios=[1.3, 1]))

    # top panel: side view (world y horizontal, z vertical)
    ax1.add_patch(plt.Rectangle((ob['near_y'], 0.0), ob['far_y'] - ob['near_y'], ob['height'],
                                color='0.7', zorder=1))
    F = len(seq.frames)
    pick = np.linspace(0, F - 1, 10).round().astype(int)
    cmap = plt.get_cmap('viridis')
    edges = []
    for c in BODY:
        if c not in idx:
            continue
        par = seq.bones[idx[c]]['parent']
        pname = nm.get(seq.bones[par]['hash']) if par >= 0 else None
        if pname in BODY:
            edges.append((c, pname))
    edges.append(('Spine', 'Hips'))
    xs_all = []
    for k, f in enumerate(pick):
        col = cmap(k / max(1, len(pick) - 1))
        for c, p in edges:
            a = world[f, idx[c]]
            b = world[f, idx[p]]
            ax1.plot([a[1], b[1]], [a[2], b[2]], '-', color=col, lw=2, zorder=3)
        for n in ('LeftHand', 'RightHand', 'LeftFoot', 'RightFoot'):
            pnt = world[f, idx[n]]
            ax1.plot(pnt[1], pnt[2], 'o', color=col, ms=4, zorder=4)
        ax1.text(world[f, idx['Hips'], 1], world[f, idx['Hips'], 2] + 1.25,
                 '%.2fs' % (seq.frames[f] / acuclip.FPS), color=col, fontsize=7, ha='center')
        xs_all.append(world[f, :, 1])
    ax1.axhline(0.0, color='k', lw=1)
    lo = min(float(np.min(x)) for x in xs_all) - 1.0
    hi = max(float(np.max(x)) for x in xs_all) + 1.0
    ax1.set_xlim(min(lo, ob['near_y'] - 1.0), max(hi, ob['far_y'] + 2.0))
    ax1.set_ylim(-0.2, max(ob['height'], 1.0) + 2.2)
    ax1.set_aspect('equal')
    ax1.set_xlabel('world y (m), direction of travel')
    ax1.set_ylabel('z (m)')
    ax1.set_title('%s  h=%.2f d=%.2f  (stick figures at 10 frames, time-coloured)' % (
        ev['name'], ob['height'], ob['depth']), fontsize=10)

    # bottom panel: z of hands and feet over time, contacts shaded, events as vertical lines
    t = seq.frames / float(acuclip.FPS)
    for n, col in POINT_COLORS.items():
        ax2.plot(t, world[:, idx[n], 2], color=col, lw=1.4, label=n)
        for s, e, kind in ev['contacts'][n]:
            ax2.axvspan(s / acuclip.FPS, e / acuclip.FPS, color=col,
                        alpha=0.30 if kind == 'top' else 0.10)
    ax2.axhline(ob['height'], color='k', ls='--', lw=1, label='obstacle top')
    ax2.axhline(0.0, color='0.4', lw=1, label='ground')
    for key, col in EVENT_COLORS.items():
        e = ev.get(key)
        if e:
            ax2.axvline(e['seconds'], color=col, ls=':', lw=1.6)
            ax2.text(e['seconds'], 2.5, key, color=col,
                     rotation=90, fontsize=8, va='top', ha='right')
    ax2.set_xlabel('time (s); shaded = contact (dark = on obstacle top, light = ground)')
    ax2.set_ylabel('z (m)')
    ax2.set_ylim(-0.1, max(2.6, ob['height'] + 0.8))
    ax2.legend(fontsize=7, ncol=3, loc='upper right')
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    os.makedirs(SHEETS, exist_ok=True)
    rows = []
    results = {}
    vaults = sorted([(h, n) for h, n, _p in dumps.list_dumps() if acuclip.is_vault(n)], key=lambda t: t[1])
    for handle, name in vaults:
        dump = dumps.load_by_handle(handle)
        seq, ev = analyse(handle, name, dump)
        rows.append(ev)
        ob = ev['obstacle']
        print('%-50s L=%.2fs h=%.2f d=%.2f  takeoff=%-18s hand_plant=%-22s foot_on_top=%-20s landing=%s' % (
            name, ev['length'], ob['height'], ob['depth'], _fmt(ev['takeoff']), _fmt(ev['hand_plant']),
            _fmt(ev['foot_on_top']), _fmt(ev['landing'])))
        results[name] = dict(handle=handle, length=ev['length'], obstacle=ob, takeoff=ev['takeoff'],
                             hand_plant=ev['hand_plant'], foot_on_top=ev['foot_on_top'],
                             landing=ev['landing'], near_frame=ev['near_frame'], far_frame=ev['far_frame'],
                             contacts=ev['contacts'])
        draw_sheet(seq, dump, ev, os.path.join(SHEETS, 'acu_%s.png' % name))
    with open(EVENTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=1)
    print('\n%d vault clips. Wrote %s and %d sheets in %s' % (len(rows), EVENTS_JSON, len(rows), SHEETS))


if __name__ == '__main__':
    main()
