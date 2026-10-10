"""Muraille_Exit chain model: what the game plays after a SlideOver, and what else uses the Exit clips.

usage: python exit_model.py [--captures-root DIR] [--out-json FILE] [--out-md FILE] [--plot-dir DIR] [-q]

Chains: consecutive Exit starts (within CHAIN_TOL) form one chain. The game starts Muraille_Exit_Fd100cm-Dn500cm (or
Fd250cm-Dn300cm) and 1 ms later Muraille_Exit_Dn100 (or Fd250cm-Dn100cm); the last member is the clip that plays. A
chain's played window runs from its last start to the next clip start (any name), as in slideover_model.

Outputs (out/match): play_model_exit.json, exit_report.md, exit_entity.png (entity z and forward distance through each
SlideOver -> Exit pair) and exit_dumps.png (root z and y and the ground contacts of the Exit dumps).
"""
import argparse
import collections
import json
import os
import sys
import zlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import acuclip  # noqa: E402
import capture  # noqa: E402
import contacts  # noqa: E402
import dumps  # noqa: E402
import ik  # noqa: E402
import matchclip as mc  # noqa: E402
import slideover_model as sm  # noqa: E402

FPS = acuclip.FPS
EXIT_PREFIX = 'xx_Parkour_Muraille_Exit'
CHAIN_TOL = 0.01          # s: starts closer than this are one chain
TRANSIENT_S = 0.05        # s: a played window shorter than this means the next start replaces the clip
GROUND_BAND = (-0.03, 0.12)   # m above the ground level of a contact (contacts.py tolerances)
SPEED_MAX = 1.0           # m/s
MIN_RUN = 2               # frames
MAIN_EXIT = 'Fd100cm-Dn100cm'
PAIR_CLIPS = ('xx_Parkour_Sprint_SlideOver_Fd100cm-Up100cm', 'xx_Parkour_Sprint_SlideOver_Fd200cm-Up200cm',
              'xx_Parkour_Sprint_SlideOver_Fd200cm-Up100cm', 'xx_Parkour_Sprint_SlideOver_Fd300cm-Up100cm')
HANDOFF_NEXT = ('xx_Nav_Run_3-Loop', 'xx_Nav_Sprint_3-Loop_V02', 'xx_Parkour_LandingFwd_LandSoft_tr_Run02_A',
                'xx_Parkour_LandingFwd_LandMid_tr_RunImpultion')
OUT_JSON = os.path.join(sm.OUT_DIR, 'play_model_exit.json')
OUT_MD = os.path.join(sm.OUT_DIR, 'exit_report.md')
PLOT_DIR = sm.OUT_DIR


def short(name):
    """Exit name without its prefix, e.g. 'Fd100cm-Dn100cm'."""
    return name[len(EXIT_PREFIX) + 1:] if name and name.startswith(EXIT_PREFIX) else (name or '-')


def is_exit(name):
    return bool(name) and name.startswith(EXIT_PREFIX)


def pred_family(name):
    """Family of a clip that precedes an Exit chain (None = the chain starts the capture)."""
    if name is None:
        return 'capture start'
    if name.startswith(sm.SLIDE_PREFIX):
        return 'SlideOver'
    if name.startswith(sm.ROLL_PREFIX):
        return 'LandRoll'
    if 'SkipOver' in name:
        return 'SkipOver'
    if 'Leaping' in name:
        return 'Leaping'
    if 'Muraille_Exit' in name:
        return 'Muraille_Exit'
    if 'NavSneak' in name or 'DropDown' in name:
        return 'drop / sneak'
    if name.startswith('xx_Parkour_'):
        return 'other parkour'
    if name.startswith('xx_Nav') or name.startswith('acu_nav'):
        return 'nav'
    return 'other'


# ---------------------------------------------------------------------------------------------- chains and timing

def exit_chains(cap, tol=CHAIN_TOL):
    """Chains of consecutive Exit starts in a capture.

    Each chain: first, last (clip indices), t_first, t_last, names (members), eff (name of the last member), prev
    (clip before the first member, or None), nxt (clip after the last member, or None), window (s from the last
    start to nxt; None when the capture ends first) and window_end.
    """
    clips = cap.clips
    out = []
    i = 0
    while i < len(clips):
        if is_exit(clips[i]['name']):
            j = i
            while j + 1 < len(clips) and is_exit(clips[j + 1]['name']) and clips[j + 1]['t'] - clips[j]['t'] < tol:
                j += 1
            nxt = clips[j + 1] if j + 1 < len(clips) else None
            out.append(dict(
                first=i, last=j, t_first=float(clips[i]['t']), t_last=float(clips[j]['t']),
                names=[clips[k]['name'] for k in range(i, j + 1)], eff=clips[j]['name'],
                prev=clips[i - 1] if i > 0 else None, nxt=nxt,
                window=(float(nxt['t']) - float(clips[j]['t'])) if nxt is not None else None,
                window_end='next clip' if nxt is not None else 'capture end'))
            i = j + 1
        else:
            i += 1
    return out


def chain_window(cap, ch):
    """Window for matchclip.fit_time_map: t_start = last member start, t_end = next clip start (any name)."""
    t0 = ch['t_last']
    t1 = float(ch['nxt']['t']) if ch['nxt'] is not None else float(cap.t[-1])
    return dict(clip_index=ch['last'], clip_name=ch['eff'], t_start=t0, t_end=t1, length_s=t1 - t0,
                next_name=ch['nxt']['name'] if ch['nxt'] is not None else None,
                next_t=float(ch['nxt']['t']) if ch['nxt'] is not None else None, end_reason=ch['window_end'])


def place_events(events, takeoff_played, k, W, slide_map, exit_map, wd2):
    """Where each WD2 event lands when WD2 durations are scaled by k from the ACU takeoff (rate k, takeoff fixed).

    events: name -> WD2 clip time. slide_map / exit_map: (o, r). W: the played SlideOver window end (= Exit start).
    Returns name -> dict(played, clip, clip_s): clip 'SlideOver' with clip time o_s + r_s * played while played < W,
    else 'Exit' with clip time o_e + r_e * (played - W) on the Exit's own clock.
    """
    out = {}
    for name, tau in events.items():
        t = takeoff_played + (tau - wd2['takeoff']) * k
        if t < W:
            out[name] = dict(played=t, clip='SlideOver', clip_s=slide_map[0] + slide_map[1] * t)
        else:
            out[name] = dict(played=t, clip='Exit', clip_s=exit_map[0] + exit_map[1] * (t - W))
    return out


def budget(takeoff_played, W, exit_window, wd2, k=1.0):
    """Time budget from the SlideOver takeoff to the Exit played end, against the WD2 run-out span (k scales WD2)."""
    end = W + exit_window
    span = (wd2['run_out'] - wd2['takeoff']) * k
    run_out = takeoff_played + span
    return dict(exit_end_played=end, total_from_takeoff=end - takeoff_played, wd2_takeoff_to_run_out=span,
                run_out_played=run_out, run_out_margin=end - run_out)


def first_run(mask, frames):
    """First frame value of a run of at least MIN_RUN true samples (None if there is none)."""
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        if run >= MIN_RUN:
            return float(frames[i - MIN_RUN + 1])
    return None


def ground_touch(z, speed, fps=FPS, support=None):
    """First time a point (height z[F], speed[F] in m/s) is in ground contact. Support: the level it rests at over
    the last 0.2 s unless given. Returns (time_s or None, support)."""
    z = np.asarray(z, float)
    if support is None:
        tail = max(1, int(round(0.2 * fps)))
        support = float(np.median(z[-tail:]))
    h = z - support
    mask = (h >= GROUND_BAND[0]) & (h <= GROUND_BAND[1]) & (np.asarray(speed) < SPEED_MAX)
    f = first_run(mask, np.arange(len(z)))
    return (None if f is None else f / fps), support


def _crc(name):
    return zlib.crc32(name.encode())


def exit_geometry(dump):
    """Root and ground contacts of one Exit clip (clip seconds). The arrays t, disp, seq and P are kept for the
    hand-off and the plots (not written to the JSON)."""
    seq = acuclip.pose_sequence(dump)
    fps = FPS / max(1, seq.step)
    t = seq.frames / float(FPS)
    disp = acuclip.markers(dump, seq.frames)['disp']
    P = acuclip.world_positions(seq, dump)
    H = seq.hash_index
    touch = {}
    for label, names in (('feet', ('LeftFoot', 'RightFoot')), ('hands', ('LeftHand', 'RightHand'))):
        cand = []
        for n in names:
            pts = P[:, H[_crc(n)]]
            tt, sup = ground_touch(pts[:, 2], contacts.smooth_speed(pts, fps), fps)
            cand.append((tt, sup))
        ts = [c[0] for c in cand if c[0] is not None]
        touch[label] = dict(time=min(ts) if ts else None, support=float(np.mean([c[1] for c in cand])))
    zend = float(disp[-1, 2])
    settle = np.nonzero(np.abs(disp[:, 2] - zend) <= 0.02)[0]
    try:
        ob = acuclip.obstacle(dump)
        markers = dict(height=round(ob['height'], 3), depth=round(ob['depth'], 3),
                       landing_y=round(ob['landing_y'], 3) if 'landing_y' in ob else None,
                       landing_z=round(ob['landing_z'], 3) if 'landing_z' in ob else None)
    except ValueError:
        markers = None
    grid = np.arange(0.0, float(dump['Length']) + 1e-9, 0.1)
    return dict(
        length_s=float(dump['Length']),
        disp_t=[round(float(x), 2) for x in grid],
        disp_y=[round(float(x), 3) for x in np.interp(grid, t, disp[:, 1])],
        disp_z=[round(float(x), 3) for x in np.interp(grid, t, disp[:, 2])],
        disp_y_end=float(disp[-1, 1]), disp_z_end=zend,
        root_settle_s=round(float(t[settle[0]]), 3) if settle.size else None,
        feet_touch_s=None if touch['feet']['time'] is None else round(touch['feet']['time'], 3),
        hand_touch_s=None if touch['hands']['time'] is None else round(touch['hands']['time'], 3),
        feet_support_z=round(touch['feet']['support'], 3), hands_support_z=round(touch['hands']['support'], 3),
        markers=markers, _t=t, _disp=disp, _P=P, _seq=seq)


def posture(seq, P, disp_t, disp, at_s):
    """Posture at one clip time: bone positions relative to Hips, Hips height above the lower foot (m), root
    velocity from the displacement track (m/s; y forward, z vertical)."""
    names = mc.FIT_BONES + (mc.ROOT_BONE,)
    idx = {b: ik.bone_index(seq.bones, b) for b in names}
    f = int(np.clip(round(at_s * FPS), 1, P.shape[0] - 2))
    rel = np.stack([P[f, idx[b]] - P[f, idx[mc.ROOT_BONE]] for b in mc.FIT_BONES])
    feet_z = min(P[f, idx['LeftFoot'], 2], P[f, idx['RightFoot'], 2])
    dt = 1.0 / FPS
    vy = (np.interp(at_s + dt, disp_t, disp[:, 1]) - np.interp(at_s - dt, disp_t, disp[:, 1])) / (2 * dt)
    vz = (np.interp(at_s + dt, disp_t, disp[:, 2]) - np.interp(at_s - dt, disp_t, disp[:, 2])) / (2 * dt)
    return dict(rel=rel, hips_above_feet=float(P[f, idx[mc.ROOT_BONE], 2] - feet_z), vy=float(vy), vz=float(vz))


def handoff_gap(exit_geo, next_dump, exit_at_s):
    """Pose and velocity at the Exit clip's played end (exit_at_s, clip time) against the next clip's first frame."""
    nseq = acuclip.pose_sequence(next_dump)
    ndisp = acuclip.markers(next_dump, nseq.frames)['disp']
    nt = nseq.frames / float(FPS)
    e = posture(exit_geo['_seq'], exit_geo['_P'], exit_geo['_t'], exit_geo['_disp'],
                min(max(exit_at_s, 0.0), exit_geo['length_s'] - 0.02))
    nworld = acuclip.world_positions(nseq, next_dump)
    n = posture(nseq, nworld, nt, ndisp, 0.0)
    rms = float(np.sqrt(np.mean(np.sum((e['rel'] - n['rel']) ** 2, axis=1))) * 100.0)
    return dict(bone_rms_cm=round(rms, 1), hips_above_feet_exit=round(e['hips_above_feet'], 3),
                hips_above_feet_next=round(n['hips_above_feet'], 3),
                vy_exit=round(e['vy'], 2), vy_next=round(n['vy'], 2),
                vz_exit=round(e['vz'], 2), vz_next=round(n['vz'], 2))


# ---------------------------------------------------------------------------------------------- captures

def chain_rows(paths):
    """One row per Exit chain in all captures."""
    rows = []
    for p in paths:
        cap = capture.load(p)
        for ch in exit_chains(cap):
            prev = ch['prev']
            pname = prev['name'] if prev is not None else None
            rows.append(dict(
                session=os.path.basename(os.path.dirname(p)).replace('_20261008', ''),
                file=os.path.basename(p)[4:-5], variant=cap.variant or '-', eff=ch['eff'],
                chain=[short(n) for n in ch['names']], prev_name=pname, prev_family=pred_family(pname),
                prev_gap=round(ch['t_first'] - float(prev['t']), 3) if prev is not None else None,
                next_name=ch['nxt']['name'] if ch['nxt'] is not None else None,
                window=ch['window'], window_end=ch['window_end'],
                hips=cap.bones.get(mc.ROOT_BONE) is not None, cap=cap, ch=ch))
    return rows


def entity_pair(cap, ch, prev):
    """SlideOver -> Exit pair: entity z and forward distance relative to the Exit start (W) and the SlideOver start."""
    t0 = float(prev['t'])
    W = ch['t_first']
    h = float(prev['name'].split('Up')[1].rstrip('cm')) / 100.0
    z = cap.entity_pos[:, 2]
    y = cap.entity_pos[:, 1]
    z0 = float(np.interp(t0, cap.t, z))
    y0 = float(np.interp(t0, cap.t, y))
    Wend = float(ch['nxt']['t']) if ch['nxt'] is not None else float(cap.t[-1])
    m = (cap.t >= W) & (cap.t <= Wend)
    zW = float(np.interp(W, cap.t, z))
    yW = float(np.interp(W, cap.t, y))
    return dict(
        slide=prev['name'][len(sm.SLIDE_PREFIX):], h=h, W_from_slide=W - t0, exit_window=Wend - W,
        exit=short(ch['eff']), z_W_rise=zW - z0, z_end_rise=float(np.interp(Wend, cap.t, z)) - z0,
        z_max_rise=float(np.max(z[(cap.t >= t0) & (cap.t <= Wend)])) - z0,
        y_W=yW - y0, y_end=float(np.interp(Wend, cap.t, y)) - y0,
        z_rel_W=float(z[m].mean() - zW) if m.any() else None,
        t=cap.t - W, z=z - zW, y=y - yW, tw=Wend - W)


# ---------------------------------------------------------------------------------------------- run

def run(captures_root=sm.CAPTURES_ROOT, verbose=True):
    paths = sm.capture_paths(captures_root)
    rows = chain_rows(paths)
    slide = json.load(open(os.path.join(sm.OUT_DIR, 'play_model_slideover.json'), encoding='utf-8'))
    dump_names = {nm: h for h, nm, _ in dumps.list_dumps()}

    # Q3: what precedes each Exit chain
    pred_full = collections.Counter((short(r['eff']), r['prev_name'] or 'capture start') for r in rows)
    pred_fam = collections.Counter((short(r['eff']), r['prev_family']) for r in rows)

    # Q1: SlideOver -> Exit pairs
    pairs = []
    for r in rows:
        prev = r['ch']['prev']
        if prev is not None and (prev['name'] or '').startswith(sm.SLIDE_PREFIX):
            pairs.append(dict(file=r['file'], session=r['session'], variant=r['variant'],
                              **entity_pair(r['cap'], r['ch'], prev)))

    # Q2: fits of the played Exit clips (windows >= TRANSIENT_S, with a pose)
    geos = {}
    for name in sorted({n for r in rows for n in r['ch']['names']}):
        d = dumps.load_by_name(name)
        geos[name] = exit_geometry(d) if d is not None else None
    # Exit starts that are replaced 1 ms later (never the played clip) and what replaces them
    transient_first = collections.Counter()
    for r in rows:
        names = r['ch']['names']
        for a, b in zip(names[:-1], names[1:]):
            transient_first[(short(a), short(b))] += 1
    fits = []
    for r in rows:
        ch = r['ch']
        if ch['nxt'] is None:
            continue
        row = dict(file=r['file'], session=r['session'], variant=r['variant'], exit=short(r['eff']),
                   prev=r['prev_name'], window=round(r['window'], 3), next=r['next_name'], pose=bool(r['hips']),
                   o=None, r=None, rms=None, n_frames=0, usable=False, skip=None)
        dump = mc.dump_for_clip(r['cap'].clips[ch['last']])
        if r['window'] < TRANSIENT_S:
            row['skip'] = 'transient (next start %.0f ms later)' % (r['window'] * 1000)
        elif dump is None:
            row['skip'] = 'no dump'
        elif not r['hips']:
            row['skip'] = 'no Hips (no pose)'
        else:
            try:
                fit = mc.fit_time_map(r['cap'], dump, chain_window(r['cap'], ch), partial=True)
                in_grid = (mc.O_GRID[0] <= fit['o'] <= mc.O_GRID[1]) and (mc.R_GRID[0] <= fit['r'] <= mc.R_GRID[1])
                row.update(o=round(fit['o'], 3), r=round(fit['r'], 3), rms=round(fit['rms_cm'], 1),
                           n_frames=fit['n_frames'], yaw=round(fit['yaw_deg'], 1))
                row['usable'] = bool(fit['rms_cm'] <= sm.USABLE_RMS_CM and in_grid and fit['n_frames'] >= 4)
            except ValueError as e:
                row['skip'] = str(e)
        fits.append(row)
        if verbose:
            print('%-8s %-22s win %.3f o=%s r=%s rms=%s %s' % (
                r['session'], short(r['eff']), r['window'], row['o'], row['r'], row['rms'], row['skip'] or ''),
                file=sys.stderr)

    # per played clip
    per = {}
    for name in sorted({r['eff'] for r in rows}):
        s = short(name)
        mine = [r for r in rows if r['eff'] == name]
        real = [r['window'] for r in mine if r['window'] is not None and r['window'] >= TRANSIENT_S]
        fr = [f for f in fits if f['exit'] == s]
        us = [f for f in fr if f['usable']]
        g = geos.get(name)
        e = dict(
            dump=dump_names.get(name), occurrences=len(mine),
            as_chain_start=sum(1 for r in mine if r['chain'][0] == s),
            transient=sum(1 for r in mine if r['window'] is None or r['window'] < TRANSIENT_S),
            after_slideover=sum(1 for r in mine if r['prev_family'] == 'SlideOver'),
            window_median=float(np.median(real)) if real else None,
            window_range=[float(min(real)), float(max(real))] if real else None,
            fitted=sum(1 for f in fr if f['o'] is not None), usable=len(us),
            o=float(np.median([f['o'] for f in us])) if us else None,
            r=float(np.median([f['r'] for f in us])) if us else None,
            predecessor_families=dict(collections.Counter(r['prev_family'] for r in mine).most_common()),
            next_counts=dict(collections.Counter(r['next_name'] or '(capture end)' for r in mine).most_common()),
        )
        if g is not None:
            e['geometry'] = {k: g[k] for k in ('length_s', 'disp_y_end', 'disp_z_end', 'root_settle_s',
                                                'feet_touch_s', 'hand_touch_s', 'feet_support_z', 'hands_support_z',
                                                'markers')}
        per[s] = e
    for name in sorted({n for r in rows for n in r['ch']['names']}):
        s_ = short(name)
        if s_ in per:
            continue
        per[s_] = dict(dump=dump_names.get(name), occurrences=0, transient_only=True,
                       replaced_by={b: v for (a, b), v in transient_first.items() if a == s_},
                       geometry={k: geos[name][k] for k in ('length_s', 'disp_y_end', 'disp_z_end', 'root_settle_s',
                                                             'feet_touch_s', 'hand_touch_s', 'markers')}
                       if geos.get(name) is not None else None)

    # Q4: time budget per SlideOver -> Exit pair
    wd2 = sm.wd2_dive()
    budget_rows = []
    for slide_name in PAIR_CLIPS:
        sc = slide['per_clip'].get(slide_name)
        if sc is None or not sc.get('acu_played_s') or sc.get('window') is None:
            continue
        # the Exit that follows each SlideOver in the captures: Dn100 after Up100 obstacles, Fd250-Dn100 after Up200
        exit_key = 'Fd250cm-Dn100cm' if slide_name.endswith('Up200cm') else MAIN_EXIT
        ex = per.get(exit_key, {})
        eo = ex.get('o') if ex.get('o') is not None else 0.0
        er = ex.get('r') if ex.get('r') is not None else 1.0
        ex_map_src = 'fit' if ex.get('o') is not None else 'assumed o = 0, r = 1'
        so, sr = sc['map_used']['o'], sc['map_used']['r']
        W = sc['window']
        t_take = sc['acu_played_s']['takeoff']
        feet_clip = sc['acu_clip_s']['feet_down']
        feet_played_slide = sc['acu_played_s']['feet_down']
        ex_feet = ex.get('geometry', {}).get('feet_touch_s')
        ex_feet_played = None if (ex_feet is None or ex_feet < eo) else W + (ex_feet - eo) / er
        ex_window = ex.get('window_median') or 0.0
        exit_end = W + ex_window
        k_feet = None
        if ex_feet_played is not None and t_take is not None:
            k_feet = (ex_feet_played - t_take) / (wd2['feet_down'] - wd2['takeoff'])
        placements = {}
        for label, k in (('k = 1', 1.0), ('k = k_feet', k_feet)):
            if k is None:
                continue
            placements[label] = dict(k=round(k, 3), events=place_events(
                dict(takeoff=wd2['takeoff'], hands_start=wd2['hands_start'], hands_end=wd2['hands_end'],
                     roll_end=wd2['roll_end'], feet_down=wd2['feet_down'], run_out=wd2['run_out']),
                t_take, k, W, (so, sr), (eo, er), wd2))
        budget_rows.append(dict(
            slide=slide_name[len(sm.SLIDE_PREFIX):], exit=exit_key, exit_map_source=ex_map_src,
            slide_map=dict(o=so, r=sr), takeoff_played=t_take, W=W,
            slide_feet_played=feet_played_slide, slide_feet_clip=feet_clip,
            exit_feet_clip=ex_feet, exit_feet_played=ex_feet_played,
            exit_window=ex_window, exit_end=exit_end, total_from_takeoff=exit_end - t_take,
            boundary_wd2_clock_k1=wd2['takeoff'] + (W - t_take),
            wd2_run_out_from_takeoff_k1=wd2['run_out'] - wd2['takeoff'],
            k_feet=None if k_feet is None else round(k_feet, 3), placements=placements))

    # Q5: hand-off to the next clip, from the Exit dump (played end) and the next clip's first frame
    hand = []
    main = geos.get('xx_Parkour_Muraille_Exit_' + MAIN_EXIT)
    ex_end_s = per.get(MAIN_EXIT, {}).get('window_median') or 0.23
    for nm in HANDOFF_NEXT:
        nd = dumps.load_by_name(nm)
        if main is None or nd is None:
            continue
        g = handoff_gap(main, nd, (per.get(MAIN_EXIT, {}).get('o') or 0.0)
                        + (per.get(MAIN_EXIT, {}).get('r') or 1.0) * ex_end_s)
        g.update(next=nm, next_in_dump=True, exit_at_clip_s=round(
            (per.get(MAIN_EXIT, {}).get('o') or 0.0) + (per.get(MAIN_EXIT, {}).get('r') or 1.0) * ex_end_s, 3))
        hand.append(g)

    return dict(paths=paths, rows=rows, pred_full=pred_full, pred_fam=pred_fam, pairs=pairs, fits=fits, per=per,
                geos=geos, budget=budget_rows, handoff=hand, wd2=wd2, slide=slide)


# ---------------------------------------------------------------------------------------------- plots

def dump_dz(res, exit_short, t_a, t_b):
    g = res['geos'].get(EXIT_PREFIX + '_' + exit_short)
    if g is None:
        return None
    return float(np.interp(t_b, g['_t'], g['_disp'][:, 2]) - np.interp(t_a, g['_t'], g['_disp'][:, 2]))


def plot_entity(res, path):
    pairs = res['pairs']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap('viridis')
    hs = sorted({p['h'] for p in pairs})
    for p in pairs:
        c = cmap(hs.index(p['h']) / max(1, len(hs) - 1))
        ax1.plot(p['t'], p['z'], '-', color=c, lw=0.9, alpha=0.8)
        ax1.plot([p['tw']], [p['z_end_rise'] - p['z_W_rise']], marker='o', ms=3, color=c)
        ax2.plot(p['t'], p['y'], '-', color=c, lw=0.9, alpha=0.8)
    for ax in (ax1, ax2):
        ax.axvline(0, color='k', lw=1)
        ax.axvspan(0, 0.25, color='0.9', zorder=0)
        ax.set_xlabel('time since the Exit start W (s); W = SlideOver played window end')
        ax.grid(alpha=0.3)
        ax.set_xlim(-1.0, 0.6)
    ax1.set_ylabel('entity z relative to its value at W (m)')
    ax2.set_ylabel('entity forward distance relative to W (m)')
    ax1.set_title('Entity z through SlideOver -> Exit (colour = obstacle height)', fontsize=9)
    ax2.set_title('Entity forward distance through the same pairs', fontsize=9)
    sm_ = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(hs), max(hs)))
    fig.colorbar(sm_, ax=[ax1, ax2], label='obstacle height (m)', fraction=0.03)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_dumps(res, path):
    names = [n for n in ('xx_Parkour_Muraille_Exit_Fd100cm-Dn100cm', 'xx_Parkour_Muraille_Exit_Fd250cm-Dn100cm',
                         'xx_Parkour_Muraille_Exit_Fd250cm-Dn300cm', 'xx_Parkour_Muraille_Exit_Fd100cm-Dn500cm')
             if res['geos'].get(n) is not None]
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4), squeeze=False)
    for ax, n in zip(axes[0], names):
        g = res['geos'][n]
        ax.plot(g['disp_t'], g['disp_z'], '-', color='tab:purple', label='root z')
        ax.plot(g['disp_t'], [y / 5.0 for y in g['disp_y']], ':', color='tab:blue', label='root y / 5 (m)')
        if g['feet_touch_s'] is not None:
            ax.axvline(g['feet_touch_s'], color='tab:green', ls='--', label='feet touch')
        if g['root_settle_s'] is not None:
            ax.axvline(g['root_settle_s'], color='tab:purple', ls=':', label='root settled')
        ax.set_title(short(n), fontsize=9)
        ax.set_xlabel('clip time (s)')
        ax.set_ylabel('m')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- JSON

def play_model_json(res):
    return dict(
        _note=('Muraille_Exit clips after the SlideOver and other moves, from the captures (sessions 1-4) and the '
               'dumps. A chain is consecutive Exit starts (1 ms apart); the last member is the played clip. window: '
               'played seconds from its start to the next clip start (any name). o, r: fit (usable fits, median). '
               'geometry: clip seconds from the dump.'),
        per_clip=res['per'],
        budget_pairs=res['budget'],
        handoff=res['handoff'],
        predecessors=[dict(exit=k[0], predecessor=k[1], n=v) for k, v in res['pred_full'].most_common()],
        predecessor_families=[dict(exit=k[0], family=k[1], n=v) for k, v in res['pred_fam'].most_common()],
        pairs=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items() if k not in ('t', 'z', 'y')}
               for p in res['pairs']],
        fits=res['fits'],
        wd2_frontdive=res['wd2'],
    )


# ---------------------------------------------------------------------------------------------- report

FINDINGS = """**1. Is the entity still at obstacle height at W, and does Exit carry the descent?** Depends on the obstacle.
- Up100 (Fd100-Up100 n = 20, Fd300-Up100 n = 4, Fd200-Up100 n = 4): the entity has already landed when W comes.
  z at W: median 0.05 m for Fd100-Up100, 0.02 m for Fd300-Up100; Fd200-Up100 is still 0.2-0.65 m up at W. The
  SlideOver carries the descent and the landing, and the Exit does not.
- Up200 (Fd200-Up200, n = 3): still at obstacle height at W (1.66-2.00 m) and on the ground 0.3 s later (0.02-0.14 m).
  Here the Exit carries the descent.
- The dumps disagree with the captures about the Exit descent. Fd100-Dn100's root drops 1.00 m in its first 0.17 s, but
  the played Exit starts at clip time 0.21 s, after the drop, so the drop is never played (dump root dz over the window
  0.00; entity z flat). For Fd250-Dn100 the offset is -0.08 s, so the drop is played: the entity falls 1.6 m in the window
  against 1.0 m in the dump (0.6 m unexplained).

**2. Exit clips (played).** Fd100cm-Dn100cm (0.867 s dump): 80 chains, of which 41 start one and 39 follow Fd100-Dn500
1 ms later; 29 follow a SlideOver. Window median 0.228 s (0.05-0.27). o 0.21, r 0.97 (12 usable fits of 42). Root drop
-1.00 m settled at clock 0.17 s; feet on the ground at 0.37 s; no hand contact. Next clip: Nav_Run_3-Loop (28),
Nav_Sprint_3-Loop_BankLeft_Light (27), others. Fd250cm-Dn100cm (1.067 s): 12 chains, all after Fd250-Dn300 (1 ms);
3 after a SlideOver. Window median 0.317 s (0.28-0.38). Fits are weak: 1 usable of 12 (o -0.08, r 1.34). Root drop
-1.00 m, settled at 0.27 s, feet touch 0.40 s. Next: LandSoft_tr_Run02_A (8), LandMid_tr_RunImpultion (3). Fd100-Dn500 and
Fd250-Dn300 are never the played clip (39 and 12 starts, each replaced 1 ms later); their dumps drop 5 m and 3 m.
The Exit dumps' edge markers do not follow the clip names (height 0, landing 1 m down), so they are not used.

**3. Who uses the Exit clips.** 92 chains in the captures: SkipOver 50 (Fd100-Up100 26, Fd010-Up100 13, others), SlideOver
32 (Fd100-Up100 20, Fd200-Up100 4, Fd300-Up100 4, Fd100-Up050 1 into Dn100; Fd200-Up200 3 into Fd250-Dn100), drop/sneak 5,
Leaping 4, other 1. SkipOver, Leaping and the drop/sneak moves enter the same Exit clips, so the swap is needed and must be
conditioned on a SlideOver having just started (the plugin has to check the previous clip). These are capture samples, so
other moves may use them too.

**4. Time budget and WD2 placement (played seconds from the SlideOver takeoff).**
- Fd100-Up100 -> Dn100 (main case, 20 pairs; own map o 0.42, r 1.05): takeoff 0.025; W 0.744; ACU landing in the SlideOver
  at 0.695; Exit played 0.744-0.972 (0.23 s). Takeoff to Exit end: 0.95 s. WD2 run-out at k = 1 would fall 0.2 s after
  the Exit ends, in the next clip.
- k = 1 is possible across the two clips here. The SlideOver/Exit boundary falls at WD2 clock 1.17, i.e. the roll end, so
  takeoff, hands and roll are in the SlideOver, roll end at the boundary, feet down at Exit clock 0.24 (played 0.775),
  and run-out after the Exit ends. Feet down lands 0.08 s after the ACU landing, which is acceptable.
- k = 0.89 matches the ACU landing exactly (feet down played 0.70, roll end 0.68, both in the SlideOver). Either way the
  roll is not compressed beyond 0.89.
- The proposed split (hands and roll in the Exit) does not fit this pair. The Exit starts after the landing, at W,
  and the hands plant (played 0.29-0.50) and roll are before W. Moving them into the Exit needs W earlier, which the
  SlideOver's own clip timing sets (the switch comes at its clip time 1.2 s). A swap of the Exit tracks alone cannot
  do it.
- Fd200-Up200 -> Fd250-Dn100 (3 pairs, 1 usable fit): W 0.821 (WD2 clock 1.24 at k = 1); the entity is still 1.7-2 m up
  at W and lands at about 1.1-1.2 played, inside the Exit. k = 1 would land feet on the ground 0.4 s early, while the
  body is still in the air. k = 1.53 puts feet down at the landing (1.18); the roll then straddles the boundary (its
  end at Exit clock 0.36).
- Fd200-Up100 (4 pairs): entity 0.2-0.65 m up at W; landing at 0.97-1.11 played; k 1.25-1.45.
- Fd300-Up100 (4 pairs, 1 usable fit): landing in the SlideOver at 1.22 (W 1.29); k = 1.53 (feet down 1.22, roll
  end 1.19, inside W). The 0.4 s of spare window is for a held landing, or k = 1 with a held landing.

**5. Hand-off back to running (Exit end, clock 0.43 s, to the next clip's first frame; dumps).**
Next = Nav_Run_3-Loop: bone RMS 27 cm, Hips 0.60 vs 0.72 m above the feet, root speed 4.8 vs 2.5 m/s. Nav_Sprint_3-Loop_V02:
29 cm, 0.60 vs 0.75 m, 4.8 vs 3.5 m/s. The landing clips (LandSoft, LandMid) start from rest at 0.3-0.4 m/s and differ by
48-57 cm. Nav_Run and Nav_Sprint are the nearest targets; the blend has to absorb about 0.1 m of height and 2 m/s of speed.

**6. Caveats.** The root speed in the dumps (4.6-4.8 m/s) is not the entity's: in game the entity moves 2.5 m/s in the
SlideOver and 2.8 m/s in the Exit window (medians of |dy| over the window), so the dump root travel does not set the
game's speed. The Exit fits are weak (Dn100: 12 of 42 usable; Fd250-Dn100: 1 of 12). The Up200 and Fd300 groups have
1-3 pairs each. The WD2 root travel is not measured here.

**Recommendation.**
1. Keep the dive in the SlideOver for Up100 obstacles (Fd100-Up100 is 20 of 32 SlideOver->Exit pairs): takeoff at
   clock 0.45, hands and roll before W, feet down at the ACU landing (k = 0.89, or k = 1 with feet down 0.08 s late).
   Use the Exit only as the stand-up and transition into Nav_Run (0.744-0.97 played). Do not move the hands or the roll
   into the Exit.
2. For Up200 (Fd200-Up200) the Exit carries the descent and the landing. Place WD2 feet down at about 1.18 played (k ~1.5)
   and the roll tail inside the Exit.
3. Swap only when the previous clip is a SlideOver. SkipOver, Leaping and drop/sneak use the same Exit clips.
4. The Exit's clock offset (o 0.21 for Dn100, -0.08 for Fd250-Dn100) decides whether its own descent is played. Keep the
   offset when replacing the tracks, or bake the replacement's own drop and landing against the same clock.
5. Hand-off target: Nav_Run_3-Loop or Nav_Sprint_3-Loop_V02, from the Exit end pose (clock 0.43 s)."""


def write_report(res, md_path, plot_dir, findings=None):
    os.makedirs(plot_dir, exist_ok=True)
    ent_png = os.path.join(plot_dir, 'exit_entity.png')
    dump_png = os.path.join(plot_dir, 'exit_dumps.png')
    plot_entity(res, ent_png)
    plot_dumps(res, dump_png)
    L = []
    rows = res['rows']
    L += ['# Muraille_Exit chain: SlideOver -> Exit, predecessors, and the hand-off', '',
          'Generated by `pipeline/exit_model.py`. Clip time = o + r * played s for an Exit clip. "W" is the SlideOver played '
          'window end, which is the Exit start. Exit chains: 1 ms apart starts are one chain, the last member plays.',
          '', '## Chains and predecessors (Q3)', '']
    eff_counts = collections.Counter(short(r['eff']) for r in rows)
    fam = collections.Counter()
    for (e, f), v in res['pred_fam'].items():
        fam[f] += v
    L += ['- Chains in the captures: %d (%s).' % (
        len(rows), ', '.join('%s %d' % (k, v) for k, v in sorted(collections.Counter(r['session'] for r in rows).items()))),
          '- Predecessor families over all chains: ' + ', '.join('%s %d' % (k, v) for k, v in fam.most_common())
          + '. Non-SlideOver moves (SkipOver, Leaping, drop/sneak) enter the same Exit clips, so a swap has to be '
          'conditioned on a SlideOver having just started.']
    L += ['', '### Predecessor of each played Exit (counts)', '']
    L += _table(['Exit (played)', 'predecessor clip', 'n'],
                [[k[0], _clip_s(k[1]), v] for k, v in res['pred_full'].most_common()])
    L += ['', 'Exit starts that are replaced 1 ms later (never played), with the start that replaces them:', '']
    trans = [(s_, e) for s_, e in sorted(res['per'].items()) if e.get('transient_only')]
    L += _table(['Exit start (replaced)', 'replaced by (n)', 'dump root dz over clip', 'feet touch s (clip)'],
                [[s_, '; '.join('%s (%d)' % (b, v) for b, v in e['replaced_by'].items()) or '-',
                  _f((e.get('geometry') or {}).get('disp_z_end'), '%.2f'),
                  _f((e.get('geometry') or {}).get('feet_touch_s'), '%.3f')] for s_, e in trans])
    L += ['', '## Exit clips (Q2)', '']
    head = ['Exit (played)', 'dump', 'chains (as start)', 'transient', 'after SlideOver', 'window W median s (range)',
            'fits (usable)', 'o', 'r', 'root y end m', 'root z end m', 'root settled s', 'feet touch s (clip)',
            'hand touch s (clip)']
    body = []
    for s, e in sorted(res['per'].items()):
        if e.get('transient_only'):
            continue
        g = e.get('geometry', {})
        body.append([s, e['dump'] or '-', '%d (%d)' % (e['occurrences'], e['as_chain_start']), e['transient'],
                     e['after_slideover'],
                     '%s (%s-%s)' % (_f(e['window_median'], '%.3f'), _f(e['window_range'] and e['window_range'][0], '%.3f'),
                                     _f(e['window_range'] and e['window_range'][1], '%.3f')) if e['window_median'] else '-',
                     '%d (%d)' % (e['fitted'], e['usable']), _f(e['o'], '%.3f'), _f(e['r'], '%.3f'),
                     _f(g.get('disp_y_end'), '%.2f'), _f(g.get('disp_z_end'), '%.2f'), _f(g.get('root_settle_s'), '%.3f'),
                     _f(g.get('feet_touch_s'), '%.3f'), _f(g.get('hand_touch_s'), '%.3f')])
    L += _table(head, body)
    L += ['', 'Next clip after each played Exit:', '']
    L += _table(['Exit (played)', 'next clip (n)'],
                [[s, '; '.join('%s (%d)' % (_clip_s(k), v) for k, v in e['next_counts'].items())]
                 for s, e in sorted(res['per'].items()) if not e.get('transient_only')])
    L += ['', '## Entity through SlideOver -> Exit (Q1)', '']
    head = ['SlideOver (h)', 'Exit', 'SlideOver W s', 'Exit window s', 'z at W (rise)', 'z at Exit end (rise)',
            'z max (rise)', 'z at W / h', 'y at W (m)', 'y at Exit end (m)', 'dump root dz over the same window']
    body = []
    eo_map = {}
    for b in res['per'].items():
        eo_map[b[0]] = (b[1].get('o') or 0.0, b[1].get('r') or 1.0)
    for p in sorted(res['pairs'], key=lambda p: (p['slide'], p['session'], p['file'])):
        eo, er = eo_map.get(p['exit'], (0.0, 1.0))
        dz = dump_dz(res, p['exit'], eo, eo + er * p['exit_window'])
        body.append(['%s (%.1f)' % (p['slide'], p['h']), p['exit'], '%.3f' % p['W_from_slide'],
                     '%.3f' % p['exit_window'], '%.2f' % p['z_W_rise'], '%.2f' % p['z_end_rise'],
                     '%.2f' % p['z_max_rise'], '%.2f' % (p['z_W_rise'] / p['h']), '%.2f' % p['y_W'],
                     '%.2f' % p['y_end'], _f(dz, '%.2f')])
    L += _table(head, body)
    L += ['', 'Forward distance and z are from the entity matrix (captures, all sessions; no pose needed). '
          '"dump root dz" is the Exit dump root z change over the same clip window: where it is about -1 m and the '
          'entity z is flat, the drop is not played.', '']
    L += ['![entity z and forward distance through the pairs](%s)' % os.path.basename(ent_png), '']
    L += ['![Exit dumps: root z and y, ground contact](%s)' % os.path.basename(dump_png), '']
    L += ['## Time budget and WD2 placement (Q4)', '']
    head = ['SlideOver', 'Exit', 'map (SlideOver)', 'takeoff played', 'W (boundary)', 'boundary on WD2 clock (k=1)',
            'SlideOver feet down played',
            'Exit feet touch (clip s)', 'Exit feet touch played', 'Exit window', 'Exit end played',
            'takeoff -> Exit end', 'WD2 run-out span (k=1)', 'k_feet']
    body = []
    for b in res['budget']:
        body.append([b['slide'], b['exit'] + ('' if b['exit_map_source'] == 'fit' else ' (ASSUMED map)'),
                     '%.3f, %.3f' % (b['slide_map']['o'], b['slide_map']['r']), _f(b['takeoff_played'], '%.3f'),
                     _f(b['W'], '%.3f'), _f(b['boundary_wd2_clock_k1'], '%.3f'), _f(b['slide_feet_played'], '%.3f'),
                     _f(b['exit_feet_clip'], '%.3f'),
                     _f(b['exit_feet_played'], '%.3f'), _f(b['exit_window'], '%.3f'), _f(b['exit_end'], '%.3f'),
                     _f(b['total_from_takeoff'], '%.3f'), _f(b['wd2_run_out_from_takeoff_k1'], '%.3f'),
                     _f(b['k_feet'], '%.3f')])
    L += _table(head, body)
    L += ['', 'WD2 event placement (played seconds from the SlideOver start; clip = which clip and its clock):', '']
    for b in res['budget']:
        if not b['placements']:
            continue
        L += ['### %s -> %s' % (b['slide'], b['exit']), '']
        names = ('takeoff', 'hands_start', 'hands_end', 'roll_end', 'feet_down', 'run_out')
        cols = list(b['placements'].items())
        head = ['WD2 event (WD2 s)'] + ['%s: played / clip / clip s' % lbl for lbl, _ in cols]
        body = []
        for nm in names:
            row = [nm + ' (%.2f)' % res['wd2'][nm if nm != 'run_out' else 'run_out']]
            for lbl, pl in cols:
                ev = pl['events'][nm]
                row.append('%.3f / %s / %s' % (ev['played'], ev['clip'], _f(ev['clip_s'], '%.3f')))
            body.append(row)
        L += _table(head, body)
        L += ['']
    L += ['## Hand-off to the next clip (Q5)', '']
    head = ['next clip', 'bone RMS cm (Hips-relative)', 'Hips above feet: Exit end / next start (m)',
            'root vy Exit end / next start (m/s)', 'root vz Exit end / next start (m/s)']
    body = []
    for h in res['handoff']:
        body.append([_clip_s(h['next']), '%.1f' % h['bone_rms_cm'],
                     '%.2f / %.2f' % (h['hips_above_feet_exit'], h['hips_above_feet_next']),
                     '%.2f / %.2f' % (h['vy_exit'], h['vy_next']), '%.2f / %.2f' % (h['vz_exit'], h['vz_next'])])
    L += _table(head, body)
    L += ['', 'Exit side: Muraille_Exit_Fd100cm-Dn100cm at its played end (clip time %s s). Next side: the next clip at its '
          'first frame (clip time 0), dump only.' % _f(res['handoff'][0]['exit_at_clip_s'] if res['handoff'] else None,
                                                    '%.3f'), '']
    L += ['## Findings and recommendation', '', findings if findings is not None else FINDINGS, '']
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    return md_path


def _clip_s(name):
    if name is None:
        return '-'
    return name[len('xx_Parkour_'):] if name.startswith('xx_Parkour_') else name


def _f(v, fmt='%.2f'):
    if v is None:
        return '-'
    try:
        if not np.isfinite(v):
            return '-'
    except TypeError:
        return str(v)
    return fmt % v


def _table(header, rows):
    out = ['| ' + ' | '.join(str(h) for h in header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--captures-root', default=sm.CAPTURES_ROOT)
    ap.add_argument('--out-json', default=OUT_JSON)
    ap.add_argument('--out-md', default=OUT_MD)
    ap.add_argument('--plot-dir', default=PLOT_DIR)
    ap.add_argument('-q', action='store_true')
    ap.add_argument('--cache', default=None, help='pickle the analysis result here')
    ap.add_argument('--from-cache', default=None, help='rebuild the outputs from a cached analysis result')
    args = ap.parse_args(argv)
    if args.from_cache:
        import pickle
        with open(args.from_cache, 'rb') as f:
            res = pickle.load(f)
    else:
        res = run(args.captures_root, verbose=not args.q)
    if args.cache:
        import pickle
        with open(args.cache, 'wb') as f:
            pickle.dump(res, f)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(play_model_json(res), f, indent=2, default=_jsonable)
        f.write('\n')
    write_report(res, args.out_md, args.plot_dir)
    print('wrote %s\nwrote %s' % (args.out_json, args.out_md), file=sys.stderr)
    return 0


def _jsonable(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (tuple, set)):
        return list(o)
    raise TypeError('not JSON serialisable: %r' % (o,))


if __name__ == '__main__':
    sys.exit(main())
