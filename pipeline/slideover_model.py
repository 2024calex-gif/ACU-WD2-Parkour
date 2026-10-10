"""Play model for the ACU Sprint SlideOver clips (and the LandRoll clips that follow a vault), from the captures.

usage: python slideover_model.py [--captures-root DIR] [--out-json FILE] [--out-md FILE] [--plot-dir DIR] [-q]

Per occurrence (every unpatched clip entry named xx_Parkour_Sprint_SlideOver_* or xx_Parkour_LandingFwd_LandRoll_*
in the four session folders):
  - window: matchclip.match_window: the played span from the clip start to the start of the next wall-exit, landing
    or nav clip, capped at the clip length;
  - next clip: the first clip that starts after it, of any name;
  - fit: matchclip.fit_time_map(partial=True): clip time = o + r * (t - t_start), yaw about Z, RMS in cm. Partial
    poses are used (a capture that misses RightHand fits with the other fit bones); captures without bones are not fitted.

Per clip name (from the original dump; acuclip.py / contacts.py):
  - obstacle height and depth, near / far / landing edge heights, root (displacement TrackID 0) crossings of them;
  - root forward distance (y) and vertical root motion (z) over clip time;
  - ACU takeoff, hand plant and feet down (landing) in clip seconds, and in played seconds (clip time - o) / r;
  - the WD2 frontdive (data/wd2_events.json, "diveroll_frontdive") aligned at the ACU takeoff, and whether it fits the
    played window.

Writes out/match/play_model_slideover.json, out/match/slideover_report.md and PNG plots in out/match/.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

import numpy as np

import acuclip
import capture
import config
import contacts
import dumps
import matchclip as mc

FPS = acuclip.FPS
SLIDE_PREFIX = 'xx_Parkour_Sprint_SlideOver_'
ROLL_PREFIX = 'xx_Parkour_LandingFwd_LandRoll_'
SESSION_DIRS = ('session1_20261008', 'session2_20261008', 'session3_20261008', 'session4_20261008')
CAPTURES_ROOT = os.path.join(config.V2, 'data', 'captures')
WD2_EVENTS = os.path.join(config.V2, 'data', 'wd2_events.json')
OUT_DIR = os.path.join(config.OUT, 'match')
OUT_JSON = os.path.join(OUT_DIR, 'play_model_slideover.json')
OUT_MD = os.path.join(OUT_DIR, 'slideover_report.md')
REF_CLIP = 'xx_Parkour_Sprint_SlideOver_Fd100cm-Up100cm'   # reference time map: its usable fits
USABLE_RMS_CM = 15.0      # above this the time map does not describe the in-game pose (best SkipOver group: 12-14 cm)
BLEND_OUT_S = 0.25        # the game samples the played clip for about this long while it blends out
ROOT_STEP_S = 0.1         # root curve sampling in the JSON


# ---------------------------------------------------------------------------------------------- names and WD2

def family(name):
    """'slide' for Sprint SlideOver clips, 'roll' for LandRoll clips, None for anything else."""
    if not name:
        return None
    if name.startswith(SLIDE_PREFIX):
        return 'slide'
    if name.startswith(ROLL_PREFIX):
        return 'roll'
    return None


def wd2_dive(path=WD2_EVENTS):
    """WD2 frontdive event times (WD2 clip seconds) from data/wd2_events.json."""
    with open(path, encoding='utf-8') as f:
        d = json.load(f)['diveroll_frontdive']
    return dict(takeoff=float(d['takeoff']), hands_start=float(d['hand_plant_start']),
                hands_end=float(d['hand_plant_end']), roll_start=float(d['roll'][0]),
                roll_end=float(d['roll'][1]), feet_down=float(d['landing']), run_out=float(d['next_step']))


# ---------------------------------------------------------------------------------------------- pure math

def played_time(clip_s, o, r):
    """Played seconds since the clip start of a clip time: clip time = o + r * played."""
    return (float(clip_s) - float(o)) / float(r)


def clip_time(played_s, o, r):
    """Clip time at played seconds since the clip start."""
    return float(o) + float(r) * float(played_s)


def next_clip_any(cap, clip_index):
    """(name, t) of the first clip after cap.clips[clip_index] that starts later, of any name; (None, None) if none."""
    t0 = float(cap.clips[clip_index]['t'])
    for c in cap.clips:
        if float(c['t']) > t0:
            return c.get('name'), float(c['t'])
    return None, None


def first_crossing(t, v, level):
    """First time at which v reaches level, linearly interpolated between samples; None when it never does."""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    hit = np.nonzero(v >= level)[0]
    if hit.size == 0:
        return None
    i = int(hit[0])
    if i == 0:
        return float(t[0])
    v0, v1 = v[i - 1], v[i]
    f = 0.0 if v1 == v0 else (level - v0) / (v1 - v0)
    return float(t[i - 1] + f * (t[i] - t[i - 1]))


def dive_on_played(takeoff_played, wd2):
    """WD2 frontdive events on the played time line, with the WD2 takeoff placed at takeoff_played (rate 1)."""
    off = float(takeoff_played) - wd2['takeoff']
    return {k: v + off for k, v in wd2.items()}


def dive_check(takeoff_played, window_s, wd2, blend_out=BLEND_OUT_S):
    """Does the WD2 dive, aligned at the ACU takeoff, fit the played window of the SlideOver clip?

    spill_*: time past the window end (0 when inside). roll_inside: the roll ends inside the window. fits_window:
    the feet are down inside the window. fits_with_blend: feet down inside the window plus the out-blend.
    takeoff_played_ok: the ACU takeoff is played at all (not before the clip time o).
    """
    ev = dive_on_played(takeoff_played, wd2)
    W = float(window_s)
    return dict(
        events=ev,
        window_s=W,
        spill_roll_end=max(0.0, ev['roll_end'] - W),
        spill_feet_down=max(0.0, ev['feet_down'] - W),
        spill_run_out=max(0.0, ev['run_out'] - W),
        roll_inside=bool(ev['roll_end'] <= W),
        fits_window=bool(ev['feet_down'] <= W),
        fits_with_blend=bool(ev['feet_down'] <= W + blend_out),
        takeoff_played_ok=bool(takeoff_played >= 0.0),
    )


def dive_factor(takeoff_played, window_s, acu_span_s, wd2):
    """Time factor k for the WD2 dive (WD2 durations x k, takeoff fixed at takeoff_played).

    k_acu: the WD2 takeoff-to-feet-down span equals the ACU span (feet down match the ACU's).
    k_roll: the roll ends at the window end (W - takeoff) / (roll end - takeoff); a larger k would run the roll past
    the next clip. k = min(k_acu, k_roll). Returns dict(k_acu, k_roll, k), None values when not computable.
    """
    k_acu = float(acu_span_s) / (wd2['feet_down'] - wd2['takeoff']) if acu_span_s is not None else None
    room = float(window_s) - float(takeoff_played)
    k_roll = room / (wd2['roll_end'] - wd2['takeoff']) if room > 0 else None
    cands = [k for k in (k_acu, k_roll) if k is not None]
    return dict(k_acu=k_acu, k_roll=k_roll, k=min(cands) if cands else None)


def dive_check_k(takeoff_played, window_s, wd2, k, blend_out=BLEND_OUT_S):
    """dive_check with the WD2 durations scaled by k (k = 1 is dive_check)."""
    scaled = {key: wd2['takeoff'] + (v - wd2['takeoff']) * k for key, v in wd2.items()}
    return dive_check(takeoff_played, window_s, scaled, blend_out)


def scale_to_wd2(acu_span_s, wd2):
    """Factor on the WD2 timing that makes the WD2 takeoff-to-feet-down span equal acu_span_s (>1: slow the WD2)."""
    return float(acu_span_s) / (wd2['feet_down'] - wd2['takeoff'])


# ---------------------------------------------------------------------------------------------- occurrences

def capture_paths(root=CAPTURES_ROOT, sessions=SESSION_DIRS):
    out = []
    for s in sessions:
        out += glob.glob(os.path.join(root, s, 'cap_*.json'))

    def key(p):
        m = re.search(r'_(\d+)\.json$', p)
        return (SESSION_DIRS.index(os.path.basename(os.path.dirname(p))), int(m.group(1)) if m else 0)
    return sorted(out, key=key)


def occurrence_indices(cap):
    """Clip indices of the unpatched SlideOver and LandRoll entries of a capture."""
    return [i for i, c in enumerate(cap.clips) if family(c.get('name')) and not c.get('patched')]


def analyse_occurrence(cap, clip_index, dump, path=''):
    """One occurrence: window, next clip (of the window kind and of any kind), and the time-map fit if the capture
    has a pose. Fields o, r, yaw, rms are None when there is no fit (skip says why)."""
    clip = cap.clips[clip_index]
    w = mc.match_window(cap, clip_index)
    nany, nany_t = next_clip_any(cap, clip_index)
    m = re.search(r'_(\d+)\.json$', path)
    row = dict(file=os.path.basename(path), session=os.path.basename(os.path.dirname(path)),
               index=int(m.group(1)) if m else None,
               clip=clip['name'], family=family(clip['name']), t_start=w['t_start'], window_s=w['length_s'],
               end_reason=w['end_reason'], next_name=w['next_name'], next_t=w['next_t'],
               next_any=nany, next_any_t=nany_t, trigger=cap.trigger, variant=cap.variant,
               o=None, r=None, yaw=None, rms=None, rms_naive=None, rms_r1=None, n_frames=0, n_window=0,
               bones=None, poor=None, at_edge=None, usable=False, skip=None,
               dump=dump['Name'] if dump else None)
    if dump is None:
        row['skip'] = 'no original dump'
        return row
    if cap.bones_space != 'entity' or cap.bones.get(mc.ROOT_BONE) is None:
        row['skip'] = 'no entity-space Hips (no pose)'
        return row
    try:
        fit = mc.fit_time_map(cap, dump, w, partial=True)
    except ValueError as e:
        row['skip'] = str(e)
        return row
    row.update(o=fit['o'], r=fit['r'], yaw=fit['yaw_deg'], rms=fit['rms_cm'], rms_naive=fit['rms_naive_cm'],
               rms_r1=fit['rms_r1']['rms_cm'], n_frames=fit['n_frames'],
               n_window=int(np.count_nonzero((cap.t >= w['t_start']) & (cap.t <= w['t_end']))),
               bones=list(fit['bones']), poor=fit['poor'], at_edge=fit['at_edge'])
    in_grid = (mc.O_GRID[0] <= fit['o'] <= mc.O_GRID[1]) and (mc.R_GRID[0] <= fit['r'] <= mc.R_GRID[1])
    row['in_grid'] = bool(in_grid)
    row['usable'] = bool(row['rms'] <= USABLE_RMS_CM and in_grid and not (fit['at_edge']['o'] or fit['at_edge']['r']))
    return row


def rms_at_reference(cap, clip_index, dump, o_ref, r_ref):
    """Best-yaw RMS (cm) of the window with the time map fixed at (o_ref, r_ref): does a shared time map explain it."""
    w = mc.match_window(cap, clip_index)
    try:
        ctx = mc._prepare(cap, dump, w, partial=True)
    except ValueError:
        return None
    cost = mc._cost(ctx, float(o_ref), float(r_ref), mc._grid(mc.YAW_GRID))
    return float(np.min(cost)) if np.isfinite(cost).any() else None


# ---------------------------------------------------------------------------------------------- dump geometry

def _sec(e):
    return None if not e else e.get('seconds')


def _sec_frame(f):
    return None if f is None else round(float(f) / FPS, 4)


def dump_geometry(dump):
    """Obstacle, root and contact events of an original SlideOver clip, in clip seconds.

    None for a clip without edge markers. The root is the root displacement track (TrackID 0): its y is the forward
    distance and its z the vertical root motion (entity world = entity + displacement). Edge crossings use the
    root y against the obstacle edges (world y, constant over the clip).
    """
    try:
        ob = acuclip.obstacle(dump)
    except ValueError:
        return None
    seq = acuclip.pose_sequence(dump)
    ev = contacts.events(seq, dump)
    t = seq.frames / float(FPS)
    disp = acuclip.markers(dump, seq.frames)['disp']
    grid = np.arange(0.0, float(dump['Length']) + 1e-9, ROOT_STEP_S)
    geo = dict(
        length_s=float(dump['Length']),
        height=ob['height'], depth=ob['depth'], near_y=ob['near_y'], far_y=ob['far_y'],
        landing_y=ob.get('landing_y'), landing_z=ob.get('landing_z'),
        root_end_y=float(disp[-1, 1]), root_z_max=float(np.max(disp[:, 2])),
        root_cross_near_s=first_crossing(t, disp[:, 1], ob['near_y']),
        root_cross_far_s=first_crossing(t, disp[:, 1], ob['far_y']),
        root_cross_landing_s=(first_crossing(t, disp[:, 1], ob['landing_y'])
                              if ob.get('landing_y') is not None else None),
        takeoff_s=_sec(ev['takeoff']), hand_plant_s=_sec(ev['hand_plant']),
        foot_on_top_s=_sec(ev['foot_on_top']), feet_down_s=_sec(ev['landing']),
        near_foot_s=_sec_frame(ev['near_frame']), far_foot_s=_sec_frame(ev['far_frame']),
        takeoff_foot=(ev['takeoff'] or {}).get('foot'), feet_down_foot=(ev['landing'] or {}).get('foot'),
        hand_plant_hand=(ev['hand_plant'] or {}).get('hand'),
        hand_top_contacts={k: [[round(a / FPS, 3), round(b / FPS, 3)] for a, b, c in v if c == 'top']
                           for k, v in ev['contacts'].items() if k in ('LeftHand', 'RightHand')},
        root_t=[round(float(x), 3) for x in grid],
        root_y=[round(float(x), 3) for x in np.interp(grid, t, disp[:, 1])],
        root_z=[round(float(x), 3) for x in np.interp(grid, t, disp[:, 2])],
    )
    # full-rate curves for the plots only (not written to the JSON)
    geo['_t'] = t
    geo['_disp'] = disp
    return geo


# ---------------------------------------------------------------------------------------------- per clip

def _median(vals):
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.median(vals)) if vals else None


def clip_summary(name, rows, geo, wd2, handle, ref=None):
    """Play model entry of one clip name.

    o, r: median over the usable fits of this clip (None when it has none). The played times use that map (source
    'own'); without a usable fit they use the reference map ref = (o, r) (source 'assumed'), flagged wherever printed.
    window: median over the occurrences whose window ends at a next clip (measured). The other windows are censored
    (clip length or capture end) and are only listed.
    """
    occ = [r for r in rows if r['clip'] == name]
    usable = [r for r in occ if r['usable']]
    fitted = [r for r in occ if r['o'] is not None]
    o_own = _median([r['o'] for r in usable])
    r_own = _median([r['r'] for r in usable])
    if o_own is not None:
        o_map, r_map, map_src = o_own, r_own, 'own'
    elif ref is not None:
        o_map, r_map, map_src = ref[0], ref[1], 'assumed (reference clip map)'
    else:
        o_map, r_map, map_src = 0.0, 1.0, 'assumed (o = 0, r = 1)'
    measured = [r['window_s'] for r in occ if r['end_reason'] == 'next clip']
    window = _median(measured)
    nxt = collections.Counter(r['next_name'] or ('(capture end)' if r['end_reason'] == 'capture end'
                                                 else '(clip length)') for r in occ)
    nxt_any = collections.Counter(r['next_any'] or '(none in capture)' for r in occ)
    entry = dict(
        family=family(name), dump=handle, n=len(occ), n_fit=len(fitted), n_usable=len(usable),
        o=o_own, r=r_own, map_used=dict(o=o_map, r=r_map, source=map_src),
        n_window_measured=len(measured), window=window,
        window_min=float(min(measured)) if measured else None,
        window_max=float(max(measured)) if measured else None,
        window_censored=[round(r['window_s'], 3) for r in occ if r['end_reason'] != 'next clip'],
        next_clip_counts=dict(nxt.most_common()), next_any_counts=dict(nxt_any.most_common()),
    )
    if geo is None:
        return entry
    entry['obstacle'] = dict(height=geo['height'], depth=geo['depth'], near_y=geo['near_y'], far_y=geo['far_y'],
                             landing_y=geo['landing_y'], landing_z=geo['landing_z'], length_s=geo['length_s'])
    entry['root'] = dict(end_y=geo['root_end_y'], z_max=geo['root_z_max'],
                         cross_near_clip_s=geo['root_cross_near_s'], cross_far_clip_s=geo['root_cross_far_s'],
                         cross_landing_clip_s=geo['root_cross_landing_s'])
    to_c, hp_c, fd_c = geo['takeoff_s'], geo['hand_plant_s'], geo['feet_down_s']
    entry['acu_clip_s'] = dict(takeoff=to_c, hand_plant=hp_c, foot_on_top=geo['foot_on_top_s'],
                               feet_down=fd_c, near_foot=geo['near_foot_s'], far_foot=geo['far_foot_s'])
    entry['acu_feet'] = dict(takeoff=geo['takeoff_foot'], feet_down=geo['feet_down_foot'],
                             hand_plant=geo['hand_plant_hand'])
    entry['hand_top_contacts_clip_s'] = geo['hand_top_contacts']
    o, rr = o_map, r_map
    played = {}
    for k, v in (('takeoff', to_c), ('hand_plant', hp_c), ('feet_down', fd_c),
                 ('near_foot', geo['near_foot_s']), ('far_foot', geo['far_foot_s'])):
        played[k] = None if v is None else round(played_time(v, o, rr), 3)
    entry['acu_played_s'] = played
    if window is not None:
        entry['window_clip_end_s'] = round(clip_time(window, o, rr), 3)
    if to_c is not None and fd_c is not None:
        entry['acu_takeoff_to_feet_clip_s'] = round(fd_c - to_c, 3)
        entry['acu_takeoff_to_feet_played_s'] = round((fd_c - to_c) / rr, 3)
        entry['wd2_scale_to_acu_span'] = round(scale_to_wd2((fd_c - to_c) / rr, wd2), 3)
    if to_c is not None and window is not None:
        chk = dive_check(played_time(to_c, o, rr), window, wd2)
        t_to = played_time(to_c, o, rr)
        span_acu = None if fd_c is None else (fd_c - to_c) / rr
        fac = dive_factor(t_to, window, span_acu, wd2)
        k_used = fac['k'] if fac['k'] is not None else 1.0
        chk_k = dive_check_k(t_to, window, wd2, k_used)
        entry['wd2_dive_k'] = dict(
            k_acu=None if fac['k_acu'] is None else round(fac['k_acu'], 3),
            k_roll=None if fac['k_roll'] is None else round(fac['k_roll'], 3),
            k_used=round(k_used, 3),
            events_played={key: round(v, 3) for key, v in chk_k['events'].items()},
            spill_roll_end=round(chk_k['spill_roll_end'], 3), spill_feet_down=round(chk_k['spill_feet_down'], 3),
            spill_run_out=round(chk_k['spill_run_out'], 3), roll_inside=chk_k['roll_inside'],
            fits_window=chk_k['fits_window'], fits_with_blend=chk_k['fits_with_blend'],
        )
        entry['wd2_dive'] = dict(
            takeoff_played=round(played_time(to_c, o, rr), 3),
            events_played={k: round(v, 3) for k, v in chk['events'].items()},
            spill_roll_end=round(chk['spill_roll_end'], 3), spill_feet_down=round(chk['spill_feet_down'], 3),
            spill_run_out=round(chk['spill_run_out'], 3), roll_inside=chk['roll_inside'],
            fits_window=chk['fits_window'], fits_with_blend=chk['fits_with_blend'],
            takeoff_played_ok=chk['takeoff_played_ok'],
            hands_on_ground_vs_acu_hand_top_s=None if hp_c is None else round((hp_c - to_c) / rr, 3),
        )
    return entry


def run(captures_root=CAPTURES_ROOT, verbose=True):
    """All occurrences, per-clip geometry and summaries. Returns dict(rows, per_clip, geos, wd2, paths)."""
    wd2 = wd2_dive()
    rows = []
    dump_cache = {}
    occ_list = []
    paths = capture_paths(captures_root)
    for p in paths:
        cap = capture.load(p)
        for i in occurrence_indices(cap):
            clip = cap.clips[i]
            key = (clip.get('handle'), clip.get('name'))
            if key not in dump_cache:
                dump_cache[key] = mc.dump_for_clip(clip)
            row = analyse_occurrence(cap, i, dump_cache[key], p)
            rows.append(row)
            occ_list.append((row, cap, i, dump_cache[key]))
            if verbose:
                print('%-9s %-28s %-46s o=%s r=%s rms=%s %s' % (
                    row['session'][:9], row['file'][4:-5], row['clip'][-46:],
                    '-' if row['o'] is None else '%.3f' % row['o'],
                    '-' if row['r'] is None else '%.3f' % row['r'],
                    '-' if row['rms'] is None else '%.1f' % row['rms'], row['skip'] or ''), file=sys.stderr)
    # reference time map: median of the usable fits of the most-measured SlideOver clip
    ref_rows = [r for r in rows if r['clip'] == REF_CLIP and r['usable']]
    ref = (_median([r['o'] for r in ref_rows]), _median([r['r'] for r in ref_rows])) if ref_rows else None
    for row, cap, i, dump in occ_list:
        row['rms_ref'] = None
        if ref is not None and row['family'] == 'slide':
            row['rms_ref'] = rms_at_reference(cap, i, dump, ref[0], ref[1])
    handles = {nm: h for h, nm, _ in dumps.list_dumps()}
    per_clip, geos = {}, {}
    for name in sorted({r['clip'] for r in rows}):
        dump = next((v for v in dump_cache.values() if v is not None and v.get('Name') == name), None)
        geo = dump_geometry(dump) if (dump is not None and family(name) == 'slide') else None
        geos[name] = geo
        per_clip[name] = clip_summary(name, rows, geo, wd2, handles.get(name), ref=ref)
    return dict(rows=rows, per_clip=per_clip, geos=geos, wd2=wd2, paths=paths, ref=ref)


def play_model_json(result):
    """JSON in the style of play_model.json: per_clip with o, r, window, n, plus the new fields."""
    return dict(
        _note=('Sprint SlideOver and LandRoll clips as played in the game, from the flight-recorder captures '
               '(sessions 1-4, unpatched occurrences). Clip time = o + r * (played s). window = played s from the clip '
               'start to the next wall-exit / landing / nav clip start. "*_clip_s" are clip seconds, "*_played_s" are '
               '(clip time - o) / r. wd2_dive: the WD2 frontdive (data/wd2_events.json) aligned at the ACU takeoff, '
               'rate 1. usable_rms_cm: fits above this are not used for o and r when another fit is available.'),
        usable_rms_cm=USABLE_RMS_CM,
        reference=dict(clip=REF_CLIP, o=result['ref'][0] if result['ref'] else None,
                       r=result['ref'][1] if result['ref'] else None),
        blend_out_s=BLEND_OUT_S,
        wd2_frontdive=result['wd2'],
        per_clip=result['per_clip'],
        occurrences=result['rows'],
    )


# ---------------------------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--captures-root', default=CAPTURES_ROOT)
    ap.add_argument('--out-json', default=OUT_JSON)
    ap.add_argument('--out-md', default=OUT_MD)
    ap.add_argument('--plot-dir', default=OUT_DIR)
    ap.add_argument('-q', action='store_true', help='no per-occurrence progress')
    args = ap.parse_args(argv)
    import slideover_report as rep
    res = run(args.captures_root, verbose=not args.q)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(play_model_json(res), f, indent=2, default=_jsonable)
        f.write('\n')
    rep.write_report(res, args.out_md, args.plot_dir)
    print('wrote %s\nwrote %s' % (args.out_json, args.out_md), file=sys.stderr)
    return 0


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (tuple, set)):
        return list(o)
    raise TypeError('not JSON serialisable: %r' % (o,))


if __name__ == '__main__':
    sys.exit(main())
