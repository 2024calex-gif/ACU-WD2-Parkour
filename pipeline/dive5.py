"""T18 (dive5): the dive roll on the 1 m SlideOver xx_Parkour_Sprint_SlideOver_Fd100cm-Up100cm (HEX 47BFB02DF).

Reuses dive4b (time map, world target, Exit carry, world metrics, judgement) and dive4c (the floor that excludes the hands
during the plant). The differences are:
  * the context is built for the Up100 clip (dive4.Ctx with its module-level SO name / hex set during the build only);
  * the closer hand must be at or beyond the far edge (4.30 + 1.00 ... 5.30) + 0.20 = 5.50 m;
  * a raised-cosine bump on the Reference z (the pelvis target) lifts the flight over the 1 m top. Its span starts before
    T_to (the body gathers upward) and ends at the plant start or at the last sample that still needs box clearance;
    its amplitude is the smallest one that gives 5 cm clearance of the non-finger body over the box top (box need from a
    zero-bump bake);
  * the scan is widened to T_to 0.47 .. 0.90 and s 0.90 .. 1.50, with the hand plant required to start by W (so the hands
    plant inside the SlideOver window, where the IK acts);
  * ease off, k_roll 1.0, WD2 tail, gated Exit (F7919BDF0), both entries enabled with the preview override.

Outputs (only under out\\dive5): patches\\47BFB02DF.baked.json, patches\\F7919BDF0.baked.json, patches\\mapping_dive5.json,
joined_sheet.png, joined_sheet.gif, report.md, scan.md, scan.csv, summary.json.
Run: python dive5.py            (writes out\\dive5)
     python dive5.py --scan     (prints the candidates, writes nothing)
"""
import csv
import json
import os
import sys

import numpy as np

import acuclip
import bake2
import config
import dive4 as D4
import dive4b as B
import dive4c as C
import dive_chain2 as DC2
import wd2_path as WP

SO_NAME = 'xx_Parkour_Sprint_SlideOver_Fd100cm-Up100cm'
SO_HEX = '47BFB02DF'
EXIT_NAME, EXIT_HEX = D4.EXIT_NAME, D4.EXIT_HEX
OVERRIDE = 'preview: dive roll on 1 m SlideOver'
OUT_DIR = os.path.join(config.OUT, 'dive5')
PATCH_DIR = os.path.join(OUT_DIR, 'patches')
MAPPING = os.path.join(PATCH_DIR, 'mapping_dive5.json')
REPORT = os.path.join(OUT_DIR, 'report.md')
SCAN_CSV = os.path.join(OUT_DIR, 'scan.csv')
SCAN_MD = os.path.join(OUT_DIR, 'scan.md')
SUMMARY = os.path.join(OUT_DIR, 'summary.json')
SHEET_PNG = os.path.join(OUT_DIR, 'joined_sheet.png')
SHEET_GIF = os.path.join(OUT_DIR, 'joined_sheet.gif')

HAND_MIN5 = 5.50                 # m: far edge 5.30 + 0.20
BOX_TOP = 1.00                   # m
BOX_CLEAR = 0.05                 # m, the bump gives this clearance for the non-finger body
T_GRID5 = np.round(np.arange(0.47, 0.9001, 0.01), 2)
S_GRID5 = np.round(np.arange(0.90, 1.5001, 0.01), 2)
SHIFT5 = np.round(np.arange(-0.20, 0.3001, 0.01), 2)
PHASE_T = (0.76, 0.80, 0.84, 0.88)  # takeoffs whose best row is also baked (W-phase candidates)
PRE_TO = 0.20                    # s, the bump starts this long before T_to (the body gathers upward)
PLANT_LEAD = 0.30                # s, WD2 0.75 s after the takeoff 0.45 s at k = 1: the plant start is T_to + 0.30

BUMP = dict(t0=None, t1=None, A=0.0)          # the bump used by the patched target (set per case)
_ORIG_TARGET = B.target_world


def make_ctx():
    """dive4.Ctx for the Up100 clip: the module-level SO name / hex are set only while the context is built."""
    saved = (D4.SO_NAME, D4.SO_HEX)
    D4.SO_NAME, D4.SO_HEX = SO_NAME, SO_HEX
    try:
        ctx = D4.Ctx()
    finally:
        D4.SO_NAME, D4.SO_HEX = saved
    return ctx


def raised_cosine_span(t, t0, t1):
    t = np.asarray(t, float)
    u = (t - t0) / max(t1 - t0, 1e-9)
    return np.where((u > 0) & (u < 1), 0.5 * (1.0 - np.cos(2.0 * np.pi * np.clip(u, 0, 1))), 0.0)


def target_world_bump(ctx, case, t, lift_k):
    """dive4b.target_world plus the raised-cosine bump on the Reference z (BUMP global, set per case)."""
    q, pl = _ORIG_TARGET(ctx, case, t, lift_k)
    if BUMP['A'] > 0 and BUMP['t0'] is not None:
        q[:, 2] = q[:, 2] + BUMP['A'] * raised_cosine_span(t, BUMP['t0'], BUMP['t1'])
    return q, pl


B.target_world = target_world_bump


def box_need(ctx, res):
    """Per played sample of the SlideOver: the lift (m) the non-finger body needs over the 1 m top (BOX_TOP + BOX_CLEAR)."""
    seg, fin, win = res['seg_S'], res['fin_S'], res['win_S']
    P = B.D3._world(seg['seq'], seg['dump'], fin['final'], fin['ref_out'], fin['disp'])
    pts = P[win][:, ctx.body_nf]
    y, z = pts[..., 1], pts[..., 2]
    inside = (y > float(ctx.ob['near_y'])) & (y < float(ctx.ob['far_y']))
    need = np.where(inside, BOX_TOP + BOX_CLEAR - z, 0.0).max(axis=1)
    return np.asarray(win, float) / D4.FPS, np.maximum(need, 0.0)


def bump_span_and_amplitude(ctx, case):
    """Zero-bump, zero-floor bake of the case: the box need, the span of the bump and its minimum amplitude.
    The span starts PRE_TO before T_to and ends at the plant start, or at the last sample that still needs clearance."""
    BUMP.update(t0=None, t1=None, A=0.0)
    res0 = B.bake_case5(ctx, case, np.zeros(B.LIFT_K), ease=False)
    t, need = box_need(ctx, res0)
    t0 = case.T_to - PRE_TO
    t1 = case.T_to + PLANT_LEAD
    shape = raised_cosine_span(t, t0, t1)
    # the raised cosine has a long tail of need (0.12 m at the end of the span) that would need an amplitude of ~1.3 m, so
    # the amplitude is the minimum over the middle of the span (shape >= 0.5); the ground-floor envelope covers the tails
    mid = shape >= 0.5
    A = float(np.max(need[mid] / shape[mid])) if np.any(need[mid] > 0) else 0.0
    outside = (shape <= 1e-12) & (need > 1e-4)
    tail_need = float(need[outside].max()) if np.any(outside) else 0.0     # left to the ground-floor envelope
    info = dict(t0=t0, t1=t1, A=A, box_need_max=float(need.max()), box_need_t=float(t[int(np.argmax(need))]),
                tail_need_m=tail_need)
    return A, t0, t1, info


def bake_case(ctx, case):
    """Span and amplitude from the box need, then the bump + floor (dive4c floor, ease off, WD2 tail)."""
    A, t0, t1, info = bump_span_and_amplitude(ctx, case)
    BUMP.update(t0=t0, t1=t1, A=A)
    original_need = B.need_samples
    B.need_samples = C.make_need_samples(ctx, case.T_to)
    try:
        res = B.bake_with_floor(ctx, case, ease=False, floor=True)
    finally:
        B.need_samples = original_need
    st = B.judge(res)
    # the box clearance after the bump (before the floor), for the report
    res['bump'] = dict(A=A, t0=t0, t1=t1, **{k: v for k, v in info.items() if k in ('box_need_max', 'box_need_t')})
    res['apex_hips_z'] = apex_hips(ctx, res)
    return res, st


def apex_hips(ctx, res):
    """Highest world Hips (pelvis) height in the flight, from the takeoff to the plant start (m), and its clip time."""
    P = B.D3._world(res['seg_S']['seq'], res['seg_S']['dump'], res['fin_S']['final'], res['fin_S']['ref_out'],
                    res['fin_S']['disp'])
    win = np.asarray(res['win_S'], int)
    tc = win / D4.FPS
    sel = (tc >= res['case'].T_to - 1e-9) & (tc <= res['case'].T_to + PLANT_LEAD + 1e-9)
    z = P[win[sel], ctx.hips, 2]
    k = int(np.argmax(z))
    return dict(z=float(z[k]), t=float(tc[sel][k]))


def scan_candidates(ctx):
    """dive4b.scan_rows with the 5.50 m hand rule; the candidates must also plant by W (T_to + 0.30 <= W)."""
    saved = B.HAND_MIN
    B.HAND_MIN = HAND_MIN5
    try:
        rows = B.scan_rows(ctx, T_grid=T_GRID5, s_grid=S_GRID5, shift_grid=SHIFT5)
    finally:
        B.HAND_MIN = saved
    W = float(ctx.t_W)
    n_w = 0
    for r in rows:
        ok_w = r['T_to'] + PLANT_LEAD <= W + 1e-9
        r['plant_by_W'] = bool(ok_w)
        if r['feasible'] and not ok_w:
            r['feasible'] = False
            n_w += 1
    return rows, n_w


def write_scan(rows, top, n_w):
    os.makedirs(OUT_DIR, exist_ok=True)
    keys = ['T_to', 's', 'shift', 'y_take', 'hand_L', 'hand_R', 'hand_clear', 'dW', 'cost', 'feasible', 'plant_by_W']
    with open(SCAN_CSV, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(keys)
        for r in rows:
            wr.writerow([('%.4f' % r[k]) if isinstance(r[k], float) else r[k] for k in keys])
    feas = [r for r in rows if r['feasible']]
    lines = ['# dive5 placement scan (1 m SlideOver)', '',
             'Grid: T_to %d x s %d x shift %d = %d rows. Closer hand >= %.2f m. Plant by W: %d rows dropped. Feasible: %d.'
             % (len(T_GRID5), len(S_GRID5), len(SHIFT5), len(rows), HAND_MIN5, n_w, len(feas)),
             'Cost = shift^2 + 0.5 (s-1)^2 + 0.3 dW^2 + 0.5 (T_to-0.47)^2. Shift is relative to the original Hips at T_to.', '',
             '| T_to | s | shift | y_take | closer hand | clearance | dW at W | cost | feet-down minus Exit end |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in top:
        lines.append('| %.2f | %.2f | %+.2f | %.3f | %.3f | %+.3f | %+.3f | %.4f | %+.3f |'
                     % (r['T_to'], r['s'], r['shift'], r['y_take'], min(r['hand_L'], r['hand_R']), r['hand_clear'],
                        r['dW'], r['cost'], r['feet_rel_exit_end']))
    lines.append('')
    with open(SCAN_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def run_candidates(ctx, top, verbose=True):
    results = []
    for r in top:
        case = B.Case(r['T_to'], r['s'], r['shift'], 1.0)
        res, st = bake_case(ctx, case)
        results.append(dict(res=res, st=st, row=r))
        if verbose:
            w = res['world']
            print('T %.2f s %.2f sh %+.2f | bump A %.3f span %.2f..%.2f apex %.2f | S %s E %s | pen %.1f/%.1f touch %d '
                  'pelvis step %.1f head step %.1f dv %.2f off %.2f cont %.0f floor %.2f' % (
                      r['T_to'], r['s'], r['shift'], res['bump']['A'], res['bump']['t0'], res['bump']['t1'],
                      res['apex_hips_z']['z'], st['S']['status'], st['E']['status'], res['met_S']['pen_box_cm'],
                      res['met_S']['pen_ground_cm'], res['met_S']['touch']['min'], w['pelvis']['step_cm'],
                      w['head']['step_cm'], w['dv_ms'], w['offset_max_m'], res['cont']['max_deg'],
                      res['floor_residual_cm']))
    return results


def write_outputs(ctx, res, st):
    os.makedirs(PATCH_DIR, exist_ok=True)
    DC2.write_patch(os.path.join(PATCH_DIR, SO_HEX + '.baked.json'), ctx.rig, res['fin_S']['rot_tracks'],
                    res['fin_S']['trans_keys'], None, ctx.so['dump']['Length'])
    DC2.write_patch(os.path.join(PATCH_DIR, EXIT_HEX + '.baked.json'), ctx.rig, res['fin_E']['rot_tracks'],
                    res['fin_E']['trans_keys'], None, ctx.ex_dump['Length'])
    mapping = dict(clips=[
        DC2.mapping_entry(SO_HEX, SO_NAME, SO_HEX + '.baked.json', True, override=OVERRIDE),
        DC2.mapping_entry(EXIT_HEX, EXIT_NAME, EXIT_HEX + '.baked.json', True, only_after=[SO_NAME], override=OVERRIDE),
    ])
    with open(MAPPING, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=1)
    return mapping


def write_sheet(ctx, res):
    """dive4b.write_sheet with the dive5 title and the Up100 name (same plot helpers)."""
    import bake_chain as BC
    rig = ctx.rig
    PfS = res['world']['_PS']
    PfE = B.D3._world(res['seg_E']['seq'], res['seg_E']['dump'], res['fin_E']['final'], res['fin_E']['ref_out'],
                      res['fin_E']['disp'])
    PoS = acuclip.world_positions(ctx.so['seq'], ctx.so['dump'])
    PoE = acuclip.world_positions(ctx.ex_seq, ctx.ex_dump)
    arrS = dict(Pf=PfS, Po=PoS)
    arrE = dict(Pf=PfE, Po=PoE)
    tl = B.Tl(ctx, res['case'], res['w_exit'])
    S = dict(F=res['seg_S']['F'], disp=ctx.disp_so, name=SO_NAME)
    E = dict(F=res['seg_E']['F'], disp=ctx.disp_ex, name=EXIT_NAME)
    c = res['case']
    title = ('dive5 (1 m wall): T_to %.2f, s %.2f, shift %+.2f, bump %.2f m, k_roll %.1f, continuity %.0f deg'
             % (c.T_to, c.s, c.shift, res['bump']['A'], c.k_roll, res['cont']['max_deg']))
    BC.plot_joined(SHEET_PNG, rig, S, arrS, E, arrE, ctx.ob, tl, title, show_root=True)
    BC.write_gif(SHEET_GIF, rig, S, arrS, E, arrE, ctx.ob, tl)


def write_report(ctx, res, st, rows, n_w, top, results, chosen_idx):
    mS, mE, w = res['met_S'], res['met_E'], res['world']
    c = res['case']
    lines = [
        '# T18 dive5: dive roll on the 1 m SlideOver (Fd100cm-Up100cm, 47BFB02DF)', '',
        'Generated by `v2/pipeline/dive5.py` (reuses `dive4b.py` and `dive4c.py`). Written only under `out\\dive5`.', '',
        'Case: T_to %.2f s, s %.2f, shift %+.2f m (relative to the original Hips at T_to), k_roll 1.0, no ease, WD2 tail, '
        'the floor leaves the hands out during the plant (dive4c). Bump: amplitude %.3f m, span %.2f .. %.2f s.'
        % (c.T_to, c.s, c.shift, res['bump']['A'], res['bump']['t0'], res['bump']['t1']), '',
        '## Enable state (coordinator preview)', '',
        'Both clips are written **enabled: true** with the override "%s", as requested. Measured spec status: SlideOver '
        '**%s**, Exit **%s**.' % (OVERRIDE, st['S']['status'], st['E']['status']), '',
        'Spec fails: %s' % ('; '.join(st['S']['fails']) or 'none'), '',
        '## 1. Scan', '',
        'Closer hand >= %.2f m at the plant; the plant must start by W = %.3f s. %d rows dropped by the W rule.'
        % (HAND_MIN5, ctx.t_W, n_w), '',
        '| T_to | s | shift | closer hand | dW | cost | feet-down minus Exit end |', '|---|---|---|---|---|---|---|']
    for r in top:
        lines.append('| %.2f | %.2f | %+.2f | %.3f | %+.3f | %.4f | %+.3f |'
                     % (r['T_to'], r['s'], r['shift'], min(r['hand_L'], r['hand_R']), r['dW'], r['cost'],
                        r['feet_rel_exit_end']))
    lines += ['', '## 2. Candidates baked (floor and bump on; the chosen one is %s)' % ('row %d' % chosen_idx), '',
              '| T_to | s | shift | bump A (m) | span (s) | apex hips z (m) | box pen / ground pen (cm) | touch | pelvis step '
              '(cm/f) | head step (cm/f) | dv (m/s/f) | offset (m) | cont (deg) | floor (cm) | status S / E |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for i, r in enumerate(results):
        rr, s2 = r['res'], r['st']
        ww = rr['world']
        lines.append('| %s%.2f | %.2f | %+.2f | %.3f | %.2f..%.2f | %.2f | %.1f / %.1f | %d | %.1f | %.1f | %.2f | %.2f | %.0f | '
                     '%.2f | %s / %s |'
                     % ('*' if i == chosen_idx else '', rr['case'].T_to, rr['case'].s, rr['case'].shift, rr['bump']['A'],
                        rr['bump']['t0'], rr['bump']['t1'], rr['apex_hips_z']['z'], rr['met_S']['pen_box_cm'],
                        rr['met_S']['pen_ground_cm'], rr['met_S']['touch']['min'], ww['pelvis']['step_cm'],
                        ww['head']['step_cm'], ww['dv_ms'], ww['offset_max_m'], rr['cont']['max_deg'],
                        rr['floor_residual_cm'], s2['S']['status'], s2['E']['status']))
    lines += ['', '## 3. Validation (chosen)', '', '| metric | SlideOver | Exit | rule |', '|---|---|---|---|',
              '| non-finger penetration, box / ground (cm) | %.1f / %.1f | %.1f / %.1f | <= 5 |'
              % (mS['pen_box_cm'], mS['pen_ground_cm'], mE['pen_box_cm'], mE['pen_ground_cm']),
              '| hand touch (frames, min) | %d | n/a | >= 4 |' % mS['touch']['min'],
              '| world pelvis: step cm/frame / dv m/s per frame / accel cm/frame^2 | %.2f / %.2f / %.2f | joined | <= 12 / 1.5 |'
              % (w['pelvis']['step_cm'], w['pelvis']['dv_ms'], w['pelvis']['acc_cm']),
              '| world head: step cm/frame / dv m/s per frame / accel cm/frame^2 | %.2f / %.2f / %.2f | joined | <= 12 / 1.5 |'
              % (w['head']['step_cm'], w['head']['dv_ms'], w['head']['acc_cm']),
              '| Reference offset max (m) | %.3f | | <= 1.0 |' % w['offset_max_m'],
              '| continuity at W (deg) | %.1f (%s) | | <= 45 |' % (res['cont']['max_deg'], res['cont']['worst']),
              '| apex of the world Hips in the flight (m, clip time) | %.2f at %.2f s | | info |'
              % (res['apex_hips_z']['z'], res['apex_hips_z']['t']),
              '| floor residual / lift max (cm) | %.2f / %.1f | | info |' % (res['floor_residual_cm'], res['lift_max_cm']),
              '| pre-ease gap to Nav_Run (deg) | | %.1f | no ease |' % res['gap_pre']['max_deg'], '']
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ctx = make_ctx()
    rows, n_w = scan_candidates(ctx)
    top = B.top_choices(rows, 5)
    # W-phase candidates: the cheapest feasible row at each later takeoff (W then sits nearer the flight apex, where the
    # WD2 pelvis is not falling as fast as at the plant end; see report section 2)
    for T in PHASE_T:
        best = B.top_choices([r for r in rows if abs(r['T_to'] - T) < 1e-9], 1)
        if best and all(abs(best[0]['T_to'] - x['T_to']) > 1e-9 or abs(best[0]['s'] - x['s']) > 1e-9 or abs(best[0]['shift'] - x['shift']) > 1e-9 for x in top):
            top.append(best[0])
    if '--scan' in argv:
        for r in top:
            print(r['T_to'], r['s'], r['shift'], round(r['cost'], 4), round(r['dW'], 3), round(r['hand_clear'], 3))
        print('plant-by-W dropped', n_w, 'feasible', sum(1 for r in rows if r['feasible']))
        return 0
    write_scan(rows, top, n_w)
    results = run_candidates(ctx, top)
    chosen_idx = min(range(len(results)), key=lambda i: (B.rank_key(results[i]['st']), results[i]['res']['case'].tm.t_feet))
    chosen = results[chosen_idx]
    res, st = chosen['res'], chosen['st']
    os.makedirs(OUT_DIR, exist_ok=True)
    write_outputs(ctx, res, st)
    write_sheet(ctx, res)
    write_report(ctx, res, st, rows, n_w, top, results, chosen_idx)
    summary = dict(chosen=dict(T_to=res['case'].T_to, s=res['case'].s, shift=res['case'].shift, bump=res['bump'],
                               apex=res['apex_hips_z']),
                   status=st, world=dict((k, v) for k, v in res['world'].items() if not k.startswith('_')),
                   touch=res['met_S']['touch'], cont=res['cont']['max_deg'],
                   variants=[dict(T_to=r['res']['case'].T_to, s=r['res']['case'].s, shift=r['res']['case'].shift,
                                  bump=r['res']['bump'], S=r['st']['S']['status'], E=r['st']['E']['status'])
                             for r in results])
    with open(SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps(dict(T_to=res['case'].T_to, s=res['case'].s, shift=res['case'].shift, S=st['S']['status'],
                          E=st['E']['status'], touch=res['met_S']['touch']['min'])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
