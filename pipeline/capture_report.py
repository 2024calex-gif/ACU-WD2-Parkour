"""Print a summary of flight-recorder captures and write one PNG per capture.

usage: python capture_report.py <capture.json or folder> [--out-dir DIR]

PNG (default v2\\out\\captures\\<capture name>.png): top view of the entity path with the obstacle edges, side view
(distance along the path vs z, with hand/foot traces and the obstacle), and the clip timeline with patched clips
highlighted.
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402

import capture  # noqa: E402
import config  # noqa: E402
import dumps  # noqa: E402

DEFAULT_OUT = os.path.join(config.OUT, 'captures')
TRACE_BONES = ('LeftHand', 'RightHand', 'LeftFoot', 'RightFoot')


def _fmt_t(x):
    return '%.3f s' % x if x is not None else 'none'


def _has_pose(cap):
    """True when the capture has entity-space bones for at least one trace bone (partial poses count)."""
    return cap.bones_space == 'entity' and any(cap.bones.get(n) is not None for n in TRACE_BONES)


def _path_axis(cap):
    d = cap.entity_pos[-1, 0:2] - cap.entity_pos[0, 0:2]
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.array([0.0, 1.0])


def _along(points_xy, origin_xy, axis):
    return (points_xy - origin_xy) @ axis


def print_report(cap, obstacle, ev, conv):
    dts = np.diff(cap.t)
    med_dt = float(np.median(dts)) if len(dts) else float('nan')
    fps = 1.0 / med_dt if med_dt > 0 else float('nan')
    print('== %s' % os.path.basename(cap.path))
    print('  trigger        %s' % cap.trigger)
    print('  patching       %s' % ('ON' if cap.patching_enabled else 'off'))
    print('  duration       %.3f s  (%d frames)' % (cap.t[-1] - cap.t[0], len(cap.t)))
    print('  frame rate     %.2f fps  (median dt %.2f ms)' % (fps, med_dt * 1000))
    print('  pose_ok        %s  (bones_space %s)' % (cap.pose_ok, cap.bones_space))
    if cap.pose_bones_missing:
        print('  bones missing  %s (in at least one frame)' % ', '.join(cap.pose_bones_missing))
    print('  variant        %s' % (cap.variant or '(default)'))
    rows_deg = conv['rows_deg'] if conv['rows_deg'] is not None else float('nan')
    cols_deg = conv['cols_deg'] if conv['cols_deg'] is not None else float('nan')
    print('  convention     forward=+Y as rows: %.1f deg, as columns: %.1f deg (%d moving frames) -> %s'
          % (rows_deg, cols_deg, conv['n_frames'], conv['verdict']))
    if conv['verdict'] in ('rows', 'columns') and not conv['matches_constant']:
        print('  WARNING        data fits %r but capture.ROW_MAJOR_BASIS is %s; world positions above use the constant'
              % (conv['verdict'], capture.ROW_MAJOR_BASIS))
    print('  clip timeline:')
    for c in cap.clips:
        mark = '  [PATCHED]' if c['patched'] else ''
        print('    %8.3f s  %6.3f s  %s%s' % (c['t'], c['length'] if c.get('length') is not None else float('nan'),
                                              c['name'] or '<unknown>', mark))
    if obstacle is not None:
        print('  obstacle       height %.3f m, depth %.3f m, near (%.3f, %.3f), far (%.3f, %.3f), top z %.3f'
              % (obstacle.height, obstacle.depth, obstacle.near[0], obstacle.near[1],
                 obstacle.far[0], obstacle.far[1], obstacle.top_z))
    else:
        print('  obstacle       not available (no trigger dump)')
    print('  events:')
    if not ev['pose']:
        print('    (no pose) trigger start %s' % _fmt_t(ev['trigger_start']))
    else:
        print('    hand_plant  %s' % _fmt_t(ev['hand_plant']))
        print('    takeoff     %s' % _fmt_t(ev['takeoff']))
        print('    landing     %s' % _fmt_t(ev['landing']))
    return fps


def plot(cap, obstacle, ev, out_path):
    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_side = fig.add_subplot(gs[0, 1])
    ax_tl = fig.add_subplot(gs[1, :])
    fig.suptitle('%s  (trigger %s, patching %s)' % (os.path.basename(cap.path), cap.trigger,
                                                     'ON' if cap.patching_enabled else 'off'))

    # 1. top view
    xy = cap.entity_pos[:, 0:2]
    ax_top.plot(xy[:, 0], xy[:, 1], '-', color='tab:blue', label='entity path')
    ax_top.plot(xy[0, 0], xy[0, 1], 'o', color='tab:green', label='start')
    ax_top.plot(xy[-1, 0], xy[-1, 1], 's', color='tab:blue', label='end')
    if obstacle is not None:
        ax_top.plot([obstacle.near[0], obstacle.far[0]], [obstacle.near[1], obstacle.far[1]], '-', lw=4,
                    color='tab:red', label='obstacle edges')
        ax_top.annotate('near', obstacle.near[0:2], fontsize=8, color='tab:red')
        ax_top.annotate('far', obstacle.far[0:2], fontsize=8, color='tab:red')
    ax_top.set_aspect('equal', adjustable='datalim')
    ax_top.set_xlabel('x (m)')
    ax_top.set_ylabel('y (m)')
    ax_top.set_title('Top view')
    ax_top.grid(alpha=0.3)
    ax_top.legend(fontsize=8, loc='best')

    # 2. side view: distance along path vs z
    axis = _path_axis(cap)
    origin = xy[0]
    ax_side.plot(_along(xy, origin, axis), cap.entity_pos[:, 2], '-', color='tab:blue', label='entity origin z')
    has_pose = _has_pose(cap)
    if has_pose:
        for name in TRACE_BONES:
            w = capture.bone_world(cap, name)
            if w is None:
                continue
            ax_side.plot(_along(w[:, 0:2], origin, axis), w[:, 2], '-', lw=1, alpha=0.8, label=name)
    if obstacle is not None:
        d_n = _along(obstacle.near[0:2], origin, axis)
        d_f = _along(obstacle.far[0:2], origin, axis)
        lo, hi = min(d_n, d_f), max(d_n, d_f)
        ax_side.add_patch(Rectangle((lo, obstacle.ground_z), hi - lo, obstacle.top_z - obstacle.ground_z,
                                    facecolor='tab:red', alpha=0.25, edgecolor='tab:red', label='obstacle'))
    ax_side.set_xlabel('distance along path (m)')
    ax_side.set_ylabel('z (m)')
    ax_side.set_title('Side view' + ('' if has_pose else ' (no pose: entity only)'))
    ax_side.grid(alpha=0.3)
    ax_side.legend(fontsize=7, loc='best')

    # 3. clip timeline
    clips = cap.clips
    for i, c in enumerate(clips):
        length = c.get('length')
        if length is None:
            length = (clips[i + 1]['t'] if i + 1 < len(clips) else cap.t[-1]) - c['t']
        if c['patched']:
            ax_tl.barh(i, length, left=c['t'], color='tab:orange', hatch='//', edgecolor='black', alpha=0.9)
        else:
            color = 'tab:purple' if c['name'] == cap.trigger else 'tab:gray'
            ax_tl.barh(i, length, left=c['t'], color=color, alpha=0.6)
        ax_tl.text(c['t'] + 0.02, i, '%s%s' % (c['name'] or '<unknown>', '  [PATCHED]' if c['patched'] else ''),
                   va='center', fontsize=7)
    for key, color in (('hand_plant', 'tab:red'), ('takeoff', 'tab:green'), ('landing', 'tab:blue')):
        if ev.get(key) is not None:
            ax_tl.axvline(ev[key], color=color, ls='--', lw=1)
            ax_tl.text(ev[key], len(clips) - 0.4, key, color=color, fontsize=7, rotation=90, va='top')
    ax_tl.set_ylim(len(clips) - 0.5, -0.8 if clips else 0.5)
    ax_tl.set_yticks([])
    ax_tl.set_xlabel('time (s)')
    ax_tl.set_title('Clip timeline (orange hatched = patched, purple = trigger)')
    ax_tl.grid(alpha=0.3, axis='x')

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def process(path, out_dir):
    cap = capture.load(path)
    conv = capture.check_convention(cap)
    obstacle = capture.obstacle_from_clip(cap, dumps)
    ev = capture.events(cap, obstacle)
    name = os.path.splitext(os.path.basename(path))[0]
    print_report(cap, obstacle, ev, conv)
    out_path = os.path.join(out_dir, name + '.png')
    plot(cap, obstacle, ev, out_path)
    print('  png            %s' % out_path)
    print()
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', help='a capture .json or a folder of them')
    ap.add_argument('--out-dir', default=DEFAULT_OUT, help='where the PNGs go (default %(default)s)')
    args = ap.parse_args(argv)
    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, '*.json')))
        if not files:
            print('no .json captures in %s' % args.path, file=sys.stderr)
            return 1
    else:
        files = [args.path]
    for f in files:
        process(f, args.out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
