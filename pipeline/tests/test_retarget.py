"""WD2 -> ACU retargeting (retarget.py): bind-pose axes, and identical output to the v1 tools copy."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

import config
import retarget
import wd2

KONG = 'b00-000-00_vaultover-kong_mplyr_000f_norm_nowep'
V1_TOOLS = os.path.join(config.ROOT, 'tools')

# Runs retarget_clip in a fresh interpreter so the two copies never share module caches.
# argv: [1] folder holding mab.py/skel.py/retarget.py, [2] .mab path, [3] output json path
_RUN = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import retarget
rig = retarget.Rig()
m, tracks = retarget.retarget_clip(sys.argv[2], rig)
out = {
    'mapped': len(rig.map),
    'tracks': {str(ai): [[int(t), [float(x) for x in q]] for t, q in keys] for ai, keys in tracks.items()},
    'heights': [[int(t), float(h)] for t, h in m['heights']],
}
with open(sys.argv[3], 'w') as f:
    json.dump(out, f)
'''


def _run_retarget(folder, mab_path, out_path, cwd):
    # -B: do not write .pyc files into the v1 tools folder
    subprocess.run([sys.executable, '-I', '-B', '-c', _RUN, folder, mab_path, out_path],
                   check=True, cwd=cwd, capture_output=True, text=True)
    with open(out_path) as f:
        return json.load(f)


def test_rig_maps_52_bones():
    rig = retarget.Rig()
    print('\nmapped bones: %d of %d WD2' % (len(rig.map), len(rig.wd)))
    assert len(rig.map) == 52


def test_bind_pose_local_x_axes_match_wd2_after_retarget():
    """At the WD2 bind pose, every mapped ACU bone points its local X axis where the WD2 bone does (in ACU world)."""
    rig = retarget.Rig()
    src_locals = [retarget.qn(b['lrot']) for b in rig.wd]
    _L, Wt = rig.target_locals(src_locals)
    worst = 0.0
    for wi, ai in rig.map.items():
        expected = retarget.qrot(retarget.M, retarget.qrot(rig.Wb_wd[wi], [1.0, 0.0, 0.0]))
        got = retarget.qrot(Wt[ai], [1.0, 0.0, 0.0])
        cosang = float(np.clip(np.dot(expected, got) / (np.linalg.norm(expected) * np.linalg.norm(got)), -1, 1))
        ang = math.degrees(math.acos(cosang))
        worst = max(worst, ang)
        assert ang < 1.0, 'WD2 bone %s (index %d) -> ACU index %d off by %.3f deg' % (
            rig.wd[wi]['name'], wi, ai, ang)
    print('\nworst bind-pose axis error: %.6f deg' % worst)


def test_retarget_clip_matches_v1_tools(tmp_path):
    """Same tracks (bone set, key times, quaternions within 1e-6) as the original tools/retarget.py."""
    mab_path = wd2.mab_index()[KONG]
    v2_out = tmp_path / 'v2.json'
    v1_out = tmp_path / 'v1.json'
    new = _run_retarget(config.PIPELINE, mab_path, str(v2_out), cwd=str(tmp_path))
    old = _run_retarget(V1_TOOLS, mab_path, str(v1_out), cwd=str(tmp_path))

    assert new['mapped'] == old['mapped'] == 52
    assert set(new['tracks']) == set(old['tracks']), 'bone sets differ'
    assert new['tracks'], 'no tracks produced'
    for ai in sorted(new['tracks'], key=int):
        a, b = new['tracks'][ai], old['tracks'][ai]
        assert [k[0] for k in a] == [k[0] for k in b], 'key times differ for bone %s' % ai
        for (_ta, qa), (_tb, qb) in zip(a, b):
            assert np.max(np.abs(np.array(qa) - np.array(qb))) <= 1e-6, 'quaternion differs for bone %s' % ai

    assert [h[0] for h in new['heights']] == [h[0] for h in old['heights']]
    assert np.max(np.abs(np.array([h[1] for h in new['heights']]) - np.array([h[1] for h in old['heights']]))) <= 1e-6
