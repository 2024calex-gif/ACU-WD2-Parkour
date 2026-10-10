"""Retarget a Watch Dogs 2 .mab onto the AC Unity (Arno) skeleton, rotations only.

Method (per mapped bone b, per sample time t):
    Ds   = Wsrc_anim(t) * Wsrc_bind^-1         world-space change of the source bone
    Dt   = M * Ds * M^-1                       same change, expressed in ACU world axes
    Rb   = arc(Wtgt_bind.x -> M * Wsrc_bind.x) constant fix for rest-pose differences (A-pose angle)
    Wtgt = Dt * Rb * Wtgt_bind
    local = Wparent_tgt^-1 * Wtgt
WD2 world: X right, Y forward, Z up. ACU world: X forward, Y left, Z up.  M = (x,y,z) -> (y,-x,z).
ACU 'Reference' is kept at identity (the game owns root motion); Hips and Spine are both its children.
"""
import json, math, sys
import numpy as np
import mab, skel

# ---- quaternion helpers (x, y, z, w) -------------------------------------------------------------
def qmul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw, aw*bw - ax*bx - ay*by - az*bz])
def qconj(q): return np.array([-q[0], -q[1], -q[2], q[3]])
def qn(q):
    q = np.array(q, float); return q / np.linalg.norm(q)
def qrot(q, v):
    return qmul(qmul(q, np.array([v[0], v[1], v[2], 0.0])), qconj(q))[:3]
def qarc(a, b):
    """shortest rotation taking unit vector a to unit vector b"""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    d = float(a @ b)
    if d > 0.999999: return np.array([0, 0, 0, 1.0])
    if d < -0.999999:
        ax = np.cross(a, [1, 0, 0]);
        if np.linalg.norm(ax) < 1e-6: ax = np.cross(a, [0, 1, 0])
        ax /= np.linalg.norm(ax); return np.array([ax[0], ax[1], ax[2], 0.0])
    c = np.cross(a, b); return qn([c[0], c[1], c[2], 1 + d])
def slerp(a, b, t):
    a = qn(a); b = qn(b); d = float(a @ b)
    if d < 0: b = -b; d = -d
    if d > 0.9995: return qn(a + (b - a) * t)
    th = math.acos(d); s = math.sin(th)
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / s

A_REF = qn([0, 0, math.sqrt(0.5), math.sqrt(0.5)])  # Reference carries this base turn in the game's clips (measured)
M = qn([0, 0, -math.sqrt(0.5), math.sqrt(0.5)])  # -90 deg about Z : WD2 world -> ACU world
Minv = qconj(M)

# ---- bone name mapping ---------------------------------------------------------------------------
def build_map():
    m = {'Pelvis': 'Hips', 'Spine': 'Spine', 'Spine1': 'Spine1', 'Spine2': 'Spine2', 'Neck': 'Neck', 'Head': 'Head'}
    for s, S in (('L', 'Left'), ('R', 'Right')):
        m[s + ' Thigh'] = S + 'UpLeg'; m[s + ' Calf'] = S + 'Leg'; m[s + ' Foot'] = S + 'Foot'; m[s + ' Toe'] = S + 'FootThumb1'
        m[s + ' Clavicle'] = S + 'Clavicle'; m[s + ' UpperArm'] = S + 'Arm'; m[s + ' Forearm'] = S + 'ForeArm'; m[s + ' Hand'] = S + 'Hand'
        for n in (1, 2, 3):
            m['%s Index0%d' % (s, n)] = '%sHandIndex%d' % (S, n)
            m['%s Middle0%d' % (s, n)] = '%sHandMiddle%d' % (S, n)
            m['%s Ring0%d' % (s, n)] = '%sHandRing%d' % (S, n)
            m['%s Pinky0%d' % (s, n)] = '%sHandPinky%d' % (S, n)
            m['%s Thumb0%d' % (s, n)] = '%sHandThumb%d' % (S, n)
        m[s + ' Middle_Meta'] = S + 'HandMiddle0'; m[s + ' Ring_Meta'] = S + 'HandRing0'; m[s + ' Hand_Meta'] = S + 'HandPinky0'
    return m


class Rig:
    def __init__(self):
        self.wd = skel.load_wd2()
        self.ac = skel.load_acu()
        # fix the root record (different layout): identity at origin
        self.ac[0]['lrot'] = (0, 0, 0, 1.0); self.ac[0]['lpos'] = (0, 0, 0)
        self.ac_by_hash = {b['hash']: i for i, b in enumerate(self.ac)}
        self.wd_by_name = {b['name']: i for i, b in enumerate(self.wd)}
        self.map = {}  # wd index -> ac index
        for wn, an in build_map().items():
            wi = self.wd_by_name.get(wn)
            ai = self.ac_by_hash.get(skel.zlib.crc32(an.encode()))
            if wi is not None and ai is not None:
                self.map[wi] = ai
        self.Wb_wd = self.fk([qn(b['lrot']) for b in self.wd], self.wd)
        self.Wb_ac = self.fk([qn(b['lrot']) for b in self.ac], self.ac)
        # constant rest-pose correction per mapped bone
        self.Rb = {}
        for wi, ai in self.map.items():
            src_dir = qrot(M, qrot(self.Wb_wd[wi], [1, 0, 0]))
            tgt_dir = qrot(self.Wb_ac[ai], [1, 0, 0])
            self.Rb[ai] = qarc(tgt_dir, src_dir)

    @staticmethod
    def fk(locals_, bones):
        W = [None] * len(bones)
        for i, b in enumerate(bones):
            W[i] = locals_[i] if b['parent'] < 0 else qn(qmul(W[b['parent']], locals_[i]))
        return W

    def source_locals(self, m, tick):
        """local rotations of every WD2 bone at a given 30Hz tick (animated, const, else bind)."""
        by_hash = {b['hash']: i for i, b in enumerate(self.wd)}
        L = [qn(b['lrot']) for b in self.wd]
        for bi, h in enumerate(m['hashes']):
            wi = by_hash.get(h)
            if wi is None: continue
            if bi in m['rot']:
                keys = m['rot'][bi]
                if tick <= keys[0][0]: q = keys[0][1]
                elif tick >= keys[-1][0]: q = keys[-1][1]
                else:
                    for (t0, q0), (t1, q1) in zip(keys, keys[1:]):
                        if t0 <= tick <= t1:
                            q = slerp(q0, q1, (tick - t0) / max(1, t1 - t0)); break
                L[wi] = qn(q)
            elif bi in m['const']:
                L[wi] = qn(m['const'][bi])
        return L

    def target_locals(self, src_locals):
        Ws = self.fk(src_locals, self.wd)
        Wt = [None] * len(self.ac)
        L = [None] * len(self.ac)
        inv = {ai: wi for wi, ai in self.map.items()}
        for ai, b in enumerate(self.ac):
            par = b['parent']
            if ai in inv:
                wi = inv[ai]
                Ds = qmul(Ws[wi], qconj(self.Wb_wd[wi]))
                Dt = qmul(qmul(M, Ds), Minv)
                W = qn(qmul(qmul(Dt, self.Rb[ai]), self.Wb_ac[ai]))
            else:
                bind_l = qn(b['lrot'])
                W = bind_l if par < 0 else qn(qmul(Wt[par], bind_l))
            Wt[ai] = W
            L[ai] = W if par < 0 else qn(qmul(qconj(Wt[par]), W))
        # In ACU the pelvis is carried by the root bone 'Reference' (Hips has no track in the game's clips, it stays at
        # its bind local rotation). So rebase: Reference := pelvis delta, Hips := bind local, Spine etc. relative to it.
        hips = self.ac_by_hash[skel.zlib.crc32(b'Hips')]
        Rp = qn(qmul(Wt[hips], qconj(qn(self.ac[hips]['lrot']))))
        Wt[0] = Rp; L[0] = Rp
        for ai, b in enumerate(self.ac):
            if b['parent'] == 0:
                L[ai] = qn(qmul(qconj(Rp), Wt[ai]))
        return L, Wt


def retarget_clip(path, rig=None, drop_tol_deg=0.15, ground_offset=0.03):
    rig = rig or Rig()
    m = mab.parse(path)
    ticks = sorted(set(m['key_times']) | {0, m['last_key']})
    samples = []
    heights = []
    body = sorted(set(rig.map.values()))
    # In-game calibration: with a zero offset the bind pose stands with its feet on the ground, so heights are
    # relative to the bind pose: raise by (bind lowest point) - (current lowest point).
    Pb = [None] * len(rig.ac); Wb = [None] * len(rig.ac)
    for i, b in enumerate(rig.ac):
        lr = qn(b['lrot'])
        if b['parent'] < 0: Wb[i] = lr; Pb[i] = np.zeros(3)
        else:
            Wb[i] = qn(qmul(Wb[b['parent']], lr)); Pb[i] = Pb[b['parent']] + qrot(Wb[b['parent']], np.array(b['lpos'], float))
    bind_min = min(Pb[i][2] for i in body)
    for t in ticks:
        L, _ = rig.target_locals(rig.source_locals(m, t))
        samples.append((t, L))
        # forward kinematics for positions; the lowest mapped body point is assumed to rest on the ground
        P = [None] * len(rig.ac); W = [None] * len(rig.ac)
        for i, b in enumerate(rig.ac):
            if b['parent'] < 0: W[i] = L[i]; P[i] = np.zeros(3)
            else:
                W[i] = qn(qmul(W[b['parent']], L[i])); P[i] = P[b['parent']] + qrot(W[b['parent']], np.array(b['lpos'], float))
        heights.append((t, -min(P[i][2] for i in body) + bind_min + ground_offset))
    # per ACU bone track: only mapped bones (+ their hierarchy locals change only if mapped parents changed => all descendants)
    out = {}
    mapped = set(rig.map.values()) | {0}
    changed = set(mapped)
    # unmapped bones below mapped parents inherit rigidly: local stays bind => no track needed
    for ai in sorted(changed):
        keys = [(t, (qn(qmul(A_REF, L[ai])) if ai == 0 else L[ai])) for t, L in samples]
        # make quats continuous
        fixed = [keys[0]]
        for t, q in keys[1:]:
            pq = fixed[-1][1]
            fixed.append((t, q if float(pq @ q) >= 0 else -q))
        # drop keys that lerp predicts within tolerance
        keep = [fixed[0]]
        for i in range(1, len(fixed) - 1):
            t0, q0 = keep[-1]; t1, q1 = fixed[i + 1]; t, q = fixed[i]
            pred = slerp(q0, q1, (t - t0) / max(1, t1 - t0))
            ang = math.degrees(2 * math.acos(min(1.0, abs(float(pred @ q)))))
            if ang > drop_tol_deg: keep.append(fixed[i])
        keep.append(fixed[-1])
        out[ai] = keep
    m['heights'] = heights
    return m, out


def to_json(rig, m, tracks, hips_height=None, bind_fill=True):
    """ACU .anim.json dict. Frames are 60 Hz (WD2 ticks are 30 Hz). bind_fill: add bind translation tracks."""
    tr = []
    for ai, b in enumerate(rig.ac):
        if ai == 0 and 0 not in tracks: continue
        h = b['hash']
        pos = list(b['lpos'])
        if ai == 0: pos = None
        if bind_fill and ai != 0:
            if b['parent'] == 0 and m.get('heights'):
                # Hips and Spine are siblings under Reference: lift every child of the root by the same
                # (time-varying) amount, equivalent to moving Reference without touching its track.
                kfs_t = [{'Frame': int(t) * 2, 'LocalPosition': [round(float(pos[0]), 6), round(float(pos[1]), 6), round(float(pos[2] + hh), 6)]}
                         for t, hh in m['heights']]
                tr.append((h, 0, {'TrackID': str(h), 'codec': 27, 'infotype': 3, 'Keyframes': kfs_t}))
                pos = None
            elif hips_height is not None and b['parent'] == 0:
                pos[2] += hips_height
        if bind_fill and pos is not None:
            tr.append((h, 0, {'TrackID': str(h), 'codec': 27, 'infotype': 3,
                              'Keyframes': [{'Frame': 0, 'LocalPosition': [round(float(x), 6) for x in pos]}]}))
        if ai in tracks:
            kfs = [{'Frame': int(t) * 2, 'LocalRotation': [round(float(x), 6) for x in q]} for t, q in tracks[ai]]
        elif bind_fill:
            kfs = [{'Frame': 0, 'LocalRotation': [round(float(x), 6) for x in qn(b['lrot'])]}]
        else:
            continue
        tr.append((h, 1, {'TrackID': str(h), 'codec': 1, 'infotype': 0, 'Keyframes': kfs}))
    tr.sort(key=lambda x: (x[0], x[1]))
    return {'Length': round(float(m['dur']), 6), 'Tracks': [t[2] for t in tr]}


if __name__ == '__main__':
    src = sys.argv[1]; dst = sys.argv[2]
    hh = float(sys.argv[3]) if len(sys.argv) > 3 else None
    rig = Rig()
    print('mapped bones: %d of %d WD2' % (len(rig.map), len(rig.wd)))
    m, tracks = retarget_clip(src, rig)
    j = to_json(rig, m, tracks, hips_height=hh)
    json.dump(j, open(dst, 'w'), indent=1)
    print('wrote', dst, 'length', j['Length'], 'tracks', len(j['Tracks']), 'rot keys', sum(len(v) for v in tracks.values()))
