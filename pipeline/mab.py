"""WD2 .mab (Disrupt engine, magic 0x46AF) rotation decoder.

Own implementation; layout derived by inspecting WD2 clips and cross-checking against the
publicly described WD1 variant (same section scheme, shifted header).

Header (WD2):
  0x00 u16 magic 0x46AF      0x20 'aNi'     0x24 u32 size-19   0x28 f32 duration  0x2C f32 fps
  0x30 u16 nbones(&0x7FFF)   0x32 u16[9] counts   0x44 u32[11] section offsets (relative to +0x10)
  0x70 u16 last key index    0x78 u32[nbones] bone-name hashes, then u8[nbones] flags
Flags per bone: (f & 0x30)==0x30 -> constant rotation; f & 0x10 -> animated; (f & 0x0F)+1 = bits/sample.
Sections: 0 key times (u16, in ticks), 4 const rotations (3*u16 each), 6 chunked bitstream (8 keys/chunk).
"""
import math
import struct

_INTERP_SCALE = [0, 0, 0.33333334, 0.14285715, 0.06666667, 0.032258064, 0.015873017,
                 0.0078740157, 0.0039215689, 0.0019569471, 0.00097751711, 0.00048851978,
                 0.00024420026, 0.00012208521, 0.000061038882, 0.000030518509, 0.000015259022]


class _Bits:
    def __init__(self, d, start, end):
        self.d, self.n, self.bp = d, end, start * 8

    def read(self, nbits):
        bo, bi = self.bp >> 3, self.bp & 7
        num = 0
        for k in range(4):
            if bo + k < self.n:
                num |= self.d[bo + k] << (8 * k)
        self.bp += nbits
        return (num >> bi) & ((1 << nbits) - 1)


def _const_quat(w0, w1, w2):
    S, OFF = 4.315969e-05, 0.7071068
    f1 = (w0 & 0x7FFF) * S - OFF
    f2 = (w1 & 0x7FFF) * S - OFF
    f3 = w2 * S - OFF
    s = 1.0 - f1 * f1 - f2 * f2 - f3 * f3
    f4 = s ** 0.5 if s > 0 else 0.0
    b0, b1 = w0 & 0x8000, w1 & 0x8000
    if not b0:
        return (f1, f2, f4, f3) if b1 else (f4, f1, f2, f3)
    return (f1, f2, f3, f4) if b1 else (f1, f4, f2, f3)


def _bone_block(r, nbits, nframes):
    cflags = r.read(6)
    bsig = (cflags >> 3) & 1
    wind = (cflags >> 4) & 3
    sign_bits = r.read(nframes) if bsig else 0
    comp = [[0.0] * nframes for _ in range(3)]
    for c in range(3):
        if (cflags >> c) & 1:
            v = r.read(16) / 32768.0 - 1.0
            comp[c] = [v] * nframes
        else:
            wv = r.read(16)
            base = (wv & 0xFF) / 127.0 - 1.0
            slope = (wv >> 8) / 127.5 * _INTERP_SCALE[nbits]
            for f in range(nframes):
                comp[c][f] = r.read(nbits) * slope + base
    out = []
    for f in range(nframes):
        q = [0.0] * 4
        s = 0.0
        for c in range(3):
            t = c if c < wind else c + 1
            q[t] = comp[c][f]
            s += q[t] * q[t]
        imp = (1.0 - s) ** 0.5 if s < 1.0 else 0.0
        if (sign_bits >> f) & 1:
            imp = -imp
        q[wind] = imp
        out.append(tuple(q))  # x, y, z, w
    return out


def parse(path):
    d = open(path, 'rb').read()
    magic, = struct.unpack_from('<H', d, 0)
    if d[0x20:0x23] != b'aNi':
        raise ValueError('no aNi marker')
    size, dur, fps = struct.unpack_from('<Iff', d, 0x24)
    nb = struct.unpack_from('<H', d, 0x30)[0] & 0x7FFF
    counts = struct.unpack_from('<9H', d, 0x32)
    offs = struct.unpack_from('<11I', d, 0x44)
    last_key, = struct.unpack_from('<H', d, 0x70)
    p = 0x78
    hashes = struct.unpack_from('<%dI' % nb, d, p)
    flags = d[p + 4 * nb:p + 5 * nb]
    # key times
    n_keys = counts[1]
    kt = struct.unpack_from('<%dH' % n_keys, d, offs[0] + 0x10)
    # const rotations
    const = {}
    sec4 = offs[4] + 0x10
    ci = 0
    for bi in range(nb):
        if (flags[bi] & 0x30) == 0x30:
            const[bi] = _const_quat(*struct.unpack_from('<3H', d, sec4 + ci * 6))
            ci += 1
    # animated rotations
    rot = {}
    jr = offs[6] + 0x10
    table_size, = struct.unpack_from('<I', d, jr)
    nch = table_size // 4 - 1
    ends = struct.unpack_from('<%dI' % nch, d, jr + 4)
    bones = [(bi, (flags[bi] & 0x0F) + 1) for bi in range(nb)
             if (flags[bi] & 0x10) and (flags[bi] & 0x30) != 0x30]
    prev = table_size
    for ci_ in range(nch):
        r = _Bits(d, jr + prev, jr + ends[ci_])
        first = ci_ * 8
        nf = min(8, n_keys - first)
        if nf <= 0:
            break
        for bi, nbits in bones:
            qs = _bone_block(r, nbits, nf)
            cur = rot.setdefault(bi, [])
            for f in range(nf):
                cur.append((kt[first + f], qs[f]))
        prev = ends[ci_]
    return dict(raw=d, dur=dur, fps=fps, nb=nb, counts=counts, offs=offs, hashes=hashes,
                flags=bytes(flags), key_times=kt, const=const, rot=rot, last_key=last_key)


def qdot(a, b):
    return sum(x * y for x, y in zip(a, b))


def ang_delta(a, b):
    d = abs(qdot(a, b))
    return 2 * math.acos(min(1.0, d))


if __name__ == '__main__':
    import sys
    m = parse(sys.argv[1])
    print('bones', m['nb'], 'keys', len(m['key_times']), 'const', len(m['const']), 'animated', len(m['rot']),
          'dur', m['dur'], 'ticks/s', m['key_times'][-1] / m['dur'])
    deltas = []
    for bi, cur in m['rot'].items():
        for (t0, q0), (t1, q1) in zip(cur, cur[1:]):
            deltas.append(ang_delta(q0, q1))
    deltas.sort()
    n = len(deltas)
    print('key-to-key angular delta (rad): median %.4f  p90 %.4f  p99 %.4f  max %.4f' %
          (deltas[n // 2], deltas[int(n * .9)], deltas[int(n * .99)], deltas[-1]))
