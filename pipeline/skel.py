"""Load the WD2 and ACU bind-pose skeletons into one simple structure.

Bone = dict(name, hash, parent(index or -1), lpos(x,y,z), lrot(x,y,z,w)).
Rotations are quaternions (x, y, z, w). Local = relative to parent.
"""
import re, struct, zlib
import xml.etree.ElementTree as ET

import config


def load_wd2(xml_path=config.WD2_SKEL_XML):
    tree = ET.parse(xml_path)
    bones = []
    for obj in tree.getroot().iter('object'):
        if obj.get('hash') != 'DCB67730':
            continue
        f = {c.get('hash'): (c.text or '') for c in obj.findall('field')}
        rot = struct.unpack('<4f', bytes.fromhex(f['E7D23810']))
        pos = struct.unpack('<3f', bytes.fromhex(f['4DE430A9']))
        par = int(f['498FB513'], 16) if f['498FB513'] else 0
        if par == 0xFF:
            par = -1
        name = bytes.fromhex(f['F96FAF0D']).rstrip(b'\0').decode('latin1')
        bones.append(dict(name=name, hash=zlib.crc32(name.encode()), parent=par, lpos=pos, lrot=rot))
    return bones


def load_acu(path=config.ACU_SKEL_BIN):
    d = open(path, 'rb').read()
    marker = bytes.fromhex('49107495')
    pos = [i for i in range(len(d)) if d[i:i + 4] == marker]
    ids = {}
    recs = []
    for k, p in enumerate(pos):
        end = pos[k + 1] if k + 1 < len(pos) else len(d)
        b = d[p:end]
        nh = struct.unpack_from('<I', b, 4)[0]
        # object id of this bone sits right before the class hash: 8 bytes ending at p
        oid = struct.unpack_from('<Q', d, p - 8)[0] if p >= 8 else None
        ids[oid] = k
        # parent pointer: 0x02 + u64 at +8 (root has a different prefix)
        par_id = struct.unpack_from('<Q', b, 9)[0] if b[8] == 2 else None
        gp = struct.unpack_from('<4f', b, 26)
        gr = struct.unpack_from('<4f', b, 42)
        lp = struct.unpack_from('<4f', b, 58)
        lr = struct.unpack_from('<4f', b, 74)
        recs.append(dict(hash=nh, par_id=par_id, gpos=gp[:3], grot=gr, lpos=lp[:3], lrot=lr))
    for r in recs:
        r['parent'] = ids.get(r['par_id'], -1) if r['par_id'] is not None else -1
    return recs


def acu_names():
    """Names known for ACU hashes (from the ACUFixes bone list + simple guesses)."""
    names = ['Reference', 'Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftFootThumb1', 'RightUpLeg', 'RightLeg',
             'RightFoot', 'RightFootThumb1', 'Spine', 'Spine1', 'Spine2', 'LeftClavicle', 'LeftArm', 'LeftForeArm',
             'LeftHand', 'Neck', 'Head', 'RightClavicle', 'RightArm', 'RightForeArm', 'RightHand']
    for side in ('Left', 'Right'):
        for fng in ('Index', 'Middle', 'Ring', 'Pinky'):
            for n in range(0, 4):
                names.append('%sHand%s%d' % (side, fng, n))
            names.append('%sHand%s1Rot' % (side, fng))
        for n in range(1, 4):
            names.append('%sHandThumb%d' % (side, n))
    return {zlib.crc32(n.encode()): n for n in names}


if __name__ == '__main__':
    w = load_wd2()
    print('WD2 bones', len(w))
    for i, b in enumerate(w[:12]):
        print(i, b['name'], b['parent'], ['%.3f' % x for x in b['lpos']], ['%.3f' % x for x in b['lrot']])
    a = load_acu()
    nm = acu_names()
    print('ACU bones', len(a))
    for i, b in enumerate(a[:24]):
        print(i, nm.get(b['hash'], '%08x' % b['hash']), b['parent'], ['%.3f' % x for x in b['lpos']], ['%.3f' % x for x in b['lrot']])
