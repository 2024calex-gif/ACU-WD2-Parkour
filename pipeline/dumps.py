"""Loader for the original ACU clip dumps (game-decoded JSON) in config.ACU_DUMPS.

Replaces tools/dumpread.py. Each dump file is named <HEXHANDLE>_<clipname>.json and has
Name, Length, Additive, NumTracks, Tracks. A track has TrackID (decimal string of a bone hash),
infotype (0 = rotation, 3 = position, 4 = events), codec and Keyframes (Frame + LocalRotation or
LocalPosition, at 60 Hz frames).
"""
import json
import os
import re

import numpy as np

import config
import retarget
import skel

_NAMES_CACHE = None


def bone_names():
    """dict bone-hash(int) -> bone name: ACU name list plus the table in AnimationEditor.cpp."""
    global _NAMES_CACHE
    if _NAMES_CACHE is None:
        nm = skel.acu_names()
        with open(config.ACUFIXES_ANIM_EDITOR, encoding='utf-8') as f:
            src = f.read()
        for m in re.finditer(r'\{\s*(\d+),\s*"([^"]*)",\s*"([^"]*)"\s*\}', src):
            nm.setdefault(int(m.group(1)), m.group(3) or m.group(2))
        _NAMES_CACHE = nm
    return _NAMES_CACHE


def _norm_handle(hex_str):
    h = str(hex_str).strip().upper()
    if h.startswith('0X'):
        h = h[2:]
    return h


def list_dumps():
    """list of (handle_hex, clip_name, path), sorted by file name."""
    out = []
    for fn in sorted(os.listdir(config.ACU_DUMPS)):
        if not fn.lower().endswith('.json'):
            continue
        stem = fn[:-5]
        handle, _, name = stem.partition('_')
        out.append((handle.upper(), name, os.path.join(config.ACU_DUMPS, fn)))
    return out


def load_by_handle(hex_str):
    """Parsed dump for a handle such as '0x296F0C7D9' or '296F0C7D9' (case-insensitive), or None."""
    h = _norm_handle(hex_str)
    for handle, _, path in list_dumps():
        if handle == h:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
    return None


def load_by_name(clip_name):
    """Parsed dump for an exact clip name, or None."""
    for _, name, path in list_dumps():
        if name == clip_name:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
    return None


def track(dump, track_id, infotype):
    """Keyframes of the first track with this TrackID (int or decimal string) and infotype, or None."""
    tid = int(track_id)
    for t in dump.get('Tracks', []):
        if int(t['TrackID']) == tid and t.get('infotype') == infotype:
            return t.get('Keyframes', [])
    return None


def _bracket(keys, frame):
    """Index pair (a, b) of keys around frame, or None when frame is outside the key range."""
    for i in range(len(keys) - 1):
        if keys[i]['Frame'] <= frame <= keys[i + 1]['Frame']:
            return i, i + 1
    return None


def sample_rotation(keys, frame):
    """Rotation (x, y, z, w) at frame: slerp between the surrounding keys, clamped at both ends.

    Same logic as bake.sample_track with key 'LocalRotation'.
    """
    if not keys:
        return None
    if frame <= keys[0]['Frame']:
        return retarget.qn(keys[0]['LocalRotation'])
    if frame >= keys[-1]['Frame']:
        return retarget.qn(keys[-1]['LocalRotation'])
    pair = _bracket(keys, frame)
    a, b = keys[pair[0]], keys[pair[1]]
    u = (frame - a['Frame']) / max(1, b['Frame'] - a['Frame'])
    return retarget.slerp(retarget.qn(a['LocalRotation']), retarget.qn(b['LocalRotation']), u)


def sample_position(keys, frame):
    """Position (x, y, z) at frame: linear interpolation between the surrounding keys, clamped at both ends."""
    if not keys:
        return None
    if frame <= keys[0]['Frame']:
        return np.array(keys[0]['LocalPosition'], float)
    if frame >= keys[-1]['Frame']:
        return np.array(keys[-1]['LocalPosition'], float)
    pair = _bracket(keys, frame)
    a, b = keys[pair[0]], keys[pair[1]]
    u = (frame - a['Frame']) / max(1, b['Frame'] - a['Frame'])
    pa = np.array(a['LocalPosition'], float)
    pb = np.array(b['LocalPosition'], float)
    return pa + (pb - pa) * u
