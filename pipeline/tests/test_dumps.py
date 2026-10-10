"""Original ACU clip dumps (dumps.py)."""
import zlib

import numpy as np

import dumps
import retarget

REF_HASH = zlib.crc32(b'Reference')


ORIGINAL_CLIPS = [
    'xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm',
    'xx_Parkour_Muraille_Exit_Fd100cm-Dn100cm',
]


def test_thirty_dumps_listed():
    """The 30 original dumps are still present (the folder grows; do not assert the total)."""
    listed = dumps.list_dumps()
    names = {name for _h, name, _p in listed}
    print('\ndumps listed: %d' % len(listed))
    assert len(listed) >= 30
    missing = [n for n in ORIGINAL_CLIPS if n not in names]
    assert not missing, missing


def test_every_dump_loads_and_has_reference_rotation():
    for handle, name, path in dumps.list_dumps():
        d = dumps.load_by_handle(handle)
        assert d is not None, handle
        assert d['Name'] == name, (handle, d['Name'], name)
        assert d['Length'] > 0, name
        keys = dumps.track(d, REF_HASH, 0)
        assert keys, 'no Reference rotation track in %s' % name


def test_load_by_handle_accepts_prefix_and_any_case():
    a = dumps.load_by_handle('0x296F0C7D9')
    b = dumps.load_by_handle('296f0c7d9')
    assert a is not None and b is not None
    assert a['Name'] == b['Name'] == 'xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm'
    assert dumps.load_by_handle('DEADBEEF') is None


def test_load_by_name():
    d = dumps.load_by_name('xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm')
    assert d is not None and d['Length'] > 0
    assert dumps.load_by_name('no_such_clip') is None


def test_bone_names_cover_reference_and_hips():
    nm = dumps.bone_names()
    assert nm[REF_HASH] == 'Reference'
    assert nm[zlib.crc32(b'Hips')] == 'Hips'


def test_sample_rotation_at_key_frame_returns_key_value():
    for handle, name, _path in dumps.list_dumps():
        d = dumps.load_by_handle(handle)
        keys = dumps.track(d, REF_HASH, 0)
        for k in keys:
            got = dumps.sample_rotation(keys, k['Frame'])
            want = retarget.qn(k['LocalRotation'])
            # q and -q are the same rotation; slerp takes the shortest path, so it may return the negated key
            err = min(float(np.max(np.abs(got - want))), float(np.max(np.abs(got + want))))
            assert err <= 1e-5, '%s frame %d: error %.2e' % (name, k['Frame'], err)


def test_sample_position_at_key_frame_returns_key_value():
    d = dumps.load_by_handle('296F0C7D9')
    keys = dumps.track(d, 0, 3)          # root position track (TrackID 0, infotype 3)
    assert keys
    for k in keys[:: max(1, len(keys) // 10)]:
        got = dumps.sample_position(keys, k['Frame'])
        assert np.allclose(got, k['LocalPosition'], atol=1e-6)
