"""WD2 .mab rotation decoder (mab.py) on the unpacked game files."""
import statistics

import mab
import wd2

KONG = 'b00-000-00_vaultover-kong_mplyr_000f_norm_nowep'


def test_mab_index_covers_all_files():
    index = wd2.mab_index()
    assert len(index) == 398, 'expected 398 unique .mab stems, got %d' % len(index)


def test_most_mab_files_parse():
    index = wd2.mab_index()
    failed = []
    for stem, path in sorted(index.items()):
        try:
            mab.parse(path)
        except Exception as exc:  # noqa: BLE001 - report every failure, not just the first
            failed.append((stem, repr(exc)))
    print('\n.mab parse failures: %d of %d' % (len(failed), len(index)))
    for stem, err in failed:
        print('  ', stem, err)
    assert len(index) - len(failed) >= 397


def test_kong_clip_duration_fps_and_smoothness():
    m = wd2.load(KONG)
    assert m['dur'] > 2.5, 'duration %.3f s' % m['dur']
    assert m['fps'] == 30, 'fps %r' % m['fps']

    deltas = []
    for _bone, keys in m['rot'].items():
        for (_t0, q0), (_t1, q1) in zip(keys, keys[1:]):
            deltas.append(mab.ang_delta(q0, q1))
    assert deltas, 'kong clip has no animated rotation keys'
    median = statistics.median(deltas)
    print('\nkong key-to-key angular delta: median %.4f rad over %d steps' % (median, len(deltas)))
    assert median < 0.1, 'median key-to-key delta %.4f rad' % median
