"""Access to the unpacked Watch Dogs 2 animation files (config.WD2_DATA, searched recursively)."""
import os

import config
import mab


def mab_index():
    """dict stem -> path for every .mab under config.WD2_DATA (stem = file name without '.mab')."""
    index = {}
    for dirpath, _, files in os.walk(config.WD2_DATA):
        for fn in files:
            if fn.lower().endswith('.mab'):
                index[fn[:-4]] = os.path.join(dirpath, fn)
    return index


def load(stem):
    """Parsed .mab (mab.parse) for a stem from mab_index(). Raises KeyError for an unknown stem."""
    index = mab_index()
    if stem not in index:
        raise KeyError('no .mab named %r under %s' % (stem, config.WD2_DATA))
    return mab.parse(index[stem])
