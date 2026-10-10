"""ACU clip name index (config.ACU_CLIP_NAMES): tab separated lines of hex id, name, size, container."""
import config


def load():
    """Returns (name -> hex id string as written in the file, int id -> name).

    53 names occur twice with different ids in the source file; the name map keeps the last one (same as bake.py).
    """
    by_name = {}
    by_id = {}
    with open(config.ACU_CLIP_NAMES, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if not line:
                continue
            cid, name, _size, _container = line.split('\t')
            by_name[name] = cid
            by_id[int(cid, 16)] = name
    return by_name, by_id
