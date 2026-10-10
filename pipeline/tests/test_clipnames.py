"""ACU clip name index (clipnames.py)."""
import clipnames


def test_skip_over_clip_id():
    by_name, by_id = clipnames.load()
    cid = by_name['xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm']
    assert cid.upper() == '296F0C7D9'
    assert by_id[int('296F0C7D9', 16)] == 'xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm'


def test_index_is_consistent():
    by_name, by_id = clipnames.load()
    assert len(by_name) > 31000
    # 53 names appear twice in anim_clip_names.txt with different ids; the index keeps the last one (as bake.py did).
    # Every name must still round-trip through its id.
    for name, cid in by_name.items():
        assert by_id[int(cid, 16)] == name, name
    assert len(by_id) >= len(by_name)
