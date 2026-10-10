"""capture_report.py CLI on synthetic captures, output in a temp folder."""
import os

import capture_report
import synth_capture as sc


def test_cli_writes_png_for_one_capture(tmp_path, capsys):
    src = tmp_path / 'in'
    src.mkdir()
    cap_path = str(src / 'vault_test.json')
    sc.write(cap_path, convention='rows', heading_deg=30.0)
    out = tmp_path / 'out'

    rc = capture_report.main([cap_path, '--out-dir', str(out)])

    assert rc == 0
    png = out / 'vault_test.png'
    assert png.exists() and png.stat().st_size > 5000
    text = capsys.readouterr().out
    assert 'trigger' in text and sc.CLIP_SKIP in text
    assert '60.00 fps' in text
    assert 'pose_ok        True' in text
    assert '-> rows' in text
    assert 'hand_plant' in text and 'takeoff' in text and 'landing' in text
    assert 'WARNING' not in text


def test_cli_folder_and_columns_warning(tmp_path, capsys):
    src = tmp_path / 'folder'
    src.mkdir()
    sc.write(str(src / 'a_rows.json'), convention='rows', heading_deg=30.0)
    sc.write(str(src / 'b_cols.json'), convention='columns', heading_deg=30.0)
    out = tmp_path / 'out'

    rc = capture_report.main([str(src), '--out-dir', str(out)])

    assert rc == 0
    assert os.path.exists(out / 'a_rows.png') and os.path.exists(out / 'b_cols.png')
    text = capsys.readouterr().out
    assert '-> columns' in text
    assert 'WARNING' in text        # the module constant is ROW_MAJOR_BASIS=True


def test_cli_no_pose_still_writes_png(tmp_path, capsys):
    cap_path = str(tmp_path / 'nopose.json')
    sc.write(cap_path, pose=False)
    out = tmp_path / 'out'
    assert capture_report.main([cap_path, '--out-dir', str(out)]) == 0
    assert (out / 'nopose.png').exists()
    assert '(no pose)' in capsys.readouterr().out


def test_cli_empty_folder_returns_error(tmp_path):
    empty = tmp_path / 'empty'
    empty.mkdir()
    assert capture_report.main([str(empty), '--out-dir', str(tmp_path / 'o')]) == 1
