import os
from pathlib import Path

from eyewear_vto.operations import cleanup_stale_workspaces


def test_cleanup_removes_only_stale_real_directories(tmp_path: Path) -> None:
    stale = tmp_path / "stale"
    current = tmp_path / "current"
    stale.mkdir()
    current.mkdir()
    symlink = tmp_path / "linked"
    symlink.symlink_to(stale, target_is_directory=True)
    os.utime(stale, (100, 100))
    os.utime(current, (950, 950))

    removed = cleanup_stale_workspaces(tmp_path, max_age_seconds=100, now=1000)

    assert removed == (stale,)
    assert not stale.exists()
    assert current.exists()
    assert symlink.is_symlink()
