from __future__ import annotations

import shutil
import time
from pathlib import Path


def cleanup_stale_workspaces(root: Path, max_age_seconds: int, now: float | None = None) -> tuple[Path, ...]:
    if not root.exists():
        return ()
    cutoff = (now if now is not None else time.time()) - max_age_seconds
    removed: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir() or child.is_symlink():
            continue
        if child.stat().st_mtime >= cutoff:
            continue
        shutil.rmtree(child)
        removed.append(child)
    return tuple(removed)

