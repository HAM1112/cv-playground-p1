"""Reset the project to a fresh state by removing everything training produced."""

from __future__ import annotations

from pathlib import Path

from .dataset import DEFAULT_CACHE, DEFAULT_REAL_DIR, REPO_ROOT
from .harvest import DEFAULT_PHOTOS_DIR
from .model import DEFAULT_MODEL_PATH

CAPTURES_DIR = REPO_ROOT / "captures"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def _images_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def plan_reset(include_photos: bool = False) -> dict[str, list[Path]]:
    """What a reset would delete, grouped by kind (nothing is removed here)."""
    plan: dict[str, list[Path]] = {
        "trained model": [DEFAULT_MODEL_PATH] if DEFAULT_MODEL_PATH.exists() else [],
        "synthetic dataset": [DEFAULT_CACHE] if DEFAULT_CACHE.exists() else [],
        "harvested cell crops": _images_in(DEFAULT_REAL_DIR),
        "camera snapshots": _images_in(CAPTURES_DIR),
    }
    if include_photos:
        plan["your training photos"] = _images_in(DEFAULT_PHOTOS_DIR)
    return plan


def run_reset(plan: dict[str, list[Path]]) -> int:
    """Delete everything in the plan. Folders (and .gitkeep files) are left in place."""
    n = 0
    for paths in plan.values():
        for p in paths:
            if p.exists():
                p.unlink()
                n += 1
    return n
