"""Harvest real cell crops from photos for training.

Works even when the whole board cannot be detected (e.g. the border runs off the
frame): every clean convex quad of plausible cell size is treated as a cell, warped
to a square crop and auto-labelled by the amount of dark ink in its centre. The
result goes to data/real/{circle,empty}/ plus a contact sheet per label for review.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .board import CROP_SIZE, _binarize, _order_corners, to_gray
from .dataset import DEFAULT_REAL_DIR
from .model import LABELS

MIN_CELL_AREA_FRAC = 0.01
MAX_CELL_AREA_FRAC = 0.20
INNER_SHRINK = 0.08          # pull the quad toward its centre to drop the stroke edges
INK_DARK_RATIO = 0.55        # pixel darker than this * paper level counts as ink
CIRCLE_MIN_INK = 0.04        # fraction of the central region that must be ink (a real
                             # circle is 0.15+; faint erased marks stay below this)
MIN_PAPER_LEVEL = 90         # a crop whose bright level is below this is not a paper cell


def _cell_quads(binary: np.ndarray) -> list[np.ndarray]:
    h, w = binary.shape
    area_frame = float(h * w)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quads: list[tuple[float, np.ndarray]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (MIN_CELL_AREA_FRAC * area_frame <= area <= MAX_CELL_AREA_FRAC * area_frame):
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        pts = approx.reshape(4, 2).astype(np.float32)
        if (pts[:, 0] <= 1).any() or (pts[:, 1] <= 1).any() or (pts[:, 0] >= w - 2).any() or (pts[:, 1] >= h - 2).any():
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        if not (0.35 <= bw / max(bh, 1) <= 2.8):
            continue
        quads.append((area, _order_corners(pts)))
    # Drop near-duplicates (inner/outer edge of the same stroke): keep the smaller.
    quads.sort(key=lambda t: t[0])
    kept: list[np.ndarray] = []
    for _area, q in quads:
        c = q.mean(axis=0)
        if any(np.linalg.norm(c - k.mean(axis=0)) < 0.04 * min(h, w) for k in kept):
            continue
        kept.append(q)
    return kept


def crop_quad(gray: np.ndarray, quad: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    centre = quad.mean(axis=0, keepdims=True)
    q = (centre + (quad - centre) * (1.0 - INNER_SHRINK)).astype(np.float32)
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(gray, M, (size, size), flags=cv2.INTER_AREA)


def auto_label(crop: np.ndarray) -> tuple[int, float]:
    """Return (label index, ink fraction) from the dark pixels in the central 70%."""
    m = int(0.15 * crop.shape[0])
    centre = crop[m:-m, m:-m]
    paper = float(np.percentile(crop, 80))
    ink = float((centre < INK_DARK_RATIO * paper).mean())
    return (1 if ink >= CIRCLE_MIN_INK else 0), ink


def looks_like_paper(crop: np.ndarray) -> bool:
    """False for crops of hand, desk or background that happen to be quad-shaped."""
    return float(np.percentile(crop, 80)) >= MIN_PAPER_LEVEL


DEFAULT_PHOTOS_DIR = DEFAULT_REAL_DIR.parent / "photos"
PHOTO_LABELS = ("full", "available")


def _photo_cells(gray: np.ndarray) -> list[np.ndarray]:
    """Cell crops from a photo: the full detector if it succeeds, else every clean box."""
    from .board import find_board

    res = find_board(gray)
    if res is not None:
        return list(res.crops)
    return [crop_quad(gray, q) for q in _cell_quads(_binarize(gray))]


def harvest_labeled(
    photos_dir: Path | str = DEFAULT_PHOTOS_DIR,
    out_dir: Path | str = DEFAULT_REAL_DIR,
    verbose: bool = False,
) -> dict:
    """Turn photos sorted into <photos_dir>/full and <photos_dir>/available into cell crops.

    full/      -> every box holds a circle, so every crop is labelled `circle`.
    available/ -> boxes are labelled individually by the ink in their centre.
    Crops are written to <out_dir>/{circle,empty}/ with deterministic names, so
    re-running overwrites instead of duplicating. Returns per-folder statistics.
    """
    photos_dir, out_dir = Path(photos_dir), Path(out_dir)
    for name in LABELS:
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    # Regenerate from scratch so crops of deleted or relabelled photos do not linger.
    for name in LABELS:
        for old in (out_dir / name).glob("full_*.png"):
            old.unlink()
        for old in (out_dir / name).glob("available_*.png"):
            old.unlink()
    stats: dict[str, dict] = {}
    for folder in PHOTO_LABELS:
        d = photos_dir / folder
        st = {"photos": 0, "circle": 0, "empty": 0, "no_boxes": [], "suspicious": []}
        stats[folder] = st
        if not d.is_dir():
            continue
        files = [p for p in sorted(d.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        bar = None
        if verbose and files:
            from .progress import Progress

            bar = Progress(len(files), f"  {folder}/")
        for k_photo, p in enumerate(files, start=1):
            img = cv2.imread(str(p))
            if img is None:
                continue
            st["photos"] += 1
            crops = [c for c in _photo_cells(to_gray(img)) if looks_like_paper(c)]
            if bar is not None:
                bar.update(k_photo, extra=f"{p.name}: {len(crops)} boxes", force=True)
            if not crops:
                st["no_boxes"].append(p.name)
                continue
            labels = []
            for k, crop in enumerate(crops):
                if folder == "full":
                    label = 1
                else:
                    label, _ink = auto_label(crop)
                labels.append(label)
                name = LABELS[label]
                fname = f"{folder}_{p.stem.replace(' ', '_')}_{k:02d}.png"
                cv2.imwrite(str(out_dir / name / fname), crop)
                st[name] += 1
            if folder == "available" and all(labels):
                st["suspicious"].append(p.name)   # labelled available but every box looks filled
        if bar is not None:
            from .progress import TICK, c

            plain = f"  {TICK} {folder}/: {st['photos']} photos processed"
            bar.close(f"  {c(TICK, 'bright_green')} {c(folder + '/', 'bold')}: {st['photos']} photos processed",
                      plain_len=len(plain))
    return stats


def harvest(image_dir: Path | str, out_dir: Path | str = DEFAULT_REAL_DIR, sheet_dir: Path | str | None = None) -> dict:
    image_dir, out_dir = Path(image_dir), Path(out_dir)
    for name in LABELS:
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    tiles: dict[str, list[np.ndarray]] = {name: [] for name in LABELS}
    counts = {name: 0 for name in LABELS}
    n_images = 0
    for p in sorted(image_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        n_images += 1
        gray = to_gray(img)
        quads = _cell_quads(_binarize(gray))
        for k, q in enumerate(quads):
            crop = crop_quad(gray, q)
            if not looks_like_paper(crop):
                continue
            label, ink = auto_label(crop)
            name = LABELS[label]
            fname = f"{p.stem.replace(' ', '_')}_{k:02d}.png"
            cv2.imwrite(str(out_dir / name / fname), crop)
            counts[name] += 1
            tile = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_NEAREST)
            cv2.putText(tile, f"{ink:.2f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
            cv2.putText(tile, f"{n_images - 1}:{k}", (2, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
            tiles[name].append(tile)
    if sheet_dir:
        sheet_dir = Path(sheet_dir)
        sheet_dir.mkdir(parents=True, exist_ok=True)
        for name, ts in tiles.items():
            if not ts:
                continue
            cols = 10
            rows = (len(ts) + cols - 1) // cols
            sheet = np.full((rows * 98, cols * 98), 255, np.uint8)
            for i, t in enumerate(ts):
                r, c = divmod(i, cols)
                sheet[r * 98:r * 98 + 96, c * 98:c * 98 + 96] = t
            cv2.imwrite(str(sheet_dir / f"sheet_{name}.png"), sheet)
    return {"images": n_images, **counts}
