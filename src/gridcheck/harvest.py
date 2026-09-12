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
CIRCLE_MIN_INK = 0.02        # fraction of the central region that must be ink


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
