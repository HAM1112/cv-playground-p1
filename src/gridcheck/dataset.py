"""Build the (crop, label) dataset for CellNet.

Synthetic boards are rendered with `synth.render_board` and pushed through the SAME
`board.find_board` pipeline used at inference, so training crops match real crops.
Boards whose detected grid shape differs from the ground truth are dropped.
Optional real crops are read from data/real/{circle,empty}/*.png.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .board import CROP_SIZE, find_board
from .model import LABELS
from .synth import render_board

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO_ROOT / "data" / "synth_cells.npz"
DEFAULT_REAL_DIR = REPO_ROOT / "data" / "real"


def build_synthetic_cells(
    n_cells: int = 20000,
    seed: int = 0,
    log_every: int = 500,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Render boards until at least n_cells crops are collected.

    Returns (crops uint8 (N, 64, 64), labels int64 (N,), stats).
    """
    rng = np.random.default_rng(seed)
    crops: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    n_boards = n_detected = 0
    total = 0
    t0 = time.time()
    while total < n_cells:
        frame, gt, _corners = render_board(rng)
        n_boards += 1
        res = find_board(frame)
        if res is not None and (res.rows, res.cols) == gt.shape:
            n_detected += 1
            crops.append(res.crops)
            labels.append(gt.reshape(-1).astype(np.int64))
            total += res.n_cells
        if log_every and n_boards % log_every == 0:
            print(
                f"  boards={n_boards} detected={n_detected} "
                f"({100.0 * n_detected / n_boards:.1f}%) cells={total} "
                f"[{time.time() - t0:.0f}s]"
            )
    X = np.concatenate(crops, axis=0)
    y = np.concatenate(labels, axis=0)
    stats = {
        "boards": n_boards,
        "detected": n_detected,
        "detect_rate": n_detected / max(n_boards, 1),
        "cells": int(len(y)),
        "circle_frac": float(y.mean()),
        "seconds": time.time() - t0,
    }
    return X, y, stats


def load_real_cells(real_dir: Path | str = DEFAULT_REAL_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Read hand-labelled crops from <real_dir>/<label>/*.{png,jpg}."""
    real_dir = Path(real_dir)
    xs, ys = [], []
    for idx, name in enumerate(LABELS):
        for p in sorted((real_dir / name).glob("*")):
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            xs.append(cv2.resize(img, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA))
            ys.append(idx)
    if not xs:
        return np.zeros((0, CROP_SIZE, CROP_SIZE), np.uint8), np.zeros((0,), np.int64)
    return np.stack(xs).astype(np.uint8), np.array(ys, dtype=np.int64)


def save_cells(path: Path | str, X: np.ndarray, y: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)


def load_cells(path: Path | str = DEFAULT_CACHE) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as f:
        return f["X"], f["y"]
