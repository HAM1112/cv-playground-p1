"""Frame -> "Full" | "Available" | "invalid field view"."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .board import BoardResult, find_board_explained
from .model import load_model, pick_device, preprocess

FULL = "Full"
AVAILABLE = "Available"
INVALID = "invalid field view"
CIRCLE_THRESHOLD = 0.5


@dataclass
class Verdict:
    status: str
    rows: int = 0
    cols: int = 0
    filled: np.ndarray | None = None      # (rows, cols) bool
    probs: np.ndarray | None = None       # (rows, cols) P(circle)
    board: BoardResult | None = None
    reason: str = ""                      # why the view is invalid (empty when valid)

    @property
    def n_filled(self) -> int:
        return int(self.filled.sum()) if self.filled is not None else 0

    @property
    def n_cells(self) -> int:
        return self.rows * self.cols


class Classifier:
    def __init__(self, model_path: str | None = None, device: str | None = None) -> None:
        self.device = pick_device(device)
        self.model = load_model(model_path, self.device)

    @torch.no_grad()
    def predict_cells(self, crops: np.ndarray) -> np.ndarray:
        """uint8 (N, 64, 64) -> P(circle) float (N,)."""
        if len(crops) == 0:
            return np.zeros((0,), np.float32)
        x = preprocess(crops).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[:, 1]
        return probs.cpu().numpy()

    def classify_frame(self, frame: np.ndarray, pad_frac: float = 0.0) -> Verdict:
        board, reason = find_board_explained(frame, pad_frac=pad_frac)
        if board is None:
            return Verdict(INVALID, reason=reason)
        probs = self.predict_cells(board.crops).reshape(board.rows, board.cols)
        filled = probs >= CIRCLE_THRESHOLD
        status = FULL if filled.all() else AVAILABLE
        return Verdict(status, board.rows, board.cols, filled, probs, board)


def draw_debug(frame: np.ndarray, verdict: Verdict) -> np.ndarray:
    """Overlay border, per-cell predictions and the status text on a copy of the frame."""
    out = frame.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    h = out.shape[0]
    scale = max(0.5, h / 720.0)
    thick = max(1, int(round(2 * scale)))

    if verdict.board is not None:
        b = verdict.board
        cv2.polylines(out, [b.corners.astype(np.int32)], True, (255, 160, 0), thick + 1)
        for idx in range(b.n_cells):
            r, c = divmod(idx, b.cols)
            poly = b.cell_polygon_in_frame(idx).astype(np.int32)
            is_filled = bool(verdict.filled[r, c])
            color = (60, 60, 230) if is_filled else (60, 200, 60)
            cv2.polylines(out, [poly], True, color, thick)
            p = float(verdict.probs[r, c])
            cx, cy = poly.mean(axis=0).astype(int)
            cv2.putText(out, f"{p:.2f}", (cx - int(18 * scale), cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45 * scale, color, thick, cv2.LINE_AA)
        text = f"{verdict.status}  ({verdict.n_filled}/{verdict.n_cells} filled, {b.rows}x{b.cols})"
        color = (60, 60, 230) if verdict.status == FULL else (60, 200, 60)
    else:
        text = verdict.status
        color = (0, 200, 255)

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, thick)
    cv2.rectangle(out, (8, 8), (16 + tw, 20 + th), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 14 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, color, thick, cv2.LINE_AA)
    if verdict.reason:
        (rw, rh), _ = cv2.getTextSize(verdict.reason, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, 1)
        y0 = 26 + th
        cv2.rectangle(out, (8, y0), (16 + rw, y0 + rh + 10), (0, 0, 0), -1)
        cv2.putText(out, verdict.reason, (12, y0 + rh + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale,
                    (200, 200, 200), 1, cv2.LINE_AA)
    return out
