import numpy as np

from gridcheck.board import find_board
from gridcheck.synth import render_board, render_invalid


def test_render_board_shapes_and_labels():
    rng = np.random.default_rng(0)
    frame, labels, corners = render_board(rng, rows=2, cols=4, fill_prob=1.0)
    assert frame.ndim == 3 and frame.dtype == np.uint8
    assert labels.shape == (2, 4) and labels.all()
    assert corners.shape == (4, 2)


def test_detection_recovers_grid_shape_on_most_synthetic_boards():
    rng = np.random.default_rng(1)
    n, ok = 60, 0
    for _ in range(n):
        frame, labels, _ = render_board(rng)
        res = find_board(frame)
        ok += res is not None and (res.rows, res.cols) == labels.shape
    assert ok / n >= 0.85, f"only {ok}/{n} synthetic boards detected correctly"


def test_invalid_frames_are_rejected():
    rng = np.random.default_rng(2)
    accepted = 0
    for kind in range(3):
        for _ in range(15):
            accepted += find_board(render_invalid(rng, kind)) is not None
    assert accepted <= 2, f"{accepted} invalid frames were accepted as boards"
