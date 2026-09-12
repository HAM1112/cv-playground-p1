from pathlib import Path

import cv2
import numpy as np
import pytest

from gridcheck.model import DEFAULT_MODEL_PATH
from gridcheck.infer import AVAILABLE, FULL, INVALID, Classifier, draw_debug
from gridcheck.synth import render_board

FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.exists(), reason="no trained weights; run `gridcheck train`"
)


@pytest.fixture(scope="module")
def clf():
    return Classifier(device="cpu")


def test_empty_grid_is_available(clf):
    img = cv2.imread(str(FIXTURES / "empty_grid.png"))
    v = clf.classify_frame(img, pad_frac=0.05)
    assert v.status == AVAILABLE
    assert v.n_filled == 0 and v.n_cells == 8


def test_partial_grid_is_available_with_expected_cells(clf):
    img = cv2.imread(str(FIXTURES / "partial_grid.png"))
    v = clf.classify_frame(img, pad_frac=0.05)
    assert v.status == AVAILABLE
    expected = np.array([[True, True, True, False], [True, False, True, False]])
    assert v.filled.tolist() == expected.tolist()


def test_full_synthetic_board_is_full(clf):
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(5):
        frame, labels, _ = render_board(rng, rows=2, cols=3, fill_prob=1.0)
        v = clf.classify_frame(frame)
        if v.status == FULL:
            hits += 1
    assert hits >= 4


def test_noise_is_invalid(clf):
    noise = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    v = clf.classify_frame(noise)
    assert v.status == INVALID
    assert v.board is None


def test_debug_overlay_has_frame_shape(clf):
    img = cv2.imread(str(FIXTURES / "partial_grid.png"))
    v = clf.classify_frame(img, pad_frac=0.05)
    vis = draw_debug(img, v)
    assert vis.shape == img.shape
