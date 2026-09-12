from pathlib import Path

import cv2
import numpy as np
import pytest

from gridcheck.board import CROP_SIZE, find_board

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("name", ["empty_grid.png", "partial_grid.png"])
def test_fixtures_detect_2x4(name):
    img = cv2.imread(str(FIXTURES / name))
    assert img is not None
    res = find_board(img, pad_frac=0.05)
    assert res is not None
    assert (res.rows, res.cols) == (2, 4)
    assert res.crops.shape == (8, CROP_SIZE, CROP_SIZE)
    assert res.crops.dtype == np.uint8


def test_partial_grid_ink_lands_in_expected_cells():
    img = cv2.imread(str(FIXTURES / "partial_grid.png"))
    res = find_board(img, pad_frac=0.05)
    ink = (res.crops < 128).mean(axis=(1, 2))
    filled = ink > 0.05
    expected = np.array([True, True, True, False, True, False, True, False])
    assert filled.tolist() == expected.tolist()


def test_empty_grid_crops_are_blank():
    img = cv2.imread(str(FIXTURES / "empty_grid.png"))
    res = find_board(img, pad_frac=0.05)
    assert (res.crops < 128).mean() < 0.01


def test_blank_and_noise_frames_are_invalid():
    blank = np.full((480, 640, 3), 255, np.uint8)
    assert find_board(blank) is None
    noise = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    assert find_board(noise) is None


def test_cell_polygons_map_back_into_frame():
    img = cv2.imread(str(FIXTURES / "partial_grid.png"))
    res = find_board(img, pad_frac=0.05)
    h, w = img.shape[:2]
    for idx in range(res.n_cells):
        poly = res.cell_polygon_in_frame(idx)
        assert poly.shape == (4, 2)
        assert (poly[:, 0] >= -2).all() and (poly[:, 0] <= w + 2).all()
        assert (poly[:, 1] >= -2).all() and (poly[:, 1] <= h + 2).all()
