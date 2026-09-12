"""Synthetic board renderer: draws bordered grids with random filled circles and
applies camera-like augmentation (perspective, lighting, noise, blur, JPEG).

Used to train the cell classifier and to stress-test board detection.
"""

from __future__ import annotations

import cv2
import numpy as np

FRAME_SIZES = [(640, 480), (800, 600), (960, 540), (1280, 720)]


def _rand_paper_color(rng: np.random.Generator) -> tuple[int, int, int]:
    base = int(rng.integers(215, 256))
    tint = rng.integers(-8, 9, size=3)
    return tuple(int(np.clip(base + t, 0, 255)) for t in tint)


def _rand_ink_color(rng: np.random.Generator, hi: int = 70) -> tuple[int, int, int]:
    base = int(rng.integers(0, hi))
    tint = rng.integers(-10, 11, size=3)
    return tuple(int(np.clip(base + t, 0, 255)) for t in tint)


def draw_board_canvas(
    rng: np.random.Generator,
    rows: int | None = None,
    cols: int | None = None,
    fill_prob: float | None = None,
):
    """Render the paper sheet (top-down, no camera effects).

    Returns (paper_bgr, labels[rows, cols] bool, board_corners (4,2) float32 in paper coords).
    """
    rows = int(rng.integers(1, 5)) if rows is None else rows
    cols = int(rng.integers(1, 9)) if cols is None else cols
    if fill_prob is None:
        u = rng.random()
        if u < 0.10:
            fill_prob = 0.0
        elif u < 0.25:
            fill_prob = 1.0
        else:
            fill_prob = float(rng.uniform(0.2, 0.8))

    cell_w = int(rng.integers(50, 111))
    cell_h = int(np.clip(round(cell_w * rng.uniform(0.8, 1.2)), 40, 130))
    t = int(rng.integers(2, 9))                 # line thickness
    board_w = cols * cell_w + t
    board_h = rows * cell_h + t
    # Paper margin around the board: at least ~2 line widths of white space.
    margin_x = int(rng.uniform(0.08, 0.35) * board_w) + 2 * t
    margin_y = int(rng.uniform(0.08, 0.35) * board_h) + 2 * t
    W, H = board_w + 2 * margin_x, board_h + 2 * margin_y

    paper = np.empty((H, W, 3), dtype=np.uint8)
    paper[:] = _rand_paper_color(rng)
    ink = _rand_ink_color(rng)
    x0, y0 = margin_x, margin_y
    x1, y1 = x0 + board_w - 1, y0 + board_h - 1

    # Outer border.
    cv2.rectangle(paper, (x0, y0), (x1, y1), ink, thickness=t, lineType=cv2.LINE_AA)

    # Internal lines drawn per cell-segment with small jitter (hand-drawn look).
    jitter = max(1, round(0.5 * t))
    for c in range(1, cols):
        x = x0 + c * cell_w
        for r in range(rows):
            dx = int(rng.integers(-jitter, jitter + 1))
            ya = y0 + r * cell_h - t
            yb = y0 + (r + 1) * cell_h + t
            cv2.line(paper, (x + dx, ya), (x + dx, yb), ink, t, cv2.LINE_AA)
    for r in range(1, rows):
        y = y0 + r * cell_h
        for c in range(cols):
            dy = int(rng.integers(-jitter, jitter + 1))
            xa = x0 + c * cell_w - t
            xb = x0 + (c + 1) * cell_w + t
            cv2.line(paper, (xa, y + dy), (xb, y + dy), ink, t, cv2.LINE_AA)

    # Circles.
    labels = rng.random((rows, cols)) < fill_prob
    for r in range(rows):
        for c in range(cols):
            if not labels[r, c]:
                continue
            # "A black circle in the box": clearly inside, only rarely grazing a line.
            cx = x0 + c * cell_w + cell_w / 2 + rng.uniform(-0.12, 0.12) * cell_w
            cy = y0 + r * cell_h + cell_h / 2 + rng.uniform(-0.12, 0.12) * cell_h
            rad = rng.uniform(0.15, 0.38) * min(cell_w, cell_h)
            axes = (int(rad), int(rad * rng.uniform(0.85, 1.15)))
            angle = float(rng.uniform(0, 180))
            cv2.ellipse(
                paper, (int(cx), int(cy)), axes, angle, 0, 360,
                _rand_ink_color(rng, 80), thickness=-1, lineType=cv2.LINE_AA,
            )

    corners = np.array(
        [[x0 - t / 2, y0 - t / 2], [x1 + t / 2, y0 - t / 2],
         [x1 + t / 2, y1 + t / 2], [x0 - t / 2, y1 + t / 2]],
        dtype=np.float32,
    )
    return paper, labels, corners


def _rand_background(rng: np.random.Generator, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    bg = np.empty((h, w, 3), dtype=np.uint8)
    if rng.random() < 0.4:
        bg[:] = _rand_paper_color(rng)        # white desk / another sheet
    else:
        c0 = rng.integers(20, 200, size=3).astype(np.float32)
        c1 = np.clip(c0 + rng.integers(-60, 61, size=3), 0, 255).astype(np.float32)
        ramp = np.linspace(0, 1, w if rng.random() < 0.5 else h, dtype=np.float32)
        if len(ramp) == w:
            grad = c0[None, None, :] * (1 - ramp)[None, :, None] + c1[None, None, :] * ramp[None, :, None]
            bg[:] = np.broadcast_to(grad, (h, w, 3)).astype(np.uint8)
        else:
            grad = c0[None, None, :] * (1 - ramp)[:, None, None] + c1[None, None, :] * ramp[:, None, None]
            bg[:] = np.broadcast_to(grad, (h, w, 3)).astype(np.uint8)
    return bg


def _place_on_frame(rng: np.random.Generator, paper: np.ndarray, board_corners: np.ndarray):
    """Perspective-place the paper into a random camera frame. Returns (frame, board_corners_in_frame)."""
    fw, fh = FRAME_SIZES[int(rng.integers(len(FRAME_SIZES)))]
    ph, pw = paper.shape[:2]
    bw = board_corners[1, 0] - board_corners[0, 0]
    bh = board_corners[2, 1] - board_corners[1, 1]

    # Target board size: 35-88% of frame width, but keep it inside the frame height too.
    scale = rng.uniform(0.35, 0.88) * fw / bw
    scale = min(scale, 0.88 * fh / bh)
    angle = np.deg2rad(rng.uniform(-12, 12))
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    centre = board_corners.mean(axis=0)
    dst = (board_corners - centre) @ R.T * scale
    # Perspective jitter per corner.
    jitter = 0.06 * scale * min(bw, bh)
    dst += rng.uniform(-jitter, jitter, size=dst.shape).astype(np.float32)
    # Translate so the quad sits fully inside the frame with a margin.
    mn, mx = dst.min(axis=0), dst.max(axis=0)
    margin = 0.03 * min(fw, fh)
    lo = -mn + margin
    hi = np.array([fw, fh], dtype=np.float32) - mx - margin
    while (hi < lo).any():
        # Too big after rotation/jitter: shrink around the centre until it fits.
        dst *= 0.9
        mn, mx = dst.min(axis=0), dst.max(axis=0)
        lo = -mn + margin
        hi = np.array([fw, fh], dtype=np.float32) - mx - margin
    shift = np.array([rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])], dtype=np.float32)
    dst += shift

    M = cv2.getPerspectiveTransform(board_corners.astype(np.float32), dst.astype(np.float32))
    bg = _rand_background(rng, (fw, fh))
    warped = cv2.warpPerspective(paper, M, (fw, fh), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), M, (fw, fh), flags=cv2.INTER_LINEAR)
    m = (mask.astype(np.float32) / 255.0)[..., None]
    frame = (warped.astype(np.float32) * m + bg.astype(np.float32) * (1 - m)).astype(np.uint8)
    return frame, dst.astype(np.float32)


def photometric_augment(rng: np.random.Generator, frame: np.ndarray) -> np.ndarray:
    img = frame.astype(np.float32)
    h, w = img.shape[:2]

    # Lighting ramp + global gain.
    if rng.random() < 0.7:
        a, b = rng.uniform(0.55, 1.0), rng.uniform(0.55, 1.0)
        if rng.random() < 0.5:
            ramp = np.linspace(a, b, w, dtype=np.float32)[None, :, None]
        else:
            ramp = np.linspace(a, b, h, dtype=np.float32)[:, None, None]
        img *= ramp
    img *= rng.uniform(0.7, 1.25)

    # Soft shadow blob.
    if rng.random() < 0.25:
        shadow = np.ones((h, w), np.float32)
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        ax, ay = int(rng.uniform(0.2, 0.6) * w), int(rng.uniform(0.2, 0.6) * h)
        cv2.ellipse(shadow, (int(cx), int(cy)), (ax, ay), float(rng.uniform(0, 180)), 0, 360,
                    float(rng.uniform(0.55, 0.85)), -1)
        shadow = cv2.GaussianBlur(shadow, (0, 0), 0.08 * min(w, h))
        img *= shadow[..., None]

    # Gamma.
    if rng.random() < 0.5:
        g = rng.uniform(0.75, 1.35)
        img = 255.0 * np.power(np.clip(img / 255.0, 0, 1), g)

    # Blur (defocus / motion).
    u = rng.random()
    if u < 0.25:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    elif u < 0.35:
        k = int(rng.integers(3, 8))
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        img = cv2.filter2D(img, -1, kern)

    # Sensor noise.
    sigma = rng.uniform(0, 8)
    img += rng.normal(0, sigma, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # JPEG re-encode.
    if rng.random() < 0.5:
        q = int(rng.integers(40, 96))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def render_board(
    rng: np.random.Generator,
    rows: int | None = None,
    cols: int | None = None,
    fill_prob: float | None = None,
    augment: bool = True,
):
    """Full synthetic camera frame.

    Returns (frame_bgr uint8, labels[rows, cols] bool, board_corners (4,2) in frame coords).
    """
    paper, labels, corners = draw_board_canvas(rng, rows, cols, fill_prob)
    frame, corners_f = _place_on_frame(rng, paper, corners)
    if augment:
        frame = photometric_augment(rng, frame)
    return frame, labels, corners_f


def render_invalid(rng: np.random.Generator, kind: int | None = None) -> np.ndarray:
    """A frame with no usable grid: clutter (0), a bare sheet (1), or a board whose
    border runs off the frame edge (2)."""
    fw, fh = FRAME_SIZES[int(rng.integers(len(FRAME_SIZES)))]
    kind = int(rng.integers(0, 3)) if kind is None else kind
    if kind == 0:
        # Random clutter: strokes and blobs on a background.
        frame = _rand_background(rng, (fw, fh))
        for _ in range(int(rng.integers(3, 15))):
            p1 = (int(rng.integers(0, fw)), int(rng.integers(0, fh)))
            p2 = (int(rng.integers(0, fw)), int(rng.integers(0, fh)))
            color = tuple(int(v) for v in rng.integers(0, 255, size=3))
            if rng.random() < 0.5:
                cv2.line(frame, p1, p2, color, int(rng.integers(1, 8)))
            else:
                cv2.circle(frame, p1, int(rng.integers(5, 80)), color, -1)
    elif kind == 1:
        # Bare sheet of paper, no border.
        paper = np.empty((int(fh * 0.7), int(fw * 0.7), 3), np.uint8)
        paper[:] = _rand_paper_color(rng)
        corners = np.array([[0, 0], [paper.shape[1] - 1, 0],
                            [paper.shape[1] - 1, paper.shape[0] - 1], [0, paper.shape[0] - 1]], np.float32)
        frame, _ = _place_on_frame(rng, paper, corners)
    else:
        # A real board, zoomed so its border is guaranteed to run off the frame.
        paper, _labels, corners = draw_board_canvas(rng)
        frame, c = _place_on_frame(rng, paper, corners)
        bw = float(c[:, 0].max() - c[:, 0].min())
        bh = float(c[:, 1].max() - c[:, 1].min())
        zoom = max(fw / bw, fh / bh) * rng.uniform(1.15, 1.8)
        big = cv2.resize(frame, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
        cx, cy = c.mean(axis=0) * zoom
        x = int(np.clip(cx - fw / 2 + rng.uniform(-0.2, 0.2) * fw, 0, big.shape[1] - fw))
        y = int(np.clip(cy - fh / 2 + rng.uniform(-0.2, 0.2) * fh, 0, big.shape[0] - fh))
        frame = big[y:y + fh, x:x + fw]
    return photometric_augment(rng, frame)
