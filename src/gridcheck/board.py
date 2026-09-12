"""Locate the bordered grid in a frame and cut it into cell crops (OpenCV only).

Pipeline
--------
1. Adaptive threshold (ink -> white) and external contours.
2. Candidate quads: largest 4-vertex convex polygons, not touching the frame edge.
3. For each candidate (largest first): perspective-warp to a top-down canvas,
   keep only ink connected to the canvas border (drops circles), find grid lines
   with row/column projection profiles, validate the resulting cells.
4. Return the first candidate that yields a valid grid, else None (= invalid view).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CROP_SIZE = 64          # cell crops are resized to CROP_SIZE x CROP_SIZE grayscale
WARP_WIDTH = 800        # width of the top-down canvas
WARP_GROW_PX = 3        # grow the quad by at least this many frame px per side before warping
WARP_GROW_FRAC = 0.03   # ... or by this fraction of its longer side, whichever is larger
                        # (wobbly hand-drawn borders make approxPolyDP cut inside the stroke)
WARP_GROW_FRACS = (WARP_GROW_FRAC, 0.012)  # tried in order per candidate quad
GRID_SPAN_FRAC = 0.7    # the grid's ink component must span this much of the canvas
BLOB_OPEN_FRAC = 0.12   # ink thicker than this (of canvas min side) in both directions is a blob
INK_DARKNESS_MAX = 0.6  # border ink must be darker than this fraction of the paper level
CORNER_EXT_FRAC = 0.03  # probe this far (of the edge length) past each corner for lines that continue
CORNER_EXT_MAX_HITS = 3 # >= this many of the 8 probes on ink -> it's a cell of a bigger grid
MIN_QUAD_AREA_FRAC = 0.01  # a thin 1-row strip can be small relative to the frame
EDGE_MARGIN_FRAC = 0.005
LINE_INK_FRAC = 0.40    # fraction of span that must be ink for a row/col to count as a line
LINE_MERGE_FRAC = 0.05  # wobble/misalignment up to this (of span) still reads as one line
LINE_OPEN_FRAC = 0.08   # strokes shorter than this (of span) are not grid lines
LINE_CORE_FRAC = 0.70   # a true grid line reaches this coverage somewhere; blobs do not
LINE_MAX_THICK_FRAC = 0.12
BORDER_TOL_FRAC = 0.06  # first/last line must sit within this of the canvas edge
MAX_CELLS_PER_AXIS = 12
MAX_CANDIDATES = 12     # largest candidate quads to evaluate per frame
BORDER_MIN_COVERAGE = 0.6  # the outer border lines must cover at least this fraction of the span
CELL_UNIFORMITY = 0.30  # std/mean of cell sizes must be below this
CELL_INNER_MARGIN = 0.08


@dataclass
class Line:
    start: int
    end: int

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0

    @property
    def thickness(self) -> int:
        return self.end - self.start + 1


@dataclass
class BoardResult:
    corners: np.ndarray            # (4, 2) float32, order TL, TR, BR, BL in the original frame
    warped: np.ndarray             # top-down grayscale canvas
    rows: int
    cols: int
    cell_boxes: list[tuple[int, int, int, int]]   # (x0, y0, x1, y1) in warped coords, row-major
    crops: np.ndarray              # (rows*cols, CROP_SIZE, CROP_SIZE) uint8, row-major
    homography: np.ndarray         # frame -> warped

    @property
    def n_cells(self) -> int:
        return self.rows * self.cols

    def cell_polygon_in_frame(self, idx: int) -> np.ndarray:
        """Return the 4 corners (in original frame coords) of cell idx."""
        x0, y0, x1, y1 = self.cell_boxes[idx]
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        inv = np.linalg.inv(self.homography)
        return cv2.perspectiveTransform(pts[None], inv)[0]


# --------------------------------------------------------------------------- helpers

def _binarize(gray: np.ndarray, block_frac: float = 0.05) -> np.ndarray:
    """Adaptive threshold, ink -> 255. The block must be wider than the thickest
    stroke, otherwise thick lines come out hollow."""
    h, w = gray.shape
    block = int(max(11, round(block_frac * min(h, w))))
    if block % 2 == 0:
        block += 1
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 10
    )


def _order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _candidate_quads(binary: np.ndarray) -> list[np.ndarray]:
    h, w = binary.shape
    frame_area = float(h * w)
    margin = max(2.0, EDGE_MARGIN_FRAC * min(h, w))
    # RETR_LIST (not EXTERNAL): the board may sit inside a larger closed contour
    # such as the edge of the sheet of paper on a dark desk.
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quads: list[tuple[float, np.ndarray]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_QUAD_AREA_FRAC * frame_area:
            continue
        peri = cv2.arcLength(c, True)
        # Thin rectangles need a tighter tolerance, bumpy hand-drawn ones a looser
        # one, so try a few and take the first that yields a convex quad.
        pts = None
        for eps in (0.01, 0.02, 0.04):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(4, 2).astype(np.float32)
                break
        if pts is None:
            continue
        # Border cut off by the frame edge -> not a usable view.
        if (pts[:, 0] < margin).any() or (pts[:, 1] < margin).any():
            continue
        if (pts[:, 0] > w - 1 - margin).any() or (pts[:, 1] > h - 1 - margin).any():
            continue
        quads.append((area, _order_corners(pts)))
    quads.sort(key=lambda t: -t[0])
    return [q for _, q in quads[:MAX_CANDIDATES]]


def _offset_quad(corners: np.ndarray, e: float) -> np.ndarray:
    """Move every edge of a convex quad outward by e pixels (corners = TL,TR,BR,BL)."""
    c = corners.astype(np.float64)
    centre = c.mean(axis=0)
    normals = []
    for i in range(4):
        p, q = c[i], c[(i + 1) % 4]
        d = q - p
        nrm = np.array([d[1], -d[0]]) / max(np.linalg.norm(d), 1e-6)
        if np.dot(nrm, (p + q) / 2 - centre) < 0:   # make it point outward
            nrm = -nrm
        normals.append(nrm)
    out = np.empty_like(c)
    for i in range(4):
        n_prev, n_next = normals[(i - 1) % 4], normals[i]   # edges meeting at corner i
        denom = 1.0 + float(np.dot(n_prev, n_next))
        out[i] = c[i] + e * (n_prev + n_next) / max(denom, 0.2)
    return out.astype(np.float32)


def _warp(gray: np.ndarray, corners: np.ndarray, grow_frac: float = WARP_GROW_FRAC) -> tuple[np.ndarray, np.ndarray]:
    """Top-down canvas of the quad, grown by a few frame pixels on every side so the
    border stroke (approxPolyDP lands on its outer edge, give or take a pixel) sits
    fully inside the canvas with paper visible outside it."""
    tl, tr, br, bl = corners
    top_w = np.linalg.norm(tr - tl)
    bot_w = np.linalg.norm(br - bl)
    left_h = np.linalg.norm(bl - tl)
    right_h = np.linalg.norm(br - tr)
    long_side = max((top_w + bot_w) / 2.0, (left_h + right_h) / 2.0)
    corners = _offset_quad(corners, max(WARP_GROW_PX, grow_frac * long_side))
    aspect = ((left_h + right_h) / 2.0) / max((top_w + bot_w) / 2.0, 1e-6)
    W = WARP_WIDTH
    H = int(np.clip(round(W * aspect), 60, 3000))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    # Replicate edge pixels so a quad expanded past the frame edge does not get a
    # black band (which would read as ink touching the canvas edge).
    warped = cv2.warpPerspective(gray, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped, M


def _grid_ink(warped_gray: np.ndarray) -> np.ndarray:
    """Binary ink mask keeping only components that span the canvas (the grid).

    The border + grid lines form one connected component covering nearly the whole
    canvas; circles are small separate components and are dropped here.
    """
    # Lines are magnified by the warp, so use a large block (see _binarize).
    binary = _binarize(warped_gray, block_frac=0.3)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h, w = binary.shape
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        _x, _y, bw, bh, _area = stats[i]
        if bw >= GRID_SPAN_FRAC * w and bh >= GRID_SPAN_FRAC * h:
            keep[i] = True
    mask = np.where(keep[labels], 255, 0).astype(np.uint8)
    # A circle that touches a line becomes part of the grid component. Lines are
    # thin in one direction while blobs are thick in both, so a square opening
    # isolates the blobs; subtract them. The kernel is sized by the canvas width
    # (fixed) so it always exceeds any plausible line thickness.
    k = max(3, int(BLOB_OPEN_FRAC * w)) | 1
    blobs = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    if blobs.any():
        blobs = cv2.dilate(blobs, np.ones((5, 5), np.uint8))
        mask[blobs > 0] = 0
    return mask


def _line_profile(ink: np.ndarray, horizontal: bool) -> tuple[np.ndarray, int]:
    """Coverage profile for horizontal (per row) or vertical (per column) grid lines.

    Opening with a long thin kernel keeps only strokes running in the line direction
    (drops circles and the perpendicular lines); dilating perpendicular to it lets
    slightly misaligned hand-drawn segments land on the same profile index.
    Returns (profile, smear, raw) where smear is the dilation width in pixels and
    raw is the un-smeared profile.
    """
    h, w = ink.shape
    span, n = (w, h) if horizontal else (h, w)
    length = max(5, int(LINE_OPEN_FRAC * span))
    smear = max(3, int(LINE_MERGE_FRAC * n)) | 1
    if horizontal:
        opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
        smeared = cv2.dilate(opened, np.ones((smear, 1), np.uint8))
        return (smeared > 0).mean(axis=1), smear, (opened > 0).mean(axis=1)
    opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8))
    smeared = cv2.dilate(opened, np.ones((1, smear), np.uint8))
    return (smeared > 0).mean(axis=0), smear, (opened > 0).mean(axis=0)


def _find_lines(profile: np.ndarray, smear: int = 1, raw: np.ndarray | None = None) -> list[Line] | None:
    """profile[i] = fraction of ink along index i. Returns line groups or None if invalid."""
    n = len(profile)
    hot = profile > LINE_INK_FRAC
    if not hot.any():
        return None
    # Background touching the canvas edge was already stripped (_strip_edge_bands),
    # but a border must still start a little way in, never at index 0.
    if raw is not None and (raw[0] > LINE_INK_FRAC or raw[-1] > LINE_INK_FRAC):
        return None
    # The smear already merged misaligned segments; here only bridge tiny gaps.
    merge_px = max(2, smear // 4)
    max_thick = max(3, int(LINE_MAX_THICK_FRAC * n)) + smear
    idx = np.flatnonzero(hot)
    groups: list[Line] = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev > merge_px:
            groups.append(Line(start, prev))
            start = i
        prev = i
    groups.append(Line(start, prev))
    # Keep only the high-coverage core of each group. A circle touching a line
    # widens the group but never reaches full coverage, so the core is still the
    # line; a group with no core (a lone blob) is not a grid line at all.
    # Two neighbouring lines bridged by a circle form one wide group with two
    # separate cores, so split the core indices into runs.
    refined: list[Line] = []
    for g in groups:
        core = np.flatnonzero(profile[g.start:g.end + 1] >= LINE_CORE_FRAC)
        if len(core) == 0:
            continue
        run_start = run_prev = int(core[0])
        runs: list[tuple[int, int]] = []
        for c in core[1:]:
            c = int(c)
            if c - run_prev > merge_px:
                runs.append((run_start, run_prev))
                run_start = c
            run_prev = c
        runs.append((run_start, run_prev))
        for a, b in runs:
            line = Line(g.start + a, g.start + b)
            if line.thickness <= max_thick:
                refined.append(line)
    groups = refined
    # Undo the smear so cell interiors are not eaten by the dilation.
    half = smear // 2
    for g in groups:
        c = int(round(g.center))
        g.start = min(max(0, g.start + half), c)
        g.end = max(min(n - 1, g.end - half), c)
    if len(groups) < 2:
        return None
    tol = max(BORDER_TOL_FRAC * n, 12)
    if groups[0].start > tol or groups[-1].end < n - 1 - tol:
        return None
    # The outer border must be a solid line, not a patchy shadow (e.g. the paper's edge).
    for g in (groups[0], groups[-1]):
        if profile[g.start:g.end + 1].max() < BORDER_MIN_COVERAGE:
            return None
    if len(groups) - 1 > MAX_CELLS_PER_AXIS:
        return None
    return groups


def _cells_from_lines(lines: list[Line]) -> list[tuple[int, int]] | None:
    cells = []
    for a, b in zip(lines[:-1], lines[1:]):
        lo, hi = a.end + 1, b.start - 1
        if hi - lo < 4:
            return None
        cells.append((lo, hi))
    sizes = np.array([hi - lo for lo, hi in cells], dtype=np.float32)
    if sizes.std() / max(sizes.mean(), 1e-6) > CELL_UNIFORMITY:
        return None
    return cells


def _strip_edge_bands(ink: np.ndarray) -> np.ndarray | None:
    """Remove dark bands that touch the canvas edge.

    The canvas is grown past the quad, so a drawn border always has some paper
    outside it. Ink touching the canvas edge is therefore the background beyond
    the sheet (a bare sheet on a dark desk, or a board drawn right up to the
    paper's edge). Strip it; if a real border exists it survives as a separate
    line further in, and a bare sheet is left with nothing.
    Returns None if the bands are so deep that nothing sensible remains.
    """
    h, w = ink.shape
    out = ink.copy()
    max_depth_r, max_depth_c = int(0.15 * h), int(0.15 * w)
    rows = (out > 0).mean(axis=1) > LINE_INK_FRAC
    cols = (out > 0).mean(axis=0) > LINE_INK_FRAC

    def run_len(flags: np.ndarray, limit: int) -> int:
        n = 0
        while n < min(limit, len(flags)) and flags[n]:
            n += 1
        return n

    top, bot = run_len(rows, max_depth_r), run_len(rows[::-1], max_depth_r)
    left, right = run_len(cols, max_depth_c), run_len(cols[::-1], max_depth_c)
    if top >= max_depth_r or bot >= max_depth_r or left >= max_depth_c or right >= max_depth_c:
        return None
    if top:
        out[:top + 1, :] = 0
    if bot:
        out[h - bot - 1:, :] = 0
    if left:
        out[:, :left + 1] = 0
    if right:
        out[:, w - right - 1:] = 0
    return out


def _analyse_warped(warped: np.ndarray):
    ink = _strip_edge_bands(_grid_ink(warped))
    if ink is None:
        return None
    row_profile, smear_r, raw_r = _line_profile(ink, horizontal=True)
    col_profile, smear_c, raw_c = _line_profile(ink, horizontal=False)
    h_lines = _find_lines(row_profile, smear_r, raw_r)
    v_lines = _find_lines(col_profile, smear_c, raw_c)
    if h_lines is None or v_lines is None:
        return None
    row_cells = _cells_from_lines(h_lines)
    col_cells = _cells_from_lines(v_lines)
    if row_cells is None or col_cells is None:
        return None
    if not _border_is_ink(warped, ink, h_lines, v_lines):
        return None
    return row_cells, col_cells


def _border_is_ink(warped: np.ndarray, ink: np.ndarray, h_lines: list[Line], v_lines: list[Line]) -> bool:
    """A drawn border is much darker than the paper inside it. The edge of a bare
    sheet against a desk of similar brightness, or a brighter desk, is not."""
    h, w = ink.shape
    border = np.zeros((h, w), dtype=bool)
    for g in (h_lines[0], h_lines[-1]):
        border[g.start:g.end + 1, :] = True
    for g in (v_lines[0], v_lines[-1]):
        border[:, g.start:g.end + 1] = True
    ink_px = warped[border & (ink > 0)]
    interior = warped[h_lines[0].end + 1:h_lines[-1].start, v_lines[0].end + 1:v_lines[-1].start]
    if ink_px.size < 10 or interior.size == 0:
        return False
    paper = float(np.percentile(interior, 75))
    return float(np.median(ink_px)) < INK_DARKNESS_MAX * paper


def _crop_cells(warped: np.ndarray, row_cells, col_cells):
    boxes: list[tuple[int, int, int, int]] = []
    crops = []
    for (y0, y1) in row_cells:
        for (x0, x1) in col_cells:
            mh = int(CELL_INNER_MARGIN * (y1 - y0))
            mw = int(CELL_INNER_MARGIN * (x1 - x0))
            bx0, by0, bx1, by1 = x0 + mw, y0 + mh, x1 - mw, y1 - mh
            boxes.append((bx0, by0, bx1, by1))
            crop = warped[by0:by1 + 1, bx0:bx1 + 1]
            crops.append(cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA))
    return boxes, np.stack(crops).astype(np.uint8)


def _lines_continue_past_corners(binary: np.ndarray, corners: np.ndarray) -> bool:
    """True if the quad's edges keep going past its corners, i.e. the quad is one
    cell of a larger grid (the outer border must have paper around it)."""
    h, w = binary.shape
    hits = 0
    for i in range(4):
        p, q = corners[i], corners[(i + 1) % 4]
        d = q - p
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            continue
        d = d / length
        # Probe at two distances past the corner. A neighbouring grid line runs on
        # for a whole cell and hits both; a hand-drawn overshoot at the corner is
        # short and hits only the near one.
        near = float(np.clip(CORNER_EXT_FRAC * length, 4.0, 20.0))
        far = float(np.clip(2.5 * CORNER_EXT_FRAC * length, near + 4.0, 60.0))
        r = max(1, int(0.005 * length))

        def on_ink(pt: np.ndarray) -> bool:
            x, y = int(round(pt[0])), int(round(pt[1]))
            return bool(r <= x < w - r and r <= y < h - r
                        and binary[y - r:y + r + 1, x - r:x + r + 1].any())

        for base, sign in ((p, -1.0), (q, 1.0)):
            if on_ink(base + sign * d * near) and on_ink(base + sign * d * far):
                hits += 1
    return hits >= CORNER_EXT_MAX_HITS


# --------------------------------------------------------------------------- public API

def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def find_board(frame: np.ndarray, pad_frac: float = 0.0) -> BoardResult | None:
    """Detect the bordered grid in a BGR (or grayscale) frame.

    pad_frac: add a white margin of this fraction of the frame size before detecting.
    Use it for tightly cropped image files whose border touches the image edge; keep
    it at 0 for camera frames so a cut-off border is reported as invalid.

    Returns None when no usable grid is found (-> "invalid field view").
    """
    gray = to_gray(frame)
    px = py = 0
    if pad_frac > 0:
        h, w = gray.shape
        py, px = int(round(pad_frac * h)), int(round(pad_frac * w))
        gray = cv2.copyMakeBorder(gray, py, py, px, px, cv2.BORDER_CONSTANT, value=255)
    binary = _binarize(gray)
    best: tuple[tuple[int, float], BoardResult] | None = None
    for corners in _candidate_quads(binary):
        if _lines_continue_past_corners(binary, corners):
            continue
        # A generous margin copes with wobbly hand-drawn borders; a tight one with
        # boards drawn close to the edge of the sheet. Try generous first.
        analysed = None
        for grow in WARP_GROW_FRACS:
            warped, M = _warp(gray, corners, grow)
            analysed = _analyse_warped(warped)
            if analysed is not None:
                break
        if analysed is None:
            continue
        row_cells, col_cells = analysed
        boxes, crops = _crop_cells(warped, row_cells, col_cells)
        area = float(cv2.contourArea(corners))
        if px or py:
            # Express corners / homography in the un-padded frame's coordinates.
            shift = np.array([[1, 0, px], [0, 1, py], [0, 0, 1]], dtype=np.float64)
            M = M @ shift
            corners = corners - np.array([px, py], dtype=np.float32)
        result = BoardResult(
            corners=corners,
            warped=warped,
            rows=len(row_cells),
            cols=len(col_cells),
            cell_boxes=boxes,
            crops=crops,
            homography=M,
        )
        # Prefer the candidate that explains the most cells (a single cell or the
        # sheet of paper can also look like a 1x1 grid); on a tie prefer the
        # tightest quad (nested outer/inner stroke edges give the same grid).
        score = (result.n_cells, -area)
        if best is None or score > best[0]:
            best = (score, result)
    return best[1] if best else None
