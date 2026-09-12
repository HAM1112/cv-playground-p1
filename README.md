# cv-playground-p1 — gridcheck

Version 1 of a small computer-vision project: point a camera at a white sheet with a
bordered grid of boxes and get one of three answers.

| Output               | Meaning                                              |
|----------------------|------------------------------------------------------|
| `Full`               | every box contains a black circle                    |
| `Available`          | at least one box is empty                            |
| `invalid field view` | no usable bordered grid is visible in the frame      |

The number of boxes (rows × columns) is not fixed; it is read from the image.

## How it works

1. **Board detection (OpenCV, `src/gridcheck/board.py`)** – adaptive threshold, contour
   search for the rectangular border, perspective warp to a top-down canvas, grid lines from
   row/column ink profiles, one crop per cell. Frames without a clean border + grid (or with
   the border cut off by the frame edge) return `None` → `invalid field view`.
2. **Cell classifier (PyTorch, `src/gridcheck/model.py`)** – `CellNet`, a ~25k-parameter CNN
   that labels each 64×64 grayscale cell crop as `empty` or `circle`.
3. **Decision (`src/gridcheck/infer.py`)** – all cells `circle` → `Full`, else `Available`.

Training data is synthetic (`src/gridcheck/synth.py`): random grids with random circles,
perspective, lighting, blur, noise and JPEG artefacts, pushed through the same detection
pipeline so training crops match inference crops. Hand-labelled real crops can be added under
`data/real/circle/` and `data/real/empty/` and are picked up automatically by `train`.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 and all dependencies are installed by:

```
uv sync --extra dev
```

CPU wheels of PyTorch are the default; see the commented block in `pyproject.toml` for CUDA.

## Usage

```
uv run gridcheck image tests/fixtures/partial_grid.png          # prints: Available
uv run gridcheck image path/to/photo.jpg --debug --save out.png # overlay + per-cell probabilities
uv run gridcheck cam --debug                                    # live webcam, q to quit
```

`image` pads the picture with a white margin so tightly cropped scans work; `cam` does not,
so a border touching the edge of the camera view is reported as invalid.

## Training (already done; weights in `models/cellnet.pt`)

```
uv run gridcheck synth --n 20000            # -> data/synth_cells.npz
uv run gridcheck train --epochs 10          # -> models/cellnet.pt
uv run pytest                               # detection + end-to-end tests
```

`gridcheck synth --samples some/dir --n 0` writes a few full synthetic frames for inspection.

## Training on your own photos

Drop photos of the sheet into these two folders (git-ignored), sorted by what the answer
should be:

```
data/photos/full/         every box has a circle
data/photos/available/    at least one box is empty
```

Then run one command:

```
uv run gridcheck train
```

It cuts one crop per box out of every photo (`full/` crops are all labelled circle; in
`available/` each box is labelled by the ink in its centre), mixes them with the synthetic data
at ~15% weight, trains for 10 epochs and writes `models/cellnet.pt`. Re-running is safe: crops
are regenerated with fixed names. Options: `--epochs N`, `--photos DIR`, `--no-photos`.

Photos whose border runs off the frame still contribute: every clean box in them is used.
The command warns about `available/` photos in which every box looks filled.

Unsorted photos can be cut into crops for manual sorting with
`uv run gridcheck harvest some/folder --sheets review/`, which also writes contact sheets.

Framing tips for the camera: keep the whole border inside the view with some paper visible
around it. A border that touches the edge of the frame is reported as `invalid field view`.
