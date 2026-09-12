# gridcheck — the complete beginner's guide

This document explains the whole project from top to bottom: what it does, how to run it,
how to train it, which tools it uses and why, where every file lives, and how the code works
inside. You don't need prior computer-vision or machine-learning experience to follow it.

---

## 1. What the project does

You point a camera at a white sheet of paper. On the sheet is a rectangle (the **border**)
divided into boxes (a **grid**). Some boxes contain a black filled circle. The program looks
at the camera picture and prints exactly one of three answers:

| Output               | Meaning                                                              |
|----------------------|----------------------------------------------------------------------|
| `Full`               | every box contains a circle                                          |
| `Available`          | at least one box is empty                                            |
| `invalid field view` | the picture does not show a complete, usable bordered grid           |

The number of boxes is **not fixed**. The program counts the rows and columns itself, so a
2×4 sheet, a 1×6 strip and a 3×5 grid all work with the same code and the same model.

---

## 2. How it works in one picture

```
 camera / image file
        │
        ▼
 ┌───────────────────────────────┐
 │ 1. FIND THE BOARD (OpenCV)    │  classical image processing, no learning
 │   • threshold: ink → white    │
 │   • find the rectangular      │
 │     border                    │
 │   • flatten it (perspective)  │
 │   • find grid lines           │
 │   • cut one crop per box      │──── nothing usable? ──► "invalid field view"
 └───────────────┬───────────────┘
                 │  N small 64×64 grey images (one per box)
                 ▼
 ┌───────────────────────────────┐
 │ 2. CLASSIFY EACH BOX (PyTorch)│  a tiny neural network, trained by us
 │   CellNet: crop → "circle"    │
 │            or "empty"         │
 └───────────────┬───────────────┘
                 │  a yes/no per box
                 ▼
 ┌───────────────────────────────┐
 │ 3. DECIDE                     │
 │   all circle → "Full"         │
 │   otherwise  → "Available"    │
 └───────────────────────────────┘
```

Why two stages? Finding straight lines and rectangles is something classical image processing
does reliably and explains well. Deciding whether a blurry, unevenly lit blob is "a circle" is
what a small neural network does well. Splitting the job keeps each part simple and testable.

---

## 3. Project layout

```
cv-playground-p1/
├── pyproject.toml            project definition: name, Python version, dependencies, the
│                             `gridcheck` command
├── uv.lock                   exact versions of every installed package (reproducible installs)
├── README.md                 short overview
├── GUIDE.md                  this file
├── .gitignore                which files git must NOT track (photos, generated data, venv)
│
├── src/gridcheck/            the Python package (all the code)
│   ├── cli.py                the `gridcheck` command line: parses arguments, calls the modules
│   ├── board.py              stage 1: find the border, grid lines and cells (OpenCV)
│   ├── model.py              CellNet, the neural network, plus save/load helpers
│   ├── infer.py              stage 2+3: run the model on the cells, produce the verdict,
│   │                         draw the debug overlay
│   ├── synth.py              draws random synthetic boards for training
│   ├── dataset.py            turns synthetic boards + real crops into a training dataset
│   ├── train.py              the training loop
│   ├── harvest.py            cuts labelled cell crops out of your photos
│   └── reset.py              deletes everything training produced
│
├── models/
│   └── cellnet.pt            THE TRAINED MODEL (≈96 KB). Committed to git.
│
├── data/
│   ├── photos/
│   │   ├── full/             YOUR photos where every box has a circle      (git-ignored)
│   │   └── available/        YOUR photos where at least one box is empty   (git-ignored)
│   ├── real/
│   │   ├── circle/           64×64 crops cut from your photos, label circle (git-ignored)
│   │   └── empty/            64×64 crops cut from your photos, label empty  (git-ignored)
│   └── synth_cells.npz       20 000 synthetic crops + labels, one file      (git-ignored)
│
├── captures/                 frames you saved from the camera with the `s` key (git-ignored)
│
└── tests/
    ├── fixtures/             the two reference sketches (empty_grid.png, partial_grid.png)
    ├── test_board.py         board detection tests
    ├── test_synth.py         synthetic renderer + detection-rate tests
    └── test_infer.py         end-to-end tests with the trained model
```

`.venv/` (created by `uv sync`) holds the installed Python and packages. Never edit it.

---

## 4. Setting up

### 4.1 Requirements

- Windows, macOS or Linux.
- [`uv`](https://docs.astral.sh/uv/) — a fast Python package manager. It also downloads the
  right Python version for you, so you do **not** need Python installed separately.
- A webcam (only for `gridcheck cam`).

### 4.2 Install

```
git clone https://github.com/HAM1112/cv-playground-p1.git
cd cv-playground-p1
uv sync --extra dev
```

`uv sync` reads `pyproject.toml` and `uv.lock`, downloads Python 3.12 if needed, creates
`.venv/` and installs PyTorch, OpenCV, NumPy and pytest with the exact locked versions.
`--extra dev` adds pytest (only needed to run the tests).

Every command in this guide is run as `uv run gridcheck ...`. `uv run` means "run this inside
the project's virtual environment", so you never have to activate anything by hand.

### 4.3 Optional: GPU

The default installs the CPU build of PyTorch, which is plenty: the model has 23 538
parameters and trains in about 3 minutes on a laptop CPU. If you want to use an NVIDIA GPU,
uncomment the two blocks at the bottom of `pyproject.toml` and run `uv sync` again.

---

## 5. All commands

Run `uv run gridcheck --help` or `uv run gridcheck <command> --help` at any time.

### `gridcheck image <path>` — classify one picture

```
uv run gridcheck image tests/fixtures/partial_grid.png
uv run gridcheck image photo.jpg --debug
uv run gridcheck image photo.jpg --debug --save overlay.png
```

Prints `Full`, `Available` or `invalid field view` to standard output (so it can be piped into
another program). Everything else goes to standard error.

| Option          | Purpose                                                                 |
|-----------------|-------------------------------------------------------------------------|
| `--debug`       | also print the grid size, the count of filled boxes, the per-box probability of "circle", and open a window with the overlay |
| `--save FILE`   | with `--debug`: write the overlay image to a file instead of opening a window |
| `--no-pad`      | treat the file like a camera frame. By default the image is padded with a white margin so tightly cropped scans (border touching the image edge) still work |
| `--model FILE`  | use another weights file than `models/cellnet.pt`                       |
| `--device cpu|cuda` | force the PyTorch device                                            |

### `gridcheck cam` — live webcam

```
uv run gridcheck cam --debug
```

First it looks for connected cameras (indices 0–4). With one camera it just uses it. With
several it lists them and asks *"Are you using an external webcam? [y/N]"*: `y` picks the
external one (or asks which index if there are several), `n` uses the built-in camera
(index 0). Pass `--device N` to skip the question.

Then it opens the camera, shows a window, prints the status **every time it changes** (a
majority vote over the last 5 frames stops flicker). Keys inside the window: `q` quits, `s`
saves the current raw frame to `captures/`.

| Option              | Purpose                                                         |
|---------------------|-----------------------------------------------------------------|
| `--device N`        | camera index (0 = built-in); omit to be asked                   |
| `--width`, `--height` | requested frame size (default 1280×720)                       |
| `--debug`           | draw the border, boxes, probabilities and, when invalid, the reason |
| `--no-window`       | headless: only print statuses                                   |
| `--snapshots DIR`   | where the `s` key saves frames (default `captures/`)            |
| `--model FILE`, `--device-torch cpu|cuda` | as for `image`                            |

### `gridcheck train` — train (or retrain) the model

```
uv run gridcheck train
uv run gridcheck train --epochs 20
```

All commands print colour-coded logs: a `gridcheck <command>` banner at start, cyan stage
headers, green ✔ ticks for completed steps, yellow ⚠ warnings, red ✖ errors, a green progress
bar with time remaining, and the validation accuracy in green (≥99%), yellow (≥95%) or red.
Verdicts are coloured too: `Full` red, `Available` green, `invalid field view` yellow.

Logs go to **stderr**; only the verdict itself goes to stdout, so `gridcheck image x.jpg |
some-script` still receives exactly `Full`, `Available` or `invalid field view`. Colours
switch off automatically when the output is not a terminal; set `NO_COLOR=1` to disable them
in the terminal too, or `FORCE_COLOR=1` to keep them when piping.

The one command that does everything, in this order:

1. **Harvest your photos.** Looks in `data/photos/full/` and `data/photos/available/`, cuts
   one crop per box from every photo, labels them, writes them to `data/real/`.
2. **Build the synthetic dataset** if `data/synth_cells.npz` does not exist yet
   (≈5 minutes the first time, then cached).
3. **Train CellNet** on synthetic + real crops and save `models/cellnet.pt`.

| Option            | Purpose                                                            |
|-------------------|--------------------------------------------------------------------|
| `--epochs N`      | passes over the data (default 10)                                  |
| `--batch-size N`  | crops per training step (default 128)                              |
| `--lr X`          | learning rate (default 0.001)                                      |
| `--seed N`        | random seed, for reproducible runs                                 |
| `--photos DIR`    | another folder with `full/` and `available/` inside                |
| `--no-photos`     | ignore photos, train on synthetic data only                        |
| `--data FILE`     | another synthetic dataset file                                     |
| `--out FILE`      | write the weights somewhere else                                   |
| `--device cpu|cuda` | force the PyTorch device                                         |

### `gridcheck synth` — (re)generate synthetic training data

```
uv run gridcheck synth --n 20000
uv run gridcheck synth --n 0 --samples preview/ --n-samples 12
```

Normally you don't need this: `train` builds the dataset automatically. Use it to rebuild with
a different size or seed, or to write a few full synthetic frames to a folder so you can see
what the generator produces.

| Option            | Purpose                                                  |
|-------------------|----------------------------------------------------------|
| `--n N`           | number of cell crops to collect (default 20000)          |
| `--seed N`        | random seed                                              |
| `--out FILE`      | output file (default `data/synth_cells.npz`)             |
| `--samples DIR`   | also save example frames here                            |
| `--n-samples N`   | how many example frames (default 12)                     |

### `gridcheck harvest <folder>` — cut crops from unsorted photos

```
uv run gridcheck harvest some_photos/ --sheets review/
```

For a folder of photos that are **not** yet sorted into full/available. Cuts every clean box
into a crop, guesses the label from the ink in its centre, writes the crops to `data/real/`,
and, with `--sheets`, writes one contact sheet image per label so you can eyeball the result
and delete wrong crops. (`train` does this automatically for the sorted folders, so most
people never need this command.)

### `gridcheck reset` — start over

```
uv run gridcheck reset                # interactive, see below
uv run gridcheck reset --dry-run      # list only, delete nothing
uv run gridcheck reset --photos       # delete photos too, without asking
uv run gridcheck reset --keep-photos  # keep photos, without asking
uv run gridcheck reset --yes          # no questions at all (photos are kept)
```

Interactive flow: it first asks *"you have N training photo(s) in data/photos. Remove them as
well? [y/N]"*. Answer `y` and the photos go too; answer `n` (or just press Enter) and they are
kept. Then it lists everything that will be deleted (trained model, synthetic dataset,
harvested crops, camera snapshots, and the photos if you said yes) and asks *"proceed? [y/N]"*.
Folders stay in place, so `train` can rebuild at once.

### Tests

```
uv run pytest
```

Runs 14 tests: the two reference sketches must detect as a 2×4 grid with the right filled
cells, synthetic boards must be detected ≥85% of the time, invalid frames must be rejected,
and the end-to-end verdicts must be right.

---

## 6. Setting up training with your own photos

1. Print or draw a sheet: a rectangle border with straight lines dividing it into boxes.
   Leave some white paper around the border. Fill some boxes with solid black circles.
2. Take photos with the webcam (`gridcheck cam`, press `s`) or a phone.
3. Sort them:
   - every box filled → `data/photos/full/`
   - at least one box empty → `data/photos/available/`
4. Run `uv run gridcheck train`.
5. Read the summary it prints:
   - `photos/available: 16 photos -> 24 circle, 57 empty crops` — how many crops it got
   - `no boxes found in: ...` — photos too blurry/tilted to use; retake or delete them
   - `WARNING every box looks filled` — an `available/` photo that probably belongs in `full/`
6. Watch the `val acc` (validation accuracy) per epoch. It should end near 100%.

What happens to your photos internally: for a `full/` photo every crop is labelled *circle*.
For an `available/` photo each box is labelled separately by how much dark ink is in its
centre (≥4% of the pixels → circle). Crops that aren't paper at all (a hand, the desk) are
thrown away. The crops are then **repeated** until they make up about 15% of the training
data, so a few dozen real photos still influence a model trained on 20 000 synthetic cells.

Photos where the border runs off the edge of the frame are still useful: the harvester
falls back to cutting every clean box it can find. But well-framed photos are better, because
then the boxes are cut exactly the way the live detector will cut them.

Tips for good photos: whole border inside the frame, some white margin around it, sheet
roughly facing the camera, fingers not covering the lines, even light.

---

## 7. Tools and libraries, and why each is used

| Tool / library | Version | What it is | Why we use it |
|---|---|---|---|
| **uv** | 0.12 | Python package and project manager | installs Python + dependencies reproducibly with one command; runs the `gridcheck` command |
| **Python** | 3.12 | the language | |
| **PyTorch** (`torch`) | 2.14 | deep-learning framework | defines and trains CellNet, runs it at inference time |
| **torchvision** | 0.29 | PyTorch's vision helpers | installed alongside torch; reserved for image transforms |
| **OpenCV** (`opencv-python`, imported as `cv2`) | 5.0 | classical computer-vision library | reading images and the camera, thresholding, contours, perspective warps, morphology, drawing the overlay |
| **NumPy** | 2.5 | fast arrays | every image is a NumPy array; profiles, masks and statistics are NumPy operations |
| **pytest** | 9.1 | test runner | runs `tests/` |
| **hatchling** | — | build backend | packages `src/gridcheck` so `uv` can install it and create the `gridcheck` command |

---

## 8. Where data is saved

| Data | File / folder | Format | Created by | In git? |
|---|---|---|---|---|
| Trained model | `models/cellnet.pt` | PyTorch checkpoint: the network's weights (`state_dict`), plus `labels` and `crop_size` | `gridcheck train` | **yes** |
| Synthetic training cells | `data/synth_cells.npz` | compressed NumPy archive with `X` (N×64×64 uint8 images) and `y` (N labels, 0 = empty, 1 = circle) | `gridcheck train` (first run) or `gridcheck synth` | no |
| Your photos | `data/photos/full/`, `data/photos/available/` | JPG/PNG as you saved them | you | no |
| Crops from your photos | `data/real/circle/*.png`, `data/real/empty/*.png` | 64×64 grayscale PNG, named `<folder>_<photo>_<box>.png` | `gridcheck train` / `harvest` | no |
| Camera snapshots | `captures/snap_<date>_<time>.jpg` | full camera frame | `s` key in `gridcheck cam` | no |
| Predictions | nowhere | printed to the terminal only | | |

There is **no database**. Everything is plain files, and everything except the model and your
photos can be regenerated. The `.gitkeep` files inside the empty folders exist only so git
keeps the folder structure.

Why is the model committed but the data not? The model is small (96 KB) and is what someone
cloning the repo needs to run the program. The data is large, private (your photos) or
regenerable (synthetic cells).

---

## 9. How the code works, module by module

### 9.1 `board.py` — finding the board (stage 1)

Entry point: `find_board(frame)` returns a `BoardResult` (corners, rows, cols, one 64×64
crop per box, the homography) or `None`. `find_board_explained` additionally returns a
one-line reason when nothing is found.

Steps:

1. **Grayscale + adaptive threshold.** The image is converted to grey and thresholded so dark
   ink becomes white and paper becomes black. "Adaptive" means each pixel is compared to the
   average of its neighbourhood, which makes it robust to shadows and uneven light.
2. **Candidate rectangles.** OpenCV finds contours (outlines of white regions). Each contour is
   simplified to a polygon; the ones with 4 corners that are convex and not touching the image
   edge become candidates. Up to the 12 largest are tried.
3. **Corner probe.** For each candidate, the code looks a little past each corner along the
   edge direction. If lines continue there, the candidate is just one box of a bigger grid,
   not the outer border, and is skipped.
4. **Perspective warp.** The candidate is grown a few pixels and mapped onto a flat 800-pixel
   wide canvas, as if seen from straight above. A generous margin is tried first (copes with
   wobbly hand-drawn borders), then a tight one (copes with borders drawn close to the paper
   edge).
5. **Grid ink only.** On the canvas the image is thresholded again; only the connected ink
   component that spans the canvas (the border with its lines) is kept — circles are separate
   blobs and are dropped. Circles that touch a line are removed with a square morphological
   opening (lines are thin in one direction, blobs are thick in both). Dark bands touching the
   canvas edge (the desk beyond the sheet) are stripped.
6. **Line profiles.** For horizontal lines, the ink is opened with a long horizontal kernel
   (keeps only horizontal strokes), smeared vertically (so slightly misaligned hand-drawn
   segments line up), and the fraction of ink per row is computed. Rows above 40% coverage
   form groups; each group must reach 70% coverage somewhere (a real line does, a blob does
   not) and be thin enough. The same is done for columns.
7. **Validation.** At least two lines in each direction, the outer ones near the canvas edge,
   cells of roughly uniform size, and the border ink clearly darker than the paper. Otherwise
   this candidate is rejected and the next one tried.
8. **Crops.** Each cell interior is cut with an 8% inner margin (drops line ink) and resized to
   64×64. When several candidates succeed, the one explaining the most cells wins.

Everything is controlled by named constants at the top of the file (thresholds, fractions),
each with a comment. Tuning means changing a constant and re-running `pytest`.

### 9.2 `model.py` — CellNet (stage 2)

```
input  1×64×64 grey image, values 0..1
  → Conv 3×3 (16 ch) → GroupNorm → ReLU → MaxPool 2   (32×32)
  → Conv 3×3 (32 ch) → GroupNorm → ReLU → MaxPool 2   (16×16)
  → Conv 3×3 (64 ch) → GroupNorm → ReLU → MaxPool 2   (8×8)
  → global average pool → Linear 64 → 2 scores (empty, circle)
```

23 538 parameters. Tiny by design: the task is easy once the crop is clean, and a tiny model
trains in minutes and runs in a millisecond. GroupNorm instead of BatchNorm means the network
behaves identically in training and inference even after very short training runs.

`save_model`/`load_model` write and read `models/cellnet.pt`. `preprocess` turns uint8 crops
into the float tensor the network expects.

### 9.3 `infer.py` — verdict (stage 3)

`Classifier.classify_frame(frame)`:
- calls `find_board_explained`; if nothing is found → `Verdict("invalid field view", reason=...)`
- runs all crops through CellNet in one batch, applies softmax, takes P(circle) ≥ 0.5 as filled
- all filled → `Full`, else `Available`

`draw_debug` draws the border (blue), each box (red = circle, green = empty) with its
probability, the status, and the invalid reason.

### 9.4 `synth.py` — synthetic boards

Real labelled photos are expensive; synthetic ones are free. `render_board` draws a random
grid (1–4 rows, 1–8 columns, random cell size, line thickness, slight hand-drawn jitter) on a
random paper tint, adds circles to random cells (with the labels known exactly), then makes it
look like a camera photo: random background, perspective, rotation, lighting ramp, shadow,
gamma, blur, sensor noise, JPEG compression. `render_invalid` produces frames with no usable
grid (clutter, a bare sheet, a board cut by the frame edge) for testing the "invalid" path.

### 9.5 `dataset.py` — building the training set

`build_synthetic_cells` renders boards and pushes each through the **same** `find_board`
used at run time. Only boards whose detected grid shape matches the truth are kept, and their
crops are paired with the known labels. This guarantees training crops look exactly like
inference crops. `load_real_cells` reads `data/real/`. `save_cells`/`load_cells` handle the
`.npz` file.

### 9.6 `train.py` — training

Harvests photos → loads synthetic + real crops (real ones repeated to ≈15%) → random 80/20
train/validation split → Adam optimiser, cosine learning-rate schedule, cross-entropy loss →
light on-the-fly augmentation (small shifts, flips, brightness/contrast jitter, noise) → after
each epoch the validation accuracy is printed and the best weights are kept → saved to
`models/cellnet.pt`.

### 9.7 `harvest.py` — crops from photos

`harvest_labeled` handles `data/photos/{full,available}`; `harvest` handles an unsorted
folder. Both try `find_board` first and fall back to "every clean quadrilateral of box size"
when the whole board can't be found. `auto_label` measures dark ink in the crop's centre;
`looks_like_paper` rejects crops of hands or desk.

### 9.8 `led.py` — Raspberry Pi status LEDs (version 2)

`StatusLeds` wraps two `gpiozero.LED` objects (BCM GPIO 17 = Available, GPIO 18 = Full).
`set_status(status)` maps the three verdict strings to the LEDs: Available → 17 on, Full → 18
on, anything else → both off. `close()` switches both off and releases the pins.
`NullLeds` has the same interface and does nothing, and `make_leds(mode, …)` picks between
them: `auto` (default) uses real LEDs when gpiozero and a GPIO driver are present, otherwise
prints a yellow note and continues without; `on` insists; `off` never touches GPIO.

`gridcheck cam` calls `set_status` at the exact moment the debounced status changes (the same
moment it prints), and `close` when it exits. `gridcheck led-test` lights each LED in turn to
verify the wiring without a camera. On any machine, `GPIOZERO_PIN_FACTORY=mock` simulates the
header. Wiring, pin map, Pi setup and troubleshooting: **[RASPBERRY_PI.md](RASPBERRY_PI.md)**.

**Feature flag.** All of this is gated by `raspberry_connected` in `gridcheck.toml`
(`src/gridcheck/config.py` reads it; the environment variable `GRIDCHECK_RASPBERRY_CONNECTED`
overrides the file). It is `false` in the repository. While it is off, `make_leds` never
imports gpiozero or touches GPIO: `auto` returns `NullLeds` with a yellow note, `on` raises,
so `led-test` and `cam --leds on` refuse to run. Turn it on only on the Pi.

### 9.9 `cli.py` and `reset.py`

`cli.py` is the only file that talks to the user: it defines the sub-commands, parses options
and calls the modules above. `reset.py` lists and deletes generated files.

---

## 10. Design decisions worth knowing

- **Output strings are exact.** `Full`, `Available`, `invalid field view` go to standard
  output and nothing else does, so scripts can rely on them.
- **A border touching the frame edge is invalid**, even if the sheet is otherwise fine. The
  program cannot know whether boxes are hidden outside the picture. Move the sheet back.
- **A single box of a bigger grid is not a board.** Otherwise the camera zoomed on one cell
  would answer `Full` or `Available` about that one cell.
- **A bare sheet is not a board.** The edge of the paper against a desk forms a rectangle
  too; it is rejected because there is no dark ink stroke with paper on both sides.
- **Real photos are repeated, not just added.** 80 real crops next to 20 000 synthetic ones
  would be ignored by the optimiser; repetition gives them weight.

---

## 11. Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| always `invalid field view` from the camera | border cut off by the frame edge, or sheet not in view | run with `--debug`; the overlay/terminal shows the reason. Hold the sheet further away so the whole border plus a white margin is visible |
| `no rectangular border found` | too dark, too blurry, or the border isn't a closed rectangle | more light, hold still, make sure the border is drawn all the way round |
| `... lines continue past the corners` | only part of a bigger grid is in view | move back |
| `... none contains a clean grid` | lines faint or broken, boxes very unequal, or border not dark enough | draw thicker, darker, straighter lines; use uniform boxes |
| a box is misclassified | model hasn't seen that kind of mark | take photos of it, sort them into `data/photos/`, run `gridcheck train` |
| `No trained weights at models/cellnet.pt` | model deleted (e.g. after `reset`) | run `gridcheck train` |
| `could not open camera 0` | camera in use or a different index | close other apps; try `--device 1` |
| tests in `test_infer.py` skipped | no model file | run `gridcheck train` |

---

## 12. Glossary

- **Adaptive threshold** — turning a grey image into black/white by comparing each pixel to
  its local neighbourhood instead of one global cut-off.
- **Contour** — the outline of a connected white region in a binary image.
- **Perspective warp / homography** — the mapping that makes a tilted rectangle look like it
  is seen from straight above.
- **Morphological opening** — erode then dilate; removes structures thinner than the kernel.
- **CNN** — convolutional neural network; learns small filters that detect local patterns.
- **Epoch** — one pass through the whole training set.
- **Validation accuracy** — accuracy on the 20% of data the model never trained on; the honest
  number.
- **Softmax** — turns the network's two raw scores into two probabilities that sum to 1.
- **Checkpoint / `state_dict`** — the saved weights of a PyTorch model.
- **Synthetic data** — training images generated by a program instead of photographed.
