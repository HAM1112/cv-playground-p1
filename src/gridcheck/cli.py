"""Command line interface.

    gridcheck image <path> [--debug] [--no-pad]
    gridcheck cam [--device 0] [--debug] [--no-window] [--leds auto|on|off]
    gridcheck led-test [--seconds 1.5]                      (Raspberry Pi, see RASPBERRY_PI.md)
    gridcheck synth [--n 20000] [--seed 0] [--out data/synth_cells.npz] [--samples DIR]
    gridcheck harvest <folder> [--sheets DIR]
    gridcheck train [--epochs 10] [--batch-size 128] [--lr 1e-3] [--device cpu|cuda]
    gridcheck reset [--dry-run] [--photos]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

from .infer import AVAILABLE, FULL, INVALID, Classifier, draw_debug
from .led import AVAILABLE_PIN, FULL_PIN, make_leds
from .progress import banner, c, fail, info, num, ok, path_text, prompt, status_text, warn

VOTE_WINDOW = 5


def cmd_image(args: argparse.Namespace) -> int:
    banner("image", str(args.path))
    frame = cv2.imread(str(args.path), cv2.IMREAD_COLOR)
    if frame is None:
        fail(f"could not read image: {args.path}")
        return 2
    clf = Classifier(args.model, args.device)
    verdict = clf.classify_frame(frame, pad_frac=0.0 if args.no_pad else 0.05)
    print(status_text(verdict.status), flush=True)
    if args.debug:
        if verdict.reason:
            warn(f"why invalid: {verdict.reason}")
        if verdict.board is not None:
            ok(f"grid {num(verdict.rows)}x{num(verdict.cols)}, "
               f"{num(verdict.n_filled)}/{num(verdict.n_cells)} filled")
            info("P(circle) per box:")
            for row in verdict.probs:
                print("    " + "  ".join(
                    c(f"{p:.2f}", "bright_red" if p >= 0.5 else "bright_green") for p in row),
                    file=sys.stderr)
        vis = draw_debug(frame, verdict)
        if args.save:
            cv2.imwrite(str(args.save), vis)
        else:
            cv2.imshow("gridcheck", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return 0


def _list_cameras(backend: int, max_index: int = 5) -> list[tuple[int, int, int]]:
    """Return (index, width, height) for every camera index that opens and yields a frame."""
    found = []
    try:  # probing unused indices makes OpenCV print warnings; keep the terminal clean
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        opened = cap.isOpened()
        frame = None
        if opened:
            opened, frame = cap.read()
        cap.release()
        if opened and frame is not None:
            found.append((i, frame.shape[1], frame.shape[0]))
    return found


def _choose_camera(backend: int) -> int | None:
    """Ask which camera to use when several are connected. Returns an index or None."""
    info("looking for cameras ...")
    cams = _list_cameras(backend)
    if not cams:
        fail("no camera found")
        return None
    if len(cams) == 1:
        idx, w, h = cams[0]
        ok(f"using the only camera found: index {num(idx)} ({w}x{h})")
        return idx
    ok(f"{num(len(cams))} cameras found:")
    for idx, w, h in cams:
        note = "(built-in / default)" if idx == 0 else "(external webcam?)"
        print(f"    {c(f'[{idx}]', 'bold', 'bright_cyan')} {w}x{h}  {c(note, 'dim')}", file=sys.stderr)
    default = cams[0][0]
    if prompt("Are you using an external webcam? [y/N]").lower() in {"y", "yes"}:
        others = [cam[0] for cam in cams if cam[0] != default]
        if len(others) == 1:
            return others[0]
        while True:
            raw = prompt(f"which camera index? {others}:")
            if raw.isdigit() and int(raw) in others:
                return int(raw)
            warn("please type one of the listed indices")
    return default


def cmd_cam(args: argparse.Namespace) -> int:
    banner("cam", "live camera")
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    device = args.device
    if device is None:
        device = _choose_camera(backend)
        if device is None:
            return 2
    cap = cv2.VideoCapture(device, backend)
    if not cap.isOpened():
        fail(f"could not open camera {device}")
        return 2
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    clf = Classifier(args.model, args.device_torch)
    votes: deque[str] = deque(maxlen=VOTE_WINDOW)
    last_printed: str | None = None
    last_reason: str | None = None
    snap_dir = Path(args.snapshots)
    ok(f"camera {num(device)} open; model loaded on {clf.device}")
    leds = make_leds(args.leds, args.led_available, args.led_full, warn=warn)
    if leds.enabled:
        ok(f"LEDs: GPIO{num(leds.available_pin)} = {status_text(AVAILABLE)}, "
           f"GPIO{num(leds.full_pin)} = {status_text(FULL)}, both off when invalid")
    info(f"keys: {c('q', 'bold')} = quit, {c('s', 'bold')} = save a snapshot of the current frame")
    info("status is printed whenever it changes:")
    try:
        while True:
            got, frame = cap.read()
            if not got:
                fail("camera read failed")
                break
            verdict = clf.classify_frame(frame)
            votes.append(verdict.status)
            if len(votes) == VOTE_WINDOW:
                stable = Counter(votes).most_common(1)[0][0]
                if stable != last_printed:
                    stamp = c(time.strftime("%H:%M:%S"), "dim")
                    print(f"{stamp}  {status_text(stable)}", flush=True)
                    leds.set_status(stable)
                    last_printed = stable
            if args.debug and verdict.reason and verdict.reason != last_reason:
                warn(f"why invalid: {verdict.reason}")
                last_reason = verdict.reason
            if not args.no_window:
                vis = draw_debug(frame, verdict) if args.debug else frame
                cv2.imshow("gridcheck", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    snap_dir.mkdir(parents=True, exist_ok=True)
                    path = snap_dir / f"snap_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(str(path), frame)
                    ok(f"saved {path_text(path)}")
    finally:
        leds.close()          # both LEDs off on q, Ctrl-C or error
        cap.release()
        cv2.destroyAllWindows()
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    from .dataset import DEFAULT_CACHE, build_synthetic_cells, save_cells
    from .synth import render_board

    banner("synth", "synthetic training data")
    if args.samples:
        out = Path(args.samples)
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(args.seed)
        for i in range(args.n_samples):
            frame, labels, _ = render_board(rng)
            cv2.imwrite(str(out / f"board_{i:03d}_{labels.shape[0]}x{labels.shape[1]}_{int(labels.sum())}filled.png"), frame)
        ok(f"wrote {num(args.n_samples)} sample frames to {path_text(out)}")
        if args.n == 0:
            return 0

    info(f"generating ~{args.n} synthetic cells (seed {args.seed}) ...")
    X, y, stats = build_synthetic_cells(args.n, seed=args.seed)
    save_cells(args.out or DEFAULT_CACHE, X, y)
    ok(f"boards rendered: {num(stats['boards'])}, detected: {num(stats['detected'])} "
       f"({100 * stats['detect_rate']:.1f}%), cells: {num(stats['cells'])}, "
       f"circle fraction: {stats['circle_frac']:.2f}, {stats['seconds']:.0f}s")
    ok(f"saved to {path_text(args.out or DEFAULT_CACHE)}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .dataset import DEFAULT_CACHE
    from .model import DEFAULT_MODEL_PATH
    from .train import train

    from .harvest import DEFAULT_PHOTOS_DIR

    banner("train", "photos -> synthetic data -> CellNet")
    train(
        data_path=args.data or DEFAULT_CACHE,
        out_path=args.out or DEFAULT_MODEL_PATH,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        photos_dir=None if args.no_photos else (args.photos or DEFAULT_PHOTOS_DIR),
    )
    return 0


def cmd_harvest(args: argparse.Namespace) -> int:
    from .dataset import DEFAULT_REAL_DIR
    from .harvest import harvest

    banner("harvest", str(args.image_dir))
    stats = harvest(args.image_dir, args.out or DEFAULT_REAL_DIR, args.sheets)
    ok(f"images: {num(stats['images'])}, circle crops: {c(str(stats['circle']), 'bright_red')}, "
       f"empty crops: {c(str(stats['empty']), 'bright_green')}")
    ok(f"saved under {path_text(args.out or DEFAULT_REAL_DIR)}"
       + (f", contact sheets in {path_text(args.sheets)}" if args.sheets else ""))
    return 0


def cmd_led_test(args: argparse.Namespace) -> int:
    """Light each LED in turn so the wiring can be checked without camera or model."""
    banner("led-test", f"GPIO{args.led_available} = Available, GPIO{args.led_full} = Full")
    leds = make_leds("on", args.led_available, args.led_full)
    steps = [
        (AVAILABLE, f"GPIO{args.led_available} on  (the {status_text(AVAILABLE)} LED)"),
        (FULL, f"GPIO{args.led_full} on  (the {status_text(FULL)} LED)"),
        (INVALID, "both off  (what an invalid view looks like)"),
    ]
    try:
        for _round in range(args.rounds):
            for status, text in steps:
                leds.set_status(status)
                ok(text)
                time.sleep(args.seconds)
        ok("LED test finished; both LEDs are off")
    finally:
        leds.close()
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    from .reset import plan_reset, run_reset

    banner("reset", "back to a fresh project")

    def ask(question: str) -> bool:
        return prompt(question).lower() in {"y", "yes"}

    # Decide about the photos first: flags win, otherwise ask.
    n_photos = len(plan_reset(include_photos=True).get("your training photos", []))
    if args.photos:
        include_photos = True
    elif args.keep_photos or args.dry_run or args.yes or n_photos == 0:
        include_photos = False
    else:
        include_photos = ask(
            f"you have {n_photos} training photo(s) in data/photos. Remove them as well? [y/N]"
        )

    plan = plan_reset(include_photos=include_photos)
    total = sum(len(v) for v in plan.values())
    if total == 0:
        ok("nothing to remove: the project is already fresh")
        return 0
    warn("this will delete:")
    for kind, paths in plan.items():
        if paths:
            shown = ", ".join(p.name for p in paths[:3]) + (", ..." if len(paths) > 3 else "")
            print(f"    {c('-', 'bright_red')} {c(kind, 'bold')}: {num(len(paths))} file(s)  {c('(' + shown + ')', 'dim')}",
                  file=sys.stderr)
    if not include_photos and n_photos:
        ok(f"your {num(n_photos)} photo(s) in data/photos are kept")
    if args.dry_run:
        info("dry run, nothing deleted")
        return 0
    if not args.yes:
        if prompt("proceed? [y/N]").lower() not in {"y", "yes"}:
            info("cancelled")
            return 1
    n = run_reset(plan)
    ok(f"removed {num(n)} file(s). Run {c('uv run gridcheck train', 'bold')} to build everything again.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gridcheck", description="Grid occupancy checker.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("image", help="classify a single image file")
    pi.add_argument("path")
    pi.add_argument("--debug", action="store_true", help="show / save an overlay and per-cell probabilities")
    pi.add_argument("--save", help="with --debug: write the overlay to this file instead of showing it")
    pi.add_argument("--no-pad", action="store_true", help="do not pad the image (treat it like a camera frame)")
    pi.add_argument("--model", help="path to cellnet.pt")
    pi.add_argument("--device", help="torch device (cpu / cuda)")
    pi.set_defaults(func=cmd_image)

    pc = sub.add_parser("cam", help="classify a live webcam feed")
    pc.add_argument("--device", type=int, default=None,
                    help="camera index; if omitted, connected cameras are listed and you are asked which to use")
    pc.add_argument("--width", type=int, default=1280)
    pc.add_argument("--height", type=int, default=720)
    pc.add_argument("--debug", action="store_true", help="draw the detection overlay")
    pc.add_argument("--no-window", action="store_true", help="headless: print status only")
    pc.add_argument("--snapshots", default="captures", help="folder for frames saved with the s key")
    pc.add_argument("--leds", choices=["auto", "on", "off"], default="auto",
                    help="drive the Raspberry Pi status LEDs: auto = if GPIO is available (default)")
    pc.add_argument("--led-available", type=int, default=AVAILABLE_PIN, help="BCM GPIO of the Available LED (default 17)")
    pc.add_argument("--led-full", type=int, default=FULL_PIN, help="BCM GPIO of the Full LED (default 18)")
    pc.add_argument("--model", help="path to cellnet.pt")
    pc.add_argument("--device-torch", dest="device_torch", help="torch device (cpu / cuda)")
    pc.set_defaults(func=cmd_cam)

    ps = sub.add_parser("synth", help="generate the synthetic cell dataset")
    ps.add_argument("--n", type=int, default=20000, help="number of cell crops to collect")
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--out", help="output .npz (default data/synth_cells.npz)")
    ps.add_argument("--samples", help="also write a few full synthetic frames to this directory")
    ps.add_argument("--n-samples", type=int, default=12)
    ps.set_defaults(func=cmd_synth)

    ph = sub.add_parser("harvest", help="cut and auto-label cell crops from a folder of photos")
    ph.add_argument("image_dir")
    ph.add_argument("--out", help="output root (default data/real)")
    ph.add_argument("--sheets", help="write contact sheets for review to this directory")
    ph.set_defaults(func=cmd_harvest)

    pl = sub.add_parser("led-test", help="Raspberry Pi: light each status LED in turn to check the wiring")
    pl.add_argument("--seconds", type=float, default=1.5, help="how long each step stays lit (default 1.5)")
    pl.add_argument("--rounds", type=int, default=1, help="repeat the sequence this many times")
    pl.add_argument("--led-available", type=int, default=AVAILABLE_PIN, help="BCM GPIO of the Available LED (default 17)")
    pl.add_argument("--led-full", type=int, default=FULL_PIN, help="BCM GPIO of the Full LED (default 18)")
    pl.set_defaults(func=cmd_led_test)

    pr = sub.add_parser("reset", help="delete the trained model and all generated data (fresh project)")
    pr.add_argument("--photos", action="store_true", help="also delete your photos in data/photos (no question asked)")
    pr.add_argument("--keep-photos", action="store_true", help="keep your photos (no question asked)")
    pr.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation (photos are kept)")
    pr.add_argument("--dry-run", action="store_true", help="only show what would be deleted")
    pr.set_defaults(func=cmd_reset)

    pt = sub.add_parser("train", help="train the cell classifier")
    pt.add_argument("--epochs", type=int, default=10)
    pt.add_argument("--batch-size", type=int, default=128)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--data", help="input .npz (default data/synth_cells.npz)")
    pt.add_argument("--out", help="output weights (default models/cellnet.pt)")
    pt.add_argument("--device", help="torch device (cpu / cuda)")
    pt.add_argument("--photos", help="folder with full/ and available/ photo subfolders (default data/photos)")
    pt.add_argument("--no-photos", action="store_true", help="train on synthetic data only")
    pt.set_defaults(func=cmd_train)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, RuntimeError) as e:
        fail(str(e))
        return 2
    except KeyboardInterrupt:
        print()
        info("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
