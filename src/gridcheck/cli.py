"""Command line interface.

    gridcheck image <path> [--debug] [--no-pad]
    gridcheck cam [--device 0] [--debug] [--no-window]
    gridcheck synth [--n 20000] [--seed 0] [--out data/synth_cells.npz] [--samples DIR]
    gridcheck train [--epochs 10] [--batch-size 128] [--lr 1e-3] [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

from .infer import INVALID, Classifier, draw_debug

VOTE_WINDOW = 5


def cmd_image(args: argparse.Namespace) -> int:
    frame = cv2.imread(str(args.path), cv2.IMREAD_COLOR)
    if frame is None:
        print(f"could not read image: {args.path}", file=sys.stderr)
        return 2
    clf = Classifier(args.model, args.device)
    verdict = clf.classify_frame(frame, pad_frac=0.0 if args.no_pad else 0.05)
    print(verdict.status)
    if args.debug:
        if verdict.board is not None:
            print(f"  grid {verdict.rows}x{verdict.cols}, {verdict.n_filled}/{verdict.n_cells} filled",
                  file=sys.stderr)
            print("  P(circle):\n" + np.array2string(verdict.probs, precision=2, suppress_small=True),
                  file=sys.stderr)
        vis = draw_debug(frame, verdict)
        if args.save:
            cv2.imwrite(str(args.save), vis)
        else:
            cv2.imshow("gridcheck", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return 0


def cmd_cam(args: argparse.Namespace) -> int:
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.device, backend)
    if not cap.isOpened():
        print(f"could not open camera {args.device}", file=sys.stderr)
        return 2
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    clf = Classifier(args.model, args.device_torch)
    votes: deque[str] = deque(maxlen=VOTE_WINDOW)
    last_printed: str | None = None
    print("press q to quit", file=sys.stderr)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed", file=sys.stderr)
                break
            verdict = clf.classify_frame(frame)
            votes.append(verdict.status)
            if len(votes) == VOTE_WINDOW:
                stable = Counter(votes).most_common(1)[0][0]
                if stable != last_printed:
                    print(stable, flush=True)
                    last_printed = stable
            if not args.no_window:
                vis = draw_debug(frame, verdict) if args.debug else frame
                cv2.imshow("gridcheck", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    from .dataset import DEFAULT_CACHE, build_synthetic_cells, save_cells
    from .synth import render_board

    if args.samples:
        out = Path(args.samples)
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(args.seed)
        for i in range(args.n_samples):
            frame, labels, _ = render_board(rng)
            cv2.imwrite(str(out / f"board_{i:03d}_{labels.shape[0]}x{labels.shape[1]}_{int(labels.sum())}filled.png"), frame)
        print(f"wrote {args.n_samples} sample frames to {out}")
        if args.n == 0:
            return 0

    print(f"Generating ~{args.n} synthetic cells (seed {args.seed}) ...")
    X, y, stats = build_synthetic_cells(args.n, seed=args.seed)
    save_cells(args.out or DEFAULT_CACHE, X, y)
    print(f"boards rendered: {stats['boards']}, detected: {stats['detected']} "
          f"({100 * stats['detect_rate']:.1f}%), cells: {stats['cells']}, "
          f"circle fraction: {stats['circle_frac']:.2f}, {stats['seconds']:.0f}s")
    print(f"saved to {args.out or DEFAULT_CACHE}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .dataset import DEFAULT_CACHE
    from .model import DEFAULT_MODEL_PATH
    from .train import train

    train(
        data_path=args.data or DEFAULT_CACHE,
        out_path=args.out or DEFAULT_MODEL_PATH,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )
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
    pc.add_argument("--device", type=int, default=0, help="camera index")
    pc.add_argument("--width", type=int, default=1280)
    pc.add_argument("--height", type=int, default=720)
    pc.add_argument("--debug", action="store_true", help="draw the detection overlay")
    pc.add_argument("--no-window", action="store_true", help="headless: print status only")
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

    pt = sub.add_parser("train", help="train the cell classifier")
    pt.add_argument("--epochs", type=int, default=10)
    pt.add_argument("--batch-size", type=int, default=128)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--data", help="input .npz (default data/synth_cells.npz)")
    pt.add_argument("--out", help="output weights (default models/cellnet.pt)")
    pt.add_argument("--device", help="torch device (cpu / cuda)")
    pt.set_defaults(func=cmd_train)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
