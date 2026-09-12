"""Tiny in-place progress line for long-running steps (no extra dependencies)."""

from __future__ import annotations

import sys
import time


def fmt_secs(s: float) -> str:
    s = max(0, int(round(s)))
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


class Progress:
    """Prints `label  [#####-----]  42%  123/300  0m40s elapsed, ~0m55s left  <extra>` on one
    line, rewriting it in place. Call update() as often as you like (it rate-limits itself),
    then close() to move to the next line."""

    def __init__(self, total: float, label: str, width: int = 24, min_interval: float = 0.1) -> None:
        self.total = max(float(total), 1e-9)
        self.label = label
        self.width = width
        self.min_interval = min_interval
        self.t0 = time.time()
        self.last_draw = 0.0
        self.last_len = 0

    def update(self, done: float, extra: str = "", force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_draw < self.min_interval:
            return
        self.last_draw = now
        frac = min(done / self.total, 1.0)
        filled = int(round(frac * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = now - self.t0
        eta = (elapsed / frac - elapsed) if frac > 0.01 else float("nan")
        eta_txt = f"~{fmt_secs(eta)} left" if eta == eta else "estimating..."
        line = (f"{self.label}  [{bar}] {100 * frac:5.1f}%  {int(done)}/{int(self.total)}  "
                f"{fmt_secs(elapsed)} elapsed, {eta_txt}  {extra}")
        pad = " " * max(0, self.last_len - len(line))
        sys.stdout.write("\r" + line + pad)
        sys.stdout.flush()
        self.last_len = len(line)

    def close(self, final: str | None = None) -> None:
        if final is not None:
            pad = " " * max(0, self.last_len - len(final))
            sys.stdout.write("\r" + final + pad + "\n")
        else:
            sys.stdout.write("\n")
        sys.stdout.flush()


def stage(n: int, total: int, title: str) -> None:
    print(f"\n[{n}/{total}] {title}", flush=True)
