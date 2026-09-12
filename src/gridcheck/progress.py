"""Colourful in-place progress lines for long-running steps (no extra dependencies).

Colours are plain ANSI escape codes. They are enabled on Windows terminals via the
console's virtual-terminal mode and switched off automatically when stdout is not a
terminal (e.g. redirected to a file) or when the NO_COLOR environment variable is set.
"""

from __future__ import annotations

import os
import sys
import time


def _enable_windows_ansi() -> bool:
    if sys.platform != "win32":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:  # pragma: no cover - best effort only
        return False


USE_COLOR = bool(os.environ.get("FORCE_COLOR")) or (
    sys.stdout.isatty() and not os.environ.get("NO_COLOR") and _enable_windows_ansi()
)

_CODES = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m", "white": "\033[37m",
    "bright_green": "\033[92m", "bright_yellow": "\033[93m", "bright_cyan": "\033[96m",
}


def _sym(unicode_char: str, ascii_fallback: str) -> str:
    """Use the nice symbol only if stdout can encode it (a redirected file may be cp1252)."""
    try:
        unicode_char.encode(sys.stdout.encoding or "ascii")
        return unicode_char
    except (UnicodeEncodeError, LookupError):
        return ascii_fallback


TICK = _sym("✔", "+")
WARN = _sym("⚠", "!")
CROSS = _sym("✖", "x")


def c(text: str, *styles: str) -> str:
    """Wrap text in the given styles (names from _CODES) when colour is enabled."""
    if not USE_COLOR or not styles:
        return text
    return "".join(_CODES[s] for s in styles) + text + _CODES["reset"]


def fmt_secs(s: float) -> str:
    s = max(0, int(round(s)))
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def stage(n: int, total: int, title: str) -> None:
    print("\n" + c(f"[{n}/{total}]", "bold", "bright_cyan") + " " + c(title, "bold"), flush=True)


def info(text: str) -> None:
    print("  " + c(text, "dim"), flush=True)


def ok(text: str) -> None:
    print("  " + c(TICK + " ", "bright_green") + text, flush=True)


def warn(text: str) -> None:
    print("  " + c(WARN + " " + text, "bright_yellow"), flush=True)


def fail(text: str) -> None:
    print("  " + c(CROSS + " " + text, "red"), flush=True)


class Progress:
    """`label  [#####-----]  42%  123/300  0m40s elapsed, ~0m55s left  <extra>` on one line,
    rewritten in place. Call update() as often as you like (it rate-limits itself), then
    close() to leave a final line and move on."""

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
        bar = c("#" * filled, "bright_green") + c("-" * (self.width - filled), "dim")
        elapsed = now - self.t0
        eta = (elapsed / frac - elapsed) if frac > 0.01 else float("nan")
        eta_txt = f"~{fmt_secs(eta)} left" if eta == eta else "estimating..."
        plain_len = (len(self.label) + 2 + self.width + 2 + 8 + 2 + len(f"{int(done)}/{int(self.total)}")
                     + 2 + len(f"{fmt_secs(elapsed)} elapsed, {eta_txt}") + 2 + len(extra))
        line = (f"{c(self.label, 'bold')}  [{bar}] {c(f'{100 * frac:5.1f}%', 'bright_yellow')}  "
                f"{int(done)}/{int(self.total)}  {c(f'{fmt_secs(elapsed)} elapsed, {eta_txt}', 'dim')}  "
                f"{c(extra, 'cyan')}")
        pad = " " * max(0, self.last_len - plain_len)
        sys.stdout.write("\r" + line + pad)
        sys.stdout.flush()
        self.last_len = plain_len

    def close(self, final: str | None = None, plain_len: int | None = None) -> None:
        if final is not None:
            n = plain_len if plain_len is not None else len(final)
            pad = " " * max(0, self.last_len - n)
            sys.stdout.write("\r" + final + pad + "\n")
        else:
            sys.stdout.write("\n")
        sys.stdout.flush()
