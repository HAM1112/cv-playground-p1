"""Status LEDs on the Raspberry Pi GPIO header (version 2).

    Available          -> GPIO 17 LED on,  GPIO 18 LED off
    Full               -> GPIO 17 LED off, GPIO 18 LED on
    invalid field view -> both off

Built on gpiozero, which works on Pi 4 and Pi 5 and has a mock pin factory so this module
can be exercised on any machine:  GPIOZERO_PIN_FACTORY=mock uv run gridcheck led-test
Wiring and setup: RASPBERRY_PI.md.
"""

from __future__ import annotations

from typing import Callable

from .infer import AVAILABLE, FULL

AVAILABLE_PIN = 17
FULL_PIN = 18

# status -> (available LED on?, full LED on?); anything else (invalid view) -> both off
_STATES: dict[str, tuple[bool, bool]] = {
    AVAILABLE: (True, False),
    FULL: (False, True),
}


class NullLeds:
    """Same interface as StatusLeds but drives nothing (no gpiozero, not a Pi, or --leds off)."""

    enabled = False

    def __init__(self, available_pin: int = AVAILABLE_PIN, full_pin: int = FULL_PIN) -> None:
        self.available_pin, self.full_pin = available_pin, full_pin
        self.state: tuple[bool, bool] = (False, False)

    def set_status(self, status: str) -> None:
        self.state = _STATES.get(status, (False, False))

    def all_off(self) -> None:
        self.state = (False, False)

    def close(self) -> None:
        self.all_off()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class StatusLeds(NullLeds):
    """Real LEDs via gpiozero. Raises if gpiozero or a GPIO pin factory is unavailable."""

    enabled = True

    def __init__(self, available_pin: int = AVAILABLE_PIN, full_pin: int = FULL_PIN) -> None:
        super().__init__(available_pin, full_pin)
        from gpiozero import LED  # imported here so the rest of gridcheck works without it

        self._available = LED(available_pin)
        self._full = LED(full_pin)
        self.all_off()

    def set_status(self, status: str) -> None:
        super().set_status(status)
        on_available, on_full = self.state
        (self._available.on if on_available else self._available.off)()
        (self._full.on if on_full else self._full.off)()

    def all_off(self) -> None:
        super().all_off()
        self._available.off()
        self._full.off()

    def close(self) -> None:
        try:
            self.all_off()
        finally:
            self._available.close()
            self._full.close()


def make_leds(
    mode: str = "auto",
    available_pin: int = AVAILABLE_PIN,
    full_pin: int = FULL_PIN,
    warn: Callable[[str], None] | None = None,
) -> NullLeds:
    """Build the LED driver.

    mode "off"  -> NullLeds.
    mode "on"   -> StatusLeds; raises RuntimeError with a clear message if impossible.
    mode "auto" -> StatusLeds if gpiozero and a GPIO pin factory are available, otherwise
                   NullLeds after calling warn(reason).

    Everything hardware-related sits behind the `raspberry_connected` feature flag
    (gridcheck.toml). With the flag off, "auto" yields NullLeds without touching GPIO and
    "on" raises, whatever the machine.
    """
    if mode == "off":
        return NullLeds(available_pin, full_pin)
    from .config import feature, how_to_enable

    if not feature("raspberry_connected"):
        msg = f"Raspberry Pi features are off (raspberry_connected = false); {how_to_enable('raspberry_connected')}"
        if mode == "on":
            raise RuntimeError(msg)
        if warn:
            warn(msg)
        return NullLeds(available_pin, full_pin)
    try:
        import gpiozero  # noqa: F401
    except ImportError:
        reason = "gpiozero is not installed (uv sync --extra pi)"
    else:
        # gpiozero tries several pin drivers and warns about each one it skips; that is
        # normal on a laptop, so keep the terminal quiet and report only the outcome.
        import warnings

        from gpiozero.exc import PinFactoryFallback

        warnings.filterwarnings("ignore", category=PinFactoryFallback)
        try:
            return StatusLeds(available_pin, full_pin)
        except Exception as e:  # BadPinFactory (subclasses ImportError!), PinInvalidPin, permissions ...
            reason = f"{type(e).__name__}: {e}".strip()
            if type(e).__name__ == "BadPinFactory":
                reason = "no GPIO pin driver found: not a Raspberry Pi, or python3-lgpio is missing"
    if mode == "on":
        raise RuntimeError(f"LEDs requested but unavailable: {reason}")
    if warn:
        warn(f"LEDs disabled: {reason}")
    return NullLeds(available_pin, full_pin)
