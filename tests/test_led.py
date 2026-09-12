import pytest

gpiozero = pytest.importorskip("gpiozero")

from gpiozero import Device  # noqa: E402
from gpiozero.pins.mock import MockFactory  # noqa: E402

from gridcheck.infer import AVAILABLE, FULL, INVALID  # noqa: E402
from gridcheck.led import AVAILABLE_PIN, FULL_PIN, NullLeds, StatusLeds, make_leds  # noqa: E402


@pytest.fixture
def mock_pins():
    factory = MockFactory()
    Device.pin_factory = factory
    yield factory
    factory.reset()
    Device.pin_factory = None


def _levels(factory):
    return factory.pin(AVAILABLE_PIN).state, factory.pin(FULL_PIN).state


def test_status_mapping(mock_pins):
    with StatusLeds() as leds:
        assert _levels(mock_pins) == (0, 0)
        leds.set_status(AVAILABLE)
        assert _levels(mock_pins) == (1, 0)
        leds.set_status(FULL)
        assert _levels(mock_pins) == (0, 1)
        leds.set_status(INVALID)
        assert _levels(mock_pins) == (0, 0)
        leds.set_status("something unexpected")
        assert _levels(mock_pins) == (0, 0)


def test_close_turns_everything_off(mock_pins):
    leds = StatusLeds()
    leds.set_status(FULL)
    assert _levels(mock_pins) == (0, 1)
    leds.close()
    assert _levels(mock_pins) == (0, 0)


def test_custom_pins(mock_pins):
    with StatusLeds(available_pin=5, full_pin=6) as leds:
        leds.set_status(AVAILABLE)
        assert (mock_pins.pin(5).state, mock_pins.pin(6).state) == (1, 0)


def test_make_leds_modes(mock_pins):
    assert isinstance(make_leds("off"), NullLeds) and not make_leds("off").enabled
    leds = make_leds("auto")
    assert isinstance(leds, StatusLeds) and leds.enabled
    leds.close()
    leds = make_leds("on")
    assert leds.enabled
    leds.close()


def test_make_leds_falls_back_without_hardware(monkeypatch):
    import gridcheck.led as led_mod

    class Boom(StatusLeds):
        def __init__(self, *a, **k):
            raise RuntimeError("Unable to load any default pin factory!")

    monkeypatch.setattr(led_mod, "StatusLeds", Boom)
    warnings = []
    leds = led_mod.make_leds("auto", warn=warnings.append)
    assert isinstance(leds, NullLeds) and not leds.enabled
    assert warnings and "LEDs disabled" in warnings[0]
    with pytest.raises(RuntimeError):
        led_mod.make_leds("on")


def test_null_leds_track_state_without_hardware():
    leds = NullLeds()
    leds.set_status(FULL)
    assert leds.state == (False, True)
    leds.close()
    assert leds.state == (False, False)
