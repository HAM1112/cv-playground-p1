# gridcheck on a Raspberry Pi — wiring and setup (version 2)

Version 2 runs the same program on a Raspberry Pi and shows the result on two LEDs:

| Status the program prints | LED on GPIO 17 ("Available", green suggested) | LED on GPIO 18 ("Full", red suggested) |
|---------------------------|:---:|:---:|
| `Available`               | **ON** | off |
| `Full`                    | off | **ON** |
| `invalid field view`      | off | off |

Both LEDs are also switched off when the program exits (`q`, Ctrl-C or an error).

---

## 1. What you need

| Part | Notes |
|---|---|
| Raspberry Pi 4 or Pi 5 | Pi 3 / Zero 2 W also work, just slower (a few frames per second) |
| 64-bit Raspberry Pi OS (Bookworm or newer) | 64-bit is required for the PyTorch wheels |
| USB webcam | any UVC webcam; the Pi Camera Module (ribbon cable) is **not** covered here |
| 2 × LEDs | e.g. one green (Available) and one red (Full), standard 5 mm |
| 2 × resistors, 330 Ω | anything from 220 Ω to 1 kΩ is fine; **never connect an LED without one** |
| Jumper wires, optional breadboard | female-to-male wires if you go straight from the header |

---

## 2. Find the pins (BCM number vs physical pin)

The program uses **BCM GPIO numbers** (the chip's numbering, what gpiozero uses). The header
also has **physical pin numbers** (1–40, counting from the corner). Don't mix them up:

| Purpose | BCM name | Physical pin | Where it is |
|---|---|---|---|
| Available LED | **GPIO 17** | **pin 11** | 6th pin from the corner, inner row |
| Full LED | **GPIO 18** | **pin 12** | 6th pin from the corner, outer row (right next to pin 11) |
| Ground | GND | **pin 9** (or 14, 6, 20, 25 …) | pin 9 is directly above pin 11 |

Header excerpt (Pi seen from above, USB ports pointing down, header at the top-right).
Pin 1 is the corner pin nearest the SD card; odd pins are the inner row, even pins the outer row.

```
                inner row (odd)          outer row (even)
   pin  1  3V3                      pin  2  5V
   pin  3  GPIO 2 (SDA)             pin  4  5V
   pin  5  GPIO 3 (SCL)             pin  6  GND
   pin  7  GPIO 4                   pin  8  GPIO 14 (TXD)
   pin  9  GND   <── ground here    pin 10  GPIO 15 (RXD)
   pin 11  GPIO 17  <── Available   pin 12  GPIO 18  <── Full
   pin 13  GPIO 27                  pin 14  GND
   ...
```

Tip: `pinout` in a terminal on the Pi prints the full header with the numbers for your model.

---

## 3. Wiring

An LED has a long leg (anode, +) and a short leg (cathode, −; also the flat side of the rim).
Current flows from the GPIO pin, through the resistor and the LED, to ground.

```
  GPIO 17 (pin 11) ──[ 330 Ω ]──►|── GND (pin 9)        Available LED
  GPIO 18 (pin 12) ──[ 330 Ω ]──►|── GND (pin 9)        Full LED

  ►|  = LED, arrow points towards the short leg (cathode) which goes to GND
```

Step by step (power the Pi **off** while wiring):

1. Connect a jumper from **pin 9 (GND)** to the breadboard's ground rail.
2. **Available LED:** jumper from **pin 11 (GPIO 17)** → one end of a 330 Ω resistor →
   other end of the resistor → **long leg** of the green LED → **short leg** → ground rail.
3. **Full LED:** jumper from **pin 12 (GPIO 18)** → 330 Ω resistor → **long leg** of the red LED
   → **short leg** → ground rail.
4. Double-check that no wire touches the 5 V pins (2 and 4). GPIO pins are 3.3 V.

The resistor can be on either side of the LED; only the LED's direction matters. If an LED
never lights during the test in section 6, turn it around.

Why the resistor: a GPIO pin is 3.3 V and rated for a few milliamps. A bare LED would draw far
more, dim quickly and could damage the pin. 330 Ω gives roughly 4–5 mA, bright enough and safe.

---

## 4. Software setup on the Pi

Run these in a terminal on the Pi (or over SSH).

```bash
# 1. system packages: git, the GPIO driver used by gpiozero on Pi 4/5, OpenCV runtime libs
sudo apt update
sudo apt install -y git python3-lgpio python3-gpiozero libgl1 libglib2.0-0 v4l-utils

# 2. let your user use the GPIO header and the camera without sudo (log out/in afterwards)
sudo usermod -aG gpio,video $USER

# 3. install uv (Python project manager) and reload the shell so `uv` is on PATH
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 4. get the project (version 2 branch)
git clone https://github.com/HAM1112/cv-playground-p1.git
cd cv-playground-p1
git checkout version-2-raspberrypi

# 5. create the environment on the system Python so gpiozero can see apt's lgpio driver,
#    then install everything (PyTorch + OpenCV download ~300 MB the first time)
uv venv --system-site-packages --python /usr/bin/python3
uv sync --extra pi
```

The trained model `models/cellnet.pt` is in the repository, so nothing needs training on the
Pi. If you retrain on your laptop later, commit and pull the new file, or copy it over with
`scp`.

### 4.1 Switch the Raspberry Pi features on (feature flag)

All Pi/LED code is behind the `raspberry_connected` feature flag, which is **off** in the
repository so that nothing ever touches GPIO on a laptop. On the Pi, turn it on by editing
`gridcheck.toml` in the project folder:

```toml
[features]
raspberry_connected = true
```

or, for a single run, with an environment variable: `GRIDCHECK_RASPBERRY_CONNECTED=1 uv run gridcheck led-test`.

With the flag off, `gridcheck led-test` refuses to run, `cam --leds on` refuses to start, and
`cam` (default `--leds auto`) prints a yellow note and runs without LEDs.

---

## 5. Test the LEDs (no camera needed)

```bash
uv run gridcheck led-test
```

Expected: the GPIO 17 LED lights for 1.5 s, then the GPIO 18 LED, then both go off, with a
coloured line in the terminal for each step. `--seconds 3` slows it down, `--rounds 3` repeats.

If a step prints but the LED stays dark, that LED is reversed or on the wrong pin (section 7).

---

## 6. Run it

Headless Pi (SSH, no monitor):

```bash
uv run gridcheck cam --no-window --debug
```

Pi with a desktop and screen: drop `--no-window` to see the camera view with the overlay.

You should see `✔ LEDs: GPIO17 = Available, GPIO18 = Full` at start. Hold the sheet in front of
the webcam with the **whole border and some white margin visible**; the status prints on each
change and the matching LED lights. `q` (window) or Ctrl-C (terminal) quits and turns both off.

Options: `--device N` picks a camera index, `--leds off` runs without LEDs, `--leds on` refuses
to start unless the LEDs work, `--led-available 22 --led-full 23` moves the LEDs to other
pins.

---

## 7. Start automatically at boot (optional)

Create `/etc/systemd/system/gridcheck.service` (replace `pi` with your user name):

```ini
[Unit]
Description=gridcheck grid occupancy with LEDs
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/cv-playground-p1
ExecStart=/home/pi/.local/bin/uv run gridcheck cam --no-window --device 0
Restart=on-failure
RestartSec=5
Environment=NO_COLOR=1

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now gridcheck
journalctl -u gridcheck -f          # watch the log
```

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Raspberry Pi features are off (raspberry_connected = false)` | the feature flag is still off | set `raspberry_connected = true` in `gridcheck.toml` (section 4.1) |
| `LEDs disabled: no GPIO pin driver found: not a Raspberry Pi, or python3-lgpio is missing` | gpiozero found no GPIO driver | `sudo apt install python3-lgpio`; make sure the venv was created with `--system-site-packages` |
| `PermissionError` / `Permission denied: /dev/gpiochip*` | your user is not in the `gpio` group | `sudo usermod -aG gpio $USER`, log out and back in |
| `RuntimeError: LEDs requested but unavailable` with `--leds on` | as above | fix the driver, or drop `--leds on` |
| Works on Pi 4, nothing lights on Pi 5 | old `RPi.GPIO` driver on Pi 5 | this project uses gpiozero + lgpio, which is Pi 5 safe; install `python3-lgpio` |
| `led-test` prints steps but one LED stays dark | LED reversed, or wrong pin | swap the LED's legs; check GPIO 17 = physical 11, GPIO 18 = physical 12 |
| Both LEDs always dark, even in `led-test` | GND not connected | check the wire to pin 9 |
| `could not open camera 0` / `no camera found` | webcam not detected | `v4l2-ctl --list-devices`; try `--device 1`; use a powered hub for power-hungry webcams |
| `ImportError: libGL.so.1` | OpenCV runtime libs missing | `sudo apt install libgl1 libglib2.0-0` |
| Status flickers between values | poor light or camera shake | more light, fixed camera mount; the 5-frame vote smooths the rest |
| Slow (1–3 frames/s on a Pi 3) | CPU bound | use a Pi 4/5, or lower the camera size: `--width 640 --height 480` |
| Only `invalid field view` | border outside the frame | run with `--debug`; the reason is printed. Keep the whole border plus a white margin in view |

---

## 9. How the code does it

`src/gridcheck/led.py` has a small `StatusLeds` class around two `gpiozero.LED` objects with
one method, `set_status(status)`, that maps the three verdict strings to the LED table at the
top. `gridcheck cam` calls it exactly when the debounced status changes (the same moment the
status is printed) and calls `close()` on exit. `make_leds()` decides between real LEDs and a
do-nothing stand-in so the identical code runs on a laptop without GPIO. On any machine you can
run the LED logic with a simulated header:

```bash
GPIOZERO_PIN_FACTORY=mock uv run gridcheck led-test        # Linux / macOS
$env:GPIOZERO_PIN_FACTORY="mock"; uv run gridcheck led-test  # Windows PowerShell
```
