#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo bash scripts/install_pi.sh" >&2
  exit 1
fi

if ! grep -qi raspberry /proc/device-tree/model 2>/dev/null; then
  echo "Warning: this does not look like a Raspberry Pi. Continue only for packaging/testing." >&2
fi

python3 - <<'PY'
import sys
major, minor = sys.version_info[:2]
if major != 3 or minor < 9:
    raise SystemExit("Python 3.9+ is required")
print(f"Python {major}.{minor}: OK")
PY

apt update
apt install -y \
  git curl python3-pip python3-venv python3-picamera2 python3-opencv \
  python3-flask python3-pil python3-gpiozero i2c-tools \
  python3-smbus

# Install project Python dependencies without replacing OS-managed camera packages.
python3 -m pip install -r requirements-pi.txt
python3 -m pip install adafruit-circuitpython-ads1x15 adafruit-circuitpython-bme280 PyYAML
python3 -m pip install -e .

mkdir -p captures models
chmod 755 captures models

echo
printf '%s\n' \
  "Pi base dependencies installed." \
  "Next: sudo raspi-config  -> enable I2C and camera if needed." \
  "Check I2C: i2cdetect -y 1" \
  "Fetch model files: python3 scripts/fetch_models.py" \
  "Run without pump pulse: python3 scripts/hardware_selftest.py" \
  "Run pump pulse only after safety check: python3 scripts/hardware_selftest.py --pump"
