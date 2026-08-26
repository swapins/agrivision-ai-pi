#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo bash scripts/install_pi.sh" >&2
  exit 1
fi

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
printf '%s\n' "Pi base dependencies installed." \
  "Next: sudo raspi-config  -> enable I2C if needed." \
  "Then run: i2cdetect -y 1"
