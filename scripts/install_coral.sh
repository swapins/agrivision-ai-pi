#!/usr/bin/env bash
set -euo pipefail

RUNTIME_PACKAGE="libedgetpu1-std"
if [[ "${1:-}" == "--max-frequency" ]]; then
  RUNTIME_PACKAGE="libedgetpu1-max"
  echo "Warning: max-frequency Edge TPU runtime can run hotter. Use cooling and stable power." >&2
elif [[ $# -gt 0 ]]; then
  echo "Usage: sudo bash scripts/install_coral.sh [--max-frequency]" >&2
  exit 2
fi

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo bash scripts/install_coral.sh" >&2
  exit 1
fi

python3 - <<'PY'
import sys
major, minor = sys.version_info[:2]
if major != 3 or minor < 6 or minor > 9:
    print(
        "Warning: Debian/Raspberry Pi PyCoral packages are usually happiest with Python 3.6-3.9.",
        file=sys.stderr,
    )
print(f"Python {major}.{minor}: checked")
PY

cat >/etc/apt/sources.list.d/coral-edgetpu.list <<'LIST'
deb https://packages.cloud.google.com/apt coral-edgetpu-stable main
LIST
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
apt update
apt install -y "$RUNTIME_PACKAGE" python3-pycoral

echo
printf '%s\n' \
  "Coral runtime installed: $RUNTIME_PACKAGE" \
  "Unplug/reconnect the USB Accelerator before testing." \
  "Verify: python3 scripts/coral_smoke_test.py path/to/leaf.jpg"
