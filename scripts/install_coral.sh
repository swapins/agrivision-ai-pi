#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo bash scripts/install_coral.sh" >&2
  exit 1
fi

cat >/etc/apt/sources.list.d/coral-edgetpu.list <<'LIST'
deb https://packages.cloud.google.com/apt coral-edgetpu-stable main
LIST
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
apt update
apt install -y libedgetpu1-std python3-pycoral

echo "Coral standard-frequency runtime installed. Unplug/reconnect the USB Accelerator before testing."
