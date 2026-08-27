#!/usr/bin/env bash
set -euo pipefail

SERVICE="scripts/systemd/agrivision.service"
TARGET="/etc/systemd/system/agrivision.service"

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo bash scripts/install_service.sh" >&2
  exit 1
fi

if [[ ! -f "$SERVICE" ]]; then
  echo "Missing service template: $SERVICE" >&2
  exit 1
fi

cp "$SERVICE" "$TARGET"
systemctl daemon-reload
systemctl enable agrivision.service
systemctl restart agrivision.service
systemctl --no-pager --full status agrivision.service || true

echo
printf '%s\n' \
  "Service installed at $TARGET" \
  "Logs: journalctl -u agrivision.service -f" \
  "Stop: sudo systemctl stop agrivision.service"
