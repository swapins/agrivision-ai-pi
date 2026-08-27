#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo bash scripts/install_service.sh" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET="/etc/systemd/system/agrivision.service"
SERVICE_USER="${SUDO_USER:-$(stat -c '%U' "$ROOT") }"
SERVICE_USER="${SERVICE_USER// /}"

if ! id "$SERVICE_USER" >/dev/null 2>&1; then
  echo "Cannot resolve service user: $SERVICE_USER" >&2
  exit 1
fi

cat >"$TARGET" <<EOF
[Unit]
Description=AgriVision AI Smart Farm
After=network.target
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$ROOT
Environment=PYTHONPATH=$ROOT/src
ExecStart=$ROOT/scripts/run.sh
Restart=on-failure
RestartSec=5
TimeoutStopSec=15
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable agrivision.service
systemctl restart agrivision.service
systemctl --no-pager --full status agrivision.service || true

echo
printf '%s\n' \
  "Service installed at $TARGET" \
  "User: $SERVICE_USER" \
  "Working directory: $ROOT" \
  "Logs: journalctl -u agrivision.service -f" \
  "Stop: sudo systemctl stop agrivision.service"
