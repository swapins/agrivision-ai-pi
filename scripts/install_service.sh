#!/usr/bin/env bash
set -euo pipefail
SERVICE=scripts/systemd/agrivision.service
if [[ $EUID -ne 0 ]]; then echo "Run with sudo"; exit 1; fi
cp "$SERVICE" /etc/systemd/system/agrivision.service
systemctl daemon-reload
systemctl enable agrivision.service
systemctl restart agrivision.service
systemctl --no-pager status agrivision.service
