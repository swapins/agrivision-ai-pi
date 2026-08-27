#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -x ".venv/bin/python" ]]; then
  exec .venv/bin/python -m agrivision.app
fi
exec python3 -m agrivision.app
