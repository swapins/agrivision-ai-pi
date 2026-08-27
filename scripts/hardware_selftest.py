#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agrivision.config import load_config, validate_config
from agrivision.hardware import make_hardware
from agrivision.logic import HealthStatus


def simulation_enabled() -> bool:
    return os.environ.get("AGRIVISION_SIMULATION", "0").lower() in {"1", "true", "yes"}


def main() -> int:
    parser = argparse.ArgumentParser()
    config_default = ROOT / ("config.yaml" if (ROOT / "config.yaml").exists() else "config.example.yaml")
    parser.add_argument("--config", default=str(config_default))
    parser.add_argument("--pump", action="store_true", help="Pulse the pump for one second after explicit physical safety check")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sim = simulation_enabled()
    validate_config(cfg, simulation=sim)
    hw = make_hardware(cfg, sim)
    try:
        print("Environment:", hw.read_environment())
        for status in (HealthStatus.HEALTHY, HealthStatus.STRESS, HealthStatus.DISEASE, HealthStatus.NOT_SCANNED):
            print("LED:", status.value)
            hw.set_health(status)
            time.sleep(0.5)
        print("Buzzer")
        hw.beep(0.15)
        if args.pump:
            print("Pump pulse: 1 second")
            hw.pump_on()
            time.sleep(1)
            hw.pump_off()
        else:
            print("Pump pulse skipped. Re-run with --pump after confirming wiring and water isolation.")
        out = ROOT / "captures" / "selftest.jpg"
        print("Camera capture:", out)
        hw.capture(out)
        print("PASS: hardware self-test completed")
        return 0
    finally:
        hw.close()


if __name__ == "__main__":
    raise SystemExit(main())
