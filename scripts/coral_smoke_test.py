#!/usr/bin/env python3
"""Validate the deployed Edge TPU model and, optionally, one image."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agrivision.config import load_config, validate_config
from agrivision.inference import EdgeTpuInferencer


def lsusb_lines() -> list[str]:
    try:
        result = subprocess.run(["lsusb"], check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return ["lsusb unavailable"]
    return [line for line in result.stdout.splitlines() if "Google" in line or "Coral" in line or "1a6e" in line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", type=Path, help="Optional test image for real Edge TPU inference")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg, simulation=False)
    print("Config: OK")
    print("USB Coral candidates:")
    for line in lsusb_lines() or ["No obvious Coral USB device reported by lsusb"]:
        print(f"  {line}")

    inf = EdgeTpuInferencer(cfg)
    print(f"Backend: {inf.backend_name}")
    print(f"Coral status: {inf.coral_status}")

    if args.image is None:
        print("No image supplied; interpreter allocation passed, inference skipped.")
        print("Run with a real leaf image path to verify end-to-end inference latency.")
        return 0

    start = time.perf_counter()
    predictions = inf.predict(args.image, top_k=5)
    latency_ms = (time.perf_counter() - start) * 1000.0
    print(f"Inference latency: {latency_ms:.1f} ms")
    for pred in predictions:
        print(f"{pred.label}: {pred.confidence * 100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
