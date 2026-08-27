#!/usr/bin/env python3
"""Kaggle GPU runner for the official AgriVision AI training path."""
from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf

import train_tf_mobilenetv2


DEFAULT_OUTPUT = Path("training/output/agrivision-mobilenetv2")


def main() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"Kaggle GPU check: detected physical GPUs: {[gpu.name for gpu in gpus]}")
    else:
        print(
            "WARNING: No TensorFlow GPU is visible. Kaggle official training should run "
            "with GPU acceleration enabled.",
            file=sys.stderr,
        )

    sys.argv = [
        "training/train_tf_mobilenetv2.py",
        "--task",
        "binary",
        "--init",
        "scratch",
        "--require-official-lineage",
        "--skip-sanity-check",
        "--batch-size",
        "16",
        "--min-int8-accuracy",
        "0.80",
        "--min-int8-balanced-accuracy",
        "0.75",
        "--min-int8-macro-f1",
        "0.75",
        "--require-all-classes-predicted",
        "--output",
        str(DEFAULT_OUTPUT),
    ]
    train_tf_mobilenetv2.main()


if __name__ == "__main__":
    main()
