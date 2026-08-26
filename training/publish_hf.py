#!/usr/bin/env python3
"""Publish verified AgriVision AI artifacts to Hugging Face.

Authentication is supplied externally. In GitHub Actions, the official workflow
uses Hugging Face Trusted Publishers/OIDC to mint a short-lived repo-scoped
HF_TOKEN. No permanent token is stored in this repository.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from huggingface_hub import HfApi

OFFICIAL_DATASET = "geraldmc/plantvillage-full"
OFFICIAL_REVISION = "v0.1.0"


def validate_release(manifest: dict, require_official: bool, min_accuracy: float | None):
    if manifest.get("smoke_dataset"):
        raise SystemExit("Refusing to publish a smoke/debug model.")

    int8_metrics = manifest.get("int8_test_metrics") or {}
    measured = float(int8_metrics.get("accuracy", 0.0))
    if min_accuracy is not None and measured < min_accuracy:
        raise SystemExit(
            f"Refusing publication: INT8 accuracy {measured:.4f} < {min_accuracy:.4f}."
        )

    if require_official:
        checks = {
            "official_release_lineage": manifest.get("official_release_lineage") is True,
            "dataset": manifest.get("dataset") == OFFICIAL_DATASET,
            "dataset_revision": manifest.get("dataset_revision") == OFFICIAL_REVISION,
            "dataset_license": manifest.get("dataset_license") == "CC0-1.0",
            "task": manifest.get("task") == "binary",
            "initialization": manifest.get("initialization") == "scratch",
            "developer": manifest.get("developer") == "Isis Saritha Swapin",
            "publisher": manifest.get("publisher") == "PeachBot AI",
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise SystemExit(
                "Refusing publication: official-lineage validation failed for "
                + ", ".join(failed)
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--repo-id",
        required=True,
        help="e.g. peachbotAI/agrivision-mobilenetv2-edge-tpu",
    )
    p.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("training/output/agrivision-mobilenetv2"),
    )
    p.add_argument("--require-official-lineage", action="store_true")
    p.add_argument("--min-int8-accuracy", type=float, default=None)
    args = p.parse_args()

    manifest_path = args.artifact_dir / "model_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing model manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_release(manifest, args.require_official_lineage, args.min_int8_accuracy)

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is not available. Use the Trusted Publisher workflow or authenticate externally."
        )

    release = args.artifact_dir / "hf_release"
    if release.exists():
        shutil.rmtree(release)
    release.mkdir(parents=True, exist_ok=True)

    wanted = [
        "plant_health_int8.tflite",
        "labels.txt",
        "model_manifest.json",
        "README.md",
    ]
    for name in wanted:
        src = args.artifact_dir / name
        if src.exists():
            shutil.copy2(src, release / name)

    edgetpu_dir = args.artifact_dir / "edgetpu"
    candidates = list(edgetpu_dir.glob("*_edgetpu.tflite")) if edgetpu_dir.exists() else []
    if candidates:
        shutil.copy2(candidates[0], release / "plant_health_edgetpu.tflite")

    if not (release / "README.md").exists():
        template = Path(__file__).with_name("model-card-template.md")
        shutil.copy2(template, release / "README.md")

    api = HfApi(token=token)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=release,
        commit_message="Publish verified AgriVision AI model release",
    )
    print(f"Published https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
