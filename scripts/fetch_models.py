#!/usr/bin/env python3
"""Fetch AgriVision AI release artifacts from Hugging Face."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ID = "peachbotAI/agrivision-mobilenetv2-edge-tpu"
BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main"
REQUIRED_FILES = (
    "plant_health_edgetpu.tflite",
    "plant_health_int8.tflite",
    "labels.txt",
    "model_manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_sha(manifest: dict, filename: str) -> str | None:
    candidates = [
        manifest.get(f"{filename}_sha256"),
        manifest.get(filename, {}).get("sha256") if isinstance(manifest.get(filename), dict) else None,
        manifest.get("sha256", {}).get(filename) if isinstance(manifest.get("sha256"), dict) else None,
        manifest.get("files", {}).get(filename, {}).get("sha256")
        if isinstance(manifest.get("files"), dict) and isinstance(manifest["files"].get(filename), dict)
        else None,
    ]
    if filename == "plant_health_edgetpu.tflite":
        candidates.extend(
            [
                manifest.get("plant_health_edgetpu_sha256"),
                manifest.get("edge_tpu_sha256"),
                manifest.get("edgetpu_sha256"),
            ]
        )
    for value in candidates:
        if isinstance(value, str) and len(value) == 64:
            return value.lower()
    return None


def download_file(filename: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"present: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{filename}"
    print(f"download: {url}")
    request = Request(url, headers={"User-Agent": "AgriVisionAI-model-fetch/1.0"})
    try:
        with urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                "wb", delete=False, dir=str(destination.parent), prefix=f".{filename}.", suffix=".tmp"
            ) as tmp:
                tmp_path = Path(tmp.name)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise SystemExit(f"Failed to download {filename}: {exc}") from exc

    if tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        raise SystemExit(f"Downloaded empty file for {filename}; existing file was not replaced.")
    os.replace(tmp_path, destination)
    print(f"saved: {destination} ({destination.stat().st_size} bytes)")


def load_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid model manifest JSON: {path}: {exc}") from exc


def verify_models(models_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (models_dir / name).exists()]
    if missing:
        raise SystemExit(f"Missing required model files in {models_dir}: {', '.join(missing)}")

    manifest = load_manifest(models_dir / "model_manifest.json")
    expected = expected_sha(manifest, "plant_health_edgetpu.tflite")
    edge_tpu = models_dir / "plant_health_edgetpu.tflite"
    actual = sha256_file(edge_tpu)
    if expected:
        if actual != expected:
            raise SystemExit(
                "Checksum mismatch for plant_health_edgetpu.tflite: "
                f"expected {expected}, got {actual}. Keeping existing files for inspection."
            )
        print("verified: plant_health_edgetpu.tflite SHA256 matches manifest")
    else:
        print(f"verified: plant_health_edgetpu.tflite present; SHA256 {actual}")
        print("manifest does not declare an Edge TPU SHA256; checksum comparison skipped")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--force", action="store_true", help="replace existing downloaded files")
    parser.add_argument("--verify-only", action="store_true", help="verify existing files without downloading")
    args = parser.parse_args()

    if not args.verify_only:
        for filename in REQUIRED_FILES:
            download_file(filename, args.models_dir / filename, args.force)
    verify_models(args.models_dir)
    print(f"Model deployment files are ready in {args.models_dir.resolve()}")


if __name__ == "__main__":
    main()
