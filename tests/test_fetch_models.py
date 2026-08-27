import json
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch_models", ROOT / "scripts" / "fetch_models.py")
fetch_models = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(fetch_models)

expected_sha = fetch_models.expected_sha
sha256_file = fetch_models.sha256_file
verify_models = fetch_models.verify_models
install_downloads = fetch_models.install_downloads


def test_sha256_file(tmp_path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"agrivision")
    assert sha256_file(path) == "d5a44fc22930024c4141ffa91b7d649a470389a23a65741547a36f5d53f7ae28"


def test_expected_sha_reads_manifest_shapes():
    digest = "a" * 64
    assert expected_sha({"edge_tpu_sha256": digest}, "plant_health_edgetpu.tflite") == digest
    assert expected_sha({"edge_tpu_model_sha256": digest}, "plant_health_edgetpu.tflite") == digest
    assert expected_sha({"files": {"plant_health_edgetpu.tflite": {"sha256": digest}}}, "plant_health_edgetpu.tflite") == digest


def test_verify_models_detects_checksum_mismatch(tmp_path):
    for name in ("plant_health_edgetpu.tflite", "plant_health_int8.tflite", "labels.txt"):
        (tmp_path / name).write_text("placeholder", encoding="utf-8")
    (tmp_path / "model_manifest.json").write_text(
        json.dumps({"edge_tpu_sha256": "0" * 64}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="Checksum mismatch"):
        verify_models(tmp_path)


def test_failed_staged_verification_preserves_existing_release(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    existing = models / "plant_health_edgetpu.tflite"
    existing.write_bytes(b"known-good")

    staged = {}
    for name in fetch_models.REQUIRED_FILES:
        path = tmp_path / f"staged-{name}"
        if name == "model_manifest.json":
            path.write_text(json.dumps({"edge_tpu_model_sha256": "0" * 64}), encoding="utf-8")
        else:
            path.write_bytes(b"replacement")
        staged[name] = path

    with pytest.raises(SystemExit, match="Checksum mismatch"):
        install_downloads(staged, models)

    assert existing.read_bytes() == b"known-good"
