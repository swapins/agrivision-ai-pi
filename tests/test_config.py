from pathlib import Path
import pytest

from agrivision.config import AppConfig, load_config, validate_config


def test_example_config_loads():
    root=Path(__file__).resolve().parents[1]
    cfg=load_config(root/'config.example.yaml')
    assert cfg.section('pump')['gpio'] == 17
    assert cfg.section('model')['confidence_min'] == 0.60


def test_example_config_validates_in_simulation():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config.example.yaml")
    validate_config(cfg, simulation=True)


def test_real_config_requires_model_files(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config.example.yaml")
    missing = AppConfig(cfg.raw, tmp_path)
    with pytest.raises(FileNotFoundError, match="Edge TPU model not found"):
        validate_config(missing, simulation=False)


def test_config_rejects_impossible_pump_thresholds(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config.example.yaml")
    raw = dict(cfg.raw)
    raw["soil"] = dict(raw["soil"], pump_on_below_percent=50.0, pump_stop_above_percent=40.0)
    with pytest.raises(ValueError, match="pump_stop_above_percent"):
        validate_config(AppConfig(raw, tmp_path), simulation=True)


def test_config_rejects_duplicate_gpio(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config.example.yaml")
    raw = dict(cfg.raw)
    raw["outputs"] = dict(raw["outputs"], green_gpio=17)
    with pytest.raises(ValueError, match="GPIO assignments must be unique"):
        validate_config(AppConfig(raw, tmp_path), simulation=True)
