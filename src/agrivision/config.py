from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass(frozen=True)
class AppConfig:
    raw: dict[str, Any]
    root: Path

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"config section {name!r} must be a mapping")
        return value

    def resolve(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else self.root / p


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        path = Path.cwd() / "config.yaml"
        if not Path(path).exists():
            path = Path.cwd() / "config.example.yaml"
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping")
    return AppConfig(data, path.parent)


def validate_config(cfg: AppConfig, simulation: bool) -> None:
    model = cfg.section("model")
    soil = cfg.section("soil")
    pump = cfg.section("pump")
    outputs = cfg.section("outputs")
    camera = cfg.section("camera")

    required_model_keys = ("path", "labels")
    for key in required_model_keys:
        if not model.get(key):
            raise ValueError(f"model.{key} is required")
    if str(model.get("path")) != "models/plant_health_edgetpu.tflite":
        raise ValueError("model.path must point to models/plant_health_edgetpu.tflite")
    if str(model.get("labels")) != "models/labels.txt":
        raise ValueError("model.labels must point to models/labels.txt")

    top_k = int(model.get("top_k", 1))
    if top_k < 1:
        raise ValueError("model.top_k must be at least 1")
    confidence = float(model.get("confidence_min", 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("model.confidence_min must be between 0 and 1")

    dry_raw = float(soil["dry_raw"])
    wet_raw = float(soil["wet_raw"])
    if dry_raw == wet_raw:
        raise ValueError("soil.dry_raw and soil.wet_raw cannot be equal")
    on_below = float(soil["pump_on_below_percent"])
    stop_above = float(soil["pump_stop_above_percent"])
    if not 0 <= on_below <= 100 or not 0 <= stop_above <= 100:
        raise ValueError("soil pump thresholds must be between 0 and 100 percent")
    if stop_above <= on_below:
        raise ValueError("soil.pump_stop_above_percent must be greater than pump_on_below_percent")

    max_run = float(pump["max_run_seconds"])
    cooldown = float(pump["cooldown_seconds"])
    if max_run <= 0:
        raise ValueError("pump.max_run_seconds must be positive")
    if cooldown < 0:
        raise ValueError("pump.cooldown_seconds cannot be negative")

    width = int(camera.get("width", 0))
    height = int(camera.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("camera width and height must be positive")

    gpio_values = {
        "pump.gpio": int(pump["gpio"]),
        "outputs.green_gpio": int(outputs["green_gpio"]),
        "outputs.yellow_gpio": int(outputs["yellow_gpio"]),
        "outputs.red_gpio": int(outputs["red_gpio"]),
        "outputs.buzzer_gpio": int(outputs["buzzer_gpio"]),
    }
    for name, value in gpio_values.items():
        if not 0 <= value <= 27:
            raise ValueError(f"{name} must be a BCM GPIO number from 0 to 27")
    if len(set(gpio_values.values())) != len(gpio_values):
        raise ValueError(f"GPIO assignments must be unique: {gpio_values}")

    if not simulation:
        model_path = cfg.resolve(str(model["path"]))
        labels_path = cfg.resolve(str(model["labels"]))
        if not model_path.exists():
            raise FileNotFoundError(
                f"Edge TPU model not found: {model_path}. Run scripts/fetch_models.py first."
            )
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Labels file not found: {labels_path}. Run scripts/fetch_models.py first."
            )
