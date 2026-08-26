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
