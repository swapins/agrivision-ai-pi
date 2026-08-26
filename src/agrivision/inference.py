from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from .config import AppConfig


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


class Inferencer:
    def predict(self, image_path: Path, top_k: int = 3) -> list[Prediction]:
        raise NotImplementedError


class EdgeTpuInferencer(Inferencer):
    def __init__(self, cfg: AppConfig):
        from pycoral.utils.edgetpu import make_interpreter
        from pycoral.adapters import common, classify

        self._common = common
        self._classify = classify
        model_cfg = cfg.section("model")
        self._model_path = cfg.resolve(str(model_cfg["path"]))
        self._labels_path = cfg.resolve(str(model_cfg["labels"]))
        if not self._model_path.exists():
            raise FileNotFoundError(f"Edge TPU model not found: {self._model_path}")
        if not self._labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self._labels_path}")

        self._labels = _read_labels(self._labels_path)
        self._interpreter = make_interpreter(str(self._model_path))
        self._interpreter.allocate_tensors()

    def predict(self, image_path: Path, top_k: int = 3) -> list[Prediction]:
        image = Image.open(image_path).convert("RGB")
        size = self._common.input_size(self._interpreter)
        image = image.resize(size, Image.Resampling.LANCZOS)
        self._common.set_input(self._interpreter, image)
        self._interpreter.invoke()
        classes = self._classify.get_classes(self._interpreter, top_k=top_k)
        return [
            Prediction(
                self._labels.get(int(item.id), str(item.id)),
                float(item.score),
            )
            for item in classes
        ]


class SimulatedInferencer(Inferencer):
    def predict(self, image_path: Path, top_k: int = 3) -> list[Prediction]:
        # Intentionally deterministic and clearly simulation-only.
        return [Prediction("healthy", 0.93), Prediction("disease", 0.05), Prediction("stress", 0.02)][:top_k]


def _read_labels(path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for sequential_index, raw in enumerate(fh):
            line = raw.strip()
            if not line:
                continue
            # Accept either `0 healthy` or plain `healthy` formats.
            first, *rest = line.split(maxsplit=1)
            if rest and first.isdigit():
                labels[int(first)] = rest[0]
            else:
                labels[sequential_index] = line
    return labels


def make_inferencer(cfg: AppConfig, simulation: bool) -> Inferencer:
    return SimulatedInferencer() if simulation else EdgeTpuInferencer(cfg)
