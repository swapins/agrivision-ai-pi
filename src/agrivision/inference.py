from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .config import AppConfig


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


class Inferencer:
    @property
    def backend_name(self) -> str:
        return "unknown"

    @property
    def coral_status(self) -> str:
        return "unknown"

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
        if not self._labels:
            raise ValueError(f"No labels found in {self._labels_path}")
        self._interpreter = make_interpreter(str(self._model_path))
        self._interpreter.allocate_tensors()
        input_details = self._interpreter.get_input_details()[0]
        input_dtype = input_details.get("dtype")
        if str(input_dtype).split(".")[-1].strip("'>") != "uint8":
            raise RuntimeError(
                "Edge TPU runtime requires a full-integer UINT8 TFLite input; "
                f"got {input_dtype}"
            )
        self._backend_name = "Coral Edge TPU"
        self._coral_status = "Edge TPU interpreter allocated"

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def coral_status(self) -> str:
        return self._coral_status

    def predict(self, image_path: Path, top_k: int = 3) -> list[Prediction]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
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
    @property
    def backend_name(self) -> str:
        return "simulation"

    @property
    def coral_status(self) -> str:
        return "simulation only - no Coral inference"

    def predict(self, image_path: Path, top_k: int = 3) -> list[Prediction]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        # Deliberately deterministic and limited to the official binary labels.
        # These are UI-development values only and are never represented as Coral output.
        return [Prediction("healthy", 0.93), Prediction("problem", 0.07)][:top_k]


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
