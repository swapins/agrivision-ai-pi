from __future__ import annotations

from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Any


@dataclass
class RuntimeState:
    soil: float = 0.0
    temperature: float = 0.0
    humidity: float = 0.0
    pump: str = "OFF"
    plant: str = "NOT SCANNED"
    raw_label: str = "-"
    confidence: float = 0.0
    last_scan: str = "-"
    message: str = "System starting"
    simulation: bool = False
    top_predictions: list[dict[str, Any]] = field(default_factory=list)


class StateStore:
    def __init__(self, initial: RuntimeState | None = None):
        self._state = initial or RuntimeState()
        self._lock = Lock()

    def update(self, **values: Any) -> None:
        with self._lock:
            for key, value in values.items():
                if not hasattr(self._state, key):
                    raise AttributeError(key)
                setattr(self._state, key, value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return asdict(self._state)
