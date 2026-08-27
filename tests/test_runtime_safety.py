from __future__ import annotations

from pathlib import Path

from PIL import Image

from agrivision.app import create_app
from agrivision.config import AppConfig, load_config
from agrivision.controller import AgriVisionController
from agrivision.hardware import EnvironmentReading, Hardware
from agrivision.inference import Inferencer, Prediction, SimulatedInferencer
from agrivision.logic import HealthStatus
from agrivision.state import RuntimeState, StateStore


class FakeHardware(Hardware):
    def __init__(self):
        self.pump = False
        self.closed = False
        self.health = HealthStatus.NOT_SCANNED

    def read_environment(self) -> EnvironmentReading:
        return EnvironmentReading(25.0, 28.0, 70.0, raw_soil=12345.0)

    def pump_on(self) -> None:
        self.pump = True

    def pump_off(self) -> None:
        self.pump = False

    def set_health(self, status: HealthStatus) -> None:
        self.health = status

    def beep(self, seconds: float) -> None:
        return None

    def capture(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), "green").save(destination)

    def close(self) -> None:
        self.closed = True


class FakeInferencer(Inferencer):
    @property
    def backend_name(self) -> str:
        return "test backend"

    @property
    def coral_status(self) -> str:
        return "test status"

    def predict(self, image_path: Path, top_k: int = 3) -> list[Prediction]:
        return [Prediction("healthy", 0.99)][:top_k]


def test_simulation_health_and_status_are_explicit(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("AGRIVISION_CONFIG", str(root / "config.example.yaml"))
    monkeypatch.setenv("AGRIVISION_SIMULATION", "1")

    app = create_app()
    client = app.test_client()
    try:
        health = client.get("/health").get_json()
        status = client.get("/api/status").get_json()
    finally:
        app.config["AGRIVISION_CONTROLLER"].close()

    assert health["simulation"] is True
    assert health["backend"] == "simulation"
    assert "no Coral inference" in health["coral_status"]
    assert status["simulation"] is True
    assert status["backend"] == "simulation"


def test_simulation_uses_official_binary_labels(tmp_path):
    predictions = SimulatedInferencer().predict(tmp_path / "unused.jpg", top_k=3)
    assert [prediction.label for prediction in predictions] == ["healthy", "problem"]


def test_non_simulation_requires_model_files(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("AGRIVISION_CONFIG", str(root / "config.example.yaml"))
    monkeypatch.delenv("AGRIVISION_SIMULATION", raising=False)
    monkeypatch.chdir(tmp_path)

    try:
        create_app()
    except FileNotFoundError as exc:
        assert "plant_health_edgetpu.tflite" in str(exc)
    else:
        raise AssertionError("real mode should require the Edge TPU model")


def test_controller_scan_marks_simulation_outputs(tmp_path):
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.example.yaml")
    raw = dict(cfg.raw)
    raw["camera"] = dict(raw["camera"], capture_path="latest.jpg")
    cfg = AppConfig(raw, tmp_path)
    store = StateStore(RuntimeState(simulation=True))
    controller = AgriVisionController(cfg, FakeHardware(), FakeInferencer(), store, simulation=True)

    payload = controller.scan_leaf()
    snapshot = store.snapshot()

    assert payload["simulation"] is True
    assert payload["backend"] == "test backend"
    assert snapshot["message"].startswith("SIMULATION:")
    assert snapshot["inference_latency_ms"] is not None


def test_manual_pump_off_sets_cooldown_state(tmp_path):
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.example.yaml")
    hardware = FakeHardware()
    hardware.pump_on()
    store = StateStore(RuntimeState())
    controller = AgriVisionController(cfg, hardware, FakeInferencer(), store)

    payload = controller.emergency_pump_off()

    assert payload == {"pump": "OFF", "reason": "manual stop", "simulation": False}
    assert hardware.pump is False
    assert store.snapshot()["pump_reason"] == "manual stop"
