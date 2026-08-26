from __future__ import annotations

from datetime import datetime
from pathlib import Path
import threading
import time

from .config import AppConfig
from .hardware import Hardware
from .inference import Inferencer
from .logic import HealthStatus, irrigation_start_decision, label_to_health_status
from .state import StateStore


class AgriVisionController:
    def __init__(self, cfg: AppConfig, hardware: Hardware, inferencer: Inferencer, state: StateStore):
        self.cfg = cfg
        self.hardware = hardware
        self.inferencer = inferencer
        self.state = state
        self.stop_event = threading.Event()
        self._last_pump_time = 0.0
        self._pump_lock = threading.Lock()

    @property
    def capture_path(self) -> Path:
        return self.cfg.resolve(str(self.cfg.section("camera")["capture_path"]))

    def start_background(self) -> threading.Thread:
        thread = threading.Thread(target=self._background_loop, name="agrivision-sensors", daemon=True)
        thread.start()
        return thread

    def _background_loop(self) -> None:
        interval = float(self.cfg.section("loop").get("sensor_interval_seconds", 2.0))
        while not self.stop_event.is_set():
            try:
                reading = self.hardware.read_environment()
                self.state.update(
                    soil=round(reading.soil_percent, 1),
                    temperature=round(reading.temperature_c, 1),
                    humidity=round(reading.humidity_percent, 1),
                )
                self._maybe_irrigate(reading.soil_percent)
            except Exception as exc:
                self.hardware.pump_off()
                self.state.update(pump="OFF", message=f"Sensor error: {exc}")
            self.stop_event.wait(interval)

    def _maybe_irrigate(self, moisture: float) -> None:
        soil = self.cfg.section("soil")
        pump_cfg = self.cfg.section("pump")
        decision = irrigation_start_decision(
            moisture=moisture,
            on_below=float(soil["pump_on_below_percent"]),
            seconds_since_last_run=time.time() - self._last_pump_time,
            cooldown_seconds=float(pump_cfg["cooldown_seconds"]),
        )
        if not decision.should_start:
            # Never switch a currently-running pump off from this path; the pump worker owns it.
            return
        if self._pump_lock.acquire(blocking=False):
            threading.Thread(
                target=self._pump_cycle,
                args=(self._pump_lock,),
                name="agrivision-pump",
                daemon=True,
            ).start()

    def _pump_cycle(self, lock: threading.Lock) -> None:
        pump_cfg = self.cfg.section("pump")
        soil_cfg = self.cfg.section("soil")
        max_seconds = float(pump_cfg["max_run_seconds"])
        stop_above = float(soil_cfg["pump_stop_above_percent"])
        try:
            self.hardware.pump_on()
            self.state.update(pump="ON", message="Dry soil — irrigation cycle active")
            deadline = time.time() + max_seconds
            while time.time() < deadline and not self.stop_event.is_set():
                time.sleep(0.5)
                reading = self.hardware.read_environment()
                self.state.update(
                    soil=round(reading.soil_percent, 1),
                    temperature=round(reading.temperature_c, 1),
                    humidity=round(reading.humidity_percent, 1),
                )
                if reading.soil_percent >= stop_above:
                    break
        finally:
            self.hardware.pump_off()
            self._last_pump_time = time.time()
            self.state.update(pump="OFF", message="Irrigation cycle complete")
            lock.release()

    def scan_leaf(self) -> dict:
        model_cfg = self.cfg.section("model")
        outputs_cfg = self.cfg.section("outputs")
        min_conf = float(model_cfg.get("confidence_min", 0.60))
        top_k = int(model_cfg.get("top_k", 3))

        self.state.update(message="Capturing leaf image…")
        self.hardware.capture(self.capture_path)
        self.state.update(message="Analysing on Edge TPU…")
        predictions = self.inferencer.predict(self.capture_path, top_k=top_k)
        if not predictions:
            status = HealthStatus.UNCERTAIN
            top_label, top_score = "unknown", 0.0
        else:
            top_label, top_score = predictions[0].label, predictions[0].confidence
            status = label_to_health_status(top_label, top_score, min_conf)

        self.hardware.set_health(status)
        if status in {HealthStatus.DISEASE, HealthStatus.STRESS}:
            self.hardware.beep(float(outputs_cfg.get("buzzer_seconds", 0.20)))

        if status == HealthStatus.DISEASE:
            message = "Possible visible disease/problem — inspect the plant"
        elif status == HealthStatus.STRESS:
            message = "Possible visible stress — inspect water/nutrient conditions"
        elif status == HealthStatus.HEALTHY:
            message = "Plant appears healthy"
        else:
            message = "Low-confidence result — scan again under controlled lighting"

        top_payload = [
            {"label": p.label, "confidence": round(p.confidence * 100.0, 1)}
            for p in predictions
        ]
        now = datetime.now().strftime("%H:%M:%S")
        self.state.update(
            plant=status.value,
            raw_label=top_label,
            confidence=round(top_score * 100.0, 1),
            last_scan=now,
            message=message,
            top_predictions=top_payload,
        )
        return {
            "status": status.value,
            "label": top_label,
            "confidence": round(top_score * 100.0, 1),
            "top_predictions": top_payload,
        }

    def close(self) -> None:
        self.stop_event.set()
        self.hardware.close()
