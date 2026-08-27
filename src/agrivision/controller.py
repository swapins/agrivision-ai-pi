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
    def __init__(
        self,
        cfg: AppConfig,
        hardware: Hardware,
        inferencer: Inferencer,
        state: StateStore,
        simulation: bool = False,
    ):
        self.cfg = cfg
        self.hardware = hardware
        self.inferencer = inferencer
        self.state = state
        self.simulation = simulation
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
                    raw_soil=reading.raw_soil,
                    backend=self.inferencer.backend_name,
                    coral_status=self.inferencer.coral_status,
                    error="",
                )
                self._maybe_irrigate(reading.soil_percent)
            except Exception as exc:
                self.hardware.pump_off()
                self.state.update(
                    pump="OFF",
                    pump_reason="sensor error",
                    message=f"Sensor error: {exc}",
                    error=str(exc),
                )
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
        self.state.update(pump_reason=decision.reason)
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
            prefix = "SIMULATION: " if self.simulation else ""
            self.hardware.pump_on()
            self.state.update(
                pump="ON",
                pump_reason="dry soil",
                message=f"{prefix}Dry soil - irrigation cycle active",
            )
            deadline = time.time() + max_seconds
            while time.time() < deadline and not self.stop_event.is_set():
                time.sleep(0.5)
                reading = self.hardware.read_environment()
                self.state.update(
                    soil=round(reading.soil_percent, 1),
                    temperature=round(reading.temperature_c, 1),
                    humidity=round(reading.humidity_percent, 1),
                    raw_soil=reading.raw_soil,
                )
                if reading.soil_percent >= stop_above:
                    break
        finally:
            self.hardware.pump_off()
            self._last_pump_time = time.time()
            self.state.update(pump="OFF", pump_reason="cycle complete", message="Irrigation cycle complete")
            lock.release()

    def scan_leaf(self) -> dict:
        model_cfg = self.cfg.section("model")
        outputs_cfg = self.cfg.section("outputs")
        min_conf = float(model_cfg.get("confidence_min", 0.60))
        top_k = int(model_cfg.get("top_k", 3))

        prefix = "SIMULATION: " if self.simulation else ""
        self.state.update(message=f"{prefix}Capturing leaf image")
        self.hardware.capture(self.capture_path)
        self.state.update(
            message=f"{prefix}Analysing image" if self.simulation else "Analysing on Coral Edge TPU",
            backend=self.inferencer.backend_name,
            coral_status=self.inferencer.coral_status,
        )
        started = time.perf_counter()
        predictions = self.inferencer.predict(self.capture_path, top_k=top_k)
        latency_ms = (time.perf_counter() - started) * 1000.0
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
            message = "Possible visible disease/problem - inspect the plant"
        elif status == HealthStatus.STRESS:
            message = "Possible visible stress - inspect water/nutrient conditions"
        elif status == HealthStatus.HEALTHY:
            message = "Plant appears healthy"
        else:
            message = "Low-confidence result - scan again under controlled lighting"
        if self.simulation:
            message = "SIMULATION: " + message

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
            last_inference_at=now,
            inference_latency_ms=round(latency_ms, 1),
            message=message,
            top_predictions=top_payload,
            backend=self.inferencer.backend_name,
            coral_status=self.inferencer.coral_status,
            error="",
        )
        return {
            "status": status.value,
            "label": top_label,
            "confidence": round(top_score * 100.0, 1),
            "top_predictions": top_payload,
            "simulation": self.simulation,
            "backend": self.inferencer.backend_name,
            "coral_status": self.inferencer.coral_status,
            "inference_latency_ms": round(latency_ms, 1),
        }

    def emergency_pump_off(self) -> dict:
        self.hardware.pump_off()
        self._last_pump_time = time.time()
        self.state.update(
            pump="OFF",
            pump_reason="manual stop",
            message="Pump manually stopped from dashboard",
            error="",
        )
        return {"pump": "OFF", "reason": "manual stop", "simulation": self.simulation}

    def close(self) -> None:
        self.stop_event.set()
        self.hardware.pump_off()
        self.hardware.close()
