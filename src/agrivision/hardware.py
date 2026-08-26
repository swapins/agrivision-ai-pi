from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
import time
from pathlib import Path

from .config import AppConfig
from .logic import HealthStatus, moisture_percent


@dataclass
class EnvironmentReading:
    soil_percent: float
    temperature_c: float
    humidity_percent: float
    raw_soil: float | None = None


class Hardware(ABC):
    @abstractmethod
    def read_environment(self) -> EnvironmentReading: ...

    @abstractmethod
    def pump_on(self) -> None: ...

    @abstractmethod
    def pump_off(self) -> None: ...

    @abstractmethod
    def set_health(self, status: HealthStatus) -> None: ...

    @abstractmethod
    def beep(self, seconds: float) -> None: ...

    @abstractmethod
    def capture(self, destination: Path) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class SimulatedHardware(Hardware):
    """Development-only hardware simulator.

    The dashboard is explicitly marked SIMULATION MODE when this backend is active.
    """

    def __init__(self, cfg: AppConfig):
        sim = cfg.section("simulation")
        self.soil = float(sim.get("default_soil_percent", 56.0))
        self.temp = float(sim.get("default_temperature_c", 29.0))
        self.hum = float(sim.get("default_humidity_percent", 71.0))
        self.pump = False
        self.health = HealthStatus.NOT_SCANNED

    def read_environment(self) -> EnvironmentReading:
        drift = random.uniform(-0.5, 0.5)
        if self.pump:
            self.soil = min(100.0, self.soil + 1.0)
        return EnvironmentReading(
            round(self.soil + drift, 1),
            round(self.temp + random.uniform(-0.2, 0.2), 1),
            round(self.hum + random.uniform(-0.5, 0.5), 1),
        )

    def pump_on(self) -> None:
        self.pump = True

    def pump_off(self) -> None:
        self.pump = False

    def set_health(self, status: HealthStatus) -> None:
        self.health = status

    def beep(self, seconds: float) -> None:
        time.sleep(min(seconds, 0.05))

    def capture(self, destination: Path) -> None:
        # Generate a neutral placeholder instead of pretending to be a real camera frame.
        from PIL import Image, ImageDraw
        destination.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (640, 480), "#e8f1e7")
        draw = ImageDraw.Draw(image)
        draw.text((30, 30), "SIMULATION MODE — NO REAL CAMERA IMAGE", fill="#173d24")
        image.save(destination, quality=90)

    def close(self) -> None:
        self.pump = False


class PiHardware(Hardware):
    def __init__(self, cfg: AppConfig):
        # Imports happen only on the Pi so the package remains testable on a PC.
        from gpiozero import LED, OutputDevice, Buzzer
        import board
        import busio
        from adafruit_ads1x15.ads1115 import ADS1115
        from adafruit_ads1x15.analog_in import AnalogIn
        import adafruit_bme280
        from picamera2 import Picamera2

        soil_cfg = cfg.section("soil")
        outputs = cfg.section("outputs")
        pump_cfg = cfg.section("pump")
        camera_cfg = cfg.section("camera")

        self._dry = float(soil_cfg["dry_raw"])
        self._wet = float(soil_cfg["wet_raw"])

        self._pump = OutputDevice(int(pump_cfg["gpio"]), active_high=True, initial_value=False)
        self._green = LED(int(outputs["green_gpio"]))
        self._yellow = LED(int(outputs["yellow_gpio"]))
        self._red = LED(int(outputs["red_gpio"]))
        self._buzzer = Buzzer(int(outputs["buzzer_gpio"]))

        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ADS1115(i2c)
        self._soil = AnalogIn(ads, int(soil_cfg.get("ads_channel", 0)))
        self._bme = adafruit_bme280.Adafruit_BME280_I2C(i2c)

        self._camera = Picamera2()
        self._camera.configure(
            self._camera.create_still_configuration(
                main={"size": (int(camera_cfg.get("width", 640)), int(camera_cfg.get("height", 480)))}
            )
        )
        self._camera.start()
        time.sleep(float(camera_cfg.get("warmup_seconds", 1.5)))

    def read_environment(self) -> EnvironmentReading:
        raw = float(self._soil.value)
        return EnvironmentReading(
            soil_percent=moisture_percent(raw, self._dry, self._wet),
            temperature_c=float(self._bme.temperature),
            humidity_percent=float(self._bme.relative_humidity),
            raw_soil=raw,
        )

    def pump_on(self) -> None:
        self._pump.on()

    def pump_off(self) -> None:
        self._pump.off()

    def set_health(self, status: HealthStatus) -> None:
        self._green.off(); self._yellow.off(); self._red.off()
        if status == HealthStatus.HEALTHY:
            self._green.on()
        elif status == HealthStatus.STRESS:
            self._yellow.on()
        elif status == HealthStatus.DISEASE:
            self._red.on()

    def beep(self, seconds: float) -> None:
        self._buzzer.on()
        time.sleep(seconds)
        self._buzzer.off()

    def capture(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._camera.capture_file(str(destination))

    def close(self) -> None:
        try:
            self._pump.off(); self._buzzer.off()
            self._green.off(); self._yellow.off(); self._red.off()
        finally:
            try:
                self._camera.stop()
            except Exception:
                pass


def make_hardware(cfg: AppConfig, simulation: bool) -> Hardware:
    return SimulatedHardware(cfg) if simulation else PiHardware(cfg)
