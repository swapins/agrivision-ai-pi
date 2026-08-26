# AgriVision AI — Raspberry Pi + Coral Edge TPU Smart Farm

A complete school-exhibition implementation for the **AgriVision AI** project: a Raspberry Pi 4 controls sensors, irrigation, camera capture, LEDs/buzzer and a local dashboard, while a Google Coral USB Accelerator performs local TensorFlow Lite inference.

> **Project author:** Isis Saritha Swapin — Class VIII, PeachBot Author / student researcher in precision agriculture.

## What this repository contains

- Raspberry Pi runtime with real hardware and explicit simulation mode.
- Capacitive soil-moisture sensing through ADS1115.
- BME280 temperature/humidity monitoring.
- Safe 5 V pump control through a MOSFET module, with hysteresis, runtime limit and cooldown.
- Raspberry Pi Camera / Picamera2 capture.
- Coral Edge TPU inference through PyCoral.
- Local Flask dashboard with a one-click **SCAN LEAF** workflow.
- Green / yellow / red LED health indicators and brief buzzer alert.
- Soil calibration and hardware self-test utilities.
- Systemd service for exhibition-day auto-start.
- Reproducible MobileNetV2 training, full-integer TFLite export and Edge TPU compilation pipeline.
- Hugging Face publishing script and model-card template.
- Unit tests that run on a normal PC without Raspberry Pi hardware.

## Recommended model strategy

The runtime accepts either a compact binary model (`healthy`, `problem`) or a multiclass PlantVillage-style disease model. For the most reliable school demonstration, the recommended first release is **MobileNetV2 alpha 0.35, 224×224**, fine-tuned on a leaf-grouped PlantVillage split and exported as fully quantized INT8 TFLite before Edge TPU compilation.

`stress` can be produced by a dedicated model if a validated stress dataset is available, but this repository does **not** silently fabricate a visual stress class. Soil/water stress is already demonstrated using the real moisture sensor. See `docs/MODEL.md`.

## Hardware

| Component | Purpose |
|---|---|
| Raspberry Pi 4 (4 GB recommended) | Main controller and dashboard |
| Google Coral USB Accelerator | Edge TPU inference |
| Raspberry Pi Camera Module 3 | Leaf image capture |
| Capacitive soil moisture sensor | Soil moisture |
| ADS1115 ADC | Reads analog soil sensor |
| BME280 | Temperature / humidity |
| 5 V mini pump | Irrigation |
| Logic-level MOSFET driver | Safe pump switching |
| Green / yellow / red LEDs | Plant status |
| 3.3 V logic-compatible active buzzer | Alert |

**Never power the pump from a GPIO pin.** Use a separate 5 V pump supply and a proper MOSFET driver, with common logic ground. Keep water physically separated from the Pi and power electronics.

## GPIO plan (BCM numbering)

- Pump MOSFET: GPIO17
- Green LED: GPIO22
- Yellow LED: GPIO23
- Red LED: GPIO24
- Buzzer: GPIO27
- I²C SDA: GPIO2
- I²C SCL: GPIO3
- ADS1115 soil channel: A0
- Coral: USB 3
- Camera: CSI

## Quick start on Raspberry Pi

The compatibility-first route for the original school build is Raspberry Pi OS Bullseye 64-bit on Pi 4 because PyCoral packaging is older than current Raspberry Pi OS Python releases.

```bash
git clone https://github.com/swapins/agrivision-ai-pi.git agrivision-ai-pi
cd agrivision-ai-pi
sudo bash scripts/install_pi.sh
sudo bash scripts/install_coral.sh
cp config.example.yaml config.yaml
python3 scripts/calibrate_soil.py
python3 scripts/hardware_selftest.py
```

Copy the compiled Edge TPU model into `models/`:

```text
models/plant_health_edgetpu.tflite
models/labels.txt
```

Then run:

```bash
./scripts/run.sh
```

Open `http://127.0.0.1:5000` on the Pi, or `http://<PI-IP>:5000` from another device on the same LAN.

## Simulation mode on PC

Simulation exists for software development only and is visibly marked in the dashboard. It must not be presented as real hardware inference.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp config.example.yaml config.yaml
AGRIVISION_SIMULATION=1 python -m agrivision.app
```

## Training and Coral deployment

See `training/README.md`. The pipeline is:

```text
PlantVillage full dataset
        ↓
MobileNetV2 α=0.35 / 224 px
        ↓
Transfer learning + fine-tuning
        ↓
Full integer INT8 TFLite
        ↓
Edge TPU Compiler
        ↓
plant_health_edgetpu.tflite
        ↓
Raspberry Pi + Coral USB Accelerator
```

## Scientific scope

This is a **school prototype**, not a crop-diagnosis device. PlantVillage images are predominantly controlled-background leaf images, so high held-out accuracy does not prove field robustness. The UI therefore says “possible disease/problem” and exposes confidence / uncertainty instead of presenting predictions as professional advice.

## License

Source code is released under the MIT License. Dataset and model artifacts may have their own terms; see `docs/MODEL.md` and the model card before redistribution.
