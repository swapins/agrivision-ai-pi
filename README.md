<div align="center">

<img src="src/agrivision/static/logo.png" alt="PeachBot — SBC for Biology" width="320">

# AgriVision AI
## Raspberry Pi + Coral Edge TPU Smart Agriculture Project

**Edge AI plant-health monitoring · automatic irrigation · local/offline inference · school science exhibition**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4-C51A4A?logo=raspberrypi&logoColor=white)](https://www.raspberrypi.com/)
[![Coral Edge TPU](https://img.shields.io/badge/Google%20Coral-Edge%20TPU-4285F4)](https://coral.ai/)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow-Lite-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/lite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Developed by [Isis Saritha Swapin](https://peachbot.in/people/4)**  
Studying in **Class 8** at **[Syndesmos Public School, Parumala, Thiruvalla](http://syndesmospublicschool.org/)** · PeachBot Author

**Contributor: Swapin Vidya**

</div>

---

## About AgriVision AI

**AgriVision AI** is an open-source **smart agriculture working model** built for a school science and technology exhibition. It combines a **Raspberry Pi 4**, **Google Coral USB Edge TPU**, camera, soil-moisture sensing, environmental monitoring and low-voltage irrigation into one practical edge-AI system.

The project demonstrates two independent capabilities:

1. **Real sensor-controlled irrigation** — a capacitive soil-moisture sensor is read through an ADS1115 ADC and the Raspberry Pi safely controls a 5 V water pump through a MOSFET driver.
2. **Real local AI inference** — a leaf image captured by the Raspberry Pi Camera is classified by a quantized TensorFlow Lite model accelerated by the Coral Edge TPU.

After installation and model deployment, the core demonstration can run **without sending each image to a cloud service**.

### Search-friendly project themes

Raspberry Pi smart agriculture · AI agriculture project for students · smart farming working model · Coral Edge TPU project · plant disease detection using Raspberry Pi · automatic irrigation project · edge AI agriculture · school science fair AI project · TensorFlow Lite plant health classification · offline AI smart farm

---

## Project authorship

### Primary developer — Isis Saritha Swapin

**Isis Saritha Swapin** is the primary developer and student author of AgriVision AI. She is studying in **Class 8 at Syndesmos Public School, Parumala, Thiruvalla**. The repository accompanies her work communicating science and technology through PeachBot.

- Role: **Primary Developer / Project Author**
- Project: **AgriVision AI**
- School: **Syndesmos Public School, Parumala, Thiruvalla**
- Current class: **Class 8**
- Focus: Raspberry Pi, Edge AI, smart farming and school STEM demonstration
- PeachBot author profile: https://peachbot.in/people/4

### Contributor — Swapin Vidya

**Swapin Vidya** is credited as a project contributor supporting the technical architecture, engineering review, documentation and implementation guidance.

See [`AUTHORS.md`](AUTHORS.md) and [`CITATION.cff`](CITATION.cff) for formal attribution.

---

## What the system does

- Measures **soil moisture** using a capacitive probe and ADS1115.
- Measures **temperature and humidity** using BME280.
- Automatically starts a **5 V irrigation pump** when soil is too dry.
- Applies **hysteresis, maximum runtime and cooldown** to make the pump demo safer and more reliable.
- Captures leaves with **Raspberry Pi Camera Module 3 / Picamera2**.
- Runs a **MobileNetV2-based TensorFlow Lite classifier** on the **Coral Edge TPU**.
- Displays plant status, confidence and sensor readings on a **local Flask dashboard**.
- Uses **green / yellow / red LEDs** and a buzzer for exhibition-friendly feedback.
- Includes an explicit **simulation mode for development**, visibly marked so it cannot be confused with a real hardware run.
- Includes model training, INT8 export, Edge TPU compilation and future Hugging Face publishing tooling.

---

## System architecture

```text
Camera Module 3
      │
      ▼
Raspberry Pi 4 ───────► Coral USB Edge TPU
      │                  │
      │                  └── Plant-health inference
      │
      ├── ADS1115 ◄── Capacitive soil-moisture sensor
      ├── BME280  ◄── Temperature / humidity
      ├── MOSFET ───► 5 V irrigation pump
      ├── LEDs + buzzer
      └── Local Flask dashboard
```

All essential decisions are local to the model. Internet access is not required for each inference after the system and model are installed.

---

## Hardware

| Component | Purpose |
|---|---|
| Raspberry Pi 4 (4 GB recommended) | Main controller and dashboard |
| Google Coral USB Accelerator | Edge TPU neural-network inference |
| Raspberry Pi Camera Module 3 | Leaf image capture |
| Capacitive soil moisture sensor | Soil moisture measurement |
| ADS1115 16-bit ADC | Converts soil-sensor analog output |
| BME280 | Temperature and humidity |
| 5 V mini pump | Irrigation demonstration |
| Logic-level MOSFET driver | Safe pump switching |
| Green / yellow / red LEDs | Plant-health / warning status |
| 3.3 V-compatible active buzzer | Alert |
| Separate 5 V pump supply | Keeps pump current away from Pi GPIO |

> **Safety:** never power the pump directly from a Raspberry Pi GPIO pin. Keep the water reservoir and tubing physically separated from the Pi, Coral and power electronics.

---

## GPIO plan — BCM numbering

| Function | GPIO / bus |
|---|---|
| Pump MOSFET | GPIO17 |
| Green LED | GPIO22 |
| Yellow LED | GPIO23 |
| Red LED | GPIO24 |
| Buzzer | GPIO27 |
| I²C SDA | GPIO2 |
| I²C SCL | GPIO3 |
| Soil sensor | ADS1115 A0 |
| Coral accelerator | USB 3 |
| Camera | CSI |

See [`docs/HARDWARE.md`](docs/HARDWARE.md) for the full wiring notes.

---

## Quick start on Raspberry Pi

The original exhibition build uses a compatibility-first Raspberry Pi 4 software path for the Coral/PyCoral stack.

```bash
git clone https://github.com/swapins/agrivision-ai-pi.git
cd agrivision-ai-pi
sudo bash scripts/install_pi.sh
sudo bash scripts/install_coral.sh
cp config.example.yaml config.yaml
python3 scripts/calibrate_soil.py
python3 scripts/hardware_selftest.py
```

Place the final compiled model in:

```text
models/plant_health_edgetpu.tflite
models/labels.txt
```

Run:

```bash
./scripts/run.sh
```

Open:

```text
http://127.0.0.1:5000
```

or from another device on the same LAN:

```text
http://<PI-IP>:5000
```

---

## PC simulation mode

Simulation mode is for software development and UI testing only. The dashboard explicitly labels it as simulation.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp config.example.yaml config.yaml
AGRIVISION_SIMULATION=1 python -m agrivision.app
```

It must not be presented as a real Coral or sensor result.

---

## AI model strategy

The default training target is:

**MobileNetV2 α=0.35 · 224×224 RGB · transfer learning · full-integer INT8 TFLite · Edge TPU compilation**

The first recommended exhibition classifier is intentionally robust and simple:

- `healthy`
- `problem`

A multiclass mode can preserve individual PlantVillage disease classes. Water stress is independently demonstrated by the real soil sensor rather than manufacturing an unsupported visual stress label.

```text
PlantVillage full dataset
        ↓
MobileNetV2 α=0.35 / 224 px
        ↓
Transfer learning + fine-tuning
        ↓
Held-out evaluation
        ↓
Full integer INT8 TFLite
        ↓
Edge TPU Compiler
        ↓
plant_health_edgetpu.tflite
        ↓
Raspberry Pi 4 + Coral USB Accelerator
```

See [`training/README.md`](training/README.md) and [`docs/MODEL.md`](docs/MODEL.md).

---

## Repository structure

```text
agrivision-ai-pi/
├── src/agrivision/          Raspberry Pi runtime and dashboard
├── scripts/                 install, calibration, self-test and service scripts
├── models/                  model deployment location and label examples
├── training/                MobileNetV2 training, INT8 export and HF release tools
├── tests/                   PC-safe unit tests
├── docs/                    hardware, safety, model and exhibition documentation
├── .github/workflows/       CI and model-training workflows
├── config.example.yaml      project / GPIO / thresholds configuration
├── AUTHORS.md               project authorship and contribution statement
├── CITATION.cff             citation metadata
└── LICENSE                  MIT license
```

---

## Testing

Run the unit tests on a normal PC:

```bash
pip install -e '.[dev]'
pytest -q
```

Hardware is abstracted so core configuration and decision logic can be tested without a Raspberry Pi attached.

---

## School exhibition scope

AgriVision AI is designed to demonstrate:

- edge artificial intelligence,
- computer vision,
- sensor calibration,
- automatic irrigation,
- GPIO control,
- ADC use,
- safe load switching,
- AI confidence and uncertainty,
- local/offline inference.

The project is **not a professional crop-diagnosis device**. PlantVillage contains largely controlled-background leaf images, so held-out dataset accuracy must not be presented as proof of general field performance.

---

## Citation

If you reuse AgriVision AI in a school project, article, demonstration or derivative implementation, please credit:

> **Isis Saritha Swapin (Syndesmos Public School, Parumala, Thiruvalla). AgriVision AI: Raspberry Pi + Coral Edge TPU Smart Agriculture Project. PeachBot, 2026. Contributor: Swapin Vidya.**

Machine-readable citation metadata is available in [`CITATION.cff`](CITATION.cff).

---

## License

Code is released under the [MIT License](LICENSE).

Dataset and trained-model artifacts may have their own licensing or redistribution terms; verify those terms before publishing model weights.

---

<div align="center">

**SEE · THINK · WATER · PROTECT**

Developed by **Isis Saritha Swapin**, studying in **Class 8 at Syndesmos Public School, Parumala, Thiruvalla**  
Contributor **Swapin Vidya** · **PeachBot — SBC for Biology**

</div>
