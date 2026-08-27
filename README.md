<div align="center">

<img src="src/agrivision/static/logo.png" alt="PeachBot - SBC for Biology" width="320">

# AgriVision AI
## Raspberry Pi + Coral Edge TPU Smart Agriculture Project

**Edge AI plant-health monitoring | automatic irrigation | local/offline inference | school science exhibition**

**Developed by [Isis Saritha Swapin](https://peachbot.in/people/4)**  
Class 8, Syndesmos Public School, Parumala, Thiruvalla

**Contributor: Swapin Vidya**

**Publisher: PeachBot AI**

</div>

---

## About

AgriVision AI is an open-source smart agriculture working model for a school science and technology exhibition. It combines a Raspberry Pi 4, Google Coral USB Accelerator, Raspberry Pi camera, soil-moisture sensing, environmental monitoring, low-voltage irrigation, LEDs, buzzer feedback, and a local Flask dashboard.

The system demonstrates:

1. Real sensor-controlled irrigation through ADS1115 soil sensing and bounded pump control.
2. Real local AI inference using a Coral Edge TPU compiled TensorFlow Lite model.
3. A clearly labeled simulation mode for PC development only.

After setup, inference runs locally on the Pi/Coral. The dashboard and APIs explicitly identify simulation output when `AGRIVISION_SIMULATION=1` is used.

## Validated Model Release

Official model repository:

https://huggingface.co/peachbotAI/agrivision-mobilenetv2-edge-tpu

Required local runtime files:

- `models/plant_health_edgetpu.tflite`
- `models/labels.txt`

Download the published release files:

```bash
python3 scripts/fetch_models.py
```

Published model lineage:

- Dataset: `geraldmc/plantvillage-full`
- Dataset revision: `v0.1.0`
- Dataset license: CC0-1.0
- Task: binary `healthy` / `problem`
- Architecture: MobileNetV2 alpha 0.35
- Input: 224x224 RGB
- Initialization: scratch
- Export: full integer UINT8 TensorFlow Lite
- Compilation: Coral Edge TPU
- Edge TPU Compiler: 16.0.384591198
- Compiler result: 69 ops mapped to Edge TPU, 0 ops on CPU
- Compiled artifact: `plant_health_edgetpu.tflite`

Held-out PlantVillage evaluation:

| Artifact | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| FLOAT | 0.9865728900255755 | 0.9823966671562662 | 0.9830538063308789 |
| INT8 | 0.9857508220679576 | 0.9809942841356456 | 0.9820030388618228 |

Defensible summary: **98.58% held-out INT8 test accuracy on the PlantVillage dataset.**

Do not describe this as field accuracy. PlantVillage uses controlled-background leaf imagery and does not prove real farm performance under changing light, camera distance, cultivar, or disease conditions.

## Hardware

| Component | Purpose |
|---|---|
| Raspberry Pi 4 | Main controller and dashboard |
| Google Coral USB Accelerator | Edge TPU neural-network inference |
| Raspberry Pi Camera Module 3 | Leaf image capture |
| Capacitive soil moisture sensor | Soil moisture measurement |
| ADS1115 16-bit ADC | Analog-to-digital conversion |
| BME280 | Temperature and humidity |
| 5 V mini pump | Irrigation demonstration |
| Logic-level MOSFET driver | Safe pump switching |
| Green / yellow / red LEDs | Visible plant-health status |
| 3.3 V-compatible active buzzer | Alert |
| Separate 5 V pump supply | Keeps pump current away from Pi GPIO |

Never power a pump directly from a Raspberry Pi GPIO pin. Keep water, tubing, and wet soil physically separated from Pi, Coral, and power electronics.

## Quick Start On Raspberry Pi

```bash
git clone https://github.com/swapins/agrivision-ai-pi.git
cd agrivision-ai-pi
sudo bash scripts/install_pi.sh
sudo bash scripts/install_coral.sh
cp config.example.yaml config.yaml
python3 scripts/fetch_models.py
python3 scripts/calibrate_soil.py
python3 scripts/hardware_selftest.py
python3 scripts/hardware_selftest.py --pump   # only after checking wiring and water isolation
python3 scripts/coral_smoke_test.py path/to/leaf.jpg
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

Real deployment remains capable of requiring `models/plant_health_edgetpu.tflite` and `models/labels.txt`; startup fails outside simulation mode if those files are missing.

## PC Simulation Mode

Simulation mode is for software development and UI testing only:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp config.example.yaml config.yaml
AGRIVISION_SIMULATION=1 python -m agrivision.app
```

Simulation output must not be presented as real Raspberry Pi sensor, camera, or Coral Edge TPU output.

## Training

Kaggle GPU training support is kept in `training/`, with `training/kaggle_train.py` and `training/kaggle_train.ipynb` preserving the official model lineage and quality gates. Do not lower those gates for release claims.

Manual training workflows are available in GitHub Actions, but full training is not triggered by ordinary source-code pushes.

## Testing

```bash
pip install -e '.[dev]'
pytest -q
```

The tests are PC-safe and do not activate Raspberry Pi GPIO, camera, pump hardware, Kaggle training, or Hugging Face publishing.

## Repository Structure

```text
src/agrivision/          Raspberry Pi runtime and dashboard
scripts/                 install, calibration, self-test, model fetch, service scripts
models/                  runtime model location; downloaded artifacts are ignored
training/                reproducible MobileNetV2 training and release tooling
tests/                   PC-safe unit tests
docs/                    deployment, safety, hardware, model documentation
.github/workflows/       unit tests and manual training workflows
```

## Project Scope

AgriVision AI is a school STEM prototype, not a professional crop-diagnosis device. Use cautious wording such as "possible visible disease/problem" and keep the `UNCERTAIN` result enabled for low-confidence scans.

## Citation

If you reuse this project, please credit:

**Isis Saritha Swapin (Syndesmos Public School, Parumala, Thiruvalla). AgriVision AI: Raspberry Pi + Coral Edge TPU Smart Agriculture Project. PeachBot AI, 2026. Contributor: Swapin Vidya.**

Machine-readable citation metadata is available in `CITATION.cff`.

## License

Code is released under the MIT License. Dataset and trained-model artifacts may have their own terms; the official model release documents the dataset lineage and license.
