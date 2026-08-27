# Model Design And Scientific Scope

## Official Release

AgriVision AI uses the published PeachBot AI model release:

https://huggingface.co/peachbotAI/agrivision-mobilenetv2-edge-tpu

Expected release files:

- `plant_health_int8.tflite`
- `plant_health_edgetpu.tflite`
- `labels.txt`
- `model_manifest.json`
- `training_history.json`
- `README.md`

Runtime deployment requires:

- `models/plant_health_edgetpu.tflite`
- `models/labels.txt`

Use `python3 scripts/fetch_models.py` to download the public release artifacts.

## Lineage

- Dataset: `geraldmc/plantvillage-full`
- Dataset revision: `v0.1.0`
- Dataset license: CC0-1.0
- Task: binary `healthy` / `problem`
- Architecture: MobileNetV2 alpha 0.35
- Input: 224x224 RGB
- Initialization: scratch
- Export: full integer UINT8 TensorFlow Lite
- Deployment target: Raspberry Pi 4 + Google Coral USB Accelerator

## Metrics

| Artifact | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| FLOAT | 0.9865728900255755 | 0.9823966671562662 | 0.9830538063308789 |
| INT8 | 0.9857508220679576 | 0.9809942841356456 | 0.9820030388618228 |

Defensible wording: **98.58% held-out INT8 test accuracy on the PlantVillage dataset.**

Do not call this field accuracy. The dataset contains controlled-background plant leaf images.

## Coral Compilation

- Edge TPU Compiler: 16.0.384591198
- Compiled artifact: `plant_health_edgetpu.tflite`
- Compiler result: 69 ops mapped to Edge TPU, 0 ops on CPU

Do not claim Coral inference from an uncompiled CPU-only `.tflite` file. The runtime expects the compiled Edge TPU artifact outside simulation mode.

## Labels And Runtime Mapping

The release is a binary classifier:

- `healthy`
- `problem`

The dashboard maps healthy outputs to `HEALTHY`. Problem outputs are shown as `DISEASE` / possible visible disease-problem so the exhibition can demonstrate alert behavior without claiming a specific field diagnosis.

Water stress is demonstrated through the soil sensor. The project does not fabricate a visual stress class from unrelated data.

## Limitations

PlantVillage is useful for a controlled school proof of concept, but real farms introduce lighting, background, camera, cultivar, and disease-stage differences. Keep the `UNCERTAIN` pathway enabled and validate physically on the Pi/Coral before presenting a hardware demo.
