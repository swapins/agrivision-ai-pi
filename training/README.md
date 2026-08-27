# Training And Model Release

**Project developer:** Isis Saritha Swapin  
**Contributor:** Swapin Vidya  
**Publisher:** PeachBot AI

## Official Release

The validated model is published at:

https://huggingface.co/peachbotAI/agrivision-mobilenetv2-edge-tpu

Runtime deployments should download the public release with:

```bash
python3 scripts/fetch_models.py
```

## Lineage

The official training path is fixed:

- Dataset: `geraldmc/plantvillage-full`
- Dataset revision: `v0.1.0`
- Dataset license: CC0-1.0
- Task: binary `healthy` / `problem`
- Architecture: MobileNetV2 alpha 0.35 at 224x224 RGB
- Initialization: scratch
- Export: full integer UINT8 TFLite
- Target: Raspberry Pi 4 + Coral USB Accelerator

Do not change this lineage or lower release quality gates when reproducing the official model.

## Kaggle GPU Path

Use GitHub as the source of truth for code and Kaggle for GPU-backed reproduction:

```bash
python training/kaggle_train.py
```

The notebook `training/kaggle_train.ipynb` is provided for Kaggle's notebook UI.

Do not run full PlantVillage training on a normal development PC. The `--smoke` option is only for checking the pipeline and cannot support release claims.

## Local Smoke Test

```bash
python training/train_tf_mobilenetv2.py --task binary --init scratch --smoke
```

Smoke artifacts are explicitly marked non-release in `model_manifest.json`.

## Metrics From Official Release

| Artifact | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| FLOAT | 0.9865728900255755 | 0.9823966671562662 | 0.9830538063308789 |
| INT8 | 0.9857508220679576 | 0.9809942841356456 | 0.9820030388618228 |

Use this wording:

> 98.58% held-out INT8 test accuracy on the PlantVillage dataset.

Do not call it field accuracy.

## Coral Compilation

The official release includes:

- `plant_health_int8.tflite`
- `plant_health_edgetpu.tflite`
- `labels.txt`
- `model_manifest.json`
- `training_history.json`

Compilation facts:

- Edge TPU Compiler: 16.0.384591198
- 69 ops mapped to Edge TPU
- 0 ops on CPU

Verify on the Pi/Coral:

```bash
python3 scripts/coral_smoke_test.py path/to/leaf.jpg
```

## Publishing

Do not publish new Hugging Face artifacts unless you intentionally trained and validated a new official release. Never commit tokens or model binaries.
