---
library_name: tflite
tags:
- image-classification
- agriculture
- plant-disease
- raspberry-pi
- coral-edge-tpu
- tflite
license: mit
---

# AgriVision AI - MobileNetV2 Plant Health Classifier

**Developer:** Isis Saritha Swapin  
**School:** Syndesmos Public School, Parumala, Thiruvalla  
**Current class:** Class 8  
**Contributor:** Swapin Vidya  
**Publisher:** PeachBot AI

This model is developed for AgriVision AI by Isis Saritha Swapin, studying in Class 8 at Syndesmos Public School, Parumala, Thiruvalla. Swapin Vidya is credited as project contributor.

## Intended Use

Local leaf-image classification on a Raspberry Pi 4 + Google Coral USB Accelerator school demonstration.

Release files:

- `plant_health_int8.tflite` - full-integer TFLite before Edge TPU compilation.
- `plant_health_edgetpu.tflite` - Edge TPU compiled deployment artifact.
- `labels.txt` - output labels.
- `model_manifest.json` - architecture, dataset, split, evaluation, and provenance metadata.
- `training_history.json` - training history.

## Architecture

MobileNetV2 alpha 0.35, 224x224 RGB, scratch initialization, binary `healthy` / `problem` task, full integer UINT8 TFLite export.

## Dataset

Dataset: `geraldmc/plantvillage-full`

Revision: `v0.1.0`

License: CC0-1.0

## Validation

Use exact metrics from the release manifest. Safe public wording:

> 98.58% held-out INT8 test accuracy on the PlantVillage dataset.

Do not call this field accuracy.

## Important Limitation

PlantVillage images are largely controlled-background leaf images. Strong held-out performance on this dataset does not establish robustness in real farms. This is a student-developed school science and technology project, not a crop-diagnosis device. Predictions should be phrased as possible visible disease/problem and confirmed by appropriate agricultural expertise.

## Reproducibility

Training and TFLite export code is in the accompanying AgriVision AI GitHub repository. See `training/train_tf_mobilenetv2.py`, `training/kaggle_train.py`, and `training/compile_edgetpu.sh`.
