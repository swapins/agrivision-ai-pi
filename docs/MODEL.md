# Model design and scientific scope

## Why MobileNetV2

AgriVision AI targets a Raspberry Pi 4 plus Coral USB Accelerator. MobileNetV2 is compact, widely supported, and is a natural fit for integer edge inference. The training pipeline defaults to width multiplier `alpha=0.35` and 224×224 RGB images to reduce compute while retaining enough spatial detail for a school demonstration.

## Default classification task

The recommended first release is **Healthy vs Problem** using PlantVillage. A multiclass mode preserves all source disease classes.

The project PDF originally describes Healthy / Disease / Stress as a three-class image classifier. This repository avoids creating an unsupported visual `stress` label from synthetic yellowing or by mixing unrelated datasets without validation. Instead:

- vision model: healthy / disease-problem (or exact disease class),
- soil sensor: actual water-stress signal,
- dashboard: can still show HEALTHY / DISEASE / STRESS based on the appropriate evidence source.

If a validated visual-stress dataset is later added, the runtime already understands labels containing `stress`, `wilt`, or `yellow`.

## Dataset

Default: `geraldmc/plantvillage-full` on Hugging Face.

Training respects its source `split` metadata and constructs validation data by deterministic hashing of `leaf_id`, so the same physical leaf does not intentionally appear in both train and validation folds.

## Limitations

PlantVillage contains controlled-background leaf imagery. A model can score highly on held-out PlantVillage data while performing much worse in field conditions. Do not use the model as professional agricultural diagnosis. Use fixed lighting / camera distance at the exhibition and show `UNCERTAIN` when confidence is low.

## Optional field robustness test

A later research extension can evaluate on PlantDoc, which contains more heterogeneous real-world imagery. Treat this as a domain-shift test, not as a direct replacement for crop-specific field validation.
