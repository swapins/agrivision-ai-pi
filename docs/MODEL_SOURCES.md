# Model selection evidence

AgriVision AI uses a **MobileNetV2 alpha 0.35, 224×224** transfer-learning target for the Coral path.

Why this family:

- Google Coral is built for mobile/edge TensorFlow Lite inference, and MobileNetV2 is a canonical Edge TPU vision architecture.
- STMicroelectronics publishes a MobileNetV2 α=0.35 / 224 px reference trained on a PlantVillage-style plant-leaf dataset; its reported INT8 top-1 accuracy is 99.68% on that controlled dataset. This is evidence that the architecture is sufficiently expressive for leaf classification, **not** a claim about this AgriVision model.
- The release dataset is `geraldmc/plantvillage-full`: 54,304 controlled-background images across 38 host/disease classes with a provided leaf-grouped held-out split. The project preserves the source test split to reduce same-leaf leakage.

Primary public references:

- https://huggingface.co/datasets/geraldmc/plantvillage-full
- https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/main/image_classification/mobilenetv2/README.md
- https://www.coral.ai/

## Important distinction

The ST model is a **reference**, not redistributed as AgriVision weights and not presented as a model trained by this project. AgriVision's `training/train_tf_mobilenetv2.py` trains its own classifier and records its own held-out float/INT8 metrics in `model_manifest.json`.

PlantVillage's controlled backgrounds make it good for a school proof-of-concept but insufficient for claims of general field diagnosis. Exhibition language must remain “possible disease/problem.”
