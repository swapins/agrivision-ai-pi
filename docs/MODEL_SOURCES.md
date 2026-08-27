# Model Source Evidence

AgriVision AI trains and deploys its own MobileNetV2 alpha 0.35 classifier for Raspberry Pi 4 + Coral USB Accelerator deployment.

Primary project release:

- Hugging Face model: https://huggingface.co/peachbotAI/agrivision-mobilenetv2-edge-tpu
- Dataset: https://huggingface.co/datasets/geraldmc/plantvillage-full
- Dataset revision: `v0.1.0`
- Dataset license: CC0-1.0

Public architecture references:

- Google Coral Edge TPU: https://www.coral.ai/
- TensorFlow Lite: https://www.tensorflow.org/lite
- STMicroelectronics MobileNetV2 alpha 0.35 / 224 px reference: https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/main/image_classification/mobilenetv2/README.md

The ST model zoo entry is evidence that this architecture family is appropriate for compact edge vision. It is not redistributed as AgriVision AI weights and is not presented as a model trained by this project.

AgriVision release metrics are the metrics recorded for the PeachBot AI trained model and manifest, not copied from a third-party model.

Controlled-background PlantVillage results must not be described as general field performance.
