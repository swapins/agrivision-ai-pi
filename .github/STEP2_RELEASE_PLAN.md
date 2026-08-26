# Step 2 — AgriVision AI Model Release Plan

**Developer:** Isis Saritha Swapin  
**Contributor:** Swapin Vidya  
**Publisher:** PeachBot AI

Target model repository: `peachbotAI/agrivision-mobilenetv2-edge-tpu`.

The first official release is a binary `healthy` / `problem` MobileNetV2 alpha 0.35 model at 224×224, trained from scratch on the CC0 PlantVillage release, evaluated on its held-out leaf-grouped test assignment, quantized to full-integer INT8 TFLite, and gated before publication.

ImageNet initialization remains available only as an explicitly marked research/educational comparison and is not the default official PeachBot release lineage.

Publication should use Hugging Face Trusted Publishers/OIDC from `swapins/agrivision-ai-pi` / `main` / `train-and-publish-hf.yml`, avoiding a permanent Hugging Face token in GitHub secrets.
