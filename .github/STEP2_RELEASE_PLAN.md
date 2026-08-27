# Step 2 - AgriVision AI Model Release Notes

**Developer:** Isis Saritha Swapin  
**Contributor:** Swapin Vidya  
**Publisher:** PeachBot AI

Official model repository:

`peachbotAI/agrivision-mobilenetv2-edge-tpu`

The official release is a binary `healthy` / `problem` MobileNetV2 alpha 0.35 model at 224x224 RGB, trained from scratch on `geraldmc/plantvillage-full` revision `v0.1.0`, evaluated on held-out PlantVillage data, quantized to full-integer UINT8 TFLite, and compiled for Coral Edge TPU.

Release artifacts are downloaded for Pi deployment with:

```bash
python3 scripts/fetch_models.py
```

Release wording must remain cautious: 98.58% held-out INT8 test accuracy on the PlantVillage dataset, not field accuracy.

Full training and publication workflows are manual-only. Do not publish a new Hugging Face model unless a new official release has been intentionally trained and validated.
