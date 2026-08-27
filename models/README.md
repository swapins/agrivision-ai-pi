# Model artifacts

Final Pi deployment expects:

- `plant_health_edgetpu.tflite` — fully integer TFLite model compiled with the Edge TPU Compiler.
- `labels.txt` — one label per line in model output order, or `index label` format.
- `model_manifest.json` — optional provenance / training metadata.

Model binaries and downloaded release metadata are ignored by Git by default. Fetch the public PeachBot AI release during deployment:

```bash
python3 scripts/fetch_models.py
```

The script downloads the public Hugging Face release from:

https://huggingface.co/peachbotAI/agrivision-mobilenetv2-edge-tpu

Required local runtime files:

- `plant_health_edgetpu.tflite`
- `plant_health_int8.tflite`
- `labels.txt`
- `model_manifest.json`

For the school demo, do not rename a CPU-only `.tflite` file to `_edgetpu.tflite`. Validate the compiled model with `scripts/coral_smoke_test.py`.
