# Model artifacts

Final Pi deployment expects:

- `plant_health_edgetpu.tflite` — fully integer TFLite model compiled with the Edge TPU Compiler.
- `labels.txt` — one label per line in model output order, or `index label` format.
- `model_manifest.json` — optional provenance / training metadata.

Model binaries are ignored by Git by default because they can be regenerated or hosted on Hugging Face. A release script can download the published artifacts during deployment.

For the school demo, do not rename a CPU-only `.tflite` file to `_edgetpu.tflite`. Validate the compiled model with `scripts/coral_smoke_test.py`.
