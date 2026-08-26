#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:-training/output/agrivision-mobilenetv2/plant_health_int8.tflite}"
OUTDIR="${2:-training/output/agrivision-mobilenetv2/edgetpu}"
mkdir -p "$OUTDIR"
command -v edgetpu_compiler >/dev/null || { echo "edgetpu_compiler not found. Install it on supported x86-64 Linux." >&2; exit 2; }
edgetpu_compiler -s -o "$OUTDIR" "$MODEL"
COMPILED=$(find "$OUTDIR" -maxdepth 1 -name '*_edgetpu.tflite' -print -quit)
[[ -n "$COMPILED" ]] || { echo "Compiler did not produce an Edge TPU model" >&2; exit 3; }
cp "$COMPILED" models/plant_health_edgetpu.tflite
cp "$(dirname "$MODEL")/labels.txt" models/labels.txt
cp "$(dirname "$MODEL")/model_manifest.json" models/model_manifest.json
python3 - <<'PY'
import json
p='models/model_manifest.json'
d=json.load(open(p))
d['edge_tpu_compiled']=True
json.dump(d,open(p,'w'),indent=2)
PY
echo "Installed: models/plant_health_edgetpu.tflite"
