# Training and model release

**Project developer:** Isis Saritha Swapin  
**Contributor:** Swapin Vidya  
**Brand:** PeachBot

## Selected base model

**MobileNetV2 alpha 0.35 at 224×224** is the default. It is small, fast, well understood, and designed for mobile/edge inference. The project uses transfer learning, then exports a fully integer INT8 TFLite model for the Coral compiler.

## Dataset choice

The default is `geraldmc/plantvillage-full` because it packages 54k+ PlantVillage images with metadata and a leaf-grouped held-out split. A `--smoke` mode uses `geraldmc/plantvillage-tiny` only to verify code; the tiny dataset is explicitly not suitable for final claims.

The reliable first release is `--task binary`:

- `healthy`
- `problem`

This is deliberately simpler than inventing a visual “stress” class from unrelated datasets. The physical project already demonstrates water stress with the soil sensor. Use `--task multiclass` if you want the full PlantVillage disease classes.

## Train

Use a Linux/Colab environment with TensorFlow 2.15 and, ideally, a GPU:

```bash
python -m venv .venv-train
source .venv-train/bin/activate
pip install -r training/requirements.txt
python training/train_tf_mobilenetv2.py --task binary
```

Fast pipeline smoke test only:

```bash
python training/train_tf_mobilenetv2.py --task binary --smoke
```

Artifacts are written under `training/output/agrivision-mobilenetv2/`.

## Compile for Coral

The Edge TPU compiler runs on supported x86-64 Linux environments. Install the compiler per Coral documentation, then:

```bash
training/compile_edgetpu.sh
```

Read the compiler summary. Do not claim Coral acceleration if the model has not been compiled and validated.

## Publish to Hugging Face

Authenticate locally without embedding a token in the repository:

```bash
hf auth login
python training/publish_hf.py --repo-id YOUR_HF_USERNAME/agrivision-mobilenetv2-edge-tpu
```

The publishing script uploads the INT8 model, compiled Edge TPU model if present, labels, manifest, and model card.

### GitHub Actions publication

After this repository is hosted on GitHub, you may also add a repository secret named `HF_TOKEN` and manually run `.github/workflows/train-and-publish-hf.yml`. The workflow refuses to publish a `--smoke` run, so the debug dataset cannot accidentally become the final Hugging Face release.

## Recommended release gate

Before calling a model “final”:

1. Train on the full dataset, not the debug subset.
2. Preserve leaf-level split integrity.
3. Record float and INT8 held-out metrics in `model_manifest.json`.
4. Confirm the Edge TPU compiler succeeds.
5. Run `scripts/coral_smoke_test.py` on the actual Raspberry Pi + Coral.
6. Test at least 10 previously unseen exhibition images/cards under the actual room lighting.
7. Keep the UI confidence threshold and `UNCERTAIN` result enabled.
