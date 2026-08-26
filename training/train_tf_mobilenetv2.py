#!/usr/bin/env python3
"""Train and export a Coral-friendly MobileNetV2 classifier.

Default release task: binary ``healthy`` vs ``problem`` using the full
PlantVillage dataset.  ``--task multiclass`` preserves all PlantVillage
classes.  A tiny dataset mode exists only to smoke-test the pipeline.

The script preserves the source dataset's leaf-grouped held-out split and
creates a validation fold from training leaf IDs by deterministic hashing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
import tensorflow as tf
from datasets import load_dataset

SEED = 260826


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    p.add_argument("--dataset", default="geraldmc/plantvillage-full")
    p.add_argument("--revision", default="v0.1.0")
    p.add_argument("--output", type=Path, default=Path("training/output/agrivision-mobilenetv2"))
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--head-epochs", type=int, default=4)
    p.add_argument("--finetune-epochs", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--finetune-learning-rate", type=float, default=1e-5)
    p.add_argument("--representative-samples", type=int, default=300)
    p.add_argument("--smoke", action="store_true", help="Use debug-grade plantvillage-tiny and short training")
    p.add_argument("--no-imagenet", action="store_true", help="Do not download/use ImageNet weights")
    return p.parse_args()


def stable_bucket(text: str, modulus: int = 10) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) % modulus


def build_index(ds, task: str):
    rows = []
    all_classes = sorted(set(ds["class_label"])) if task == "multiclass" else ["healthy", "problem"]
    class_to_id = {name: i for i, name in enumerate(all_classes)}

    class_labels = ds["class_label"]
    diseases = ds["disease"]
    source_splits = ds["split"]
    leaf_ids = ds["leaf_id"]

    for i, (class_label, disease, source_split, leaf_id) in enumerate(
        zip(class_labels, diseases, source_splits, leaf_ids)
    ):
        if task == "binary":
            target_name = "healthy" if str(disease).strip().lower() == "healthy" else "problem"
        else:
            target_name = class_label
        if source_split == "test":
            split = "test"
        else:
            split = "val" if stable_bucket(str(leaf_id), 10) == 0 else "train"
        rows.append((i, class_to_id[target_name], split))
    return rows, all_classes


def image_generator(ds, rows, image_size):
    for index, target, _ in rows:
        image = ds[index]["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        yield np.asarray(image, dtype=np.float32), np.int32(target)


def make_tf_dataset(ds, rows, image_size, batch_size, training):
    signature = (
        tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    out = tf.data.Dataset.from_generator(
        lambda: image_generator(ds, rows, image_size), output_signature=signature
    )
    if training:
        out = out.shuffle(min(len(rows), 4096), seed=SEED, reshuffle_each_iteration=True)
    out = out.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return out


def make_model(image_size, alpha, num_classes, imagenet=True):
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.12),
        ],
        name="augmentation",
    )
    x = aug(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_scaling")(x)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        alpha=alpha,
        include_top=False,
        weights="imagenet" if imagenet else None,
    )
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="plant_health")(x)
    return tf.keras.Model(inputs, outputs, name="agrivision_mobilenetv2"), base


def class_weights(rows):
    counts = Counter(target for _, target, split in rows if split == "train")
    total = sum(counts.values())
    n = len(counts)
    return {c: total / (n * count) for c, count in counts.items()}


def export_int8(model, ds, train_rows, out_path: Path, image_size: int, samples: int):
    indices = [r for r in train_rows if r[2] == "train"][:samples]

    def representative():
        for image, _ in image_generator(ds, indices, image_size):
            yield [np.expand_dims(image, 0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite = converter.convert()
    out_path.write_bytes(tflite)


def evaluate_tflite(path: Path, ds, test_rows, image_size: int):
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    correct = total = 0
    for image, target in image_generator(ds, test_rows, image_size):
        if inp["dtype"] == np.uint8:
            arr = np.clip(np.round(image), 0, 255).astype(np.uint8)
        else:
            arr = image.astype(inp["dtype"])
        interpreter.set_tensor(inp["index"], np.expand_dims(arr, 0))
        interpreter.invoke()
        scores = interpreter.get_tensor(out["index"])[0]
        pred = int(np.argmax(scores))
        correct += int(pred == int(target))
        total += 1
    return correct / total if total else 0.0


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)
    args.output.mkdir(parents=True, exist_ok=True)

    dataset_id = "geraldmc/plantvillage-tiny" if args.smoke else args.dataset
    if args.smoke:
        args.head_epochs = min(args.head_epochs, 1)
        args.finetune_epochs = min(args.finetune_epochs, 1)
    print(f"Loading {dataset_id} @ {args.revision}")
    ds = load_dataset(dataset_id, revision=args.revision, split="train")
    rows, labels = build_index(ds, args.task)
    train_rows = [r for r in rows if r[2] == "train"]
    val_rows = [r for r in rows if r[2] == "val"]
    test_rows = [r for r in rows if r[2] == "test"]
    print({"train": len(train_rows), "val": len(val_rows), "test": len(test_rows), "classes": len(labels)})

    train_ds = make_tf_dataset(ds, train_rows, args.image_size, args.batch_size, True)
    val_ds = make_tf_dataset(ds, val_rows, args.image_size, args.batch_size, False)
    test_ds = make_tf_dataset(ds, test_rows, args.image_size, args.batch_size, False)

    model, base = make_model(args.image_size, args.alpha, len(labels), not args.no_imagenet)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=1, factor=0.3),
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("Stage 1: classifier head")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        class_weight=class_weights(rows),
        callbacks=callbacks,
    )

    print("Stage 2: fine-tune final MobileNetV2 layers")
    base.trainable = True
    freeze_to = max(0, len(base.layers) - 30)
    for layer in base.layers[:freeze_to]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.finetune_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    if args.finetune_epochs:
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.finetune_epochs,
            class_weight=class_weights(rows),
            callbacks=callbacks,
        )

    float_loss, float_acc = model.evaluate(test_ds, verbose=1)
    keras_path = args.output / "plant_health.keras"
    model.save(keras_path)
    (args.output / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    int8_path = args.output / "plant_health_int8.tflite"
    export_int8(model, ds, train_rows, int8_path, args.image_size, args.representative_samples)
    int8_acc = evaluate_tflite(int8_path, ds, test_rows, args.image_size)

    manifest = {
        "project": "AgriVision AI",
        "architecture": "MobileNetV2",
        "alpha": args.alpha,
        "input_size": [args.image_size, args.image_size, 3],
        "task": args.task,
        "labels": labels,
        "dataset": dataset_id,
        "dataset_revision": args.revision,
        "split_policy": "source leaf-grouped test split; validation by deterministic leaf_id hash",
        "train_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "test_examples": len(test_rows),
        "float_test_accuracy": float(float_acc),
        "int8_test_accuracy": float(int8_acc),
        "smoke_dataset": bool(args.smoke),
        "edge_tpu_compiled": False,
        "limitations": [
            "PlantVillage is largely controlled-background imagery and does not establish field robustness.",
            "This model is a school/research prototype and is not professional agricultural diagnosis.",
        ],
    }
    (args.output / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"\nNext: training/compile_edgetpu.sh {int8_path}")


if __name__ == "__main__":
    main()
