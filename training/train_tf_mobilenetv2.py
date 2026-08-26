#!/usr/bin/env python3
"""Train and export the AgriVision AI Coral-friendly MobileNetV2 classifier.

Official PeachBot release lineage:
- Dataset: geraldmc/plantvillage-full @ v0.1.0 (CC0-1.0)
- Task: binary healthy vs problem
- Architecture: MobileNetV2 alpha 0.35, 224x224 RGB
- Initialization: scratch (no ImageNet weights)
- Export: full-integer UINT8 TensorFlow Lite

ImageNet initialization remains available only as an explicit research comparison.
The tiny dataset mode is strictly for pipeline smoke tests and must never be
presented as a trained release model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

SEED = 260826
OFFICIAL_DATASET = "geraldmc/plantvillage-full"
SMOKE_DATASET = "geraldmc/plantvillage-tiny"
DATASET_REVISION = "v0.1.0"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    p.add_argument("--dataset", default=OFFICIAL_DATASET)
    p.add_argument("--revision", default=DATASET_REVISION)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("training/output/agrivision-mobilenetv2"),
    )
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--init",
        choices=["scratch", "imagenet"],
        default="scratch",
        help="Official releases use scratch. ImageNet is research-comparison only.",
    )
    p.add_argument("--scratch-epochs", type=int, default=15)
    p.add_argument("--head-epochs", type=int, default=4)
    p.add_argument("--finetune-epochs", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--head-learning-rate", type=float, default=1e-3)
    p.add_argument("--finetune-learning-rate", type=float, default=1e-5)
    p.add_argument("--representative-samples", type=int, default=300)
    p.add_argument(
        "--min-int8-accuracy",
        type=float,
        default=None,
        help="Fail after evaluation when INT8 held-out accuracy is below this value.",
    )
    p.add_argument(
        "--require-official-lineage",
        action="store_true",
        help="Fail unless using the official full dataset, revision, binary task, and scratch init.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Use debug-grade plantvillage-tiny and one short epoch. Never a final release.",
    )
    p.add_argument("--no-imagenet", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def stable_bucket(text: str, modulus: int = 10) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) % modulus


def build_index(ds, task: str):
    rows = []
    all_classes = (
        sorted(set(ds["class_label"]))
        if task == "multiclass"
        else ["healthy", "problem"]
    )
    class_to_id = {name: i for i, name in enumerate(all_classes)}

    class_labels = ds["class_label"]
    diseases = ds["disease"]
    source_splits = ds["split"]
    leaf_ids = ds["leaf_id"]

    for i, (class_label, disease, source_split, leaf_id) in enumerate(
        zip(class_labels, diseases, source_splits, leaf_ids)
    ):
        if task == "binary":
            target_name = (
                "healthy" if str(disease).strip().lower() == "healthy" else "problem"
            )
        else:
            target_name = class_label

        if source_split == "test":
            split = "test"
        else:
            split = "val" if stable_bucket(str(leaf_id), 10) == 0 else "train"

        rows.append((i, class_to_id[target_name], split))
    return rows, all_classes


def validate_split_integrity(ds, rows):
    """Verify no leaf_id appears across train/val/test in our constructed index."""
    split_by_leaf = defaultdict(set)
    for index, _, split in rows:
        split_by_leaf[str(ds[index]["leaf_id"])].add(split)
    leaking = sorted(
        leaf_id for leaf_id, splits in split_by_leaf.items() if len(splits) > 1
    )
    if leaking:
        raise RuntimeError(
            f"Split-integrity failure: {len(leaking)} leaf_id values cross split boundaries."
        )


def image_generator(ds, rows, image_size):
    for index, target, _ in rows:
        image = ds[index]["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB").resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
        yield np.asarray(image, dtype=np.float32), np.int32(target)


def make_tf_dataset(ds, rows, image_size, batch_size, training):
    signature = (
        tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    out = tf.data.Dataset.from_generator(
        lambda: image_generator(ds, rows, image_size),
        output_signature=signature,
    )
    if training:
        out = out.shuffle(
            min(len(rows), 4096), seed=SEED, reshuffle_each_iteration=True
        )
    return out.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(image_size, alpha, num_classes, init_mode):
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
    x = tf.keras.layers.Rescaling(
        1.0 / 127.5, offset=-1.0, name="mobilenet_scaling"
    )(x)

    weights = "imagenet" if init_mode == "imagenet" else None
    base = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        alpha=alpha,
        include_top=False,
        weights=weights,
    )
    x = base(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="plant_health"
    )(x)
    return tf.keras.Model(inputs, outputs, name="agrivision_mobilenetv2"), base


def class_weights(rows):
    counts = Counter(target for _, target, split in rows if split == "train")
    total = sum(counts.values())
    n = len(counts)
    return {c: total / (n * count) for c, count in counts.items()}


def callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=1, factor=0.3, min_lr=1e-7
        ),
    ]


def train_model(model, base, args, train_ds, val_ds, weights):
    if args.init == "scratch":
        base.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print("Training official scratch-initialized MobileNetV2")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.scratch_epochs,
            class_weight=weights,
            callbacks=callbacks(),
        )
        return

    base.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.head_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("Stage 1: ImageNet research comparison — classifier head")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        class_weight=weights,
        callbacks=callbacks(),
    )

    print("Stage 2: ImageNet research comparison — fine-tune final layers")
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
            class_weight=weights,
            callbacks=callbacks(),
        )


def representative_rows(train_rows, samples, num_classes):
    """Choose a deterministic, roughly class-balanced calibration sample."""
    by_class = defaultdict(list)
    for row in train_rows:
        by_class[row[1]].append(row)

    rng = np.random.default_rng(SEED)
    for values in by_class.values():
        rng.shuffle(values)

    selected = []
    per_class = max(1, samples // max(1, num_classes))
    for class_id in sorted(by_class):
        selected.extend(by_class[class_id][:per_class])

    if len(selected) < samples:
        used = {(r[0], r[1], r[2]) for r in selected}
        remaining = [r for r in train_rows if (r[0], r[1], r[2]) not in used]
        rng.shuffle(remaining)
        selected.extend(remaining[: samples - len(selected)])
    return selected[:samples]


def export_int8(
    model,
    ds,
    train_rows,
    out_path: Path,
    image_size: int,
    samples: int,
    num_classes: int,
):
    indices = representative_rows(train_rows, samples, num_classes)

    def representative():
        for image, _ in image_generator(ds, indices, image_size):
            yield [np.expand_dims(image, 0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    out_path.write_bytes(converter.convert())


def metrics_from_predictions(y_true, y_pred, labels):
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(labels)))
        ).tolist(),
        "classification_report": report,
    }


def evaluate_float(model, ds, test_rows, image_size, labels):
    y_true = []
    y_pred = []
    for image, target in image_generator(ds, test_rows, image_size):
        scores = model.predict(np.expand_dims(image, 0), verbose=0)[0]
        y_true.append(int(target))
        y_pred.append(int(np.argmax(scores)))
    return metrics_from_predictions(y_true, y_pred, labels)


def evaluate_tflite(path: Path, ds, test_rows, image_size: int, labels):
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    y_true = []
    y_pred = []
    for image, target in image_generator(ds, test_rows, image_size):
        if inp["dtype"] == np.uint8:
            arr = np.clip(np.round(image), 0, 255).astype(np.uint8)
        else:
            arr = image.astype(inp["dtype"])

        interpreter.set_tensor(inp["index"], np.expand_dims(arr, 0))
        interpreter.invoke()
        scores = interpreter.get_tensor(out["index"])[0]
        y_true.append(int(target))
        y_pred.append(int(np.argmax(scores)))

    return metrics_from_predictions(y_true, y_pred, labels)


def enforce_official_lineage(args, dataset_id):
    if not args.require_official_lineage:
        return
    expected = {
        "dataset": OFFICIAL_DATASET,
        "revision": DATASET_REVISION,
        "task": "binary",
        "init": "scratch",
        "smoke": False,
    }
    actual = {
        "dataset": dataset_id,
        "revision": args.revision,
        "task": args.task,
        "init": args.init,
        "smoke": bool(args.smoke),
    }
    if actual != expected:
        raise RuntimeError(
            "Official-lineage gate failed.\n"
            f"Expected: {expected}\nActual: {actual}"
        )


def main():
    args = parse_args()
    if args.no_imagenet:
        args.init = "scratch"

    if args.smoke:
        args.dataset = SMOKE_DATASET
        args.scratch_epochs = 1
        args.head_epochs = 1
        args.finetune_epochs = 0
        args.representative_samples = min(args.representative_samples, 96)

    dataset_id = args.dataset
    enforce_official_lineage(args, dataset_id)

    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading {dataset_id} @ {args.revision}")
    ds = load_dataset(dataset_id, revision=args.revision, split="train")
    rows, labels = build_index(ds, args.task)
    validate_split_integrity(ds, rows)

    train_rows = [r for r in rows if r[2] == "train"]
    val_rows = [r for r in rows if r[2] == "val"]
    test_rows = [r for r in rows if r[2] == "test"]
    print(
        {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
            "classes": len(labels),
            "initialization": args.init,
        }
    )

    train_ds = make_tf_dataset(
        ds, train_rows, args.image_size, args.batch_size, True
    )
    val_ds = make_tf_dataset(ds, val_rows, args.image_size, args.batch_size, False)

    model, base = build_model(args.image_size, args.alpha, len(labels), args.init)
    train_model(model, base, args, train_ds, val_ds, class_weights(rows))

    float_metrics = evaluate_float(model, ds, test_rows, args.image_size, labels)

    keras_path = args.output / "plant_health.keras"
    model.save(keras_path)
    (args.output / "labels.txt").write_text(
        "\n".join(labels) + "\n", encoding="utf-8"
    )

    int8_path = args.output / "plant_health_int8.tflite"
    export_int8(
        model,
        ds,
        train_rows,
        int8_path,
        args.image_size,
        args.representative_samples,
        len(labels),
    )
    int8_metrics = evaluate_tflite(
        int8_path, ds, test_rows, args.image_size, labels
    )

    manifest = {
        "project": "AgriVision AI",
        "developer": "Isis Saritha Swapin",
        "contributor": "Swapin Vidya",
        "publisher": "PeachBot AI",
        "architecture": "MobileNetV2",
        "alpha": args.alpha,
        "input_size": [args.image_size, args.image_size, 3],
        "initialization": args.init,
        "official_release_lineage": (
            not args.smoke
            and dataset_id == OFFICIAL_DATASET
            and args.revision == DATASET_REVISION
            and args.task == "binary"
            and args.init == "scratch"
        ),
        "task": args.task,
        "labels": labels,
        "dataset": dataset_id,
        "dataset_revision": args.revision,
        "dataset_license": "CC0-1.0",
        "split_policy": (
            "Source held-out test assignment; validation by deterministic leaf_id hash. "
            "PlantVillage metadata has genuine leaf grouping for most, but not all, classes."
        ),
        "train_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "test_examples": len(test_rows),
        "float_test_metrics": float_metrics,
        "int8_test_metrics": int8_metrics,
        "smoke_dataset": bool(args.smoke),
        "edge_tpu_compiled": False,
        "limitations": [
            "PlantVillage is controlled-background imagery and does not establish field robustness.",
            "Some PlantVillage classes use synthetic leaf IDs; see the source dataset card.",
            "This model is a school/research prototype and is not professional agricultural diagnosis.",
        ],
    }
    (args.output / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2))
    print(f"\nNext: training/compile_edgetpu.sh {int8_path}")

    if args.min_int8_accuracy is not None:
        measured = float(int8_metrics["accuracy"])
        if measured < args.min_int8_accuracy:
            raise SystemExit(
                "Release gate failed: "
                f"INT8 held-out accuracy {measured:.4f} "
                f"< required {args.min_int8_accuracy:.4f}"
            )


if __name__ == "__main__":
    main()
