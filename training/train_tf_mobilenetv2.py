#!/usr/bin/env python3
"""Train and export the AgriVision AI Coral-friendly MobileNetV2 classifier.

Official PeachBot release lineage:
- Dataset: geraldmc/plantvillage-full @ v0.1.0 (CC0-1.0)
- Task: binary healthy vs problem
- Architecture: MobileNetV2 alpha 0.35, 224x224 RGB
- Initialization: scratch (no ImageNet weights)
- Export: full-integer UINT8 TensorFlow Lite

The official binary path deliberately uses deterministic class-balanced
oversampling and multiple release metrics. A model that predicts only one
class is rejected even when raw accuracy looks acceptable.
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
        "--output", type=Path,
        default=Path("training/output/agrivision-mobilenetv2"),
    )
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--init", choices=["scratch", "imagenet"], default="scratch",
        help="Official releases use scratch. ImageNet is research-comparison only.",
    )
    p.add_argument("--scratch-epochs", type=int, default=20)
    p.add_argument("--head-epochs", type=int, default=4)
    p.add_argument("--finetune-epochs", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--head-learning-rate", type=float, default=1e-3)
    p.add_argument("--finetune-learning-rate", type=float, default=1e-5)
    p.add_argument("--representative-samples", type=int, default=300)
    p.add_argument("--min-int8-accuracy", type=float, default=None)
    p.add_argument("--min-int8-balanced-accuracy", type=float, default=None)
    p.add_argument("--min-int8-macro-f1", type=float, default=None)
    p.add_argument("--require-all-classes-predicted", action="store_true")
    p.add_argument("--sanity-samples-per-class", type=int, default=32)
    p.add_argument("--sanity-epochs", type=int, default=10)
    p.add_argument("--min-sanity-accuracy", type=float, default=0.85)
    p.add_argument("--skip-sanity-check", action="store_true")
    p.add_argument(
        "--require-official-lineage", action="store_true",
        help="Fail unless using the official full dataset, revision, binary task, and scratch init.",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Use debug-grade plantvillage-tiny and one short epoch. Never a final release.",
    )
    p.add_argument("--no-imagenet", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def stable_bucket(text: str, modulus: int = 10) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) % modulus


def build_index(ds, task: str):
    rows = []
    all_classes = sorted(set(ds["class_label"])) if task == "multiclass" else ["healthy", "problem"]
    class_to_id = {name: i for i, name in enumerate(all_classes)}

    for i, (class_label, disease, source_split, leaf_id) in enumerate(
        zip(ds["class_label"], ds["disease"], ds["split"], ds["leaf_id"])
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


def validate_split_integrity(ds, rows):
    split_by_leaf = defaultdict(set)
    for index, _, split in rows:
        split_by_leaf[str(ds[index]["leaf_id"])].add(split)
    leaking = sorted(leaf_id for leaf_id, splits in split_by_leaf.items() if len(splits) > 1)
    if leaking:
        raise RuntimeError(
            f"Split-integrity failure: {len(leaking)} leaf_id values cross split boundaries."
        )


def split_class_counts(rows, labels):
    result = {}
    for split in ("train", "val", "test"):
        counts = Counter(target for _, target, row_split in rows if row_split == split)
        result[split] = {label: int(counts.get(i, 0)) for i, label in enumerate(labels)}
    return result


def balance_rows(rows):
    """Deterministically oversample every training class to the largest class."""
    by_class = defaultdict(list)
    for row in rows:
        by_class[row[1]].append(row)
    if not by_class:
        raise RuntimeError("Cannot balance an empty training split.")

    target_size = max(len(values) for values in by_class.values())
    rng = np.random.default_rng(SEED)
    balanced = []
    for class_id in sorted(by_class):
        values = list(by_class[class_id])
        if not values:
            raise RuntimeError(f"Training class {class_id} has no examples.")
        balanced.extend(values)
        if len(values) < target_size:
            extra_idx = rng.choice(len(values), size=target_size - len(values), replace=True)
            balanced.extend(values[int(i)] for i in extra_idx)
    rng.shuffle(balanced)
    return balanced


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
        lambda: image_generator(ds, rows, image_size), output_signature=signature
    )
    if training:
        out = out.shuffle(min(len(rows), 8192), seed=SEED, reshuffle_each_iteration=True)
    return out.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def print_tensorflow_runtime():
    gpus = tf.config.list_physical_devices("GPU")
    gpu_names = [gpu.name for gpu in gpus]
    built_with_cuda = bool(tf.test.is_built_with_cuda())
    using_cuda_gpu = bool(gpus) and built_with_cuda
    policy = tf.keras.mixed_precision.global_policy()
    print("TensorFlow runtime:")
    print(f"  version: {tf.__version__}")
    print(f"  physical GPUs: {gpu_names if gpu_names else 'none'}")
    print(f"  CUDA GPU in use: {using_cuda_gpu}")
    print(f"  mixed precision policy: {policy.name}")


def build_model(image_size, alpha, num_classes, init_mode, augment=True):
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = inputs
    if augment:
        x = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.08),
                tf.keras.layers.RandomZoom(0.10),
                tf.keras.layers.RandomContrast(0.12),
            ],
            name="augmentation",
        )(x)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_scaling")(x)

    weights = "imagenet" if init_mode == "imagenet" else None
    base = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3), alpha=alpha,
        include_top=False, weights=weights,
    )
    x = base(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="plant_health")(x)
    return tf.keras.Model(inputs, outputs, name="agrivision_mobilenetv2"), base


def callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, min_delta=1e-4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=2, factor=0.3, min_lr=1e-7
        ),
    ]


def history_to_json(history):
    return {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }


def train_model(model, base, args, train_ds, val_ds):
    histories = []
    if args.init == "scratch":
        base.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate),
            loss="sparse_categorical_crossentropy", metrics=["accuracy"],
        )
        print("Training scratch-initialized MobileNetV2 with balanced oversampling")
        h = model.fit(
            train_ds, validation_data=val_ds, epochs=args.scratch_epochs,
            callbacks=callbacks(),
        )
        histories.append({"stage": "scratch", "history": history_to_json(h)})
        return histories

    base.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.head_learning_rate),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"],
    )
    h = model.fit(
        train_ds, validation_data=val_ds, epochs=args.head_epochs, callbacks=callbacks()
    )
    histories.append({"stage": "imagenet_head", "history": history_to_json(h)})

    base.trainable = True
    freeze_to = max(0, len(base.layers) - 30)
    for layer in base.layers[:freeze_to]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.finetune_learning_rate),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"],
    )
    if args.finetune_epochs:
        h = model.fit(
            train_ds, validation_data=val_ds, epochs=args.finetune_epochs,
            callbacks=callbacks(),
        )
        histories.append({"stage": "imagenet_finetune", "history": history_to_json(h)})
    return histories


def balanced_sanity_rows(train_rows, per_class):
    by_class = defaultdict(list)
    for row in train_rows:
        by_class[row[1]].append(row)
    rng = np.random.default_rng(SEED)
    selected = []
    for class_id in sorted(by_class):
        values = list(by_class[class_id])
        rng.shuffle(values)
        selected.extend(values[: min(per_class, len(values))])
    rng.shuffle(selected)
    return selected


def run_overfit_sanity(ds, train_rows, args, labels):
    """Prove the same scratch architecture can learn both labels on a tiny balanced subset."""
    sanity_rows = balanced_sanity_rows(train_rows, args.sanity_samples_per_class)
    counts = Counter(target for _, target, _ in sanity_rows)
    if len(counts) != len(labels):
        raise RuntimeError(f"Sanity check missing class: {dict(counts)}")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model, base = build_model(args.image_size, args.alpha, len(labels), "scratch", augment=False)
    base.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"],
    )
    sanity_ds = make_tf_dataset(
        ds, sanity_rows, args.image_size, min(args.batch_size, 16), True
    )
    model.fit(sanity_ds, epochs=args.sanity_epochs, verbose=2)

    y_true, y_pred = [], []
    for image, target in image_generator(ds, sanity_rows, args.image_size):
        scores = model.predict(np.expand_dims(image, 0), verbose=0)[0]
        y_true.append(int(target))
        y_pred.append(int(np.argmax(scores)))
    result = metrics_from_predictions(y_true, y_pred, labels)
    result["samples_per_class_requested"] = int(args.sanity_samples_per_class)
    result["epochs"] = int(args.sanity_epochs)
    result["passed"] = bool(
        result["accuracy"] >= args.min_sanity_accuracy
        and result["classes_predicted"] == len(labels)
    )
    print("Balanced overfit sanity:", json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError(
            "Balanced overfit sanity failed: the scratch architecture/pipeline did not "
            "learn both classes before full training."
        )
    tf.keras.backend.clear_session()
    return result


def representative_rows(train_rows, samples, num_classes):
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


def export_int8(model, ds, train_rows, out_path, image_size, samples, num_classes):
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
        y_true, y_pred, labels=list(range(len(labels))), target_names=labels,
        output_dict=True, zero_division=0,
    )
    pred_counts_raw = Counter(y_pred)
    prediction_counts = {
        label: int(pred_counts_raw.get(i, 0)) for i, label in enumerate(labels)
    }
    total = max(1, len(y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(labels)))
        ).tolist(),
        "classification_report": report,
        "prediction_counts": prediction_counts,
        "prediction_fractions": {
            label: float(count / total) for label, count in prediction_counts.items()
        },
        "classes_predicted": int(sum(count > 0 for count in prediction_counts.values())),
        "collapsed_prediction": bool(sum(count > 0 for count in prediction_counts.values()) < len(labels)),
    }


def evaluate_float(model, ds, test_rows, image_size, labels, batch_size=32):
    test_ds = make_tf_dataset(ds, test_rows, image_size, batch_size, False)
    scores = model.predict(test_ds, verbose=0)
    y_true = [int(target) for _, target, _ in test_rows]
    y_pred = [int(index) for index in np.argmax(scores, axis=1)]
    return metrics_from_predictions(y_true, y_pred, labels)


def evaluate_tflite(path, ds, test_rows, image_size, labels):
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    y_true, y_pred = [], []
    for image, target in image_generator(ds, test_rows, image_size):
        arr = np.clip(np.round(image), 0, 255).astype(np.uint8) if inp["dtype"] == np.uint8 else image.astype(inp["dtype"])
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
        "dataset": OFFICIAL_DATASET, "revision": DATASET_REVISION,
        "task": "binary", "init": "scratch", "smoke": False,
    }
    actual = {
        "dataset": dataset_id, "revision": args.revision,
        "task": args.task, "init": args.init, "smoke": bool(args.smoke),
    }
    if actual != expected:
        raise RuntimeError(f"Official-lineage gate failed.\nExpected: {expected}\nActual: {actual}")


def enforce_release_gates(args, metrics, labels):
    failures = []
    checks = [
        ("accuracy", args.min_int8_accuracy),
        ("balanced_accuracy", args.min_int8_balanced_accuracy),
        ("macro_f1", args.min_int8_macro_f1),
    ]
    for name, minimum in checks:
        if minimum is not None and float(metrics[name]) < minimum:
            failures.append(f"{name} {float(metrics[name]):.4f} < {minimum:.4f}")
    if args.require_all_classes_predicted and metrics["classes_predicted"] != len(labels):
        failures.append(
            f"predicted {metrics['classes_predicted']}/{len(labels)} classes; distribution={metrics['prediction_counts']}"
        )
    if failures:
        raise SystemExit("Release gate failed: " + "; ".join(failures))


def main():
    args = parse_args()
    print_tensorflow_runtime()
    if args.no_imagenet:
        args.init = "scratch"
    if args.smoke:
        args.dataset = SMOKE_DATASET
        args.scratch_epochs = 1
        args.head_epochs = 1
        args.finetune_epochs = 0
        args.representative_samples = min(args.representative_samples, 96)
        args.skip_sanity_check = True

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
    counts = split_class_counts(rows, labels)
    print("Split class counts:", json.dumps(counts, indent=2))

    sanity = None
    if not args.skip_sanity_check:
        sanity = run_overfit_sanity(ds, train_rows, args, labels)
        tf.keras.utils.set_random_seed(SEED)

    balanced_train_rows = balance_rows(train_rows) if args.task == "binary" else train_rows
    balanced_counts = Counter(target for _, target, _ in balanced_train_rows)
    print("Training rows after balancing:", {labels[i]: int(balanced_counts[i]) for i in range(len(labels))})

    train_ds = make_tf_dataset(
        ds, balanced_train_rows, args.image_size, args.batch_size, True
    )
    val_ds = make_tf_dataset(ds, val_rows, args.image_size, args.batch_size, False)

    model, base = build_model(args.image_size, args.alpha, len(labels), args.init)
    histories = train_model(model, base, args, train_ds, val_ds)
    (args.output / "training_history.json").write_text(
        json.dumps(histories, indent=2), encoding="utf-8"
    )

    float_metrics = evaluate_float(
        model, ds, test_rows, args.image_size, labels, args.batch_size
    )
    keras_path = args.output / "plant_health.keras"
    model.save(keras_path)
    (args.output / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    int8_path = args.output / "plant_health_int8.tflite"
    export_int8(
        model, ds, train_rows, int8_path, args.image_size,
        args.representative_samples, len(labels),
    )
    int8_metrics = evaluate_tflite(int8_path, ds, test_rows, args.image_size, labels)

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
            not args.smoke and dataset_id == OFFICIAL_DATASET
            and args.revision == DATASET_REVISION and args.task == "binary"
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
        "split_class_counts": counts,
        "train_examples": len(train_rows),
        "balanced_train_examples": len(balanced_train_rows),
        "validation_examples": len(val_rows),
        "test_examples": len(test_rows),
        "training_strategy": {
            "binary_balancing": "deterministic minority oversampling to equal class counts" if args.task == "binary" else "none",
            "class_weight": False,
            "early_stopping_monitor": "val_loss",
            "scratch_epochs_max": args.scratch_epochs,
            "learning_rate": args.learning_rate,
        },
        "balanced_overfit_sanity": sanity,
        "float_test_metrics": float_metrics,
        "int8_test_metrics": int8_metrics,
        "release_gate": {
            "min_int8_accuracy": args.min_int8_accuracy,
            "min_int8_balanced_accuracy": args.min_int8_balanced_accuracy,
            "min_int8_macro_f1": args.min_int8_macro_f1,
            "require_all_classes_predicted": bool(args.require_all_classes_predicted),
        },
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
    enforce_release_gates(args, int8_metrics, labels)


if __name__ == "__main__":
    main()
