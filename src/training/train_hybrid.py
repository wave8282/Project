"""Train and evaluate an LSTM + LSTM-Autoencoder fall detector.

Dataset must be created by build_sequences.py and contain X, y, and groups.
Splitting is performed by source video, never by overlapping windows. At least
three independent source videos per class are required to form train/validation/
test partitions without source-video leakage.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def split_groups(
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/validation/test indices with video-level class stratification."""
    y = np.asarray(y)
    groups = np.asarray(groups).astype(str)
    if y.ndim != 1 or groups.ndim != 1 or len(y) != len(groups) or len(y) == 0:
        raise ValueError("y and groups must be non-empty 1D arrays of equal length")

    group_labels: dict[str, int] = {}
    for group in np.unique(groups):
        labels = np.unique(y[groups == group])
        if len(labels) != 1:
            raise ValueError(f"Source group {group!r} contains multiple labels")
        group_labels[group] = int(labels[0])

    rng = np.random.default_rng(seed)
    partitions: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for label in sorted(np.unique(y).tolist()):
        class_groups = [g for g, value in group_labels.items() if value == label]
        if len(class_groups) < 3:
            raise ValueError(
                f"Class {label} has only {len(class_groups)} independent source video(s); "
                "at least 3 are required for group-separated train/validation/test. "
                "Collect more independent videos; do not split overlapping windows randomly."
            )
        rng.shuffle(class_groups)
        n_test = max(1, int(round(len(class_groups) * 0.2)))
        n_val = max(1, int(round(len(class_groups) * 0.2)))
        n_train = len(class_groups) - n_test - n_val
        if n_train < 1:
            raise ValueError(f"Not enough source videos for class {label}")
        partitions["test"].extend(class_groups[:n_test])
        partitions["validation"].extend(class_groups[n_test:n_test + n_val])
        partitions["train"].extend(class_groups[n_test + n_val:])

    train_idx = np.flatnonzero(np.isin(groups, partitions["train"]))
    val_idx = np.flatnonzero(np.isin(groups, partitions["validation"]))
    test_idx = np.flatnonzero(np.isin(groups, partitions["test"]))
    if set(groups[train_idx]) & set(groups[val_idx]) or set(groups[train_idx]) & set(groups[test_idx]) or set(groups[val_idx]) & set(groups[test_idx]):
        raise RuntimeError("Internal error: source-video leakage across partitions")
    for name, indices in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
        if set(np.unique(y[indices]).tolist()) != set(np.unique(y).tolist()):
            raise RuntimeError(f"{name} partition does not contain every class")
    return train_idx, val_idx, test_idx


def build_models(sequence_length: int, feature_count: int):
    """Import TensorFlow lazily so split logic remains testable without it."""
    import tensorflow as tf
    from tensorflow.keras import Model, Sequential
    from tensorflow.keras.layers import Dense, Input, LSTM, RepeatVector, TimeDistributed

    classifier = Sequential(
        [
            Input(shape=(sequence_length, feature_count)),
            LSTM(64, return_sequences=False),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ],
        name="fall_lstm_classifier",
    )
    classifier.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    inputs = Input(shape=(sequence_length, feature_count))
    encoded = LSTM(64, activation="relu")(inputs)
    repeated = RepeatVector(sequence_length)(encoded)
    decoded = LSTM(64, activation="relu", return_sequences=True)(repeated)
    outputs = TimeDistributed(Dense(feature_count))(decoded)
    autoencoder = Model(inputs, outputs, name="normal_lstm_autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return classifier, autoencoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/pose_sequences.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/fall_detector"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    args = parser.parse_args()

    if not 0.5 < args.threshold_quantile < 1.0:
        parser.error("--threshold-quantile must be between 0.5 and 1.0")

    random.seed(args.seed)
    np.random.seed(args.seed)

    with np.load(args.dataset, allow_pickle=False) as data:
        required = {"X", "y", "groups"}
        if not required.issubset(data.files):
            raise ValueError(f"Dataset must contain {sorted(required)}")
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        groups = data["groups"].astype(str)

    if X.ndim != 3 or X.shape[0] != len(y) or len(y) != len(groups):
        raise ValueError("Invalid dataset dimensions")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("Both labels 0 (normal) and 1 (fall) are required")

    train_idx, val_idx, test_idx = split_groups(y, groups, args.seed)
    normal_train = train_idx[y[train_idx] == 0]
    normal_val = val_idx[y[val_idx] == 0]
    if len(normal_train) == 0 or len(normal_val) == 0:
        raise ValueError("Normal training and validation sequences are required")

    import tensorflow as tf
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    tf.keras.utils.set_random_seed(args.seed)
    classifier, autoencoder = build_models(X.shape[1], X.shape[2])
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-6),
    ]

    classifier.fit(
        X[train_idx], y[train_idx],
        validation_data=(X[val_idx], y[val_idx]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    autoencoder.fit(
        X[normal_train], X[normal_train],
        validation_data=(X[normal_val], X[normal_val]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-6),
        ],
        verbose=1,
    )

    normal_val_reconstruction = autoencoder.predict(X[normal_val], verbose=0)
    normal_val_errors = np.mean(np.square(X[normal_val] - normal_val_reconstruction), axis=(1, 2))
    threshold = float(np.quantile(normal_val_errors, args.threshold_quantile))

    test_probabilities = classifier.predict(X[test_idx], verbose=0).reshape(-1)
    classifier_predictions = (test_probabilities >= 0.5).astype(np.int64)
    reconstructed = autoencoder.predict(X[test_idx], verbose=0)
    test_errors = np.mean(np.square(X[test_idx] - reconstructed), axis=(1, 2))
    anomaly_predictions = (test_errors > threshold).astype(np.int64)
    hybrid_predictions = ((classifier_predictions == 1) & (anomaly_predictions == 1)).astype(np.int64)

    y_test = y[test_idx]
    metrics = {
        "test_sequences": int(len(test_idx)),
        "test_source_videos": sorted(np.unique(groups[test_idx]).tolist()),
        "classifier": {
            "accuracy": float(accuracy_score(y_test, classifier_predictions)),
            "precision": float(precision_score(y_test, classifier_predictions, zero_division=0)),
            "recall": float(recall_score(y_test, classifier_predictions, zero_division=0)),
            "f1": float(f1_score(y_test, classifier_predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, classifier_predictions, labels=[0, 1]).tolist(),
            "report": classification_report(y_test, classifier_predictions, labels=[0, 1], target_names=["normal", "fall"], zero_division=0, output_dict=True),
        },
        "hybrid": {
            "accuracy": float(accuracy_score(y_test, hybrid_predictions)),
            "precision": float(precision_score(y_test, hybrid_predictions, zero_division=0)),
            "recall": float(recall_score(y_test, hybrid_predictions, zero_division=0)),
            "f1": float(f1_score(y_test, hybrid_predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, hybrid_predictions, labels=[0, 1]).tolist(),
        },
        "autoencoder_threshold": threshold,
        "threshold_quantile_on_normal_validation": args.threshold_quantile,
        "split_source_videos": {
            name: sorted(np.unique(groups[idx]).tolist())
            for name, idx in (("train", train_idx), ("validation", val_idx), ("test", test_idx))
        },
        "warning": "Metrics are meaningful only if source videos are independent and labels are correct.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    classifier.save(args.output_dir / "fall_lstm_classifier.keras")
    autoencoder.save(args.output_dir / "normal_lstm_autoencoder.keras")
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
