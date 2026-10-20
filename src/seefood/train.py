"""Train, select the best validation checkpoint, and report held-out test metrics."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import MODEL_PATH, PREDICTION_THRESHOLD, RANDOM_SEED, REPORTS_DIR, EPOCHS
from .data import load_datasets
from .model import build_model


def _save_curves(history: tf.keras.callbacks.History) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set(title="Accuracy by epoch", xlabel="Epoch", ylabel="Accuracy")
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set(title="Loss by epoch", xlabel="Epoch", ylabel="Binary cross-entropy")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "training_curves.png", dpi=150)
    plt.close(figure)


def main() -> None:
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    train_ds, val_ds, test_ds, counts = load_datasets()
    print(f"Image counts: {counts}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model = build_model()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )
    _save_curves(history)

    # Restore the checkpoint selected only by validation loss before final test scoring.
    best_model = build_model(compile_model=False, pretrained=False)
    best_model.load_weights(MODEL_PATH)
    y_true = np.concatenate([labels.numpy().astype(int) for _, labels in test_ds])
    y_probability = best_model.predict(test_ds, verbose=0).ravel()
    y_pred = (y_probability >= PREDICTION_THRESHOLD).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": PREDICTION_THRESHOLD,
        "test_image_count": int(len(y_true)),
    }
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORTS_DIR / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    figure, axis = plt.subplots(figsize=(5, 4))
    display = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Not Hotdog", "Hotdog"],
        yticklabels=["Not Hotdog", "Hotdog"],
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion matrix (held-out test set)",
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(display, ax=axis)
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    print("Held-out test metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Plots and metrics saved in: {REPORTS_DIR}")


if __name__ == "__main__":
    main()

