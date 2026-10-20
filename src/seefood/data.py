"""Dataset discovery, leakage-safe splitting, image preprocessing, and batching."""

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .config import (
    BATCH_SIZE,
    DATA_ROOT,
    IMAGE_SIZE,
    RANDOM_SEED,
    VALIDATION_FRACTION,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _class_label(folder_name: str) -> int | None:
    """Map common Kaggle folder spellings to our fixed 0/1 labels."""
    normalized = "".join(character for character in folder_name.lower() if character.isalnum())
    if normalized == "hotdog":
        return 1
    if normalized in {"nothotdog", "nonhotdog"}:
        return 0
    return None


def _images_by_class(split_dir: Path) -> tuple[list[str], list[int]]:
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Expected a '{split_dir.name}' folder at {split_dir}. "
            "See the README for the required dataset layout."
        )

    paths: list[str] = []
    labels: list[int] = []
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        label = _class_label(class_dir.name)
        if label is None:
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(str(image_path))
                labels.append(label)

    counts = {label: labels.count(label) for label in (0, 1)}
    if not paths or any(count == 0 for count in counts.values()):
        raise ValueError(
            f"Both hotdog and not-hotdog image folders must contain images in {split_dir}. "
            f"Found counts: not_hot_dog={counts[0]}, hot_dog={counts[1]}"
        )
    return paths, labels


def _decode_and_resize(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.cast(label, tf.float32)


def _make_dataset(paths: list[str], labels: list[int], training: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths), seed=RANDOM_SEED, reshuffle_each_iteration=True
        )
    dataset = dataset.map(_decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset


def load_datasets(data_root: Path = DATA_ROOT):
    """Use Kaggle's train split for train/validation, and hold test split untouched."""
    train_paths, train_labels = _images_by_class(data_root / "train")
    test_paths, test_labels = _images_by_class(data_root / "test")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=VALIDATION_FRACTION,
        random_state=RANDOM_SEED,
        stratify=train_labels,
    )

    datasets = (
        _make_dataset(train_paths, train_labels, training=True),
        _make_dataset(val_paths, val_labels, training=False),
        _make_dataset(test_paths, test_labels, training=False),
    )
    counts = {
        "train": len(train_paths),
        "validation": len(val_paths),
        "test": len(test_paths),
    }
    return (*datasets, counts)

