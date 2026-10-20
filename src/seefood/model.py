"""Transfer-learning classifier plus the small CNN used as a baseline."""

import tensorflow as tf

from .config import IMAGE_SIZE


def _augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.08),
        ],
        name="training_augmentation",
    )


def build_model(compile_model: bool = True, pretrained: bool = True) -> tf.keras.Model:
    """Build a classifier using frozen ImageNet features from MobileNetV2."""
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="rgb_image")
    x = _augmentation()(inputs)
    # MobileNetV2 expects pixels scaled from [0, 1] to [-1, 1].
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenet_scaling")(x)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet" if pretrained else None,
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="hotdog_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="seefood_mobilenetv2")
    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )
    return model


def build_small_cnn(compile_model: bool = True) -> tf.keras.Model:
    """Build the original from-scratch CNN for comparison and explanation."""
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="rgb_image")
    x = _augmentation()(inputs)
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="hotdog_probability")(x)

    model = tf.keras.Model(inputs, outputs, name="seefood_cnn")
    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )
    return model

