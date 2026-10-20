"""Load a saved model and classify one image without applying training augmentation."""

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from .config import CLASS_NAMES, IMAGE_SIZE, MODEL_PATH, PREDICTION_THRESHOLD
from .model import build_model


def predict_image(image_path: str | Path, model_path: str | Path = MODEL_PATH) -> dict:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Download the dataset and run "
            "python -m src.seefood.train first."
        )
    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize(IMAGE_SIZE)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("Please provide a valid image file.") from error

    batch = np.asarray(image, dtype=np.float32)[None, ...] / 255.0
    # The trained checkpoint contains both the ImageNet backbone and classifier weights,
    # so inference can run without downloading the pretrained backbone again.
    model = build_model(compile_model=False, pretrained=False)
    model.load_weights(model_path)
    hotdog_probability = float(model.predict(batch, verbose=0)[0][0])
    label = int(hotdog_probability >= PREDICTION_THRESHOLD)
    confidence = hotdog_probability if label == 1 else 1.0 - hotdog_probability
    return {
        "label": CLASS_NAMES[label],
        "confidence": confidence,
        "hotdog_probability": hotdog_probability,
    }

