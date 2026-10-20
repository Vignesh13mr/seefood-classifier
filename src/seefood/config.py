from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "hotdog"
MODEL_PATH = PROJECT_ROOT / "models" / "seefood_mobilenet.weights.h5"
REPORTS_DIR = PROJECT_ROOT / "reports"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 25
VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42
PREDICTION_THRESHOLD = 0.50

# Label 1 is positive (hotdog); label 0 is negative (not hotdog).
CLASS_NAMES = {0: "Not Hotdog", 1: "Hotdog"}

