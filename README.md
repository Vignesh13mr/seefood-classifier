# SeeFood: Hotdog or Not Hotdog

A first-year image-classification project comparing a small CNN with transfer learning using MobileNetV2. It includes a reproducible train/validation/test pipeline, data augmentation during training, evaluation metrics, plots, single-image inference, and a Streamlit interface.

## 1. Get the dataset

Create or sign in to a Kaggle account, open the [Hot Dog - Not Hot Dog dataset](https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog), and download its ZIP. Extract it locally; do not put the images in this Git repository.

Place the extracted `train` and `test` folders under `data/raw/hotdog/` so the result looks like this (class folder names such as `hotdog`/`not-hotdog` and `hot_dog`/`not_hot_dog` are both accepted):

```text
data/raw/hotdog/
├── train/
│   ├── hot_dog/       # or hotdog/
│   └── not_hot_dog/   # or not-hotdog/
└── test/
    ├── hot_dog/
    └── not_hot_dog/
```

The training script splits Kaggle's `train` images into training and validation sets (80/20, stratified). Kaggle's separate `test` images remain untouched until the final evaluation. This keeps test examples out of training and validation. The first training run downloads MobileNetV2's public ImageNet initialization; the resulting app checkpoint contains all model weights and can run without repeating that download.

## 2. Set up Python

Use Python 3.12 for the smoothest setup, or a Python version supported by TensorFlow. (TensorFlow's current install guide lists Python 3.10-3.13; Python 3.14 is not listed.) On Windows Command Prompt, from this project folder:

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.12` is not available, install Python 3.12 and reopen Command Prompt. Google Colab is another option; upload the project there and follow its TensorFlow setup.

## 3. Train and evaluate

```cmd
python -m src.seefood.train
```

The program selects the checkpoint with the lowest validation loss, then evaluates that checkpoint on the held-out test set. It prints accuracy, precision, recall, and F1-score, and writes:

- `models/seefood_mobilenet.weights.h5` - trained model weights (the app rebuilds MobileNetV2 and loads these weights)
- `reports/test_metrics.json` - held-out test metrics
- `reports/confusion_matrix.png` - test-set confusion matrix
- `reports/training_curves.png` - training/validation accuracy and loss

Do not quote any metric until you have trained this model; the values depend on the downloaded images and training run.

## 4. Run the image-upload app

```cmd
streamlit run app.py
```

Open the local address printed by Streamlit, usually `http://localhost:8501`, then upload a JPG, JPEG, PNG, WebP, or BMP image.

## 5. Use inference from Python

```python
from src.seefood.inference import predict_image

result = predict_image("my_image.jpg")
print(result["label"], f"{result['confidence']:.1%}")
```

## Model choices and how to explain them

- **Input:** RGB images resized to 128 x 128 and scaled from byte values 0-255 to floats 0-1.
- **Augmentation:** random horizontal flips, small rotations, and zooms run during training only. They make the small training set less repetitive.
- **Transfer learning:** MobileNetV2 starts with visual features learned from ImageNet. Its convolutional backbone is frozen while a small binary classifier learns the hotdog labels. This reduces the number of trainable parameters and helps when the labeled dataset is small. The earlier from-scratch CNN is kept as `build_small_cnn` for comparison.
- **Output:** one sigmoid value is the estimated probability of `hotdog`. A 0.5 threshold maps it to hotdog (1) or not hotdog (0).
- **Loss and optimizer:** binary cross-entropy compares the predicted probability with the true 0/1 label; Adam updates weights using gradients from backpropagation.
- **Training settings:** 32 images per batch, at most 25 epochs, Adam learning rate 0.001, early stopping after five validation-loss epochs without improvement. The best validation-loss checkpoint is used for final test scoring.
- **Metrics:** precision answers “of the images called hotdog, how many were hotdogs?”; recall answers “of all actual hotdogs, how many did it find?”; F1 balances those two. The confusion matrix shows correct and incorrect predictions by class.

## Results from this run

The held-out Kaggle test set contains 500 images. The original from-scratch CNN scored 53.8% accuracy and 0.608 F1. MobileNetV2 transfer learning scored 89.2% accuracy, 91.9% precision, 86.0% recall, and 0.888 F1 on the same test set. These are results from this run, not a guarantee for other data. The baseline metrics are in `reports/cnn_baseline_test_metrics.json`; current metrics and plots are in `reports/`.

This dataset is small, so results can vary and may not generalize to every real-world image. Confidence is not a guarantee. Inspect false positives and false negatives before making performance claims.

## Suggested technical challenge to investigate

The dataset is small, so a model can memorize training examples (overfit). Compare training and validation loss curves. If training loss falls while validation loss rises, try stronger augmentation, dropout, or fewer epochs, and explain the change. Only report a challenge you actually observed and how you addressed it.

## GitHub files

The `.gitignore` excludes downloaded images, virtual environments, and temporary files. It allows the trained MobileNetV2 weights and evaluation reports so the project can be reproduced and reviewed without committing the Kaggle images.

