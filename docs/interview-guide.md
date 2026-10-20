# SeeFood interview guide

Use this to understand the project. Explain it in your own words, and only describe results you actually observed after training.

## What happens to an uploaded image?

1. The app accepts an image file and Pillow converts it to RGB.
2. The image is resized to 128 x 128 pixels so every example has the same shape.
3. Pixel values are changed from integers in the range 0-255 to floating-point values in the range 0-1.
4. MobileNetV2 extracts visual features learned from ImageNet. A small classifier on top produces one sigmoid output. A value near 1 means the model leans toward hotdog; a value near 0 means it leans toward not hotdog.
5. The app applies a 0.5 decision threshold, chooses a label, and shows the probability for that chosen class as confidence.

## What does a convolution learn?

A convolution filter is a small grid of numbers that combines nearby pixel values into a feature map. Early filters often respond to edges or colors; deeper layers combine those responses into more complex patterns. In this project's MobileNetV2 transfer-learning model, these filters were first learned from ImageNet and then frozen. Training adjusts the new classifier on top for hotdog labels. The project also keeps its original CNN trained from scratch as a baseline.

## Why max pooling?

Max pooling keeps the strongest value in a small neighborhood. It shrinks the feature maps, reduces computation, and makes the representation somewhat less sensitive to a feature moving by a few pixels.

## What is backpropagation?

The model compares its predicted probability with the true label using binary cross-entropy. Backpropagation calculates how the trainable classifier weights affected that loss. Adam uses those gradients to update the classifier. The pretrained MobileNetV2 backbone stays frozen during this run. Training repeats this over batches and epochs.

## Why binary cross-entropy and Adam?

There are exactly two labels, represented as 0 and 1, and the last layer outputs a probability, so binary cross-entropy is a natural loss. Adam adapts update sizes for each parameter and is a common, practical starting optimizer. Neither choice guarantees good generalization; validation and test results must still be checked.

## Why three data splits?

- **Training:** the examples used to update model weights.
- **Validation:** used while training to select the checkpoint and stop before overfitting.
- **Test:** kept separate until training is finished, to estimate performance on examples the model did not use for fitting or checkpoint selection.

The code makes the validation split only from Kaggle's training folder and keeps Kaggle's test folder untouched. This reduces leakage between the two evaluation sets.

## Reading the metrics

- **Accuracy:** fraction of all predictions that are correct.
- **Precision:** among predicted hotdogs, fraction that are truly hotdogs.
- **Recall:** among actual hotdogs, fraction found by the model.
- **F1-score:** harmonic mean of precision and recall.
- **Confusion matrix:** rows are true labels and columns are predicted labels; off-diagonal counts are mistakes.

For a missed hotdog, ask whether it was small, obscured, oddly shaped, or under unusual lighting. For a false hotdog, check whether another food or a strong color/shape fooled the model. The confusion matrix points to the count of each kind of error; inspect the corresponding images to understand why.

## Be careful with confidence

The displayed confidence is the model's chosen-class probability. Neural-network probabilities are not automatically calibrated, and this simple project does not include out-of-distribution detection. A high number does not prove the image resembles the training data or that the prediction is correct.

## Questions worth practicing

1. Why do we resize and normalize images?
2. How does a convolution differ from a dense layer on raw pixels?
3. What would a rising validation loss with falling training loss suggest?
4. What is the difference between precision and recall?
5. Why must the test set stay untouched while choosing a model?
6. What failure cases did you actually see, and what change did you try?

## Results from the current run

On the untouched 500-image test set, the original CNN reached 53.8% accuracy and 0.608 F1. The MobileNetV2 model reached 89.2% accuracy, 91.9% precision, 86.0% recall, and 0.888 F1. This is an observed comparison on one dataset split; do not claim it proves the model will perform equally well on new images. The transfer-learning model uses pretrained ImageNet features, freezes that backbone, and trains a small output classifier. The test set was not used to choose checkpoints.

