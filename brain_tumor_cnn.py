"""
Brain Tumor Detection & Classification using CNN
==================================================

Classifies brain MRI scans as:
    - notumor    -> no tumor detected
    - glioma     -> glioma tumor
    - meningioma -> meningioma tumor
    - pituitary  -> pituitary tumor

Dataset: Brain Tumor MRI Dataset (Kaggle)
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Expected directory structure:
    dataset/
        Training/
            glioma/
            meningioma/
            notumor/
            pituitary/
        Testing/
            glioma/
            meningioma/
            notumor/
            pituitary/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_DIR = "dataset/Training"
TEST_DIR = "dataset/Testing"
CATEGORIES = ["glioma", "meningioma", "notumor", "pituitary"]

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 25


# ---------------------------------------------------------------------------
# Data loading & augmentation
# ---------------------------------------------------------------------------
def build_data_generators(train_dir: str, test_dir: str):
    """Create training (with augmentation) and testing data generators."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, test_generator


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
def build_model(num_classes: int) -> Sequential:
    """Build and compile the CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(*IMAGE_SIZE, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(model, train_generator, test_generator):
    """Train the model and return the training history."""
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=test_generator,
        validation_steps=test_generator.samples // BATCH_SIZE,
    )
    return history


def plot_training_curves(history):
    """Plot accuracy and loss curves for training and validation."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, test_generator):
    """Evaluate the model on the test set and print loss/accuracy."""
    loss, accuracy = model.evaluate(
        test_generator, steps=test_generator.samples // BATCH_SIZE
    )
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    return loss, accuracy


def plot_confusion_matrix(model, test_generator):
    """Generate predictions and plot a confusion matrix."""
    predictions = model.predict(test_generator)
    predicted_categories = np.argmax(predictions, axis=1)
    true_categories = test_generator.classes

    cm = tf.math.confusion_matrix(true_categories, predicted_categories)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=np.arange(len(CATEGORIES)), labels=CATEGORIES)
    plt.yticks(ticks=np.arange(len(CATEGORIES)), labels=CATEGORIES)
    plt.show()

    return cm, predicted_categories, true_categories


def print_classification_metrics(cm):
    """Compute and print precision, recall, and F1-score per class."""
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)

    for i, category in enumerate(CATEGORIES):
        print(f"Class: {category}")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall:    {recall[i]:.4f}")
        print(f"  F1-Score:  {f1_score[i]:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    train_generator, test_generator = build_data_generators(TRAIN_DIR, TEST_DIR)

    model = build_model(num_classes=len(CATEGORIES))
    model.summary()

    history = train_model(model, train_generator, test_generator)
    plot_training_curves(history)

    evaluate_model(model, test_generator)
    cm, _, _ = plot_confusion_matrix(model, test_generator)
    print_classification_metrics(cm)

    model.save("brain_tumor_detection_model.h5")
    print("Model saved to brain_tumor_detection_model.h5")


if __name__ == "__main__":
    main()
