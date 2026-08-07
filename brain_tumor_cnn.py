import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------
TRAIN_DIR = "dataset/Training"
TEST_DIR = "dataset/Testing"
CATEGORIES = ["glioma", "meningioma", "notumor", "pituitary"]

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 25

# ---------------------------------------------------------------------------
# Data Loading & Augmentation
# ---------------------------------------------------------------------------
def build_data_generators(train_dir: str, test_dir: str):
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
# Model Architecture
# ---------------------------------------------------------------------------
def build_model(num_classes: int) -> Sequential:
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
# Training & Evaluation
# ---------------------------------------------------------------------------
def main():
    train_generator, test_generator = build_data_generators(TRAIN_DIR, TEST_DIR)

    model = build_model(num_classes=len(CATEGORIES))
    model.summary()

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=test_generator,
        validation_steps=test_generator.samples // BATCH_SIZE,
    )

    loss, accuracy = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    model.save("brain_tumor_detection_model.h5")
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
