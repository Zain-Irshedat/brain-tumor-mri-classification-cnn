# brain-tumor-mri-classification-cnn
CNN-based deep learning model that detects whether a brain MRI scan shows a tumor, and if so, classifies it as glioma, meningioma, or pituitary tumor — built with TensorFlow/Keras, ~96% test accuracy.
# Brain Tumor Detection & Classification (CNN, ~96% Accuracy)

A convolutional neural network (CNN) built with TensorFlow/Keras that analyzes brain MRI scans in two conceptual steps:

1. **Detection** — determines whether a tumor is present at all
2. **Classification** — if a tumor is present, identifies its type as one of:
   - **Glioma**
   - **Meningioma**
   - **Pituitary tumor**

Scans with no abnormality are labeled **No Tumor**.

## Overview

Under the hood, this is implemented as a single 4-class CNN (`glioma`, `meningioma`, `pituitary`, `notumor`), which naturally handles both the detection step (notumor vs. the rest) and the classification step (which tumor type) in one pass. The model is trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and achieves approximately **96% test accuracy**.

## Model Architecture

A sequential CNN with:
- 4 Conv2D + MaxPooling2D blocks (32 → 64 → 128 → 128 filters, ReLU activation)
- Flatten layer
- Dense layer (512 units, ReLU)
- Dropout (0.5) for regularization
- Output Dense layer with softmax activation over 4 classes

## Pipeline

1. Load and preprocess MRI images from the training/testing directories
2. Visualize class distribution and sample images per tumor type
3. Train the CNN over 25 epochs (image size 150x150, batch size 32)
4. Evaluate on the test set (loss, accuracy)
5. Generate a confusion matrix
6. Compute precision, recall, and F1-score per class — including how well the model distinguishes "no tumor" scans from tumor-present scans, and how accurately it identifies the tumor type when one is present

## Repository Contents

| File | Description |
|---|---|
| `brain_tumor_cnn.py` | Core model code: data loading/augmentation, CNN architecture, training, evaluation, and metrics |
| `Brain_Tumor_Diagnosis_Deck.pptx` | Presentation slides summarizing the project |

## Getting Started

```bash
pip install -r requirements.txt
```

Download the dataset from Kaggle, arrange it as shown above, then run:

```bash
python brain_tumor_cnn.py
```

## Dataset Structure
