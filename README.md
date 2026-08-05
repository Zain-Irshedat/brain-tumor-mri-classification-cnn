🧠 Brain Tumor Detection & Classification (CNN)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN-brightgreen" alt="CNN">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License">
</p>

A deep learning system that analyzes brain MRI scans through a two-stage classification pipeline to first detect the presence of a tumor and then classify its specific type. Built end-to-end with a custom Convolutional Neural Network (CNN) in TensorFlow/Keras.

⸻

📋 Table of Contents

* Overview
* How It Works (Two-Stage Pipeline)
* Model Architecture
* Results
* Repository Contents
* Getting Started
* Dataset Structure
* Tech Stack
* Disclaimer

⸻

🔍 Overview

Given a brain MRI scan, this project implements a structured diagnostic pipeline divided into two distinct classification stages:

1. Stage 1 (Binary Classification): Determines whether the brain MRI scan is healthy (No Tumor) or contains an abnormality (Tumor).
2. Stage 2 (Multi-class Classification): If a tumor is detected, it further categorizes the specific type into one of three major classes:
    * Glioma
    * Meningioma
    * Pituitary Tumor

Trained on the Brain Tumor MRI Dataset, the model follows a cascaded deep learning approach designed to improve interpretability and mimic a realistic diagnostic workflow.

⸻

⚙️ How It Works (Two-Stage Pipeline)

MRI Scan
│
▼
[Preprocessing & Augmentation]
│
▼
Stage 1: Binary Classification (Tumor vs. No Tumor)
├── If "No Tumor" ──> Result: No Tumor Detected
└── If "Tumor" ──> Passes to Stage 2
│
▼
Stage 2: Multi-class Classification
(Glioma / Meningioma / Pituitary)
│
▼
Final Diagnosis

⸻

🏗️ Model Architecture

Layer	Details
Conv2D + MaxPooling	32 filters, 3×3, ReLU
Conv2D + MaxPooling	64 filters, 3×3, ReLU
Conv2D + MaxPooling	128 filters, 3×3, ReLU
Conv2D + MaxPooling	128 filters, 3×3, ReLU
Flatten	—
Dense	512 units, ReLU
Dropout	0.5
Dense (Output)	Softmax Output Layer

Training config: image size 150×150, batch size 32, 25 epochs, Adam optimizer, categorical cross-entropy loss, with data augmentation (rotation, shift, shear, zoom, and horizontal flipping).

⸻

📊 Results

Stage	Task	Accuracy
Stage 1	Tumor Detection (Binary Classification)	91.0%
Stage 2	Tumor Type Classification	83.8%

Additional evaluation metrics include:

* Precision
* Recall
* F1-Score
* Confusion Matrix

The model demonstrates strong performance in detecting brain tumors and classifying major tumor categories through a cascaded CNN framework.

⸻

📁 Repository Contents

File	Description
brain_tumor_detection.ipynb	Complete pipeline including preprocessing, CNN training, evaluation, and visualization
Brain_Tumor_Diagnosis_Deck.pptx	Presentation slides summarizing the project
requirements.txt	Python dependencies

⸻

🚀 Getting Started

# 1. Clone the repository
git clone https://github.com/Zain-Irshedat/brain-tumor-mri-classification-cnn.git
cd brain-tumor-mri-classification-cnn
# 2. Install dependencies
pip install -r requirements.txt
# 3. Download the dataset from Kaggle and arrange it (see structure below)
# 4. Run the notebook
jupyter notebook brain_tumor_detection.ipynb

🗂️ Dataset Structure

dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/

Dataset source: Brain Tumor MRI Dataset — Kaggle

⸻

🛠️ Tech Stack

* Python 3.9+
* TensorFlow / Keras — model building & training
* NumPy / Pandas — data handling
* Matplotlib / Seaborn — visualization
* Jupyter Notebook

⸻

⚠️ Disclaimer

This project is for educational and research purposes only and is not intended for clinical or diagnostic use.