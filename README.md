🧠 Brain Tumor Detection and Classification using a Two-Stage CNN Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN-brightgreen" alt="CNN">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License">
</p>

A deep learning system for automated brain tumor diagnosis from MRI scans using a two-stage Convolutional Neural Network (CNN) pipeline. The model first determines whether a tumor is present and then classifies the detected tumor into one of three major categories: Glioma, Meningioma, or Pituitary tumor.

⸻

📋 Table of Contents

* Overview
* Dataset
* Methodology
* Two-Stage Classification Pipeline
* Model Architecture
* Results
* Repository Structure
* Getting Started
* Tech Stack
* Future Improvements
* Disclaimer

⸻

🔍 Overview

Brain tumors are among the most serious neurological conditions, where early and accurate diagnosis can significantly improve treatment planning and patient outcomes. Manual MRI interpretation requires expert radiologists and can be time-consuming.

This project explores the use of Deep Learning and Computer Vision techniques to assist in brain tumor diagnosis by automatically analyzing MRI scans through a structured two-stage classification approach.

⸻

📂 Dataset

This project uses the Brain Tumor MRI Dataset from Kaggle, containing MRI scans divided into four categories:

* Glioma
* Meningioma
* Pituitary Tumor
* No Tumor

The dataset includes separate training and testing subsets and provides sufficient diversity for evaluating CNN-based classification models.

Dataset Source:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

⸻

⚙️ Methodology

The project follows a cascaded classification strategy designed to simplify the diagnostic process.

Data Preprocessing

* Image resizing
* Normalization
* Data augmentation
* Dataset balancing and preparation

Data Augmentation Techniques

* Rotation
* Width and height shifting
* Zooming
* Shearing
* Horizontal flipping

These techniques help improve model generalization and reduce overfitting.

⸻

🔄 Two-Stage Classification Pipeline

MRI Scan
   │
   ▼
Preprocessing & Augmentation
   │
   ▼
Stage 1: Tumor Detection
(Tumor vs No Tumor)
   │
   ├── No Tumor
   │      ▼
   │  Final Result
   │
   └── Tumor
          ▼
Stage 2: Tumor Classification
(Glioma / Meningioma / Pituitary)
          ▼
      Final Diagnosis

Stage 1 – Binary Classification

Determines whether an MRI scan contains a brain tumor.

Classes:

* Tumor
* No Tumor

Stage 2 – Multi-Class Classification

If a tumor is detected, the MRI scan is passed to a second CNN model that classifies the tumor type into:

* Glioma
* Meningioma
* Pituitary Tumor

This cascaded approach improves interpretability and mirrors a realistic clinical diagnostic workflow.

⸻

🏗️ Model Architecture

The CNN architecture consists of multiple convolutional and pooling layers followed by fully connected layers.

Main Components

* Convolutional Layers
* Max Pooling Layers
* ReLU Activation Functions
* Dropout Regularization
* Fully Connected Dense Layers
* Softmax Output Layer

Training Configuration

Parameter	Value
Image Size	150 × 150
Batch Size	32
Epochs	25
Optimizer	Adam
Loss Function	Categorical Cross-Entropy

⸻

📊 Results

Stage	Task	Accuracy
Stage 1	Tumor Detection (Binary Classification)	91.0%
Stage 2	Tumor Type Classification	83.8%

The model demonstrates strong performance in identifying brain tumors and provides reliable classification of major tumor categories through a cascaded CNN framework.

Additional evaluation metrics include:

* Precision
* Recall
* F1-Score
* Confusion Matrix

⸻

📁 Repository Structure

├── Brain_Tumor_Detection.ipynb
├── Brain_Tumor_Diagnosis_Deck.pptx
├── requirements.txt
└── README.md

⸻

🚀 Getting Started

Clone the Repository

git clone https://github.com/Zain-Irshedat/brain-tumor-mri-classification-cnn.git
cd brain-tumor-mri-classification-cnn

Install Dependencies

pip install -r requirements.txt

Download Dataset

Download the dataset from Kaggle and organize it as follows:

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

Run the Notebook

Open and run:

Brain_Tumor_Detection.ipynb

⸻

🛠️ Tech Stack

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Jupyter Notebook

⸻

🚀 Future Improvements

Potential enhancements include:

* Transfer Learning (ResNet, EfficientNet)
* Model Explainability using Grad-CAM
* Hyperparameter Optimization
* Clinical Validation on Larger MRI Datasets
* Deployment as a Web Application

⸻

⚠️ Disclaimer

This project was developed for educational and research purposes only.

It is not intended for clinical diagnosis, medical decision-making, or patient treatment. Any medical application would require extensive validation, regulatory approval, and evaluation by healthcare professionals.