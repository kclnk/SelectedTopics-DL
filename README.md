# Selected Topics: Deep Learning — Course Project

This repository contains the implementation, experiments, trained models, evaluation results, and documentation for the **Selected Topics: Deep Learning** course project.

The project focuses on applying deep learning techniques for automatic plant leaf disease classification using both:
- A custom CNN architecture
- Transfer learning with EfficientNet-B1

---

# Project Overview

Two independent image classification models were developed:

| Dataset | Model |
|---|---|
| Rice Leaf Disease Dataset | Custom CNN |
| Mango Leaf Disease Dataset | EfficientNet-B1 |

The models classify plant leaf diseases based on leaf image appearance and visual disease symptoms.

---

# Project Features

- Deep learning image classification
- Custom CNN architecture
- Transfer learning using EfficientNet-B1
- Data preprocessing and augmentation
- Learning curve visualization
- Confusion matrices
- Classification metrics
- FLOPs and parameter analysis
- Energy consumption and carbon emission tracking
- Streamlit-based user interface for inference

---

# Evaluation Metrics

The models were evaluated using:

- Accuracy (ACC)
- Precision
- Recall
- F1-Score
- ROC-AUC
- Cohen’s Kappa

Additional computational metrics:
- FLOPs
- Number of Parameters
- Training Time
- Carbon Emissions

---

# Repository Structure

```text
├── docs
├── metrics
├── models
├── notebooks
├── ui
├── outputs
├── training_results
└── README.md
```

---

# Team Members

| Name | ID |
|---|---|
| Abdelrahman Omar Ali | 191900004 |
| Mohamed Safwat Hassan | 192100140 |
| Abdallah Abdelmonem Abdelaziz | 192100154 |
| Moomen Diaa Elbanna | 191800129 |
| Osama Abdelhakem Mohamed | 192100032 |

---

# Course

**Selected Topics: Deep Learning**

---

# Datasets

## Rice Leaf Disease Dataset

Classes:
- Bacterial Blight
- Blast
- Brown Spot
- Tungro

---

## Mango Leaf Disease Dataset

Classes:
- Anthracnose
- Bacterial Canker
- Cutting Weevil
- Die Back
- Gall Midge
- Healthy
- Powdery Mildew
- Sooty Mould

---

# Technologies Used

- Python
- PyTorch
- Torchvision
- Scikit-learn
- Matplotlib
- Streamlit
- CodeCarbon
- THOP

---

# Instructions for Running the Project

## 1. Clone the Repository

```bash
git clone <repository-link>
cd <repository-folder>
```

---

## 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 3. Install Required Libraries

```bash
pip install torch torchvision torchaudio
pip install matplotlib numpy pandas pillow scikit-learn
pip install streamlit torchinfo thop codecarbon
```

---

# Running the Training Notebooks

## Rice Leaf Disease Model

Run:

```bash
rice_leaf_custom_cnn.ipynb
```

This notebook includes:
- Custom CNN architecture
- Training and validation
- Evaluation metrics
- FLOPs calculation
- Carbon emission tracking

---

## Mango Leaf Disease Model

Run:

```bash
mango_leaf_efficientnet_b1.ipynb
```

This notebook includes:
- EfficientNet-B1 transfer learning
- Fine-tuning
- Evaluation metrics
- FLOPs calculation
- Carbon emission tracking

---

# Running the User Interface (UI)

## 1. Navigate to the UI Directory

```bash
cd ui
```

---

## 2. Run the Streamlit Application

```bash
streamlit run app.py
```

---

# Using the Web Interface

The interface allows users to test both trained models directly using leaf images.

## Steps

1. Select the desired model:
   - Rice Leaf Disease Model
   - Mango Leaf Disease Model

2. Upload a leaf image:
   - `.jpg`
   - `.jpeg`
   - `.png`

3. Click:
   
```text
Predict
```

4. The system will display:
   - Predicted disease class
   - Prediction confidence score
   - Uploaded image preview

---

# Saved Models

Place trained model weights inside the:

```text
models/
```

directory.

Example:

```text
models/
├── best_rice_leaf_cnn.pth
└── best_mango_effb1.pth
```

---

# Results Summary

## Rice Leaf Disease Classification
- Custom CNN achieved high classification performance with strong generalization across disease classes.

## Mango Leaf Disease Classification
- EfficientNet-B1 achieved:
  - Accuracy: 99.17%
  - F1-Score: 99.17%
  - ROC-AUC: 1.0000

The confusion matrix demonstrated near-perfect classification across all mango disease classes.

---

# Notes
- Training was performed using GPU acceleration.
- Results may vary slightly depending on hardware and random initialization.
