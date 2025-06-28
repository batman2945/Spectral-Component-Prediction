
# Spectral Component Prediction using Deep Learning

## 🔍 Overview

This project builds a deep learning model to predict spectral component distributions from structural input data. It focuses on **multi-output regression**, predicting four spectral components (each with 61 values), with special optimization and visualization for **Component 4**, which may be of higher scientific or practical interest.

The model is designed in PyTorch and incorporates:
- A custom neural architecture with a shared base and specialized heads
- A **custom loss function** that prioritizes accuracy and correlation, especially for Component 4
- **Advanced visualization** tools to analyze residuals and model predictions
- **Mixed precision training**, **OneCycleLR**, and **early stopping**

---

## 📁 Files

- `Spectral_Model.ipynb`: Main Jupyter notebook
- `best_spectral_model.pth`: Saved model with best validation performance

---

## 🧠 Model Architecture

- Input: 8 normalized structural features
- Output: 4 components × 61 spectrum points
- Architecture:
  - Shared base: 3 fully connected layers with GELU, BatchNorm, and Dropout
  - 4 specialized heads for each component
  - Component 4 head has deeper layers for refined accuracy

---

## 🧪 Loss Function

A composite custom loss:
- **Weighted MSE** for each component
- **Cosine similarity** to preserve spectral shape
- **Peak-sensitive error**, especially for Component 4

---

## 📊 Visualizations

- Component-wise plots comparing predicted vs actual spectra
- Residual plots
- Diagnostics focused on Component 4 across multiple samples

---

## ⚙️ How to Run

1. Prepare your data:
   - `structures_120k.txt` (features)
   - `spectra_120k.dat` (target spectra)

2. Run the notebook:
   - It trains the model with monitoring, saves best checkpoint, and visualizes results.

3. Optional:
   - Load `best_spectral_model.pth` for inference or extended analysis.

---

## 🛠️ Libraries Used

- `PyTorch`
- `scikit-learn`
- `NumPy`
- `Matplotlib`
- `SciPy`

---

## 🎯 Goal

To build a high-performance model that accurately learns structural–spectral relationships, with a focused enhancement on **Component 4**, potentially important in materials science or remote sensing tasks.
