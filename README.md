# DeepResAutoencoder: Spectral Data Reconstruction using Residual Autoencoders

This project implements a deep **residual autoencoder** for unsupervised reconstruction of spectral data using PyTorch. The model is optimized using a **composite loss** that combines MSE, cosine similarity, and spectral smoothness. Visual diagnostics include spectral plots and UMAP latent space projection.

---

## 🚀 Features

- Residual skip connections for better learning
- Composite loss (MSE + Cosine + Smoothness)
- Mixed-Precision training with AMP (if CUDA available)
- Early stopping & OneCycleLR scheduler
- Spectral reconstruction and residual visualization
- Latent space exploration with UMAP

---

## 📁 Dataset

- Input: 3D spectral dataset (`(samples, 61 wavelengths, 4 components)`)
- Format: `.dat` file loaded via NumPy

---

## 🧠 Model Architecture

```
Input → [512 → 256] → Latent (128-d) → [256 → 512] → Output (+ Residual)
```

- Activations: SELU  
- Normalization: LayerNorm  
- Regularization: Dropout (0.2)  
- Loss: `α*MSE + β*Cosine + γ*Smoothness`

---

## 🛠️ Installation

```bash
pip install torch numpy scikit-learn matplotlib umap-learn
```

---

## ▶️ Run Training

```bash
python train.py
```

Where `train.py` contains:
- Model definition
- Data loading
- Training loop
- Visualizations

---

## 📊 Visualizations

### 📉 Training Curves
- MSE loss vs epoch  
- Validation tracking with early stopping

### 🔬 Spectral Reconstruction
- Original vs Reconstructed vs Residuals  
- Per component (4 channels)

### 🌌 Latent Space (UMAP)
- 2D UMAP embedding from the 128-d latent vector

---

## 📦 Output

- `best_deep_res_autoencoder.pth`: Saved model with best validation loss

---

## 👨‍💻 Author

Dharmik Dudhat  
Built using PyTorch + NumPy + Matplotlib  
Feel free to ⭐ this repo and contribute!
