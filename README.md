# DeepResAutoencoder (Spectral Data Reconstruction)

This project focuses on reconstructing high-dimensional spectral data using a deep residual autoencoder built with PyTorch. It is based on unsupervised learning and explores how well the model can learn meaningful representations from spectral inputs.

---

## What I did

* Built a residual autoencoder for spectral data reconstruction
* Used a combination of losses (MSE, cosine similarity, smoothness)
* Added residual connections to improve learning
* Visualized reconstruction and latent space using UMAP

---

## Dataset

* Shape: `(samples, 61 wavelengths, 4 components)`
* Loaded using NumPy from `.dat` file

---

## Model

Basic structure:

```
Input → Encoder → Latent → Decoder → Output
```

* Latent size: 128
* Activation: SELU
* Dropout: 0.2
* Normalization: LayerNorm

---

## Training

* Optimizer with learning rate scheduling (OneCycleLR)
* Early stopping based on validation loss
* Mixed precision training (if GPU available)

Run:

```bash
python train.py
```

---

## Results

* Compared original vs reconstructed spectra
* Checked residual errors
* Used UMAP to visualize latent space

---

## Output

* Saved model: `best_deep_res_autoencoder.pth`

---

## Tech used

* PyTorch
* NumPy
* Matplotlib
* scikit-learn
* UMAP

---

## Notes

This was mainly a learning + experimental project to understand:

* autoencoders
* high-dimensional data
* representation learning

---

## Author

Dharmik Dudhat
