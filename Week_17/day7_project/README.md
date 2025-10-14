# AI Art Generator (MNIST) — GAN & VAE

A lightweight Streamlit app that trains and generates MNIST digits using:
- **GAN** for sharp samples.
- **VAE** for smooth latent interpolation and reconstruction.

## Features
- Generate image grids (GAN & VAE)
- Latent interpolation (z₁→z₂)
- Reconstruction for VAE
- Mini and full training options directly from UI
- Diversity & sharpness proxy metrics
- Downloadable generated grids

## Setup
1. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. (Optional) Pre-train models:
   ```bash
   python train_vae.py
   python train_gan.py
   ```

3. Launch app:
   ```bash
   streamlit run app.py
   ```

## Controls
- **Model**: Choose GAN or VAE.
- **Latent dim (z)**: Adjust latent vector size.
- **Samples**: Number of images to generate.
- **Interpolate**: Morph between latent points.
- **Reconstruct**: Show VAE reconstructions of real digits.
- **Train Buttons**:
  - ⚡ Mini-Train (3 epochs)
  - 🧠 Train (customizable epochs)

## Tips for Better GAN Results
- Train 30–60 epochs for sharper digits.
- Use label smoothing (real=0.9).
- Try latent dim 64–96.
- DCGAN architectures yield the cleanest images.

## File Structure
- `models.py` — architectures (GAN & VAE)
- `train_vae.py` — VAE trainer
- `train_gan.py` — GAN trainer
- `app.py` — Streamlit app
- `requirements.txt` — dependencies
- `weights/` — saved model checkpoints

## Author
AI Mastery Course — Week 17, Day 7 Mini Project
