# Week 7 · Day 7 — CIFAR‑10 Mini‑App 🚀

**Goal:** Train a baseline CNN on CIFAR‑10, save the best checkpoint, and ship a small demo app (Streamlit/CLI) to classify uploaded images.

## Files
- `app.py` — Streamlit app (upload image → top‑K predictions + probs).
- `predict.py` — CLI script for single‑image classification.
- `cifar_cnn_best.pt` — saved model checkpoint (not included here).
- `requirements.txt` — dependencies.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
For CLI mode:
```bash
python predict.py path/to/image.jpg
```

## Features
- Loads a 3‑block CNN trained on CIFAR‑10 (32×32 color images, 10 classes).
- Preprocessing: resize + normalize to match training pipeline.
- **Top‑K predictions** with probabilities (adjustable in sidebar).
- Upload support for PNG/JPG/JPEG/WEBP formats.
- Probability bar chart for visual feedback.
- CLI script for quick classification outside Streamlit.

## What I learned
- How to save and reload model checkpoints for reproducibility.
- How to wrap PyTorch inference into both CLI and Streamlit apps.
- Importance of showing **top‑K** predictions to capture class confusions (e.g., cat vs dog).
- This app is a baseline reference (~65–70% test accuracy) for improving in Week 8 with transfer learning and advanced architectures.
