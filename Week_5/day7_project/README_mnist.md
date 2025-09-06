# Week 5 · Day 7 — Streamlit Digit Recognizer

This is the mini-project app for drawing a digit and getting a prediction using a simple PyTorch MLP.

## Run
```bash
pip install -r requirements.txt
streamlit run app_mnist.py
```

On the first run, if `mnist_mlp.pt` is not present, the app will train a quick baseline (3 epochs) on MNIST and cache weights.

## Features
- Drawable canvas (280×280) with adjustable stroke width and color.
- Preprocessing to 28×28 grayscale (MNIST format) with inversion and normalization.
- PyTorch MLP (256→128) with ReLU, trained with Adam.
- Top-3 probability display and visualization of the processed input.
