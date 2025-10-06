# Week 6 · Day 7 — Fashion-MNIST Mini-Project (Streamlit)

**Visual mini-project**: train a Fashion-MNIST classifier, watch live metrics, browse predictions, and inspect a confusion matrix.

## Run
```bash
pip install -r requirements_fashion.txt
streamlit run fashion_app.py
```

## Features
- Sidebar controls: optimizer (Adam / SGD+Momentum), learning rate, weight decay, dropout, hidden sizes, epochs, batch size.
- Live training with progress and plots (loss + validation accuracy).
- Prediction browser: grid of random test images with true/pred labels and **top-3 probabilities**; toggle to show only misclassifications.
- Confusion matrix tab for full test set.
- Caches model weights to `fashion_net.pt` for reuse.
