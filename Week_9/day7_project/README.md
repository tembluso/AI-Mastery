# Week 9 · Day 7 — IMDB Sentiment Baseline (LSTM)

This project is part of the **AI Mastery 20-Week Program** (Week 9: Sequences & RNNs).  
It trains a simple **LSTM text classifier** on the IMDB reviews dataset and ships it with both a **CLI** and a **Streamlit app** for sentiment analysis.
---

## ⚙️ Setup

1. Create and activate your virtual environment (Windows PowerShell example):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

   Versions known to work:
   - `numpy<2` (1.26.4)
   - `torch==2.2.0`
   - `torchtext==0.17.0`
   - `scikit-learn`
   - `streamlit`
   - `pandas`, `matplotlib`

---

## 🏋️ Training

Run:
```powershell
python train.py
```

This will:
- Load the IMDB dataset,
- Normalize labels (`1` = negative, `2` = positive),
- Train a 1-layer LSTM for the specified epochs,
- Save a single **`bundle.pt`** with model weights, vocab, and config.

💡 Example result after 2 epochs: **~77% test accuracy**.

---

## 🔮 Inference (CLI)

Test the trained model directly:
```powershell
python predict.py "I loved this movie, it was fantastic!"
python predict.py "Worst movie ever."
```

Output example:
```
{'label': 'Positive', 'prob': 0.82}
{'label': 'Negative', 'prob': 0.91}
```

---

## 🌐 Streamlit App

Launch the demo UI:
```powershell
streamlit run app.py
```

Features:
- **Single Review** tab → paste a review and see sentiment + probability
- **Batch** tab → analyze multiple lines or upload a `.txt` file
- Threshold slider to mark low-confidence predictions as *Uncertain*.

---

## 📊 Results

- Model: LSTM (embed=64, hidden=128, dropout=0.3)
- Dataset: IMDB (25k train, 25k test)
- Epochs: 2
- Test Accuracy: ~0.77
- Notes: label mapping adjusted (`1` = negative, `2` = positive`).

---

## 📌 Key Learnings

- How to handle variable-length text sequences with padding + packing.
- Why vocab/tokenizer consistency matters for inference.
- How to save a single bundle (`bundle.pt`) for reproducible inference.
- Deploying a PyTorch model with both CLI and Streamlit.

---

## 🚀 Next Steps

- Train for more epochs (5–6) → accuracy ~85% possible.
- Try a **BiLSTM** or **GRU** (Week 10).
- Add pretrained embeddings (e.g., GloVe).
- Expand the Streamlit app with probability bars or token saliency.

---
