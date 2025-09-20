# Week 10 · Day 7 — IMDB Sentiment App (Final)

This mini‑project upgrades the IMDB sentiment model with a **BiLSTM + Attention** architecture.  
The app classifies reviews as **positive/negative** and highlights influential words using **Integrated Gradients**.

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training

```bash
python train.py
```

This trains the model on IMDB (subset for CPU‑friendliness) and saves:
- `bilstm.pt` — model weights  
- `vocab.json` — vocabulary  
- `config.json` — config with hyperparameters  

---

## 🔮 Inference (CLI)

```bash
python predict.py "I loved this movie, it was amazing!" "Terrible acting and boring plot."
```

Output example:
```json
[
  {"label": "pos", "probs": {"pos": 0.88, "neg": 0.12}},
  {"label": "neg", "probs": {"pos": 0.07, "neg": 0.93}}
]
```

Each prediction also includes token‑level saliency scores.

---

## 🌐 Streamlit App

```bash
streamlit run app.py
```

Features:
- Paste one or many reviews.  
- See predicted label + probability.  
- Highlights show which tokens influenced the decision most.

---

## 📊 Results

- Model: BiLSTM + Attention + small regularizers  
- Dataset: IMDB subset (train=3k, val=0.8k, test=0.8k)  
- Epochs: 3  
- Expected Accuracy/F1: ~0.80+ on test subset (CPU‑friendly run).  

---

## 📌 Key Learnings

- Attention pooling gives the model a mechanism to focus on sentiment words.  
- Integrated Gradients produce cleaner highlights than raw gradients.  
- Small penalties (entropy, stopwords) steer attention away from filler words.

---

