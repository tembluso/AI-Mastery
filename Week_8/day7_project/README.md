# Week 8 · Day 7 — Transfer Learning Demo

This mini-project fine-tunes a pretrained **ResNet18** on a small dataset (e.g., CIFAR-10 subset or a custom 3–10 class dataset).  
The app allows users to upload an image and see **top-3 predictions** along with a **Grad-CAM heatmap** for explainability.

---

## 🚀 Setup

```bash
pip install -r requirements.txt
```

Ensure you have your trained checkpoint:
- Place your fine-tuned weights in the project root as **`best_resnet18.pt`**.

---

## ▶️ Run

```bash
streamlit run app.py
```

---

## ✨ Features
- Loads a fine-tuned ResNet18 model.
- Upload `.png`, `.jpg`, `.jpeg`, `.webp` images.
- Returns **top-3 class predictions** with probabilities.
- Shows **Grad-CAM heatmap** overlay for transparency.

---

## 📊 Results
- Validation accuracy target: **85–90%** (on CIFAR-10 subset or small domain dataset).  
- Grad-CAM highlights show model focus regions.  
- Example failure cases: e.g., ship misclassified as airplane due to blue background.  

---

## 📂 Files
- `app.py` — Streamlit app with predictions + Grad-CAM.
- `requirements.txt` — dependencies.
- `best_resnet18.pt` — fine-tuned checkpoint (not included in repo).
- `README.md` — this file.

---

## ✅ Acceptance Criteria
1. Clear data prep steps documented.  
2. ≥85–90% val acc on your small dataset (or justify if harder).  
3. Grad-CAM explainability surfaced in the UI.  
4. Working README + requirements.txt.  
