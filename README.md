# 🧠 AI Mastery Journey

This repository documents my **20-week self-designed AI Mastery course**, built week by week with daily exercises and projects.  

The goal: **learn AI from scratch by building, experimenting, and documenting everything**.  
Each week has 6 days of learning (notebooks) + 1 final project (Day 7).  

---

## 📂 Repository Structure

```txt
AI-Mastery/
├─ Week1/
│  ├─ Day1.ipynb
│  ├─ Day2.ipynb
│  ├─ Day6.ipynb
│  └─ Day7-Project/
│     ├─ train_and_save.py
│     ├─ app.py
│     └─ README.md   (optional, explains this project)
├─ Week2/
│  └─ Day1.ipynb
└─ Week20/
```


- **Daily notebooks (Day1–Day6):** step-by-step exercises, notes, and experiments.  
- **Day 7 Project:** a small end-to-end project for the week, applying everything learned.  
- **Notes:** sometimes you’ll find `.txt` files with my reflections or explanations. These are part of the journey.  

---

## ✅ Completed Projects

- **Week 1 Project — Titanic Survival Predictor**  
  Logistic Regression model trained on the Titanic dataset, wrapped in a Streamlit web app.  
  - Preprocessing pipeline with `scikit-learn`  
  - Logistic Regression classifier  
  - Streamlit app for interactive predictions 

- **Week 2 Project — Model Playground — California Housing** 
  
  An interactive **Streamlit app** to compare machine learning models on the California Housing dataset.  
  - Train & evaluate models with adjustable hyperparameters.
  - See test metrics: **R²** and **RMSE**.
  - Inspect **feature importances** (trees) or **coefficients** (linear regression).
  - Side-by-side **model comparison**.
  - Make **custom predictions** with input values.
  - Visualize **learning curves** to diagnose underfitting/overfitting.

- **Week 3 Project — SMS Spam Classifier** 
  
  An interactive **Streamlit app** that classifies SMS messages as **Spam** or **Ham** (not spam).  
  - Built with **Naive Bayes + TF-IDF**
  - Enter your own SMS text or choose a sample message.
  - Adjustable **decision threshold** to trade off **precision vs recall**.
  - Displays **spam probability** and model prediction.
  - Shows **top spammy words** learned by the Naive Bayes model.

- **Week 4 Project — Movie Recommender 🎬** 
  
  A small, ship‑able item/user‑based collaborative filtering app on MovieLens 100k.
  - Item‑based or user‑based cosine similarity.
  - Filter by minimum number of ratings per movie.
  - Option to exclude already‑watched items.
  - Optional minimum release year filter.
  - Cached data loading and similarity computation for speed.


- **Week 5 Project — Digit Recognizer** 
  
  This is the mini-project app for drawing a digit and getting a prediction using a simple PyTorch MLP.
  - Drawable canvas (280×280) with adjustable stroke width and color.
  - Preprocessing to 28×28 grayscale (MNIST format) with inversion and normalization.
  - PyTorch MLP (256→128) with ReLU, trained with Adam.
  - Top-3 probability display and visualization of the processed input.

- **Week 6 Project — Fashion Recognizer** 
  
  This is the mini-project app where you can train a Fashion-MNIST classifier, watch live metrics,
  browse predictions, and inspect a confusion matrix.
  - Sidebar controls: optimizer (Adam / SGD+Momentum), learning rate, weight decay, dropout, hidden sizes, epochs, batch size.
  - Live training with progress and plots (loss + validation accuracy).
  - Prediction browser: grid of random test images with true/pred labels and **top-3 probabilities**; toggle to show only misclassifications.
  - Confusion matrix tab for full test set.


---

## 🎯 Goals of this repo
- Show my **learning journey** transparently (not just polished code).  
- Build a foundation in AI/ML step by step.  
- Keep a record of every exercise, experiment, and project.  

---

## 🚀 Future
- Weeks 2–20 will include more advanced topics (ML models, deep learning, NLP, etc.).  
- Final weeks will focus on **larger projects** applying everything together.  

---

## ⚠️ Disclaimer
This repo is a **learning journal**.  
Code may be messy, overly commented, or experimental. That’s intentional — it reflects real learning progress.  
