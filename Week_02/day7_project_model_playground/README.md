# 🏡 Model Playground — California Housing

An interactive **Streamlit app** to compare machine learning models on the California Housing dataset.  
This project is the **Week 2 mini-project** of my AI Mastery journey.

---

## ✨ Features

- **Models supported**
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor

- **What you can do**
  - Train & evaluate models with adjustable hyperparameters.
  - See test metrics: **R²** and **RMSE**.
  - Inspect **feature importances** (trees) or **coefficients** (linear regression).
  - Side-by-side **model comparison**.
  - Make **custom predictions** with input values.
  - Visualize **learning curves** to diagnose underfitting/overfitting.

---

## 📊 Why this project?

Week 2 focused on **trees, ensembles, and boosting**.  
Instead of building another “single predictor app,” this project shows:

- How different model families perform on the same dataset.  
- Why ensembles (RF & GB) usually beat linear models.  
- How boosting improves by sequentially correcting errors.  
- The importance of **evaluation beyond a single score**.

---

## 🚀 Run locally

Clone the repo and install requirements:

```bash
pip install -r requirements.txt
streamlit run app.py
```