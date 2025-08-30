# 🚢 Titanic Survival Predictor

This is the final project for **Week 1** of my AI Mastery journey.  
It trains a Logistic Regression model on the Titanic dataset and wraps it in a simple Streamlit web app.

---

## 📊 What it does
- Preprocesses data with scikit-learn (imputation + one-hot encoding).  
- Trains a Logistic Regression model to predict survival.  
- Saves the full pipeline using `joblib`.  
- Provides a **Streamlit app** where you can input passenger details and see survival probability.  

---

## ▶️ How to run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train and save the pipeline:
    ```bash
   python train_and_save.py
   ```
3. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

⚠️ Note
This project is for learning purposes only — it’s part of my Week 1 journey in AI Mastery.