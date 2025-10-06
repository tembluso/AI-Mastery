# 📨 SMS Spam Classifier

An interactive **Streamlit app** that classifies SMS messages as **Spam** or **Ham** (not spam).  
Built with **Naive Bayes + TF-IDF**, trained on the [UCI SMS Spam Collection dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).

---

## 🚀 Features
- Enter your own SMS text or choose a sample message.
- Adjustable **decision threshold** to trade off **precision vs recall**.
- Displays **spam probability** and model prediction.
- Shows **top spammy words** learned by the Naive Bayes model.

---

## 📊 Models
- **Multinomial Naive Bayes** with TF-IDF (default).
- Optionally extendable to Logistic Regression or other models.

# 🚀 Run locally

Clone the repo and install requirements:

```bash
pip install -r requirements.txt
python train_and_save.py
streamlit run app.py
```
