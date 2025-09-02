# app.py
import joblib
import streamlit as st
import numpy as np

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📨", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("spam_nb_pipeline.joblib")

pipe = load_model()
tfidf = pipe.named_steps["tfidf"]
clf = pipe.named_steps["clf"]

st.title("📨 SMS Spam Classifier")
st.write("Type a message or pick a sample, then click **Predict**.")

samples = [
    "Win a free iPhone now!!! Click this link to claim your prize",
    "Are we still meeting at 6pm?",
    "URGENT! Your account has been suspended. Verify your details here",
    "Can you send the notes from today’s lecture?",
    "Congratulations! You’ve been selected for a $1000 gift card"
]

col1, col2 = st.columns([2,1])
with col1:
    mode = st.radio("Input mode", ["Type message", "Choose sample"], horizontal=True)
    if mode == "Type message":
        text = st.text_area("Message", height=120, placeholder="Enter SMS text...")
    else:
        choice = st.selectbox("Sample messages", samples, index=0)
        text = st.text_area("Message", value=choice, height=120)

with col2:
    st.caption("Threshold")
    threshold = st.slider("Decision threshold (spam)", 0.1, 0.9, 0.5, 0.05, label_visibility="collapsed")

if st.button("Predict", type="primary"):
    if not text.strip():
        st.warning("Please enter a message.")
    else:
        proba_spam = pipe.predict_proba([text])[0,1]
        pred_label = "Spam" if proba_spam >= threshold else "Ham"
        st.subheader(f"Prediction: **{pred_label}**")
        st.metric("Spam probability", f"{proba_spam:.3f}")
        st.caption("Tip: lower the threshold to catch more spam (higher recall), raise it to avoid false alarms (higher precision).")

st.divider()
st.subheader("🔍 What words scream *spam*?")
st.caption("Top tokens with highest spam score learned by the model.")
# For MultinomialNB: larger log prob for spam class → more spammy
try:
    feature_names = np.array(tfidf.get_feature_names_out())
    # difference between spam and ham log-probs to rank tokens
    log_spam, log_ham = clf.feature_log_prob_[1], clf.feature_log_prob_[0]
    spam_score = log_spam - log_ham
    top_idx = np.argsort(spam_score)[-20:][::-1]
    top_tokens = feature_names[top_idx]
    top_vals = spam_score[top_idx]
    for t, v in zip(top_tokens, top_vals):
        st.write(f"- **{t}**  (score: {v:.2f})")
except Exception:
    st.info("Feature names unavailable (different vectorizer). Retrain with TF-IDF to see token importances.")
