# app.py
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")
st.title("🚢 Titanic Survival Predictor")

# Load pipeline
@st.cache_resource #Make sure pipeline is only loaded once at the start
def load_pipeline():
    return joblib.load("titanic_pipeline.joblib")

pipe = load_pipeline()

st.markdown("Enter passenger details to predict the probability of survival.")

# Inputs
col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Sex", ["male", "female"])
    pclass = st.selectbox("Passenger class", [1, 2, 3])
    alone = st.selectbox("Alone?", [0, 1], format_func=lambda x: "No (with family)" if x == 0 else "Yes (alone)")
with col2:
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=30.0, step=1.0)
    threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Predict
if st.button("Predict"):
    X = pd.DataFrame([{
        "sex": sex,
        "pclass": pclass,
        "age": age,
        "fare": fare,
        "alone": alone
    }])
    proba = pipe.predict_proba(X)[0, 1]
    pred = (proba >= threshold).astype(int)

    st.subheader("Result")
    st.write(f"**Survival probability:** {proba:.3f}")
    st.write("**Prediction:** " + ("✅ Survives" if pred == 1 else "❌ Does not survive"))

    # Nice hint about thresholds
    st.caption("Tip: The decision uses a 0.5 threshold. In safety-critical apps you might tune this.")
    st.caption("Young alone females in first class that have paid a high fare have the highest probability of survival.")

