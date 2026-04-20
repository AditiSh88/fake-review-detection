import streamlit as st
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from preprocessing import clean_text
from utils.model_loader import load_models

tfidf, logreg, xgb = load_models()

st.set_page_config(page_title="Advanced Analysis")

st.title("Advanced Analysis")

text = st.text_area("Enter review")

model_choice = st.selectbox("Select Model", ["Hybrid", "Logistic Regression", "XGBoost"])

if st.button("Analyze"):

    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])

    prob_lr = logreg.predict_proba(vec)[0][1]
    prob_xgb = xgb.predict_proba(vec)[0][1]

    prob_hybrid = (prob_lr + prob_xgb) / 2

    def show_result(prob, label):
        pred = 1 if prob > 0.5 else 0
        trust = (1 - prob) * 100 if pred else prob * 100

        st.subheader(label)
        st.write("Prediction:", "Fake" if pred else "Genuine")
        st.write("Confidence:", f"{trust:.2f}%")

    # SELECTED MODEL
    if model_choice == "Hybrid":
        show_result(prob_hybrid, "Hybrid Model")
    elif model_choice == "Logistic Regression":
        show_result(prob_lr, "Logistic Regression")
    else:
        show_result(prob_xgb, "XGBoost")

    st.divider()

    # COMPARISON
    st.markdown("### Model Comparison")

    col1, col2, col3 = st.columns(3)

    col1.metric("Hybrid", f"{prob_hybrid:.2f}")
    col2.metric("LogReg", f"{prob_lr:.2f}")
    col3.metric("XGBoost", f"{prob_xgb:.2f}")
