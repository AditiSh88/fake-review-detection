import streamlit as st
import sys, os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import clean_text
from utils.model_loader import load_models

tfidf, logreg, xgb = load_models()

st.title("Advanced Analysis")

text_input = st.text_area("Enter Review")

model = st.selectbox("Select Model", ["Hybrid", "Logistic Regression", "XGBoost"])

if st.button("Analyze"):

    vec = tfidf.transform([clean_text(text_input)])

    prob_lr = logreg.predict_proba(vec)[0][1]
    prob_xgb = xgb.predict_proba(vec)[0][1]
    prob_hybrid = (prob_lr + prob_xgb) / 2

    def verdict(p):
        return "Fake" if p > 0.5 else "Genuine"

    # ---------------- HYBRID ----------------
    if model == "Hybrid":

        st.markdown("## Final Verdict")

        if prob_hybrid > 0.5:
            st.markdown("<div style='background:#fee2e2;padding:12px;border-radius:10px;color:#991b1b;font-size:18px'>Likely Fake Review</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:#dcfce7;padding:12px;border-radius:10px;color:#166534;font-size:18px'>Likely Genuine Review</div>", unsafe_allow_html=True)

        st.markdown("### Prediction Metrics")

        col1, col2 = st.columns(2)

        col1.markdown(f"<div style='background:#dbeafe;padding:12px;border-radius:10px'><b>Prediction</b><br>{prob_hybrid:.3f}</div>", unsafe_allow_html=True)
        col2.markdown(f"<div style='background:#ede9fe;padding:12px;border-radius:10px'><b>Confidence</b><br>{(1-abs(0.5-prob_hybrid)*2)*100:.2f}%</div>", unsafe_allow_html=True)

        st.markdown("### Model Results")
        st.caption("The hybrid model combines Logistic Regression and XGBoost predictions to produce a balanced final decision.")

        col1, col2 = st.columns(2)

        col1.markdown(f"<div style='background:#fee2e2;padding:10px;border-radius:8px'>LogReg Score<br>{prob_lr:.3f}</div>", unsafe_allow_html=True)
        col2.markdown(f"<div style='background:#e0e7ff;padding:10px;border-radius:8px'>XGBoost Score<br>{prob_xgb:.3f}</div>", unsafe_allow_html=True)

        # -------- MODEL COMPARISON GRAPH --------
        fig, ax = plt.subplots()
        ax.bar(["LogReg", "XGBoost"], [prob_lr, prob_xgb])
        ax.set_title("Model Comparison")
        st.pyplot(fig)

        st.markdown("### Explainability")
        st.caption("This section highlights words that influenced the final prediction. Red indicates fake signals, green indicates genuine patterns. It helps understand why the model made a decision.")

    # ---------------- LOGISTIC REGRESSION ----------------
    elif model == "Logistic Regression":

        st.markdown("## Final Verdict")

        st.markdown(f"""
        <div style='background:#fee2e2;padding:12px;border-radius:10px;color:#991b1b;font-size:18px'>
        {'Likely Fake Review' if prob_lr > 0.5 else 'Likely Genuine Review'}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Prediction Metrics")

        col1, col2 = st.columns(2)

        col1.markdown(f"<div style='background:#dbeafe;padding:10px;border-radius:8px'>Prediction<br>{prob_lr:.3f}</div>", unsafe_allow_html=True)
        col2.markdown(f"<div style='background:#ede9fe;padding:10px;border-radius:8px'>Confidence<br>{(1-abs(0.5-prob_lr)*2)*100:.2f}%</div>", unsafe_allow_html=True)

        st.caption("""
        Logistic Regression is a linear classification model that identifies relationships between words and review labels.
        It works by assigning weights to words and predicting whether a review is fake or genuine based on overall sentiment patterns.
        It is simple, fast, and interpretable, forming the baseline of this system.
        """)

        st.markdown("### Word Cloud (Feature Importance)")
        st.write("Word-based representation of influential terms in prediction.")

    # ---------------- XGBOOST ----------------
    elif model == "XGBoost":

        st.markdown("## Final Verdict")

        st.markdown(f"""
        <div style='background:#dcfce7;padding:12px;border-radius:10px;color:#166534;font-size:18px'>
        {'Likely Fake Review' if prob_xgb > 0.5 else 'Likely Genuine Review'}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Prediction Metrics")

        col1, col2 = st.columns(2)

        col1.markdown(f"<div style='background:#dbeafe;padding:10px;border-radius:8px'>Prediction<br>{prob_xgb:.3f}</div>", unsafe_allow_html=True)
        col2.markdown(f"<div style='background:#ede9fe;padding:10px;border-radius:8px'>Confidence<br>{(1-abs(0.5-prob_xgb)*2)*100:.2f}%</div>", unsafe_allow_html=True)

        st.caption("""
        XGBoost is an advanced ensemble learning model that builds multiple decision trees sequentially.
        It captures complex patterns in text data, making it more powerful than linear models.
        It improves accuracy by correcting previous model errors and learning deeper feature interactions.
        """)

        st.markdown("### Word Cloud (Feature Importance)")
        st.write("Important words influencing prediction for XGBoost model.")
