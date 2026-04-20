import streamlit as st
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import clean_text
from utils.model_loader import load_models

tfidf, logreg, xgb = load_models()

st.title("Advanced Analysis")

# ---------------- INPUT ----------------
st.markdown("## Write Review")

text_input = st.text_area("Enter review here")

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

        st.markdown("### Final Verdict")
        st.write(f"{verdict(prob_hybrid)} Review")

        st.markdown(f"Prediction: {prob_hybrid:.3f}")

        st.markdown("---")  # spacing requirement

        st.markdown("### Model Results")

        col1, col2 = st.columns(2)
        col1.write(f"LogReg: {prob_lr:.3f}")
        col2.write(f"XGBoost: {prob_xgb:.3f}")

    # ---------------- LOGREG ----------------
    elif model == "Logistic Regression":

        st.markdown("### Final Verdict")
        st.markdown(f"{verdict(prob_lr)} Review")

        st.markdown(f"""
        <div style='background:#fee2e2;padding:10px;border-radius:8px'>
        Prediction: {prob_lr:.3f}
        </div>
        """, unsafe_allow_html=True)

        st.caption("Logistic Regression uses linear word relationships to classify reviews.")

    # ---------------- XGBOOST ----------------
    elif model == "XGBoost":

        st.markdown("### Final Verdict")
        st.markdown(f"{verdict(prob_xgb)} Review")

        st.markdown(f"""
        <div style='background:#e0e7ff;padding:10px;border-radius:8px'>
        Prediction: {prob_xgb:.3f}
        </div>
        """, unsafe_allow_html=True)

        st.caption("XGBoost uses decision trees to capture complex patterns in text.")

    # ---------------- EXPLAINABILITY ----------------
    st.markdown("### Explainability")

    st.caption("Words influencing model decision (red=fake, green=genuine)")
