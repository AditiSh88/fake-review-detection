import streamlit as st
import os, sys

st.set_page_config(page_title="Review Detector", layout="wide")

# ✅ GLOBAL SIDEBAR STYLE (applies everywhere)
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a8a, #4f46e5);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stSidebarNav"] a {
    background: rgba(255,255,255,0.12);
    margin: 6px 10px;
    padding: 10px;
    border-radius: 10px;
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: rgba(255,255,255,0.25);
}
</style>
""", unsafe_allow_html=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import clean_text
from utils.model_loader import load_models

tfidf, logreg, xgb = load_models()

# ✅ CAPITALIZED TITLE
st.title("REVIEW DETECTOR")

text_input = st.text_area("Enter Review")

if st.button("Analyze Review"):

    vec = tfidf.transform([clean_text(text_input)])

    prob_lr = logreg.predict_proba(vec)[0][1]
    prob_xgb = xgb.predict_proba(vec)[0][1]
    prob = (prob_lr + prob_xgb) / 2

    st.markdown("## Final Verdict")

    # ✅ RESULT FIRST
    if prob > 0.5:
        st.markdown("<div style='background:#fee2e2;padding:12px;border-radius:10px;color:#991b1b'>Likely Fake Review</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background:#dcfce7;padding:12px;border-radius:10px;color:#166534'>Likely Genuine Review</div>", unsafe_allow_html=True)

    # ✅ THEN DESCRIPTION
    st.write("This result is generated using a hybrid machine learning system combining Logistic Regression and XGBoost.")

    # ✅ THEN METRICS
    col1, col2 = st.columns(2)

    col1.markdown(f"<div style='background:#dbeafe;padding:10px;border-radius:8px'><b>Prediction</b><br>{prob:.3f}</div>", unsafe_allow_html=True)
    col2.markdown(f"<div style='background:#ede9fe;padding:10px;border-radius:8px'><b>Confidence</b><br>{(1-abs(0.5-prob)*2)*100:.2f}%</div>", unsafe_allow_html=True)
