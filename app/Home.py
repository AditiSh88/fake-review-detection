import streamlit as st
import os, sys, time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import clean_text
from utils.model_loader import load_models

tfidf, logreg, xgb = load_models()

st.set_page_config(page_title="Review Detector", layout="wide")

# ---------------- SIDEBAR (COLORED + BOX MENU) ----------------
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #3b82f6, #6366f1);
}

.sidebar-item {
    background: white;
    padding: 10px;
    border-radius: 10px;
    margin: 8px 0;
    text-align: center;
    font-weight: 500;
    color: #111;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Navigation")
    st.markdown("<div class='sidebar-item'>Home</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-item'>Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-item'>Insights</div>", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("Review Detector")

st.write("A hybrid machine learning system for detecting fake and manipulated reviews using explainable AI.")

# ---------------- INPUT ----------------
st.markdown("## Enter Review")

text_input = st.text_area("", height=140)

uploaded_file = st.file_uploader("Upload review (.txt)", type=["txt"])
if uploaded_file:
    text_input = uploaded_file.read().decode("utf-8")

# ---------------- ANALYZE ----------------
if st.button("Analyze Review"):

    if text_input.strip() == "":
        st.warning("Please enter a review")
    else:

        with st.spinner("Analyzing..."):
            time.sleep(1)

        cleaned = clean_text(text_input)
        vec = tfidf.transform([cleaned])

        prob_lr = logreg.predict_proba(vec)[0][1]
        prob_xgb = xgb.predict_proba(vec)[0][1]
        prob = (prob_lr + prob_xgb) / 2

        pred = 1 if prob > 0.5 else 0
        trust = (1 - prob) * 100 if pred else prob * 100
        agreement = abs(prob_lr - prob_xgb)

        # ---------------- FINAL VERDICT ----------------
        st.markdown("## Final Verdict")

        if pred:
            st.markdown("### Likely Fake Review")
        else:
            st.markdown("### Likely Genuine Review")

        st.write("This result is generated using a hybrid machine learning system combining Logistic Regression and XGBoost.")

        st.caption("Disclaimer: This prediction is for analytical assistance only and should not be considered absolute verification.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------- METRICS ----------------
        col1, col2 = st.columns(2)

        col1.markdown(f"""
        <div style='background:#dbeafe;padding:14px;border-radius:10px'>
        <b>Prediction Score</b><br>{prob:.3f}
        </div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
        <div style='background:#ede9fe;padding:14px;border-radius:10px'>
        <b>Confidence Score</b><br>{trust:.2f}%
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------- EXPLAINABILITY ----------------
        st.markdown("### Explainability")

        st.caption("""
        This section highlights words that influenced the model’s decision.
        Red indicates patterns associated with fake reviews, while green indicates patterns associated with genuine reviews.
        Each word contributes based on learned patterns from training data.
        """)

        feature_names = tfidf.get_feature_names_out()
        coefs = logreg.coef_[0]
        vec_array = vec.toarray()[0]
        indices = vec_array.nonzero()[0]

        weights = {feature_names[i]: coefs[i] for i in indices}

        def highlight(w):
            key = w.lower()
            if key in weights:
                color = "rgba(255,0,0,0.25)" if weights[key] > 0 else "rgba(0,255,0,0.25)"
                return f"<span style='background:{color};padding:3px;border-radius:4px'>{w}</span>"
            return w

        highlighted = " ".join([highlight(w) for w in text_input.split()])
        st.markdown(f"<div style='background:#f8fafc;padding:10px;border-radius:10px'>{highlighted}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------- MODEL INSIGHTS ----------------
        st.markdown("### Model Insights")

        st.caption("""
        Logistic Regression identifies linear relationships between words and labels.
        XGBoost captures complex non-linear patterns and feature interactions.
        Both models are combined in a hybrid system for better stability and accuracy.
        """)

        col1, col2 = st.columns(2)

        col1.markdown(f"""
        <div style='background:#fee2e2;padding:12px;border-radius:10px'>
        <b>Logistic Regression</b><br>{prob_lr:.3f}
        </div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
        <div style='background:#e0e7ff;padding:12px;border-radius:10px'>
        <b>XGBoost</b><br>{prob_xgb:.3f}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------- MODEL AGREEMENT ----------------
        st.markdown("### Model Agreement")

        st.caption("""
        Model agreement measures how consistently Logistic Regression and XGBoost produce similar predictions.
        High agreement indicates strong confidence, while low agreement indicates conflicting interpretations.
        """)

        if agreement < 0.1:
            st.success("High Agreement")
        elif agreement < 0.25:
            st.warning("Moderate Agreement")
        else:
            st.error("Low Agreement")
