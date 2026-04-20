import streamlit as st
import os, sys, time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import clean_text
from utils.model_loader import load_models

tfidf, logreg, xgb = load_models()

st.set_page_config(page_title="Review Detector", layout="wide")

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

        st.write("This result is generated using a hybrid machine learning system combining Logistic Regression and XGBoost.")

        if pred:
            st.markdown("<div style='background:#fee2e2;padding:12px;border-radius:10px;color:#991b1b;font-size:18px'>Likely Fake Review</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:#dcfce7;padding:12px;border-radius:10px;color:#166534;font-size:18px'>Likely Genuine Review</div>", unsafe_allow_html=True)

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

        # ---------------- EXPLAINABILITY ----------------
        st.markdown("### Explainability")

        st.caption("""
        Words highlighted here influence the model’s prediction. Red indicates fake-related signals, green indicates genuine patterns.
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

        # ---------------- MODEL INSIGHTS ----------------
        st.markdown("### Model Insights")

        st.caption("""
        Logistic Regression works by finding simple relationships between words and labels.
        XGBoost uses advanced decision trees to capture complex patterns in text data.
        """)

        col1, col2 = st.columns(2)

        col1.markdown(f"<div style='background:#fee2e2;padding:12px;border-radius:10px'><b>LogReg</b><br>{prob_lr:.3f}</div>", unsafe_allow_html=True)
        col2.markdown(f"<div style='background:#e0e7ff;padding:12px;border-radius:10px'><b>XGBoost</b><br>{prob_xgb:.3f}</div>", unsafe_allow_html=True)

        # ---------------- MODEL AGREEMENT ----------------
        st.markdown("### Model Agreement")

        st.caption("""
        Agreement shows how similarly both Logistic Regression and XGBoost interpret the review.
        Higher agreement means more reliable prediction.
        """)

        if agreement < 0.1:
            st.success("High Agreement")
        elif agreement < 0.25:
            st.warning("Moderate Agreement")
        else:
            st.error("Low Agreement")
