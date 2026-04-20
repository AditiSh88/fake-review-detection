import streamlit as st
import os, sys, time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import clean_text
from utils.model_loader import load_models

tfidf, logreg, xgb = load_models()

st.set_page_config(page_title="Review Detector", layout="wide")

theme = st.toggle("Dark Mode", value=True)

bg = "#0f172a" if theme else "#f8fafc"
text = "#e2e8f0" if theme else "#111827"

st.markdown(f"""
<style>
body {{
    background-color: {bg};
    color: {text};
}}

.hero {{
    padding: 24px;
    border-radius: 14px;
    background: linear-gradient(135deg,#3b82f6,#6366f1,#a855f7);
    color: white;
}}

.box {{
    padding: 12px;
    border-radius: 10px;
    background: #f1f5f9;
    color: #111;
}}

.small {{
    font-size: 12px;
    opacity: 0.7;
}}
</style>
""", unsafe_allow_html=True)

# title
st.markdown("""
<div class="hero">
<h2>Review Detector</h2>
<p>Hybrid machine learning system for detecting fake and manipulated reviews with explainable AI insights.</p>
</div>
""", unsafe_allow_html=True)

# input
st.markdown("### Enter Review")
text_input = st.text_area("", height=140)

uploaded_file = st.file_uploader("Upload review (.txt)", type=["txt"])
if uploaded_file:
    text_input = uploaded_file.read().decode("utf-8")

# analyze
if st.button("Analyze Review"):

    if text_input.strip() == "":
        st.warning("Enter review")
    else:

        with st.spinner("Analyzing review patterns..."):
            time.sleep(1)

        cleaned = clean_text(text_input)
        vec = tfidf.transform([cleaned])

        prob_lr = logreg.predict_proba(vec)[0][1]
        prob_xgb = xgb.predict_proba(vec)[0][1]
        prob = (prob_lr + prob_xgb) / 2

        pred = 1 if prob > 0.5 else 0
        trust = (1 - prob) * 100 if pred else prob * 100
        agreement = abs(prob_lr - prob_xgb)

        # final verdict
        st.markdown("## Final Verdict")

        st.markdown("""
        <div style='background:#dbeafe;padding:10px;border-radius:8px;color:#1e3a8a'>
        This prediction is generated using a hybrid ML model combining Logistic Regression and XGBoost.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if pred:
            st.markdown("<div style='background:#fee2e2;padding:12px;border-radius:10px;color:#991b1b;font-size:18px'>Likely Fake Review</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:#dcfce7;padding:12px;border-radius:10px;color:#166534;font-size:18px'>Likely Genuine Review</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # metrics
        st.markdown("### Prediction Metrics")

        col1, col2 = st.columns(2)

        col1.markdown(f"<div class='box'><b>Prediction</b><br>{prob:.3f}</div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='box'><b>Confidence</b><br>{trust:.2f}%</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # explainability
        st.markdown("### Explainability")

        st.caption("Highlighted words indicate influence on prediction (red=fake signals, green=genuine signals)")

        feature_names = tfidf.get_feature_names_out()
        coefs = logreg.coef_[0]

        vec_array = vec.toarray()[0]
        indices = vec_array.nonzero()[0]

        weights = {feature_names[i]: coefs[i] for i in indices}

        def highlight(w):
            if w.lower() in weights:
                color = "rgba(255,0,0,0.25)" if weights[w.lower()] > 0 else "rgba(0,255,0,0.25)"
                return f"<span style='background:{color};padding:3px;border-radius:4px'>{w}</span>"
            return w

        highlighted = " ".join([highlight(w) for w in text_input.split()])

        st.markdown(f"<div class='box'>{highlighted}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # model insights
        st.markdown("### Model Insights")

        st.caption("Logistic Regression = linear patterns, XGBoost = complex nonlinear patterns")

        col1, col2 = st.columns(2)

        col1.markdown(f"<div class='box'><b>LogReg</b><br>{prob_lr:.3f}</div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='box'><b>XGBoost</b><br>{prob_xgb:.3f}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # model agreement
        st.markdown("### Model Agreement")

        st.caption("Measures consistency between both models")

        if agreement < 0.1:
            st.success("High Agreement")
        elif agreement < 0.25:
            st.warning("Moderate Agreement")
        else:
            st.error("Low Agreement")
