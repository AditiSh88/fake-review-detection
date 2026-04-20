import streamlit as st
import os, sys, time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import clean_text
from utils.model_loader import load_models

tfidf, logreg, xgb = load_models()

st.set_page_config(page_title="ReviewShield AI", layout="wide")

# theme
theme = st.toggle("Dark Mode", value=True)

bg = "#0f172a" if theme else "#f8fafc"
card = "#f1f5f9"
text = "#e2e8f0" if theme else "#111827"

# css
st.markdown(f"""
<style>
body {{
    background-color: {bg};
    color: {text};
}}

.block-container {{
    max-width: 900px;
}}

.hero {{
    padding: 26px;
    border-radius: 16px;
    background: linear-gradient(135deg, #3b82f6, #6366f1, #a855f7);
    color: white;
    margin-bottom: 20px;
}}

.card {{
    background: {card};
    padding: 14px;
    border-radius: 10px;
    color: #111;
}}

.small {{
    font-size: 12px;
    opacity: 0.7;
}}

.upload-btn {{
    font-size: 12px;
    padding: 4px;
}}
</style>
""", unsafe_allow_html=True)

# markdown
st.markdown("""
<div class="hero">
<h2>Review Detector</h2>
<p>Hybrid machine learning system for detecting manipulated and authentic user reviews with explainable AI insights.</p>
</div>
""", unsafe_allow_html=True)

# input
st.markdown("### Enter Review")

text_input = st.text_area("", height=140)

# upload button INSIDE box feel
uploaded_file = st.file_uploader("Upload .txt (optional)", type=["txt"])

if uploaded_file:
    text_input = uploaded_file.read().decode("utf-8")

# analyze
if st.button("Analyze Review"):

    if text_input.strip() == "":
        st.warning("Please enter a review")
    else:

        with st.spinner("Analyzing linguistic patterns and model agreement..."):
            time.sleep(1)

        cleaned = clean_text(text_input)
        vec = tfidf.transform([cleaned])

        prob_lr = logreg.predict_proba(vec)[0][1]
        prob_xgb = xgb.predict_proba(vec)[0][1]

        prob = (prob_lr + prob_xgb) / 2
        pred = 1 if prob > 0.5 else 0

        trust = (1 - prob) * 100 if pred else prob * 100

        # final verdict
        st.markdown("### Final Verdict")

        if pred:
            st.markdown("<div style='background:#fee2e2;padding:10px;border-radius:8px;color:#991b1b'>Likely Fake Review</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:#dcfce7;padding:10px;border-radius:8px;color:#166534'>Likely Genuine Review</div>", unsafe_allow_html=True)

        # metrics
        col1, col2 = st.columns(2)

        col1.metric("Prediction Score", f"{prob:.2f}")
        col2.metric("Confidence", f"{trust:.2f}%")

        st.divider()

        # model comparison
        st.markdown("### Model Comparison")

        st.caption("""
Logistic Regression captures linear relationships in text features, while XGBoost captures complex non-linear patterns. 
Both models are combined in a hybrid system for improved robustness.
""")

        col1, col2 = st.columns(2)

        col1.metric("Logistic Regression", f"{prob_lr:.2f}")
        col2.metric("XGBoost", f"{prob_xgb:.2f}")

        st.divider()

        # explainability
        st.markdown("### Explainability")

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

        st.markdown(f"""
        <div style="background:#f1f5f9;padding:14px;border-radius:10px;color:#111;">
        {highlighted}
        </div>
        """, unsafe_allow_html=True)
