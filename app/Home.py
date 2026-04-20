import streamlit as st
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import clean_text
from utils.model_loader import load_models

tfidf, logreg, xgb = load_models()

st.set_page_config(page_title="Review Detector", layout="wide")

# theme
theme = st.toggle("Dark Mode", value=True)

bg = "#0f172a" if theme else "#f8fafc"
text = "#e2e8f0" if theme else "#111827"

# css
st.markdown(f"""
<style>
body {{
    background-color: {bg};
    color: {text};
    font-family: 'Segoe UI', sans-serif;
}}

.block-container {{
    max-width: 950px;
    padding-top: 2rem;
}}

.hero {{
    padding: 26px;
    border-radius: 16px;
    background: linear-gradient(135deg, #3b82f6, #6366f1, #a855f7);
    color: white;
    margin-bottom: 24px;
}}

textarea {{
    border-radius: 10px !important;
}}

.small-upload {{
    font-size: 12px;
    opacity: 0.7;
}}
</style>
""", unsafe_allow_html=True)

# hero
st.markdown("""
<div class="hero">
<h2>Review Detector</h2>
<p>Hybrid machine learning system for detecting fake and manipulated reviews with explainable outputs.</p>
</div>
""", unsafe_allow_html=True)

# input
st.markdown("### Enter Review")

text_input = st.text_area("", height=150, placeholder="Paste review here...")

uploaded_file = st.file_uploader("", type=["txt"])
if uploaded_file:
    text_input = uploaded_file.read().decode("utf-8")

st.caption("Upload a .txt file (optional)")

# analysis
if st.button("Analyze"):

    if text_input.strip() == "":
        st.warning("Please enter a review")
    else:
        cleaned = clean_text(text_input)
        vec = tfidf.transform([cleaned])

        prob_lr = logreg.predict_proba(vec)[0][1]
        prob_xgb = xgb.predict_proba(vec)[0][1]

        prob = (prob_lr + prob_xgb) / 2
        pred = 1 if prob > 0.5 else 0

        trust = (1 - prob) * 100 if pred else prob * 100
        agreement = abs(prob_lr - prob_xgb)

        # RESULT
        st.markdown("## Result")

        if pred:
            st.error("Likely Fake Review")
        else:
            st.success("Likely Genuine Review")

        st.progress(trust/100)
        st.write(f"Trust Score: {trust:.2f}%")

        st.divider()

        # EXPLAINABILITY
        st.markdown("### Why this prediction?")
        st.caption("""
The highlighted words below contributed most to the model’s decision. 
Words in red indicate signals commonly associated with fake or promotional content, 
while green words indicate patterns seen in genuine reviews.
This helps in understanding not just the result, but the reasoning behind it.
""")

        feature_names = tfidf.get_feature_names_out()
        coefs = logreg.coef_[0]

        vec_array = vec.toarray()[0]
        indices = vec_array.nonzero()[0]

        weights = {feature_names[i]: coefs[i] for i in indices}

        def highlight(w):
            if w.lower() in weights:
                score = weights[w.lower()]
                color = "rgba(255,0,0,0.3)" if score > 0 else "rgba(0,255,0,0.3)"
                return f"<span style='background:{color};padding:4px;border-radius:5px'>{w}</span>"
            return w

        highlighted = " ".join([highlight(w) for w in text_input.split()])

        st.markdown(f"""
        <div style="background:#f1f5f9;padding:15px;border-radius:10px;color:#111;">
        {highlighted}
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # RISK SIGNALS
        st.markdown("### Risk Signals")
        st.caption("""
Risk signals highlight linguistic or structural patterns that are often found in fake or manipulated reviews. 
These include overly promotional language, unnatural emphasis, or lack of detail. 
They are not definitive proof but provide supporting evidence for the prediction.
""")

        if len(text_input.split()) < 5:
            st.warning("Very short review")

        if "buy" in text_input.lower():
            st.warning("Promotional language detected")

        if text_input.count("!") > 2:
            st.warning("Excessive punctuation")

        st.divider()

        # AGREEMENT
        st.markdown("### Model Agreement")
        st.caption("""
This measures how closely both machine learning models agree on the prediction. 
High agreement indicates strong confidence, while low agreement suggests uncertainty 
and that the review may be more complex or ambiguous.
""")

        if agreement < 0.1:
            st.success("High agreement")
        elif agreement < 0.25:
            st.warning("Moderate agreement")
        else:
            st.error("Low agreement")
