import streamlit as st
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import clean_text
from utils.model_loader import load_models

tfidf, logreg, xgb = load_models()

st.set_page_config(page_title="Review Detector", layout="wide")

# theme
theme = st.toggle("Dark Mode", value=True)

bg = "#0b1220" if theme else "#f8fafc"
text = "#e5e7eb" if theme else "#111827"

# css
st.markdown(f"""
<style>
body {{
    background-color: {bg};
    color: {text};
    font-family: "Segoe UI", sans-serif;
}}

.block-container {{
    max-width: 950px;
    padding-top: 2rem;
}}

.hero {{
    padding: 24px;
    border-radius: 16px;
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: white;
    margin-bottom: 24px;
}}

textarea {{
    border-radius: 10px !important;
}}

</style>
""", unsafe_allow_html=True)

# hero
st.markdown("""
<div class="hero">
<h2>Review Detector AI</h2>
<p>Detect fake and manipulated reviews using hybrid machine learning.</p>
</div>
""", unsafe_allow_html=True)

# input
st.markdown("### Enter Review")

col1, col2 = st.columns([4,1])

with col1:
    text_input = st.text_area("", height=140, placeholder="Paste or upload review...")

with col2:
    uploaded_file = st.file_uploader("📄", type=["txt"])

# handle txt upload properly
if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    reviews = content.split("\n")
    text_input = "\n".join(reviews)

# -analyze
if st.button("Analyze"):

    if text_input.strip() == "":
        st.warning("Enter a review")
    else:
        cleaned = clean_text(text_input)
        vec = tfidf.transform([cleaned])

        prob_lr = logreg.predict_proba(vec)[0][1]
        prob_xgb = xgb.predict_proba(vec)[0][1]

        prob = (prob_lr + prob_xgb) / 2
        pred = 1 if prob > 0.5 else 0

        trust = (1 - prob) * 100 if pred else prob * 100
        agreement = abs(prob_lr - prob_xgb)

        st.markdown("## Verdict")

        if pred:
            st.error("Likely Fake Review")
        else:
            st.success("Likely Genuine Review")

        st.progress(trust/100)
        st.write(f"Trust Score: {trust:.2f}%")

        st.divider()

        # explainability
        st.markdown("### Why this prediction?")
        st.caption("Highlighted words influenced the model decision")

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
        <div style="background:#f9fafb;padding:15px;border-radius:10px;color:#111;">
        {highlighted}
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # signals
        st.markdown("### Risk Signals")
        st.caption("Patterns that may indicate manipulation")

        if len(text_input.split()) < 5:
            st.warning("Very short review")

        if "buy" in text_input.lower():
            st.warning("Promotional language")

        if text_input.count("!") > 2:
            st.warning("Excessive punctuation")

        st.divider()

        # agreement
        st.markdown("### Model Agreement")
        st.caption("How consistent both models are")

        if agreement < 0.1:
            st.success("High agreement")
        elif agreement < 0.25:
            st.warning("Moderate agreement")
        else:
            st.error("Low agreement")
