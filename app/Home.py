import streamlit as st
import os, sys, time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

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
    max-width: 820px;
    padding-top: 1.5rem;
}}

/* Chat bubbles */
.user {{
    background: #1e293b;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
}}

.bot {{
    background: #f1f5f9;
    color: #111;
    padding: 14px;
    border-radius: 12px;
    margin-bottom: 12px;
}}

.hero {{
    padding: 20px;
    border-radius: 14px;
    background: linear-gradient(135deg, #3b82f6, #6366f1, #a855f7);
    color: white;
    margin-bottom: 20px;
}}

textarea {{
    border-radius: 10px !important;
}}
</style>
""", unsafe_allow_html=True)

# hero
st.markdown("""
<div class="hero">
<h3>Review Detector</h3>
<p>AI system that evaluates authenticity of reviews using hybrid machine learning and explainability.</p>
</div>
""", unsafe_allow_html=True)

# input
user_input = st.text_area("", height=120, placeholder="Paste a review...")

uploaded_file = st.file_uploader("", type=["txt"])
if uploaded_file:
    user_input = uploaded_file.read().decode("utf-8")

if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("Enter a review")
    else:

        # USER MESSAGE
        st.markdown(f"<div class='user'>{user_input}</div>", unsafe_allow_html=True)

        # AI THINKING
        with st.spinner("Analyzing linguistic patterns, model signals, and consistency..."):
            time.sleep(1.2)

        cleaned = clean_text(user_input)
        vec = tfidf.transform([cleaned])

        prob_lr = logreg.predict_proba(vec)[0][1]
        prob_xgb = xgb.predict_proba(vec)[0][1]

        prob = (prob_lr + prob_xgb) / 2
        pred = 1 if prob > 0.5 else 0

        trust = (1 - prob) * 100 if pred else prob * 100
        agreement = abs(prob_lr - prob_xgb)

        response = ""

        # RESULT
        if pred:
            response += "<b>Verdict:</b> Likely Fake Review<br>"
        else:
            response += "<b>Verdict:</b> Likely Genuine Review<br>"

        response += f"<b>Confidence:</b> {trust:.2f}%<br><br>"

        # CONFIDENCE EXPLANATION
        if trust > 75:
            response += "The system is highly confident based on strong language patterns.<br><br>"
        elif trust > 50:
            response += "The prediction has moderate confidence with mixed signals.<br><br>"
        else:
            response += "The prediction is uncertain due to ambiguous patterns.<br><br>"

        # AGREEMENT
        if agreement < 0.1:
            response += "<b>Model Agreement:</b> High consistency between models.<br><br>"
        elif agreement < 0.25:
            response += "<b>Model Agreement:</b> Moderate consistency.<br><br>"
        else:
            response += "<b>Model Agreement:</b> Low consistency, review is complex.<br><br>"

        # EXPLAINABILITY
        feature_names = tfidf.get_feature_names_out()
        coefs = logreg.coef_[0]

        vec_array = vec.toarray()[0]
        indices = vec_array.nonzero()[0]

        weights = {feature_names[i]: coefs[i] for i in indices}

        def highlight(w):
            if w.lower() in weights:
                score = weights[w.lower()]
                color = "rgba(255,0,0,0.3)" if score > 0 else "rgba(0,255,0,0.3)"
                return f"<span style='background:{color};padding:3px;border-radius:4px'>{w}</span>"
            return w

        highlighted = " ".join([highlight(w) for w in user_input.split()])

        response += "<b>Key Signals:</b><br>"
        response += f"<div style='background:#f1f5f9;padding:10px;border-radius:8px;color:#111'>{highlighted}</div><br>"

        # FINAL INTERPRETATION
        if pred:
            response += "This review likely contains promotional or manipulated content."
        else:
            response += "This review appears to reflect a genuine user experience."

        st.markdown(f"<div class='bot'>{response}</div>", unsafe_allow_html=True)
