import streamlit as st
import os, sys
import pandas as pd
import matplotlib.pyplot as plt

# imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import clean_text
from utils.model_loader import load_models

tfidf, logreg, xgb = load_models()

st.set_page_config(page_title="ReviewGuard AI", layout="wide")

# -----------------------------
# THEME
# -----------------------------
theme = st.toggle("Dark Mode", value=True)

if theme:
    bg = "#0b1220"
    card_bg = "rgba(17,24,39,0.7)"
    text = "#e5e7eb"
else:
    bg = "#f8fafc"
    card_bg = "rgba(255,255,255,0.7)"
    text = "#111827"

# css
st.markdown(f"""
<style>
html, body {{
    background-color: {bg};
    color: {text};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto;
}}

.block-container {{
    max-width: 1000px;
    padding-top: 2.5rem;
}}

.hero {{
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: white;
    margin-bottom: 28px;
}}

.gb {{
    border-radius: 20px;
    padding: 1px;
    background: linear-gradient(120deg, #3b82f6, #22c55e, #a855f7);
    margin-bottom: 18px;
}}

.card {{
    padding: 22px;
    border-radius: 18px;
    background: {card_bg};
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}}

.fade-in {{
    animation: fadeInUp .5s ease both;
}}

@keyframes fadeInUp {{
    from {{opacity:0; transform:translateY(10px);}}
    to {{opacity:1; transform:translateY(0);}}
}}
</style>
""", unsafe_allow_html=True)

# hero
st.markdown("""
<div class="hero">
<h1>🚀 ReviewGuard AI</h1>
<p>Hybrid ML system for detecting fake reviews with explainable AI.</p>
</div>
""", unsafe_allow_html=True)

# input
st.markdown("### Enter Review")

text_input = st.text_area("Paste review here...", height=120)

st.markdown("#### OR")
uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])

if uploaded_file:
    text_input = uploaded_file.read().decode("utf-8")
    st.success("File loaded")

# analyze button
if st.button("Analyze Review"):

    if text_input.strip() == "":
        st.warning("Enter review")
    else:
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)

        cleaned = clean_text(text_input)
        vec = tfidf.transform([cleaned])

        prob_lr = logreg.predict_proba(vec)[0][1]
        prob_xgb = xgb.predict_proba(vec)[0][1]

        prob = (prob_lr + prob_xgb) / 2
        pred = 1 if prob > 0.5 else 0

        trust_score = (1 - prob) * 100 if pred == 1 else prob * 100
        agreement = abs(prob_lr - prob_xgb)

        # result
        st.markdown("## Verdict")

        if pred == 1:
            st.markdown("### Likely Fake Review")
        else:
            st.markdown("### Likely Genuine Review")

        st.progress(trust_score / 100)
        st.write(f"Trust Score: {trust_score:.2f}%")

        st.info("This system provides decision support, not absolute truth.")

        st.markdown("---")

        # explainability
        st.markdown("### Why this prediction?")

        feature_names = tfidf.get_feature_names_out()
        coefs = logreg.coef_[0]

        vec_array = vec.toarray()[0]
        indices = vec_array.nonzero()[0]

        word_weights = {feature_names[i]: coefs[i] for i in indices}
        max_weight = max(abs(v) for v in word_weights.values()) if word_weights else 1

        def highlight(word):
            w = word.lower()
            if w in word_weights:
                score = word_weights[w]
                norm = score / max_weight
                color = f"rgba(255,0,0,{abs(norm)})" if norm > 0 else f"rgba(0,255,0,{abs(norm)})"
                return f"<span style='background:{color};padding:4px;border-radius:6px'>{word}</span>"
            return word

        highlighted = " ".join([highlight(w) for w in text_input.split()])
        st.markdown(f"<div class='gb'><div class='card'>{highlighted}</div></div>", unsafe_allow_html=True)

        st.markdown("---")

        # signals
        st.markdown("### Risk Signals")

        signals = []

        if len(text_input.split()) < 5:
            signals.append("Very short review")

        if "buy" in text_input.lower():
            signals.append("Promotional language")

        if text_input.count("!") > 2:
            signals.append("Excessive punctuation")

        if signals:
            for s in signals:
                st.warning(s)
        else:
            st.success("No suspicious signals")

        st.markdown("---")

        # model agreement
        st.markdown("### Model Agreement")

        if agreement < 0.1:
            st.success("High agreement")
        elif agreement < 0.25:
            st.warning("Moderate agreement")
        else:
            st.error("Low agreement")

        st.markdown('</div>', unsafe_allow_html=True)

# bulk analysis
st.markdown("---")
st.markdown("### Bulk Analysis")

bulk_file = st.file_uploader("Upload CSV", type=["csv"], key="bulk")

if bulk_file:
    df = pd.read_csv(bulk_file)

    if "review" not in df.columns:
        st.error("CSV must have 'review'")
    else:
        if st.button("Run Bulk Analysis"):

            results = []

            for review in df["review"]:
                cleaned = clean_text(review)
                vec = tfidf.transform([cleaned])

                prob = (logreg.predict_proba(vec)[0][1] +
                        xgb.predict_proba(vec)[0][1]) / 2

                pred = 1 if prob > 0.5 else 0
                trust = (1 - prob) * 100 if pred == 1 else prob * 100

                results.append({
                    "review": review,
                    "prediction": "Fake" if pred else "Genuine",
                    "trust_score": round(trust, 2)
                })

            result_df = pd.DataFrame(results)

            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode()
            st.download_button("Download Results", csv, "results.csv")

            # dashboard
            st.markdown("### Analytics Dashboard")

            fake = (result_df["prediction"] == "Fake").sum()
            real = (result_df["prediction"] == "Genuine").sum()

            fig, ax = plt.subplots()
            ax.bar(["Fake", "Genuine"], [fake, real])
            st.pyplot(fig)

            fig2, ax2 = plt.subplots()
            ax2.hist(result_df["trust_score"], bins=10)
            st.pyplot(fig2)
