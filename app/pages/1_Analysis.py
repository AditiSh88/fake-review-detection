import streamlit as st
import sys, os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="Analysis", layout="wide")

# ✅ SAME SIDEBAR STYLE
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

st.title("Advanced Analysis")

text_input = st.text_area("Enter Review")

model = st.selectbox("Select Model", ["Hybrid", "Logistic Regression", "XGBoost"])

if st.button("Analyze"):

    vec = tfidf.transform([clean_text(text_input)])

    prob_lr = logreg.predict_proba(vec)[0][1]
    prob_xgb = xgb.predict_proba(vec)[0][1]
    prob_hybrid = (prob_lr + prob_xgb) / 2

    def verdict_box(prob):
        if prob > 0.5:
            return "<div style='background:#fee2e2;padding:12px;border-radius:10px;color:#991b1b'>Likely Fake Review</div>"
        else:
            return "<div style='background:#dcfce7;padding:12px;border-radius:10px;color:#166534'>Likely Genuine Review</div>"

    # ---------------- HYBRID ----------------
    if model == "Hybrid":

        st.markdown("## Final Verdict")
        st.markdown(verdict_box(prob_hybrid), unsafe_allow_html=True)

        # Prediction + Confidence
        col1, col2 = st.columns(2)
        col1.markdown(f"<div style='background:#dbeafe;padding:10px;border-radius:8px'><b>Prediction</b><br>{prob_hybrid:.3f}</div>", unsafe_allow_html=True)
        col2.markdown(f"<div style='background:#ede9fe;padding:10px;border-radius:8px'><b>Confidence</b><br>{(1-abs(0.5-prob_hybrid)*2)*100:.2f}%</div>", unsafe_allow_html=True)

        # ✅ MODEL SCORES
        st.markdown("### Model Scores")

        c1, c2 = st.columns(2)
        c1.markdown(f"<div style='background:#fee2e2;padding:10px;border-radius:8px'>Logistic Regression<br>{prob_lr:.3f}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div style='background:#e0e7ff;padding:10px;border-radius:8px'>XGBoost<br>{prob_xgb:.3f}</div>", unsafe_allow_html=True)

        # ✅ EXPLAINABILITY
        st.markdown("### Explainability")
        st.caption("Highlighted words indicate their contribution to prediction.")

        feature_names = tfidf.get_feature_names_out()
        coefs = logreg.coef_[0]
        vec_arr = vec.toarray()[0]

        weights = {feature_names[i]: coefs[i] for i in range(len(vec_arr)) if vec_arr[i] > 0}

        def highlight(w):
            key = w.lower()
            if key in weights:
                color = "rgba(255,0,0,0.25)" if weights[key] > 0 else "rgba(0,255,0,0.25)"
                return f"<span style='background:{color};padding:3px;border-radius:4px'>{w}</span>"
            return w

        highlighted = " ".join([highlight(w) for w in text_input.split()])
        st.markdown(f"<div style='background:#f8fafc;padding:10px;border-radius:10px'>{highlighted}</div>", unsafe_allow_html=True)

        # ✅ GRAPH LAST
        st.markdown("### Model Comparison")

        fig, ax = plt.subplots()
        ax.bar(["LogReg", "XGBoost"], [prob_lr, prob_xgb])
        st.pyplot(fig)

    # ---------------- LOGREG & XGB ----------------
    # ✅ LEFT UNCHANGED (as you requested)
